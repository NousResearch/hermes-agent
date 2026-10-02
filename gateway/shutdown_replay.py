"""Revalidate persisted pending input before startup admission."""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, Protocol

from gateway.input_owner import gateway_input_owner
from gateway.pending_execution import PendingExecutionOwner, PendingExecutionRunner
from gateway.config import GatewayConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.base_pending import (
    merge_recorded,
    pending_dispatch_scope,
    pending_part,
    reserve_pending_dispatch,
    release_pending_dispatch_record,
)
from gateway.platforms.event import MessageEvent
from gateway.shutdown_withdrawal import bind_spooled_event
from gateway.shutdown_pending import PENDING_SCHEMA, PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_event, decode_pending_source
from gateway.shutdown_recovery import consume_executed_pending
from gateway.session_transcript import TranscriptReadError
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class PendingReplayRunner(PendingExecutionRunner, Protocol):
    config: GatewayConfig

    def _session_key_for_source(self, source: SessionSource) -> str: ...

    def _is_user_authorized_for_source(self, source: SessionSource) -> bool: ...


async def replay_pending_snapshots(runner: PendingReplayRunner) -> int:
    from gateway.run import _multiplex_profile_homes, _profile_runtime_scope
    from hermes_constants import get_hermes_home

    launch = get_hermes_home().resolve()
    homes = [launch]
    if runner.config.multiplex_profiles:
        homes.extend(
            Path(home).resolve()
            for _name, home in _multiplex_profile_homes(runner.config)
            if Path(home).resolve() != launch
        )
    admitted = 0
    for home in homes:
        with _profile_runtime_scope(home):
            consume_executed_pending(runner.session_store)
            snapshots = []
            for path in (home / "pending_messages").glob("*.json"):
                try:
                    original = path.read_bytes()
                    payload = json.loads(original)
                    if (
                        not isinstance(payload, dict)
                        or payload.get("schema") != PENDING_SCHEMA
                        or payload.get("projection") is not None
                    ):
                        continue
                    snapshot = PendingQueueSnapshot.from_payload(payload)
                    if Path(snapshot.runtime_home).resolve() != home:
                        continue
                    snapshots.append((snapshot, path, original))
                except (OSError, ValueError, TypeError):
                    logger.warning(
                        "Could not read pending input in %s; preserving it",
                        path,
                        exc_info=True,
                    )
            snapshots.sort(key=lambda item: (item[0].ts, item[1].name))
            copies: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for snapshot, _path, _original in snapshots:
                for record in snapshot.events:
                    copies.setdefault((snapshot.session_key, record["uid"]), []).append(
                        record
                    )
            seen: set[tuple[str, str]] = set()
            for snapshot, path, original in snapshots:
                for record in snapshot.events:
                    if record.get("withdrawn") is True:
                        continue
                    identity = (snapshot.session_key, record["uid"])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    if any(copy != record for copy in copies[identity]):
                        logger.warning(
                            "Conflicting pending input copies for %s; preserving them",
                            record["uid"],
                        )
                        continue
                    try:
                        admitted += await _replay_record(
                            runner, snapshot, path, original, record, home
                        )
                    except (
                        OSError,
                        ValueError,
                        TypeError,
                        KeyError,
                        TranscriptReadError,
                    ):
                        logger.warning(
                            "Could not restore pending input %s; preserving it",
                            record["uid"],
                            exc_info=True,
                        )
    return admitted


async def _replay_record(
    runner: PendingReplayRunner,
    snapshot: PendingQueueSnapshot,
    path: Path,
    original: bytes,
    record: dict[str, Any],
    home: Path,
) -> int:
    source = decode_pending_source(record)
    routing = record.get("routing")
    if routing is None and runner.config.multiplex_profiles:
        return 0
    profile = routing.get("transport_profile") if isinstance(routing, dict) else None
    adapter = runner._adapters_for_profile(profile).get(source.platform)
    if adapter is None or not adapter.is_connected or "input_owner" not in record:
        return 0
    resolved = runner.session_store.resolve_session_id_for_key(
        snapshot.session_key, not_after=snapshot.ts
    )
    if resolved is None:
        return 0
    session_id, db = resolved
    row = db.get_session(session_id)
    if (
        Path(db.db_path).resolve().parent != home
        or not row
        or float(row["started_at"]) > snapshot.ts
    ):
        return 0
    event = decode_pending_event(record, adapter=adapter)
    if event.internal or event.source.delivered_via_upstream_relay:
        return 0
    if not bind_spooled_event(runner, adapter, event, home, snapshot.session_key):
        return 0
    if not runner._is_user_authorized_for_source(event.source):
        return 0
    owner = gateway_input_owner(event, event.source)
    if owner != record["input_owner"]["owner"] or runner.session_store.has_input_owner(
        session_id, owner
    ):
        return 0
    reserved = reserve_pending_dispatch(
        adapter, snapshot.session_key, event, accepted=False
    )
    revision = reserved.revision
    try:
        with pending_dispatch_scope(adapter, snapshot.session_key, event):
            verified = await _revalidate_event(adapter, event)
            if (
                verified is None
                or reserved.withdrawn
                or reserved.revision != revision
                or reserved.event is not event
                or path.read_bytes() != original
                or not adapter.is_connected
                or runner._adapters_for_profile(profile).get(source.platform)
                is not adapter
            ):
                return 0
            if (
                not bind_spooled_event(
                    runner, adapter, verified, home, snapshot.session_key
                )
                or not runner._is_user_authorized_for_source(verified.source)
                or runner.session_store.resolve_session_id_for_key(
                    snapshot.session_key, not_after=snapshot.ts
                )
                != resolved
                or runner.session_store.has_input_owner(session_id, owner)
            ):
                return 0
            for item in fields(event):
                if item.init:
                    setattr(event, item.name, getattr(verified, item.name))
            event._merged_parts = verified._merged_parts
            event._pending_native_input = verified._pending_native_input
            event._prepared_inbound = None
            event._pending_execution_owner = PendingExecutionOwner(
                home, snapshot.session_key, session_id, owner
            )
            event.defer_until_idle = True
            setattr(event, "_hermes_startup_restore_replay", True)
            await adapter.handle_message(event)
            return int(getattr(event, "_gateway_accepted", False) is True)
    finally:
        release_pending_dispatch_record(adapter, snapshot.session_key, reserved)


async def _revalidate_event(
    adapter: BasePlatformAdapter, event: MessageEvent
) -> MessageEvent | None:
    if not event._merged_parts:
        return await adapter.revalidate_pending_event(pending_part(event))
    rebuilt = None
    for part, merge in list(event._merged_parts):
        current = await _revalidate_event(adapter, part)
        if current is None:
            return None
        if rebuilt is None:
            rebuilt = current
        elif merge is not None:
            merge_recorded(rebuilt, current, merge)
        else:
            return None
    return rebuilt
