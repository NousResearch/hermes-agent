"""Withdraw original constituents from profile-owned shutdown snapshots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from gateway.input_owner import gateway_input_owner
from gateway.platforms.base_pending import Withdraw
from gateway.platforms.event import MessageEvent
from gateway.session_identity import canonical_identity, clear_identity
from gateway.shutdown_pending_lock import pending_snapshot_lock
from gateway.shutdown_pending import (
    PENDING_SCHEMA,
    PendingQueueSnapshot,
    _capture_event,
)
from gateway.shutdown_pending_codec import decode_pending_event

logger = logging.getLogger("gateway.run")


def bind_spooled_event(
    runner: Any, adapter: Any, event: MessageEvent, home: Path, session_key: str
) -> bool:
    owner = gateway_input_owner(event, event.source)
    if event.internal or event.source.delivered_via_upstream_relay:
        return False
    clear_identity(event.source)
    event.source.role_authorized = False
    current = canonical_identity(event.source, runner=runner, adapter=adapter)
    if (
        current is None
        or current.adapter() is not adapter
        or current.runtime_home.resolve() != home
        or runner._session_key_for_source(event.source) != session_key
        or gateway_input_owner(event, event.source) != owner
    ):
        return False
    return all(
        bind_spooled_event(runner, adapter, part, home, session_key)
        for part, _merge in event._merged_parts
    )


def withdraw_spooled_pending(runner: Any, adapter: Any, withdraw: Withdraw) -> bool:
    from gateway.run import _multiplex_profile_homes, _profile_runtime_scope
    from hermes_constants import get_hermes_home
    from utils import atomic_json_write

    launch = get_hermes_home().resolve()
    homes = [launch]
    if runner.config.multiplex_profiles:
        homes.extend(
            Path(home).resolve()
            for _name, home in _multiplex_profile_homes(runner.config)
            if Path(home).resolve() != launch
        )
    found = False
    for home in homes:
        with _profile_runtime_scope(home):
            for path in (home / "pending_messages").glob("*.json"):
                try:
                    with pending_snapshot_lock(path):
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
                        changed = False
                        records = []
                        for record in snapshot.events:
                            records.append(record)
                            if (
                                record.get("withdrawn") is True
                                or "input_owner" not in record
                            ):
                                continue
                            routing = record.get("routing")
                            if routing is None and runner.config.multiplex_profiles:
                                continue
                            profile = (
                                routing.get("transport_profile")
                                if isinstance(routing, dict)
                                else None
                            )
                            event = decode_pending_event(record, adapter=adapter)
                            if (
                                runner._adapters_for_profile(profile).get(
                                    event.source.platform
                                )
                                is not adapter
                                or event.internal
                                or event.source.delivered_via_upstream_relay
                            ):
                                continue
                            if not bind_spooled_event(
                                runner, adapter, event, home, snapshot.session_key
                            ):
                                continue
                            matched, remaining = withdraw(event)
                            if not matched:
                                continue
                            records[-1] = (
                                {**record, "withdrawn": True}
                                if remaining is None
                                else _capture_event(remaining)
                            )
                            changed = True
                        if changed and path.read_bytes() == original:
                            atomic_json_write(
                                path, {**payload, "events": records}, mode=0o600
                            )
                            found = True
                except (OSError, ValueError, TypeError, KeyError):
                    logger.warning(
                        "Could not withdraw spooled input in %s; preserving it",
                        path,
                        exc_info=True,
                    )
    return found
