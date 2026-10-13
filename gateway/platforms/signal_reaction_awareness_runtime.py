"""Gateway lifecycle runtime for private, silent Signal reaction awareness.

This module intentionally does not route through GatewayRunner message handling or
messaging APIs. A receipt is persisted synchronously, then a bounded worker uses the
configured no-tools reaction client and stores semantic context for a later turn.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from gateway.platforms.signal_reaction_awareness import (
    ConfiguredAIAgentModelClient,
    ReactionAwarenessLane,
    _bounded_id,
)

logger = logging.getLogger(__name__)
_DB_NAME = "signal-reaction-awareness.sqlite3"
_MAX_ACTIVE_SCOPES = 4
_MAX_ATTEMPTS = 5
_RETRY_BASE_SECONDS = 0.5
_RETRY_MAX_SECONDS = 30.0
_MAINTENANCE_INTERVAL_SECONDS = 3600.0

# Adapter keys are weakly coupled to lifecycle; shutdown owns callback removal and workers.
_services: dict[int, "_Runtime"] = {}


def _trusted_owner_ids(adapter: Any) -> frozenset[str]:
    """Only explicit configured Signal identities count; '*' and empty deny by default."""
    configured = getattr(adapter, "dm_allow_from", None)
    if not isinstance(configured, (set, frozenset, list, tuple)):
        return frozenset()
    from gateway.platforms.signal import _safe_reaction_identifier
    values = {value.strip() for value in configured
              if isinstance(value, str) and value.strip() and _safe_reaction_identifier(value)}
    if "*" in configured:
        return frozenset()
    return frozenset(values)


def _reaction_awareness_enabled(user_config: Any) -> bool:
    """Require an explicit, profile-local Signal opt-in; environment allowlists do not enable it."""
    if not isinstance(user_config, dict):
        return False
    sections = [user_config.get("signal")]
    platforms = user_config.get("platforms")
    if isinstance(platforms, dict):
        sections.append(platforms.get("signal"))
    gateway = user_config.get("gateway")
    if isinstance(gateway, dict):
        gateway_platforms = gateway.get("platforms")
        if isinstance(gateway_platforms, dict):
            sections.append(gateway_platforms.get("signal"))
        sections.append(gateway.get("signal"))
    for section in sections:
        if not isinstance(section, dict):
            continue
        extra = section.get("extra")
        if section.get("reaction_awareness") is True:
            return True
        if isinstance(extra, dict) and extra.get("reaction_awareness") is True:
            return True
    return False


def _model_context(adapter: Any) -> tuple[Any, Any, dict] | None:
    # GatewayAdapterLifecycleMixin._create_adapter installs this public reference before connect.
    # Do not accept private test/integration side channels or the process-global runner fallback:
    # those can select the wrong profile in a multiplexed gateway.
    runner = getattr(adapter, "gateway_runner", None)
    owners = _trusted_owner_ids(adapter)
    build_source = getattr(adapter, "build_source", None)
    if runner is None or len(owners) != 1 or not callable(build_source):
        return None
    owner = next(iter(owners))
    try:
        # connect() is executed inside the owning profile's runtime scope at startup/reconnect.
        # Thus config loading and source construction inherit the actual profile's route.
        from gateway.run import _load_gateway_config
        user_config = _load_gateway_config()
        if not _reaction_awareness_enabled(user_config):
            return None
        source = build_source(chat_id=owner, chat_name=owner, chat_type="dm",
                              user_id=owner, user_name=owner)
    except Exception:
        logger.warning("Could not resolve Signal awareness route for configured profile", exc_info=True)
        return None
    if not isinstance(user_config, dict) or source is None:
        return None
    return runner, source, user_config


def _open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        logger.warning("Cannot enforce private permissions on Signal awareness directory")
    db = sqlite3.connect(path, timeout=5.0)
    try:
        os.chmod(path, 0o600)
    except OSError:
        db.close()
        raise
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db


class _Runtime:
    def __init__(self, adapter: Any, lane: ReactionAwarenessLane, owner_ids: frozenset[str],
                 owner_scopes: dict[str, str]):
        self.adapter, self.lane, self.owner_ids = adapter, lane, owner_ids
        self.owner_scopes = dict(owner_scopes)
        self.max_active_scopes = _MAX_ACTIVE_SCOPES
        self.tasks: dict[tuple[str, str], asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(_MAX_ACTIVE_SCOPES)
        self.closed = False
        self.profile = str(get_hermes_home().resolve())
        self.wakeups: dict[tuple[str, str], asyncio.TimerHandle] = {}
        self.maintenance_task: asyncio.Task | None = None

    def authorized_owner(self, scope: tuple[str, str]) -> str | None:
        if (self.closed or scope[0] != self.profile
                or _trusted_owner_ids(self.adapter) != self.owner_ids):
            return None
        return next((owner for owner in self.owner_ids if self.owner_scopes.get(owner) == scope[1]), None)

    def wake(self, scope: tuple[str, str]) -> None:
        self.wakeups.pop(scope, None)
        self.schedule(scope)

    def ingress(self, payload: dict[str, Any]) -> str:
        if self.closed or not isinstance(payload, dict):
            return "denied"
        if payload.get("chat_type") != "dm":
            return "denied"
        # Scope is adapter-generated from the authenticated DM route. For reaction events
        # sender must also match; target authors may be the bot account for outbound receipts.
        scope = payload.get("scope")
        owner = next((candidate for candidate in self.owner_ids
                      if scope == self.owner_scopes.get(candidate)), None)
        if owner is None or payload.get("profile") != str(get_hermes_home().resolve()):
            return "denied"
        if payload.get("kind") == "reaction":
            if payload.get("sender") != owner:
                return "denied"
        elif payload.get("kind") == "target":
            if payload.get("author") is None:
                return "denied"
        else:
            return "invalid"
        trusted = dict(payload, owner_authorized=True)
        if payload.get("kind") == "target":
            if not _bounded_id(trusted.get("profile")) or not _bounded_id(trusted.get("scope")):
                return "invalid_target"
            return "target_registered" if self.lane.register_target(
                profile=trusted.get("profile"), scope=trusted.get("scope"),
                timestamp=trusted.get("timestamp"), author=trusted.get("author"), body=trusted.get("body")) else "invalid_target"
        if payload.get("kind") != "reaction":
            return "invalid"
        status = self.lane.enqueue(trusted)
        if status == "queued":
            self.schedule((trusted["profile"], trusted["scope"]))
        return status

    def schedule(self, scope: tuple[str, str]) -> None:
        owner = self.authorized_owner(scope)
        if owner is None or scope in self.tasks or len(self.tasks) >= self.max_active_scopes:
            return
        row = self.lane.db.execute(
            "SELECT MIN(lease_until) FROM awareness_outbox WHERE profile=? AND scope=? AND sender=? "
            "AND status IN ('pending','retry','claimed')", (*scope, owner)).fetchone()
        if row is None or row[0] is None:
            return
        delay = float(row[0]) - time.time()
        existing = self.wakeups.get(scope)
        if existing is not None:
            if delay > 0 and existing.when() <= asyncio.get_running_loop().time() + delay:
                return
            existing.cancel()
            self.wakeups.pop(scope, None)
        if delay > 0:
            self.wakeups[scope] = asyncio.get_running_loop().call_later(delay, self.wake, scope)
            return
        task = asyncio.create_task(self.drain(scope), name="signal-reaction-awareness")
        self.tasks[scope] = task
        task.add_done_callback(lambda done, key=scope: self.finished(done, key))

    async def drain(self, scope: tuple[str, str]) -> None:
        while not self.closed:
            owner = self.authorized_owner(scope)
            if owner is None:
                return
            async with self.semaphore:
                result = await self.lane.process_one(
                    *scope, authorized_sender=owner,
                    authorization_check=lambda sender: self.authorized_owner(scope) == sender)
            if result in {"completed", "removed", "superseded"}:
                continue
            if result in {"retry", "tool_call_refused"}:
                row = self.lane.db.execute(
                    "SELECT attempts FROM awareness_outbox WHERE profile=? AND scope=? "
                    "AND status='retry' ORDER BY updated DESC LIMIT 1", scope).fetchone()
                attempts = int(row["attempts"]) if row else _MAX_ATTEMPTS
                if attempts < _MAX_ATTEMPTS:
                    await asyncio.sleep(min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * (2 ** (attempts - 1))))
                    continue
                self.lane.db.execute(
                    "UPDATE awareness_outbox SET status='failed',lease_until=0,updated=strftime('%s','now') "
                    "WHERE profile=? AND scope=? AND status='retry' AND attempts>=?",
                    (*scope, _MAX_ATTEMPTS))
                self.lane.db.commit()
                return
            return

    async def maintain(self) -> None:
        """Purge expired records periodically without waiting for Signal traffic."""
        while not self.closed:
            await asyncio.sleep(_MAINTENANCE_INTERVAL_SECONDS)
            if not self.closed:
                self.lane.prune_expired()

    def finished(self, task: asyncio.Task, scope: tuple[str, str]) -> None:
        if self.tasks.get(scope) is task:
            self.tasks.pop(scope, None)
        if not task.cancelled():
            try:
                error = task.exception()
            except asyncio.CancelledError:
                error = None
            if error:
                logger.warning("Signal reaction awareness worker failed",
                               exc_info=(type(error), error, error.__traceback__))
        if self.closed:
            return
        for next_scope in self.pending_scopes():
            if len(self.tasks) >= self.max_active_scopes:
                break
            self.schedule(next_scope)

    def pending_scopes(self) -> list[tuple[str, str]]:
        rows = self.lane.db.execute(
            "SELECT profile,scope,sender FROM awareness_outbox "
            "WHERE status IN ('pending','retry','claimed') GROUP BY profile,scope,sender ORDER BY MIN(created)").fetchall()
        return list(dict.fromkeys((row[0], row[1]) for row in rows
                                 if row[2] == self.authorized_owner((row[0], row[1]))))

    def get_notes(self, profile: str, scope: str) -> list[dict[str, Any]]:
        if (self.closed or not self.owner_ids or profile != str(get_hermes_home().resolve())
                or scope not in self.owner_scopes.values()):
            return []
        return self.lane.notes(profile, scope)

    async def close(self) -> None:
        self.closed = True
        for handle in self.wakeups.values():
            handle.cancel()
        self.wakeups.clear()
        for task in tuple(self.tasks.values()):
            task.cancel()
        if self.maintenance_task is not None:
            self.maintenance_task.cancel()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks.values()), return_exceptions=True)
        if self.maintenance_task is not None:
            await asyncio.gather(self.maintenance_task, return_exceptions=True)
            self.maintenance_task = None
        self.tasks.clear()
        self.lane.db.close()


async def start_awareness(adapter: Any) -> None:
    """Attach durable ingress/note callbacks and recover pending jobs after Signal connects.

    The normal GatewayRunner adapter factory supplies its public runner reference. Configuration,
    owner source, and model are resolved through the same session/runtime resolver as ordinary turns.
    """
    await stop_awareness(adapter)
    context = _model_context(adapter)
    owner_ids = _trusted_owner_ids(adapter)
    if context is None or not owner_ids:
        logger.info("Signal reaction awareness disabled: configured model or exact owner identity unavailable")
        return
    runner, source, user_config = context
    try:
        model_client = ConfiguredAIAgentModelClient.from_gateway_runner(runner, source, user_config)
        model_name = model_client.model
        path = Path(get_hermes_home()) / _DB_NAME
        db = _open_database(path)
        lane = ReactionAwarenessLane(db, model_name=model_name, model_client=model_client)
        lane.prune_expired()
        owner_scopes = {next(iter(owner_ids)): adapter._source_session_key(source)}
        runtime = _Runtime(adapter, lane, owner_ids, owner_scopes)
        runtime.maintenance_task = asyncio.create_task(
            runtime.maintain(), name="signal-reaction-awareness-maintenance")
        adapter.set_reaction_awareness_callback(runtime.ingress)
        adapter.set_reaction_awareness_note_getter(runtime.get_notes)
        _services[id(adapter)] = runtime
        for scope in runtime.pending_scopes():
            runtime.schedule(scope)
    except Exception:
        logger.exception("Signal reaction awareness runtime could not start")
        raise


async def stop_awareness(adapter: Any) -> None:
    """Detach hooks and cancel all runtime work without emitting Signal messages."""
    runtime = _services.pop(id(adapter), None)
    if runtime is not None:
        await runtime.close()
    setter = getattr(adapter, "set_reaction_awareness_callback", None)
    if callable(setter):
        setter(None)
    note_setter = getattr(adapter, "set_reaction_awareness_note_getter", None)
    if callable(note_setter):
        note_setter(None)
