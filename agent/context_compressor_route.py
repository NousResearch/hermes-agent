"""Component-owned staged route updates for the built-in context compressor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


_DURABLE_ROUTE_SQL = (
    "SELECT compression_ineffective_count, compression_fallback_streak, "
    "compression_failure_cooldown_until, compression_failure_error, model_config "
    "FROM sessions WHERE id = ?"
)
_COMPRESSOR_ROUTE_REVISION_KEY = "_compressor_route_revision"


class CompressorRouteTicketState(Enum):
    PREPARED = auto()
    COMMITTED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class CompressorRouteTarget:
    """Fully-derived live state and durable reset owned by one route update."""

    live_values: tuple[tuple[str, Any], ...]
    runtime_changed: bool


@dataclass(frozen=True)
class _LiveRouteSnapshot:
    values: tuple[tuple[str, bool, Any], ...]


@dataclass(frozen=True)
class _DurableRouteSnapshot:
    store: Any
    session_id: str
    ineffective_count: Any
    fallback_streak: Any
    cooldown_until: Any
    cooldown_error: Any
    model_config: Any


class CompressorRouteTicket:
    """Single-owner prepare/commit/abort ticket for ``ContextCompressor``."""

    def __init__(
        self,
        owner: Any,
        target: CompressorRouteTarget,
        live_snapshot: _LiveRouteSnapshot,
        owner_generation: int,
        store: Any,
        session_id: str,
        *,
        require_atomic: bool,
    ) -> None:
        self._owner = owner
        self.target = target
        self._live_snapshot = live_snapshot
        self._owner_generation = owner_generation
        self._store = store
        self._session_id = session_id
        self._require_atomic = require_atomic
        self._durable_snapshot: _DurableRouteSnapshot | None = None
        self._committed_durable_snapshot: _DurableRouteSnapshot | None = None
        self._committed_generation: int | None = None
        self._commit_started = False
        self.state = CompressorRouteTicketState.PREPARED

    def commit(self) -> None:
        if self.state is CompressorRouteTicketState.COMMITTED:
            return
        if self.state is CompressorRouteTicketState.ABORTED:
            raise RuntimeError("cannot commit an aborted compressor route ticket")
        self._owner._commit_compressor_route_ticket(self)

    def abort(self) -> None:
        if self.state is CompressorRouteTicketState.ABORTED:
            return
        if self.state is CompressorRouteTicketState.PREPARED and not self._commit_started:
            self.state = CompressorRouteTicketState.ABORTED
            return
        self._owner._abort_compressor_route_ticket(self)
        self.state = CompressorRouteTicketState.ABORTED


class ContextCompressorRouteMixin:
    """Narrow owner API; the transition coordinator sees only the ticket."""

    def prepare_route_update(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: Any = "",
        provider: str = "",
        api_mode: str = "",
        max_tokens: int | None = None,
    ) -> CompressorRouteTicket:
        return self._prepare_route_update(
            model,
            context_length,
            base_url,
            api_key,
            provider,
            api_mode,
            max_tokens,
            require_atomic=True,
        )

    def _prepare_route_update(
        self,
        model: str,
        context_length: int,
        base_url: str,
        api_key: Any,
        provider: str,
        api_mode: str,
        max_tokens: int | None,
        *,
        require_atomic: bool,
    ) -> CompressorRouteTicket:
        target = self._build_compressor_route_target(
            model, context_length, base_url, api_key, provider, api_mode, max_tokens
        )
        owner_values = vars(self)
        live_snapshot = _LiveRouteSnapshot(tuple(
            (name, name in owner_values, getattr(self, name, None))
            for name, _value in target.live_values
        ))
        return CompressorRouteTicket(
            self,
            target,
            live_snapshot,
            int(getattr(self, "_route_generation", 0)),
            getattr(self, "_session_db", None),
            getattr(self, "_session_id", "") or "",
            require_atomic=require_atomic,
        )

    @staticmethod
    def _capture_durable_route_snapshot(store: Any, session_id: str) -> _DurableRouteSnapshot | None:
        reader = getattr(store, "_read_one", None)
        if not session_id or not callable(reader):
            return None
        row = reader(_DURABLE_ROUTE_SQL, (session_id,))
        if row is None:
            return None
        return _DurableRouteSnapshot(
            store=store,
            session_id=session_id,
            ineffective_count=row["compression_ineffective_count"],
            fallback_streak=row["compression_fallback_streak"],
            cooldown_until=row["compression_failure_cooldown_until"],
            cooldown_error=row["compression_failure_error"],
            model_config=row["model_config"],
        )

    @staticmethod
    def _derive_committed_durable_snapshot(
        snapshot: _DurableRouteSnapshot | None,
        *,
        runtime_changed: bool,
        route_revision: int,
    ) -> _DurableRouteSnapshot | None:
        if snapshot is None:
            return None
        try:
            model_config = json.loads(snapshot.model_config) if snapshot.model_config else {}
        except (TypeError, json.JSONDecodeError):
            model_config = {}
        if not isinstance(model_config, dict):
            model_config = {}
        model_config.pop("_proactive_prune_rearm_tokens", None)
        model_config[_COMPRESSOR_ROUTE_REVISION_KEY] = route_revision
        return _DurableRouteSnapshot(
            store=snapshot.store,
            session_id=snapshot.session_id,
            ineffective_count=0,
            fallback_streak=0 if runtime_changed else snapshot.fallback_streak,
            cooldown_until=None if runtime_changed else snapshot.cooldown_until,
            cooldown_error=None if runtime_changed else snapshot.cooldown_error,
            model_config=json.dumps(model_config) if model_config else None,
        )

    @staticmethod
    def _restore_durable_route_snapshot(
        snapshot: _DurableRouteSnapshot | None,
        committed: _DurableRouteSnapshot | None,
    ) -> bool:
        if snapshot is None or committed is None:
            return False
        writer = getattr(snapshot.store, "_execute_write", None)
        if not callable(writer):
            raise RuntimeError("cannot compensate compressor route without atomic durable restore")

        def _restore(conn):
            cursor = conn.execute(
                "UPDATE sessions SET compression_ineffective_count = ?, "
                "compression_fallback_streak = ?, compression_failure_cooldown_until = ?, "
                "compression_failure_error = ?, model_config = ? WHERE id = ? "
                "AND compression_ineffective_count IS ? "
                "AND compression_fallback_streak IS ? "
                "AND compression_failure_cooldown_until IS ? "
                "AND compression_failure_error IS ? AND model_config IS ?",
                (
                    snapshot.ineffective_count,
                    snapshot.fallback_streak,
                    snapshot.cooldown_until,
                    snapshot.cooldown_error,
                    snapshot.model_config,
                    snapshot.session_id,
                    committed.ineffective_count,
                    committed.fallback_streak,
                    committed.cooldown_until,
                    committed.cooldown_error,
                    committed.model_config,
                ),
            )
            return cursor.rowcount == 1

        return bool(writer(_restore))

    def _restore_live_route_snapshot(self, snapshot: _LiveRouteSnapshot) -> None:
        for name, present, value in snapshot.values:
            if present:
                setattr(self, name, value)
            elif hasattr(self, name):
                delattr(self, name)

    def _apply_legacy_route_reset(self, ticket: CompressorRouteTicket) -> None:
        previous = {name: value for name, present, value in ticket._live_snapshot.values if present}
        if previous.get("_ineffective_compression_count") != 0:
            self._durable_write(
                "set_compression_ineffective_count", "compression ineffective count", 0
            )
        if ticket.target.runtime_changed:
            self._durable_write(
                "set_compression_fallback_streak", "compression fallback streak", 0
            )
            self._durable_write(
                "clear_compression_failure_cooldown", "compression failure cooldown clear"
            )
        self._durable_write(
            "patch_session_model_config",
            "proactive prune runway clear",
            {"_proactive_prune_rearm_tokens": None},
        )

    def _commit_compressor_route_ticket(self, ticket: CompressorRouteTicket) -> None:
        if int(getattr(self, "_route_generation", 0)) != ticket._owner_generation:
            raise RuntimeError("stale compressor route ticket")
        if (
            getattr(self, "_session_db", None) is not ticket._store
            or (getattr(self, "_session_id", "") or "") != ticket._session_id
        ):
            raise RuntimeError("stale compressor route ticket: session binding changed")

        atomic_reset = getattr(ticket._store, "apply_compressor_route_reset", None)
        bound = ticket._store is not None and bool(ticket._session_id)
        if bound and ticket._require_atomic and not callable(atomic_reset):
            raise RuntimeError("bound durable store lacks atomic compressor route reset")

        ticket._commit_started = True
        ticket._durable_snapshot = self._capture_durable_route_snapshot(
            ticket._store, ticket._session_id
        )
        try:
            for name, value in ticket.target.live_values:
                setattr(self, name, value)
            if bound and callable(atomic_reset):
                route_revision = atomic_reset(
                    ticket._session_id,
                    ineffective_count=0,
                    fallback_streak=0 if ticket.target.runtime_changed else None,
                    clear_failure_cooldown=ticket.target.runtime_changed,
                    clear_proactive_prune_rearm=True,
                    advance_route_revision=True,
                )
                if type(route_revision) is not int:
                    raise RuntimeError("atomic compressor route reset did not advance revision")
                ticket._committed_durable_snapshot = self._derive_committed_durable_snapshot(
                    ticket._durable_snapshot,
                    runtime_changed=ticket.target.runtime_changed,
                    route_revision=route_revision,
                )
            elif bound:
                self._apply_legacy_route_reset(ticket)
        except Exception:
            self._restore_live_route_snapshot(ticket._live_snapshot)
            ticket.state = CompressorRouteTicketState.ABORTED
            raise

        committed_generation = ticket._owner_generation + 1
        self._route_generation = committed_generation
        ticket._committed_generation = committed_generation
        ticket.state = CompressorRouteTicketState.COMMITTED

    def _abort_compressor_route_ticket(self, ticket: CompressorRouteTicket) -> None:
        if ticket._committed_generation is None:
            self._restore_live_route_snapshot(ticket._live_snapshot)
            return
        if int(getattr(self, "_route_generation", 0)) != ticket._committed_generation:
            return
        self._restore_durable_route_snapshot(
            ticket._durable_snapshot, ticket._committed_durable_snapshot
        )
        self._restore_live_route_snapshot(ticket._live_snapshot)
        self._route_generation = ticket._committed_generation + 1
