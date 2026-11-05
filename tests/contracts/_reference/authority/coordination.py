"""In-memory coordination repository for the reference backend (test fixture only).

Implements ``persistence.authority.coordination.CoordinationRepository`` in memory. Storage inside one scope store: ``coord_leases`` maps a session id to the compression lock ``(holder, expires_at)``; ``coord_turn_leases`` maps a conversation key to the turn lease ``(holder, expires_at)`` (the message, maintenance and worker references read both through ``_lease_fence``); ``hygiene_streaks`` maps a session key to its streak; ``handoffs`` maps a session id to its handoff columns (``status``, ``platform``, ``error``), which count only while the session row exists; ``async_delegations`` is the delegation ledger the worker references share (``coordination_delegations.py``); ``delivery_obligations`` is the obligation ledger (``coordination_obligations.py``).

The turn-lease key is the lineage root (:func:`conversation_key`); the dead-holder probe is :func:`_holder_process_is_dead` (strict namespace rule: on Linux a holder without this process's ``pidns=`` stamp is never probed).

Reference limits: every verb runs under the backend lock, so ``patience_s`` has no effect and :class:`persistence.StoreContention` never arises; the store-failure arms of the lease and handoff verbs are reached only through a closed backend.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Any, Optional

from persistence import PersistenceError, ProfileScope, SessionId, StoreReplaced
from persistence.authority.coordination import (
    TURN_LEASE_SWEEP_GRACE_SECONDS,
    HandoffState,
    HandoffStatus,
    PendingHandoff,
    turn_lease_holder_pid,
)
from persistence.conversation.sessions import END_REASON_COMPRESSION, SessionLineageKind

from .._lineage_intent import direct_lineage_kind
from ..backend import InMemoryBackend, register_repository
from .coordination_delegations import DelegationLedger
from .coordination_obligations import ObligationLedger

try:
    import psutil
except ImportError:  # pragma: no cover - optional probe dependency
    psutil = None

logger = logging.getLogger(__name__)

__all__ = ["InMemoryCoordinationRepository", "conversation_key", "pidns_token"]

_PIDNS_TOKEN = re.compile(r"(?:^|:)pidns=(\d+)(?::|$)")
_STORE_FAILURES = (RuntimeError, PersistenceError)


def _local_pid_namespace() -> tuple[Optional[str], bool]:
    """``(id, supported)`` of this process's PID namespace."""
    if sys.platform != "linux":
        return None, False
    try:
        link = os.readlink("/proc/self/ns/pid")
    except OSError:
        return None, True
    match = re.search(r"\[(\d+)\]", link)
    return (match.group(1) if match else None), True


def pidns_token() -> str:
    """The ``:pidns=<id>`` fragment a holder string carries, or ``""``."""
    ns, _supported = _local_pid_namespace()
    return f":pidns={ns}" if ns else ""


def _holder_pid_checkable(holder: str) -> bool:
    """Strict: probe only a holder stamped with this namespace, or any holder where namespaces do not exist."""
    ns, supported = _local_pid_namespace()
    if not supported:
        return True
    match = _PIDNS_TOKEN.search(holder or "")
    if match is None or ns is None:
        return False
    return match.group(1) == ns


def _holder_process_is_dead(holder: str) -> bool:
    """True only when the holder's ``pid=<n>`` names a local process that is provably gone: the current process, an unstructured or foreign-namespace holder, an absurd pid and any probe doubt read as alive."""
    pid = turn_lease_holder_pid(holder)
    if pid is None or pid == os.getpid() or not _holder_pid_checkable(holder):
        return False
    if psutil is not None:
        try:
            return not bool(psutil.pid_exists(pid))
        except Exception:
            return False
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, OverflowError):
        return False
    return False


def _is_explicit_fork_child(row: dict[str, Any]) -> bool:
    """An explicit fork child, resets included: a tool child, or a row whose ``_branched_from`` / ``_delegate_from`` / ``_reset_from`` marker names its parent (a marker only counts when it points at the parent)."""
    if row.get("source") == "tool" or direct_lineage_kind(row) in (SessionLineageKind.BRANCH, SessionLineageKind.DELEGATE, SessionLineageKind.RESET):
        return True
    cfg = row.get("model_config") or {}
    markers = (cfg.get("_branched_from"), cfg.get("_delegate_from"), cfg.get("_reset_from"))
    parent_id = row.get("parent_session_id")
    return parent_id in markers if parent_id else any(m is not None for m in markers)


def conversation_key(backend: InMemoryBackend, scope: ProfileScope, session_id: str) -> str:
    """The turn-lease key: walk to the parent while the current row is a plain continuation and the parent ended by compression; an unknown id is its own key."""
    table: dict[str, dict[str, Any]] = backend.scope_store(scope).setdefault("sessions", {})
    current = table.get(session_id)
    key = session_id
    seen = {session_id}
    while current is not None:
        parent_id = current.get("parent_session_id")
        if not parent_id or parent_id in seen or _is_explicit_fork_child(current):
            break
        parent = table.get(parent_id)
        if parent is None or parent.get("end_reason") != END_REASON_COMPRESSION:
            break
        seen.add(parent_id)
        key = parent_id
        current = parent
    return key


def _claim_lease_row(leases: dict[str, tuple[str, float]], key: str, holder: str, expires_at: float, stale: Any) -> bool:
    """Delete a stale row, insert-or-ignore ours, then whether the row names ``holder``."""
    row = leases.get(key)
    if row is not None and stale(row[0], row[1]):
        del leases[key]
    leases.setdefault(key, (holder, expires_at))
    return leases[key][0] == holder


class InMemoryCoordinationRepository(DelegationLedger, ObligationLedger):
    """Reference implementation; see ``persistence/authority/coordination.py`` for the contract text."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    def _table(self, name: str) -> dict[Any, Any]:
        table: dict[Any, Any] = self._backend.scope_store(self._backend.scope).setdefault(name, {})
        return table

    def _key(self, session_id: str) -> str:
        return conversation_key(self._backend, self._backend.scope, session_id)

    # ── compression lock ─────────────────────────────────────────────────

    def try_acquire_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        if not session_id:
            return False
        now = time.time()
        try:
            self._backend.backend.require_unreplaced(self._backend.scope)
            leases = self._table("coord_leases")
            row = leases.get(str(session_id))
            acquired = _claim_lease_row(leases, str(session_id), holder, now + ttl_seconds, lambda h, e: e < now or _holder_process_is_dead(h))
        except StoreReplaced:
            raise
        except _STORE_FAILURES as exc:
            logger.warning("try_acquire_compression_lock(%s) failed: %s", session_id, exc)
            return False
        if row is not None and leases.get(str(session_id)) != row:
            logger.warning("Reclaimed stale compression lock for session=%s (holder=%s)", session_id, row[0])
        return acquired

    def refresh_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        if not session_id or not holder:
            return False
        try:
            self._backend.backend.require_unreplaced(self._backend.scope)
            leases = self._table("coord_leases")
            row = leases.get(str(session_id))
            if row is None or row[0] != holder:
                return False
            leases[str(session_id)] = (holder, time.time() + ttl_seconds)
            return True
        except StoreReplaced:
            raise
        except _STORE_FAILURES as exc:
            logger.warning("refresh_compression_lock(%s) failed: %s", session_id, exc)
            return False

    def release_compression_lock(self, *, session_id: SessionId, holder: str) -> None:
        if not session_id:
            return
        try:
            self._backend.backend.require_unreplaced(self._backend.scope)
            leases = self._table("coord_leases")
            if leases.get(str(session_id), ("",))[0] == holder and str(session_id) in leases:
                del leases[str(session_id)]
        except StoreReplaced:
            raise
        except _STORE_FAILURES as exc:
            logger.warning("release_compression_lock(%s) failed: %s", session_id, exc)

    def get_compression_lock_holder(self, *, session_id: SessionId) -> Optional[str]:
        if not session_id:
            return None
        row = self._table("coord_leases").get(str(session_id))
        return row[0] if row is not None and row[1] >= time.time() else None

    # ── turn lease ───────────────────────────────────────────────────────

    def try_acquire_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0, patience_s: Optional[float] = None) -> bool:
        if not session_id or not holder:
            return False
        now = time.time()
        leases = self._table("coord_turn_leases")
        for key in [key for key, (_holder, expires) in leases.items() if expires < now - TURN_LEASE_SWEEP_GRACE_SECONDS]:
            del leases[key]
        return _claim_lease_row(leases, self._key(str(session_id)), holder, now + max(0.1, float(ttl_seconds)),
                                lambda h, e: float(e) <= now or _holder_process_is_dead(h))

    def refresh_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0, patience_s: Optional[float] = None) -> bool:
        if not session_id or not holder:
            return False
        expires_at = time.time() + max(0.1, float(ttl_seconds))
        leases = self._table("coord_turn_leases")
        key = self._key(str(session_id))
        if leases.get(key, ("",))[0] != holder or key not in leases:
            return False
        leases[key] = (holder, expires_at)
        return True

    def release_turn_lease(self, *, session_id: SessionId, holder: str) -> None:
        if not session_id or not holder:
            return
        leases = self._table("coord_turn_leases")
        key = self._key(str(session_id))
        if key in leases and leases[key][0] == holder:
            del leases[key]

    def turn_lease_expires_at(self, *, session_id: SessionId, holder: str) -> Optional[float]:
        if not session_id or not holder:
            return None
        row = self._table("coord_turn_leases").get(self._key(str(session_id)))
        return float(row[1]) if row is not None and row[0] == holder else None

    # ── hygiene streak ───────────────────────────────────────────────────

    def increment_hygiene_failure_streak(self, *, session_key: str) -> int:
        if not session_key:
            return 1
        streaks = self._table("hygiene_streaks")
        streaks[session_key] = streaks.get(session_key, 0) + 1
        return int(streaks[session_key])

    def reset_hygiene_failure_streak(self, *, session_key: str) -> None:
        if session_key:
            self._table("hygiene_streaks").pop(session_key, None)

    # ── handoffs ─────────────────────────────────────────────────────────

    def _handoff_row(self, session_id: str) -> Optional[dict[str, Any]]:
        """The handoff columns of an existing session row (created empty on first touch), else None."""
        if session_id not in self._table("sessions"):
            return None
        row: dict[str, Any] = self._table("handoffs").setdefault(session_id, {"status": None, "platform": None, "error": None})
        return row

    def _live_handoffs(self) -> list[tuple[str, dict[str, Any]]]:
        sessions = self._table("sessions")
        return [(sid, row) for sid, row in self._table("handoffs").items() if sid in sessions]

    def request_handoff(self, *, session_id: SessionId, platform: str) -> bool:
        row = self._handoff_row(str(session_id))
        if row is None or row["status"] not in (None, HandoffStatus.COMPLETED, HandoffStatus.FAILED):
            return False
        row.update(status=HandoffStatus.PENDING.value, platform=platform, error=None)
        return True

    def claim_handoff(self, *, session_id: SessionId) -> bool:
        row = self._handoff_row(str(session_id))
        if row is None or row["status"] != HandoffStatus.PENDING:
            return False
        row["status"] = HandoffStatus.RUNNING.value
        return True

    def complete_handoff(self, *, session_id: SessionId) -> None:
        row = self._handoff_row(str(session_id))
        if row is not None:
            row.update(status=HandoffStatus.COMPLETED.value, error=None)

    def fail_handoff(self, *, session_id: SessionId, error: str, only_states: Optional[tuple[HandoffStatus, ...]] = None) -> bool:
        row = self._handoff_row(str(session_id))
        if row is None or (only_states and row["status"] not in {str(state) for state in only_states}):
            return False
        row.update(status=HandoffStatus.FAILED.value, error=error[:500])
        return True

    def reclaim_stale_running_handoffs(self, *, error: str) -> tuple[SessionId, ...]:
        try:
            running = [(sid, row) for sid, row in self._live_handoffs() if row["status"] == HandoffStatus.RUNNING]
        except _STORE_FAILURES:
            logger.warning("reclaim_stale_running_handoffs failed; stale running handoffs stay in place", exc_info=True)
            return ()
        for _sid, row in running:
            row.update(status=HandoffStatus.FAILED.value, error=error[:500])
        return tuple(SessionId(sid) for sid, _row in running)

    def get_handoff(self, *, session_id: SessionId) -> Optional[HandoffState]:
        try:
            if str(session_id) not in self._table("sessions"):
                return None
            row = self._table("handoffs").get(str(session_id))
        except _STORE_FAILURES:
            return None
        if row is None or row["status"] is None:
            return None
        return HandoffState(state=HandoffStatus(row["status"]), platform=row["platform"], error=row["error"])

    def list_pending_handoffs(self) -> tuple[PendingHandoff, ...]:
        try:
            sessions = self._table("sessions")
            pending = [(sessions[sid], sid, row) for sid, row in self._live_handoffs() if row["status"] == HandoffStatus.PENDING]
        except _STORE_FAILURES:
            return ()
        pending.sort(key=lambda item: item[0]["started_at"].timestamp())
        return tuple(PendingHandoff(session_id=SessionId(sid), platform=row["platform"], title=session.get("title"),
                                    started_at=session["started_at"].timestamp()) for session, sid, row in pending)

    def has_pending_handoffs(self) -> bool:
        return any(row["status"] == HandoffStatus.PENDING for _sid, row in self._live_handoffs())


register_repository("coordination", InMemoryCoordinationRepository, {"transactional_claims": True})
