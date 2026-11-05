"""Shared lease-fence verification for the in-memory reference backend.

``MessageRepository.compact`` is lease-fenced: when the caller passes a ``lock_holder``, the operation re-verifies inside its own atomic step that the holder still owns the live compression lock on the session and raises ``CompressionInProgress`` otherwise, without mutating anything. The fence reads the same ``coord_leases`` table the coordination repository's compression lock writes, so a holder whose lock was lost (expiry, reclaim, release) fails the fence. QUIRK(lease-aba): the check is holder identity, not a generation, so a holder that lost and reacquired the lease passes.

``MessageRepository.replace_transcript(reject_active_turn_lease=True)`` reads the same two tables holder-blind: :func:`refuse_live_leases` refuses on any live row, :func:`drop_stale_leases` reclaims stale ones once every refusal has passed.

This module underscores its name because it is fixture plumbing, not a registered capability; capability modules import it without modifying each other or ``backend.py``.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from persistence import CompressionInProgress, ProfileScope, TurnLeaseLost
from persistence.core.identity import SessionId

from .backend import InMemoryBackend

__all__ = ["drop_stale_leases", "lease_guarded", "refuse_live_leases", "verify_lease", "verify_turn_lease"]


def verify_lease(
    backend: InMemoryBackend,
    scope: ProfileScope,
    session_id: SessionId,
    holder: Optional[str],
) -> None:
    """Raise ``CompressionInProgress`` unless *holder* is ``None`` or names the live compression lock on this session; an expired lock is lost, never renewed."""
    if holder is None:
        return
    leases: dict[str, tuple[str, float]] = backend.scope_store(scope).setdefault(
        "coord_leases", {}
    )
    live = leases.get(str(session_id))
    if live is None or live[1] <= time.time() or live[0] != holder:
        raise CompressionInProgress(session_id)


def verify_turn_lease(
    backend: InMemoryBackend,
    scope: ProfileScope,
    session_id: SessionId,
    holder: Optional[str],
    ttl_seconds: float,
    *,
    renew: bool = True,
) -> None:
    """The turn-lease fence of ``MessageRepository.append`` and ``append_batch``: raise ``TurnLeaseLost`` unless *holder* is ``None`` or empty or the stored turn lease on this session's conversation still names it, even if expired; with *renew*, extend an expired row that still names the holder for *ttl_seconds* (the starved-refresher rule). Callers that must refuse for another reason after the lease check (the closed-parent refusal runs after the fence) verify with ``renew=False`` first and renew only once every refusal has passed, so a refused append commits no renewal.

    Reads and writes the same ``coord_turn_leases`` table the coordination reference owns, keyed by the same lineage-root resolution, so a holder whose ``try_acquire_turn_lease`` row was reclaimed or released fails here. Runs inside the caller's critical section, under the backend lock the caller holds from fence to insert.
    """
    if not holder:
        return
    from .authority.coordination import conversation_key

    key = conversation_key(backend, scope, str(session_id))
    leases: dict[str, tuple[str, float]] = backend.scope_store(scope).setdefault("coord_turn_leases", {})
    live = leases.get(key)
    if live is None or live[0] != holder:
        raise TurnLeaseLost(session_id)
    now = time.time()
    if renew and live[1] <= now:
        leases[key] = (holder, now + max(0.1, float(ttl_seconds)))


def _lease_rows(backend: InMemoryBackend, scope: ProfileScope, session_id: SessionId) -> list[tuple[Literal["turn", "compaction"], dict[str, tuple[str, float]], str, bool]]:
    """``(kind, table, key, live)`` for the compaction lease on the session and the turn lease on its conversation, in that check order; absent rows are omitted. Stale (``live`` False) = expired or a provably dead local holder."""
    from .authority.coordination import _holder_process_is_dead, conversation_key

    store = backend.scope_store(scope)
    now = time.time()
    rows: list[tuple[Literal["turn", "compaction"], dict[str, tuple[str, float]], str, bool]] = []
    for kind, table_name, key in (
        ("compaction", "coord_leases", str(session_id)),
        ("turn", "coord_turn_leases", conversation_key(backend, scope, str(session_id))),
    ):
        table: dict[str, tuple[str, float]] = store.setdefault(table_name, {})
        row = table.get(key)
        if row is not None:
            rows.append((kind, table, key, not (row[1] <= now or _holder_process_is_dead(row[0]))))
    return rows


def refuse_live_leases(backend: InMemoryBackend, scope: ProfileScope, session_id: SessionId) -> None:
    """The holder-blind reject of ``replace_transcript(reject_active_turn_lease=True)``: a live compaction lease on the session refuses first, then a live turn lease on its conversation. Leaves lease rows unchanged but may initialize empty tables; the caller drops stale rows with :func:`drop_stale_leases` only after every refusal has passed."""
    for kind, _table, _key, live in _lease_rows(backend, scope, session_id):
        if live:
            raise CompressionInProgress(session_id) if kind == "compaction" else TurnLeaseLost(session_id, f"Session has an active turn lease; refusing transcript mutation for {str(session_id)!r}")


def lease_guarded(backend: InMemoryBackend, scope: ProfileScope, session_id: SessionId) -> bool:
    """The guard of ``SessionRepository.delete(refuse_leased=True)`` and ``MaintenanceRepository.prune(refuse_leased=True)``: a live compaction lease on the session or a live turn lease on its conversation, holder-blind. Leaves rows unchanged but may initialize empty tables."""
    return any(live for _kind, _table, _key, live in _lease_rows(backend, scope, session_id))


def drop_stale_leases(backend: InMemoryBackend, scope: ProfileScope, session_id: SessionId) -> None:
    """Remove the stale compaction and turn lease rows :func:`refuse_live_leases` passed over (the reclaim, which also fences the stale holder's late writes)."""
    for _kind, table, key, live in _lease_rows(backend, scope, session_id):
        if not live:
            del table[key]
