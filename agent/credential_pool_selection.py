"""Component-owned staged selection for a credential pool.

Preparation is read-only.  The normal ``select`` side effects (refresh,
request counters, round-robin order and current cursor) are delayed until the
coordinator commits this owner ticket.
"""

from __future__ import annotations

import random
from dataclasses import replace
from enum import Enum, auto
from typing import Any, Optional


class CredentialSelectionTicketState(Enum):
    PREPARED = auto()
    COMMITTED = auto()
    ABORTED = auto()


def _selection_projection(pool: Any) -> tuple[Any, tuple[tuple[Any, int, int], ...]]:
    return (
        getattr(pool, "_current_id", None),
        tuple(
            (entry.id, int(entry.priority), int(entry.request_count))
            for entry in getattr(pool, "_entries", ())
        ),
    )


class CredentialSelectionTicket:
    """Read-only preview whose pool-owned side effects happen at commit."""

    def __init__(self, pool: Any, candidate: Any, model: Optional[str]) -> None:
        self._pool = pool
        self._candidate_id = candidate.id
        self._model = model
        self._prepared_epoch = int(getattr(pool, "_mutation_epoch", 0))
        self._prepared_generation = int(
            getattr(pool, "_entry_generations", {}).get(candidate.id, 0)
        )
        self._prepared_projection = _selection_projection(pool)
        self._prepared_unmatched_streak = int(
            getattr(pool, "_unmatched_rotation_streak", 0)
        )
        self._prepared_last_empty_log = getattr(pool, "_last_no_entries_log_at", None)
        self._committed_epoch: Optional[int] = None
        self._committed_projection = None
        self.candidate = candidate
        self.state = CredentialSelectionTicketState.PREPARED

    def commit(self):
        if self.state is CredentialSelectionTicketState.COMMITTED:
            return self.candidate
        if self.state is CredentialSelectionTicketState.ABORTED:
            raise RuntimeError("cannot commit an aborted credential selection ticket")
        return self._pool._commit_credential_selection_ticket(self)

    def abort(self) -> None:
        if self.state is CredentialSelectionTicketState.ABORTED:
            return
        if self.state is CredentialSelectionTicketState.COMMITTED:
            self._pool._abort_credential_selection_ticket(self)
        self.state = CredentialSelectionTicketState.ABORTED


class CredentialPoolSelectionMixin:
    """Staged counterpart to ``CredentialPool.select``."""

    def _preview_selection_candidate(self, model: Optional[str]) -> Optional[Any]:
        from agent.credential_pool import (
            STRATEGY_LEAST_USED,
            STRATEGY_RANDOM,
        )
        from agent.credential_pool_reclaim import _eligible

        available = [
            entry for entry in self._entries if _eligible(self, entry, model)
        ]
        if not available:
            return None
        if self._strategy == STRATEGY_RANDOM:
            return random.choice(available)
        if self._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            return min(available, key=lambda entry: entry.request_count)
        return available[0]

    def prepare_selection(
        self, *, model: Optional[str] = None,
    ) -> Optional[CredentialSelectionTicket]:
        """Preview the entry ``select`` would use without mutating pool state."""
        with self._lock:
            candidate = self._preview_selection_candidate(model)
            if candidate is None:
                return None
            return CredentialSelectionTicket(self, candidate, model)

    def _validate_selection_basis(self, ticket: CredentialSelectionTicket) -> None:
        from agent.credential_pool_reclaim import _eligible

        current = next(
            (entry for entry in self._entries if entry.id == ticket._candidate_id),
            None,
        )
        if (
            current is None
            or int(getattr(self, "_mutation_epoch", 0)) != ticket._prepared_epoch
            or int(
                getattr(self, "_entry_generations", {}).get(
                    ticket._candidate_id, 0
                )
            )
            != ticket._prepared_generation
            or not _eligible(self, current, ticket._model)
        ):
            raise RuntimeError("stale credential selection ticket")

    def _commit_credential_selection_ticket(
        self, ticket: CredentialSelectionTicket,
    ) -> Any:
        with self._lock:
            self._validate_selection_basis(ticket)
        entry = None
        try:
            entry, pending_refresh = self._select_under_lock(
                model=ticket._model, preferred_id=ticket._candidate_id
            )
            if pending_refresh:
                self._refresh_pending_entries(pending_refresh)
                if entry is None:
                    entry, _ = self._select_under_lock(
                        model=ticket._model, preferred_id=ticket._candidate_id
                    )
        finally:
            # Selection/refresh may fail after partially mutating owner state.
            # Mark that state as compensatable before propagating the error.
            with self._lock:
                ticket._committed_epoch = int(getattr(self, "_mutation_epoch", 0))
                ticket._committed_projection = _selection_projection(self)
                ticket.state = CredentialSelectionTicketState.COMMITTED
        if entry is None or entry.id != ticket._candidate_id:
            raise RuntimeError("prepared credential is no longer selectable")
        self._unmatched_rotation_streak = 0
        ticket.candidate = entry
        return entry

    def _abort_credential_selection_ticket(
        self, ticket: CredentialSelectionTicket,
    ) -> None:
        """Undo only selection bookkeeping; refreshed credential material survives."""
        if ticket._committed_epoch is None or ticket._committed_projection is None:
            return
        with self._lock:
            if (
                ticket._committed_epoch == ticket._prepared_epoch
                and ticket._committed_projection == ticket._prepared_projection
            ):
                return
            if (
                int(getattr(self, "_mutation_epoch", 0)) != ticket._committed_epoch
                or _selection_projection(self) != ticket._committed_projection
            ):
                return
            prepared_current, prepared_rows = ticket._prepared_projection
            prepared_by_id = {
                entry_id: (priority, request_count)
                for entry_id, priority, request_count in prepared_rows
            }
            current_by_id = {entry.id: entry for entry in self._entries}
            if set(current_by_id) != set(prepared_by_id):
                return
            priority_changed = False
            compensated = []
            for entry_id, _priority, _request_count in prepared_rows:
                current = current_by_id[entry_id]
                priority, request_count = prepared_by_id[entry_id]
                priority_changed = priority_changed or current.priority != priority
                compensated.append(
                    replace(
                        current,
                        priority=priority,
                        request_count=request_count,
                    )
                )
            self._replace_all_entries(compensated)
            self._current_id = prepared_current
            self._unmatched_rotation_streak = ticket._prepared_unmatched_streak
            self._last_no_entries_log_at = ticket._prepared_last_empty_log
            if priority_changed:
                self._persist()
