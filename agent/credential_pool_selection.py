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


def _selection_projection(
    pool: Any,
) -> tuple[Any, tuple[tuple[Any, int, int], ...], int, Any]:
    return (
        getattr(pool, "_current_id", None),
        tuple(
            (entry.id, int(entry.priority), int(entry.request_count))
            for entry in getattr(pool, "_entries", ())
        ),
        int(getattr(pool, "_unmatched_rotation_streak", 0)),
        getattr(pool, "_last_no_entries_log_at", None),
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
        self._owned_before_projection = None
        self._owned_after_epoch: Optional[int] = None
        self._owned_after_projection = None
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
        entry = None
        with self._lock:
            self._validate_selection_basis(ticket)
            available, pending_refresh = self._available_entries(
                clear_expired=True,
                refresh=True,
                model=ticket._model,
            )
            available_ids = {candidate.id for candidate in available}
            pending_ids = {candidate.id for candidate in pending_refresh}
            if ticket._candidate_id in available_ids:
                entry, pending_refresh = self._select_ticket_candidate_locked(ticket)
            elif ticket._candidate_id not in pending_ids:
                raise RuntimeError("prepared credential is no longer selectable")

        if pending_refresh:
            self._refresh_pending_entries(pending_refresh)
        if entry is None:
            with self._lock:
                entry, _ = self._select_ticket_candidate_locked(ticket)
        if entry is None or entry.id != ticket._candidate_id:
            raise RuntimeError("prepared credential is no longer selectable")
        ticket.candidate = entry
        return entry

    def _select_ticket_candidate_locked(
        self,
        ticket: CredentialSelectionTicket,
    ) -> tuple[Any, list[Any]]:
        """Own one selection delta while validation/selection stay atomic."""
        ticket._owned_before_projection = _selection_projection(self)
        try:
            entry, pending_refresh = self._select_unlocked(
                model=ticket._model,
                preferred_id=ticket._candidate_id,
            )
            if entry is not None:
                self._unmatched_rotation_streak = 0
            return entry, pending_refresh
        finally:
            # Selection may raise after partially changing bookkeeping.  The
            # immediate before/after pair contains only this ticket's atomic
            # delta; refreshes and prior concurrent selections are baselines.
            ticket._owned_after_epoch = int(getattr(self, "_mutation_epoch", 0))
            ticket._owned_after_projection = _selection_projection(self)
            ticket.state = CredentialSelectionTicketState.COMMITTED

    def _abort_credential_selection_ticket(
        self, ticket: CredentialSelectionTicket,
    ) -> None:
        """Undo only selection bookkeeping; refreshed credential material survives."""
        if (
            ticket._owned_before_projection is None
            or ticket._owned_after_epoch is None
            or ticket._owned_after_projection is None
        ):
            return
        with self._lock:
            if (
                int(getattr(self, "_mutation_epoch", 0))
                != ticket._owned_after_epoch
                or _selection_projection(self) != ticket._owned_after_projection
            ):
                return
            if ticket._owned_after_projection == ticket._owned_before_projection:
                return
            (
                before_current,
                before_rows,
                before_unmatched_streak,
                before_last_empty_log,
            ) = ticket._owned_before_projection
            before_by_id = {
                entry_id: (priority, request_count)
                for entry_id, priority, request_count in before_rows
            }
            current_by_id = {entry.id: entry for entry in self._entries}
            if set(current_by_id) != set(before_by_id):
                return
            priority_changed = False
            rows_changed = False
            compensated = []
            for entry_id, _priority, _request_count in before_rows:
                current = current_by_id[entry_id]
                priority, request_count = before_by_id[entry_id]
                priority_changed = priority_changed or current.priority != priority
                rows_changed = rows_changed or (
                    current.priority != priority
                    or current.request_count != request_count
                )
                compensated.append(
                    replace(
                        current,
                        priority=priority,
                        request_count=request_count,
                    )
                )
            if rows_changed:
                self._replace_all_entries(compensated)
            self._current_id = before_current
            self._unmatched_rotation_streak = before_unmatched_streak
            self._last_no_entries_log_at = before_last_empty_log
            if priority_changed:
                self._persist()
