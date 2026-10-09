"""Component-owned staged reclaim for one credential-pool row."""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from enum import Enum, auto
from typing import Any, Optional


_RECLAIM_REVISION_KEY = "_credential_reclaim_revision"
_STATUS_FIELDS = (
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
    "status_cleared_at",
)


class CredentialReclaimTicketState(Enum):
    PREPARED = auto()
    COMMITTED = auto()
    ABORTED = auto()


def _snapshot(entry: Any) -> dict[str, Any]:
    return copy.deepcopy(entry.to_dict())


def _revision(entry: Any) -> int:
    raw = getattr(entry, "extra", {}).get(_RECLAIM_REVISION_KEY, 0)
    return raw if type(raw) is int and raw >= 0 else 0


def _with_revision(entry: Any, revision: int) -> Any:
    return replace(
        entry,
        extra={**getattr(entry, "extra", {}), _RECLAIM_REVISION_KEY: revision},
    )


def _durable_row(pool: Any, credential_id: str) -> Optional[dict[str, Any]]:
    from agent.credential_pool import read_credential_pool

    return next(
        (
            copy.deepcopy(row)
            for row in read_credential_pool(pool.provider)
            if isinstance(row, dict) and row.get("id") == credential_id
        ),
        None,
    )


def _eligible(pool: Any, entry: Any, model: Optional[str]) -> bool:
    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        AUTH_TYPE_OAUTH,
        STATUS_DEAD,
        STATUS_EXHAUSTED,
        _exhausted_until,
        model_cooldown_until,
    )

    if entry.last_status == STATUS_DEAD or model_cooldown_until(entry, model) is not None:
        return False
    if entry.auth_type == AUTH_TYPE_API_KEY and not entry.runtime_api_key:
        return False
    if entry.auth_type == AUTH_TYPE_OAUTH and not (entry.access_token or "").strip():
        return False
    if entry.last_status == STATUS_EXHAUSTED:
        until = _exhausted_until(entry, sole_credential=pool._is_sole_credential())
        if until is not None and time.time() < until:
            return False
    return True


class CredentialReclaimTicket:
    """Prepared reclaim of exactly one row, with owner-local compensation."""

    def __init__(
        self,
        pool: Any,
        entry: Any,
        candidate: Any,
        model: Optional[str],
        durable_basis: Optional[dict[str, Any]],
    ) -> None:
        self._pool = pool
        self._credential_id = entry.id
        self._model = model
        self._prepared_entry = entry
        self._prepared_snapshot = _snapshot(entry)
        self._durable_basis = durable_basis
        self._prepared_revision = _revision(entry)
        self._committed_revision: Optional[int] = None
        self._committed_snapshot: Optional[dict[str, Any]] = None
        self._committed_durable: Optional[dict[str, Any]] = None
        self._commit_started = False
        self.candidate = candidate
        self.state = CredentialReclaimTicketState.PREPARED

    def commit(self):
        if self.state is CredentialReclaimTicketState.COMMITTED:
            return self.candidate
        if self.state is CredentialReclaimTicketState.ABORTED:
            raise RuntimeError("cannot commit an aborted credential reclaim ticket")
        return self._pool._commit_credential_reclaim_ticket(self)

    def abort(self) -> None:
        if self.state is CredentialReclaimTicketState.ABORTED:
            return
        if self.state is CredentialReclaimTicketState.PREPARED and not self._commit_started:
            self.state = CredentialReclaimTicketState.ABORTED
            return
        self._pool._abort_credential_reclaim_ticket(self)
        self.state = CredentialReclaimTicketState.ABORTED


class CredentialPoolReclaimMixin:
    """Narrow target-only reclaim API owned by ``CredentialPool``."""

    def prepare_reclaim(
        self, credential_id: str, *, model: Optional[str] = None,
    ) -> Optional[CredentialReclaimTicket]:
        from agent.credential_pool import STATUS_OK, _CLEAR_STATUS

        with self._lock:
            entry = next((item for item in self._entries if item.id == credential_id), None)
            if entry is None or not _eligible(self, entry, model):
                return None
            candidate = replace(entry, **{**_CLEAR_STATUS, "last_status": STATUS_OK})
            durable_basis = _durable_row(self, credential_id)
            return CredentialReclaimTicket(
                self, entry, candidate, model, durable_basis,
            )

    def _commit_credential_reclaim_ticket(self, ticket: CredentialReclaimTicket):
        from agent.credential_pool import STATUS_OK, _CLEAR_STATUS, _auth_store_lock

        ticket._commit_started = True
        with _auth_store_lock():
            with self._lock:
                current = next(
                    (item for item in self._entries if item.id == ticket._credential_id), None
                )
                if (
                    current is None
                    or _snapshot(current) != ticket._prepared_snapshot
                    or _revision(current) != ticket._prepared_revision
                    or not _eligible(self, current, ticket._model)
                ):
                    raise RuntimeError("stale credential reclaim ticket")
                if _durable_row(self, ticket._credential_id) != ticket._durable_basis:
                    raise RuntimeError("stale credential reclaim ticket: durable row changed")
                committed_revision = ticket._prepared_revision + 1
                staged = _with_revision(current, committed_revision)
                self._replace_entry(current, staged)

            try:
                if self._entry_needs_refresh(staged):
                    committed = self._refresh_entry(staged, force=False)
                    if committed is None:
                        raise RuntimeError("credential reclaim refresh failed")
                    with self._lock:
                        live = next(
                            (item for item in self._entries if item.id == ticket._credential_id), None
                        )
                        if live is None or _revision(live) != committed_revision:
                            raise RuntimeError("credential reclaim changed during refresh")
                        committed = live
                        if committed.last_status != STATUS_OK:
                            committed = replace(
                                committed, **{**_CLEAR_STATUS, "last_status": STATUS_OK},
                            )
                            self._replace_entry(live, committed)
                            self._persist(status_cleared_ids=[committed.id])
                else:
                    with self._lock:
                        live = next(
                            (item for item in self._entries if item.id == ticket._credential_id), None
                        )
                        if live is None or _snapshot(live) != _snapshot(staged):
                            raise RuntimeError("credential reclaim changed before commit")
                        committed = replace(
                            live, **{**_CLEAR_STATUS, "last_status": STATUS_OK},
                        )
                        self._replace_entry(live, committed)
                        self._persist(status_cleared_ids=[committed.id])
            except Exception:
                with self._lock:
                    live = next(
                        (item for item in self._entries if item.id == ticket._credential_id), None
                    )
                    if live is not None and _snapshot(live) == _snapshot(staged):
                        self._replace_entry(live, ticket._prepared_entry)
                ticket.state = CredentialReclaimTicketState.ABORTED
                raise

            ticket._committed_revision = committed_revision
            ticket._committed_snapshot = _snapshot(committed)
            ticket._committed_durable = _durable_row(self, ticket._credential_id)
            ticket.candidate = committed
            ticket.state = CredentialReclaimTicketState.COMMITTED
            return committed

    def _abort_credential_reclaim_ticket(self, ticket: CredentialReclaimTicket) -> None:
        if ticket._committed_revision is None or ticket._committed_snapshot is None:
            return
        from agent.credential_pool import _auth_store_lock

        with _auth_store_lock():
            with self._lock:
                current = next(
                    (item for item in self._entries if item.id == ticket._credential_id), None
                )
                durable = _durable_row(self, ticket._credential_id)
                if (
                    current is None
                    or _revision(current) != ticket._committed_revision
                    or _snapshot(current) != ticket._committed_snapshot
                    or durable != ticket._committed_durable
                ):
                    return

                status_values = {
                    name: getattr(ticket._prepared_entry, name) for name in _STATUS_FIELDS
                }
                original_failure_reason = ticket._prepared_entry.extra.get("failure_reason")
                extra = dict(current.extra)
                if original_failure_reason is None:
                    extra.pop("failure_reason", None)
                else:
                    extra["failure_reason"] = original_failure_reason
                extra[_RECLAIM_REVISION_KEY] = ticket._committed_revision + 1
                compensated = replace(current, extra=extra, **status_values)
                self._replace_entry(current, compensated)
                self._persist()
