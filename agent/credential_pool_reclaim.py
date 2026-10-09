"""Component-owned staged reclaim for one credential-pool row."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
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


@dataclass(frozen=True)
class _CredentialReclaimRefreshOutcome:
    """Unpublished target refresh result, including terminal owner mutations."""

    candidate: Optional[Any] = None
    terminal_entry: Optional[Any] = None
    terminal_removed: bool = False

    @property
    def terminal(self) -> bool:
        return self.terminal_entry is not None or self.terminal_removed


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


def _generation(pool: Any, credential_id: str) -> int:
    return int(getattr(pool, "_entry_generations", {}).get(credential_id, 0))


def _mutation_epoch(pool: Any) -> int:
    return int(getattr(pool, "_mutation_epoch", 0))


def _owning_store_path(pool: Any, credential_id: str) -> Path:
    """Exact auth.json owning this row (active profile or borrowed global root)."""
    from agent.credential_pool import (
        SINGLE_USE_REFRESH_POOL_PROVIDERS,
        _borrowed_single_use_pool_root,
    )
    import hermes_cli.auth as auth_mod

    if (
        pool.provider in SINGLE_USE_REFRESH_POOL_PROVIDERS
        and credential_id in getattr(pool, "_borrowed_root_ids", ())
    ):
        root = _borrowed_single_use_pool_root()
        if root is not None:
            return root
    return auth_mod._auth_file_path()


def _durable_row(
    pool: Any,
    credential_id: str,
    store_path: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Read one row from its exact owner instead of the profile/root merged view."""
    import hermes_cli.auth as auth_mod

    path = store_path or _owning_store_path(pool, credential_id)
    store = auth_mod._load_auth_store(path)
    rows = (store.get("credential_pool") or {}).get(pool.provider)
    if not isinstance(rows, list):
        return None
    return next(
        (
            copy.deepcopy(row)
            for row in rows
            if isinstance(row, dict) and row.get("id") == credential_id
        ),
        None,
    )


def _write_durable_row(
    pool: Any,
    credential_id: str,
    entry: Any,
    store_path: Path,
) -> dict[str, Any]:
    """Write exactly one target row while the caller holds ``store_path``'s lock."""
    import hermes_cli.auth as auth_mod
    from agent.credential_persistence import sanitize_borrowed_credential_payload

    store = auth_mod._load_auth_store(store_path)
    pool_section = store.get("credential_pool")
    if not isinstance(pool_section, dict):
        pool_section = {}
        store["credential_pool"] = pool_section
    rows = pool_section.get(pool.provider)
    rows = list(rows) if isinstance(rows, list) else []
    payload = sanitize_borrowed_credential_payload(entry.to_dict(), pool.provider)
    replaced = False
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("id") == credential_id:
            rows[index] = payload
            replaced = True
            break
    if not replaced:
        if credential_id in getattr(pool, "_borrowed_root_ids", ()):
            raise RuntimeError("borrowed credential row disappeared from its owning store")
        rows.append(payload)
    pool_section[pool.provider] = rows
    auth_mod._save_auth_store(store, target_path=store_path)
    return copy.deepcopy(payload)


def _remove_durable_row(
    pool: Any,
    credential_id: str,
    store_path: Path,
) -> None:
    """Remove exactly one target row while the caller holds ``store_path``'s lock."""
    import hermes_cli.auth as auth_mod

    store = auth_mod._load_auth_store(store_path)
    pool_section = store.get("credential_pool")
    rows = pool_section.get(pool.provider) if isinstance(pool_section, dict) else None
    if not isinstance(rows, list):
        return
    retained = [
        row for row in rows
        if not (isinstance(row, dict) and row.get("id") == credential_id)
    ]
    if len(retained) == len(rows):
        return
    pool_section[pool.provider] = retained
    auth_mod._save_auth_store(store, target_path=store_path)


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
        store_path: Path,
        generation: int,
        mutation_epoch: int,
    ) -> None:
        self._pool = pool
        self._credential_id = entry.id
        self._model = model
        self._prepared_entry = entry
        self._prepared_snapshot = _snapshot(entry)
        self._durable_basis = durable_basis
        self._store_path = store_path
        self._prepared_generation = generation
        self._prepared_mutation_epoch = mutation_epoch
        self._prepared_revision = _revision(entry)
        self._committed_revision: Optional[int] = None
        self._committed_generation: Optional[int] = None
        self._committed_mutation_epoch: Optional[int] = None
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
            store_path = _owning_store_path(self, credential_id)
            durable_basis = _durable_row(self, credential_id, store_path)
            return CredentialReclaimTicket(
                self,
                entry,
                candidate,
                model,
                durable_basis,
                store_path,
                _generation(self, credential_id),
                _mutation_epoch(self),
            )

    def _refresh_reclaim_candidate(self, entry: Any) -> _CredentialReclaimRefreshOutcome:
        """Run refresh privately and return the target result without publishing pool state."""
        from agent.credential_pool import STATUS_DEAD

        scratch = copy.copy(self)
        scratch._entries = [entry]
        scratch._lock = threading.RLock()
        scratch._entry_generations = {entry.id: _generation(self, entry.id)}
        scratch._mutation_epoch = _mutation_epoch(self)
        scratch._borrowed_root_ids = set(getattr(self, "_borrowed_root_ids", ()))
        scratch._persist = lambda **_kwargs: None
        candidate = scratch._refresh_entry(entry, force=False)
        if candidate is not None:
            return _CredentialReclaimRefreshOutcome(candidate=candidate)
        current = next(
            (item for item in scratch._entries if item.id == entry.id), None,
        )
        if current is None:
            return _CredentialReclaimRefreshOutcome(terminal_removed=True)
        if current.last_status == STATUS_DEAD:
            return _CredentialReclaimRefreshOutcome(terminal_entry=current)
        return _CredentialReclaimRefreshOutcome()

    def _publish_terminal_reclaim_outcome(
        self,
        ticket: CredentialReclaimTicket,
        outcome: _CredentialReclaimRefreshOutcome,
    ) -> None:
        """CAS-publish a terminal DEAD/removal verdict to the live and owning stores."""
        from agent.credential_pool import _auth_store_lock

        with _auth_store_lock(target_path=ticket._store_path):
            with self._lock:
                current = self._validate_reclaim_basis(ticket)
                durable = _durable_row(
                    self, ticket._credential_id, ticket._store_path,
                )
                if durable != ticket._durable_basis:
                    raise RuntimeError("stale credential reclaim ticket: durable row changed")
                if outcome.terminal_removed:
                    _remove_durable_row(
                        self, ticket._credential_id, ticket._store_path,
                    )
                    self._replace_all_entries(
                        item for item in self._entries
                        if item.id != ticket._credential_id
                    )
                    if self._current_id == ticket._credential_id:
                        self._current_id = None
                    return
                terminal_entry = outcome.terminal_entry
                if terminal_entry is None or terminal_entry.id != ticket._credential_id:
                    raise RuntimeError("invalid terminal credential reclaim outcome")
                terminal_entry = _with_revision(
                    terminal_entry, ticket._prepared_revision + 1,
                )
                _write_durable_row(
                    self,
                    ticket._credential_id,
                    terminal_entry,
                    ticket._store_path,
                )
                self._replace_entry(current, terminal_entry)

    def _validate_reclaim_basis(self, ticket: CredentialReclaimTicket) -> Any:
        current = next(
            (item for item in self._entries if item.id == ticket._credential_id), None,
        )
        if (
            current is None
            or _snapshot(current) != ticket._prepared_snapshot
            or _generation(self, ticket._credential_id) != ticket._prepared_generation
            or _mutation_epoch(self) != ticket._prepared_mutation_epoch
            or not _eligible(self, current, ticket._model)
        ):
            raise RuntimeError("stale credential reclaim ticket")
        return current

    def _commit_credential_reclaim_ticket(self, ticket: CredentialReclaimTicket):
        from agent.credential_pool import STATUS_OK, _CLEAR_STATUS, _auth_store_lock

        ticket._commit_started = True
        try:
            with self._lock:
                current = self._validate_reclaim_basis(ticket)
                needs_refresh = self._entry_needs_refresh(current)

            raw_outcome = (
                self._refresh_reclaim_candidate(current)
                if needs_refresh
                else _CredentialReclaimRefreshOutcome(candidate=current)
            )
            outcome = (
                raw_outcome
                if isinstance(raw_outcome, _CredentialReclaimRefreshOutcome)
                else _CredentialReclaimRefreshOutcome(candidate=raw_outcome)
            )
            if outcome.terminal:
                self._publish_terminal_reclaim_outcome(ticket, outcome)
                raise RuntimeError("credential reclaim refresh reached terminal state")
            refreshed = outcome.candidate
            if refreshed is None:
                raise RuntimeError("credential reclaim refresh failed")
            committed_revision = ticket._prepared_revision + 1
            committed = _with_revision(
                replace(refreshed, **{**_CLEAR_STATUS, "last_status": STATUS_OK}),
                committed_revision,
            )

            with _auth_store_lock(target_path=ticket._store_path):
                with self._lock:
                    current = self._validate_reclaim_basis(ticket)
                    durable = _durable_row(
                        self, ticket._credential_id, ticket._store_path,
                    )
                    if durable != ticket._durable_basis:
                        raise RuntimeError("stale credential reclaim ticket: durable row changed")
                    committed_durable = _write_durable_row(
                        self,
                        ticket._credential_id,
                        committed,
                        ticket._store_path,
                    )
                    self._replace_entry(current, committed)
                    committed_generation = _generation(self, ticket._credential_id)
                    committed_mutation_epoch = _mutation_epoch(self)
        except Exception:
            ticket.state = CredentialReclaimTicketState.ABORTED
            raise

        ticket._committed_revision = committed_revision
        ticket._committed_generation = committed_generation
        ticket._committed_mutation_epoch = committed_mutation_epoch
        ticket._committed_snapshot = _snapshot(committed)
        ticket._committed_durable = committed_durable
        ticket.candidate = committed
        ticket.state = CredentialReclaimTicketState.COMMITTED
        return committed

    def _abort_credential_reclaim_ticket(self, ticket: CredentialReclaimTicket) -> None:
        if (
            ticket._committed_revision is None
            or ticket._committed_generation is None
            or ticket._committed_mutation_epoch is None
            or ticket._committed_snapshot is None
        ):
            return
        from agent.credential_pool import _auth_store_lock

        with _auth_store_lock(target_path=ticket._store_path):
            with self._lock:
                current = next(
                    (item for item in self._entries if item.id == ticket._credential_id), None,
                )
                durable = _durable_row(
                    self, ticket._credential_id, ticket._store_path,
                )
                if (
                    current is None
                    or _generation(self, ticket._credential_id) != ticket._committed_generation
                    or _mutation_epoch(self) != ticket._committed_mutation_epoch
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
                _write_durable_row(
                    self,
                    ticket._credential_id,
                    compensated,
                    ticket._store_path,
                )
                self._replace_entry(current, compensated)
