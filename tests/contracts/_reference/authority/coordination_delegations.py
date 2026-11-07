"""The delegation-ledger verbs of the reference coordination repository.

Rows live in the scope store's ``async_delegations`` table in the shape the worker references write (``workers/delegations._dispatch_row``); ``owner_birth`` holds the owner's start time. ``task``, ``event`` and ``result`` hold the decoded JSON values, round-tripped through JSON so an unencodable payload refuses.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Sequence
from typing import Any, Optional

from persistence import DelegationId, JsonValue, SessionId
from persistence.authority.coordination import (
    DELEGATION_CLAIM_STALE_SECONDS,
    DELEGATION_FAILURE_WINDOW_SECONDS,
    DELEGATION_MAX_DELIVERY_ATTEMPTS,
    DELEGATION_MAX_DURABLE_PENDING,
    DELEGATION_MAX_RETAINED_TERMINAL,
    DELEGATION_ORPHAN_STALE_SECONDS,
    DELEGATION_REPLAY_MAX_AGE_SECONDS,
    DELEGATION_RETENTION_SECONDS,
    DelegationRecord,
    DelegationRecovery,
    DeliveryState,
    FinishedDelegationRow,
    InflightDelegation,
    OrphanCompletion,
    OwnerStamp,
    PendingCompletion,
    RecordedOwner,
)

from ..workers.delegations import _dispatch_row

logger = logging.getLogger(__name__)

__all__ = ["DelegationLedger"]

_IN_FLIGHT = ("running", "finalizing")


def _json_value(value: Any) -> Any:
    """``json.loads(json.dumps(value))``: the stored form of a payload."""
    return json.loads(json.dumps(value))


def _json_object(value: Any) -> Any:
    """The stored payload when it is an object, else ``{}``."""
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _owner(row: dict[str, Any]) -> RecordedOwner:
    return RecordedOwner(pid=row.get("owner_pid"), started_at=row.get("owner_birth"))


def _pending(row: dict[str, Any]) -> PendingCompletion:
    return PendingCompletion(delegation_id=DelegationId(row["delegation_id"]), event=copy.deepcopy(row["event"]),
                             completed_at=row["completed_at"], dispatched_at=row["dispatched_at"])


def _replay_order(row: dict[str, Any]) -> tuple[float, str]:
    # SQLite orders NULL first: a row without a completion time sorts before every timed one.
    completed = row["completed_at"]
    return (float("-inf") if completed is None else completed, row["delegation_id"])


def _drop(row: dict[str, Any], now: float) -> None:
    row.update(delivery_state=DeliveryState.DROPPED.value, delivery_claim=None, delivery_claimed_at=None, updated_at=now)


def _too_old(row: dict[str, Any], now: float) -> bool:
    age_basis = row["completed_at"] or row["dispatched_at"]
    return bool(age_basis) and (now - age_basis) > DELEGATION_REPLAY_MAX_AGE_SECONDS


def _orphan_candidate(row: dict[str, Any], now: float) -> bool:
    """An orphaned completion: not in flight, its event pending delivery, stale, and unclaimed or holding a stale claim."""
    claimed_at = row["delivery_claimed_at"]
    return (row["state"] not in _IN_FLIGHT and row["delivery_state"] == DeliveryState.PENDING
            and row["event"] is not None and row["updated_at"] < now - DELEGATION_ORPHAN_STALE_SECONDS
            and (row["delivery_claim"] is None or (claimed_at is not None and claimed_at < now - DELEGATION_CLAIM_STALE_SECONDS)))


class DelegationLedger:
    """The delegation-ledger verbs; the host class provides ``_table``."""

    def _table(self, name: str) -> dict[Any, Any]:
        raise NotImplementedError

    def _ledger(self) -> dict[str, dict[str, Any]]:
        return self._table("async_delegations")

    def record_delegation_dispatch(self, *, delegation_id: DelegationId, origin_session: str, origin_ui_session_id: str, parent_session_id: Optional[SessionId], dispatched_at: float, owner: OwnerStamp, task: JsonValue, origin_session_id: str = '') -> None:
        stored_task = _json_value(task)
        row = _dispatch_row(str(delegation_id), origin_session, origin_session_id, dispatched_at)
        row.update(origin_ui_session_id=origin_ui_session_id, parent_session_id=parent_session_id,
                   owner_pid=owner.pid, owner_birth=owner.started_at, task=stored_task, updated_at=time.time())
        self._ledger()[str(delegation_id)] = row

    def prune_delegation_ledger(self) -> None:
        ledger = self._ledger()
        cutoff = time.time() - DELEGATION_RETENTION_SECONDS
        for key in [key for key, row in ledger.items() if row["delivery_state"] == DeliveryState.DELIVERED and row["updated_at"] < cutoff]:
            del ledger[key]
        terminal = [row for row in ledger.values() if row["state"] not in _IN_FLIGHT]
        if len(terminal) > DELEGATION_MAX_RETAINED_TERMINAL:
            terminal.sort(key=lambda row: (0 if row["delivery_state"] == DeliveryState.DELIVERED else 1, row["updated_at"]))
            for row in terminal[:len(terminal) - DELEGATION_MAX_RETAINED_TERMINAL]:
                del ledger[row["delegation_id"]]
        pending = [row for row in ledger.values() if row["state"] not in _IN_FLIGHT and row["delivery_state"] == DeliveryState.PENDING]
        if len(pending) > DELEGATION_MAX_DURABLE_PENDING:
            pending.sort(key=lambda row: row["updated_at"])
            for row in pending[:len(pending) - DELEGATION_MAX_DURABLE_PENDING]:
                del ledger[row["delegation_id"]]

    def record_delegation_completion(self, *, delegation_id: DelegationId, state: str, completed_at: float, event: JsonValue, result: JsonValue) -> None:
        stored_event, stored_result = _json_value(event), _json_value(result)
        row = self._ledger().get(str(delegation_id))
        if row is not None:
            row.update(state=state, completed_at=completed_at, updated_at=time.time(), event=stored_event,
                       result=stored_result, delivery_state=DeliveryState.PENDING.value)

    def record_unit_child(self, *, delegation_id: DelegationId, entry: JsonValue) -> None:
        stored_entry = _json_value(entry)
        row = self._ledger().get(str(delegation_id))
        if row is None or row["state"] != "running":
            return
        partial = row["result"] or {}
        index = stored_entry.get("task_index") if isinstance(stored_entry, dict) else None
        results = [r for r in partial.get("results") or [] if r.get("task_index") != index]
        results.append(stored_entry)
        row.update(result={"results": results, "partial": True}, updated_at=time.time())

    def recover_abandoned_delegations(self, *, recoveries: Sequence[DelegationRecovery]) -> int:
        ledger, now, written = self._ledger(), time.time(), 0
        for recovery in recoveries:
            event, result = _json_value(recovery.event), _json_value(recovery.result)
            row = ledger.get(str(recovery.delegation_id))
            if row is None or row["state"] != recovery.expected_state or row["updated_at"] != recovery.expected_updated_at:
                continue
            row.update(state="unknown", completed_at=recovery.completed_at, updated_at=now, event=event, result=result,
                       delivery_state=DeliveryState.PENDING.value)
            written += 1
        return written

    def take_replayable_completions(self) -> tuple[PendingCompletion, ...]:
        now = time.time()
        rows = sorted((row for row in self._ledger().values() if row["state"] != "running"
                       and row["delivery_state"] == DeliveryState.PENDING and row["event"] is not None), key=_replay_order)
        return self._offer(rows, now)

    def has_replayable_completions(self) -> bool:
        return any(row["state"] != "running" and row["delivery_state"] == DeliveryState.PENDING and row["event"] is not None
                   for row in self._ledger().values())

    def _offer(self, rows: list[dict[str, Any]], now: float) -> tuple[PendingCompletion, ...]:
        """The pending replay without the queue: a row past the replay age is dropped, the rest returned."""
        offered = []
        for row in rows:
            if _too_old(row, now):
                if row["delivery_state"] == DeliveryState.PENDING:
                    _drop(row, now)
                logger.warning("Async delegation %s: pending completion is too old; terminally dropping the replay (result remains queryable).", row["delegation_id"])
                continue
            offered.append(_pending(row))
        return tuple(offered)

    def sweep_orphaned_completions(self, *, now: float, exclude_ids: frozenset[DelegationId] = frozenset(), dead_owners: frozenset[RecordedOwner], lost_ids: frozenset[DelegationId] = frozenset()) -> tuple[PendingCompletion, ...]:
        rows = sorted((row for row in self._ledger().values() if _orphan_candidate(row, now)), key=_replay_order)
        orphans = []
        for row in rows:
            if row["delegation_id"] not in lost_ids and (row["delegation_id"] in exclude_ids or _owner(row) not in dead_owners):
                continue
            if (row["delivery_attempts"] or 0) >= DELEGATION_MAX_DELIVERY_ATTEMPTS:
                _drop(row, now)
                logger.warning("Async delegation %s exhausted its %d delivery attempts; marking terminally dropped (result remains queryable).",
                               row["delegation_id"], DELEGATION_MAX_DELIVERY_ATTEMPTS)
                continue
            orphans.append(row)
        return self._offer(orphans, now)

    def list_orphan_completion_offers(self, *, now: float, exclude_ids: frozenset[DelegationId] = frozenset(), dead_owners: frozenset[RecordedOwner], lost_ids: frozenset[DelegationId] = frozenset()) -> tuple[PendingCompletion, ...]:
        rows = sorted((row for row in self._ledger().values() if _orphan_candidate(row, now)), key=_replay_order)
        return tuple(_pending(row) for row in rows
                     if (row["delegation_id"] in lost_ids or (row["delegation_id"] not in exclude_ids and _owner(row) in dead_owners))
                     and (row["delivery_attempts"] or 0) < DELEGATION_MAX_DELIVERY_ATTEMPTS and not _too_old(row, now))

    def list_orphan_completion_owners(self, *, now: float) -> frozenset[RecordedOwner]:
        return frozenset(_owner(row) for row in self._ledger().values() if _orphan_candidate(row, now))

    def list_orphan_completions(self, *, now: float) -> tuple[OrphanCompletion, ...]:
        rows = sorted((row for row in self._ledger().values() if _orphan_candidate(row, now)), key=_replay_order)
        return tuple(OrphanCompletion(delegation_id=DelegationId(row["delegation_id"]), owner=_owner(row), updated_at=row["updated_at"]) for row in rows)

    def list_inflight_delegations(self) -> tuple[InflightDelegation, ...]:
        return tuple(InflightDelegation(
            delegation_id=DelegationId(row["delegation_id"]), origin_session=row["origin_session"],
            origin_ui_session_id=row["origin_ui_session_id"], parent_session_id=row["parent_session_id"],
            dispatched_at=row["dispatched_at"], owner_pid=row.get("owner_pid"), owner_started_at=row.get("owner_birth"),
            task=copy.deepcopy(row["task"]), origin_session_id=row["origin_session_id"], result=copy.deepcopy(row["result"]),
            state=row["state"], updated_at=row["updated_at"]) for row in self._ledger().values() if row["state"] in _IN_FLIGHT)

    # ── delivery claims ──────────────────────────────────────────────────

    def _claimed(self, delegation_id: DelegationId, claim_id: str) -> Optional[dict[str, Any]]:
        row = self._ledger().get(str(delegation_id))
        if row is None or row["delivery_state"] != DeliveryState.PENDING or row["delivery_claim"] != claim_id:
            return None
        return row

    def claim_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        now = time.time()
        row = self._ledger().get(str(delegation_id))
        if row is None:
            return True
        claimed_at = row["delivery_claimed_at"]
        if row["delivery_state"] != DeliveryState.PENDING or not (
                row["delivery_claim"] is None or (claimed_at is not None and claimed_at < now - DELEGATION_CLAIM_STALE_SECONDS)):
            return False
        row.update(delivery_claim=claim_id, delivery_claimed_at=now, delivery_attempts=row["delivery_attempts"] + 1, updated_at=now)
        return True

    def release_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        row = self._claimed(delegation_id, claim_id)
        if row is None:
            return False
        now = time.time()
        if row["delivery_attempts"] >= DELEGATION_MAX_DELIVERY_ATTEMPTS:
            _drop(row, now)
            logger.warning("Async delegation %s exhausted its %d delivery attempts; marking terminally dropped (result remains queryable).",
                           delegation_id, DELEGATION_MAX_DELIVERY_ATTEMPTS)
            return True
        row.update(delivery_claim=None, delivery_claimed_at=None, updated_at=now)
        return True

    def defer_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        row = self._claimed(delegation_id, claim_id)
        if row is None:
            return False
        row.update(delivery_claim=None, delivery_claimed_at=None, delivery_attempts=max(0, row["delivery_attempts"] - 1), updated_at=time.time())
        return True

    def drop_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        row = self._claimed(delegation_id, claim_id)
        if row is None:
            return False
        _drop(row, time.time())
        return True

    def complete_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        row = self._claimed(delegation_id, claim_id)
        if row is None:
            return False
        now = time.time()
        row.update(delivery_state=DeliveryState.DELIVERED.value, delivered_at=now, updated_at=now, delivery_claim=None, delivery_claimed_at=None)
        return True

    def mark_delegation_delivered(self, *, delegation_id: DelegationId) -> bool:
        row = self._ledger().get(str(delegation_id))
        if row is None or row["delivery_state"] == DeliveryState.DELIVERED.value:
            return False
        now = time.time()
        row.update(delivery_state=DeliveryState.DELIVERED.value, delivered_at=now, updated_at=now)
        return True

    def reopen_admitted_completion(self, *, delegation_id: DelegationId) -> Optional[PendingCompletion]:
        now = time.time()
        row = self._ledger().get(str(delegation_id))
        if row is None or row["delivery_state"] != DeliveryState.DELIVERED:
            return None
        if row["delivery_attempts"] >= DELEGATION_MAX_DELIVERY_ATTEMPTS:
            row.update(delivery_state=DeliveryState.DROPPED.value, updated_at=now)
            logger.warning("Async delegation %s exhausted its %d delivery attempts; marking terminally dropped (result remains queryable).",
                           delegation_id, DELEGATION_MAX_DELIVERY_ATTEMPTS)
            return None
        if row["event"] is None:
            return None
        row.update(delivery_state=DeliveryState.PENDING.value, delivered_at=None, updated_at=now)
        offered = self._offer([row], now)
        return offered[0] if offered else None

    def get_admitted_completion(self, *, delegation_id: DelegationId) -> Optional[PendingCompletion]:
        row = self._ledger().get(str(delegation_id))
        if (row is None or row["delivery_state"] != DeliveryState.DELIVERED or row["event"] is None
                or row["delivery_attempts"] >= DELEGATION_MAX_DELIVERY_ATTEMPTS or _too_old(row, time.time())):
            return None
        return _pending(row)

    def delete_unscheduled_delegation(self, *, delegation_id: DelegationId) -> None:
        self._ledger().pop(str(delegation_id), None)

    # ── reads ────────────────────────────────────────────────────────────

    def get_delegation(self, *, delegation_id: DelegationId) -> Optional[DelegationRecord]:
        row = self._ledger().get(str(delegation_id))
        if row is None:
            return None
        return DelegationRecord(
            delegation_id=DelegationId(row["delegation_id"]), origin_session=row["origin_session"], state=row["state"],
            dispatched_at=row["dispatched_at"], completed_at=row["completed_at"], result=copy.deepcopy(row["result"]),
            delivery_state=DeliveryState(row["delivery_state"]), delivery_attempts=row["delivery_attempts"],
            origin_session_id=row["origin_session_id"] or "")

    def list_finished_delegations(self, *, origin_ui_session_id: str = '', parent_session_id: str = '', limit: int = 20, now: Optional[float] = None) -> tuple[FinishedDelegationRow, ...]:
        selectors = [(column, value) for column, value in (("origin_ui_session_id", origin_ui_session_id), ("parent_session_id", parent_session_id)) if value]
        if not selectors:
            return ()
        cutoff = (now if now is not None else time.time()) - DELEGATION_FAILURE_WINDOW_SECONDS
        rows = [row for row in self._ledger().values()
                if any(row[column] == value for column, value in selectors) and row["state"] not in _IN_FLIGHT
                and row["completed_at"] is not None and row["completed_at"] >= cutoff]
        rows.sort(key=lambda row: row["completed_at"], reverse=True)
        return tuple(FinishedDelegationRow(
            delegation_id=DelegationId(row["delegation_id"]), state=row["state"], dispatched_at=row["dispatched_at"],
            completed_at=row["completed_at"], task=_json_object(row["task"]), result=_json_object(row["result"]))
            for row in (rows if limit < 0 else rows[:limit]))
