"""The task-transition verbs of the in-memory ``HostedRoomDriverRepository`` and the driver's shared row steps.

A transition is one write: an optional lease check (before the row load on the lease-first paths, after the replay on the settlement paths), the row load with its identity check, the idempotent replay, the guard, then the fenced update, whose fence is tested against the row with SQL ``WHERE`` semantics, and the work-record capture.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    TERMINAL_STATUSES,
    DriverLease,
    HostedPlacement,
    HostedTaskIdentity,
    InvalidTaskTransitionError,
    RoomUnavailableError,
    StaleLeaseError,
    StaleTaskError,
    TaskAttempt,
    TaskConflictError,
    TerminalStatus,
)
from session_rules.hosted_tasks import compact_json, normalize_task_payload

from . import hosted_room_rules as rules
from .hosted_room_tables import HostedTables, task_view, write_tables
from .hosted_room_work import capture_transition_locked

__all__ = [
    "DriverTaskMixin",
    "Row",
    "active_room",
    "lease_row_matches",
    "load_task",
    "require_active_lease",
    "require_room_authority",
    "run_fence",
    "tasks_in_order",
]

Row = dict[str, Any]
Replay = Callable[[Row], "dict[str, Any] | None"]

_STARTED_STATUSES = frozenset({"running", "indeterminate", "stopping"})
_REQUEUE: Row = {"status": "queued", "run_gateway_id": None, "run_process_generation": None, "run_lease_generation": None}
# Transition name -> (fenced status, guard stale message, fence stale message).
_INDETERMINATE_STALE = "indeterminate task generation changed"
_GENERATION_TRANSITIONS = {
    "resolve": ("indeterminate", _INDETERMINATE_STALE, "indeterminate task changed during reconciliation"),
    "resolve_cancel": ("indeterminate", "indeterminate cancellation proof is stale", "indeterminate cancellation proof lost its fence"),
    "requeue": ("indeterminate", _INDETERMINATE_STALE, "indeterminate task changed during requeue"),
    "defer": ("indeterminate", _INDETERMINATE_STALE, "indeterminate task changed during deferral"),
    "defer_stopping": ("stopping", "stopping task generation changed", "stopping task changed during deferral"),
    "reopen_deferred": ("deferred", "deferred task generation changed", "deferred task changed during reopen"),
    "requeue_deferred": ("deferred", "deferred task generation changed", "deferred task changed during requeue")}


def tasks_in_order(tables: HostedTables, room_id: str, status: str | None = None) -> list[Row]:
    """The room's task rows, optionally of one status, ordered by ``source_event_seq``, ``created_at``, ``task_id``."""
    rows = tables["hosted_room_driver_tasks"].get(room_id, {}).values()
    return sorted((row for row in rows if status is None or row["status"] == status),
                  key=lambda row: (int(row["source_event_seq"]), float(row["created_at"]), row["task_id"]))


def active_room(tables: HostedTables, room_id: str) -> Row:
    """The room row, else ``RoomUnavailableError`` when it is missing or disbanded."""
    row = tables["hosted_rooms"].get(room_id)
    if row is None or row["disbanded_at"] is not None:
        raise RoomUnavailableError("hosted room does not exist" if row is None else "hosted room is disbanded")
    return row


def require_room_authority(tables: HostedTables, room_id: str, gateway_id: str, epoch: int) -> Row:
    """The active room when its authority is ``(gateway_id, epoch)``, else ``StaleLeaseError``."""
    room = active_room(tables, room_id)
    if room["authority_gateway_id"] != gateway_id or int(room["authority_epoch"]) != epoch:
        raise StaleLeaseError("hosted room authority changed")
    return room


def run_fence(lease: DriverLease) -> tuple[str, str, int]:
    """The run fence a lease stamps on a started task: ``(gateway_id, process_generation, lease_generation)``."""
    return lease.gateway_id, lease.process_generation, lease.lease_generation


def lease_row_matches(row: Row | None, lease: DriverLease) -> bool:
    """Whether the lease row names this lease: gateway, authority epoch, process generation and lease generation."""
    return row is not None and (row["gateway_id"], int(row["authority_epoch"]), row["process_generation"], int(row["lease_generation"])) == (
        lease.gateway_id, lease.authority_epoch, lease.process_generation, lease.lease_generation)


def require_active_lease(tables: HostedTables, lease: DriverLease, *, now: float) -> Row:
    """Room authority, then the matching, unreleased, unexpired lease row."""
    require_room_authority(tables, lease.room_id, lease.gateway_id, lease.authority_epoch)
    row = tables["hosted_room_driver_leases"].get(lease.room_id)
    if not lease_row_matches(row, lease) or row["released_at"] is not None or float(row["expires_at"]) <= now:
        raise StaleLeaseError("driver lease is stale or expired")
    return row


def _row_identity(row: Row) -> HostedTaskIdentity:
    return HostedTaskIdentity(row["room_id"], row["task_id"], row["thread_id"], row["turn_id"])


def load_task(tables: HostedTables, identity: HostedTaskIdentity, *, required: bool = True) -> Row | None:
    """The task row of ``identity``; ``TaskConflictError`` when it is missing (and required) or bound to another turn."""
    row = tables["hosted_room_driver_tasks"].get(identity.room_id, {}).get(identity.task_id)
    if row is None:
        if required:
            raise TaskConflictError("task does not exist")
        return None
    if _row_identity(row) != identity:
        raise TaskConflictError("task_id is already bound to a different turn")
    return row


def _generations_match(row: Row, status: str, execution_generation: int, cancel_generation: int) -> bool:
    return (row["status"], int(row["execution_generation"]), int(row["cancel_generation"])) == (status, execution_generation, cancel_generation)


def _run_fence_of(row: Row) -> tuple[Any, Any, Any]:
    return row["run_gateway_id"], row["run_process_generation"], row["run_lease_generation"]


def _settlement(settlement_id: str, status: str, result: JsonValue, now: float) -> tuple[Replay, Row]:
    """The settlement after the caller's checks: the replay and the settlement columns."""
    result_json = compact_json(result)

    def replay(row: Row) -> dict[str, Any] | None:
        if row["settlement_id"] is None:
            return None
        if (row["settlement_id"], row["settlement_status"], row["result_json"]) == (settlement_id, status, result_json):
            return task_view(row, idempotent=True)
        raise TaskConflictError("task already has a different terminal settlement")

    return replay, {"status": status, "settlement_id": settlement_id, "settlement_status": status, "result_json": result_json, "terminal_at": now, "updated_at": now}


def _cancel_replay(cancel_id: str, status: str = "cancelled") -> Replay:
    """The idempotent replay of a cancel already applied with ``cancel_id``."""
    return lambda row: task_view(row, idempotent=True) if row["status"] == status and row["cancel_id"] == cancel_id else None


def _new_task(identity: HostedTaskIdentity, *, source_event_seq: int, payload_json: str, payload_digest: str, now: float) -> Row:
    """The queued ``hosted_room_driver_tasks`` row ``admit_task`` inserts."""
    return {
        "room_id": identity.room_id, "task_id": identity.task_id, "thread_id": identity.thread_id, "turn_id": identity.turn_id,
        "source_event_seq": source_event_seq, "payload_json": payload_json, "payload_digest": payload_digest, "status": "queued",
        "execution_generation": 0, "cancel_generation": 0, "run_gateway_id": None, "run_process_generation": None,
        "run_lease_generation": None, "cancel_id": None, "settlement_id": None, "settlement_status": None, "result_json": None,
        "created_at": now, "updated_at": now, "started_at": None, "terminal_at": None, "indeterminate_at": None}


class DriverTaskMixin:
    """Task admission, start and every fenced transition; the repository provides ``_store``."""

    _store: Any

    def _transition(self, placement: HostedPlacement, identity: HostedTaskIdentity, *, changes: Row, fence: Callable[[Row], bool], stale: str, now: float, lease: DriverLease | None = None, lease_first: bool = True, replay: Replay | None = None, guard: Callable[[Row], None] | None = None) -> dict[str, Any]:
        """One transition in one write: lease check, row load, replay, guard, then the fenced update."""
        with write_tables(self._store, placement) as tables:
            if lease is not None and lease_first:
                require_active_lease(tables, lease, now=now)
            row = load_task(tables, identity)
            assert row is not None
            if replay is not None and (replayed := replay(row)) is not None:
                return replayed
            if lease is not None and not lease_first:
                require_active_lease(tables, lease, now=now)
            if guard is not None:
                guard(row)
            if not fence(row):
                raise StaleTaskError(stale)
            row.update(changes)
            capture_transition_locked(tables, identity.room_id)
            return task_view(row)

    def _generation_transition(self, placement: HostedPlacement, identity: HostedTaskIdentity, lease: DriverLease, name: str, execution_generation: int, cancel_generation: int, *, changes: Row, now: float, replay: Replay | None = None) -> dict[str, Any]:
        """Lease first, fenced on status and both generations."""
        status, generation_stale, stale = _GENERATION_TRANSITIONS[name]

        def matches(row: Row) -> bool:
            return _generations_match(row, status, execution_generation, cancel_generation)

        def guard(row: Row) -> None:
            if not matches(row):
                raise StaleTaskError(generation_stale)

        return self._transition(placement, identity, lease=lease, now=now, replay=replay, guard=guard, fence=matches, changes=changes, stale=stale)

    def _run_fence_transition(self, placement: HostedPlacement, attempt: TaskAttempt, *, guard_stale: str, lease_generation: Callable[[Any], int] = int, **transition: Any) -> dict[str, Any]:
        """The guard casts the stored ``run_lease_generation`` with ``lease_generation``; the fence compares it as stored."""
        fence = (attempt.execution_generation, attempt.cancel_generation, *run_fence(attempt.lease))

        def guard(row: Row) -> None:
            stored = (row["run_gateway_id"], row["run_process_generation"], lease_generation(row["run_lease_generation"]))
            if not _generations_match(row, "running", attempt.execution_generation, attempt.cancel_generation) or stored != run_fence(attempt.lease):
                raise StaleTaskError(guard_stale)

        def fenced(row: Row) -> bool:
            return row["status"] == "running" and (int(row["execution_generation"]), int(row["cancel_generation"]), *_run_fence_of(row)) == fence

        return self._transition(placement, attempt.identity, lease=attempt.lease, guard=guard, fence=fenced, **transition)

    def admit_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, payload: Mapping[str, JsonValue], now: float | None = None) -> Mapping[str, JsonValue]:
        normalized, payload_json, payload_digest = normalize_task_payload(dict(payload))
        now = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            active_room(tables, identity.room_id)
            existing = load_task(tables, identity, required=False)
            if existing is not None:
                if existing["payload_digest"] != payload_digest or existing["payload_json"] != payload_json:
                    raise TaskConflictError("task_id is already bound to a different payload")
                return task_view(existing, idempotent=True)
            rows = tables.rows("hosted_room_driver_tasks", identity.room_id)
            if any((row["thread_id"], row["turn_id"]) == (identity.thread_id, identity.turn_id) for row in rows.values()):
                raise TaskConflictError("thread_id and turn_id are already bound to a task")
            rows[identity.task_id] = _new_task(identity, source_event_seq=normalized["source_event_seq"], payload_json=payload_json, payload_digest=payload_digest, now=now)
            capture_transition_locked(tables, identity.room_id)
            return task_view(rows[identity.task_id])

    def start_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_cancel_generation: int, now: float | None = None) -> TaskAttempt:
        now = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            require_active_lease(tables, lease, now=now)
            row = load_task(tables, identity)
            assert row is not None
            if int(row["cancel_generation"]) != expected_cancel_generation:
                raise StaleTaskError("task cancellation generation changed")
            if row["status"] != "queued":
                raise InvalidTaskTransitionError(f"cannot start task in state '{row['status']}'")
            ordered = tasks_in_order(tables, identity.room_id)
            if any(task["status"] in _STARTED_STATUSES for task in ordered):
                raise InvalidTaskTransitionError("room recovery must resolve the prior task before starting new work")
            queued = [task for task in ordered if task["status"] == "queued"]
            if not queued or queued[0]["task_id"] != identity.task_id:
                raise InvalidTaskTransitionError("task is not next in the hosted room event order")
            execution_generation = int(row["execution_generation"]) + 1
            if row["status"] != "queued" or int(row["cancel_generation"]) != expected_cancel_generation:
                raise StaleTaskError("task changed during start")
            gateway_id, process_generation, lease_generation = run_fence(lease)
            row.update(status="running", execution_generation=execution_generation, run_gateway_id=gateway_id, run_process_generation=process_generation, run_lease_generation=lease_generation, started_at=now, updated_at=now)
            capture_transition_locked(tables, identity.room_id)
            return TaskAttempt(identity=identity, lease=lease, execution_generation=execution_generation, cancel_generation=expected_cancel_generation)

    def settle_task(self, attempt: TaskAttempt, *, placement: HostedPlacement, settlement_id: str, status: TerminalStatus, result: JsonValue, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)
        replay, changes = _settlement(settlement_id, status, result, now)
        return self._run_fence_transition(placement, attempt, guard_stale="task attempt is stale or cancelled", lease_first=False, now=now, replay=replay, changes=changes, stale="task changed during settlement")

    def settle_stopping_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, settlement_id: str, status: TerminalStatus, result: JsonValue, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)
        replay, changes = _settlement(settlement_id, status, result, now)

        def fence(row: Row) -> bool:
            return _generations_match(row, "stopping", expected_execution_generation, expected_cancel_generation)

        return self._transition(placement, identity, lease=lease, lease_first=False, now=now, replay=replay, fence=fence, changes=changes, stale="task completion lost the stop race")

    def resolve_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, settlement_id: str, status: TerminalStatus, result: JsonValue, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)
        replay, changes = _settlement(settlement_id, status, result, now)
        return self._generation_transition(placement, identity, lease, "resolve", expected_execution_generation, expected_cancel_generation, now=now, replay=replay, changes=changes)

    def resolve_indeterminate_cancellation(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, cancel_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)
        changes = {"status": "cancelled", "cancel_generation": expected_cancel_generation + 1, "cancel_id": cancel_id, "terminal_at": now, "updated_at": now}
        return self._generation_transition(placement, identity, lease, "resolve_cancel", expected_execution_generation, expected_cancel_generation, now=now, replay=_cancel_replay(cancel_id), changes=changes)

    def requeue_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)
        changes = {**_REQUEUE, "started_at": None, "indeterminate_at": None, "updated_at": now}
        return self._generation_transition(placement, identity, lease, "requeue", expected_execution_generation, expected_cancel_generation, now=now, changes=changes)

    def defer_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, reason: str, now: float | None = None) -> Mapping[str, JsonValue]:
        return self._defer(placement, identity, lease, "defer", expected_execution_generation, expected_cancel_generation, reason=reason, now=now)

    def defer_unconfirmed_stop(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, reason: str, now: float | None = None) -> Mapping[str, JsonValue]:
        return self._defer(placement, identity, lease, "defer_stopping", expected_execution_generation, expected_cancel_generation, reason=reason, now=now)

    def _defer(self, placement: HostedPlacement, identity: HostedTaskIdentity, lease: DriverLease, name: str, expected_execution_generation: int, expected_cancel_generation: int, *, reason: str, now: float | None) -> dict[str, Any]:
        """The deferral of ``name``'s status as retryable; the same deferral replays."""
        now = rules.clock(now)
        result_json = compact_json({"reason": reason, "retryable": True})

        def replay(row: Row) -> dict[str, Any] | None:
            deferred = _generations_match(row, "deferred", expected_execution_generation, expected_cancel_generation)
            return task_view(row, idempotent=True) if deferred and row["result_json"] == result_json else None

        changes = {"status": "deferred", "result_json": result_json, "terminal_at": now, "updated_at": now}
        return self._generation_transition(placement, identity, lease, name, expected_execution_generation, expected_cancel_generation, now=now, replay=replay, changes=changes)

    def requeue_deferred_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)
        changes = {**_REQUEUE, "result_json": None, "started_at": None, "terminal_at": None, "indeterminate_at": None, "updated_at": now}
        return self._generation_transition(placement, identity, lease, "requeue_deferred", expected_execution_generation, expected_cancel_generation, now=now, changes=changes)

    def reopen_deferred_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement, expected_execution_generation: int, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)
        changes = {"status": "indeterminate", "result_json": None, "terminal_at": None, "started_at": now, "indeterminate_at": now, "updated_at": now}
        return self._generation_transition(placement, identity, lease, "reopen_deferred", expected_execution_generation, expected_cancel_generation, now=now, changes=changes)

    def mark_attempt_indeterminate(self, attempt: TaskAttempt, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> None:
        now = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            require_active_lease(tables, lease, now=now)
            row = load_task(tables, attempt.identity)
            assert row is not None
            if not _generations_match(row, "running", attempt.execution_generation, attempt.cancel_generation):
                return
            row.update(status="indeterminate", indeterminate_at=now, updated_at=now)
            capture_transition_locked(tables, lease.room_id)

    def requeue_not_admitted_task(self, attempt: TaskAttempt, *, placement: HostedPlacement, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)

        def replay(row: Row) -> dict[str, Any] | None:
            requeued = _generations_match(row, "queued", attempt.execution_generation, attempt.cancel_generation) and _run_fence_of(row) == (None, None, None)
            return task_view(row, idempotent=True) if requeued else None

        return self._run_fence_transition(
            placement, attempt, guard_stale="not-admitted task attempt lost its fence", lease_generation=lambda value: int(value or 0), now=now,
            replay=replay, changes={**_REQUEUE, "started_at": None, "updated_at": now}, stale="not-admitted task changed during requeue")

    def cancel_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)

        def guard(row: Row) -> None:
            if row["status"] in TERMINAL_STATUSES:
                raise InvalidTaskTransitionError(f"cannot cancel task in state '{row['status']}'")
            if row["status"] not in {"queued", "deferred"}:
                raise InvalidTaskTransitionError("running work requires acknowledged two-phase cancellation")
            if int(row["cancel_generation"]) != expected_cancel_generation:
                raise StaleTaskError("task cancellation generation changed")

        def fence(row: Row) -> bool:
            return row["status"] in {"queued", "deferred"} and int(row["cancel_generation"]) == expected_cancel_generation

        changes = {"status": "cancelled", "cancel_generation": expected_cancel_generation + 1, "cancel_id": cancel_id, "terminal_at": now, "updated_at": now}
        return self._transition(placement, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, fence=fence, changes=changes, stale="task changed during cancellation")

    def begin_task_cancel(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)

        def guard(row: Row) -> None:
            if row["status"] in TERMINAL_STATUSES or row["status"] == "queued":
                raise InvalidTaskTransitionError(f"cannot request remote stop in state '{row['status']}'")
            if int(row["cancel_generation"]) != expected_cancel_generation:
                raise StaleTaskError("task cancellation generation changed")

        def fence(row: Row) -> bool:
            return row["status"] in {"running", "indeterminate"} and int(row["cancel_generation"]) == expected_cancel_generation

        changes = {"status": "stopping", "cancel_generation": expected_cancel_generation + 1, "cancel_id": cancel_id, "updated_at": now}
        return self._transition(placement, identity, now=now, replay=_cancel_replay(cancel_id, "stopping"), guard=guard, fence=fence, changes=changes, stale="task changed during stop request")

    def complete_task_cancel(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str, expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = rules.clock(now)

        def fence(row: Row) -> bool:
            return (row["status"], row["cancel_id"], int(row["cancel_generation"])) == ("stopping", cancel_id, expected_cancel_generation)

        def guard(row: Row) -> None:
            if not fence(row):
                raise StaleTaskError("task stop acknowledgement is stale")

        return self._transition(placement, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, fence=fence, changes={"status": "cancelled", "terminal_at": now, "updated_at": now}, stale="task changed during stop acknowledgement")
