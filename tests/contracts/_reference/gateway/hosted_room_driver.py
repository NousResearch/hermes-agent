"""In-memory reference ``HostedRoomDriverRepository``: room driver leases and tasks over the placement's hosted-room tables.

Leases are ``hosted_room_driver_leases`` rows keyed by ``room_id``; tasks are ``hosted_room_driver_tasks`` rows under ``[room_id][task_id]``, both in ``hosted_room_tables``. Every write is one ``write_tables`` block, so any refusal leaves both tables as they were; the task transitions live in ``hosted_room_driver_tasks``. The caller-side validators (identifiers, generations, TTL and clock windows, settlement shapes, the payload normalization before ``admit_task``) stay with the caller; the reference re-derives the stored payload JSON and digest with the same normalization and encodes results and deferral reasons as compact JSON. Every write that changes a task runs the work-record capture (``hosted_room_work``) after its update.

Reference limits: ``prune_published_terminal_tasks`` reads ``hosted_room_policy_publications``, which the reference creates on first touch, so the absent-table case (it returns 0) never arises. A fence the guard before it already proved (the ``... changed during ...`` refusals after a matching guard) cannot fail in one atomic step; the fences without a guard (``settle_stopping_task``, ``begin_task_cancel`` on a ``deferred`` or ``stopping`` row) refuse on the fenced update itself.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Collection, Mapping
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    SHOWN_TASK_STATUSES,
    DriverLease,
    HostedPlacement,
    HostedTaskIdentity,
    InvalidTaskTransitionError,
    LeaseHeldError,
    LeaseRelease,
    RoomChangeRows,
    RoomRecovery,
    StaleLeaseError,
    TaskStatus,
)
from session_rules.hosted_tasks import MAX_TASK_PRUNE_BATCH

from ..backend import register_repository
from . import hosted_room_rules as rules
from .hosted_room_driver_tasks import (
    DriverTaskMixin,
    Row,
    lease_row_matches,
    load_task,
    require_active_lease,
    require_room_authority,
    run_fence,
    tasks_in_order,
)
from .hosted_room_tables import read_tables, task_view, write_tables
from .hosted_room_work import capture_transition_locked

__all__ = ["InMemoryHostedRoomDriverRepository"]

# Terminal-task retention: age and count.
TERMINAL_TASK_RETENTION_SECONDS = 30 * 24 * 60 * 60
MAX_RETAINED_TERMINAL_TASKS = 2048
_TERMINAL_PUBLICATIONS = frozenset({"turn.settled", "turn.failed", "turn.cancelled"})


def _lease_from_row(row: Row, *, reclaimed: bool = False) -> DriverLease:
    """The lease record of a lease row."""
    return DriverLease(
        room_id=row["room_id"], gateway_id=row["gateway_id"], authority_epoch=int(row["authority_epoch"]),
        process_generation=row["process_generation"], lease_generation=int(row["lease_generation"]),
        expires_at=float(row["expires_at"]), reclaimed=reclaimed)


def _taken_over(row: Row, *, gateway_id: str, authority_epoch: int, now: float) -> bool:
    """The takeover fence of ``acquire_lease``: another authority, or a released or expired lease."""
    return (row["gateway_id"] != gateway_id or int(row["authority_epoch"]) != authority_epoch
            or row["released_at"] is not None or float(row["expires_at"]) <= now)


class InMemoryHostedRoomDriverRepository(DriverTaskMixin):
    """Driver leases, task transitions and task reads on either placement."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def acquire_lease(self, *, placement: HostedPlacement, room_id: str, gateway_id: str, authority_epoch: int, process_generation: str, ttl_seconds: float, now: float | None = None) -> DriverLease:
        now = rules.clock(now)
        expires_at = now + float(ttl_seconds)
        with write_tables(self._store, placement) as tables:
            require_room_authority(tables, room_id, gateway_id, authority_epoch)
            leases = tables["hosted_room_driver_leases"]
            row = leases.get(room_id)
            if row is None:
                leases[room_id] = {
                    "room_id": room_id, "gateway_id": gateway_id, "authority_epoch": authority_epoch,
                    "process_generation": process_generation, "lease_generation": 1, "expires_at": expires_at,
                    "acquired_at": now, "updated_at": now, "released_at": None}
                return _lease_from_row(leases[room_id])
            same_authority = row["gateway_id"] == gateway_id and int(row["authority_epoch"]) == authority_epoch
            live = row["released_at"] is None and float(row["expires_at"]) > now
            if same_authority and row["process_generation"] == process_generation and live:
                row.update(expires_at=max(float(row["expires_at"]), expires_at), updated_at=now)
                return _lease_from_row(row)
            if same_authority and live:
                raise LeaseHeldError("room driver lease is held by another generation")
            if not _taken_over(row, gateway_id=gateway_id, authority_epoch=authority_epoch, now=now):
                raise LeaseHeldError("room driver lease changed during acquisition")
            row.update(
                gateway_id=gateway_id, authority_epoch=authority_epoch, process_generation=process_generation,
                lease_generation=int(row["lease_generation"]) + 1, expires_at=expires_at, acquired_at=now, updated_at=now,
                released_at=None)
            return _lease_from_row(row, reclaimed=True)

    def renew_lease(self, lease: DriverLease, *, placement: HostedPlacement, ttl_seconds: float, now: float | None = None) -> DriverLease:
        now = rules.clock(now)
        requested = now + float(ttl_seconds)
        with write_tables(self._store, placement) as tables:
            current = require_active_lease(tables, lease, now=now)
            expires_at = max(float(current["expires_at"]), requested)
            if ((current["gateway_id"], current["process_generation"], int(current["lease_generation"])) != run_fence(lease)
                    or current["released_at"] is not None or float(current["expires_at"]) <= now):
                raise StaleLeaseError("driver lease changed during renewal")
            current.update(expires_at=expires_at, updated_at=now)
            return dataclasses.replace(lease, expires_at=expires_at, reclaimed=False)

    def release_lease(self, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> LeaseRelease:
        now = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            require_room_authority(tables, lease.room_id, lease.gateway_id, lease.authority_epoch)
            row = tables["hosted_room_driver_leases"].get(lease.room_id)
            if row is None or not lease_row_matches(row, lease):
                raise StaleLeaseError("driver lease is stale")
            if row["released_at"] is not None:
                return LeaseRelease(lease=_lease_from_row(row), idempotent=True)
            if float(row["expires_at"]) <= now:
                raise StaleLeaseError("driver lease expired before release")
            if any(task["status"] == "running" for task in tables["hosted_room_driver_tasks"].get(lease.room_id, {}).values()):
                raise InvalidTaskTransitionError("cannot release a room lease while tasks are running")
            row.update(expires_at=now, updated_at=now, released_at=now)
            return LeaseRelease(lease=_lease_from_row(row), idempotent=False)

    def recover_room(self, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> RoomRecovery:
        now = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            require_active_lease(tables, lease, now=now)
            fence = run_fence(lease)
            for row in tasks_in_order(tables, lease.room_id, "running"):
                # NOT (run fence equals this lease's): a stored NULL makes the comparison unknown, so only a differing stored value selects the row.
                if any(stored is not None and stored != expected for stored, expected in zip((row["run_gateway_id"], row["run_process_generation"], row["run_lease_generation"]), fence, strict=True)):
                    row.update(status="indeterminate", indeterminate_at=now, updated_at=now)
            capture_transition_locked(tables, lease.room_id)
            queued, indeterminate = (
                tuple(HostedTaskIdentity(row["room_id"], row["task_id"], row["thread_id"], row["turn_id"]) for row in tasks_in_order(tables, lease.room_id, status))
                for status in ("queued", "indeterminate"))
            return RoomRecovery(queued=queued, indeterminate=indeterminate)

    def prune_published_terminal_tasks(self, *, placement: HostedPlacement, room_id: str, now: float | None = None, retention_seconds: float = TERMINAL_TASK_RETENTION_SECONDS, retain: int = MAX_RETAINED_TERMINAL_TASKS) -> int:
        now = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            published = {key[0] for key in tables["hosted_room_policy_publications"].get(room_id, {}) if key[1] in _TERMINAL_PUBLICATIONS}
            rows = [row for row in tables["hosted_room_driver_tasks"].get(room_id, {}).values()
                    if row["status"] in {"settled", "failed", "cancelled"} and row["task_id"] in published]
            # ORDER BY terminal_at DESC, task_id ASC: SQLite sorts NULL below every number, so a NULL terminal_at comes last.
            rows.sort(key=lambda row: (row["terminal_at"] is None, -float(row["terminal_at"] or 0.0), row["task_id"]))
            cutoff = now - float(retention_seconds)
            candidates = [
                str(row["task_id"]) for index, row in enumerate(rows)
                if index >= retain or (row["terminal_at"] is not None and float(row["terminal_at"]) <= cutoff)][:MAX_TASK_PRUNE_BATCH]
            tasks = tables["hosted_room_driver_tasks"].get(room_id, {})
            for task_id in candidates:
                del tasks[task_id]
            return len(candidates)

    def get_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement) -> Mapping[str, JsonValue]:
        row = load_task(read_tables(self._store, placement), identity)
        assert row is not None
        return task_view(row)

    def list_tasks(self, *, placement: HostedPlacement, room_id: str, status: TaskStatus | None = None) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(task_view(row) for row in tasks_in_order(read_tables(self._store, placement), room_id, status))

    def room_task_summary(self, *, placement: HostedPlacement, room_id: str, statuses: Collection[str]) -> tuple[Mapping[str, int], tuple[Mapping[str, JsonValue], ...]]:
        tables = read_tables(self._store, placement)
        counts = Counter(str(row["status"]) for row in tables["hosted_room_driver_tasks"].get(room_id, {}).values())
        wanted = frozenset(statuses)
        return dict(counts), tuple(task_view(row) for row in tasks_in_order(tables, room_id) if row["status"] in wanted)

    def room_change_rows(self, *, placement: HostedPlacement, statuses: Collection[str] = SHOWN_TASK_STATUSES) -> Mapping[str, RoomChangeRows]:
        tables = read_tables(self._store, placement)
        wanted = frozenset(statuses)
        return {
            room_id: RoomChangeRows(int(room["next_seq"]), int(room["revision"]), tuple(sorted(
                (str(row["task_id"]), str(row["status"]), int(row["execution_generation"]), int(row["cancel_generation"]))
                for row in tables["hosted_room_driver_tasks"].get(room_id, {}).values() if row["status"] in wanted)))
            for room_id, room in tables["hosted_rooms"].items() if room["disbanded_at"] is None}

    def get_task_for_turn(self, identity: HostedTaskIdentity, *, placement: HostedPlacement) -> Mapping[str, JsonValue] | None:
        rows = read_tables(self._store, placement)["hosted_room_driver_tasks"].get(identity.room_id, {}).values()
        row = next((row for row in rows if (row["thread_id"], row["turn_id"]) == (identity.thread_id, identity.turn_id)), None)
        return task_view(row) if row is not None else None


register_repository("hosted_room_driver", InMemoryHostedRoomDriverRepository)
