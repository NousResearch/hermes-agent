"""``HostedRoomDriverRepository`` on SQLite: the room driver's fenced leases and task state machine in the hosted-room
tables of either file, each verb one ``BEGIN IMMEDIATE`` transaction or one
read context on the placement's file.

The caller-side validators (identifiers, generations, TTL and clock windows, settlement shapes, the payload
normalization before ``admit_task``) run in the gateway before each call; the backend re-derives the stored payload
JSON and digest with the same normalization and encodes results as compact JSON. Every write that changes a
task runs the work-record capture after its update in the same transaction.

Every driver verb opens the file with ``DRIVER_STEP`` only, without the rooms step: existing tables get the task-status migration when their CHECK is old and are validated on every open, so an incompatible draft fails the driver verbs only. ``room_change_rows`` is the exception: it reads the rooms table and the task table only where it exists, so it opens the file with ``ROOMS_STEP`` and a reader answers a file whose driver tables were never set up.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Callable

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    SHOWN_TASK_STATUSES,
    TERMINAL_STATUSES,
    DriverLease,
    DriverStateError,
    DriverValidationError,
    HostedPlacement,
    HostedTaskIdentity,
    InvalidTaskTransitionError,
    LeaseHeldError,
    LeaseRelease,
    RoomChangeRows,
    RoomRecovery,
    RoomUnavailableError,
    StaleLeaseError,
    StaleTaskError,
    TaskAttempt,
    TaskConflictError,
    TaskStatus,
    TerminalStatus,
)
from session_rules.hosted_rooms import clock, compact_json
from session_rules.hosted_tasks import (
    MAX_RETAINED_TERMINAL_TASKS, MAX_TASK_PRUNE_BATCH, TERMINAL_TASK_RETENTION_SECONDS, normalize_task_payload,
)

from .._backend import SQLiteStore, register_repository, translated
from ._hosted_rooms_common import fenced_update, table_columns
from ._hosted_rooms_schema import ROOMS_STEP
from ._hosted_store import HostedFiles, SchemaStep
from ._hosted_work import capture_transition_locked

__all__ = ["SQLiteHostedRoomDriverRepository", "_task_from_row"]

_LEASE_COLUMNS = frozenset({
    "room_id", "gateway_id", "authority_epoch", "process_generation", "lease_generation", "expires_at", "acquired_at",
    "updated_at", "released_at"})
_TASK_COLUMN_ORDER = (
    "room_id", "task_id", "thread_id", "turn_id", "source_event_seq", "payload_json", "payload_digest", "status",
    "execution_generation", "cancel_generation", "run_gateway_id", "run_process_generation", "run_lease_generation",
    "cancel_id", "settlement_id", "settlement_status", "result_json", "created_at", "updated_at", "started_at",
    "terminal_at", "indeterminate_at")
_TASK_COLUMNS = frozenset(_TASK_COLUMN_ORDER)
_TASK_ORDER = "ORDER BY source_event_seq, created_at, task_id"
_SELECT_LEASE = "SELECT * FROM hosted_room_driver_leases WHERE room_id=?"
_SELECT_TASK = "SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?"
_TASK_INDEX_SQL = """CREATE INDEX {if_not_exists}idx_hosted_room_driver_tasks_status
           ON hosted_room_driver_tasks(room_id, status, source_event_seq, created_at, task_id)"""

# --- Fenced task UPDATE statements (one per state-machine transition) ---------
# Every transition is "UPDATE ... SET <set> WHERE room_id=? AND task_id=? AND <fence>"; the fence names the
# expected status plus the generations that must not have moved.
_GENERATION_FENCE = "execution_generation=? AND cancel_generation=?"
_RUN_FENCE = "run_gateway_id=? AND run_process_generation=? AND run_lease_generation=?"
_SETTLE_SET = "status=?, settlement_id=?, settlement_status=?, result_json=?, terminal_at=?, updated_at=?"
_REQUEUE_SET = "status='queued', run_gateway_id=NULL, run_process_generation=NULL, run_lease_generation=NULL"
_CANCEL_SET = "status='cancelled', cancel_generation=?, cancel_id=?, terminal_at=?, updated_at=?"


def _task_update(set_clause: str, fence: str) -> str:
    return f"UPDATE hosted_room_driver_tasks SET {set_clause} WHERE room_id=? AND task_id=? AND {fence}"


def _generation_update(set_clause: str, status: str) -> str:
    """Transition fenced on ``status`` + both generations (terminal settlements and the recovery family)."""
    return _task_update(set_clause, f"status='{status}' AND {_GENERATION_FENCE}")


_SETTLE_RUNNING_SQL = _generation_update(_SETTLE_SET, "running") + f" AND {_RUN_FENCE}"
_SETTLE_STOPPING_SQL = _generation_update(_SETTLE_SET, "stopping")
_REQUEUE_RUNNING_SQL = _task_update(
    f"{_REQUEUE_SET}, started_at=NULL, updated_at=?", f"status='running' AND {_GENERATION_FENCE} AND {_RUN_FENCE}")
_CANCEL_QUEUED_SQL = _task_update(_CANCEL_SET, "status IN ('queued', 'deferred') AND cancel_generation=?")
_BEGIN_STOP_SQL = _task_update(
    "status='stopping', cancel_generation=?, cancel_id=?, updated_at=?",
    "status IN ('running', 'indeterminate') AND cancel_generation=?")
_COMPLETE_STOP_SQL = _task_update(
    "status='cancelled', terminal_at=?, updated_at=?", "status='stopping' AND cancel_id=? AND cancel_generation=?")

# Lease-first recovery transitions: name -> (fenced status, SET clause, generation-guard stale message,
# row stale message); the UPDATE is _generation_update(set_clause, status).
_INDETERMINATE_STALE = "indeterminate task generation changed"
_GENERATION_TRANSITIONS = {
    "resolve": ("indeterminate", _SETTLE_SET, _INDETERMINATE_STALE, "indeterminate task changed during reconciliation"),
    "resolve_cancel": (
        "indeterminate", _CANCEL_SET, "indeterminate cancellation proof is stale",
        "indeterminate cancellation proof lost its fence"),
    "requeue": (
        "indeterminate", f"{_REQUEUE_SET}, started_at=NULL, indeterminate_at=NULL, updated_at=?", _INDETERMINATE_STALE,
        "indeterminate task changed during requeue"),
    "defer": (
        "indeterminate", "status='deferred', result_json=?, terminal_at=?, updated_at=?", _INDETERMINATE_STALE,
        "indeterminate task changed during deferral"),
    "defer_stopping": (
        "stopping", "status='deferred', result_json=?, terminal_at=?, updated_at=?",
        "stopping task generation changed", "stopping task changed during deferral"),
    "reopen_deferred": (
        "deferred",
        "status='indeterminate', result_json=NULL, terminal_at=NULL, started_at=?, indeterminate_at=?, updated_at=?",
        "deferred task generation changed", "deferred task changed during reopen"),
    "requeue_deferred": (
        "deferred",
        f"{_REQUEUE_SET}, result_json=NULL, started_at=NULL, terminal_at=NULL, indeterminate_at=NULL, updated_at=?",
        "deferred task generation changed", "deferred task changed during requeue")}

Row = sqlite3.Row
Replay = Callable[[Row], "dict[str, Any] | None"]


# ── schema (``DRIVER_STEP``) ──────────────────────────────────────────────────


# The task table under ``table``; the status-constraint migration builds its replacement under another name.
def _create_task_table(conn: sqlite3.Connection, table: str = "hosted_room_driver_tasks") -> None:
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {table} (
            room_id TEXT NOT NULL, task_id TEXT NOT NULL, thread_id TEXT NOT NULL, turn_id TEXT NOT NULL,
            source_event_seq INTEGER NOT NULL CHECK (source_event_seq >= 1),
            payload_json TEXT NOT NULL, payload_digest TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN (
                'queued', 'running', 'settled', 'failed', 'cancelled', 'indeterminate', 'deferred', 'stopping')),
            execution_generation INTEGER NOT NULL DEFAULT 0 CHECK (execution_generation >= 0),
            cancel_generation INTEGER NOT NULL DEFAULT 0 CHECK (cancel_generation >= 0),
            run_gateway_id TEXT, run_process_generation TEXT, run_lease_generation INTEGER, cancel_id TEXT,
            settlement_id TEXT, settlement_status TEXT, result_json TEXT, created_at REAL NOT NULL,
            updated_at REAL NOT NULL, started_at REAL, terminal_at REAL, indeterminate_at REAL,
            PRIMARY KEY (room_id, task_id), UNIQUE (room_id, thread_id, turn_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id))""")


# Creates the lease and task tables, validates them, then adds the status index.
def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_driver_leases (
            room_id TEXT PRIMARY KEY, gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1), process_generation TEXT NOT NULL,
            lease_generation INTEGER NOT NULL CHECK (lease_generation >= 1),
            expires_at REAL NOT NULL, acquired_at REAL NOT NULL, updated_at REAL NOT NULL, released_at REAL,
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id))""")
    _create_task_table(conn)
    _validate_schema(conn)
    conn.execute(_TASK_INDEX_SQL.format(if_not_exists="IF NOT EXISTS "))


# Refuses driver tables whose columns or ``hosted_rooms`` foreign keys differ from this schema.
def _validate_schema(conn: sqlite3.Connection) -> None:
    columns = table_columns(conn, "hosted_room_driver_leases"), table_columns(conn, "hosted_room_driver_tasks")
    if columns != (_LEASE_COLUMNS, _TASK_COLUMNS):
        raise DriverStateError(
            "unsupported unpublished hosted-room driver schema; "
            "recreate the driver tables before starting the driver")
    for table in ("hosted_room_driver_leases", "hosted_room_driver_tasks"):
        if not any(
            row[2] == "hosted_rooms" and row[3] == "room_id" and row[4] == "room_id"
            for row in conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()):
            raise DriverStateError(f"{table} is missing its hosted_rooms foreign key")


# True when both driver tables and the status index exist.
def _schema_objects_exist(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("""SELECT name FROM sqlite_master WHERE type='table'
           AND name IN ('hosted_room_driver_leases', 'hosted_room_driver_tasks')""").fetchall()
    if {row[0] for row in rows} != {"hosted_room_driver_leases", "hosted_room_driver_tasks"}:
        return False
    index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_hosted_room_driver_tasks_status'").fetchone()
    return index is not None


def _migrate_task_status_constraint(conn: sqlite3.Connection) -> None:
    """Expand the unpublished task-state CHECK without losing durable work."""
    conn.execute("DROP INDEX IF EXISTS idx_hosted_room_driver_tasks_status")
    _create_task_table(conn, "hosted_room_driver_tasks_next")
    columns = ", ".join(_TASK_COLUMN_ORDER)
    conn.execute(
        f"INSERT INTO hosted_room_driver_tasks_next ({columns}) SELECT {columns} FROM hosted_room_driver_tasks")
    conn.execute("DROP TABLE hosted_room_driver_tasks")
    conn.execute("ALTER TABLE hosted_room_driver_tasks_next RENAME TO hosted_room_driver_tasks")
    conn.execute(_TASK_INDEX_SQL.format(if_not_exists=""))


def _schema_ready(conn: sqlite3.Connection, path: Path) -> bool:
    """The step's ``ready``: both tables and the status index exist and the task-status CHECK covers the current
    states. Existing tables are validated on every open, so an incompatible table refuses every driver verb."""
    if not _schema_objects_exist(conn):
        return False
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='hosted_room_driver_tasks'").fetchone()
    sql = str(row[0] or "").lower() if row else ""
    if not ("'stopping'" in sql and "'deferred'" in sql):  # task-status CHECK already covers the current states
        return False
    _validate_schema(conn)
    return True


def _initialize_store(conn: sqlite3.Connection, path: Path) -> None:
    """The step's ``initialize``: the status-constraint migration and validation of existing tables, else the schema."""
    if _schema_objects_exist(conn):
        _migrate_task_status_constraint(conn)
        _validate_schema(conn)
    else:
        _initialize_schema(conn)


DRIVER_STEP: SchemaStep = (_schema_ready, _initialize_store)


# ── row helpers ───────────────────────────────────────────────────────────────


def _lease_from_row(row: Row | dict[str, Any], *, reclaimed: bool = False) -> DriverLease:
    return DriverLease(
        room_id=row["room_id"], gateway_id=row["gateway_id"], authority_epoch=int(row["authority_epoch"]),
        process_generation=row["process_generation"], lease_generation=int(row["lease_generation"]),
        expires_at=float(row["expires_at"]), reclaimed=reclaimed)


def _task_identity_from_row(row: Row) -> HostedTaskIdentity:
    return HostedTaskIdentity(
        room_id=row["room_id"], task_id=row["task_id"], thread_id=row["thread_id"], turn_id=row["turn_id"])


def _optional(cast: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: cast(value) if value is not None else None


# Task-view casts per row column (columns after payload_digest in _TASK_COLUMN_ORDER, same key order;
# result_json is exposed as "result"). Columns not listed are passed through untouched.
_TASK_VIEW_CASTS: dict[str, Callable[[Any], Any]] = {
    "execution_generation": int, "cancel_generation": int, "run_lease_generation": _optional(int),
    "result_json": _optional(json.loads), "created_at": float, "updated_at": float, "started_at": _optional(float),
    "terminal_at": _optional(float), "indeterminate_at": _optional(float)}


def _task_from_row(row: Row, *, idempotent: bool = False) -> dict[str, Any]:
    """The task view of one stored row; a stored payload that fails its normalization or integrity check refuses
    ``TaskConflictError``."""
    try:
        payload, encoded_payload, payload_digest = normalize_task_payload(json.loads(row["payload_json"]))
    except (TypeError, json.JSONDecodeError, DriverValidationError) as exc:
        raise TaskConflictError("stored task payload is invalid") from exc
    if (encoded_payload, payload_digest, payload["source_event_seq"]) != (
        row["payload_json"], row["payload_digest"], int(row["source_event_seq"])):
        raise TaskConflictError("stored task payload failed its integrity check")
    task: dict[str, Any] = {"identity": _task_identity_from_row(row), "payload": payload}
    for column in _TASK_COLUMN_ORDER[_TASK_COLUMN_ORDER.index("payload_digest"):]:
        task["result" if column == "result_json" else column] = _TASK_VIEW_CASTS.get(column, lambda v: v)(row[column])
    task["idempotent"] = idempotent
    return task


# The stored task of ``identity``; refuses a task id bound to another turn, and a missing task when ``required``.
def _load_task(conn: sqlite3.Connection, identity: HostedTaskIdentity, *, required: bool = True) -> Row | None:
    row = conn.execute(_SELECT_TASK, (identity.room_id, identity.task_id)).fetchone()
    if row is None:
        if required:
            raise TaskConflictError("task does not exist")
        return None
    if _task_identity_from_row(row) != identity:
        raise TaskConflictError("task_id is already bound to a different turn")
    return row


# The room's tasks, optionally of one status, in event order.
def _tasks_in_order(conn: sqlite3.Connection, room_id: str, status: str | None = None) -> list[Row]:
    where, params = ("", (room_id,)) if status is None else (" AND status=?", (room_id, status))
    sql = f"SELECT * FROM hosted_room_driver_tasks WHERE room_id=?{where} {_TASK_ORDER}"
    return conn.execute(sql, params).fetchall()


# The room row; refuses a missing or disbanded room, a missing room table included.
def _load_active_room(conn: sqlite3.Connection, room_id: str) -> Row:
    try:
        row = conn.execute(
            "SELECT room_id, authority_gateway_id, authority_epoch, disbanded_at FROM hosted_rooms WHERE room_id=?",
            (room_id,)).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            raise RoomUnavailableError("hosted room does not exist") from exc
        raise
    if row is None or row["disbanded_at"] is not None:
        raise RoomUnavailableError("hosted room does not exist" if row is None else "hosted room is disbanded")
    return row


def _require_room_authority(conn: sqlite3.Connection, room_id: str, gateway_id: str, epoch: int) -> Row:
    room = _load_active_room(conn, room_id)
    if room["authority_gateway_id"] != gateway_id or int(room["authority_epoch"]) != epoch:
        raise StaleLeaseError("hosted room authority changed")
    return room


def _run_fence(lease: DriverLease) -> tuple[str, str, int]:
    """The ``_RUN_FENCE`` bind order: (gateway_id, process_generation, lease_generation)."""
    return lease.gateway_id, lease.process_generation, lease.lease_generation


def _lease_row_matches(row: Row | None, lease: DriverLease) -> bool:
    return row is not None and (
        row["gateway_id"], int(row["authority_epoch"]), row["process_generation"], int(row["lease_generation"])
    ) == (lease.gateway_id, lease.authority_epoch, lease.process_generation, lease.lease_generation)


# The lease row; refuses a changed room authority or a lease that is stale, released or (unless ``allow_expired``)
# expired.
def _require_active_lease(conn: sqlite3.Connection, lease: DriverLease, *, now: float, allow_expired: bool = False) -> Row:
    _require_room_authority(conn, lease.room_id, lease.gateway_id, lease.authority_epoch)
    row = conn.execute(_SELECT_LEASE, (lease.room_id,)).fetchone()
    if not _lease_row_matches(row, lease) or row["released_at"] is not None or (
        not allow_expired and float(row["expires_at"]) <= now
    ):
        raise StaleLeaseError("driver lease is stale or expired")
    return row


def _settlement(settlement_id: str, status: str, result: JsonValue, now: float) -> tuple[Replay, tuple[Any, ...]]:
    """(replay predicate, ``_SETTLE_SET`` params) of one settlement the caller has checked; the replay treats an
    identical committed settlement as idempotent and a different one as a conflict."""
    result_json = compact_json(result)

    def replay(row: Row) -> dict[str, Any] | None:
        if row["settlement_id"] is None:
            return None
        if (row["settlement_id"], row["settlement_status"], row["result_json"]) == (settlement_id, status, result_json):
            return _task_from_row(row, idempotent=True)
        raise TaskConflictError("task already has a different terminal settlement")
    return replay, (status, settlement_id, status, result_json, now, now)


def _cancel_replay(cancel_id: str, status: str = "cancelled") -> Replay:
    """Replay predicate: same cancel_id already committed in ``status``."""
    return lambda row: (
        _task_from_row(row, idempotent=True) if row["status"] == status and row["cancel_id"] == cancel_id else None)


def _generations_match(row: Row, status: str, execution_generation: int, cancel_generation: int) -> bool:
    return (row["status"], int(row["execution_generation"]), int(row["cancel_generation"])) == (
        status, execution_generation, cancel_generation)


def _require_cancel_generation(row: Row, expected_cancel_generation: int) -> None:
    if int(row["cancel_generation"]) != expected_cancel_generation:
        raise StaleTaskError("task cancellation generation changed")


class SQLiteHostedRoomDriverRepository:
    """Driver leases, task transitions and task reads on either hosted-room file."""

    def __init__(self, store: SQLiteStore) -> None:
        self._files = HostedFiles(store, (DRIVER_STEP,))
        self._room_files = HostedFiles(store, (ROOMS_STEP,))

    def _transition(
        self, placement: HostedPlacement, identity: HostedTaskIdentity, *, sql: str, set_params: tuple[Any, ...],
        fence_params: tuple[Any, ...], stale: str, now: float, lease: DriverLease | None = None,
        lease_first: bool = True, replay: Replay | None = None, guard: Callable[[Row], None] | None = None,
        allow_expired_lease: bool = False,
    ) -> dict[str, Any]:
        """Run one fenced task transition: load -> idempotent replay -> lease/fence guard -> UPDATE.

        ``sql`` binds ``(*set_params, room_id, task_id, *fence_params)`` and must hit exactly one row or ``stale``
        is raised. ``lease_first`` checks the lease before the row load (recovery paths) instead of after the
        replay (settlement paths: an identical replay still succeeds after the lease moved on);
        ``allow_expired_lease`` relaxes only that post-replay check and never the lease-first one.
        """
        params = (*set_params, identity.room_id, identity.task_id, *fence_params)

        def _txn(conn: sqlite3.Connection) -> dict[str, Any]:
            if lease is not None and lease_first:
                _require_active_lease(conn, lease, now=now)
            row = _load_task(conn, identity)
            if replay is not None and (replayed := replay(row)) is not None:
                return replayed
            if lease is not None and not lease_first:
                _require_active_lease(conn, lease, now=now, allow_expired=allow_expired_lease)
            if guard is not None:
                guard(row)
            fenced_update(conn, sql, params, StaleTaskError(stale))
            capture_transition_locked(conn, identity.room_id)
            return _task_from_row(_load_task(conn, identity))
        with translated():
            return self._files._hosted_write(placement, _txn, immediate=True)

    def _generation_transition(
        self, placement: HostedPlacement, identity: HostedTaskIdentity, lease: DriverLease, name: str,
        execution_generation: int, cancel_generation: int, *, now: float, set_params: tuple[Any, ...],
        replay: Replay | None = None,
    ) -> dict[str, Any]:
        """Lease-first transition from ``_GENERATION_TRANSITIONS`` fenced on status + both generations."""
        status, set_clause, generation_stale, stale = _GENERATION_TRANSITIONS[name]

        def guard(row: Row) -> None:
            if not _generations_match(row, status, execution_generation, cancel_generation):
                raise StaleTaskError(generation_stale)
        return self._transition(
            placement, identity, lease=lease, now=now, replay=replay, guard=guard,
            sql=_generation_update(set_clause, status), set_params=set_params,
            fence_params=(execution_generation, cancel_generation), stale=stale)

    def _run_fence_transition(
        self, placement: HostedPlacement, attempt: TaskAttempt, *, guard_stale: str,
        lease_generation: Callable[[Any], int] = int, **transition: Any,
    ) -> dict[str, Any]:
        """Transition fenced on this attempt's running generation under its exact lease (row guard + SQL fence).
        ``lease_generation`` casts the stored run_lease_generation: ``int`` raises on NULL, ``int(v or 0)`` reads 0."""
        lease = attempt.lease

        def guard(row: Row) -> None:
            if not _generations_match(row, "running", attempt.execution_generation, attempt.cancel_generation) or (
                row["run_gateway_id"], row["run_process_generation"], lease_generation(row["run_lease_generation"])
            ) != _run_fence(lease):
                raise StaleTaskError(guard_stale)
        return self._transition(
            placement, attempt.identity, lease=lease, guard=guard,
            fence_params=(attempt.execution_generation, attempt.cancel_generation, *_run_fence(lease)), **transition)

    # ── writes ──────────────────────────────────────────────────────────────

    # Takes or extends the room's lease for this process generation, reclaiming one that is expired, released or held
    # under another authority.
    def acquire_lease(self, *, placement: HostedPlacement, room_id: str, gateway_id: str, authority_epoch: int,
                      process_generation: str, ttl_seconds: float, now: float | None = None) -> DriverLease:
        now = clock(now)
        expires_at = now + float(ttl_seconds)

        def _txn(conn: sqlite3.Connection) -> DriverLease:
            _require_room_authority(conn, room_id, gateway_id, authority_epoch)
            row = conn.execute(_SELECT_LEASE, (room_id,)).fetchone()
            if row is None:
                conn.execute("""INSERT INTO hosted_room_driver_leases (
                           room_id, gateway_id, authority_epoch, process_generation, lease_generation,
                           expires_at, acquired_at, updated_at, released_at
                       ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, NULL)""",
                    (room_id, gateway_id, authority_epoch, process_generation, expires_at, now, now))
                return _lease_from_row(conn.execute(_SELECT_LEASE, (room_id,)).fetchone())
            same_authority = row["gateway_id"] == gateway_id and int(row["authority_epoch"]) == authority_epoch
            live = row["released_at"] is None and float(row["expires_at"]) > now
            if same_authority and row["process_generation"] == process_generation and live:
                renewed_expiry = max(float(row["expires_at"]), expires_at)
                conn.execute(
                    "UPDATE hosted_room_driver_leases SET expires_at=?, updated_at=? WHERE room_id=? AND lease_generation=?",
                    (renewed_expiry, now, room_id, int(row["lease_generation"])))
                return _lease_from_row({**dict(row), "expires_at": renewed_expiry})
            if same_authority and live:
                raise LeaseHeldError("room driver lease is held by another generation")
            fenced_update(conn, """UPDATE hosted_room_driver_leases
                   SET gateway_id=?, authority_epoch=?, process_generation=?, lease_generation=lease_generation + 1,
                       expires_at=?, acquired_at=?, updated_at=?, released_at=NULL
                   WHERE room_id=? AND lease_generation=? AND (
                       gateway_id != ? OR authority_epoch != ? OR released_at IS NOT NULL OR expires_at <= ?)""",
                (
                    gateway_id, authority_epoch, process_generation, expires_at, now, now, room_id,
                    int(row["lease_generation"]), gateway_id, authority_epoch, now),
                LeaseHeldError("room driver lease changed during acquisition"))
            return _lease_from_row(conn.execute(_SELECT_LEASE, (room_id,)).fetchone(), reclaimed=True)
        with translated():
            return self._files._hosted_write(placement, _txn, immediate=True)

    # Extends a live lease; its expiry never moves backwards.
    def renew_lease(self, lease: DriverLease, *, placement: HostedPlacement, ttl_seconds: float,
                    now: float | None = None) -> DriverLease:
        now = clock(now)
        requested_expiry = now + float(ttl_seconds)

        def _txn(conn: sqlite3.Connection) -> DriverLease:
            current = _require_active_lease(conn, lease, now=now)
            expires_at = max(float(current["expires_at"]), requested_expiry)
            fenced_update(conn, """UPDATE hosted_room_driver_leases SET expires_at=?, updated_at=?
                   WHERE room_id=? AND gateway_id=? AND process_generation=?
                     AND lease_generation=? AND released_at IS NULL AND expires_at > ?""",
                (expires_at, now, lease.room_id, *_run_fence(lease), now),
                StaleLeaseError("driver lease changed during renewal"))
            return dataclasses.replace(lease, expires_at=expires_at, reclaimed=False)
        with translated():
            return self._files._hosted_write(placement, _txn, immediate=True)

    # Releases the lease unless a task is running; releasing it again replays.
    def release_lease(self, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> LeaseRelease:
        now = clock(now)

        def _txn(conn: sqlite3.Connection) -> LeaseRelease:
            _require_room_authority(conn, lease.room_id, lease.gateway_id, lease.authority_epoch)
            row = conn.execute(_SELECT_LEASE, (lease.room_id,)).fetchone()
            if not _lease_row_matches(row, lease):
                raise StaleLeaseError("driver lease is stale")
            if row["released_at"] is not None:
                return LeaseRelease(lease=_lease_from_row(row), idempotent=True)
            if float(row["expires_at"]) <= now:
                raise StaleLeaseError("driver lease expired before release")
            if conn.execute(
                "SELECT 1 FROM hosted_room_driver_tasks WHERE room_id=? AND status='running' LIMIT 1", (lease.room_id,)
            ).fetchone() is not None:
                raise InvalidTaskTransitionError("cannot release a room lease while tasks are running")
            conn.execute("""UPDATE hosted_room_driver_leases SET expires_at=?, updated_at=?, released_at=?
                   WHERE room_id=? AND lease_generation=?""",
                (now, now, now, lease.room_id, lease.lease_generation))
            return LeaseRelease(
                lease=_lease_from_row({**dict(row), "expires_at": now, "updated_at": now, "released_at": now}),
                idempotent=False)
        with translated():
            return self._files._hosted_write(placement, _txn, immediate=True)

    # Queues a task in a live room; the same task with the same payload replays.
    def admit_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, payload: Mapping[str, JsonValue],
                   now: float | None = None) -> Mapping[str, JsonValue]:
        normalized_payload, payload_json, payload_digest = normalize_task_payload(dict(payload))
        now = clock(now)

        def _txn(conn: sqlite3.Connection) -> dict[str, Any]:
            _load_active_room(conn, identity.room_id)
            existing = _load_task(conn, identity, required=False)
            if existing is not None:
                if existing["payload_digest"] != payload_digest or existing["payload_json"] != payload_json:
                    raise TaskConflictError("task_id is already bound to a different payload")
                return _task_from_row(existing, idempotent=True)
            if conn.execute(
                "SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND thread_id=? AND turn_id=?",
                (identity.room_id, identity.thread_id, identity.turn_id)).fetchone() is not None:
                raise TaskConflictError("thread_id and turn_id are already bound to a task")
            conn.execute("""INSERT INTO hosted_room_driver_tasks (
                       room_id, task_id, thread_id, turn_id, source_event_seq, payload_json, payload_digest,
                       status, execution_generation, cancel_generation, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 0, 0, ?, ?)""",
                (
                    *dataclasses.astuple(identity), normalized_payload["source_event_seq"], payload_json, payload_digest,
                    now, now))
            capture_transition_locked(conn, identity.room_id)
            return _task_from_row(_load_task(conn, identity))
        with translated():
            return self._files._hosted_write(placement, _txn, immediate=True)

    # Starts the next queued task in event order under the lease, once no earlier task awaits recovery.
    def start_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                   expected_cancel_generation: int, now: float | None = None) -> TaskAttempt:
        now = clock(now)

        def _txn(conn: sqlite3.Connection) -> TaskAttempt:
            _require_active_lease(conn, lease, now=now)
            row = _load_task(conn, identity)
            _require_cancel_generation(row, expected_cancel_generation)
            if row["status"] != "queued":
                raise InvalidTaskTransitionError(f"cannot start task in state '{row['status']}'")
            if conn.execute(
                f"""SELECT task_id, status FROM hosted_room_driver_tasks
                   WHERE room_id=? AND status IN ('running', 'indeterminate', 'stopping') {_TASK_ORDER} LIMIT 1""",
                (identity.room_id,)).fetchone() is not None:
                raise InvalidTaskTransitionError("room recovery must resolve the prior task before starting new work")
            next_queued = conn.execute(
                f"SELECT task_id FROM hosted_room_driver_tasks WHERE room_id=? AND status='queued' {_TASK_ORDER} LIMIT 1",
                (identity.room_id,)).fetchone()
            if next_queued is None or next_queued["task_id"] != identity.task_id:
                raise InvalidTaskTransitionError("task is not next in the hosted room event order")
            execution_generation = int(row["execution_generation"]) + 1
            fenced_update(conn, """UPDATE hosted_room_driver_tasks
                   SET status='running', execution_generation=?, run_gateway_id=?, run_process_generation=?,
                       run_lease_generation=?, started_at=?, updated_at=?
                   WHERE room_id=? AND task_id=? AND status='queued' AND cancel_generation=?""",
                (
                    execution_generation, *_run_fence(lease), now, now, identity.room_id, identity.task_id,
                    expected_cancel_generation), StaleTaskError("task changed during start"))
            capture_transition_locked(conn, identity.room_id)
            return TaskAttempt(
                identity=identity, lease=lease, execution_generation=execution_generation,
                cancel_generation=expected_cancel_generation)
        with translated():
            return self._files._hosted_write(placement, _txn, immediate=True)

    # Settles a running attempt; the replay check comes before the lease check, so an identical settlement replays.
    # ``allow_expired_lease`` (the live in-process settle) drops only that check's expiry term.
    def settle_task(self, attempt: TaskAttempt, *, placement: HostedPlacement, settlement_id: str,
                    status: TerminalStatus, result: JsonValue, now: float | None = None,
                    allow_expired_lease: bool = False) -> Mapping[str, JsonValue]:
        now = clock(now)
        replay, set_params = _settlement(settlement_id, status, result, now)
        return self._run_fence_transition(
            placement, attempt, guard_stale="task attempt is stale or cancelled", lease_first=False, now=now,
            replay=replay, sql=_SETTLE_RUNNING_SQL, set_params=set_params, stale="task changed during settlement",
            allow_expired_lease=allow_expired_lease)

    # Settles a stopping task; the replay check comes before the lease check.
    def settle_stopping_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                             expected_execution_generation: int, expected_cancel_generation: int, settlement_id: str,
                             status: TerminalStatus, result: JsonValue, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        replay, set_params = _settlement(settlement_id, status, result, now)
        return self._transition(
            placement, identity, lease=lease, lease_first=False, now=now, replay=replay, sql=_SETTLE_STOPPING_SQL,
            set_params=set_params, fence_params=(expected_execution_generation, expected_cancel_generation),
            stale="task completion lost the stop race")

    # Settles an indeterminate task under the lease.
    def resolve_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                                   expected_execution_generation: int, expected_cancel_generation: int,
                                   settlement_id: str, status: TerminalStatus, result: JsonValue,
                                   now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        replay, set_params = _settlement(settlement_id, status, result, now)
        return self._generation_transition(
            placement, identity, lease, "resolve", expected_execution_generation, expected_cancel_generation, now=now,
            replay=replay, set_params=set_params)

    # Cancels an indeterminate task under the lease.
    def resolve_indeterminate_cancellation(self, identity: HostedTaskIdentity, lease: DriverLease, *,
                                           placement: HostedPlacement, expected_execution_generation: int,
                                           expected_cancel_generation: int, cancel_id: str,
                                           now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        return self._generation_transition(
            placement, identity, lease, "resolve_cancel", expected_execution_generation, expected_cancel_generation,
            now=now, replay=_cancel_replay(cancel_id), set_params=(expected_cancel_generation + 1, cancel_id, now, now))

    # Returns an indeterminate task to the queue under the lease.
    def requeue_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                                   expected_execution_generation: int, expected_cancel_generation: int,
                                   now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        return self._generation_transition(
            placement, identity, lease, "requeue", expected_execution_generation, expected_cancel_generation, now=now,
            set_params=(now,))

    # Defers an indeterminate task with a retryable reason; the same deferral replays.
    def defer_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                                 expected_execution_generation: int, expected_cancel_generation: int, reason: str,
                                 now: float | None = None) -> Mapping[str, JsonValue]:
        return self._defer(placement, identity, lease, "defer", expected_execution_generation,
                           expected_cancel_generation, reason=reason, now=now)

    # Defers a stopping task whose Stop its member never confirmed, with a retryable reason; the same deferral replays.
    def defer_unconfirmed_stop(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                               expected_execution_generation: int, expected_cancel_generation: int, reason: str,
                               now: float | None = None) -> Mapping[str, JsonValue]:
        return self._defer(placement, identity, lease, "defer_stopping", expected_execution_generation,
                           expected_cancel_generation, reason=reason, now=now)

    def _defer(self, placement: HostedPlacement, identity: HostedTaskIdentity, lease: DriverLease, name: str,
               expected_execution_generation: int, expected_cancel_generation: int, *, reason: str,
               now: float | None) -> dict[str, Any]:
        result_json = compact_json({"reason": reason, "retryable": True})
        now = clock(now)

        def replay(row: Row) -> dict[str, Any] | None:
            deferred = _generations_match(row, "deferred", expected_execution_generation, expected_cancel_generation)
            return _task_from_row(row, idempotent=True) if deferred and row["result_json"] == result_json else None
        return self._generation_transition(
            placement, identity, lease, name, expected_execution_generation, expected_cancel_generation, now=now,
            replay=replay, set_params=(result_json, now, now))

    # Returns a deferred task to the queue under the lease.
    def requeue_deferred_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                              expected_execution_generation: int, expected_cancel_generation: int,
                              now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        return self._generation_transition(
            placement, identity, lease, "requeue_deferred", expected_execution_generation, expected_cancel_generation,
            now=now, set_params=(now,))

    # Returns a deferred task to indeterminate at its same generations under the lease; its attempt clock restarts.
    def reopen_deferred_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                             expected_execution_generation: int, expected_cancel_generation: int,
                             now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        return self._generation_transition(
            placement, identity, lease, "reopen_deferred", expected_execution_generation, expected_cancel_generation,
            now=now, set_params=(now, now, now))

    # Requeues a running attempt that was not admitted, under its exact lease; a NULL run lease generation reads 0.
    def requeue_not_admitted_task(self, attempt: TaskAttempt, *, placement: HostedPlacement,
                                  now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)

        def replay(row: Row) -> dict[str, Any] | None:
            requeued = _generations_match(row, "queued", attempt.execution_generation, attempt.cancel_generation) and (
                row["run_gateway_id"], row["run_process_generation"], row["run_lease_generation"]) == (None, None, None)
            return _task_from_row(row, idempotent=True) if requeued else None
        return self._run_fence_transition(
            placement, attempt, guard_stale="not-admitted task attempt lost its fence",
            lease_generation=lambda value: int(value or 0), now=now, replay=replay, sql=_REQUEUE_RUNNING_SQL,
            set_params=(now,), stale="not-admitted task changed during requeue")

    # Cancels a queued or deferred task; running work goes through ``begin_task_cancel``.
    def cancel_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str,
                    expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)

        def guard(row: Row) -> None:
            if row["status"] in TERMINAL_STATUSES:
                raise InvalidTaskTransitionError(f"cannot cancel task in state '{row['status']}'")
            if row["status"] not in {"queued", "deferred"}:
                raise InvalidTaskTransitionError("running work requires acknowledged two-phase cancellation")
            _require_cancel_generation(row, expected_cancel_generation)
        return self._transition(
            placement, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, sql=_CANCEL_QUEUED_SQL,
            set_params=(expected_cancel_generation + 1, cancel_id, now, now), fence_params=(expected_cancel_generation,),
            stale="task changed during cancellation")

    # Moves a running or indeterminate task to ``stopping`` under the next cancel generation.
    def begin_task_cancel(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str,
                          expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)

        def guard(row: Row) -> None:
            if row["status"] in TERMINAL_STATUSES or row["status"] == "queued":
                raise InvalidTaskTransitionError(f"cannot request remote stop in state '{row['status']}'")
            _require_cancel_generation(row, expected_cancel_generation)
        return self._transition(
            placement, identity, now=now, replay=_cancel_replay(cancel_id, "stopping"), guard=guard, sql=_BEGIN_STOP_SQL,
            set_params=(expected_cancel_generation + 1, cancel_id, now), fence_params=(expected_cancel_generation,),
            stale="task changed during stop request")

    # Cancels a stopping task once its stop is acknowledged.
    def complete_task_cancel(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str,
                             expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)

        def guard(row: Row) -> None:
            if (row["status"], row["cancel_id"], int(row["cancel_generation"])) != (
                "stopping", cancel_id, expected_cancel_generation):
                raise StaleTaskError("task stop acknowledgement is stale")
        return self._transition(
            placement, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, sql=_COMPLETE_STOP_SQL,
            set_params=(now, now), fence_params=(cancel_id, expected_cancel_generation),
            stale="task changed during stop acknowledgement")

    # Marks exactly this running attempt indeterminate under an active lease of the room, whichever lease ran it;
    # an attempt no longer running at its generations is left as it is.
    def mark_attempt_indeterminate(self, attempt: TaskAttempt, lease: DriverLease, *, placement: HostedPlacement,
                                   now: float | None = None) -> None:
        now = clock(now)
        identity = attempt.identity

        def _txn(conn: sqlite3.Connection) -> None:
            _require_active_lease(conn, lease, now=now)
            row = _load_task(conn, identity)
            if not _generations_match(row, "running", attempt.execution_generation, attempt.cancel_generation):
                return
            conn.execute(
                _generation_update("status='indeterminate', indeterminate_at=?, updated_at=?", "running"),
                (now, now, identity.room_id, identity.task_id, attempt.execution_generation, attempt.cancel_generation))
            capture_transition_locked(conn, lease.room_id)
        with translated():
            self._files._hosted_write(placement, _txn, immediate=True)

    # Marks running tasks of other leases indeterminate and lists the room's queued and indeterminate tasks.
    def recover_room(self, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> RoomRecovery:
        now = clock(now)
        foreign_running = f"room_id=? AND status='running' AND NOT ({_RUN_FENCE})"
        fence = (lease.room_id, *_run_fence(lease))

        def _txn(conn: sqlite3.Connection) -> RoomRecovery:
            _require_active_lease(conn, lease, now=now)
            stale_rows = conn.execute(
                f"SELECT * FROM hosted_room_driver_tasks WHERE {foreign_running} {_TASK_ORDER}", fence).fetchall()
            if stale_rows:
                conn.execute(
                    f"""UPDATE hosted_room_driver_tasks SET status='indeterminate', indeterminate_at=?, updated_at=?
                        WHERE {foreign_running}""", (now, now, *fence))
            capture_transition_locked(conn, lease.room_id)
            queued, indeterminate = (
                tuple(_task_identity_from_row(row) for row in _tasks_in_order(conn, lease.room_id, status))
                for status in ("queued", "indeterminate"))
            return RoomRecovery(queued=queued, indeterminate=indeterminate)
        with translated():
            return self._files._hosted_write(placement, _txn, immediate=True)

    # Deletes published terminal tasks past retention or beyond the newest ``retain``, at most one batch per call.
    def prune_published_terminal_tasks(self, *, placement: HostedPlacement, room_id: str, now: float | None = None,
                                       retention_seconds: float = TERMINAL_TASK_RETENTION_SECONDS,
                                       retain: int = MAX_RETAINED_TERMINAL_TASKS) -> int:
        now = clock(now)

        def _txn(conn: sqlite3.Connection) -> int:
            publications = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_policy_publications'").fetchone()
            if publications is None:
                return 0
            rows = conn.execute("""SELECT t.task_id, t.terminal_at FROM hosted_room_driver_tasks t
                    WHERE t.room_id=? AND t.status IN ('settled', 'failed', 'cancelled')
                      AND EXISTS (SELECT 1 FROM hosted_room_policy_publications p
                                  WHERE p.room_id=t.room_id AND p.task_id=t.task_id
                                    AND p.kind IN ('turn.settled', 'turn.failed', 'turn.cancelled'))
                    ORDER BY t.terminal_at DESC, t.task_id ASC""", (room_id,)).fetchall()
            cutoff = now - float(retention_seconds)
            candidates = [
                str(row["task_id"]) for index, row in enumerate(rows)
                if index >= retain or (row["terminal_at"] is not None and float(row["terminal_at"]) <= cutoff)
            ][:MAX_TASK_PRUNE_BATCH]
            if not candidates:
                return 0
            deleted = conn.execute(
                f"DELETE FROM hosted_room_driver_tasks WHERE room_id=? AND task_id IN ({','.join('?' * len(candidates))})",
                (room_id, *candidates))
            return max(0, int(deleted.rowcount))
        with translated():
            return self._files._hosted_write(placement, _txn, immediate=True)

    # ── reads ───────────────────────────────────────────────────────────────

    def get_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement) -> Mapping[str, JsonValue]:
        def _read(conn: sqlite3.Connection) -> dict[str, Any]:
            return _task_from_row(_load_task(conn, identity))
        with translated():
            return self._files._hosted_read(placement, _read)

    def list_tasks(self, *, placement: HostedPlacement, room_id: str,
                   status: TaskStatus | None = None) -> tuple[Mapping[str, JsonValue], ...]:
        def _read(conn: sqlite3.Connection) -> tuple[dict[str, Any], ...]:
            return tuple(_task_from_row(row) for row in _tasks_in_order(conn, room_id, status))
        with translated():
            return self._files._hosted_read(placement, _read)

    def room_task_summary(self, *, placement: HostedPlacement, room_id: str,
                          statuses: Collection[str]) -> tuple[Mapping[str, int], tuple[Mapping[str, JsonValue], ...]]:
        statuses = tuple(statuses)

        def _read(conn: sqlite3.Connection) -> tuple[dict[str, int], tuple[dict[str, Any], ...]]:
            # Python's sqlite3 runs SELECTs in autocommit; the explicit BEGIN makes both reads one WAL snapshot,
            # so a task moving between them is neither doubled nor lost.
            began = not conn.in_transaction
            if began:
                conn.execute("BEGIN")
            try:
                counts = {str(status): int(count) for status, count in conn.execute(
                    "SELECT status, COUNT(*) FROM hosted_room_driver_tasks WHERE room_id=? GROUP BY status",
                    (room_id,))}
                rows = conn.execute(
                    f"""SELECT * FROM hosted_room_driver_tasks
                        WHERE room_id=? AND status IN ({','.join('?' * len(statuses))}) {_TASK_ORDER}""",
                    (room_id, *statuses)).fetchall() if statuses else []
            finally:
                if began:
                    conn.rollback()
            return counts, tuple(_task_from_row(row) for row in rows)
        with translated():
            return self._files._hosted_read(placement, _read)

    def room_change_rows(self, *, placement: HostedPlacement,
                         statuses: Collection[str] = SHOWN_TASK_STATUSES) -> Mapping[str, RoomChangeRows]:
        statuses = tuple(statuses)

        # Each room's rows are read through the status index (``room_id, status`` prefix) and sorted here: an
        # ``ORDER BY task_id`` would make SQLite walk the room's retained terminal tasks through the primary key.
        def _read(conn: sqlite3.Connection) -> dict[str, RoomChangeRows]:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('hosted_rooms', 'hosted_room_driver_tasks')")}
            if "hosted_rooms" not in tables:
                return {}
            rooms = conn.execute(
                "SELECT room_id, next_seq, revision FROM hosted_rooms WHERE disbanded_at IS NULL").fetchall()
            if "hosted_room_driver_tasks" not in tables:
                return {room_id: RoomChangeRows(next_seq, revision, ()) for room_id, next_seq, revision in rooms}
            query = f"""SELECT task_id, status, execution_generation, cancel_generation FROM hosted_room_driver_tasks
                        WHERE room_id=? AND status IN ({','.join('?' * len(statuses))})"""
            return {room_id: RoomChangeRows(
                        next_seq, revision, tuple(sorted(tuple(row) for row in conn.execute(query, (room_id, *statuses)))))
                    for room_id, next_seq, revision in rooms}
        with translated():
            return self._room_files._hosted_read(placement, _read)

    # The task bound to this thread and turn, or None.
    def get_task_for_turn(self, identity: HostedTaskIdentity, *,
                          placement: HostedPlacement) -> Mapping[str, JsonValue] | None:
        def _read(conn: sqlite3.Connection) -> dict[str, Any] | None:
            row = conn.execute(
                """SELECT * FROM hosted_room_driver_tasks
                   WHERE room_id=? AND thread_id=? AND turn_id=?""",
                (identity.room_id, identity.thread_id, identity.turn_id),
            ).fetchone()
            return _task_from_row(row) if row is not None else None
        with translated():
            return self._files._hosted_read(placement, _read)


register_repository("hosted_room_driver", SQLiteHostedRoomDriverRepository)
