"""Durable execution state for a same-gateway hosted room driver.

Owns only the driver lease and task state machine: no model calls, no sessions, no dependency on the
hosted-room event log. Callers supply the database path and clock so recovery and fencing are testable
without process-global state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
import contextlib
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Mapping, get_args

from gateway.hosted_rooms_common import (
    DbPath, bounded_int, canonical_json, compact_json, connect, fenced_update, identifier, table_columns, text,
    transaction)

Clock = Callable[[], float]
TaskStatus = Literal["queued", "running", "settled", "failed", "cancelled", "indeterminate", "deferred", "stopping"]
TerminalStatus = Literal["settled", "failed"]

MAX_IDENTIFIER_CHARS = 128
MAX_PROMPT_BYTES = 128 * 1024
MAX_RESULT_JSON_BYTES = 256 * 1024
TERMINAL_TASK_RETENTION_SECONDS = 30 * 24 * 60 * 60
MAX_RETAINED_TERMINAL_TASKS = 2048
MAX_TASK_PRUNE_BATCH = 1000
TASK_STATUSES = frozenset(get_args(TaskStatus))
TERMINAL_STATUSES = frozenset({"settled", "failed", "cancelled"})

_TASK_PAYLOAD_REQUIRED_FIELDS = frozenset({"target_profile", "prompt", "source_event_seq"})
_TASK_PAYLOAD_OPTIONAL_FIELDS = frozenset({"target_member_id", "attachments", "input_context"})
_LEASE_COLUMNS = frozenset({
    "room_id", "gateway_id", "authority_epoch", "process_generation", "lease_generation", "expires_at", "acquired_at",
    "updated_at", "released_at"})
_TASK_COLUMN_ORDER = (
    "room_id", "task_id", "thread_id", "turn_id", "source_event_seq", "payload_json", "payload_digest", "status",
    "execution_generation", "cancel_generation", "run_gateway_id", "run_process_generation", "run_lease_generation",
    "cancel_id", "settlement_id", "settlement_status", "result_json", "created_at", "updated_at", "started_at",
    "terminal_at", "indeterminate_at", "admitted_at", "owner_descriptor")
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
_REQUEUE_SET = (
    "status='queued', run_gateway_id=NULL, run_process_generation=NULL, run_lease_generation=NULL, "
    "admitted_at=NULL, owner_descriptor=NULL")
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
# ``admitted_at IS NULL`` is part of the fence, not only of the guard: direct cancellation is for
# work no transport ever accepted, and a stale routing read must not be able to commit it.
_CANCEL_QUEUED_SQL = _task_update(
    _CANCEL_SET, "status IN ('queued', 'deferred') AND cancel_generation=? AND admitted_at IS NULL")
_BEGIN_STOP_SQL = _task_update(
    "status='stopping', cancel_generation=?, cancel_id=?, updated_at=?",
    "status IN ('running', 'indeterminate', 'deferred') AND cancel_generation=?")
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
    "requeue_deferred": (
        "deferred",
        f"{_REQUEUE_SET}, result_json=NULL, started_at=NULL, terminal_at=NULL, indeterminate_at=NULL, updated_at=?",
        "deferred task generation changed", "deferred task changed during requeue"),
    # A deferred attempt is uncertain work too: an exact terminal receipt (or a proven remote
    # cancellation) resolves it on the same fenced path, never by replaying the turn.
    "resolve_deferred": (
        "deferred", _SETTLE_SET, "deferred task generation changed",
        "deferred task changed during reconciliation"),
    "resolve_deferred_cancel": (
        "deferred", _CANCEL_SET, "deferred cancellation proof is stale",
        "deferred cancellation proof lost its fence"),
    "stop_uncertain": (
        "stopping", "status='indeterminate', indeterminate_at=?, updated_at=?",
        "stopping task generation changed", "stopping task changed during uncertainty")}


# Which ``_GENERATION_TRANSITIONS`` entry resolves each uncertain status; the two states differ
# only in the row they are fenced against, never in what counts as proof.
_UNCERTAIN_RESOLUTIONS = {
    "indeterminate": {"settle": "resolve", "cancel": "resolve_cancel"},
    "deferred": {"settle": "resolve_deferred", "cancel": "resolve_deferred_cancel"}}


class DriverStateError(ValueError): """Base class for invalid or conflicting driver-state operations."""
class DriverValidationError(DriverStateError): """Raised when an identifier, clock, TTL, or payload is invalid."""
class RoomUnavailableError(DriverStateError): """Raised when the hosted room does not exist or was disbanded."""
class LeaseHeldError(DriverStateError): """Raised when another unexpired driver generation owns the room."""
class StaleLeaseError(DriverStateError): """Raised when a lease generation can no longer mutate room state."""
class TaskConflictError(DriverStateError): """Raised when an idempotency key is reused for different task state."""
class StaleTaskError(DriverStateError): """Raised when an obsolete task attempt or cancellation tries to commit."""
class InvalidTaskTransitionError(DriverStateError): """Raised when a requested task transition is not allowed."""


_identifier = partial(identifier, error=DriverValidationError, max_chars=MAX_IDENTIFIER_CHARS)
_bounded_int = partial(bounded_int, error=DriverValidationError)
_canonical_json = partial(
    canonical_json, error=DriverValidationError, label="result", max_bytes=MAX_RESULT_JSON_BYTES, ensure_ascii=True)


def _finite(compute: Callable[[], Any], message: str, *, positive: bool = False) -> float:
    try:
        value = float(compute())
    except (TypeError, ValueError, OverflowError) as exc:
        raise DriverValidationError(message) from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise DriverValidationError(message)
    return value


def _timestamp(clock: Clock) -> float:
    if not callable(clock):
        raise DriverValidationError("clock must be callable")
    return _finite(clock, "clock must return a finite number")


def _lease_window(ttl_seconds: Any, clock: Clock) -> tuple[float, float]:
    """Validate ttl (first) and clock -> ``(now, expires_at)``; the sum itself must stay finite."""
    ttl = _finite(lambda: ttl_seconds, "ttl_seconds must be a finite positive number", positive=True)
    now = _timestamp(clock)
    if not math.isfinite(now + ttl):
        raise DriverValidationError("lease expiry must be finite")
    return now, now + ttl


def _task_payload(value: Any) -> tuple[dict[str, Any], str, str]:
    if not isinstance(value, dict):
        raise DriverValidationError("payload must be an object")
    unknown = set(value) - _TASK_PAYLOAD_REQUIRED_FIELDS - _TASK_PAYLOAD_OPTIONAL_FIELDS
    missing = _TASK_PAYLOAD_REQUIRED_FIELDS - set(value)
    if unknown:
        raise DriverValidationError(f"unknown payload fields: {', '.join(sorted(unknown))}")
    if missing:
        raise DriverValidationError(f"missing payload fields: {', '.join(sorted(missing))}")
    target_profile = _identifier(value["target_profile"], label="target_profile")
    prompt = text(
        value["prompt"],
        error=DriverValidationError,
        label="prompt",
        max_bytes=MAX_PROMPT_BYTES,
        strip=False,
    )
    source_event_seq = _bounded_int(
        value["source_event_seq"], message="source_event_seq must be a positive integer", low=1
    )
    normalized = {
        "target_profile": target_profile,
        "prompt": prompt,
        "source_event_seq": source_event_seq,
    }
    if "input_context" in value:
        from gateway.hosted_room_task_input import validate_task_input

        try:
            normalized["input_context"] = validate_task_input(value["input_context"])
        except ValueError as exc:
            raise DriverValidationError(str(exc)) from exc
    if "target_member_id" in value:
        normalized["target_member_id"] = _identifier(
            value["target_member_id"], label="target_member_id"
        )
    if "attachments" in value:
        from gateway.hosted_room_attachments import validate_task_manifest

        attachments = validate_task_manifest(value["attachments"])
        if not attachments:
            raise DriverValidationError("attachments must not be empty when present")
        normalized["attachments"] = attachments
    encoded = compact_json(normalized)
    return normalized, encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskIdentity:
    """Stable identity for one admitted room turn."""
    room_id: str
    task_id: str
    thread_id: str
    turn_id: str

    def __post_init__(self) -> None:
        for field in ("room_id", "task_id", "thread_id", "turn_id"):
            object.__setattr__(self, field, _identifier(getattr(self, field), label=field))


@dataclass(frozen=True)
class DriverLease:
    """A fenced lease held by one gateway process incarnation."""
    room_id: str
    gateway_id: str
    authority_epoch: int
    process_generation: str
    lease_generation: int
    expires_at: float
    reclaimed: bool = False


@dataclass(frozen=True)
class TaskAttempt:
    """The exact running generation authorized to settle one task."""
    identity: TaskIdentity
    lease: DriverLease
    execution_generation: int
    cancel_generation: int


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
            -- When this attempt's generation passed its final admission fence (see
            -- fence_task_admission): the durable, cross-process boundary between "no prompt was
            -- handed to a transport" and "an execution may be live".
            admitted_at REAL,
            -- The owner incarnation this generation was admitted on (JSON, see
            -- hosted_room_owner_probe). NULL for legacy rows, peers and unsupported platforms,
            -- which therefore gain no death-recovery eligibility. Cleared by start and by every
            -- requeue so a new generation never inherits the old owner's identity.
            owner_descriptor TEXT,
            PRIMARY KEY (room_id, task_id), UNIQUE (room_id, thread_id, turn_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id))""")


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


def _schema_objects_exist(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("""SELECT name FROM sqlite_master WHERE type='table'
           AND name IN ('hosted_room_driver_leases', 'hosted_room_driver_tasks')""").fetchall()
    if {row[0] for row in rows} != {"hosted_room_driver_leases", "hosted_room_driver_tasks"}:
        return False
    index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_hosted_room_driver_tasks_status'").fetchone()
    return index is not None


def _migrate_task_status_constraint(conn: sqlite3.Connection) -> None:
    """Rebuild the unpublished task table without losing durable work.

    Copies only the columns the stored table already has, so a table written before a column was
    added migrates into the current shape with that column at its default.
    """
    conn.execute("DROP INDEX IF EXISTS idx_hosted_room_driver_tasks_status")
    _create_task_table(conn, "hosted_room_driver_tasks_next")
    stored = table_columns(conn, "hosted_room_driver_tasks")
    columns = ", ".join(column for column in _TASK_COLUMN_ORDER if column in stored)
    conn.execute(
        f"INSERT INTO hosted_room_driver_tasks_next ({columns}) SELECT {columns} FROM hosted_room_driver_tasks")
    conn.execute("DROP TABLE hosted_room_driver_tasks")
    conn.execute("ALTER TABLE hosted_room_driver_tasks_next RENAME TO hosted_room_driver_tasks")
    conn.execute(_TASK_INDEX_SQL.format(if_not_exists=""))


def _connect(db_path: DbPath) -> sqlite3.Connection:
    """Open the store; existing tables are validated (after the status-constraint migration when needed).
    The driver schema never shipped, so an incompatible draft fails closed in ``_validate_schema``."""
    existing: list[bool] = []
    def ready(conn: sqlite3.Connection) -> bool:
        existing.append(_schema_objects_exist(conn))
        if not existing[0]:
            return False
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='hosted_room_driver_tasks'").fetchone()
        sql = str(row[0] or "").lower() if row else ""
        # The status CHECK covers the current states AND both admission columns exist.
        return all(
            token in sql for token in ("'stopping'", "'deferred'", "admitted_at", "owner_descriptor"))
    conn = connect(
        db_path, db_label="state.db (hosted_room_driver)", ready=ready,
        initialize=lambda conn: (_migrate_task_status_constraint if existing[0] else _initialize_schema)(conn))
    if existing[0]:
        try:
            _validate_schema(conn)
        except Exception:
            conn.close()
            raise
    return conn


def _transaction(db_path: DbPath):
    return transaction(_connect, db_path, immediate=True)


def _lease_from_row(row: sqlite3.Row | dict[str, Any], *, reclaimed: bool = False) -> DriverLease:
    return DriverLease(
        room_id=row["room_id"], gateway_id=row["gateway_id"], authority_epoch=int(row["authority_epoch"]),
        process_generation=row["process_generation"], lease_generation=int(row["lease_generation"]),
        expires_at=float(row["expires_at"]), reclaimed=reclaimed)


def _task_identity_from_row(row: sqlite3.Row) -> TaskIdentity:
    return TaskIdentity(
        room_id=row["room_id"], task_id=row["task_id"], thread_id=row["thread_id"], turn_id=row["turn_id"])


def _optional(cast: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: cast(value) if value is not None else None


# Task-view casts per row column (columns after payload_digest in _TASK_COLUMN_ORDER, same key order;
# result_json is exposed as "result"). Columns not listed are passed through untouched.
_TASK_VIEW_CASTS: dict[str, Callable[[Any], Any]] = {
    "execution_generation": int, "cancel_generation": int, "run_lease_generation": _optional(int),
    "result_json": _optional(json.loads), "created_at": float, "updated_at": float, "started_at": _optional(float),
    "terminal_at": _optional(float), "indeterminate_at": _optional(float), "admitted_at": _optional(float),
    "owner_descriptor": lambda value: _descriptor_view(value)}


def _descriptor_view(value: Any) -> dict[str, Any] | None:
    """Decode a stored owner descriptor for the task view, or ``None`` if it is not one.

    Tolerant on purpose: a corrupt value must not make every read of the row raise. It costs only
    death-recovery eligibility, and the authoritative refusal happens on the RAW column inside the
    transaction, where a non-NULL unreadable descriptor is rejected rather than treated as legacy.
    """
    if value is None:
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _task_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    try:
        payload, encoded_payload, payload_digest = _task_payload(json.loads(row["payload_json"]))
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


def _load_task(conn: sqlite3.Connection, identity: TaskIdentity, *, required: bool = True) -> sqlite3.Row | None:
    row = conn.execute(_SELECT_TASK, (identity.room_id, identity.task_id)).fetchone()
    if row is None:
        if required:
            raise TaskConflictError("task does not exist")
        return None
    if _task_identity_from_row(row) != identity:
        raise TaskConflictError("task_id is already bound to a different turn")
    return row


def _tasks_in_order(conn: sqlite3.Connection, room_id: str, status: str | None = None) -> list[sqlite3.Row]:
    where, params = ("", (room_id,)) if status is None else (" AND status=?", (room_id, status))
    sql = f"SELECT * FROM hosted_room_driver_tasks WHERE room_id=?{where} {_TASK_ORDER}"
    return conn.execute(sql, params).fetchall()


def _load_active_room(conn: sqlite3.Connection, room_id: str) -> sqlite3.Row:
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


def _require_room_authority(conn: sqlite3.Connection, room_id: str, gateway_id: str, epoch: int) -> sqlite3.Row:
    room = _load_active_room(conn, room_id)
    if room["authority_gateway_id"] != gateway_id or int(room["authority_epoch"]) != epoch:
        raise StaleLeaseError("hosted room authority changed")
    return room


def _run_fence(lease: DriverLease) -> tuple[str, str, int]:
    """The ``_RUN_FENCE`` bind order: (gateway_id, process_generation, lease_generation)."""
    return lease.gateway_id, lease.process_generation, lease.lease_generation


def _lease_row_matches(row: sqlite3.Row | None, lease: DriverLease) -> bool:
    return row is not None and (
        row["gateway_id"], int(row["authority_epoch"]), row["process_generation"], int(row["lease_generation"])
    ) == (lease.gateway_id, lease.authority_epoch, lease.process_generation, lease.lease_generation)


def _require_active_lease(conn: sqlite3.Connection, lease: DriverLease, *, now: float) -> sqlite3.Row:
    _require_room_authority(conn, lease.room_id, lease.gateway_id, lease.authority_epoch)
    row = conn.execute(_SELECT_LEASE, (lease.room_id,)).fetchone()
    if not _lease_row_matches(row, lease) or row["released_at"] is not None or float(row["expires_at"]) <= now:
        raise StaleLeaseError("driver lease is stale or expired")
    return row


def _check_same_room(lease: DriverLease, identity: TaskIdentity) -> None:
    if lease.room_id != identity.room_id:
        raise DriverValidationError("lease and task belong to different rooms")


def _cancel_generation(value: int) -> int:
    # Deliberately accepts bool (a bool is an int); do not swap for non_negative_int.
    if not isinstance(value, int) or value < 0:
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    return value


def _expected_generations(
    lease: DriverLease, identity: TaskIdentity, execution_generation: int, cancel_generation: int) -> None:
    _check_same_room(lease, identity)
    if not isinstance(execution_generation, int) or execution_generation < 1:
        raise DriverValidationError("expected_execution_generation must be a positive integer")
    _cancel_generation(cancel_generation)


def _terminal_settlement_id(settlement_id: Any, status: Any) -> str:
    settlement_id = _identifier(settlement_id, label="settlement_id")
    if status not in {"settled", "failed"}:
        raise DriverValidationError("status must be 'settled' or 'failed'")
    return settlement_id


def _settlement(
    settlement_id: Any, status: Any, result: Any, clock: Clock
) -> tuple[float, Callable[[sqlite3.Row], Any], tuple[Any, ...]]:
    """Validate one terminal settlement -> (now, replay predicate, ``_SETTLE_SET`` params); the replay treats an
    identical committed settlement as idempotent and a different one as a conflict."""
    settlement_id = _terminal_settlement_id(settlement_id, status)
    result_json = _canonical_json(result)
    now = _timestamp(clock)
    def replay(row: sqlite3.Row) -> dict[str, Any] | None:
        if row["settlement_id"] is None:
            return None
        if (row["settlement_id"], row["settlement_status"], row["result_json"]) == (settlement_id, status, result_json):
            return _task_from_row(row, idempotent=True)
        raise TaskConflictError("task already has a different terminal settlement")
    return now, replay, (status, settlement_id, status, result_json, now, now)


def _cancel_replay(cancel_id: str, status: str = "cancelled") -> Callable[[sqlite3.Row], Any]:
    """Replay predicate: same cancel_id already committed in ``status``."""
    return lambda row: (
        _task_from_row(row, idempotent=True) if row["status"] == status and row["cancel_id"] == cancel_id else None)


def _generations_match(row: sqlite3.Row, status: str, execution_generation: int, cancel_generation: int) -> bool:
    return (row["status"], int(row["execution_generation"]), int(row["cancel_generation"])) == (
        status, execution_generation, cancel_generation)


def _require_cancel_generation(row: sqlite3.Row, expected_cancel_generation: int) -> None:
    if int(row["cancel_generation"]) != expected_cancel_generation:
        raise StaleTaskError("task cancellation generation changed")


def _transition(
    db_path: DbPath, identity: TaskIdentity, *, sql: str, set_params: tuple[Any, ...], fence_params: tuple[Any, ...],
    stale: str, now: float, lease: DriverLease | None = None, lease_first: bool = True,
    replay: Callable[[sqlite3.Row], dict[str, Any] | None] | None = None,
    guard: Callable[[sqlite3.Row], None] | None = None) -> dict[str, Any]:
    """Run one fenced task transition: load -> idempotent replay -> lease/fence guard -> UPDATE.

    ``sql`` binds ``(*set_params, room_id, task_id, *fence_params)`` and must hit exactly one row or ``stale``
    is raised. ``lease_first`` checks the lease before the row load (recovery paths) instead of after the
    replay (settlement paths: an identical replay still succeeds after the lease moved on).
    """
    params = (*set_params, identity.room_id, identity.task_id, *fence_params)
    with _transaction(db_path) as conn:
        if lease is not None and lease_first:
            _require_active_lease(conn, lease, now=now)
        row = _load_task(conn, identity)
        if replay is not None and (replayed := replay(row)) is not None:
            return replayed
        if lease is not None and not lease_first:
            _require_active_lease(conn, lease, now=now)
        if guard is not None:
            guard(row)
        fenced_update(conn, sql, params, StaleTaskError(stale))
        return _task_from_row(_load_task(conn, identity))


def _requeue_ownership(
    row: sqlite3.Row, lease: DriverLease, identity: TaskIdentity,
    proof: "OwnerDeathProof | None") -> None:
    """Who may return an admitted attempt to the queue, in three cases.

    Requeueing clears the admission stamp and the run owner, which is what makes a later Stop take
    the direct route, so it may only happen on evidence that the original execution is over.

    1. Never admitted: nothing was handed to a transport, so anyone may requeue it.
    2. No owner descriptor (legacy rows, peers, unsupported platforms): unchanged behaviour --
       only the process that ran it. Those rows gain no new foreign-owner privilege.
    3. Descriptor present: the attempt-bound ``ended`` proof is required **even when the process
       generations are equal**. A restart can reuse a generation value, and equality was never
       evidence; ``_unadmitted_attempts`` is not a cessation witness either.
    """
    if row["admitted_at"] is None:
        return
    if row["owner_descriptor"] is None:
        # Legacy/peer/unsupported rows only: the branch is decided on the RAW column, never on the
        # decoder, which also answers ``None`` for a corrupt or non-object value. Deciding on the
        # decode would hand exactly those rows back the equal-generation bypass.
        if row["run_process_generation"] in (None, lease.process_generation):
            return
        raise InvalidTaskTransitionError(
            "cannot requeue an attempt another process admitted; its execution has not been proven over")
    if _complete_descriptor(row) is None:
        # Unreadable OR incomplete OR wrongly typed: a parseable JSON object is not a descriptor,
        # and a partial one must not be able to authorize a replay just because the process domain
        # inside it still probes as ended.
        raise InvalidTaskTransitionError(
            "cannot requeue an admitted attempt whose stored owner descriptor is incomplete")
    if proof is None:
        raise InvalidTaskTransitionError(
            "cannot requeue an admitted attempt without proof its owner incarnation ended")
    _require_owner_death_proof(row, identity, proof, stop_intent=False)


def _generation_transition(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, name: str, execution_generation: int,
    cancel_generation: int, *, now: float, set_params: tuple[Any, ...],
    replay: Callable[[sqlite3.Row], Any] | None = None,
    owner_death_proof: "OwnerDeathProof | None" = None,
    death_stop_intent: bool = False, death_cancel_id: str | None = None) -> dict[str, Any]:
    """Lease-first transition from ``_GENERATION_TRANSITIONS`` fenced on status + both generations.

    ``owner_death_proof`` is ``None`` for every ordinary caller, whose behaviour is unchanged. When
    supplied it authorizes a death-based route and MUST validate here, in the same transaction as
    the fence; it is never an alternative to the fence.
    """
    status, set_clause, generation_stale, stale = _GENERATION_TRANSITIONS[name]
    death_guard = _owner_death_guard(
        identity, owner_death_proof, stop_intent=death_stop_intent,
        expected_cancel_id=death_cancel_id)
    def guard(row: sqlite3.Row) -> None:
        if not _generations_match(row, status, execution_generation, cancel_generation):
            raise StaleTaskError(generation_stale)
        if name.startswith("requeue"):
            _requeue_ownership(row, lease, identity, owner_death_proof)
        elif death_guard is not None:
            death_guard(row)
    return _transition(
        db_path, identity, lease=lease, now=now, replay=replay, guard=guard, sql=_generation_update(set_clause, status),
        set_params=set_params, fence_params=(execution_generation, cancel_generation), stale=stale)


def _run_fence_transition(
    db_path: DbPath, attempt: TaskAttempt, *, guard_stale: str, lease_generation: Callable[[Any], int] = int,
    **transition: Any) -> dict[str, Any]:
    """Transition fenced on this attempt's running generation under its exact lease (row guard + SQL fence).
    ``lease_generation`` casts the stored run_lease_generation: ``int`` raises on NULL, ``int(v or 0)`` reads 0."""
    lease = attempt.lease
    def guard(row: sqlite3.Row) -> None:
        if not _generations_match(row, "running", attempt.execution_generation, attempt.cancel_generation) or (
            row["run_gateway_id"], row["run_process_generation"], lease_generation(row["run_lease_generation"])
        ) != _run_fence(lease):
            raise StaleTaskError(guard_stale)
    return _transition(
        db_path, attempt.identity, lease=lease, guard=guard,
        fence_params=(attempt.execution_generation, attempt.cancel_generation, *_run_fence(lease)), **transition)


def acquire_lease(
    db_path: DbPath, *, room_id: Any, gateway_id: Any, authority_epoch: Any, process_generation: Any, ttl_seconds: Any,
    clock: Clock) -> DriverLease:
    """Acquire an empty or expired room lease with a monotonic generation."""
    room_id = _identifier(room_id, label="room_id")
    gateway_id = _identifier(gateway_id, label="gateway_id")
    authority_epoch = _bounded_int(authority_epoch, message="authority_epoch must be a positive integer", low=1)
    process_generation = _identifier(process_generation, label="process_generation")
    now, expires_at = _lease_window(ttl_seconds, clock)
    with _transaction(db_path) as conn:
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


def renew_lease(db_path: DbPath, lease: DriverLease, *, ttl_seconds: Any, clock: Clock) -> DriverLease:
    """Renew the exact active lease generation or fail closed."""
    now, requested_expiry = _lease_window(ttl_seconds, clock)
    with _transaction(db_path) as conn:
        current = _require_active_lease(conn, lease, now=now)
        expires_at = max(float(current["expires_at"]), requested_expiry)
        fenced_update(conn, """UPDATE hosted_room_driver_leases SET expires_at=?, updated_at=?
               WHERE room_id=? AND gateway_id=? AND process_generation=?
                 AND lease_generation=? AND released_at IS NULL AND expires_at > ?""",
            (expires_at, now, lease.room_id, *_run_fence(lease), now),
            StaleLeaseError("driver lease changed during renewal"))
        return dataclasses.replace(lease, expires_at=expires_at, reclaimed=False)


def release_lease(db_path: DbPath, lease: DriverLease, *, clock: Clock) -> dict[str, Any]:
    """Release the exact active lease generation idempotently."""
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_room_authority(conn, lease.room_id, lease.gateway_id, lease.authority_epoch)
        row = conn.execute(_SELECT_LEASE, (lease.room_id,)).fetchone()
        if not _lease_row_matches(row, lease):
            raise StaleLeaseError("driver lease is stale")
        if row["released_at"] is not None:
            return {"lease": _lease_from_row(row), "idempotent": True}
        if float(row["expires_at"]) <= now:
            raise StaleLeaseError("driver lease expired before release")
        if conn.execute(
            "SELECT 1 FROM hosted_room_driver_tasks WHERE room_id=? AND status='running' LIMIT 1", (lease.room_id,)
        ).fetchone() is not None:
            raise InvalidTaskTransitionError("cannot release a room lease while tasks are running")
        conn.execute("""UPDATE hosted_room_driver_leases SET expires_at=?, updated_at=?, released_at=?
               WHERE room_id=? AND lease_generation=?""",
            (now, now, now, lease.room_id, lease.lease_generation))
        return {
            "lease": _lease_from_row({**dict(row), "expires_at": now, "updated_at": now, "released_at": now}),
            "idempotent": False}


def admit_task(db_path: DbPath, identity: TaskIdentity, *, payload: Any, clock: Clock) -> dict[str, Any]:
    """Persist a queued task, or return the identical admission."""
    normalized_payload, payload_json, payload_digest = _task_payload(payload)
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
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
        return _task_from_row(_load_task(conn, identity))


def start_task(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_cancel_generation: int, clock: Clock
) -> TaskAttempt:
    """Move one queued task to running under the current driver lease."""
    _check_same_room(lease, identity)
    _cancel_generation(expected_cancel_generation)
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
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
                   run_lease_generation=?, started_at=?, updated_at=?, admitted_at=NULL, owner_descriptor=NULL
               WHERE room_id=? AND task_id=? AND status='queued' AND cancel_generation=?""",
            (
                execution_generation, *_run_fence(lease), now, now, identity.room_id, identity.task_id,
                expected_cancel_generation), StaleTaskError("task changed during start"))
        return TaskAttempt(
            identity=identity, lease=lease, execution_generation=execution_generation,
            cancel_generation=expected_cancel_generation)


def _owner_descriptor_json(
    attempt: TaskAttempt, native: Mapping[str, Any] | None) -> str | None:
    """The complete owner descriptor for one attempt, or ``None`` when capture is unavailable.

    The admission coordinates come from THIS ``TaskAttempt`` and its lease -- never reconstructed
    from mutable session state, and never taken from the adapter, which cannot see them. ``native``
    supplies only what the running process can observe about itself (its incarnation, home, profile
    and the two session identifiers). A missing native half means no descriptor at all, not a
    partial one.
    """
    from gateway.hosted_room_owner_probe import validate_native_descriptor

    if (validated := validate_native_descriptor(native)) is None:
        return None
    gateway_id, process_generation, lease_generation = _run_fence(attempt.lease)
    return _canonical_json({
        **validated,
        "room_id": attempt.identity.room_id, "task_id": attempt.identity.task_id,
        "thread_id": attempt.identity.thread_id, "turn_id": attempt.identity.turn_id,
        "execution_generation": int(attempt.execution_generation),
        "cancel_generation_at_admission": int(attempt.cancel_generation),
        "run_gateway_id": gateway_id, "run_process_generation": process_generation,
        "run_lease_generation": int(lease_generation)})


def fence_task_admission(
    db_path: DbPath, attempt: TaskAttempt, *, clock: Clock,
    owner_descriptor: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Stamp ``admitted_at`` for one running attempt at its final admission boundary, or raise.

    The caller runs this immediately before handing the prompt to a transport, so the row records
    the boundary every other process can read:

    * ``admitted_at IS NULL`` for the current generation means no prompt was handed to any
      transport. A cancellation that commits first makes this raise ``StaleTaskError`` (the fence
      names ``status='running'`` plus both generations and the run fence), so the attempt aborts
      without submitting -- a durable barrier, not a re-read.
    * ``admitted_at`` set means an execution may be live. An idle or absent session is then no
      longer proof that it stopped; only the process owning the run (or, once that owner's lease
      expires, its successor) may commit cancellation.

    The stamp cannot be atomic with the transport call itself, so a Stop landing between them is
    reported as provisional rather than acknowledged and is resolved by interrupting the execution
    that did start.
    """
    now = _timestamp(clock)
    return _run_fence_transition(
        db_path, attempt, guard_stale="task attempt is stale or cancelled", now=now,
        sql=_task_update(
            "admitted_at=?, owner_descriptor=?, updated_at=?",
            f"status='running' AND {_GENERATION_FENCE} AND {_RUN_FENCE}"),
        set_params=(now, _owner_descriptor_json(attempt, owner_descriptor), now),
        stale="task changed during admission")


# --- owner-death recovery ----------------------------------------------------
# Process death is an ADDITIONAL cessation authority. It never replaces the exact terminal receipt
# or the acknowledged native cancellation, both of which stay valid while the owner's backend is
# alive; and it never chooses the outcome, which stays whatever the recorded stop identity says.
# Every death-authorized transition takes ``owner_death_proof``: ``None`` on every ordinary path
# (exactly today's behaviour), and when supplied the proof MUST validate inside the transaction or
# the transition raises. There is no boolean and no public proof surface -- ``groups.retry``
# forwards only room and task ids.
_PROOF_STALE = "owner-death proof does not bind this attempt"


@dataclass(frozen=True)
class OwnerDeathProof:
    """A successor's own evidence that one exact admitted attempt's owner incarnation ended."""

    identity: TaskIdentity
    execution_generation: int
    cancel_generation_at_admission: int
    run: tuple[str, str, int]
    admitted_at: float
    descriptor: dict[str, Any]
    verdict: str


def owner_death_proof(task: Mapping[str, Any], verdict: str) -> OwnerDeathProof | None:
    """Build the proof for one loaded task view, or ``None`` when it cannot be made.

    The successor constructs this from the row it just read plus its own probe verdict, so a
    caller cannot hand-roll a partial one: anything short of a complete descriptor and an ``ended``
    verdict yields ``None``, and a ``None`` proof simply means the death route is not taken.
    """
    # Exactly the rule the transactional guard applies -- one definition, not a second copy -- so
    # the builder cannot mint a proof the guard would then have to reject. There is no default for
    # a missing coordinate: an absent `cancel_generation_at_admission` once became -1, which made
    # incomplete evidence look complete.
    descriptor = _validated_descriptor(task.get("owner_descriptor"))
    if verdict != "ended" or descriptor is None or task.get("admitted_at") is None:
        return None
    run = (task.get("run_gateway_id"), task.get("run_process_generation"), task.get("run_lease_generation"))
    if not isinstance(run[0], str) or not isinstance(run[1], str) or not _exact_int(run[2]):
        return None
    if not _exact_int(task.get("execution_generation")) or not isinstance(
            task.get("admitted_at"), float):
        return None
    return OwnerDeathProof(
        identity=task["identity"], execution_generation=task["execution_generation"],
        cancel_generation_at_admission=descriptor["cancel_generation_at_admission"],
        run=(run[0], run[1], run[2]), admitted_at=task["admitted_at"],
        descriptor=dict(descriptor), verdict=verdict)


# The admission half of a descriptor, with the exact type each coordinate must have. A parseable
# JSON object is NOT a valid descriptor: incomplete evidence must never be made to look complete by
# coercion (`int("3")`, `str(None)`, `True == 1`) or by a default.
_DESCRIPTOR_IDENTITY_FIELDS = ("room_id", "task_id", "thread_id", "turn_id")
_DESCRIPTOR_TEXT_FIELDS = (*_DESCRIPTOR_IDENTITY_FIELDS, "run_gateway_id", "run_process_generation")
_DESCRIPTOR_INT_FIELDS = (
    "execution_generation", "cancel_generation_at_admission", "run_lease_generation")


def _exact_int(value: Any) -> bool:
    """A real ``int``. ``bool`` is excluded: ``True == 1`` would otherwise impersonate a coordinate."""
    return isinstance(value, int) and not isinstance(value, bool)


def _stored_descriptor(row: sqlite3.Row) -> dict[str, Any] | None:
    """Decode the raw column to a JSON object, with no completeness claim. Internal use only."""
    with contextlib.suppress(TypeError, ValueError, json.JSONDecodeError):
        stored = json.loads(row["owner_descriptor"]) if row["owner_descriptor"] is not None else None
        return stored if isinstance(stored, dict) else None
    return None


def _validated_descriptor(candidate: Any) -> dict[str, Any] | None:
    """The ONE completeness/type rule for a descriptor, wherever it came from.

    Both halves are required: the native process/domain/target half (validated by the probe module
    that wrote it) and the admission half below. A descriptor missing ``home``, ``profile``, either
    session identifier, an identity field or any admission coordinate is not evidence of anything,
    even though the process domain in it could still probe as ``ended``.

    Ranges, not just types: an admitted attempt has a positive execution generation and a positive
    run-lease generation, and its admission-cancel coordinate is a real count. Negative coordinates
    are as invalid as missing ones. The row-relative upper bound lives in the guard, which is the
    only place that can see the row.
    """
    from gateway.hosted_room_owner_probe import validate_native_descriptor

    if not isinstance(candidate, Mapping) or validate_native_descriptor(candidate) is None:
        return None
    for field in _DESCRIPTOR_TEXT_FIELDS:
        value = candidate.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
    if any(not _exact_int(candidate.get(field)) for field in _DESCRIPTOR_INT_FIELDS):
        return None
    if candidate["execution_generation"] < 1 or candidate["run_lease_generation"] < 1:
        return None
    if candidate["cancel_generation_at_admission"] < 0:
        return None
    return dict(candidate)


def _complete_descriptor(row: sqlite3.Row) -> dict[str, Any] | None:
    """The stored descriptor, or ``None`` unless it is complete and strictly typed."""
    return _validated_descriptor(_stored_descriptor(row))


def _same_descriptor(stored: Mapping[str, Any], supplied: Mapping[str, Any]) -> bool:
    """Type-aware equality. ``{"pid": 1} == {"pid": True}`` is True in Python, so ``==`` alone
    would let a bool or float impersonate an integer coordinate inside an otherwise equal map."""
    if set(stored) != set(supplied):
        return False
    return all(
        type(stored[key]) is type(supplied[key]) and stored[key] == supplied[key]
        for key in stored)


def _descriptor_binds_row(stored: Mapping[str, Any], row: sqlite3.Row, identity: TaskIdentity) -> bool:
    """Whether the descriptor's OWN duplicated coordinates agree with the row it is stored on.

    The descriptor repeats the identity, the attempt and the original run owner. Those copies are
    redundant by design, so a descriptor whose nested values contradict its row is corrupt, not
    merely stale -- and comparing the proof to the stored object alone would never notice.
    """
    if tuple(stored[field] for field in _DESCRIPTOR_IDENTITY_FIELDS) != (
            identity.room_id, identity.task_id, identity.thread_id, identity.turn_id):
        return False
    if stored["execution_generation"] != int(row["execution_generation"]):
        return False
    return (
        stored["run_gateway_id"], stored["run_process_generation"], stored["run_lease_generation"]
    ) == (row["run_gateway_id"], row["run_process_generation"], row["run_lease_generation"])


def _require_owner_death_proof(
    row: sqlite3.Row, identity: TaskIdentity, proof: OwnerDeathProof, *, stop_intent: bool,
    expected_cancel_id: str | None = None) -> None:
    """Validate one death proof against the row just loaded, inside its own transaction.

    This is the authoritative check: it re-derives everything rather than trusting the builder, so
    a hand-constructed or substituted ``OwnerDeathProof`` is refused here too.

    Bound together: the COMPLETE, strictly typed stored descriptor; the descriptor's own duplicated
    identity/attempt/run coordinates against the row; the full task identity; the exact attempt; the
    ORIGINAL run owner (a different fence from the successor's active lease, which
    ``_require_active_lease`` keeps checking separately); the exact admission stamp; the proof's
    admission-cancel coordinate against the descriptor; and the descriptor field-for-field.

    ``stop_intent`` additionally requires that THIS attempt was really asked to stop: a recorded
    cancel id plus a current cancel generation advanced beyond the one recorded at admission -- a
    nonzero generation on its own can be inherited. ``expected_cancel_id`` binds a caller-supplied
    stop identity to the recorded one, so a death route cannot rewrite what was stopped.
    """
    stored = _complete_descriptor(row)
    if stored is None or proof.verdict != "ended":
        raise StaleTaskError(_PROOF_STALE)
    if not _descriptor_binds_row(stored, row, identity):
        raise StaleTaskError(_PROOF_STALE)
    if proof.identity != identity or _task_identity_from_row(row) != identity:
        raise StaleTaskError(_PROOF_STALE)
    if not _exact_int(proof.execution_generation) or (
            int(row["execution_generation"]) != proof.execution_generation):
        raise StaleTaskError(_PROOF_STALE)
    if row["admitted_at"] is None or not isinstance(proof.admitted_at, float) or (
            not math.isfinite(proof.admitted_at) or float(row["admitted_at"]) != proof.admitted_at):
        raise StaleTaskError(_PROOF_STALE)
    if not isinstance(proof.run, tuple) or len(proof.run) != 3 or not _exact_int(proof.run[2]):
        raise StaleTaskError(_PROOF_STALE)
    if (row["run_gateway_id"], row["run_process_generation"], row["run_lease_generation"]) != (
            proof.run[0], proof.run[1], proof.run[2]):
        raise StaleTaskError(_PROOF_STALE)
    # The proof carries this coordinate explicitly, so it is checked explicitly: shallow equality
    # with the stored object would pass a proof that simply copied a wrong value along with it.
    if not _exact_int(proof.cancel_generation_at_admission) or (
            proof.cancel_generation_at_admission != stored["cancel_generation_at_admission"]):
        raise StaleTaskError(_PROOF_STALE)
    # The admission-cancel coordinate cannot claim a stop the row never reached.
    if stored["cancel_generation_at_admission"] > int(row["cancel_generation"]):
        raise StaleTaskError(_PROOF_STALE)
    # The SUPPLIED descriptor is validated too, not merely compared: a hand-built proof could
    # otherwise carry bool/float impersonations that dict equality accepts.
    if _validated_descriptor(proof.descriptor) is None or not _same_descriptor(
            stored, proof.descriptor):
        raise StaleTaskError(_PROOF_STALE)
    if not stop_intent:
        return
    recorded_cancel_id = str(row["cancel_id"] or "")
    if not recorded_cancel_id:
        raise StaleTaskError("owner-death cancellation needs this attempt's own stop intent")
    if int(row["cancel_generation"]) <= stored["cancel_generation_at_admission"]:
        raise StaleTaskError("owner-death cancellation needs this attempt's own stop intent")
    if expected_cancel_id is not None and expected_cancel_id != recorded_cancel_id:
        raise StaleTaskError(
            "owner-death cancellation cannot replace this attempt's recorded stop identity")


def _require_death_route(proof: OwnerDeathProof | None, death_authorized: bool) -> None:
    """A caller whose only evidence is owner death must carry the proof, or nothing happens.

    This is what stops a death-authorized terminal route from silently degrading into the ordinary
    one: a missing or unbuildable proof raises here, before any transaction opens, rather than
    committing an unguarded cancellation. The inverse is refused too -- a proof handed to a caller
    that did not declare the death route would never be validated.
    """
    if death_authorized and proof is None:
        raise DriverValidationError(
            "owner-death authorization requires a complete owner-death proof")
    if proof is not None and not death_authorized:
        raise DriverValidationError(
            "an owner-death proof is only accepted on an explicitly death-authorized route")


def _owner_death_guard(
    identity: TaskIdentity, proof: OwnerDeathProof | None, *, stop_intent: bool,
    expected_cancel_id: str | None = None) -> Callable[[sqlite3.Row], None] | None:
    """The in-transaction guard for a death-authorized transition, or ``None`` for ordinary ones."""
    if proof is None:
        return None
    return lambda row: _require_owner_death_proof(
        row, identity, proof, stop_intent=stop_intent, expected_cancel_id=expected_cancel_id)


def mark_stop_uncertain(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, clock: Clock) -> dict[str, Any]:
    """Record that a requested stop could not be proven: uncertain, never cancelled.

    Cancellation is a claim that the execution is over. When the exact attempt was admitted by a
    process this one does not run, an absent or idle session proves nothing about it -- the former
    owner may still be between its admission fence and its dispatch. This keeps the durable answer
    honest by moving the attempt into the existing uncertain state, which never auto-resubmits and
    requires explicit user recovery.
    """
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    now = _timestamp(clock)
    return _generation_transition(
        db_path, identity, lease, "stop_uncertain", expected_execution_generation, expected_cancel_generation,
        now=now, set_params=(now, now))


def settle_task(
    db_path: DbPath, attempt: TaskAttempt, *, settlement_id: Any, status: TerminalStatus, result: Any, clock: Clock
) -> dict[str, Any]:
    """Commit one terminal result if every lease and task fence still matches."""
    now, replay, set_params = _settlement(settlement_id, status, result, clock)
    return _run_fence_transition(
        db_path, attempt, guard_stale="task attempt is stale or cancelled", lease_first=False, now=now, replay=replay,
        sql=_SETTLE_RUNNING_SQL, set_params=set_params, stale="task changed during settlement")


def settle_stopping_task(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, settlement_id: Any, status: TerminalStatus, result: Any, clock: Clock,
    death_authorized: bool = False, owner_death_proof: "OwnerDeathProof | None" = None
) -> dict[str, Any]:
    """Commit a completion that won the race with an unacknowledged Stop.

    ``death_authorized`` is for the one caller whose evidence is owner death (a deadline stop whose
    owner is proven gone): the outcome and its reason still come from the recorded stop identity,
    never from the cessation evidence."""
    _terminal_settlement_id(settlement_id, status)  # settlement errors take precedence over generation errors
    if expected_execution_generation < 1 or expected_cancel_generation < 1:
        raise DriverValidationError("stopping settlement generations are invalid")
    _require_death_route(owner_death_proof, death_authorized)
    now, replay, set_params = _settlement(settlement_id, status, result, clock)
    return _transition(
        db_path, identity, lease=lease, lease_first=False, now=now, replay=replay, sql=_SETTLE_STOPPING_SQL,
        set_params=set_params, fence_params=(expected_execution_generation, expected_cancel_generation),
        guard=_owner_death_guard(identity, owner_death_proof, stop_intent=True),
        stale="task completion lost the stop race")


def _uncertain_transition(from_status: Any, kind: str) -> str:
    """Name the resolution transition for one uncertain status (``indeterminate``/``deferred``)."""
    if from_status not in _UNCERTAIN_RESOLUTIONS:
        raise DriverValidationError("from_status must be 'indeterminate' or 'deferred'")
    return _UNCERTAIN_RESOLUTIONS[from_status][kind]


def resolve_indeterminate_task(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, settlement_id: Any, status: TerminalStatus, result: Any, clock: Clock,
    from_status: str = "indeterminate", death_authorized: bool = False,
    owner_death_proof: "OwnerDeathProof | None" = None) -> dict[str, Any]:
    """Commit a verified historical receipt under the current room lease.

    ``from_status`` names which uncertain state the receipt resolves: a deferred attempt carries
    the same execution generation and the same admission evidence as an indeterminate one, so an
    exact receipt closes it on the identical fenced path instead of replaying its turn.
    """
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    _require_death_route(owner_death_proof, death_authorized)
    now, replay, set_params = _settlement(settlement_id, status, result, clock)
    return _generation_transition(
        db_path, identity, lease, _uncertain_transition(from_status, "settle"), expected_execution_generation,
        expected_cancel_generation, now=now, replay=replay, set_params=set_params,
        owner_death_proof=owner_death_proof, death_stop_intent=True)


def resolve_indeterminate_cancellation(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, cancel_id: Any, clock: Clock,
    from_status: str = "indeterminate", death_authorized: bool = False,
    owner_death_proof: "OwnerDeathProof | None" = None) -> dict[str, Any]:
    """Commit a verified terminal cancellation for an uncertain attempt.

    ``death_authorized`` states that the CALLER's only cessation evidence is owner death; the proof
    is then mandatory and validated in-transaction. Without that flag this is the ordinary route a
    transport-reported cancellation uses, which stays valid while the owner's backend is alive."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    cancel_id = _identifier(cancel_id, label="cancel_id")
    _require_death_route(owner_death_proof, death_authorized)
    now = _timestamp(clock)
    return _generation_transition(
        db_path, identity, lease, _uncertain_transition(from_status, "cancel"), expected_execution_generation,
        expected_cancel_generation, now=now, replay=_cancel_replay(cancel_id),
        set_params=(expected_cancel_generation + 1, cancel_id, now, now),
        owner_death_proof=owner_death_proof, death_stop_intent=True,
        # The written id is the caller's, so on the death route it must equal the recorded one --
        # checked in this same transaction. `complete_task_cancel` already fences that equality;
        # this route wrote whatever it was handed.
        death_cancel_id=cancel_id)


def requeue_indeterminate_task(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, clock: Clock,
    owner_death_proof: "OwnerDeathProof | None" = None) -> dict[str, Any]:
    """Explicitly retry uncertain work after an operator accepts at-least-once risk.

    No ``death_authorized`` flag is needed here: ``_requeue_ownership`` refuses outright when the
    row carries a descriptor and no proof accompanies it, so this route cannot degrade into an
    unguarded requeue."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    now = _timestamp(clock)
    return _generation_transition(
        db_path, identity, lease, "requeue", expected_execution_generation, expected_cancel_generation, now=now,
        set_params=(now,), owner_death_proof=owner_death_proof)


def defer_indeterminate_task(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, reason: Any, clock: Clock) -> dict[str, Any]:
    """Fence one uncertain attempt and release later room work."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    reason = _identifier(reason, label="defer_reason")
    result_json = _canonical_json({"reason": reason, "retryable": True})
    now = _timestamp(clock)
    def replay(row: sqlite3.Row) -> dict[str, Any] | None:
        deferred = _generations_match(row, "deferred", expected_execution_generation, expected_cancel_generation)
        return _task_from_row(row, idempotent=True) if deferred and row["result_json"] == result_json else None
    return _generation_transition(
        db_path, identity, lease, "defer", expected_execution_generation, expected_cancel_generation, now=now,
        replay=replay, set_params=(result_json, now, now))


def requeue_deferred_task(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, clock: Clock,
    owner_death_proof: "OwnerDeathProof | None" = None) -> dict[str, Any]:
    """Explicitly retry a fenced deferred turn under a new generation.

    No ``death_authorized`` flag is needed here: ``_requeue_ownership`` refuses outright when the
    row carries a descriptor and no proof accompanies it, so this route cannot degrade into an
    unguarded requeue."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    now = _timestamp(clock)
    return _generation_transition(
        db_path, identity, lease, "requeue_deferred", expected_execution_generation, expected_cancel_generation,
        now=now, set_params=(now,), owner_death_proof=owner_death_proof)


def requeue_not_admitted_task(db_path: DbPath, attempt: TaskAttempt, *, clock: Clock) -> dict[str, Any]:
    """Return a running task to its durable queue after proven non-admission."""
    now = _timestamp(clock)
    _check_same_room(attempt.lease, attempt.identity)
    def replay(row: sqlite3.Row) -> dict[str, Any] | None:
        requeued = _generations_match(row, "queued", attempt.execution_generation, attempt.cancel_generation) and (
            row["run_gateway_id"], row["run_process_generation"], row["run_lease_generation"]) == (None, None, None)
        return _task_from_row(row, idempotent=True) if requeued else None
    return _run_fence_transition(
        db_path, attempt, guard_stale="not-admitted task attempt lost its fence",
        lease_generation=lambda value: int(value or 0), now=now, replay=replay, sql=_REQUEUE_RUNNING_SQL,
        set_params=(now,), stale="not-admitted task changed during requeue")


def cancel_task(
    db_path: DbPath, identity: TaskIdentity, *, cancel_id: Any, expected_cancel_generation: int, clock: Clock
) -> dict[str, Any]:
    """Cancel a queued task before any external work was admitted."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    _cancel_generation(expected_cancel_generation)
    now = _timestamp(clock)
    def guard(row: sqlite3.Row) -> None:
        if row["status"] in TERMINAL_STATUSES:
            raise InvalidTaskTransitionError(f"cannot cancel task in state '{row['status']}'")
        if row["status"] not in {"queued", "deferred"} or row["admitted_at"] is not None:
            # A deferred attempt that passed its admission fence may still be executing wherever
            # it was admitted; only an attempt no transport accepted can be cancelled outright.
            raise InvalidTaskTransitionError("running work requires acknowledged two-phase cancellation")
        _require_cancel_generation(row, expected_cancel_generation)
    return _transition(
        db_path, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, sql=_CANCEL_QUEUED_SQL,
        set_params=(expected_cancel_generation + 1, cancel_id, now, now), fence_params=(expected_cancel_generation,),
        stale="task changed during cancellation")


def begin_task_cancel(
    db_path: DbPath, identity: TaskIdentity, *, cancel_id: Any, expected_cancel_generation: int, clock: Clock
) -> dict[str, Any]:
    """Persist a stop intent without claiming the remote run has stopped.

    Accepts a deferred attempt too: a deferral only fences publication, so an attempt that already
    passed its admission fence still needs an acknowledged stop rather than a direct cancellation.
    """
    cancel_id = _identifier(cancel_id, label="cancel_id")
    _cancel_generation(expected_cancel_generation)
    now = _timestamp(clock)
    def guard(row: sqlite3.Row) -> None:
        if row["status"] in TERMINAL_STATUSES or row["status"] == "queued":
            raise InvalidTaskTransitionError(f"cannot request remote stop in state '{row['status']}'")
        _require_cancel_generation(row, expected_cancel_generation)
    return _transition(
        db_path, identity, now=now, replay=_cancel_replay(cancel_id, "stopping"), guard=guard, sql=_BEGIN_STOP_SQL,
        set_params=(expected_cancel_generation + 1, cancel_id, now), fence_params=(expected_cancel_generation,),
        stale="task changed during stop request")


def complete_task_cancel(
    db_path: DbPath, identity: TaskIdentity, *, cancel_id: Any, expected_cancel_generation: int, clock: Clock,
    death_authorized: bool = False, owner_death_proof: "OwnerDeathProof | None" = None
) -> dict[str, Any]:
    """Commit cancellation after the transport acknowledges exact Stop, or the owner is proven gone.

    The recorded stop identity is untouched by either route: ``cancel_id`` is the row's own, and the
    current cancel generation is still the fence. ``death_authorized`` only adds a second source of
    cessation evidence, validated in this same transaction; it never rewrites what was stopped."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    _require_death_route(owner_death_proof, death_authorized)
    death_guard = _owner_death_guard(identity, owner_death_proof, stop_intent=True)
    now = _timestamp(clock)
    def guard(row: sqlite3.Row) -> None:
        if (row["status"], row["cancel_id"], int(row["cancel_generation"])) != (
            "stopping", cancel_id, expected_cancel_generation):
            raise StaleTaskError("task stop acknowledgement is stale")
        if death_guard is not None:
            death_guard(row)
    return _transition(
        db_path, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, sql=_COMPLETE_STOP_SQL,
        set_params=(now, now), fence_params=(cancel_id, expected_cancel_generation),
        stale="task changed during stop acknowledgement")


def recover_room(db_path: DbPath, lease: DriverLease, *, clock: Clock) -> dict[str, list[TaskIdentity]]:
    """Fence abandoned running attempts without requeueing uncertain work."""
    now = _timestamp(clock)
    foreign_running = f"room_id=? AND status='running' AND NOT ({_RUN_FENCE})"
    fence = (lease.room_id, *_run_fence(lease))
    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        stale_rows = conn.execute(
            f"SELECT * FROM hosted_room_driver_tasks WHERE {foreign_running} {_TASK_ORDER}", fence).fetchall()
        if stale_rows:
            conn.execute(
                f"""UPDATE hosted_room_driver_tasks SET status='indeterminate', indeterminate_at=?, updated_at=?
                    WHERE {foreign_running}""", (now, now, *fence))
        return {
            status: [_task_identity_from_row(row) for row in _tasks_in_order(conn, lease.room_id, status)]
            for status in ("queued", "indeterminate")}


def get_task(db_path: DbPath, identity: TaskIdentity) -> dict[str, Any]:
    """Read one task without mutating its state."""
    with closing(_connect(db_path)) as conn:
        return _task_from_row(_load_task(conn, identity))


def list_tasks(db_path: DbPath, *, room_id: Any, status: TaskStatus | None = None) -> list[dict[str, Any]]:
    """Return room tasks in deterministic admission order."""
    room_id = _identifier(room_id, label="room_id")
    if status is not None and status not in TASK_STATUSES:
        raise DriverValidationError("invalid task status")
    with closing(_connect(db_path)) as conn:
        return [_task_from_row(row) for row in _tasks_in_order(conn, room_id, status)]


def prune_published_terminal_tasks(
    db_path: DbPath, *, room_id: Any, clock: Clock, retention_seconds: float = TERMINAL_TASK_RETENTION_SECONDS,
    retain: int = MAX_RETAINED_TERMINAL_TASKS) -> int:
    """Bound execution rows after outcomes are durable in the room log."""
    room_id = _identifier(room_id, label="room_id")
    now = _timestamp(clock)
    if retention_seconds <= 0:
        raise DriverValidationError("retention_seconds must be positive")
    _bounded_int(retain, message="retain must be a non-negative integer")
    with _transaction(db_path) as conn:
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Iterator  # noqa: F401,E402
from pathlib import Path  # noqa: F401,E402
from contextlib import contextmanager  # noqa: F401,E402
import re  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----


def defer_not_admitted_task(
    db_path: DbPath,
    attempt: TaskAttempt,
    *,
    reason: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Publish a proven pre-admission outage without blocking later members."""

    reason = _identifier(reason, label="defer_reason")
    result_json = _canonical_json({"reason": reason, "retryable": True})
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_active_lease(conn, attempt.lease, now=now)
        row = _load_task(conn, attempt.identity)
        if (
            row["status"] == "deferred"
            and int(row["execution_generation"]) == attempt.execution_generation
            and int(row["cancel_generation"]) == attempt.cancel_generation
            and row["result_json"] == result_json
        ):
            return _task_from_row(row, idempotent=True)
        if (
            row["status"] != "running"
            or int(row["execution_generation"]) != attempt.execution_generation
            or int(row["cancel_generation"]) != attempt.cancel_generation
        ):
            raise StaleTaskError("running task generation changed during deferral")
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
                  SET status='deferred', result_json=?, terminal_at=?, updated_at=?
                WHERE room_id=? AND task_id=? AND status='running'
                  AND execution_generation=? AND cancel_generation=?""",
            (
                result_json,
                now,
                now,
                attempt.identity.room_id,
                attempt.identity.task_id,
                attempt.execution_generation,
                attempt.cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("running task changed during deferral")
        return _task_from_row(_load_task(conn, attempt.identity))


def get_task_for_turn(
    db_path: DbPath,
    identity: TaskIdentity,
) -> dict[str, Any] | None:
    """Read the immutable admission for a turn, including older payload versions."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """SELECT * FROM hosted_room_driver_tasks
               WHERE room_id=? AND thread_id=? AND turn_id=?""",
            (identity.room_id, identity.thread_id, identity.turn_id),
        ).fetchone()
        return _task_from_row(row) if row is not None else None
    finally:
        conn.close()
