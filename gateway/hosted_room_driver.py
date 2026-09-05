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
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, get_args

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
# Source artifacts retain 30 days of bytes plus a 30-day ACK tombstone.
ARTIFACT_RETRY_RETENTION_SECONDS = 60 * 24 * 60 * 60
MAX_RETAINED_TERMINAL_TASKS = 2048
MAX_TASK_PRUNE_BATCH = 1000
TASK_STATUSES = frozenset(get_args(TaskStatus))
TERMINAL_STATUSES = frozenset({"settled", "failed", "cancelled"})

_TASK_PAYLOAD_REQUIRED_FIELDS = frozenset({"target_profile", "prompt", "source_event_seq"})
_TASK_PAYLOAD_OPTIONAL_FIELDS = frozenset({"target_member_id", "attachments", "input_context", "recipient_member_ids"})
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
        "deferred task generation changed", "deferred task changed during requeue")}


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
    if "recipient_member_ids" in value:
        raw_recipients = value["recipient_member_ids"]
        if not isinstance(raw_recipients, list) or not 1 <= len(raw_recipients) <= 6:
            raise DriverValidationError("recipient_member_ids must contain 1-6 members")
        recipients = [_identifier(item, label="recipient_member_id") for item in raw_recipients]
        if len(set(recipients)) != len(recipients):
            raise DriverValidationError("recipient_member_ids must be unique")
        normalized["recipient_member_ids"] = recipients
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
    _initialize_retry_receipt_table(conn)
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
    """Expand the unpublished task-state CHECK without losing durable work."""
    preserved_receipts = []
    receipt_table = conn.execute(
        """SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='hosted_room_retry_receipts'"""
    ).fetchone()
    if receipt_table is not None:
        preserved_receipts.extend(
            conn.execute(
                """SELECT retry_id, room_id, task_id,
                          source_execution_generation, created_at
                     FROM hosted_room_retry_receipts"""
            ).fetchall()
        )
    if _task_schema_has_legacy_retry_id(conn):
        preserved_receipts.extend(
            conn.execute(
                """SELECT retry_id, room_id, task_id, execution_generation, updated_at
                 FROM hosted_room_driver_tasks
                WHERE retry_id IS NOT NULL AND retry_id != ''"""
            ).fetchall()
        )
    conn.execute("DROP TABLE IF EXISTS hosted_room_retry_receipts")
    conn.execute("DROP INDEX IF EXISTS idx_hosted_room_driver_tasks_status")
    _create_task_table(conn, "hosted_room_driver_tasks_next")
    columns = ", ".join(_TASK_COLUMN_ORDER)
    conn.execute(
        f"""INSERT INTO hosted_room_driver_tasks_next ({columns})
             SELECT {columns} FROM hosted_room_driver_tasks"""
    )
    conn.execute("DROP TABLE hosted_room_driver_tasks")
    conn.execute(
        "ALTER TABLE hosted_room_driver_tasks_next RENAME TO hosted_room_driver_tasks"
    )
    conn.execute(
        """CREATE INDEX idx_hosted_room_driver_tasks_status
           ON hosted_room_driver_tasks(
               room_id, status, source_event_seq, created_at, task_id
           )"""
    )
    _initialize_retry_receipt_table(conn)
    for receipt in preserved_receipts:
        existing = conn.execute(
            "SELECT room_id, task_id FROM hosted_room_retry_receipts WHERE retry_id=?",
            (str(receipt["retry_id"]),),
        ).fetchone()
        if existing is not None and (
            str(existing["room_id"]),
            str(existing["task_id"]),
        ) != (str(receipt["room_id"]), str(receipt["task_id"])):
            raise DriverStateError("draft retry_id is bound to multiple tasks")
        conn.execute(
            """INSERT OR IGNORE INTO hosted_room_retry_receipts(
                   retry_id, room_id, task_id, source_execution_generation, created_at
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                str(receipt["retry_id"]),
                str(receipt["room_id"]),
                str(receipt["task_id"]),
                int(
                    receipt["source_execution_generation"]
                    if "source_execution_generation" in receipt.keys()
                    else receipt["execution_generation"]
                ),
                float(
                    receipt["created_at"]
                    if "created_at" in receipt.keys()
                    else receipt["updated_at"]
                ),
            ),
        )


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
        return ("'stopping'" in sql and "'deferred'" in sql
                and not _task_schema_has_legacy_retry_id(conn))
    conn = connect(
        db_path, db_label="state.db (hosted_room_driver)", ready=ready,
        initialize=lambda conn: (_migrate_task_status_constraint if existing[0] else _initialize_schema)(conn))
    try:
        if existing[0]:
            _validate_schema(conn)
        _initialize_retry_receipt_table(conn)
        conn.commit()
    except Exception:
        conn.rollback()
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
    "terminal_at": _optional(float), "indeterminate_at": _optional(float)}


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


def _generation_transition(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, name: str, execution_generation: int,
    cancel_generation: int, *, now: float, set_params: tuple[Any, ...],
    replay: Callable[[sqlite3.Row], Any] | None = None) -> dict[str, Any]:
    """Lease-first transition from ``_GENERATION_TRANSITIONS`` fenced on status + both generations."""
    status, set_clause, generation_stale, stale = _GENERATION_TRANSITIONS[name]
    def guard(row: sqlite3.Row) -> None:
        if not _generations_match(row, status, execution_generation, cancel_generation):
            raise StaleTaskError(generation_stale)
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
    db_path: Path | str,
    identity: TaskIdentity,
    lease: DriverLease,
    *,
    expected_cancel_generation: int,
    clock: Clock,
    not_before: float | None = None,
) -> TaskAttempt | None:
    """Move one queued task to running under the current driver lease."""
    if lease.room_id != identity.room_id:
        raise DriverValidationError("lease and task belong to different rooms")
    if (
        not isinstance(expected_cancel_generation, int)
        or expected_cancel_generation < 0
    ):
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    now = _timestamp(clock)
    if not_before is not None:
        not_before = _finite(lambda: not_before, "not_before must be a finite number")
    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        row = _load_task(conn, identity)
        if int(row["cancel_generation"]) != expected_cancel_generation:
            raise StaleTaskError("task cancellation generation changed")
        if row["status"] != "queued":
            raise InvalidTaskTransitionError(
                f"cannot start task in state '{row['status']}'"
            )
        if _cancel_task_behind_stop_fence(conn, row, now=now) is not None:
            return None
        # Stop must settle a queued task even while its route is cooling down.
        if not_before is not None and now < not_before:
            return None
        unresolved = conn.execute(
            """SELECT task_id, status FROM hosted_room_driver_tasks
               WHERE room_id=? AND status IN ('running', 'indeterminate', 'stopping')
               ORDER BY source_event_seq, created_at, task_id LIMIT 1""",
            (identity.room_id,),
        ).fetchone()
        if unresolved is not None:
            raise InvalidTaskTransitionError(
                "room recovery must resolve the prior task before starting new work"
            )
        next_queued = conn.execute(
            """SELECT task_id FROM hosted_room_driver_tasks
               WHERE room_id=? AND status='queued'
               ORDER BY source_event_seq, created_at, task_id LIMIT 1""",
            (identity.room_id,),
        ).fetchone()
        if next_queued is None or next_queued["task_id"] != identity.task_id:
            raise InvalidTaskTransitionError(
                "task is not next in the hosted room event order"
            )
        execution_generation = int(row["execution_generation"]) + 1
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='running', execution_generation=?,
                   run_gateway_id=?, run_process_generation=?,
                   run_lease_generation=?, started_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND status='queued'
                 AND cancel_generation=?""",
            (
                execution_generation,
                lease.gateway_id,
                lease.process_generation,
                lease.lease_generation,
                now,
                now,
                identity.room_id,
                identity.task_id,
                expected_cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("task changed during start")
        return TaskAttempt(
            identity=identity,
            lease=lease,
            execution_generation=execution_generation,
            cancel_generation=expected_cancel_generation,
        )


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
    expected_cancel_generation: int, settlement_id: Any, status: TerminalStatus, result: Any, clock: Clock
) -> dict[str, Any]:
    """Commit a completion that won the race with an unacknowledged Stop."""
    _terminal_settlement_id(settlement_id, status)  # settlement errors take precedence over generation errors
    if expected_execution_generation < 1 or expected_cancel_generation < 1:
        raise DriverValidationError("stopping settlement generations are invalid")
    now, replay, set_params = _settlement(settlement_id, status, result, clock)
    return _transition(
        db_path, identity, lease=lease, lease_first=False, now=now, replay=replay, sql=_SETTLE_STOPPING_SQL,
        set_params=set_params, fence_params=(expected_execution_generation, expected_cancel_generation),
        stale="task completion lost the stop race")


def resolve_indeterminate_task(
    db_path: Path | str,
    identity: TaskIdentity,
    lease: DriverLease,
    *,
    expected_execution_generation: int,
    expected_cancel_generation: int,
    settlement_id: Any,
    status: TerminalStatus,
    result: Any,
    clock: Clock,
    retry_id: Any = None,
) -> dict[str, Any]:
    """Commit a verified historical receipt under the current room lease."""
    if lease.room_id != identity.room_id:
        raise DriverValidationError("lease and task belong to different rooms")
    if (
        not isinstance(expected_execution_generation, int)
        or expected_execution_generation < 1
    ):
        raise DriverValidationError(
            "expected_execution_generation must be a positive integer"
        )
    if (
        not isinstance(expected_cancel_generation, int)
        or expected_cancel_generation < 0
    ):
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    settlement_id = _identifier(settlement_id, label="settlement_id")
    retry_id = _identifier(retry_id, label="retry_id") if retry_id is not None else None
    if status not in {"settled", "failed"}:
        raise DriverValidationError("status must be 'settled' or 'failed'")
    result_json = _canonical_json(result)
    now = _timestamp(clock)

    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        row = _load_task(conn, identity)
        if row["settlement_id"] is not None:
            if (
                row["settlement_id"] == settlement_id
                and row["settlement_status"] == status
                and row["result_json"] == result_json
            ):
                _record_retry_receipt(conn, retry_id=retry_id, row=row, now=now)
                return _task_from_row(row, idempotent=True)
            raise TaskConflictError("task already has a different terminal settlement")
        if (
            row["status"] != "indeterminate"
            or int(row["execution_generation"]) != expected_execution_generation
            or int(row["cancel_generation"]) != expected_cancel_generation
        ):
            raise StaleTaskError("indeterminate task generation changed")
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status=?, settlement_id=?, settlement_status=?,
                   result_json=?, terminal_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND status='indeterminate'
                 AND execution_generation=? AND cancel_generation=?""",
            (
                status,
                settlement_id,
                status,
                result_json,
                now,
                now,
                identity.room_id,
                identity.task_id,
                expected_execution_generation,
                expected_cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("indeterminate task changed during reconciliation")
        _record_retry_receipt(conn, retry_id=retry_id, row=row, now=now)
        return _task_from_row(_load_task(conn, identity))


def resolve_indeterminate_cancellation(
    db_path: Path | str,
    identity: TaskIdentity,
    lease: DriverLease,
    *,
    expected_execution_generation: int,
    expected_cancel_generation: int,
    cancel_id: Any,
    clock: Clock,
    retry_id: Any = None,
) -> dict[str, Any]:
    """Commit a verified terminal cancellation for an uncertain attempt."""
    if lease.room_id != identity.room_id:
        raise DriverValidationError("lease and task belong to different rooms")
    if (
        not isinstance(expected_execution_generation, int)
        or expected_execution_generation < 1
    ):
        raise DriverValidationError(
            "expected_execution_generation must be a positive integer"
        )
    if (
        not isinstance(expected_cancel_generation, int)
        or expected_cancel_generation < 0
    ):
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    cancel_id = _identifier(cancel_id, label="cancel_id")
    retry_id = _identifier(retry_id, label="retry_id") if retry_id is not None else None
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        row = _load_task(conn, identity)
        if row["status"] == "cancelled" and row["cancel_id"] == cancel_id:
            _record_retry_receipt(conn, retry_id=retry_id, row=row, now=now)
            return _task_from_row(row, idempotent=True)
        if (
            row["status"] != "indeterminate"
            or int(row["execution_generation"]) != expected_execution_generation
            or int(row["cancel_generation"]) != expected_cancel_generation
        ):
            raise StaleTaskError("indeterminate cancellation proof is stale")
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='cancelled', cancel_generation=?, cancel_id=?,
                   terminal_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND status='indeterminate'
                 AND execution_generation=? AND cancel_generation=?""",
            (
                expected_cancel_generation + 1,
                cancel_id,
                now,
                now,
                identity.room_id,
                identity.task_id,
                expected_execution_generation,
                expected_cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("indeterminate cancellation proof lost its fence")
        _record_retry_receipt(conn, retry_id=retry_id, row=row, now=now)
        return _task_from_row(_load_task(conn, identity))


def requeue_indeterminate_task(
    db_path: Path | str,
    identity: TaskIdentity,
    lease: DriverLease,
    *,
    expected_execution_generation: int,
    expected_cancel_generation: int,
    clock: Clock,
    retry_id: Any = None,
) -> dict[str, Any]:
    """Explicitly retry uncertain work after an operator accepts at-least-once risk."""
    if lease.room_id != identity.room_id:
        raise DriverValidationError("lease and task belong to different rooms")
    if (
        not isinstance(expected_execution_generation, int)
        or expected_execution_generation < 1
    ):
        raise DriverValidationError(
            "expected_execution_generation must be a positive integer"
        )
    if (
        not isinstance(expected_cancel_generation, int)
        or expected_cancel_generation < 0
    ):
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    retry_id = _identifier(retry_id, label="retry_id") if retry_id is not None else None
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        row = _load_task(conn, identity)
        if (
            row["status"] != "indeterminate"
            or int(row["execution_generation"]) != expected_execution_generation
            or int(row["cancel_generation"]) != expected_cancel_generation
        ):
            raise StaleTaskError("indeterminate task generation changed")
        stopped = _cancel_task_behind_stop_fence(conn, row, now=now)
        if stopped is not None:
            _record_retry_receipt(conn, retry_id=retry_id, row=row, now=now)
            return _task_from_row(stopped)
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='queued', run_gateway_id=NULL,
                   run_process_generation=NULL, run_lease_generation=NULL,
                   started_at=NULL, indeterminate_at=NULL, updated_at=?
               WHERE room_id=? AND task_id=? AND status='indeterminate'
                 AND execution_generation=? AND cancel_generation=?""",
            (
                now,
                identity.room_id,
                identity.task_id,
                expected_execution_generation,
                expected_cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("indeterminate task changed during requeue")
        _record_retry_receipt(conn, retry_id=retry_id, row=row, now=now)
        return _task_from_row(_load_task(conn, identity))


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
    db_path: Path | str,
    identity: TaskIdentity,
    lease: DriverLease,
    *,
    expected_execution_generation: int,
    expected_cancel_generation: int,
    clock: Clock,
    retry_id: Any = None,
) -> dict[str, Any]:
    """Explicitly retry a fenced deferred turn under a new generation."""

    if lease.room_id != identity.room_id:
        raise DriverValidationError("lease and task belong to different rooms")
    if (
        not isinstance(expected_execution_generation, int)
        or expected_execution_generation < 1
    ):
        raise DriverValidationError(
            "expected_execution_generation must be a positive integer"
        )
    if (
        not isinstance(expected_cancel_generation, int)
        or expected_cancel_generation < 0
    ):
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    retry_id = _identifier(retry_id, label="retry_id") if retry_id is not None else None
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        row = _load_task(conn, identity)
        if (
            row["status"] != "deferred"
            or int(row["execution_generation"]) != expected_execution_generation
            or int(row["cancel_generation"]) != expected_cancel_generation
        ):
            raise StaleTaskError("deferred task generation changed")
        stopped = _cancel_task_behind_stop_fence(conn, row, now=now)
        if stopped is not None:
            _record_retry_receipt(conn, retry_id=retry_id, row=row, now=now)
            return _task_from_row(stopped)
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='queued', run_gateway_id=NULL,
                   run_process_generation=NULL, run_lease_generation=NULL,
                   result_json=NULL, started_at=NULL, terminal_at=NULL,
                   indeterminate_at=NULL, updated_at=?
               WHERE room_id=? AND task_id=? AND status='deferred'
                 AND execution_generation=? AND cancel_generation=?""",
            (
                now,
                identity.room_id,
                identity.task_id,
                expected_execution_generation,
                expected_cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("deferred task changed during requeue")
        _record_retry_receipt(conn, retry_id=retry_id, row=row, now=now)
        return _task_from_row(_load_task(conn, identity))


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
        if row["status"] not in {"queued", "deferred"}:
            raise InvalidTaskTransitionError("running work requires acknowledged two-phase cancellation")
        _require_cancel_generation(row, expected_cancel_generation)
    return _transition(
        db_path, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, sql=_CANCEL_QUEUED_SQL,
        set_params=(expected_cancel_generation + 1, cancel_id, now, now), fence_params=(expected_cancel_generation,),
        stale="task changed during cancellation")


def begin_task_cancel(
    db_path: DbPath, identity: TaskIdentity, *, cancel_id: Any, expected_cancel_generation: int, clock: Clock
) -> dict[str, Any]:
    """Persist a stop intent without claiming the remote run has stopped."""
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
    db_path: DbPath, identity: TaskIdentity, *, cancel_id: Any, expected_cancel_generation: int, clock: Clock
) -> dict[str, Any]:
    """Commit cancellation only after the transport acknowledges exact Stop."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    now = _timestamp(clock)
    def guard(row: sqlite3.Row) -> None:
        if (row["status"], row["cancel_id"], int(row["cancel_generation"])) != (
            "stopping", cancel_id, expected_cancel_generation):
            raise StaleTaskError("task stop acknowledgement is stale")
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
        retries = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_artifact_retries'").fetchone()
        retry_guard, retry_params = "", ()
        if retries is not None:
            retry_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(hosted_room_artifact_retries)"
                ).fetchall()
            }
            retry_age_column = next(
                (name for name in ("created_at", "updated_at") if name in retry_columns),
                None,
            )
            retry_guard = (
                "AND NOT EXISTS (SELECT 1 FROM hosted_room_artifact_retries r "
                "WHERE r.room_id=t.room_id AND r.task_id=t.task_id "
                "AND r.execution_generation=t.execution_generation"
            )
            if retry_age_column is not None:
                retry_guard += f" AND r.{retry_age_column}>?"
                retry_params = (now - ARTIFACT_RETRY_RETENTION_SECONDS,)
            retry_guard += ")"
        rows = conn.execute(f"""SELECT t.task_id, t.terminal_at FROM hosted_room_driver_tasks t
                WHERE t.room_id=? AND t.status IN ('settled', 'failed', 'cancelled')
                  AND EXISTS (SELECT 1 FROM hosted_room_policy_publications p
                              WHERE p.room_id=t.room_id AND p.task_id=t.task_id
                                AND p.kind IN ('turn.settled', 'turn.failed', 'turn.cancelled'))
                {retry_guard}
                ORDER BY t.terminal_at DESC, t.task_id ASC""",
            (room_id, *retry_params)).fetchall()
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


def _record_retry_receipt(
    conn: sqlite3.Connection,
    *,
    retry_id: str | None,
    row: sqlite3.Row,
    now: float,
) -> None:
    if retry_id is None:
        return
    retry_id = _identifier(retry_id, label="retry_id")
    existing = conn.execute(
        """SELECT room_id, task_id FROM hosted_room_retry_receipts
            WHERE retry_id=?""",
        (retry_id,),
    ).fetchone()
    if existing is not None:
        if (str(existing["room_id"]), str(existing["task_id"])) != (
            str(row["room_id"]),
            str(row["task_id"]),
        ):
            raise TaskConflictError("retry_id is already bound to another task")
        return
    conn.execute(
        """INSERT INTO hosted_room_retry_receipts(
               retry_id, room_id, task_id, source_execution_generation, created_at
           ) VALUES (?, ?, ?, ?, ?)""",
        (
            retry_id,
            str(row["room_id"]),
            str(row["task_id"]),
            int(row["execution_generation"]),
            now,
        ),
    )


def retry_receipt_exists(
    db_path: Path | str,
    *,
    room_id: Any,
    task_id: Any,
    retry_id: Any,
) -> bool:
    room_id = _identifier(room_id, label="room_id")
    task_id = _identifier(task_id, label="task_id")
    retry_id = _identifier(retry_id, label="retry_id")
    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT 1 FROM hosted_room_retry_receipts
                WHERE retry_id=? AND room_id=? AND task_id=?""",
            (retry_id, room_id, task_id),
        ).fetchone()
        return row is not None


def require_active_lease_in_transaction(
    conn: sqlite3.Connection,
    lease: DriverLease,
    *,
    now: Any,
) -> DriverLease:
    """Validate an exact lease inside a caller-owned SQLite transaction."""

    timestamp = _timestamp(lambda: now)
    return _lease_from_row(_require_active_lease(conn, lease, now=timestamp))


def _cancel_task_behind_stop_fence(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now: float,
) -> sqlite3.Row | None:
    stop = conn.execute(
        """SELECT seq FROM hosted_room_events
            WHERE room_id=? AND kind='room.stop_requested'
            ORDER BY seq DESC LIMIT 1""",
        (row["room_id"],),
    ).fetchone()
    if stop is None or int(row["source_event_seq"]) >= int(stop["seq"]):
        return None
    cancel_id = f"stop-fence:{int(stop['seq'])}"
    changed = conn.execute(
        """UPDATE hosted_room_driver_tasks
              SET status='cancelled', cancel_generation=cancel_generation + 1,
                  cancel_id=?, terminal_at=?, updated_at=?
            WHERE room_id=? AND task_id=? AND status=?
              AND execution_generation=? AND cancel_generation=?""",
        (
            cancel_id,
            now,
            now,
            row["room_id"],
            row["task_id"],
            row["status"],
            int(row["execution_generation"]),
            int(row["cancel_generation"]),
        ),
    )
    if changed.rowcount != 1:
        raise StaleTaskError("task changed while applying the room stop fence")
    return _load_task(
        conn,
        TaskIdentity(
            room_id=str(row["room_id"]),
            task_id=str(row["task_id"]),
            thread_id=str(row["thread_id"]),
            turn_id=str(row["turn_id"]),
        ),
    )


def require_active_lease(
    db_path: Path | str,
    lease: DriverLease,
    *,
    clock: Clock,
) -> DriverLease:
    """Revalidate one exact lease generation without extending its lifetime."""

    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        current = _require_active_lease(conn, lease, now=now)
        return _lease_from_row(current)


def _initialize_retry_receipt_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_retry_receipts (
            retry_id TEXT PRIMARY KEY,
            room_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            source_execution_generation INTEGER NOT NULL
                CHECK (source_execution_generation >= 0),
            created_at REAL NOT NULL,
            FOREIGN KEY (room_id, task_id)
                REFERENCES hosted_room_driver_tasks(room_id, task_id)
                ON DELETE CASCADE
        )"""
    )


def _task_schema_has_legacy_retry_id(conn: sqlite3.Connection) -> bool:
    return any(
        row[1] == "retry_id"
        for row in conn.execute("PRAGMA table_info(hosted_room_driver_tasks)")
    )


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
