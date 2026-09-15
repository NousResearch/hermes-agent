"""Durable execution state for a same-gateway hosted room driver.

Owns leases, tasks, private terminal receipts and exact approval requests. Admission and demotion share
transactions with the room log so Stop and recovery reserves cannot race new work. Callers supply the
database path and clock; this module performs no model calls or session execution.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import threading
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, get_args

from gateway import hosted_rooms
from gateway.hosted_rooms_common import (
    DbPath, bounded_int, canonical_json, compact_json, connect, fenced_update, identifier, table_columns, text,
    transaction)

Clock = Callable[[], float]
OwnerProcessState = Literal["alive", "dead", "unknown"]
OwnerLiveness = Callable[[int, int], OwnerProcessState]
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
_TASK_PAYLOAD_OPTIONAL_FIELDS = frozenset({"target_member_id"})
_TASK_REQUIRED_PAYLOAD_FIELDS = _TASK_PAYLOAD_REQUIRED_FIELDS
_TASK_OPTIONAL_PAYLOAD_FIELDS = _TASK_PAYLOAD_OPTIONAL_FIELDS
_TASK_PAYLOAD_FIELDS = _TASK_PAYLOAD_REQUIRED_FIELDS | _TASK_PAYLOAD_OPTIONAL_FIELDS
_LEASE_COLUMNS = frozenset({
    "room_id", "gateway_id", "authority_epoch", "process_generation", "lease_generation", "expires_at", "acquired_at",
    "updated_at", "released_at", "process_pid", "process_start_time"})
_TASK_COLUMN_ORDER = (
    "room_id", "task_id", "thread_id", "turn_id", "source_event_seq", "payload_json", "payload_digest", "status",
    "execution_generation", "cancel_generation", "run_gateway_id", "run_process_generation", "run_lease_generation",
    "cancel_id", "settlement_id", "settlement_status", "result_json", "created_at", "updated_at", "started_at",
    "terminal_at", "indeterminate_at", "run_process_pid", "run_process_start_time")
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
_RUN_FENCE = ("run_gateway_id=? AND run_process_generation=? AND run_lease_generation=? "
              "AND run_process_pid IS ? AND run_process_start_time IS ?")
_SETTLE_SET = "status=?, settlement_id=?, settlement_status=?, result_json=?, terminal_at=?, updated_at=?"
_REQUEUE_SET = ("status='queued', run_gateway_id=NULL, run_process_generation=NULL, run_lease_generation=NULL, "
                "run_process_pid=NULL, run_process_start_time=NULL")
_CANCEL_SET = "status='cancelled', cancel_generation=?, cancel_id=?, terminal_at=?, updated_at=?"

_OWNER_LEASE_COLUMNS = frozenset({"process_pid", "process_start_time"})

_OWNER_TASK_COLUMNS = frozenset({"run_process_pid", "run_process_start_time"})

_LEGACY_LEASE_COLUMNS = _LEASE_COLUMNS - _OWNER_LEASE_COLUMNS

_LEGACY_TASK_COLUMNS = _TASK_COLUMNS - _OWNER_TASK_COLUMNS

_TERMINAL_RECEIPT_COLUMN_ORDER = (
    "room_id", "task_id", "execution_generation", "settlement_id", "status", "result_json", "created_at")
_TERMINAL_RECEIPT_COLUMNS = frozenset(_TERMINAL_RECEIPT_COLUMN_ORDER)
_APPROVAL_REQUEST_COLUMN_ORDER = (
    "room_id", "task_id", "execution_generation", "member_id", "request_id", "session_id", "action_json", "choice",
    "created_at", "updated_at", "consumed_at")
_APPROVAL_REQUEST_COLUMNS = frozenset(_APPROVAL_REQUEST_COLUMN_ORDER)
_ADMISSION_BARRIER_COLUMNS = frozenset({"room_id", "gateway_id", "authority_epoch", "reason", "created_at"})
_ADMISSION_BARRIER_PRIMARY_KEY = ("room_id", "gateway_id", "authority_epoch")
_DEMOTION_INTENT_COLUMNS = frozenset({
    "room_id", "gateway_id", "authority_epoch", "observed_gateway_id", "observed_epoch", "cancel_id", "created_at"})
_DEMOTION_INTENT_PRIMARY_KEY = ("room_id", "gateway_id", "authority_epoch")

_STARTUP_AUDIT_LOCK = threading.Lock()

_STARTUP_AUDITED_SCHEMAS: set[tuple[int, str, int, int, int]] = set()

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
_authority_epoch = partial(_bounded_int, message="authority_epoch must be a positive integer", low=1)
_canonical_json = partial(
    canonical_json, error=DriverValidationError, label="result", max_bytes=MAX_RESULT_JSON_BYTES, ensure_ascii=True)


class TaskAdmissionBlockedError(DriverStateError):
    """Raised when a durable Stop or authority fence rejects a new task."""


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


def _optional_process_integer(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DriverValidationError(f"{label} must be a positive integer or null")
    return value

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
    prompt = text(value["prompt"], error=DriverValidationError, label="prompt", max_bytes=MAX_PROMPT_BYTES, strip=False)
    source_event_seq = _bounded_int(
        value["source_event_seq"], message="source_event_seq must be a positive integer", low=1)
    normalized = {"target_profile": target_profile, "prompt": prompt, "source_event_seq": source_event_seq}
    if "target_member_id" in value:
        normalized["target_member_id"] = _identifier(value["target_member_id"], label="target_member_id")
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
    process_pid: int | None = None
    process_start_time: int | None = None

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
            run_process_pid INTEGER CHECK (run_process_pid IS NULL OR run_process_pid >= 1),
            run_process_start_time INTEGER CHECK (run_process_start_time IS NULL OR run_process_start_time >= 1),
            settlement_id TEXT, settlement_status TEXT, result_json TEXT, created_at REAL NOT NULL,
            updated_at REAL NOT NULL, started_at REAL, terminal_at REAL, indeterminate_at REAL,
            PRIMARY KEY (room_id, task_id), UNIQUE (room_id, thread_id, turn_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id))""")

def _create_terminal_receipt_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_terminal_receipts (
            room_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            execution_generation INTEGER NOT NULL
                CHECK (execution_generation >= 1),
            settlement_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('settled', 'failed')),
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, task_id, execution_generation),
            FOREIGN KEY (room_id, task_id)
                REFERENCES hosted_room_driver_tasks(room_id, task_id)
                ON DELETE CASCADE
        )"""
    )


def _create_approval_request_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_approval_requests (
            room_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            execution_generation INTEGER NOT NULL
                CHECK (execution_generation >= 1),
            member_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            action_json TEXT NOT NULL,
            choice TEXT CHECK (choice IN ('once', 'deny')),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            consumed_at REAL,
            PRIMARY KEY (
                room_id, task_id, execution_generation, member_id, request_id
            ),
            FOREIGN KEY (room_id, task_id)
                REFERENCES hosted_room_driver_tasks(room_id, task_id)
                ON DELETE CASCADE
        )"""
    )


def _create_admission_barrier_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_driver_admission_barriers (
            room_id TEXT NOT NULL,
            gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            reason TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, gateway_id, authority_epoch),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id)
                ON DELETE CASCADE
        )"""
    )

def _create_demotion_intent_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_driver_demotion_intents (
            room_id TEXT NOT NULL,
            gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            observed_gateway_id TEXT NOT NULL,
            observed_epoch INTEGER NOT NULL CHECK (
                observed_epoch > authority_epoch
            ),
            cancel_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, gateway_id, authority_epoch),
            FOREIGN KEY (room_id, gateway_id, authority_epoch)
                REFERENCES hosted_room_driver_admission_barriers(
                    room_id, gateway_id, authority_epoch
                ) ON DELETE CASCADE
        )"""
    )

def _demotion_intent_table_exists(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table'
                 AND name='hosted_room_driver_demotion_intents'"""
        ).fetchone()
        is not None
    )

def _raise_if_legacy_demotion_barrier_is_unrecoverable(
    conn: sqlite3.Connection,
) -> None:
    if _demotion_intent_table_exists(conn):
        return
    orphan = conn.execute(
        """SELECT 1
             FROM hosted_room_driver_admission_barriers AS barrier
             JOIN hosted_rooms AS room ON room.room_id=barrier.room_id
            WHERE barrier.reason='authority-demotion'
              AND room.disbanded_at IS NULL
              AND room.authority_gateway_id=barrier.gateway_id
              AND room.authority_epoch=barrier.authority_epoch
            LIMIT 1"""
    ).fetchone()
    if orphan is not None:
        raise DriverStateError(
            "unpublished authority-demotion barrier lacks resumable target metadata; "
            "restore the pre-update state snapshot or recreate the unpublished "
            "driver tables"
        )

def _raise_if_pending_demotion_intent_lacks_stop(
    conn: sqlite3.Connection,
) -> None:
    if not _demotion_intent_table_exists(conn):
        return
    intents = conn.execute(
        """SELECT intent.room_id, intent.gateway_id,
                  intent.authority_epoch, intent.cancel_id
             FROM hosted_room_driver_demotion_intents AS intent
             JOIN hosted_rooms AS room ON room.room_id=intent.room_id
            WHERE room.disbanded_at IS NULL
              AND room.authority_gateway_id=intent.gateway_id
              AND room.authority_epoch=intent.authority_epoch"""
    ).fetchall()
    for intent in intents:
        stop = conn.execute(
            """SELECT kind, actor_json, authority_epoch, payload_json
                 FROM hosted_room_events
                WHERE room_id=? AND event_id=?""",
            (
                str(intent["room_id"]),
                hosted_rooms._stop_event_id(str(intent["cancel_id"])),
            ),
        ).fetchone()
        try:
            actor = json.loads(stop["actor_json"]) if stop is not None else None
            payload = json.loads(stop["payload_json"]) if stop is not None else None
            stop_epoch = (
                int(stop["authority_epoch"]) if stop is not None else None
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            actor = None
            payload = None
            stop_epoch = None
        if (
            stop is None
            or str(stop["kind"]) != "room.stop_requested"
            or stop_epoch != int(intent["authority_epoch"])
            or actor
            != {"kind": "gateway", "id": str(intent["gateway_id"])}
            or payload != {"cancel_id": str(intent["cancel_id"])}
        ):
            raise DriverStateError(
                "unpublished authority-demotion intent lacks its atomic Stop "
                "fence; restore the pre-update state snapshot or recreate the "
                "unpublished driver tables"
            )

def _raise_if_terminal_recovery_headroom_is_unrecoverable(
    conn: sqlite3.Connection,
) -> None:
    liabilities = hosted_rooms._terminal_publication_liabilities(conn)
    for room_id in sorted({room_id for room_id, _ in liabilities}):
        try:
            hosted_rooms._assert_terminal_recovery_headroom(
                conn,
                room_id=room_id,
            )
        except hosted_rooms.HostedRoomError as exc:
            raise DriverStateError(
                "unpublished hosted-room tasks exceed durable terminal recovery "
                "headroom; restore the pre-update state snapshot or drain the "
                "unpublished driver state before starting this version"
            ) from exc

def _audit_terminal_recovery_headroom_once(
    conn: sqlite3.Connection,
    *,
    path: Path,
) -> None:
    """Audit one database schema once per process, not on every task read."""

    stat = path.stat()
    schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
    key = (
        os.getpid(),
        str(path.resolve()),
        int(stat.st_dev),
        int(stat.st_ino),
        schema_version,
    )
    with _STARTUP_AUDIT_LOCK:
        if key in _STARTUP_AUDITED_SCHEMAS:
            return
        _raise_if_terminal_recovery_headroom_is_unrecoverable(conn)
        _STARTUP_AUDITED_SCHEMAS.add(key)

def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_driver_leases (
            room_id TEXT PRIMARY KEY, gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1), process_generation TEXT NOT NULL,
            process_pid INTEGER CHECK (process_pid IS NULL OR process_pid >= 1),
            process_start_time INTEGER CHECK (process_start_time IS NULL OR process_start_time >= 1),
            lease_generation INTEGER NOT NULL CHECK (lease_generation >= 1),
            expires_at REAL NOT NULL, acquired_at REAL NOT NULL, updated_at REAL NOT NULL, released_at REAL,
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id))""")
    _create_task_table(conn)
    _create_terminal_receipt_table(conn)
    _create_approval_request_table(conn)
    _create_admission_barrier_table(conn)
    _create_demotion_intent_table(conn)
    _validate_schema(conn)
    conn.execute(_TASK_INDEX_SQL.format(if_not_exists="IF NOT EXISTS "))

def _validate_schema(conn: sqlite3.Connection) -> None:
    lease_columns = table_columns(conn, "hosted_room_driver_leases")
    task_columns = table_columns(conn, "hosted_room_driver_tasks")
    receipt_columns = table_columns(conn, "hosted_room_terminal_receipts")
    approval_columns = table_columns(conn, "hosted_room_approval_requests")
    barrier_info = conn.execute(
        "PRAGMA table_info(hosted_room_driver_admission_barriers)"
    ).fetchall()
    barrier_columns = frozenset(row[1] for row in barrier_info)
    barrier_primary_key = tuple(
        row[1]
        for row in sorted(barrier_info, key=lambda row: int(row[5]))
        if int(row[5]) > 0
    )
    intent_info = conn.execute(
        "PRAGMA table_info(hosted_room_driver_demotion_intents)"
    ).fetchall()
    intent_columns = frozenset(row[1] for row in intent_info)
    intent_primary_key = tuple(
        row[1]
        for row in sorted(intent_info, key=lambda row: int(row[5]))
        if int(row[5]) > 0
    )
    if (
        lease_columns != _LEASE_COLUMNS
        or task_columns != _TASK_COLUMNS
        or receipt_columns != _TERMINAL_RECEIPT_COLUMNS
        or approval_columns != _APPROVAL_REQUEST_COLUMNS
        or barrier_columns != _ADMISSION_BARRIER_COLUMNS
        or barrier_primary_key != _ADMISSION_BARRIER_PRIMARY_KEY
        or intent_columns != _DEMOTION_INTENT_COLUMNS
        or intent_primary_key != _DEMOTION_INTENT_PRIMARY_KEY
    ):
        raise DriverStateError(
            "unsupported unpublished hosted-room driver schema; "
            "recreate the driver tables before starting the driver")
    for table in ("hosted_room_driver_leases", "hosted_room_driver_tasks"):
        if not any(
            row[2] == "hosted_rooms" and row[3] == "room_id" and row[4] == "room_id"
            for row in conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()):
            raise DriverStateError(f"{table} is missing its hosted_rooms foreign key")
    receipt_foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(hosted_room_terminal_receipts)"
    ).fetchall()
    if not any(
        row[2] == "hosted_room_driver_tasks"
        and row[3] == "room_id"
        and row[4] == "room_id"
        for row in receipt_foreign_keys
    ):
        raise DriverStateError(
            "hosted_room_terminal_receipts is missing its task foreign key"
        )
    approval_foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(hosted_room_approval_requests)"
    ).fetchall()
    if not any(
        row[2] == "hosted_room_driver_tasks"
        and row[3] == "room_id"
        and row[4] == "room_id"
        for row in approval_foreign_keys
    ):
        raise DriverStateError(
            "hosted_room_approval_requests is missing its task foreign key"
        )
    barrier_foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(hosted_room_driver_admission_barriers)"
    ).fetchall()
    if not any(
        row[2] == "hosted_rooms"
        and row[3] == "room_id"
        and row[4] == "room_id"
        for row in barrier_foreign_keys
    ):
        raise DriverStateError(
            "hosted_room_driver_admission_barriers is missing its room foreign key"
        )
    intent_foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(hosted_room_driver_demotion_intents)"
    ).fetchall()
    intent_mapping = {
        (str(row[3]), str(row[4]))
        for row in intent_foreign_keys
        if row[2] == "hosted_room_driver_admission_barriers"
    }
    if intent_mapping != {
        ("room_id", "room_id"),
        ("gateway_id", "gateway_id"),
        ("authority_epoch", "authority_epoch"),
    }:
        raise DriverStateError(
            "hosted_room_driver_demotion_intents is missing its barrier foreign key"
        )

def _schema_objects_exist(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("""SELECT name FROM sqlite_master WHERE type='table'
           AND name IN ('hosted_room_driver_leases', 'hosted_room_driver_tasks')""").fetchall()
    if {row[0] for row in rows} != {"hosted_room_driver_leases", "hosted_room_driver_tasks"}:
        return False
    index = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_hosted_room_driver_tasks_status'").fetchone()
    return index is not None


def _task_schema_supports_current_statuses(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        """SELECT sql FROM sqlite_master
           WHERE type='table' AND name='hosted_room_driver_tasks'"""
    ).fetchone()
    sql = str(row[0] or "").lower() if row else ""
    return "'stopping'" in sql and "'deferred'" in sql


def _schema_is_current(conn: sqlite3.Connection) -> bool:
    if not _schema_objects_exist(conn) or not _task_schema_supports_current_statuses(conn):
        return False
    rows = conn.execute(
        """SELECT name FROM sqlite_master
           WHERE type='table' AND name IN (
                'hosted_room_terminal_receipts',
                'hosted_room_approval_requests',
                'hosted_room_driver_admission_barriers',
                'hosted_room_driver_demotion_intents'
            )"""
    ).fetchall()
    if {str(row[0]) for row in rows} != {
        "hosted_room_terminal_receipts",
        "hosted_room_approval_requests",
        "hosted_room_driver_admission_barriers",
        "hosted_room_driver_demotion_intents",
    }:
        return False
    lease_columns = table_columns(conn, "hosted_room_driver_leases")
    task_columns = table_columns(conn, "hosted_room_driver_tasks")
    if lease_columns != _LEASE_COLUMNS or task_columns != _TASK_COLUMNS:
        return False
    _validate_schema(conn)
    _raise_if_pending_demotion_intent_lacks_stop(conn)
    return True

def _migrate_owner_identity_columns(conn: sqlite3.Connection) -> None:
    """Add nullable process-identity fences to the unpublished driver schema."""

    lease_columns = table_columns(conn, "hosted_room_driver_leases")
    task_columns = table_columns(conn, "hosted_room_driver_tasks")
    if lease_columns not in {_LEGACY_LEASE_COLUMNS, _LEASE_COLUMNS}:
        raise DriverStateError(
            "unsupported unpublished hosted-room lease schema; "
            "recreate the driver tables before starting the driver"
        )
    if task_columns not in {_LEGACY_TASK_COLUMNS, _TASK_COLUMNS}:
        raise DriverStateError(
            "unsupported unpublished hosted-room task schema; "
            "recreate the driver tables before starting the driver"
        )
    if lease_columns == _LEGACY_LEASE_COLUMNS:
        conn.execute(
            "ALTER TABLE hosted_room_driver_leases ADD COLUMN process_pid INTEGER "
            "CHECK (process_pid IS NULL OR process_pid >= 1)"
        )
        conn.execute(
            "ALTER TABLE hosted_room_driver_leases ADD COLUMN process_start_time "
            "INTEGER CHECK (process_start_time IS NULL OR process_start_time >= 1)"
        )
    if task_columns == _LEGACY_TASK_COLUMNS:
        conn.execute(
            "ALTER TABLE hosted_room_driver_tasks ADD COLUMN run_process_pid INTEGER "
            "CHECK (run_process_pid IS NULL OR run_process_pid >= 1)"
        )
        conn.execute(
            "ALTER TABLE hosted_room_driver_tasks ADD COLUMN run_process_start_time "
            "INTEGER CHECK (run_process_start_time IS NULL OR "
            "run_process_start_time >= 1)"
        )

def _migrate_task_status_constraint(conn: sqlite3.Connection) -> None:
    """Expand the unpublished task-state CHECK without losing durable work."""
    dependent_rows: dict[str, list[tuple[Any, ...]]] = {}
    for table, columns in (
        ("hosted_room_terminal_receipts", _TERMINAL_RECEIPT_COLUMN_ORDER),
        ("hosted_room_approval_requests", _APPROVAL_REQUEST_COLUMN_ORDER),
    ):
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if exists is None:
            continue
        column_sql = ", ".join(columns)
        dependent_rows[table] = [
            tuple(row[column] for column in columns)
            for row in conn.execute(f"SELECT {column_sql} FROM {table}").fetchall()
        ]
        conn.execute(f"DROP TABLE {table}")
    conn.execute("DROP INDEX IF EXISTS idx_hosted_room_driver_tasks_status")
    _create_task_table(conn, "hosted_room_driver_tasks_next")
    columns = ", ".join(_TASK_COLUMN_ORDER)
    conn.execute(
        f"INSERT INTO hosted_room_driver_tasks_next ({columns}) SELECT {columns} FROM hosted_room_driver_tasks")
    conn.execute("DROP TABLE hosted_room_driver_tasks")
    conn.execute("ALTER TABLE hosted_room_driver_tasks_next RENAME TO hosted_room_driver_tasks")
    conn.execute(_TASK_INDEX_SQL.format(if_not_exists=""))
    _create_terminal_receipt_table(conn)
    _create_approval_request_table(conn)
    for table, columns in (
        ("hosted_room_terminal_receipts", _TERMINAL_RECEIPT_COLUMN_ORDER),
        ("hosted_room_approval_requests", _APPROVAL_REQUEST_COLUMN_ORDER),
    ):
        rows = dependent_rows.get(table, [])
        if not rows:
            continue
        column_sql = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(
            f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})",
            rows,
        )


def _connect(db_path: DbPath) -> sqlite3.Connection:
    """Open and audit the driver schema; migrations share one IMMEDIATE transaction."""
    path = Path(db_path)

    def ready(conn: sqlite3.Connection) -> bool:
        conn.execute("BEGIN")
        try:
            current = _schema_is_current(conn)
            if current:
                _audit_terminal_recovery_headroom_once(conn, path=path)
            return current
        finally:
            conn.rollback()

    def initialize(conn: sqlite3.Connection) -> None:
        # Another opener may have migrated while this connection waited for the write lock.
        if not _schema_is_current(conn):
            if _schema_objects_exist(conn):
                _migrate_owner_identity_columns(conn)
                if not _task_schema_supports_current_statuses(conn):
                    _migrate_task_status_constraint(conn)
                _create_terminal_receipt_table(conn)
                _create_approval_request_table(conn)
                _create_admission_barrier_table(conn)
                _raise_if_legacy_demotion_barrier_is_unrecoverable(conn)
                _create_demotion_intent_table(conn)
                _validate_schema(conn)
                _raise_if_pending_demotion_intent_lacks_stop(conn)
            else:
                _initialize_schema(conn)
        _audit_terminal_recovery_headroom_once(conn, path=path)

    return connect(
        path, db_label="state.db (hosted_room_driver)", ready=ready, initialize=initialize)


def _transaction(db_path: DbPath):
    return transaction(_connect, db_path, immediate=True)


def _lease_from_row(row: sqlite3.Row | dict[str, Any], *, reclaimed: bool = False) -> DriverLease:
    return DriverLease(
        room_id=row["room_id"], gateway_id=row["gateway_id"], authority_epoch=int(row["authority_epoch"]),
        process_generation=row["process_generation"], lease_generation=int(row["lease_generation"]),
        expires_at=float(row["expires_at"]), reclaimed=reclaimed,
        process_pid=int(row["process_pid"]) if row["process_pid"] is not None else None,
        process_start_time=int(row["process_start_time"]) if row["process_start_time"] is not None else None)

def _task_identity_from_row(row: sqlite3.Row) -> TaskIdentity:
    return TaskIdentity(
        room_id=row["room_id"], task_id=row["task_id"], thread_id=row["thread_id"], turn_id=row["turn_id"])


def _optional(cast: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: cast(value) if value is not None else None


# Task-view casts per row column (columns after payload_digest in _TASK_COLUMN_ORDER, same key order;
# result_json is exposed as "result"). Columns not listed are passed through untouched.
_TASK_VIEW_CASTS: dict[str, Callable[[Any], Any]] = {
    "execution_generation": int, "cancel_generation": int, "run_lease_generation": _optional(int),
    "run_process_pid": _optional(int), "run_process_start_time": _optional(int),
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


def _run_fence(lease: DriverLease) -> tuple[str, str, int, int | None, int | None]:
    """SQL bind order for the exact lease and process incarnation."""
    return (
        lease.gateway_id, lease.process_generation, lease.lease_generation,
        lease.process_pid, lease.process_start_time)


def _lease_row_matches(row: sqlite3.Row | None, lease: DriverLease) -> bool:
    return row is not None and (
        row["gateway_id"], int(row["authority_epoch"]), row["process_generation"], int(row["lease_generation"]),
        row["process_pid"], row["process_start_time"]
    ) == (
        lease.gateway_id, lease.authority_epoch, lease.process_generation, lease.lease_generation,
        lease.process_pid, lease.process_start_time)


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


def _reserve_deferred_transition(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    """Reopening a published, closed deferral must reacquire its terminal reserve."""
    if row["status"] != "deferred":
        return
    try:
        hosted_rooms._assert_terminal_recovery_headroom(
            conn, room_id=str(row["room_id"]),
            prospective_liability_keys=frozenset({(str(row["room_id"]), str(row["task_id"]))}))
    except hosted_rooms.HostedRoomError as exc:
        raise TaskAdmissionBlockedError(
            "deferred task transition is blocked to preserve terminal recovery headroom") from exc


def _transition(
    db_path: DbPath, identity: TaskIdentity, *, sql: str, set_params: tuple[Any, ...], fence_params: tuple[Any, ...],
    stale: str, now: float, lease: DriverLease | None = None, lease_first: bool = True,
    replay: Callable[[sqlite3.Row], dict[str, Any] | None] | None = None,
    guard: Callable[[sqlite3.Row], None] | None = None,
    storage_guard: Callable[[sqlite3.Connection, sqlite3.Row], None] | None = None) -> dict[str, Any]:
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
        if storage_guard is not None:
            storage_guard(conn, row)
        fenced_update(conn, sql, params, StaleTaskError(stale))
        return _task_from_row(_load_task(conn, identity))


def _generation_transition(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, name: str, execution_generation: int,
    cancel_generation: int, *, now: float, set_params: tuple[Any, ...],
    replay: Callable[[sqlite3.Row], Any] | None = None,
    require_active_source_event_seq: int | None = None) -> dict[str, Any]:
    """Lease-first transition from ``_GENERATION_TRANSITIONS`` fenced on status + both generations."""
    status, set_clause, generation_stale, stale = _GENERATION_TRANSITIONS[name]
    def guard(row: sqlite3.Row) -> None:
        if not _generations_match(row, status, execution_generation, cancel_generation):
            raise StaleTaskError(generation_stale)
    def storage_guard(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        _raise_if_task_fenced(
            conn,
            room_id=identity.room_id,
            source_event_seq=int(row["source_event_seq"]),
        )
        if require_active_source_event_seq is not None:
            active_source = conn.execute(
                """SELECT 1
                   FROM hosted_room_policy_events AS source
                   JOIN hosted_room_policy_threads AS active
                     ON active.room_id=source.room_id
                    AND active.thread_id=source.thread_id
                    AND active.discussion_event_id=source.discussion_event_id
                    AND active.completed=0
                   WHERE source.room_id=? AND source.seq=?""",
                (identity.room_id, require_active_source_event_seq),
            ).fetchone()
            if active_source is None:
                raise InvalidTaskTransitionError(
                    "cannot retry deferred task because its source discussion "
                    "is no longer active"
                )
        _reserve_deferred_transition(conn, row)
    return _transition(
        db_path, identity, lease=lease, now=now, replay=replay, guard=guard,
        storage_guard=storage_guard if name in {"requeue", "requeue_deferred"} else None,
        sql=_generation_update(set_clause, status),
        set_params=set_params, fence_params=(execution_generation, cancel_generation), stale=stale)


def _run_fence_transition(
    db_path: DbPath, attempt: TaskAttempt, *, guard_stale: str, lease_generation: Callable[[Any], int] = int,
    **transition: Any) -> dict[str, Any]:
    """Transition fenced on this attempt's running generation under its exact lease (row guard + SQL fence).
    ``lease_generation`` casts the stored run_lease_generation: ``int`` raises on NULL, ``int(v or 0)`` reads 0."""
    lease = attempt.lease
    def guard(row: sqlite3.Row) -> None:
        if not _generations_match(row, "running", attempt.execution_generation, attempt.cancel_generation) or (
            row["run_gateway_id"], row["run_process_generation"], lease_generation(row["run_lease_generation"]),
            row["run_process_pid"], row["run_process_start_time"]
        ) != _run_fence(lease):
            raise StaleTaskError(guard_stale)
    return _transition(
        db_path, attempt.identity, lease=lease, guard=guard,
        fence_params=(attempt.execution_generation, attempt.cancel_generation, *_run_fence(lease)), **transition)


def acquire_lease(
    db_path: DbPath, *, room_id: Any, gateway_id: Any, authority_epoch: Any, process_generation: Any, ttl_seconds: Any,
    clock: Clock, process_pid: Any = None, process_start_time: Any = None) -> DriverLease:
    """Acquire an empty or expired room lease with a monotonic generation."""
    room_id = _identifier(room_id, label="room_id")
    gateway_id = _identifier(gateway_id, label="gateway_id")
    authority_epoch = _bounded_int(authority_epoch, message="authority_epoch must be a positive integer", low=1)
    process_generation = _identifier(process_generation, label="process_generation")
    process_pid = _optional_process_integer(process_pid, label="process_pid")
    process_start_time = _optional_process_integer(process_start_time, label="process_start_time")
    now, expires_at = _lease_window(ttl_seconds, clock)
    with _transaction(db_path) as conn:
        _require_room_authority(conn, room_id, gateway_id, authority_epoch)
        row = conn.execute(_SELECT_LEASE, (room_id,)).fetchone()
        if row is None:
            conn.execute("""INSERT INTO hosted_room_driver_leases (
                       room_id, gateway_id, authority_epoch, process_generation, process_pid, process_start_time, lease_generation,
                       expires_at, acquired_at, updated_at, released_at
                   ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, NULL)""",
                (room_id, gateway_id, authority_epoch, process_generation, process_pid, process_start_time, expires_at, now, now))
            return _lease_from_row(conn.execute(_SELECT_LEASE, (room_id,)).fetchone())
        same_authority = row["gateway_id"] == gateway_id and int(row["authority_epoch"]) == authority_epoch
        live = row["released_at"] is None and float(row["expires_at"]) > now
        if (same_authority and row["process_generation"] == process_generation and live
                and row["process_pid"] == process_pid and row["process_start_time"] == process_start_time):
            renewed_expiry = max(float(row["expires_at"]), expires_at)
            conn.execute(
                "UPDATE hosted_room_driver_leases SET expires_at=?, updated_at=? WHERE room_id=? AND lease_generation=?",
                (renewed_expiry, now, room_id, int(row["lease_generation"])))
            return _lease_from_row({**dict(row), "expires_at": renewed_expiry})
        if same_authority and live:
            raise LeaseHeldError("room driver lease is held by another generation")
        fenced_update(conn, """UPDATE hosted_room_driver_leases
               SET gateway_id=?, authority_epoch=?, process_generation=?, lease_generation=lease_generation + 1,
                   process_pid=?, process_start_time=?,
                   expires_at=?, acquired_at=?, updated_at=?, released_at=NULL
               WHERE room_id=? AND lease_generation=? AND (
                   gateway_id != ? OR authority_epoch != ? OR released_at IS NOT NULL OR expires_at <= ?)""",
            (
                gateway_id, authority_epoch, process_generation, process_pid, process_start_time, expires_at, now, now, room_id,
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
                 AND lease_generation=? AND process_pid IS ? AND process_start_time IS ?
                 AND released_at IS NULL AND expires_at > ?""",
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


def _ensure_admission_barrier(
    conn: sqlite3.Connection, *, room_id: str, reason: str, gateway_id: str,
    authority_epoch: int, created_at: float) -> dict[str, Any]:
    existing = conn.execute(
        "SELECT room_id, gateway_id, authority_epoch, reason, created_at FROM hosted_room_driver_admission_barriers "
        "WHERE room_id=? AND gateway_id=? AND authority_epoch=?", (room_id, gateway_id, authority_epoch)).fetchone()
    if existing is not None:
        return {**dict(existing), "idempotent": True}
    conn.execute(
        "INSERT INTO hosted_room_driver_admission_barriers (room_id, gateway_id, authority_epoch, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?)", (room_id, gateway_id, authority_epoch, reason, created_at))
    return {
        "room_id": room_id, "gateway_id": gateway_id, "authority_epoch": authority_epoch,
        "reason": reason, "created_at": created_at, "idempotent": False}


def block_room_admissions(
    db_path: DbPath,
    *,
    room_id: Any,
    reason: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Permanently fence new tasks for one room authority epoch."""

    room_id = _identifier(room_id, label="room_id")
    reason = _identifier(reason, label="admission barrier reason")
    gateway_id = _identifier(expected_gateway_id, label="gateway_id")
    authority_epoch = _authority_epoch(expected_epoch)
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_room_authority(
            conn,
            room_id=room_id,
            gateway_id=gateway_id,
            epoch=authority_epoch,
        )
        return _ensure_admission_barrier(
            conn,
            room_id=room_id,
            reason=reason,
            gateway_id=gateway_id,
            authority_epoch=authority_epoch,
            created_at=now,
        )

def begin_room_demotion(
    db_path: DbPath,
    *,
    room_id: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
    observed_gateway_id: Any,
    observed_epoch: Any,
    cancel_id: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Atomically fence one authority epoch and persist its demotion intent."""

    room_id = _identifier(room_id, label="room_id")
    gateway_id = _identifier(expected_gateway_id, label="gateway_id")
    authority_epoch = _authority_epoch(expected_epoch)
    observed_gateway_id = _identifier(
        observed_gateway_id, label="observed_gateway_id"
    )
    observed_epoch = _authority_epoch(observed_epoch)
    cancel_id = _identifier(cancel_id, label="cancel_id")
    if observed_epoch <= authority_epoch:
        raise DriverValidationError(
            "observed_epoch must supersede the current authority epoch"
        )
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_room_authority(
            conn,
            room_id=room_id,
            gateway_id=gateway_id,
            epoch=authority_epoch,
        )
        barrier = _ensure_admission_barrier(
            conn,
            room_id=room_id,
            reason="authority-demotion",
            gateway_id=gateway_id,
            authority_epoch=authority_epoch,
            created_at=now,
        )
        if barrier["reason"] != "authority-demotion":
            raise TaskConflictError(
                "room authority epoch is blocked for a different reason"
            )
        existing = conn.execute(
            """SELECT room_id, gateway_id, authority_epoch,
                      observed_gateway_id, observed_epoch, cancel_id, created_at
                 FROM hosted_room_driver_demotion_intents
                WHERE room_id=? AND gateway_id=? AND authority_epoch=?""",
            (room_id, gateway_id, authority_epoch),
        ).fetchone()
        expected_identity = {
            "room_id": room_id,
            "gateway_id": gateway_id,
            "authority_epoch": authority_epoch,
            "observed_gateway_id": observed_gateway_id,
            "observed_epoch": observed_epoch,
        }
        if existing is not None:
            current = dict(existing)
            if any(
                current[key] != value for key, value in expected_identity.items()
            ):
                raise TaskConflictError(
                    "room authority epoch already has a different demotion intent"
                )
            hosted_rooms._request_room_stop_locked(
                conn,
                room_id=room_id,
                cancel_id=str(current["cancel_id"]),
                expected_gateway_id=gateway_id,
                expected_epoch=authority_epoch,
                now=now,
                demotion_control=True,
            )
            return {**current, "idempotent": True}
        conn.execute(
            """INSERT INTO hosted_room_driver_demotion_intents
               (room_id, gateway_id, authority_epoch, observed_gateway_id,
                observed_epoch, cancel_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                room_id,
                gateway_id,
                authority_epoch,
                observed_gateway_id,
                observed_epoch,
                cancel_id,
                now,
            ),
        )
        hosted_rooms._request_room_stop_locked(
            conn,
            room_id=room_id,
            cancel_id=cancel_id,
            expected_gateway_id=gateway_id,
            expected_epoch=authority_epoch,
            now=now,
            demotion_control=True,
        )
        return {
            **expected_identity,
            "cancel_id": cancel_id,
            "created_at": now,
            "idempotent": False,
        }

def pending_room_demotion(
    db_path: DbPath,
    *,
    room_id: Any,
) -> dict[str, Any] | None:
    """Return the resumable demotion intent for the room's current authority."""

    room_id = _identifier(room_id, label="room_id")
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """SELECT intent.room_id, intent.gateway_id,
                      intent.authority_epoch, intent.observed_gateway_id,
                      intent.observed_epoch, intent.cancel_id, intent.created_at
                 FROM hosted_room_driver_demotion_intents AS intent
                 JOIN hosted_rooms AS room ON room.room_id=intent.room_id
                WHERE intent.room_id=? AND room.disbanded_at IS NULL
                  AND room.authority_gateway_id=intent.gateway_id
                  AND room.authority_epoch=intent.authority_epoch""",
            (room_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None

def _raise_if_task_fenced(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    source_event_seq: int,
) -> None:
    barrier = conn.execute(
        """SELECT 1
             FROM hosted_room_driver_admission_barriers AS barrier
             JOIN hosted_rooms AS room ON room.room_id=barrier.room_id
            WHERE barrier.room_id=?
              AND barrier.gateway_id=room.authority_gateway_id
              AND barrier.authority_epoch=room.authority_epoch""",
        (room_id,),
    ).fetchone()
    if barrier is not None:
        raise TaskAdmissionBlockedError(
            "task admission or start is blocked by a durable room admission barrier"
        )
    stop = conn.execute(
        """SELECT MAX(seq) AS stop_seq FROM hosted_room_events
           WHERE room_id=? AND kind='room.stop_requested'""",
        (room_id,),
    ).fetchone()
    stop_seq = stop["stop_seq"] if stop is not None else None
    if stop_seq is not None and source_event_seq <= int(stop_seq):
        raise TaskAdmissionBlockedError(
            "task source event is behind the current Stop fence"
        )

def _stop_fenced_inactive_rows(
    conn: sqlite3.Connection,
    *,
    room_id: str,
) -> tuple[str, list[sqlite3.Row]] | None:
    _load_active_room(conn, room_id)
    stop = conn.execute(
        """SELECT seq, payload_json FROM hosted_room_events
           WHERE room_id=? AND kind='room.stop_requested'
           ORDER BY seq DESC LIMIT 1""",
        (room_id,),
    ).fetchone()
    if stop is None:
        return None
    try:
        payload = json.loads(stop["payload_json"])
        cancel_id = _identifier(payload.get("cancel_id"), label="cancel_id")
    except (AttributeError, json.JSONDecodeError, DriverValidationError) as exc:
        raise DriverStateError("latest Stop event payload is invalid") from exc
    rows = conn.execute(
        """SELECT * FROM hosted_room_driver_tasks
           WHERE room_id=? AND status IN ('queued', 'deferred')
             AND source_event_seq <= ?
           ORDER BY source_event_seq, created_at, task_id""",
        (room_id, int(stop["seq"])),
    ).fetchall()
    if any(row["status"] == "deferred" for row in rows):
        liabilities = hosted_rooms._terminal_publication_liabilities(conn)
        # Closed deferrals are retained retry records, not live work for this
        # Stop. Turning them into cancellations would revive unreserved work.
        rows = [row for row in rows if row["status"] != "deferred"
                or (room_id, str(row["task_id"])) in liabilities]
    return cancel_id, rows
def reconcile_stop_fenced_inactive_tasks(
    db_path: DbPath,
    *,
    room_id: Any,
    clock: Clock,
) -> list[TaskIdentity]:
    """Cancel inactive work stranded by a durable Stop before process exit."""

    room_id = _identifier(room_id, label="room_id")
    preflight = _connect(db_path)
    try:
        candidate = _stop_fenced_inactive_rows(preflight, room_id=room_id)
    finally:
        preflight.close()
    if candidate is None or not candidate[1]:
        return []

    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        current = _stop_fenced_inactive_rows(conn, room_id=room_id)
        if current is None or not current[1]:
            return []
        cancel_id, rows = current
        updated = conn.executemany(
            _CANCEL_QUEUED_SQL,
            [(int(row["cancel_generation"]) + 1, cancel_id, now, now,
              room_id, str(row["task_id"]), int(row["cancel_generation"])) for row in rows],
        )
        if updated.rowcount != len(rows):
            raise StaleTaskError(
                "stop-fenced inactive tasks changed during reconciliation"
            )
        return [_task_identity_from_row(row) for row in rows]

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

        _raise_if_task_fenced(
            conn,
            room_id=identity.room_id,
            source_event_seq=normalized_payload["source_event_seq"],
        )
        discussion_liability_key = (
            hosted_rooms._pending_discussion_liability_key_for_source(
                conn,
                room_id=identity.room_id,
                source_event_seq=normalized_payload["source_event_seq"],
                thread_id=identity.thread_id,
            )
        )
        try:
            hosted_rooms._assert_terminal_recovery_headroom(
                conn,
                room_id=identity.room_id,
                released_liability_keys=(
                    frozenset({discussion_liability_key})
                    if discussion_liability_key is not None
                    else frozenset()
                ),
                prospective_liability_keys=frozenset(
                    {(identity.room_id, identity.task_id)}
                ),
            )
        except hosted_rooms.HostedRoomError as exc:
            raise TaskAdmissionBlockedError(
                "task admission is blocked to preserve terminal recovery headroom"
            ) from exc

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
        _raise_if_task_fenced(conn, room_id=identity.room_id, source_event_seq=int(row["source_event_seq"]))
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
                   run_lease_generation=?, run_process_pid=?, run_process_start_time=?, started_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND status='queued' AND cancel_generation=?""",
            (
                execution_generation, *_run_fence(lease), now, now, identity.room_id, identity.task_id,
                expected_cancel_generation), StaleTaskError("task changed during start"))
        return TaskAttempt(
            identity=identity, lease=lease, execution_generation=execution_generation,
            cancel_generation=expected_cancel_generation)

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
    _check_same_room(lease, identity)
    now, replay, set_params = _settlement(settlement_id, status, result, clock)
    return _transition(
        db_path, identity, lease=lease, lease_first=False, now=now, replay=replay, sql=_SETTLE_STOPPING_SQL,
        set_params=set_params, fence_params=(expected_execution_generation, expected_cancel_generation),
        stale="task completion lost the stop race")


def resolve_indeterminate_task(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, settlement_id: Any, status: TerminalStatus, result: Any, clock: Clock
) -> dict[str, Any]:
    """Commit a verified historical receipt under the current room lease."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    now, replay, set_params = _settlement(settlement_id, status, result, clock)
    return _generation_transition(
        db_path, identity, lease, "resolve", expected_execution_generation, expected_cancel_generation, now=now,
        replay=replay, set_params=set_params)


def resolve_indeterminate_cancellation(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, cancel_id: Any, clock: Clock) -> dict[str, Any]:
    """Commit a verified terminal cancellation for an uncertain attempt."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    cancel_id = _identifier(cancel_id, label="cancel_id")
    now = _timestamp(clock)
    return _generation_transition(
        db_path, identity, lease, "resolve_cancel", expected_execution_generation, expected_cancel_generation, now=now,
        replay=_cancel_replay(cancel_id), set_params=(expected_cancel_generation + 1, cancel_id, now, now))


def requeue_indeterminate_task(
    db_path: DbPath, identity: TaskIdentity, lease: DriverLease, *, expected_execution_generation: int,
    expected_cancel_generation: int, clock: Clock) -> dict[str, Any]:
    """Explicitly retry uncertain work after an operator accepts at-least-once risk."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    now = _timestamp(clock)
    return _generation_transition(
        db_path, identity, lease, "requeue", expected_execution_generation, expected_cancel_generation, now=now,
        set_params=(now,))

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
    expected_cancel_generation: int, clock: Clock, require_active_source_event_seq: int | None = None) -> dict[str, Any]:
    """Explicitly retry a fenced deferred turn under a new generation."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    if require_active_source_event_seq is not None:
        _bounded_int(
            require_active_source_event_seq, low=1,
            message="require_active_source_event_seq must be a positive integer")
    now = _timestamp(clock)
    return _generation_transition(
        db_path, identity, lease, "requeue_deferred", expected_execution_generation, expected_cancel_generation,
        now=now, set_params=(now,), require_active_source_event_seq=require_active_source_event_seq)


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
        db_path, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard,
        storage_guard=_reserve_deferred_transition, sql=_CANCEL_QUEUED_SQL,
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


def record_terminal_receipt(
    db_path: DbPath,
    identity: TaskIdentity,
    *,
    execution_generation: int,
    settlement_id: Any,
    status: TerminalStatus,
    result: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Persist one exact terminal proof before process-local publication."""

    if (
        isinstance(execution_generation, bool)
        or not isinstance(execution_generation, int)
        or execution_generation < 1
    ):
        raise DriverValidationError(
            "execution_generation must be a positive integer"
        )
    settlement_id = _identifier(settlement_id, label="settlement_id")
    if status not in {"settled", "failed"}:
        raise DriverValidationError("terminal receipt status must be settled or failed")
    result_json = _canonical_json(result)
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        task = _load_task(conn, identity)
        if int(task["execution_generation"]) != execution_generation:
            raise StaleTaskError("terminal receipt belongs to a stale task generation")
        existing = conn.execute(
            """SELECT * FROM hosted_room_terminal_receipts
               WHERE room_id=? AND task_id=? AND execution_generation=?""",
            (identity.room_id, identity.task_id, execution_generation),
        ).fetchone()
        if existing is not None:
            if (
                existing["settlement_id"] != settlement_id
                or existing["status"] != status
                or existing["result_json"] != result_json
            ):
                raise TaskConflictError(
                    "terminal receipt generation already has different content"
                )
            return {
                "identity": identity,
                "execution_generation": execution_generation,
                "settlement_id": settlement_id,
                "status": status,
                "result": json.loads(result_json),
                "created_at": float(existing["created_at"]),
                "idempotent": True,
            }
        conn.execute(
            """INSERT INTO hosted_room_terminal_receipts(
                   room_id, task_id, execution_generation, settlement_id,
                   status, result_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                identity.room_id,
                identity.task_id,
                execution_generation,
                settlement_id,
                status,
                result_json,
                now,
            ),
        )
    return {
        "identity": identity,
        "execution_generation": execution_generation,
        "settlement_id": settlement_id,
        "status": status,
        "result": json.loads(result_json),
        "created_at": now,
        "idempotent": False,
    }


def get_terminal_receipt(
    db_path: DbPath,
    identity: TaskIdentity,
    *,
    execution_generation: int,
) -> dict[str, Any] | None:
    """Read the private durable terminal proof for one exact attempt."""

    if (
        isinstance(execution_generation, bool)
        or not isinstance(execution_generation, int)
        or execution_generation < 1
    ):
        raise DriverValidationError(
            "execution_generation must be a positive integer"
        )
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """SELECT * FROM hosted_room_terminal_receipts
               WHERE room_id=? AND task_id=? AND execution_generation=?""",
            (identity.room_id, identity.task_id, execution_generation),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "identity": identity,
        "execution_generation": execution_generation,
        "settlement_id": str(row["settlement_id"]),
        "status": str(row["status"]),
        "result": json.loads(row["result_json"]),
        "created_at": float(row["created_at"]),
    }


def publish_approval_request(
    db_path: DbPath,
    identity: TaskIdentity,
    *,
    execution_generation: int,
    member_id: Any,
    request_id: Any,
    session_id: Any,
    action: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Publish one exact owner-side approval request to shared durable state."""

    member_id = _identifier(member_id, label="member_id")
    request_id = _identifier(request_id, label="request_id")
    session_id = _identifier(session_id, label="session_id")
    if (
        isinstance(execution_generation, bool)
        or not isinstance(execution_generation, int)
        or execution_generation < 1
    ):
        raise DriverValidationError(
            "execution_generation must be a positive integer"
        )
    action_json = _canonical_json(action)
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        task = _load_task(conn, identity)
        if int(task["execution_generation"]) != execution_generation:
            raise StaleTaskError("approval request belongs to a stale task generation")
        if task["status"] != "running":
            raise InvalidTaskTransitionError(
                "approval request requires a running task generation"
            )
        existing = conn.execute(
            """SELECT * FROM hosted_room_approval_requests
               WHERE room_id=? AND task_id=? AND execution_generation=?
                 AND member_id=? AND request_id=?""",
            (
                identity.room_id,
                identity.task_id,
                execution_generation,
                member_id,
                request_id,
            ),
        ).fetchone()
        if existing is not None:
            if (
                existing["session_id"] != session_id
                or existing["action_json"] != action_json
            ):
                raise TaskConflictError(
                    "approval request id already has different content"
                )
            row = existing
        else:
            conn.execute(
                """INSERT INTO hosted_room_approval_requests(
                       room_id, task_id, execution_generation, member_id,
                       request_id, session_id, action_json, choice,
                       created_at, updated_at, consumed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL)""",
                (
                    identity.room_id,
                    identity.task_id,
                    execution_generation,
                    member_id,
                    request_id,
                    session_id,
                    action_json,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """SELECT * FROM hosted_room_approval_requests
                   WHERE room_id=? AND task_id=? AND execution_generation=?
                     AND member_id=? AND request_id=?""",
                (
                    identity.room_id,
                    identity.task_id,
                    execution_generation,
                    member_id,
                    request_id,
                ),
            ).fetchone()
    return {
        "identity": identity,
        "execution_generation": execution_generation,
        "member_id": member_id,
        "request_id": request_id,
        "session_id": session_id,
        "action": json.loads(action_json),
        "choice": row["choice"],
        "consumed": row["consumed_at"] is not None,
    }


def decide_approval_request(
    db_path: DbPath,
    identity: TaskIdentity,
    *,
    execution_generation: int,
    member_id: Any,
    request_id: Any,
    choice: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Record an exact dashboard decision without impersonating the owner."""

    member_id = _identifier(member_id, label="member_id")
    request_id = _identifier(request_id, label="request_id")
    if choice not in {"once", "deny"}:
        raise DriverValidationError("approval choice must be once or deny")
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        task = _load_task(conn, identity)
        if int(task["execution_generation"]) != execution_generation:
            raise StaleTaskError("approval decision belongs to a stale task generation")
        if task["status"] != "running":
            raise StaleTaskError("approval request task is no longer running")
        row = conn.execute(
            """SELECT * FROM hosted_room_approval_requests
               WHERE room_id=? AND task_id=? AND execution_generation=?
                 AND member_id=? AND request_id=?""",
            (
                identity.room_id,
                identity.task_id,
                execution_generation,
                member_id,
                request_id,
            ),
        ).fetchone()
        if row is None or row["consumed_at"] is not None:
            raise StaleTaskError("approval request is no longer pending")
        if row["choice"] is not None:
            if row["choice"] != choice:
                raise TaskConflictError("approval request already has another choice")
            return {"choice": choice, "idempotent": True}
        updated = conn.execute(
            """UPDATE hosted_room_approval_requests
               SET choice=?, updated_at=?
               WHERE room_id=? AND task_id=? AND execution_generation=?
                 AND member_id=? AND request_id=? AND choice IS NULL
                 AND consumed_at IS NULL""",
            (
                choice,
                now,
                identity.room_id,
                identity.task_id,
                execution_generation,
                member_id,
                request_id,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("approval request changed during decision")
    return {"choice": choice, "idempotent": False}


def list_pending_approval_requests(
    db_path: DbPath,
    *,
    room_id: Any,
) -> list[dict[str, Any]]:
    """List unconsumed approvals whose task generation is still current."""

    room_id = _identifier(room_id, label="room_id")
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT approvals.*, tasks.thread_id, tasks.turn_id
               FROM hosted_room_approval_requests AS approvals
               JOIN hosted_room_driver_tasks AS tasks
                 ON tasks.room_id=approvals.room_id
                AND tasks.task_id=approvals.task_id
                AND tasks.execution_generation=approvals.execution_generation
               WHERE approvals.room_id=? AND approvals.consumed_at IS NULL
                 AND tasks.status='running'
               ORDER BY approvals.created_at, approvals.task_id,
                        approvals.member_id, approvals.request_id""",
            (room_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "identity": TaskIdentity(
                room_id=str(row["room_id"]),
                task_id=str(row["task_id"]),
                thread_id=str(row["thread_id"]),
                turn_id=str(row["turn_id"]),
            ),
            "execution_generation": int(row["execution_generation"]),
            "member_id": str(row["member_id"]),
            "request_id": str(row["request_id"]),
            "session_id": str(row["session_id"]),
            "action": json.loads(row["action_json"]),
            "choice": row["choice"],
        }
        for row in rows
    ]


def mark_approval_consumed(
    db_path: DbPath,
    identity: TaskIdentity,
    *,
    execution_generation: int,
    member_id: Any,
    request_id: Any,
    choice: Any,
    clock: Clock,
) -> bool:
    """Owner-side acknowledgement after the local approval queue resolves."""

    member_id = _identifier(member_id, label="member_id")
    request_id = _identifier(request_id, label="request_id")
    if choice not in {"once", "deny"}:
        raise DriverValidationError("approval choice must be once or deny")
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        updated = conn.execute(
            """UPDATE hosted_room_approval_requests
               SET consumed_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND execution_generation=?
                 AND member_id=? AND request_id=? AND choice=?
                 AND consumed_at IS NULL""",
            (
                now,
                now,
                identity.room_id,
                identity.task_id,
                execution_generation,
                member_id,
                request_id,
                choice,
            ),
        )
        return updated.rowcount == 1


def clear_member_approval_requests(
    db_path: DbPath,
    *,
    room_id: Any,
    member_id: Any,
) -> int:
    """Drop stale owner-side requests after the member leaves approval state."""

    room_id = _identifier(room_id, label="room_id")
    member_id = _identifier(member_id, label="member_id")
    with _transaction(db_path) as conn:
        removed = conn.execute(
            """DELETE FROM hosted_room_approval_requests
               WHERE room_id=? AND member_id=? AND consumed_at IS NULL""",
            (room_id, member_id),
        )
        return max(0, int(removed.rowcount or 0))


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
from contextlib import contextmanager  # noqa: F401,E402
import re  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
