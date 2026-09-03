"""Durable execution state for a same-gateway hosted room driver.

This module owns only the driver lease and task state machine. It does not
invoke models, touch sessions, or depend on the hosted-room event log. Callers
provide both the database path and clock so recovery and fencing behavior can
be tested without process-global state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from gateway import hosted_room_driver_schema as _driver_schema
from gateway.hosted_room_driver_schema import (
    DriverStateError,
    _TASK_COLUMN_ORDER as _TASK_COLUMN_ORDER,
)


Clock = Callable[[], float]
TaskStatus = Literal[
    "queued",
    "running",
    "settled",
    "failed",
    "cancelled",
    "indeterminate",
    "deferred",
    "stopping",
]
TerminalStatus = Literal["settled", "failed"]

MAX_IDENTIFIER_CHARS = 128
MAX_PROMPT_BYTES = 128 * 1024
MAX_RESULT_JSON_BYTES = 256 * 1024
TERMINAL_TASK_RETENTION_SECONDS = 30 * 24 * 60 * 60
MAX_RETAINED_TERMINAL_TASKS = 2048
MAX_TASK_PRUNE_BATCH = 1000
TASK_STATUSES = frozenset({
    "queued",
    "running",
    "settled",
    "failed",
    "cancelled",
    "indeterminate",
    "deferred",
    "stopping",
})
TERMINAL_STATUSES = frozenset({"settled", "failed", "cancelled"})

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_TASK_PAYLOAD_REQUIRED_FIELDS = frozenset({
    "target_profile",
    "prompt",
    "source_event_seq",
})
_TASK_PAYLOAD_OPTIONAL_FIELDS = frozenset({"target_member_id"})


class DriverValidationError(DriverStateError):
    """Raised when an identifier, clock, TTL, or payload is invalid."""


class RoomUnavailableError(DriverStateError):
    """Raised when the hosted room does not exist or was disbanded."""


class LeaseHeldError(DriverStateError):
    """Raised when another unexpired driver generation owns the room."""


class StaleLeaseError(DriverStateError):
    """Raised when a lease generation can no longer mutate room state."""


class TaskConflictError(DriverStateError):
    """Raised when an idempotency key is reused for different task state."""


class StaleTaskError(DriverStateError):
    """Raised when an obsolete task attempt or cancellation tries to commit."""


class InvalidTaskTransitionError(DriverStateError):
    """Raised when a requested task transition is not allowed."""


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise DriverValidationError(f"{label} must be a string")
    value = value.strip()
    if (
        not value
        or len(value) > MAX_IDENTIFIER_CHARS
        or not _IDENTIFIER_RE.fullmatch(value)
    ):
        raise DriverValidationError(f"invalid {label}")
    return value


def _timestamp(clock: Clock) -> float:
    if not callable(clock):
        raise DriverValidationError("clock must be callable")
    try:
        value = float(clock())
    except (TypeError, ValueError, OverflowError) as exc:
        raise DriverValidationError("clock must return a finite number") from exc
    if not math.isfinite(value):
        raise DriverValidationError("clock must return a finite number")
    return value


def _ttl(value: Any) -> float:
    try:
        ttl = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DriverValidationError(
            "ttl_seconds must be a finite positive number"
        ) from exc
    if not math.isfinite(ttl) or ttl <= 0:
        raise DriverValidationError("ttl_seconds must be a finite positive number")
    return ttl


def _expiry(now: float, ttl: float) -> float:
    expires_at = now + ttl
    if not math.isfinite(expires_at):
        raise DriverValidationError("lease expiry must be finite")
    return expires_at


def _canonical_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise DriverValidationError("result must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_RESULT_JSON_BYTES:
        raise DriverValidationError("result is too large")
    return encoded


def _authority_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DriverValidationError("authority_epoch must be a positive integer")
    return value


def _task_payload(value: Any) -> tuple[dict[str, Any], str, str]:
    if not isinstance(value, dict):
        raise DriverValidationError("payload must be an object")
    unknown = set(value) - _TASK_PAYLOAD_REQUIRED_FIELDS - _TASK_PAYLOAD_OPTIONAL_FIELDS
    missing = _TASK_PAYLOAD_REQUIRED_FIELDS - set(value)
    if unknown:
        raise DriverValidationError(
            f"unknown payload fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise DriverValidationError(
            f"missing payload fields: {', '.join(sorted(missing))}"
        )

    target_profile = _identifier(value["target_profile"], label="target_profile")
    prompt = value["prompt"]
    if not isinstance(prompt, str):
        raise DriverValidationError("prompt must be a string")
    if not prompt.strip():
        raise DriverValidationError("prompt must not be empty")
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise DriverValidationError("prompt is too large")
    source_event_seq = value["source_event_seq"]
    if (
        isinstance(source_event_seq, bool)
        or not isinstance(source_event_seq, int)
        or source_event_seq < 1
    ):
        raise DriverValidationError("source_event_seq must be a positive integer")

    normalized = {
        "target_profile": target_profile,
        "prompt": prompt,
        "source_event_seq": source_event_seq,
    }
    if "target_member_id" in value:
        normalized["target_member_id"] = _identifier(
            value["target_member_id"], label="target_member_id"
        )
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return normalized, encoded, digest


@dataclass(frozen=True)
class TaskIdentity:
    """Stable identity for one admitted room turn."""

    room_id: str
    task_id: str
    thread_id: str
    turn_id: str

    def __post_init__(self) -> None:
        for field in ("room_id", "task_id", "thread_id", "turn_id"):
            object.__setattr__(
                self,
                field,
                _identifier(getattr(self, field), label=field),
            )


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


def _connect(db_path: Path | str) -> sqlite3.Connection:
    from hermes_state import apply_wal_with_fallback

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        apply_wal_with_fallback(conn, db_label="state.db (hosted_room_driver)")
        conn.execute("PRAGMA foreign_keys=ON")
        if _driver_schema._schema_objects_exist(conn):
            if _driver_schema._task_schema_has_legacy_retry_id(
                conn
            ) or not _driver_schema._task_schema_supports_current_statuses(conn):
                conn.execute("BEGIN IMMEDIATE")
                _driver_schema._migrate_task_status_constraint(conn)
                conn.commit()
            _driver_schema._initialize_retry_receipt_table(conn)
            conn.commit()
            _driver_schema._validate_schema(conn)
            return conn
        # Schema creation is one database-wide transaction. The driver schema
        # has never shipped, so an incompatible draft schema fails closed
        # instead of attempting a partial in-place migration.
        conn.execute("BEGIN IMMEDIATE")
        _driver_schema._initialize_schema(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    return conn


@contextmanager
def _transaction(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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


def _lease_from_row(
    row: sqlite3.Row | dict[str, Any], *, reclaimed: bool = False
) -> DriverLease:
    return DriverLease(
        room_id=row["room_id"],
        gateway_id=row["gateway_id"],
        authority_epoch=int(row["authority_epoch"]),
        process_generation=row["process_generation"],
        lease_generation=int(row["lease_generation"]),
        expires_at=float(row["expires_at"]),
        reclaimed=reclaimed,
    )


def _task_identity_from_row(row: sqlite3.Row) -> TaskIdentity:
    return TaskIdentity(
        room_id=row["room_id"],
        task_id=row["task_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
    )


def _task_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    try:
        raw_payload = json.loads(row["payload_json"])
        payload, encoded_payload, payload_digest = _task_payload(raw_payload)
    except (TypeError, json.JSONDecodeError, DriverValidationError) as exc:
        raise TaskConflictError("stored task payload is invalid") from exc
    if (
        encoded_payload != row["payload_json"]
        or payload_digest != row["payload_digest"]
        or payload["source_event_seq"] != int(row["source_event_seq"])
    ):
        raise TaskConflictError("stored task payload failed its integrity check")
    result = json.loads(row["result_json"]) if row["result_json"] is not None else None
    return {
        "identity": _task_identity_from_row(row),
        "payload": payload,
        "payload_digest": row["payload_digest"],
        "status": row["status"],
        "execution_generation": int(row["execution_generation"]),
        "cancel_generation": int(row["cancel_generation"]),
        "run_gateway_id": row["run_gateway_id"],
        "run_process_generation": row["run_process_generation"],
        "run_lease_generation": (
            int(row["run_lease_generation"])
            if row["run_lease_generation"] is not None
            else None
        ),
        "cancel_id": row["cancel_id"],
        "settlement_id": row["settlement_id"],
        "settlement_status": row["settlement_status"],
        "result": result,
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "started_at": (
            float(row["started_at"]) if row["started_at"] is not None else None
        ),
        "terminal_at": (
            float(row["terminal_at"]) if row["terminal_at"] is not None else None
        ),
        "indeterminate_at": (
            float(row["indeterminate_at"])
            if row["indeterminate_at"] is not None
            else None
        ),
        "idempotent": idempotent,
    }


def _load_task(conn: sqlite3.Connection, identity: TaskIdentity) -> sqlite3.Row:
    row = conn.execute(
        """SELECT * FROM hosted_room_driver_tasks
           WHERE room_id=? AND task_id=?""",
        (identity.room_id, identity.task_id),
    ).fetchone()
    if row is None:
        raise TaskConflictError("task does not exist")
    if _task_identity_from_row(row) != identity:
        raise TaskConflictError("task_id is already bound to a different turn")
    return row


def _load_active_room(conn: sqlite3.Connection, room_id: str) -> sqlite3.Row:
    try:
        row = conn.execute(
            """SELECT room_id, authority_gateway_id, authority_epoch, disbanded_at
               FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            raise RoomUnavailableError("hosted room does not exist") from exc
        raise
    if row is None:
        raise RoomUnavailableError("hosted room does not exist")
    if row["disbanded_at"] is not None:
        raise RoomUnavailableError("hosted room is disbanded")
    return row


def _require_room_authority(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    gateway_id: str,
    authority_epoch: int,
) -> sqlite3.Row:
    room = _load_active_room(conn, room_id)
    if (
        room["authority_gateway_id"] != gateway_id
        or int(room["authority_epoch"]) != authority_epoch
    ):
        raise StaleLeaseError("hosted room authority changed")
    return room


def _require_active_lease(
    conn: sqlite3.Connection,
    lease: DriverLease,
    *,
    now: float,
) -> sqlite3.Row:
    _require_room_authority(
        conn,
        room_id=lease.room_id,
        gateway_id=lease.gateway_id,
        authority_epoch=lease.authority_epoch,
    )
    row = conn.execute(
        "SELECT * FROM hosted_room_driver_leases WHERE room_id=?",
        (lease.room_id,),
    ).fetchone()
    if (
        row is None
        or row["gateway_id"] != lease.gateway_id
        or int(row["authority_epoch"]) != lease.authority_epoch
        or row["process_generation"] != lease.process_generation
        or int(row["lease_generation"]) != lease.lease_generation
        or row["released_at"] is not None
        or float(row["expires_at"]) <= now
    ):
        raise StaleLeaseError("driver lease is stale or expired")
    return row


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


def acquire_lease(
    db_path: Path | str,
    *,
    room_id: Any,
    gateway_id: Any,
    authority_epoch: Any,
    process_generation: Any,
    ttl_seconds: Any,
    clock: Clock,
) -> DriverLease:
    """Acquire an empty or expired room lease with a monotonic generation."""
    room_id = _identifier(room_id, label="room_id")
    gateway_id = _identifier(gateway_id, label="gateway_id")
    authority_epoch = _authority_epoch(authority_epoch)
    process_generation = _identifier(
        process_generation,
        label="process_generation",
    )
    ttl_seconds = _ttl(ttl_seconds)
    now = _timestamp(clock)
    expires_at = _expiry(now, ttl_seconds)

    with _transaction(db_path) as conn:
        _require_room_authority(
            conn,
            room_id=room_id,
            gateway_id=gateway_id,
            authority_epoch=authority_epoch,
        )
        row = conn.execute(
            "SELECT * FROM hosted_room_driver_leases WHERE room_id=?",
            (room_id,),
        ).fetchone()
        if row is None:
            conn.execute(
                """INSERT INTO hosted_room_driver_leases (
                       room_id, gateway_id, authority_epoch, process_generation,
                       lease_generation, expires_at, acquired_at,
                       updated_at, released_at
                   ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, NULL)""",
                (
                    room_id,
                    gateway_id,
                    authority_epoch,
                    process_generation,
                    expires_at,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM hosted_room_driver_leases WHERE room_id=?",
                (room_id,),
            ).fetchone()
            return _lease_from_row(row)

        if (
            row["gateway_id"] == gateway_id
            and int(row["authority_epoch"]) == authority_epoch
            and row["process_generation"] == process_generation
            and row["released_at"] is None
            and float(row["expires_at"]) > now
        ):
            renewed_expiry = max(float(row["expires_at"]), expires_at)
            conn.execute(
                """UPDATE hosted_room_driver_leases
                   SET expires_at=?, updated_at=?
                   WHERE room_id=? AND lease_generation=?""",
                (renewed_expiry, now, room_id, int(row["lease_generation"])),
            )
            current = dict(row)
            current["expires_at"] = renewed_expiry
            return _lease_from_row(current)

        same_authority = (
            row["gateway_id"] == gateway_id
            and int(row["authority_epoch"]) == authority_epoch
        )
        if (
            same_authority
            and row["released_at"] is None
            and float(row["expires_at"]) > now
        ):
            raise LeaseHeldError("room driver lease is held by another generation")

        previous_generation = int(row["lease_generation"])
        updated = conn.execute(
            """UPDATE hosted_room_driver_leases
               SET gateway_id=?, authority_epoch=?, process_generation=?,
                   lease_generation=lease_generation + 1,
                   expires_at=?, acquired_at=?, updated_at=?, released_at=NULL
               WHERE room_id=? AND lease_generation=?
                 AND (
                     gateway_id != ? OR authority_epoch != ?
                     OR released_at IS NOT NULL OR expires_at <= ?
                 )""",
            (
                gateway_id,
                authority_epoch,
                process_generation,
                expires_at,
                now,
                now,
                room_id,
                previous_generation,
                gateway_id,
                authority_epoch,
                now,
            ),
        )
        if updated.rowcount != 1:
            raise LeaseHeldError("room driver lease changed during acquisition")
        current = conn.execute(
            "SELECT * FROM hosted_room_driver_leases WHERE room_id=?",
            (room_id,),
        ).fetchone()
        return _lease_from_row(current, reclaimed=True)


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


def renew_lease(
    db_path: Path | str,
    lease: DriverLease,
    *,
    ttl_seconds: Any,
    clock: Clock,
) -> DriverLease:
    """Renew the exact active lease generation or fail closed."""
    ttl_seconds = _ttl(ttl_seconds)
    now = _timestamp(clock)
    requested_expiry = _expiry(now, ttl_seconds)
    with _transaction(db_path) as conn:
        current = _require_active_lease(conn, lease, now=now)
        expires_at = max(float(current["expires_at"]), requested_expiry)
        updated = conn.execute(
            """UPDATE hosted_room_driver_leases
               SET expires_at=?, updated_at=?
               WHERE room_id=? AND gateway_id=? AND process_generation=?
                 AND lease_generation=? AND released_at IS NULL AND expires_at > ?""",
            (
                expires_at,
                now,
                lease.room_id,
                lease.gateway_id,
                lease.process_generation,
                lease.lease_generation,
                now,
            ),
        )
        if updated.rowcount != 1:
            raise StaleLeaseError("driver lease changed during renewal")
        return DriverLease(
            room_id=lease.room_id,
            gateway_id=lease.gateway_id,
            authority_epoch=lease.authority_epoch,
            process_generation=lease.process_generation,
            lease_generation=lease.lease_generation,
            expires_at=expires_at,
        )


def release_lease(
    db_path: Path | str,
    lease: DriverLease,
    *,
    clock: Clock,
) -> dict[str, Any]:
    """Release the exact active lease generation idempotently."""
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_room_authority(
            conn,
            room_id=lease.room_id,
            gateway_id=lease.gateway_id,
            authority_epoch=lease.authority_epoch,
        )
        row = conn.execute(
            "SELECT * FROM hosted_room_driver_leases WHERE room_id=?",
            (lease.room_id,),
        ).fetchone()
        if (
            row is None
            or row["gateway_id"] != lease.gateway_id
            or int(row["authority_epoch"]) != lease.authority_epoch
            or row["process_generation"] != lease.process_generation
            or int(row["lease_generation"]) != lease.lease_generation
        ):
            raise StaleLeaseError("driver lease is stale")
        if row["released_at"] is not None:
            return {"lease": _lease_from_row(row), "idempotent": True}
        if float(row["expires_at"]) <= now:
            raise StaleLeaseError("driver lease expired before release")
        running = conn.execute(
            """SELECT 1 FROM hosted_room_driver_tasks
               WHERE room_id=? AND status='running' LIMIT 1""",
            (lease.room_id,),
        ).fetchone()
        if running is not None:
            raise InvalidTaskTransitionError(
                "cannot release a room lease while tasks are running"
            )
        conn.execute(
            """UPDATE hosted_room_driver_leases
               SET expires_at=?, updated_at=?, released_at=?
               WHERE room_id=? AND lease_generation=?""",
            (now, now, now, lease.room_id, lease.lease_generation),
        )
        current = dict(row)
        current["expires_at"] = now
        current["updated_at"] = now
        current["released_at"] = now
        return {"lease": _lease_from_row(current), "idempotent": False}


def admit_task(
    db_path: Path | str,
    identity: TaskIdentity,
    *,
    payload: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Persist a queued task, or return the identical admission."""
    normalized_payload, payload_json, payload_digest = _task_payload(payload)
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _load_active_room(conn, identity.room_id)
        existing = conn.execute(
            """SELECT * FROM hosted_room_driver_tasks
               WHERE room_id=? AND task_id=?""",
            (identity.room_id, identity.task_id),
        ).fetchone()
        if existing is not None:
            if _task_identity_from_row(existing) != identity:
                raise TaskConflictError("task_id is already bound to a different turn")
            if (
                existing["payload_digest"] != payload_digest
                or existing["payload_json"] != payload_json
            ):
                raise TaskConflictError(
                    "task_id is already bound to a different payload"
                )
            return _task_from_row(existing, idempotent=True)

        turn = conn.execute(
            """SELECT * FROM hosted_room_driver_tasks
               WHERE room_id=? AND thread_id=? AND turn_id=?""",
            (identity.room_id, identity.thread_id, identity.turn_id),
        ).fetchone()
        if turn is not None:
            raise TaskConflictError("thread_id and turn_id are already bound to a task")

        conn.execute(
            """INSERT INTO hosted_room_driver_tasks (
                   room_id, task_id, thread_id, turn_id,
                   source_event_seq, payload_json, payload_digest, status,
                   execution_generation, cancel_generation,
                   created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 0, 0, ?, ?)""",
            (
                identity.room_id,
                identity.task_id,
                identity.thread_id,
                identity.turn_id,
                normalized_payload["source_event_seq"],
                payload_json,
                payload_digest,
                now,
                now,
            ),
        )
        row = _load_task(conn, identity)
        return _task_from_row(row)


def start_task(
    db_path: Path | str,
    identity: TaskIdentity,
    lease: DriverLease,
    *,
    expected_cancel_generation: int,
    clock: Clock,
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
    db_path: Path | str,
    attempt: TaskAttempt,
    *,
    settlement_id: Any,
    status: TerminalStatus,
    result: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Commit one terminal result if every lease and task fence still matches."""
    settlement_id = _identifier(settlement_id, label="settlement_id")
    if status not in {"settled", "failed"}:
        raise DriverValidationError("status must be 'settled' or 'failed'")
    result_json = _canonical_json(result)
    now = _timestamp(clock)

    with _transaction(db_path) as conn:
        row = _load_task(conn, attempt.identity)
        if row["settlement_id"] is not None:
            if (
                row["settlement_id"] == settlement_id
                and row["settlement_status"] == status
                and row["result_json"] == result_json
            ):
                return _task_from_row(row, idempotent=True)
            raise TaskConflictError("task already has a different terminal settlement")

        _require_active_lease(conn, attempt.lease, now=now)
        expected = (
            row["status"] == "running"
            and int(row["execution_generation"]) == attempt.execution_generation
            and int(row["cancel_generation"]) == attempt.cancel_generation
            and row["run_gateway_id"] == attempt.lease.gateway_id
            and row["run_process_generation"] == attempt.lease.process_generation
            and int(row["run_lease_generation"]) == attempt.lease.lease_generation
        )
        if not expected:
            raise StaleTaskError("task attempt is stale or cancelled")

        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status=?, settlement_id=?, settlement_status=?,
                   result_json=?, terminal_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND status='running'
                 AND execution_generation=? AND cancel_generation=?
                 AND run_gateway_id=? AND run_process_generation=?
                 AND run_lease_generation=?""",
            (
                status,
                settlement_id,
                status,
                result_json,
                now,
                now,
                attempt.identity.room_id,
                attempt.identity.task_id,
                attempt.execution_generation,
                attempt.cancel_generation,
                attempt.lease.gateway_id,
                attempt.lease.process_generation,
                attempt.lease.lease_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("task changed during settlement")
        return _task_from_row(_load_task(conn, attempt.identity))


def settle_stopping_task(
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
) -> dict[str, Any]:
    """Commit a completion that won the race with an unacknowledged Stop."""
    settlement_id = _identifier(settlement_id, label="settlement_id")
    if status not in {"settled", "failed"}:
        raise DriverValidationError("status must be 'settled' or 'failed'")
    if expected_execution_generation < 1 or expected_cancel_generation < 1:
        raise DriverValidationError("stopping settlement generations are invalid")
    result_json = _canonical_json(result)
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        row = _load_task(conn, identity)
        if row["settlement_id"] is not None:
            if (
                row["settlement_id"] == settlement_id
                and row["settlement_status"] == status
                and row["result_json"] == result_json
            ):
                return _task_from_row(row, idempotent=True)
            raise TaskConflictError("task already has a different terminal settlement")
        _require_active_lease(conn, lease, now=now)
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status=?, settlement_id=?, settlement_status=?,
                   result_json=?, terminal_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND status='stopping'
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
            raise StaleTaskError("task completion lost the stop race")
        return _task_from_row(_load_task(conn, identity))


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
    db_path: Path | str,
    identity: TaskIdentity,
    lease: DriverLease,
    *,
    expected_execution_generation: int,
    expected_cancel_generation: int,
    reason: Any,
    clock: Clock,
) -> dict[str, Any]:
    """Fence one uncertain attempt and release later room work."""

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
    reason = _identifier(reason, label="defer_reason")
    result_json = _canonical_json({"reason": reason, "retryable": True})
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        row = _load_task(conn, identity)
        if (
            row["status"] == "deferred"
            and int(row["execution_generation"]) == expected_execution_generation
            and int(row["cancel_generation"]) == expected_cancel_generation
            and row["result_json"] == result_json
        ):
            return _task_from_row(row, idempotent=True)
        if (
            row["status"] != "indeterminate"
            or int(row["execution_generation"]) != expected_execution_generation
            or int(row["cancel_generation"]) != expected_cancel_generation
        ):
            raise StaleTaskError("indeterminate task generation changed")
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='deferred', result_json=?, terminal_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND status='indeterminate'
                 AND execution_generation=? AND cancel_generation=?""",
            (
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
            raise StaleTaskError("indeterminate task changed during deferral")
        return _task_from_row(_load_task(conn, identity))


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


def requeue_not_admitted_task(
    db_path: Path | str,
    attempt: TaskAttempt,
    *,
    clock: Clock,
) -> dict[str, Any]:
    """Return a running task to its durable queue after proven non-admission."""
    now = _timestamp(clock)
    lease = attempt.lease
    identity = attempt.identity
    if lease.room_id != identity.room_id:
        raise DriverValidationError("lease and task belong to different rooms")
    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        row = _load_task(conn, identity)
        if (
            row["status"] == "queued"
            and int(row["execution_generation"]) == attempt.execution_generation
            and int(row["cancel_generation"]) == attempt.cancel_generation
            and row["run_gateway_id"] is None
            and row["run_process_generation"] is None
            and row["run_lease_generation"] is None
        ):
            return _task_from_row(row, idempotent=True)
        if (
            row["status"] != "running"
            or int(row["execution_generation"]) != attempt.execution_generation
            or int(row["cancel_generation"]) != attempt.cancel_generation
            or row["run_gateway_id"] != lease.gateway_id
            or row["run_process_generation"] != lease.process_generation
            or int(row["run_lease_generation"] or 0) != lease.lease_generation
        ):
            raise StaleTaskError("not-admitted task attempt lost its fence")
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='queued', run_gateway_id=NULL,
                   run_process_generation=NULL, run_lease_generation=NULL,
                   started_at=NULL, updated_at=?
               WHERE room_id=? AND task_id=? AND status='running'
                 AND execution_generation=? AND cancel_generation=?
                 AND run_gateway_id=? AND run_process_generation=?
                 AND run_lease_generation=?""",
            (
                now,
                identity.room_id,
                identity.task_id,
                attempt.execution_generation,
                attempt.cancel_generation,
                lease.gateway_id,
                lease.process_generation,
                lease.lease_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("not-admitted task changed during requeue")
        return _task_from_row(_load_task(conn, identity))


def cancel_task(
    db_path: Path | str,
    identity: TaskIdentity,
    *,
    cancel_id: Any,
    expected_cancel_generation: int,
    clock: Clock,
) -> dict[str, Any]:
    """Cancel a queued task before any external work was admitted."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    if (
        not isinstance(expected_cancel_generation, int)
        or expected_cancel_generation < 0
    ):
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        row = _load_task(conn, identity)
        if row["status"] == "cancelled" and row["cancel_id"] == cancel_id:
            return _task_from_row(row, idempotent=True)
        if row["status"] in TERMINAL_STATUSES:
            raise InvalidTaskTransitionError(
                f"cannot cancel task in state '{row['status']}'"
            )
        if row["status"] not in {"queued", "deferred"}:
            raise InvalidTaskTransitionError(
                "running work requires acknowledged two-phase cancellation"
            )
        if int(row["cancel_generation"]) != expected_cancel_generation:
            raise StaleTaskError("task cancellation generation changed")

        next_generation = expected_cancel_generation + 1
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='cancelled', cancel_generation=?, cancel_id=?,
                   terminal_at=?, updated_at=?
               WHERE room_id=? AND task_id=?
                 AND status IN ('queued', 'deferred')
                 AND cancel_generation=?""",
            (
                next_generation,
                cancel_id,
                now,
                now,
                identity.room_id,
                identity.task_id,
                expected_cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("task changed during cancellation")
        return _task_from_row(_load_task(conn, identity))


def begin_task_cancel(
    db_path: Path | str,
    identity: TaskIdentity,
    *,
    cancel_id: Any,
    expected_cancel_generation: int,
    clock: Clock,
) -> dict[str, Any]:
    """Persist a stop intent without claiming the remote run has stopped."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    if (
        not isinstance(expected_cancel_generation, int)
        or expected_cancel_generation < 0
    ):
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        row = _load_task(conn, identity)
        if row["status"] == "stopping" and row["cancel_id"] == cancel_id:
            return _task_from_row(row, idempotent=True)
        if row["status"] in TERMINAL_STATUSES or row["status"] == "queued":
            raise InvalidTaskTransitionError(
                f"cannot request remote stop in state '{row['status']}'"
            )
        if int(row["cancel_generation"]) != expected_cancel_generation:
            raise StaleTaskError("task cancellation generation changed")
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='stopping', cancel_generation=?, cancel_id=?,
                   updated_at=?
               WHERE room_id=? AND task_id=?
                 AND status IN ('running', 'indeterminate', 'deferred')
                 AND cancel_generation=?""",
            (
                expected_cancel_generation + 1,
                cancel_id,
                now,
                identity.room_id,
                identity.task_id,
                expected_cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("task changed during stop request")
        return _task_from_row(_load_task(conn, identity))


def complete_task_cancel(
    db_path: Path | str,
    identity: TaskIdentity,
    *,
    cancel_id: Any,
    expected_cancel_generation: int,
    clock: Clock,
) -> dict[str, Any]:
    """Commit cancellation only after the transport acknowledges exact Stop."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        row = _load_task(conn, identity)
        if row["status"] == "cancelled" and row["cancel_id"] == cancel_id:
            return _task_from_row(row, idempotent=True)
        if (
            row["status"] != "stopping"
            or row["cancel_id"] != cancel_id
            or int(row["cancel_generation"]) != expected_cancel_generation
        ):
            raise StaleTaskError("task stop acknowledgement is stale")
        updated = conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='cancelled', terminal_at=?, updated_at=?
               WHERE room_id=? AND task_id=? AND status='stopping'
                 AND cancel_id=? AND cancel_generation=?""",
            (
                now,
                now,
                identity.room_id,
                identity.task_id,
                cancel_id,
                expected_cancel_generation,
            ),
        )
        if updated.rowcount != 1:
            raise StaleTaskError("task changed during stop acknowledgement")
        return _task_from_row(_load_task(conn, identity))


def recover_room(
    db_path: Path | str,
    lease: DriverLease,
    *,
    clock: Clock,
) -> dict[str, list[TaskIdentity]]:
    """Fence abandoned running attempts without requeueing uncertain work."""
    now = _timestamp(clock)
    with _transaction(db_path) as conn:
        _require_active_lease(conn, lease, now=now)
        stale_rows = conn.execute(
            """SELECT * FROM hosted_room_driver_tasks
               WHERE room_id=? AND status='running'
                 AND NOT (
                     run_gateway_id=? AND run_process_generation=?
                     AND run_lease_generation=?
                 )
               ORDER BY source_event_seq, created_at, task_id""",
            (
                lease.room_id,
                lease.gateway_id,
                lease.process_generation,
                lease.lease_generation,
            ),
        ).fetchall()
        if stale_rows:
            conn.execute(
                """UPDATE hosted_room_driver_tasks
                   SET status='indeterminate', indeterminate_at=?, updated_at=?
                   WHERE room_id=? AND status='running'
                     AND NOT (
                         run_gateway_id=? AND run_process_generation=?
                         AND run_lease_generation=?
                     )""",
                (
                    now,
                    now,
                    lease.room_id,
                    lease.gateway_id,
                    lease.process_generation,
                    lease.lease_generation,
                ),
            )
        queued_rows = conn.execute(
            """SELECT * FROM hosted_room_driver_tasks
               WHERE room_id=? AND status='queued'
               ORDER BY source_event_seq, created_at, task_id""",
            (lease.room_id,),
        ).fetchall()
        indeterminate_rows = conn.execute(
            """SELECT * FROM hosted_room_driver_tasks
               WHERE room_id=? AND status='indeterminate'
               ORDER BY source_event_seq, created_at, task_id""",
            (lease.room_id,),
        ).fetchall()
        return {
            "queued": [_task_identity_from_row(row) for row in queued_rows],
            "indeterminate": [
                _task_identity_from_row(row) for row in indeterminate_rows
            ],
        }


def get_task(
    db_path: Path | str,
    identity: TaskIdentity,
) -> dict[str, Any]:
    """Read one task without mutating its state."""
    conn = _connect(db_path)
    try:
        return _task_from_row(_load_task(conn, identity))
    finally:
        conn.close()


def list_tasks(
    db_path: Path | str,
    *,
    room_id: Any,
    status: TaskStatus | None = None,
) -> list[dict[str, Any]]:
    """Return room tasks in deterministic admission order."""
    room_id = _identifier(room_id, label="room_id")
    if status is not None and status not in TASK_STATUSES:
        raise DriverValidationError("invalid task status")
    conn = _connect(db_path)
    try:
        if status is None:
            rows = conn.execute(
                """SELECT * FROM hosted_room_driver_tasks
                   WHERE room_id=?
                   ORDER BY source_event_seq, created_at, task_id""",
                (room_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM hosted_room_driver_tasks
                   WHERE room_id=? AND status=?
                   ORDER BY source_event_seq, created_at, task_id""",
                (room_id, status),
            ).fetchall()
        return [_task_from_row(row) for row in rows]
    finally:
        conn.close()


def prune_published_terminal_tasks(
    db_path: Path | str,
    *,
    room_id: Any,
    clock: Clock,
    retention_seconds: float = TERMINAL_TASK_RETENTION_SECONDS,
    retain: int = MAX_RETAINED_TERMINAL_TASKS,
) -> int:
    """Bound execution rows after outcomes are durable in the room log."""

    room_id = _identifier(room_id, label="room_id")
    now = _timestamp(clock)
    if retention_seconds <= 0:
        raise DriverValidationError("retention_seconds must be positive")
    if isinstance(retain, bool) or not isinstance(retain, int) or retain < 0:
        raise DriverValidationError("retain must be a non-negative integer")

    with _transaction(db_path) as conn:
        publications = conn.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table' AND name='hosted_room_policy_publications'"""
        ).fetchone()
        if publications is None:
            return 0
        rows = conn.execute(
            """SELECT t.task_id, t.terminal_at
                 FROM hosted_room_driver_tasks t
                WHERE t.room_id=?
                  AND t.status IN ('settled', 'failed', 'cancelled')
                  AND EXISTS (
                      SELECT 1 FROM hosted_room_policy_publications p
                       WHERE p.room_id=t.room_id AND p.task_id=t.task_id
                         AND p.kind IN (
                             'turn.settled', 'turn.failed', 'turn.cancelled'
                         )
                  )
                ORDER BY t.terminal_at DESC, t.task_id ASC""",
            (room_id,),
        ).fetchall()
        cutoff = now - float(retention_seconds)
        candidates = [
            str(row["task_id"])
            for index, row in enumerate(rows)
            if index >= retain
            or (row["terminal_at"] is not None and float(row["terminal_at"]) <= cutoff)
        ][:MAX_TASK_PRUNE_BATCH]
        if not candidates:
            return 0
        placeholders = ",".join("?" for _ in candidates)
        deleted = conn.execute(
            f"""DELETE FROM hosted_room_driver_tasks
                WHERE room_id=? AND task_id IN ({placeholders})""",
            (room_id, *candidates),
        )
        return max(0, int(deleted.rowcount))
