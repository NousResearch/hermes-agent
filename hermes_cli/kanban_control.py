"""Durable per-board Kanban dispatch control state."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Optional


CONTROL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kanban_dispatch_control (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    state      TEXT NOT NULL CHECK (state IN ('running', 'paused')),
    revision   INTEGER NOT NULL CHECK (revision >= 1),
    changed_at INTEGER NOT NULL,
    changed_by TEXT NOT NULL,
    reason     TEXT
);

CREATE TABLE IF NOT EXISTS kanban_dispatch_control_events (
    revision   INTEGER PRIMARY KEY,
    old_state  TEXT NOT NULL CHECK (old_state IN ('running', 'paused')),
    new_state  TEXT NOT NULL CHECK (new_state IN ('running', 'paused')),
    changed_by TEXT NOT NULL,
    reason     TEXT,
    changed_at INTEGER NOT NULL
);
"""

_CONTROL_COLUMNS = {"id", "state", "revision", "changed_at", "changed_by", "reason"}
_VALID_STATES = {"running", "paused"}
_MAX_ACTOR_LENGTH = 200
_MAX_REASON_LENGTH = 2_000


@dataclass(frozen=True)
class ControlState:
    state: str
    revision: int
    changed_at: Optional[int] = None
    changed_by: Optional[str] = None
    reason: Optional[str] = None
    error_code: Optional[str] = None


class ControlStateUnknown(RuntimeError):
    """Raised when a mutation cannot establish authoritative current state."""


def _unknown(error_code: str) -> ControlState:
    return ControlState(state="unknown", revision=0, error_code=error_code)


def _valid_text(value, *, maximum: int, allow_none: bool) -> bool:
    if value is None:
        return allow_none
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def read_control(conn: sqlite3.Connection) -> ControlState:
    """Read authoritative board control without initializing or repairing it."""
    try:
        columns = {
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in conn.execute("PRAGMA table_info(kanban_dispatch_control)")
        }
        if not _CONTROL_COLUMNS.issubset(columns):
            return _unknown("control_schema_invalid")
        rows = conn.execute(
            "SELECT id, state, revision, changed_at, changed_by, reason "
            "FROM kanban_dispatch_control"
        ).fetchall()
    except (sqlite3.DatabaseError, UnicodeDecodeError):
        return _unknown("control_read_failed")

    if not rows:
        return ControlState(state="running", revision=0)
    if len(rows) != 1:
        return _unknown("invalid_control_state")
    row = rows[0]
    try:
        row_id = int(row["id"])
        revision = int(row["revision"])
        changed_at = int(row["changed_at"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return _unknown("invalid_control_state")
    state = row["state"]
    changed_by = row["changed_by"]
    reason = row["reason"]
    if (
        row_id != 1
        or state not in _VALID_STATES
        or revision < 1
        or changed_at < 1
        or not _valid_text(changed_by, maximum=_MAX_ACTOR_LENGTH, allow_none=False)
        or not _valid_text(reason, maximum=_MAX_REASON_LENGTH, allow_none=True)
    ):
        return _unknown("invalid_control_state")
    return ControlState(
        state=state,
        revision=revision,
        changed_at=changed_at,
        changed_by=changed_by,
        reason=reason,
    )


def _normalize_actor(changed_by: str) -> str:
    actor = str(changed_by).strip()
    if not actor:
        raise ValueError("changed_by must be non-empty")
    if len(actor) > _MAX_ACTOR_LENGTH:
        raise ValueError(f"changed_by must be at most {_MAX_ACTOR_LENGTH} characters")
    return actor


def _normalize_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    normalized = str(reason).strip() or None
    if normalized is not None and len(normalized) > _MAX_REASON_LENGTH:
        raise ValueError(f"reason must be at most {_MAX_REASON_LENGTH} characters")
    return normalized


def _set_control(
    conn: sqlite3.Connection, desired_state: str, *, changed_by: str,
    reason: Optional[str],
) -> ControlState:
    from hermes_cli.kanban_db_connect import write_txn

    actor = _normalize_actor(changed_by)
    normalized_reason = _normalize_reason(reason)
    with write_txn(conn):
        current = read_control(conn)
        if current.state == "unknown":
            raise ControlStateUnknown(current.error_code or "control_state_unknown")
        if current.state == desired_state:
            return current
        revision = current.revision + 1
        changed_at = int(time.time())
        conn.execute(
            "INSERT INTO kanban_dispatch_control "
            "(id, state, revision, changed_at, changed_by, reason) "
            "VALUES (1, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "state=excluded.state, revision=excluded.revision, "
            "changed_at=excluded.changed_at, changed_by=excluded.changed_by, reason=excluded.reason",
            (desired_state, revision, changed_at, actor, normalized_reason),
        )
        conn.execute(
            "INSERT INTO kanban_dispatch_control_events "
            "(revision, old_state, new_state, changed_by, reason, changed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (revision, current.state, desired_state, actor, normalized_reason, changed_at),
        )
        return ControlState(
            state=desired_state,
            revision=revision,
            changed_at=changed_at,
            changed_by=actor,
            reason=normalized_reason,
        )


def pause_board(
    conn: sqlite3.Connection, *, changed_by: str, reason: Optional[str] = None,
) -> ControlState:
    return _set_control(conn, "paused", changed_by=changed_by, reason=reason)


def resume_board(
    conn: sqlite3.Connection, *, changed_by: str, reason: Optional[str] = None,
) -> ControlState:
    return _set_control(conn, "running", changed_by=changed_by, reason=reason)
