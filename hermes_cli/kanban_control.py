"""Durable per-board Kanban dispatch control state."""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


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


def _control_schema_state(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' "
        "AND name IN ('kanban_dispatch_control', 'kanban_dispatch_control_events')",
    ).fetchall()
    schemas = {
        (row["name"] if isinstance(row, sqlite3.Row) else row[0]):
        (row["sql"] if isinstance(row, sqlite3.Row) else row[1])
        for row in rows
    }
    if not schemas:
        return "missing"
    if set(schemas) != {"kanban_dispatch_control", "kanban_dispatch_control_events"}:
        return "invalid"

    control_info = conn.execute("PRAGMA table_info(kanban_dispatch_control)").fetchall()
    observed_control = [
        (
            item["name"] if isinstance(item, sqlite3.Row) else item[1],
            (item["type"] if isinstance(item, sqlite3.Row) else item[2]).upper(),
            int(item["notnull"] if isinstance(item, sqlite3.Row) else item[3]),
            int(item["pk"] if isinstance(item, sqlite3.Row) else item[5]),
        )
        for item in control_info
    ]
    expected_control = [
        ("id", "INTEGER", 0, 1),
        ("state", "TEXT", 1, 0),
        ("revision", "INTEGER", 1, 0),
        ("changed_at", "INTEGER", 1, 0),
        ("changed_by", "TEXT", 1, 0),
        ("reason", "TEXT", 0, 0),
    ]
    event_info = conn.execute("PRAGMA table_info(kanban_dispatch_control_events)").fetchall()
    observed_events = [
        (
            item["name"] if isinstance(item, sqlite3.Row) else item[1],
            (item["type"] if isinstance(item, sqlite3.Row) else item[2]).upper(),
            int(item["notnull"] if isinstance(item, sqlite3.Row) else item[3]),
            int(item["pk"] if isinstance(item, sqlite3.Row) else item[5]),
        )
        for item in event_info
    ]
    expected_events = [
        ("revision", "INTEGER", 0, 1),
        ("old_state", "TEXT", 1, 0),
        ("new_state", "TEXT", 1, 0),
        ("changed_by", "TEXT", 1, 0),
        ("reason", "TEXT", 0, 0),
        ("changed_at", "INTEGER", 1, 0),
    ]
    control_sql = re.sub(
        r"\s+", " ", str(schemas["kanban_dispatch_control"] or "").lower(),
    )
    events_sql = re.sub(
        r"\s+", " ", str(schemas["kanban_dispatch_control_events"] or "").lower(),
    )
    control_constraints = (
        "check (id = 1)",
        "check (state in ('running', 'paused'))",
        "check (revision >= 1)",
    )
    event_constraints = (
        "check (old_state in ('running', 'paused'))",
        "check (new_state in ('running', 'paused'))",
    )
    valid = (
        observed_control == expected_control
        and observed_events == expected_events
        and all(fragment in control_sql for fragment in control_constraints)
        and all(fragment in events_sql for fragment in event_constraints)
    )
    return "valid" if valid else "invalid"


def read_control(conn: sqlite3.Connection) -> ControlState:
    """Read authoritative board control without initializing or repairing it."""
    try:
        schema_state = _control_schema_state(conn)
        if schema_state == "missing":
            return ControlState(state="running", revision=0)
        if schema_state != "valid":
            return _unknown("control_schema_invalid")
        rows = conn.execute(
            "SELECT id, state, revision, changed_at, changed_by, reason, "
            "typeof(id) AS id_type, typeof(state) AS state_type, "
            "typeof(revision) AS revision_type, typeof(changed_at) AS changed_at_type, "
            "typeof(changed_by) AS changed_by_type, typeof(reason) AS reason_type "
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
        values = tuple(row) if not isinstance(row, sqlite3.Row) else None
        row_id = row["id"] if values is None else values[0]
        state = row["state"] if values is None else values[1]
        revision = row["revision"] if values is None else values[2]
        changed_at = row["changed_at"] if values is None else values[3]
        changed_by = row["changed_by"] if values is None else values[4]
        reason = row["reason"] if values is None else values[5]
        storage = (
            row["id_type"], row["state_type"], row["revision_type"],
            row["changed_at_type"], row["changed_by_type"], row["reason_type"],
        ) if values is None else values[6:12]
    except (IndexError, KeyError, TypeError):
        return _unknown("invalid_control_state")
    if (
        storage[:5] != ("integer", "text", "integer", "integer", "text")
        or storage[5] not in {"null", "text"}
        or type(row_id) is not int
        or type(revision) is not int
        or type(changed_at) is not int
        or row_id != 1
        or state not in _VALID_STATES
        or revision < 1
        or changed_at < 1
        or not _valid_text(changed_by, maximum=_MAX_ACTOR_LENGTH, allow_none=False)
        or not _valid_text(reason, maximum=_MAX_REASON_LENGTH, allow_none=True)
    ):
        return _unknown("invalid_control_state")
    try:
        datetime.fromtimestamp(changed_at, tz=UTC)
    except (OSError, OverflowError, ValueError):
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


def _control_row(board: str, state: ControlState) -> dict:
    row = {
        "board": board,
        "state": state.state,
        "revision": state.revision,
        "changed_at": state.changed_at,
        "changed_at_utc": (
            datetime.fromtimestamp(state.changed_at, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z")
            if state.changed_at is not None
            else None
        ),
        "changed_by": state.changed_by,
    }
    if state.reason is not None:
        row["reason"] = state.reason
    if state.error_code is not None:
        row["error_code"] = state.error_code
    return row


def selected_boards(board: Optional[str] = None) -> list[str]:
    """Deterministic active board scope for operator and backend controls."""
    from hermes_cli import kanban_db as kb

    active = [item["slug"] for item in kb.list_boards(include_archived=False)]
    if board is None:
        return active
    normalized = kb._normalize_board_slug(board)
    if normalized is None or normalized not in active:
        raise ValueError(f"kanban board {normalized or board!r} does not exist or is archived")
    return [normalized]


def _aggregate(rows: list[dict], *, desired_state: Optional[str] = None) -> str:
    errors = [row for row in rows if row["state"] == "unknown"]
    known = [row for row in rows if row["state"] != "unknown"]
    if desired_state is not None:
        reached = [row for row in known if row["state"] == desired_state]
        if len(reached) == len(rows):
            return desired_state
        return "partial" if reached else "unknown"
    if errors:
        return "unknown"
    states = {row["state"] for row in known}
    return states.pop() if len(states) == 1 else "partial"


def control_status(*, board: Optional[str] = None) -> dict:
    """Read one or all active boards without mutating control state."""
    from hermes_cli import kanban_db as kb

    rows = []
    for slug in selected_boards(board):
        try:
            path = kb.kanban_db_path(board=slug)
            if not path.exists():
                state = ControlState(state="running", revision=0)
            else:
                wal = path.with_name(path.name + "-wal")
                shm = path.with_name(path.name + "-shm")
                if not wal.exists() and not shm.exists():
                    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
                    conn = sqlite3.connect(uri, uri=True)
                    conn.row_factory = sqlite3.Row
                    try:
                        state = read_control(conn)
                    finally:
                        conn.close()
                else:
                    with tempfile.TemporaryDirectory(prefix="hermes-kanban-status-") as tmp:
                        snapshot = Path(tmp) / path.name
                        shutil.copyfile(path, snapshot)
                        if wal.exists():
                            shutil.copyfile(
                                wal, snapshot.with_name(snapshot.name + "-wal"),
                            )
                        conn = sqlite3.connect(snapshot)
                        conn.row_factory = sqlite3.Row
                        try:
                            state = read_control(conn)
                        finally:
                            conn.close()
        except Exception as exc:
            logger.exception("failed to read Kanban pickup control for board %s", slug)
            state = ControlState(
                state="unknown", revision=0, changed_at=None, changed_by=None,
                reason=None, error_code=f"board_unreadable:{type(exc).__name__}",
            )
        rows.append(_control_row(slug, state))
    return {
        "state": _aggregate(rows),
        "observed_at": int(time.time()),
        "boards": rows,
    }


def set_control_scope(
    desired_state: str, *, changed_by: str, reason: Optional[str] = None,
    board: Optional[str] = None,
) -> dict:
    """Convergently set one or all active boards and return committed readback."""
    from hermes_cli import kanban_db_connect as kbc

    if desired_state not in _VALID_STATES:
        raise ValueError(f"invalid desired state: {desired_state!r}")
    actor = _normalize_actor(changed_by)
    normalized_reason = _normalize_reason(reason)
    transition = pause_board if desired_state == "paused" else resume_board
    rows = []
    for slug in selected_boards(board):
        try:
            with kbc.connect_closing(board=slug) as conn:
                state = transition(
                    conn, changed_by=actor, reason=normalized_reason,
                )
        except ControlStateUnknown as exc:
            state = ControlState(
                state="unknown", revision=0, changed_at=None, changed_by=None,
                reason=None, error_code=str(exc) or "control_state_unknown",
            )
        except Exception as exc:
            logger.exception(
                "failed to set Kanban pickup control to %s for board %s",
                desired_state, slug,
            )
            state = ControlState(
                state="unknown", revision=0, changed_at=None, changed_by=None,
                reason=None, error_code=f"board_write_failed:{type(exc).__name__}",
            )
        rows.append(_control_row(slug, state))
    return {
        "state": _aggregate(rows, desired_state=desired_state),
        "requested_state": desired_state,
        "observed_at": int(time.time()),
        "boards": rows,
    }
