"""SQLite operations for read-only gateway-session mirrors on a Kanban board.

Mirror rows deliberately live outside ``tasks``: dispatcher and task lifecycle
queries therefore cannot treat an observed conversation as executable work.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional

from hermes_cli import kanban_db as kb


MIRROR_STATES = frozenset({"received", "running", "completed", "failed", "cancelled"})
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_MAX_MIRROR_LIMIT = 500


def _row_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


def _validate_retention_days(retention_days: int) -> int:
    if isinstance(retention_days, bool) or not isinstance(retention_days, int):
        raise ValueError("session_mirror.retention_days must be an integer")
    if retention_days < 0 or retention_days > 3650:
        raise ValueError("session_mirror.retention_days must be between 0 and 3650")
    return retention_days


def create_or_get_mirror(
    conn: sqlite3.Connection, *, profile: str, platform: str, chat_id: str,
    thread_id: Optional[str], session_id: str, message_id: str, now: Optional[int] = None,
) -> tuple[int, bool]:
    """Create one metadata-only mirror per stable inbound message identity."""
    identity = {
        "profile": profile,
        "platform": platform,
        "chat_id": chat_id,
        "session_id": session_id,
        "message_id": message_id,
    }
    missing = [name for name, value in identity.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError(f"session mirror identity is missing {', '.join(missing)}")
    timestamp = int(time.time()) if now is None else int(now)
    title = f"{platform} session"
    with kb.write_txn(conn, allow_nested=True):
        inserted = conn.execute(
            """INSERT OR IGNORE INTO session_mirrors (
                   profile, platform, chat_id, thread_id, session_id, message_id,
                   title, status, received_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, 'received', ?, ?)""",
            (
                profile.strip(), platform.strip(), chat_id.strip(),
                str(thread_id or "").strip(), session_id.strip(), message_id.strip(),
                title, timestamp, timestamp,
            ),
        ).rowcount == 1
        row = conn.execute(
            """SELECT id FROM session_mirrors
               WHERE profile = ? AND platform = ? AND chat_id = ?
                 AND thread_id = ? AND message_id = ?""",
            (profile.strip(), platform.strip(), chat_id.strip(), str(thread_id or "").strip(), message_id.strip()),
        ).fetchone()
        if row is None:
            raise RuntimeError("session mirror insert did not produce a row")
        return int(row["id"]), inserted


def get_mirror(conn: sqlite3.Connection, mirror_id: int) -> Optional[dict[str, Any]]:
    return _row_dict(conn.execute("SELECT * FROM session_mirrors WHERE id = ?", (mirror_id,)).fetchone())


def mark_mirror_running(conn: sqlite3.Connection, mirror_id: int, *, now: Optional[int] = None) -> bool:
    timestamp = int(time.time()) if now is None else int(now)
    with kb.write_txn(conn, allow_nested=True):
        cursor = conn.execute(
            """UPDATE session_mirrors
               SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ?
               WHERE id = ? AND status = 'received'""",
            (timestamp, timestamp, mirror_id),
        )
        return cursor.rowcount == 1


def finish_mirror(
    conn: sqlite3.Connection, mirror_id: int, status: str, *, now: Optional[int] = None,
) -> bool:
    if status not in _TERMINAL_STATES:
        raise ValueError(f"mirror terminal status must be one of {sorted(_TERMINAL_STATES)}")
    timestamp = int(time.time()) if now is None else int(now)
    with kb.write_txn(conn, allow_nested=True):
        cursor = conn.execute(
            """UPDATE session_mirrors
               SET status = ?, completed_at = ?, updated_at = ?
               WHERE id = ? AND status IN ('received', 'running')""",
            (status, timestamp, timestamp, mirror_id),
        )
        return cursor.rowcount == 1


def list_mirrors(
    conn: sqlite3.Connection, *, include_archived: bool = False, limit: int = 100,
) -> list[dict[str, Any]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_MIRROR_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_MIRROR_LIMIT}")
    where = "" if include_archived else "WHERE archived_at IS NULL"
    rows = conn.execute(
        f"SELECT * FROM session_mirrors {where} ORDER BY received_at DESC, id DESC LIMIT ?", (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def archive_mirror(conn: sqlite3.Connection, mirror_id: int, *, now: Optional[int] = None) -> bool:
    timestamp = int(time.time()) if now is None else int(now)
    with kb.write_txn(conn, allow_nested=True):
        cursor = conn.execute(
            """UPDATE session_mirrors SET archived_at = ?, updated_at = ?
               WHERE id = ? AND archived_at IS NULL""",
            (timestamp, timestamp, mirror_id),
        )
        return cursor.rowcount == 1


def delete_mirror(conn: sqlite3.Connection, mirror_id: int) -> bool:
    with kb.write_txn(conn, allow_nested=True):
        cursor = conn.execute("DELETE FROM session_mirrors WHERE id = ?", (mirror_id,))
        return cursor.rowcount == 1


def prune_expired_mirrors(
    conn: sqlite3.Connection, *, retention_days: int, now: Optional[int] = None,
    profile: Optional[str] = None,
) -> int:
    """Delete old terminal mirrors within one profile; zero disables expiry and active turns remain."""
    days = _validate_retention_days(retention_days)
    if days == 0:
        return 0
    timestamp = int(time.time()) if now is None else int(now)
    cutoff = timestamp - days * 86_400
    profile_filter = " AND profile = ?" if profile is not None else ""
    params: tuple[Any, ...] = (cutoff, profile.strip()) if profile is not None else (cutoff,)
    with kb.write_txn(conn, allow_nested=True):
        cursor = conn.execute(
            """DELETE FROM session_mirrors
               WHERE status IN ('completed', 'failed', 'cancelled') AND updated_at < ?"""
            + profile_filter,
            params,
        )
        return cursor.rowcount


def promote_mirror(
    conn: sqlite3.Connection, mirror_id: int, *, title: str, body: Optional[str] = None,
    created_by: Optional[str] = None, board: Optional[str] = None,
) -> str:
    """Create a user-authored, parked task once; never copy transcript text or dispatch it."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("task title is required to promote a session mirror")
    if body is not None and not isinstance(body, str):
        raise ValueError("task body must be text")
    with kb.write_txn(conn, allow_nested=True):
        row = conn.execute("SELECT * FROM session_mirrors WHERE id = ?", (mirror_id,)).fetchone()
        if row is None:
            raise KeyError(f"session mirror {mirror_id} not found")
        if row["promoted_task_id"]:
            return str(row["promoted_task_id"])
        task_id = kb.create_task(
            conn,
            title=title.strip(),
            body=(body or "").strip() or None,
            created_by=created_by,
            initial_status="blocked",
            session_id=row["session_id"],
            idempotency_key=f"session-mirror:{mirror_id}",
            board=board or kb.get_current_board(),
        )
        conn.execute(
            "UPDATE session_mirrors SET promoted_task_id = ?, updated_at = ? WHERE id = ?",
            (task_id, int(time.time()), mirror_id),
        )
        return task_id
