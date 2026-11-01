"""Kanban board metrics and timestamp/retention helpers."""

from __future__ import annotations

import sqlite3
import time
from typing import Optional


def board_stats(conn: sqlite3.Connection) -> dict:
    """Per-status + per-assignee counts and the oldest ``ready`` age (staleness signal)."""
    by_status: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' GROUP BY status"
    ):
        by_status[row["status"]] = int(row["n"])

    by_assignee = _counts_by_assignee(conn)

    oldest_row = conn.execute(
        "SELECT MIN(created_at) AS ts FROM tasks WHERE status = 'ready'"
    ).fetchone()
    now = int(time.time())
    oldest_ready_age = (
        (now - int(oldest_row["ts"]))
        if oldest_row and oldest_row["ts"] is not None else None
    )

    return {
        "by_status": by_status,
        "by_assignee": by_assignee,
        "oldest_ready_age_seconds": oldest_ready_age,
        "now": now,
    }


def _counts_by_assignee(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """``{assignee: {status: n}}`` over non-archived tasks."""
    counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        counts.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])
    return counts


def _to_epoch(val) -> Optional[int]:
    """Epoch seconds from int/float/numeric string/ISO-8601; None for empty/invalid."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    # ISO-8601 fallback (e.g. '2026-05-10T15:00:00Z')
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp())
    except (ValueError, OSError):
        return None


def task_age(task: object) -> dict:
    """Return age metrics for a single task. All values are seconds or None."""
    now = int(time.time())
    _c, _s, _co = task.created_at, task.started_at, task.completed_at  # from_row coerced
    return {
        "created_age_seconds": now - _c if _c is not None else None,
        "started_age_seconds": now - _s if _s is not None else None,
        "time_to_complete_seconds": _co - (_s or _c) if _co is not None else None,
    }


def _retention_seconds(older_than_seconds: int) -> int:
    """Normalise a gc retention window, rejecting negatives.

    Shared by both gc sweeps: a negative window puts the cutoff in the future,
    so "older than cutoff" would match every row / file instead of none —
    refuse before any sweep runs.
    """
    older_than_seconds = int(older_than_seconds)
    if older_than_seconds < 0:
        raise ValueError(
            f"older_than_seconds must be >= 0, got {older_than_seconds!r}: "
            "a negative retention selects everything."
        )
    return older_than_seconds
