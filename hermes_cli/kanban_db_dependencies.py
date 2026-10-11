"""Canonical successful-parent policy, independent of terminal/cleanup statuses."""
from __future__ import annotations

import sqlite3
from typing import Optional


DEPENDENCY_SUCCESS_STATUS = "done"


def parent_succeeded(status: Optional[str]) -> bool:
    """Only successful completion releases a dependency; archive is not success."""
    return status == DEPENDENCY_SUCCESS_STATUS


def unsatisfied_parents(conn: sqlite3.Connection, task_id: str) -> list[tuple[str, Optional[str]]]:
    """Direct parents that have not succeeded, ordered for stable diagnostics."""
    rows = conn.execute(
        "SELECT l.parent_id, p.status FROM task_links l "
        "LEFT JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? AND (p.status IS NULL OR p.status != ?) "
        "ORDER BY l.parent_id", (task_id, DEPENDENCY_SUCCESS_STATUS),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def parents_satisfied(conn: sqlite3.Connection, task_id: str) -> bool:
    """No parents is satisfied; missing parents fail closed on damaged boards."""
    return not unsatisfied_parents(conn, task_id)


def landing_status(
    conn: sqlite3.Connection, task_id: str, resume_status: str = "ready",
) -> str:
    """Restore the intended phase only when parents succeeded (caller holds txn)."""
    return resume_status if parents_satisfied(conn, task_id) else "todo"
