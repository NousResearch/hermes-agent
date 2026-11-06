"""Board statistics — per-status counts, per-assignee counts, queue age.

Split out of ``hermes_cli.kanban_db`` (which imports this module at the tail
so the facade keeps its public names).

``ready_runnable`` is the point of the split: "ready" and "ready AND
dispatchable" are different numbers. A ``ready`` card with no assignee is
queued work the board cannot deliver — reporting only the first is what let
``ready=20, 0 runnable`` read as a healthy board.
"""

from __future__ import annotations

import sqlite3
import time


def board_stats(conn: sqlite3.Connection) -> dict:
    """Per-status + per-assignee counts and the oldest ``ready`` age (staleness signal).

    ``ready_runnable`` separates "ready" from "ready AND dispatchable": a
    ``ready`` card with no assignee (or one assigned to a profile this home
    cannot spawn) is queued work the board cannot deliver. Reporting the two
    apart is what stops ``ready=20, 0 runnable`` from reading as a healthy
    board.
    """
    by_status: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' GROUP BY status"
    ):
        by_status[row["status"]] = int(row["n"])

    by_assignee = counts_by_assignee(conn)

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
        "ready_runnable": ready_runnable_count(conn),
        "now": now,
    }


def ready_runnable_count(conn: sqlite3.Connection) -> int:
    """How many ``ready`` cards can actually be dispatched.

    "Runnable" = assigned AND unclaimed. A ``ready`` card with no assignee
    is queued work nobody can pick up, which is what the ``parked`` column
    exists to make visible.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks "
        "WHERE status = 'ready' AND claim_lock IS NULL "
        "  AND assignee IS NOT NULL AND assignee != ''",
    ).fetchone()
    return int(row["n"]) if row else 0


def counts_by_assignee(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """``{assignee: {status: n}}`` over non-archived tasks."""
    counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        counts.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])
    return counts
