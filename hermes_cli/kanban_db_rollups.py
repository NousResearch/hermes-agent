"""Per-board rollup queries split out of :mod:`hermes_cli.kanban_db`.

``kanban_db`` is a facade past the repo's 2,000-line cap and may only shrink, so
new behaviour never lands there: read-only aggregate helpers that need nothing
but a connection live here and are re-exported at the tail of ``kanban_db`` (the
same pattern as ``kanban_db_boards`` / ``kanban_db_dispatch``).

Nothing in this module imports ``kanban_db``.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable


def latest_summaries(conn: sqlite3.Connection, task_ids: Iterable[str]) -> dict[str, str]:
    """``{task_id: newest non-empty run summary}`` in one query (window function,
    SQLite >= 3.25); tasks without a summary are omitted."""
    ids = list(task_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT task_id, summary FROM (
            SELECT task_id, summary,
                   ROW_NUMBER() OVER (
                       PARTITION BY task_id
                       ORDER BY COALESCE(ended_at, started_at) DESC, id DESC
                   ) AS rn
              FROM task_runs
             WHERE task_id IN ({placeholders})
               AND summary IS NOT NULL AND summary != ''
        ) WHERE rn = 1
        """,
        ids,
    ).fetchall()
    return {r["task_id"]: r["summary"] for r in rows}


def current_run_started_ats(conn: sqlite3.Connection, task_ids: Iterable[str]) -> dict[str, int]:
    """``{task_id: started_at of the run ``tasks.current_run_id`` points at}``
    in one query; tasks with no active run (NULL or dangling pointer) are omitted."""
    ids = list(task_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT t.id AS task_id, r.started_at AS started_at FROM tasks t "
        "JOIN task_runs r ON r.id = t.current_run_id "
        f"WHERE t.id IN ({placeholders})",
        ids,
    ).fetchall()
    return {r["task_id"]: r["started_at"] for r in rows}
