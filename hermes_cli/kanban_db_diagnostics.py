"""Database context used by graph-aware Kanban diagnostics."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


def task_graph_contexts(
    conn: sqlite3.Connection, task_ids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Load dependencies, capacity state, and assignee validity for tasks."""
    ordered_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
    contexts: dict[str, dict[str, Any]] = {
        task_id: {"parents": [], "children": []} for task_id in ordered_ids
    }
    if not ordered_ids:
        return contexts

    running_rows = conn.execute(
        "SELECT assignee, COUNT(*) AS count FROM tasks "
        "WHERE status = 'running' GROUP BY assignee"
    ).fetchall()
    running_by_assignee = {
        (row["assignee"] or ""): int(row["count"]) for row in running_rows
    }

    from hermes_cli.kanban_db_dispatch import count_running_tasks_other_boards

    current_db_path = next(
        (
            Path(row[2])
            for row in conn.execute("PRAGMA database_list").fetchall()
            if row[1] == "main" and row[2]
        ),
        None,
    )
    running_total = sum(
        running_by_assignee.values()
    ) + count_running_tasks_other_boards(current_db_path=current_db_path)

    placeholders = ",".join("?" for _ in ordered_ids)
    assignee_rows = conn.execute(
        f"SELECT id, assignee FROM tasks WHERE id IN ({placeholders})",
        tuple(ordered_ids),
    ).fetchall()

    from hermes_cli.profiles import profile_exists

    for row in assignee_rows:
        try:
            exists = profile_exists(row["assignee"] or "")
        except OSError:
            exists = False
        contexts[row["id"]]["assignee_profile_exists"] = exists
    for context in contexts.values():
        context["running_total"] = running_total
        context["running_by_assignee"] = running_by_assignee

    for bucket, own, other in (
        ("parents", "child_id", "parent_id"),
        ("children", "parent_id", "child_id"),
    ):
        rows = conn.execute(
            f"SELECT l.{own} AS owner_id, t.id, t.title, t.status "
            f"FROM task_links l JOIN tasks t ON t.id = l.{other} "
            f"WHERE l.{own} IN ({placeholders}) ORDER BY l.{own}, t.id",
            tuple(ordered_ids),
        ).fetchall()
        for row in rows:
            contexts[row["owner_id"]][bucket].append({
                "id": row["id"],
                "title": row["title"],
                "status": row["status"],
            })
    return contexts
