"""Kanban task dependency links: parent/child edges, cycle guards, and
dependency promotion hooks.

Extracted verbatim from :mod:`hermes_cli.kanban_db` (godfile shard s2,
cluster c8). Public names are re-exported from :mod:`hermes_cli.kanban_db`
so existing ``import hermes_cli.kanban_db as kb`` callers are unaffected.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

# ---------------------------------------------------------------------------
# Verbatim section moved from hermes_cli/kanban_db.py (lines 3524-3631).
# The kanban_db.py -> sibling re-export cycle is import-safe because the
# parent-module imports at the bottom of this file are deferred to the end
# of the module (see kanban_db.py bottom re-export block).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def link_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    if parent_id == child_id:
        raise ValueError("a task cannot depend on itself")
    with write_txn(conn):
        missing = _find_missing_parents(conn, [parent_id, child_id])
        if missing:
            raise ValueError(f"unknown task(s): {', '.join(missing)}")
        if _would_cycle(conn, parent_id, child_id):
            raise ValueError(
                f"linking {parent_id} -> {child_id} would create a cycle"
            )
        conn.execute(
            "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )
        # If child was ready but parent is not yet done, demote child to todo.
        parent_status = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (parent_id,)
        ).fetchone()["status"]
        if parent_status != "done":
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'",
                (child_id,),
            )
        _append_event(
            conn, child_id, "linked",
            {"parent": parent_id, "child": child_id},
        )
        _inherit_notify_subs(conn, child_id, (parent_id,))


def _would_cycle(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    """Return True if adding parent->child creates a cycle.

    A cycle exists iff ``parent_id`` is already a descendant of
    ``child_id`` via existing parent->child links.  We walk downward
    from ``child_id`` and check whether we reach ``parent_id``.
    """
    seen = set()
    stack = [child_id]
    while stack:
        node = stack.pop()
        if node == parent_id:
            return True
        if node in seen:
            continue
        seen.add(node)
        rows = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (node,)
        ).fetchall()
        stack.extend(r["child_id"] for r in rows)
    return False


def unlink_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        )
        if cur.rowcount:
            _append_event(
                conn, child_id, "unlinked",
                {"parent": parent_id, "child": child_id},
            )
        removed = cur.rowcount > 0
    if removed:
        # Dependency edge removed — re-evaluate promotion eligibility for the
        # child immediately.  Matches the contract of complete_task and
        # unblock_task; without this the child stays stuck in todo until the
        # next dispatcher tick or a manual `hermes kanban recompute` (issue #22459).
        recompute_ready(conn)
    return removed


def parent_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    return [r["parent_id"] for r in rows]


def child_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
        (task_id,),
    ).fetchall()
    return [r["child_id"] for r in rows]


def parent_results(conn: sqlite3.Connection, task_id: str) -> list[tuple[str, Optional[str]]]:
    """Return ``(parent_id, result)`` for every done parent of ``task_id``."""
    rows = conn.execute(
        """
        SELECT t.id AS id, t.result AS result
        FROM tasks t
        JOIN task_links l ON l.parent_id = t.id
        WHERE l.child_id = ? AND t.status = 'done'
        ORDER BY t.completed_at ASC
        """,
        (task_id,),
    ).fetchall()
    return [(r["id"], r["result"]) for r in rows]

# Names from the parent module (helpers that stay in kanban_db.py because
# other clusters use them). Imported at the bottom of this module so the
# kanban_db <-> sibling re-export cycle resolves in any import order.
from hermes_cli.kanban_db import (  # noqa: E402
    write_txn,
    _append_event,
    _find_missing_parents,
    _inherit_notify_subs,
    recompute_ready,
)
