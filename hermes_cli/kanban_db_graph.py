"""Task graph initialization and atomic decomposition persistence."""
from __future__ import annotations

import sqlite3
from typing import Optional

def inherit_creator_origin(
    conn: sqlite3.Connection, task_id: str, creator_task_id: Optional[str], *,
    created_at: int,
) -> None:
    """Copy durable origin inside creation's transaction, never adding dependencies."""
    if not creator_task_id:
        return
    from hermes_cli.kanban_db import _inherit_notify_subs

    conn.execute(
        "UPDATE tasks SET session_id = COALESCE(session_id, "
        "(SELECT session_id FROM tasks WHERE id = ?)) WHERE id = ?",
        (creator_task_id, task_id),
    )
    _inherit_notify_subs(conn, task_id, (creator_task_id,), created_at=created_at)


def initial_task_state(
    conn: sqlite3.Connection, parents: tuple[str, ...], initial_status: str,
    triage: bool, tenant: Optional[str],
) -> tuple[str, Optional[str]]:
    """Resolve state and tenant under the creator's write transaction.

    Parent order breaks ties in this soft namespace; explicit tenant wins.
    Validate parents even for parked tasks so links never dangle.
    """
    rows = {}
    if parents:
        rows = {row["id"]: row for row in conn.execute(
            "SELECT id, status, tenant FROM tasks WHERE id IN "
            "(" + ",".join("?" * len(parents)) + ")", parents,
        )}
        missing = [pid for pid in parents if pid not in rows]
        if missing:
            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
        if tenant is None:
            tenant = next((rows[pid]["tenant"] for pid in parents if rows[pid]["tenant"]), None)
    if initial_status == "blocked":
        return "blocked", tenant
    if triage:
        return "triage", tenant
    if any(row["status"] != "done" for row in rows.values()):
        return "todo", tenant
    return "ready", tenant


def _validate_children_graph(children: list) -> None:
    """Delegates to the canonical implementation in ``kanban_db``.

    Kept here only so existing callers that import this name from
    ``kanban_db_graph`` keep working; the real logic lives beside
    ``decompose_triage_task`` to avoid two divergent copies (see history:
    this module used to own a full, independently-maintained duplicate that
    silently dropped ``project_id``/``branch_name``/``skills`` from every
    decomposed child — auto-decompose's actual code path, since
    ``kanban_decompose.py`` imports from here, not from ``kanban_db``).
    """
    from hermes_cli.kanban_db import _validate_children_graph as _impl
    return _impl(children)


def decompose_triage_task(
    conn: sqlite3.Connection, task_id: str, *, root_assignee: Optional[str], children: list[dict],
    author: Optional[str] = None, auto_promote: bool = True,
) -> Optional[list[str]]:
    """Delegates to the canonical implementation in ``kanban_db`` — see
    ``_validate_children_graph`` docstring above for why this module no
    longer owns its own copy of the fan-out logic."""
    from hermes_cli.kanban_db import decompose_triage_task as _impl
    return _impl(
        conn, task_id, root_assignee=root_assignee, children=children,
        author=author, auto_promote=auto_promote,
    )
