"""Task graph initialization and atomic decomposition persistence."""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional

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
    """DB-free shape check + Kahn's cycle check on the sibling graph (a cycle
    would deadlock every involved child in ``todo`` forever)."""
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(f"child[{idx}].parents[{p}] is not a valid index into children")
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")

    in_deg = [0] * len(children)
    adj: list[list[int]] = [[] for _ in children]
    for i, c in enumerate(children):
        for p in (c.get("parents") or []):
            adj[p].append(i)
            in_deg[i] += 1
    queue = [i for i in range(len(children)) if in_deg[i] == 0]
    seen = 0
    while queue:
        seen += 1
        for nb in adj[queue.pop()]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    if seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")


def decompose_triage_task(
    conn: sqlite3.Connection, task_id: str, *, root_assignee: Optional[str], children: list[dict],
    author: Optional[str] = None, auto_promote: bool = True,
) -> Optional[list[str]]:
    """Fan a triage task out into children and move the root to ``todo``; the root
    waits on every child and wakes (``ready``) when all are done.

    Pure plan producer: validates triage preconditions, constructs a mutation plan,
    and delegates to commit_graph_mutation under strict governance constraints.
    """
    from hermes_cli.kanban_db import (
        _canonical_assignee,
        commit_graph_mutation,
        recompute_ready,
    )

    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)

    # Check root preconditions
    root_row = conn.execute(
        "SELECT id, status, tenant, workspace_kind, workspace_path "
        "FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if root_row is None or root_row["status"] != "triage":
        return None
    if conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'decomposed' LIMIT 1",
        (task_id,),
    ).fetchone():
        return None

    root_ws_kind = root_row["workspace_kind"] or "scratch"
    tasks_to_create = []
    for idx, child in enumerate(children):
        child_ws_kind = child.get("workspace_kind") or root_ws_kind
        if child.get("workspace_path"):
            child_ws_path = child.get("workspace_path")
        elif child_ws_kind == "worktree":
            child_ws_path = None
        elif child_ws_kind == root_ws_kind:
            child_ws_path = root_row["workspace_path"]
        else:
            child_ws_path = None
        tasks_to_create.append({
            "temp_id": f"child_{idx}",
            "title": child["title"].strip(),
            "body": child.get("body") if isinstance(child.get("body"), str) else None,
            "assignee": _canonical_assignee(child.get("assignee")),
            "workspace_kind": child_ws_kind,
            "workspace_path": child_ws_path,
            "tenant": root_row["tenant"],
            "status": "todo",
            "created_by": author or "decomposer",
            "creator_task_id": task_id,
        })

    edges_to_create = []
    for idx, child in enumerate(children):
        for p_idx in (child.get("parents") or []):
            edges_to_create.append({"from": f"child_{p_idx}", "to": f"child_{idx}"})
    for idx in range(len(children)):
        edges_to_create.append({"from": f"child_{idx}", "to": task_id})

    state_changes = [{
        "task_id": task_id,
        "status": "todo",
        "assignee": root_assignee,
    }]

    comments = []
    if author and author.strip():
        comments.append({
            "task_id": task_id,
            "author": author.strip(),
            "body": "Decomposed into {child_ids}. Root will wake when all children complete.",
        })

    events = [{
        "task_id": task_id,
        "kind": "decomposed",
        "payload": {"root_assignee": root_assignee},
    }]

    plan = {
        "reason": "decompose_triage_task",
        "create_tasks": tasks_to_create,
        "create_edges": edges_to_create,
        "state_changes": state_changes,
        "comments": comments,
        "events": events,
    }
    gov_ctx = {
        "actor": author or "decomposer",
        "ingress": "graph_api",
        "details": {"root_task_id": task_id, "children_count": len(children)},
    }

    res = commit_graph_mutation(conn, board_id=None, mutation_plan=plan, governance_context=gov_ctx)
    if auto_promote:
        recompute_ready(conn)
    return res.created_task_ids


def _insert_decomposed_child(
    conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, child: dict,
    author: Optional[str], now: int,
) -> str:
    """Insert one decomposed child as ``todo`` (linked under the root later so
    the dispatcher only ever sees a coherent graph); returns its id.

    Workspace: per-child override wins, else inherit the root's kind. Path
    inherits only when kinds match (a 'dir' child must not point at the
    root's worktree) and NEVER for worktrees — siblings dispatch concurrently
    and one shared checkout would put them all on the first sibling's branch
    with no lock; leaving it unset makes dispatch materialize a fresh
    ``<repo>/.worktrees/<child-id>`` per child from the board anchor.
    """
    from hermes_cli.kanban_db import (
        _new_task_id, _canonical_assignee, _append_event,
    )

    root_ws_kind = root_row["workspace_kind"] or "scratch"
    child_ws_kind = child.get("workspace_kind") or root_ws_kind
    if child.get("workspace_path"):
        child_ws_path = child.get("workspace_path")
    elif child_ws_kind == "worktree":
        child_ws_path = None
    elif child_ws_kind == root_ws_kind:
        child_ws_path = root_row["workspace_path"]
    else:
        child_ws_path = None
    new_id = _new_task_id()
    body = child.get("body")
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, created_at, created_by) "
        "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?)",
        (
            new_id, child["title"].strip(), body if isinstance(body, str) else None,
            _canonical_assignee(child.get("assignee")), child_ws_kind, child_ws_path,
            root_row["tenant"], now, (author or "decomposer"),
        ),
    )
    _append_event(
        conn, new_id, "created", {"by": author or "decomposer", "from_decompose_of": root_id},
    )
    inherit_creator_origin(conn, new_id, root_id, created_at=now)
    return new_id
