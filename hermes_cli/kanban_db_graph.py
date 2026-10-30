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

    ``children``: dicts of ``title`` (required), ``body``, ``assignee``,
    ``parents`` (indices into this list), optional workspace overrides.
    Returns child ids in input order, or None when the root is missing / not
    in triage, or has already decomposed. Atomic: malformed entries abort fan-out.
    """
    from hermes_cli.kanban_db import (
        _canonical_assignee, _link, _append_event, _insert_comment,
        write_txn, recompute_ready,
    )
    from hermes_cli.kanban_intake import is_pending, review as intake_review, record_inherited, record_review

    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)

    # ONE txn so the fan-out is atomic; helpers that open their own write_txn
    # (create_task, link_tasks, add_comment) must not be called in here.
    now = int(time.time())
    caller_owns_txn = bool(conn.in_transaction)
    with write_txn(conn, allow_nested=True):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path, intake_state "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if root_row is None or root_row["status"] != "triage":
            return None
        receipt = intake_review(conn, task_id)
        if is_pending(conn, task_id) and receipt is None:
            return None
        # Dependency links alone do not imply lineage. The completion event is
        # committed with the graph, and survives re-triage or unlinking.
        if conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'decomposed' LIMIT 1",
            (task_id,),
        ).fetchone():
            return None
        root_parent_ids = [row[0] for row in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id", (task_id,),
        )]
        child_ids = [
            _insert_decomposed_child(conn, task_id, root_row, child, author, now, receipt)
            for child in children
        ]
        # Every fan-out child must retain the root's prerequisites. Otherwise
        # planning could make newly-created work claimable before its inputs.
        for child_id in child_ids:
            for parent_id in root_parent_ids:
                _link(conn, parent_id, child_id)
        # Sibling edges within the decomposed graph.
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id, child_id = child_ids[p_idx], child_ids[idx]
                _link(conn, parent_id, child_id)
                _append_event(conn, child_id, "linked", {"parent": parent_id, "child": child_id})
        # Root waits for the whole graph: link it under EVERY child (simpler
        # than computing leaves; cycle-free since the root is only ever a child).
        for cid in child_ids:
            _link(conn, cid, task_id)
        # Flip the root triage -> todo, assignee -> orchestrator.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", tuple(params))
        if author and author.strip():
            _insert_comment(
                conn, task_id, author.strip(),
                "Decomposed into " + ", ".join(child_ids)
                + ". Root will wake when all children complete.",
                now,
            )
        _append_event(
            conn, task_id, "decomposed", {"child_ids": child_ids, "root_assignee": root_assignee},
        )
        if receipt is not None:
            record_review(conn, task_id)
            for child_id in child_ids:
                record_inherited(conn, child_id, *receipt)
    # Outside the txn (own IMMEDIATE txn). ``auto_promote=False`` leaves the
    # children in ``todo`` for manual-review-first workflows.
    if auto_promote and not caller_owns_txn:
        recompute_ready(conn)
    return child_ids


def specify_triage_task(
    conn: sqlite3.Connection, task_id: str, *, title: Optional[str] = None,
    body: Optional[str] = None, assignee: Optional[str] = None, author: Optional[str] = None,
) -> bool:
    """Update a triage task and admit it only after a scoped planner review."""
    from hermes_cli.kanban_db import (
        _canonical_assignee, _insert_comment, _append_event, write_txn, recompute_ready,
    )
    from hermes_cli.kanban_intake import is_pending, review as intake_review, record_review

    if title is not None and not title.strip():
        raise ValueError("title cannot be blank")
    assignee = _canonical_assignee(assignee)
    caller_owns_txn = bool(conn.in_transaction)
    with write_txn(conn, allow_nested=True):
        existing = conn.execute(
            "SELECT title, body, assignee FROM tasks WHERE id = ? AND status = 'triage'",
            (task_id,),
        ).fetchone()
        if existing is None:
            return False
        receipt = intake_review(conn, task_id)
        if is_pending(conn, task_id) and receipt is None:
            return False
        sets: list[str] = ["status = 'todo'"]
        params: list[Any] = []
        changed_fields: list[str] = []
        if title is not None and title.strip() != (existing["title"] or ""):
            sets.append("title = ?")
            params.append(title.strip())
            changed_fields.append("title")
        if body is not None and (body or "") != (existing["body"] or ""):
            sets.append("body = ?")
            params.append(body)
            changed_fields.append("body")
        if assignee is not None and assignee != (existing["assignee"] or None):
            sets.append("assignee = ?")
            params.append(assignee)
            changed_fields.append("assignee")
        params.append(task_id)
        cur = conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} "
            f"WHERE id = ? AND status = 'triage'", tuple(params),
        )
        if cur.rowcount != 1:
            return False
        if changed_fields and author and author.strip():
            _insert_comment(
                conn, task_id, author.strip(),
                "Specified — updated " + ", ".join(changed_fields) + " and promoted to todo.",
                int(time.time()),
            )
        _append_event(
            conn, task_id, "specified",
            {"changed_fields": changed_fields} if changed_fields else None,
        )
        if receipt is not None:
            record_review(conn, task_id)
    if not caller_owns_txn:
        recompute_ready(conn)
    return True


def _insert_decomposed_child(
    conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, child: dict,
    author: Optional[str], now: int, receipt: Optional[tuple[str, str]] = None,
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
    from hermes_cli.kanban_intake import creation_admission

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
    intake_state, intake_profile, _ = creation_admission(conn)
    if receipt is not None:
        intake_state, intake_profile = "inherited", receipt[0]
    initial_status = "triage" if intake_state == "pending" else "todo"
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, created_at, created_by, intake_state, intake_profile) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            new_id, child["title"].strip(), body if isinstance(body, str) else None,
            _canonical_assignee(child.get("assignee")), initial_status,
            child_ws_kind, child_ws_path, root_row["tenant"], now,
            (author or "decomposer"), intake_state, intake_profile,
        ),
    )
    _append_event(
        conn, new_id, "created", {"by": author or "decomposer", "from_decompose_of": root_id},
    )
    inherit_creator_origin(conn, new_id, root_id, created_at=now)
    return new_id
