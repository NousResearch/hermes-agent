"""Task graph initialization and atomic decomposition persistence."""
from __future__ import annotations

import os
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
    """Fan a triage OR blocked task out into child tasks and promote the root to ``todo``.

    The root task stays alive and becomes a *child* of every child —
    when all children reach ``done``, the root promotes to ``ready`` and
    its assignee (typically the orchestrator profile) wakes back up to
    judge completion or spawn more work.

    The root may be ``triage`` (a fresh fan-out) or ``blocked`` (a resume
    fan-out: an operator split a stuck card into children). 2026-09-01
    incident: a resume fan-out linked children as *children of the blocked
    root* (root as parent) while also making the root a child of every
    child — the promoter only dispatches a child once all its parents are
    done, so children waited on the root and the root waited on the
    children: 63 minutes, zero runs, manual unlink to recover. Children
    are therefore NEVER linked under the root's ``parent`` column; the
    root is always a *child* of each child (children runnable immediately,
    the closing card waits on them).

    ``children`` is a list of dicts, each shaped like::

        {
            "title": "...",
            "body": "...",                     # optional
            "assignee": "profile-name",        # optional, None -> default fallback
            "parents": [0, 2],                 # indices into this same children list
        }

    Returns the list of created child task ids (in input order) on
    success. Returns ``None`` when:
      - The root task does not exist
      - The root task is not in ``triage``/``blocked``
      - A cycle would result (caller built a bad graph)

    Validation of titles/assignees happens inside the same write_txn as
    the inserts so a malformed entry aborts the whole decomposition
    cleanly (no orphan children).
    """
    from hermes_cli.kanban_db import (
        _canonical_assignee, _link, _append_event, _insert_comment,
        _inherit_notify_subs, _new_task_id, write_txn, recompute_ready,
        _assignee_is_known, _worktree_holds_unmerged, _tenant_project,
        effective_max_cost, auto_points_comment,
    )

    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)

    # ONE txn so the fan-out is atomic; helpers that open their own write_txn
    # (create_task, link_tasks, add_comment) must not be called in here.
    now = int(time.time())
    child_ids: list[str] = []
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if root_row is None:
            return None
        if root_row["status"] not in ("triage", "blocked"):
            # triage = fresh fan-out; blocked = resume fan-out (an operator
            # split a stuck card into children). Any other status cannot be
            # fanned out. 2026-09-01: a resume fan-out that reversed the link
            # direction (children linked UNDER the blocked root as its
            # children AND root waited on children) deadlocked for 63 minutes
            # with zero runs — the promoter only dispatches a child once all
            # its parents are done, so children waited on the root and the
            # root waited on the children. Children are therefore never
            # parent-dependent on the root here; the root is a child of each
            # child so children run immediately and the closing card waits on
            # them.
            return None
        # Dependency links alone do not imply lineage. The completion event is
        # committed with the graph, and survives re-triage or unlinking.
        if conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'decomposed' LIMIT 1",
            (task_id,),
        ).fetchone():
            return None
        tenant = root_row["tenant"]
        root_was_triage = root_row["status"] == "triage"

        # Children inherit the root's workspace by default so a fan-out
        # of a code-gen task lands in the parent's project dir/worktree
        # rather than throwaway scratch tmp dirs. A child dict can still
        # override with its own 'workspace_kind' / 'workspace_path'.
        root_ws_kind = root_row["workspace_kind"] or "scratch"
        root_ws_path = root_row["workspace_path"]

        # Guard 2 (auto-decomposer fix, t_c52b9bc3): WORKTREE INHERITANCE. When
        # the root is a worktree that holds unmerged work (dirty tree OR
        # unpushed commits), the IMPLEMENTATION child — the first dispatchable
        # (non-decision, non-triage) child that will actually rebuild — MUST
        # reuse that worktree/branch instead of minting a fresh one. A fresh
        # worktree would strand the parent's half-done diff (the t_f712e819
        # post-mortem: the partial round-2 diff stayed behind and was re-done).
        # We detect the worktree once; the FIRST non-parked child that will run
        # inherits it. Other children still get fresh worktrees (the per-sibling
        # isolation rule stays intact — only the implementation child touches the
        # parent's checkout).
        root_worktree_has_unmerged = bool(
            root_ws_kind == "worktree"
            and root_ws_path
            and _worktree_holds_unmerged(root_ws_path)
        )
        impl_child_worktree_claimed = False

        # Create children. Status is 'todo' regardless of parents — we
        # link them under the root AFTER creation so the dispatcher
        # sees a coherent state, and recompute_ready() at the end
        # promotes parent-free children to 'ready'. An auto-decomposer
        # child whose title is a product decision (child["triage"]) is
        # created as 'triage' instead — recompute_ready only promotes
        # 'todo'/'blocked', so it stays parked for the PM to accept.
        for idx, child in enumerate(children):
            new_id = _new_task_id()
            title = child["title"].strip()
            body = child.get("body")
            assignee = _canonical_assignee(child.get("assignee"))
            # A decision-shaped child (auto-decomposer-spawned) is parked in
            # triage, not 'todo', so the PM must explicitly accept it before it
            # can be dispatched as authoritative. An unknown (phantom)
            # assignee likewise parks the child in triage so the PM can fix the
            # routing rather than have the dispatcher strand or silently drop it.
            assignee_unknown = assignee is not None and not _assignee_is_known(
                assignee
            )
            child_status = "triage" if (
                child.get("triage") or assignee_unknown
            ) else "todo"
            # A terminal/deploy child marked "hold" is created as a REAL
            # operator_hold block (never todo), so recompute_ready() cannot
            # auto-promote it the moment its parent completes. 2026-09-04
            # GlobalAside: a deploy card created todo with a prose "HELD"
            # auto-promoted to ready as soon as its verify parent finished,
            # one dispatcher tick from deploying past an un-approved gate.
            # A typed operator_hold at creation is sticky and only a human
            # unblock ends it — this is the machine guard, not advisory prose.
            child_block_kind = "operator_hold" if child.get("hold") else None
            if child_block_kind:
                child_status = "blocked"
            # Per-child override wins; otherwise inherit the root's
            # workspace. A child that sets workspace_kind without a path
            # falls back to the root path only when kinds match (so a
            # child can't accidentally point a 'dir' at the root's
            # worktree path or vice versa).
            child_ws_kind = child.get("workspace_kind") or root_ws_kind
            if child.get("workspace_path"):
                child_ws_path = child.get("workspace_path")
            elif child_ws_kind == "worktree":
                # Guard 2: if the root's own worktree holds unmerged work and
                # this is the FIRST dispatchable (non-decision, non-triage)
                # child, inherit the parent's worktree/branch so the half-done
                # diff is not stranded behind a fresh mint. The parent becomes
                # non-dispatchable in the same txn, so only this implementation
                # child will touch that checkout — no sibling-lock violation.
                if (
                    root_worktree_has_unmerged
                    and child_status == "todo"
                    and not impl_child_worktree_claimed
                ):
                    child_ws_path = root_ws_path
                    impl_child_worktree_claimed = True
                else:
                    # Never share one worktree checkout between siblings: the
                    # root's literal path would put every child in the same
                    # directory on the first-dispatched sibling's branch, with
                    # no lock — siblings can be promoted and dispatched
                    # concurrently. Leave the path unset so dispatch
                    # materializes a fresh <repo>/.worktrees/<child-id> per
                    # child from the board anchor.
                    child_ws_path = None
            elif child_ws_kind == root_ws_kind:
                child_ws_path = root_ws_path
            else:
                child_ws_path = None
            # 2026-09-06 (W1): never mint a tenant's code child into an empty
            # scratch folder; anchor it under the tenant's repo as its own
            # worktree (one worktree per card, no sibling sharing).
            if child_ws_kind == "scratch" and tenant:
                _tp = _tenant_project(tenant)
                if _tp is not None and _tp.primary_path:
                    child_ws_kind = "worktree"
                    child_ws_path = os.path.join(
                        str(_tp.primary_path), ".worktrees", new_id
                    )
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, body, assignee, status, workspace_kind, "
                " workspace_path, tenant, created_at, created_by, max_cost, "
                " block_kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id,
                    title,
                    body if isinstance(body, str) else None,
                    assignee,
                    child_status,
                    child_ws_kind,
                    child_ws_path,
                    tenant,
                    now,
                    (author or "decomposer"),
                    # WeRoll cost policy 2026-09-02: decomposer children were the
                    # single biggest uncapped path — t_0ad7eded was minted here
                    # and reached $2.17 with no cap. Children inherit the default
                    # under the $1.00 ceiling; Steve-o re-estimates from there.
                    effective_max_cost(None),
                    child_block_kind,
                ),
            )
            _append_event(
                conn, new_id, "created",
                {"by": author or "decomposer", "from_decompose_of": task_id},
            )
            # Charter §4 (est-mintpath-20260911): a decomposed child is a real
            # work item and gets the same mint-path estimate as any other card.
            # This is the ~70% of cards that never read a SOUL — the reason the
            # 2026-09-04 prose-only attempt moved the metric not at all.
            _insert_comment(
                conn, new_id, "(system)", auto_points_comment(), now,
            )
            if child_block_kind:
                # Typed hold at creation (mirrors create_task): the kind is what
                # escalation-watch, fleet-preflight, stalled-card-watch and
                # _has_sticky_block key on, and the `blocked` event is what
                # dates the hold. A held deploy child must never auto-promote —
                # only a human kanban_unblock ends it.
                _append_event(
                    conn, new_id, "blocked",
                    {"reason": "operator_hold at decomposition",
                     "kind": child_block_kind, "recurrences": 0,
                     "source_status": "created"},
                )
            if assignee_unknown:
                conn.execute(
                    "INSERT INTO task_comments "
                    "(task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        new_id,
                        "(system)",
                        f"Decomposed child with unknown assignee {assignee!r} — "
                        "not a real profile, so this card is parked in triage for "
                        "the PM to accept or reject. Transfer to a valid assignee "
                        "to dispatch.",
                        now,
                    ),
                )
            # Durable origin (session_id + subscriptions) travels with the
            # child independently of dependency edges — upstream added this in
            # _insert_decomposed_child, which the fleet's inline insert below
            # does not go through, so it has to be called here or a decomposed
            # child loses the originating session.
            inherit_creator_origin(conn, new_id, task_id, created_at=now)
            _inherit_notify_subs(conn, new_id, (task_id,), created_at=now)
            child_ids.append(new_id)

        # Link children to their sibling parents (within the decomposed graph).
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id, child_id = child_ids[p_idx], child_ids[idx]
                _link(conn, parent_id, child_id)
                _append_event(conn, child_id, "linked", {"parent": parent_id, "child": child_id})
        # Root waits for the whole graph: link it under EVERY child (simpler
        # than computing leaves; cycle-free since the root is only ever a child).
        for cid in child_ids:
            _link(conn, cid, task_id)
        # Flip the root: triage -> todo, set assignee to the orchestrator ONLY
        # for a fresh triage fan-out. Guard 3 (auto-decomposer fix, t_c52b9bc3):
        # for a BLOCKED-resume fan-out the parent's assignee is left UNCHANGED —
        # no reassignment to switch or any router profile. The incident demoted
        # parent was reassigned to the router profile while still dispatchable;
        # the parent must park as-is (its only role after splitting is to wake
        # when children complete, and its original assignee is who the card
        # belongs to). Fresh fan-outs keep the historical orchestrator-wake
        # behavior (an orchestrator profile awaits children completion).
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None and root_was_triage:
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
    # Outside the txn (own IMMEDIATE txn). ``auto_promote=False`` leaves the
    # children in ``todo`` for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


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
        _new_task_id, _canonical_assignee, _append_event, _insert_comment,
        auto_points_comment,
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
    # Charter §4 (est-mintpath-20260911): same mint-path estimate as every
    # other card; see auto_points_comment in kanban_db.
    _insert_comment(conn, new_id, "(system)", auto_points_comment(), now)
    inherit_creator_origin(conn, new_id, root_id, created_at=now)
    return new_id
