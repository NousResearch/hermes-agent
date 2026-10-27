"""Task graph initialization and atomic decomposition persistence."""
from __future__ import annotations

import json
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

    Size gate: every child is measured with the deterministic estimator. If any
    child exceeds the configured threshold, an OversizedChildError is raised
    (typed failure, not silent pass-through). The decomposer must break oversized
    children down further before emitting them. Every accepted child records its
    estimate (``allowed``/``decompose``) and carries the budget line as the first
    line of its body, so the worker sees the budget it was sized for.
    """
    from hermes_cli.kanban_db import (
        _canonical_assignee, _link, _append_event, _insert_comment,
        write_txn, recompute_ready,
    )
    from hermes_cli.kanban_size_gate import (
        estimate_card_size, load_size_gate_threshold, record_size_estimate,
        stamp_budget_estimate, OversizedChildError,
    )

    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)

    # Size gate: measure every child and refuse oversized ones with a typed error.
    # This runs BEFORE the transaction so the failure is clean (no partial fan-out).
    # Measure the UNSTAMPED body; the budget stamp is estimator-neutral and is
    # applied afterwards, so re-measuring a stamped child returns the same score.
    threshold = load_size_gate_threshold()
    measurements: list = []
    stamped_children: list[dict] = []
    for idx, child in enumerate(children):
        child_title = child.get("title") or ""
        child_body = child.get("body") or ""
        estimate = estimate_card_size(child_title, child_body)
        if estimate.is_oversized(threshold):
            raise OversizedChildError(idx, child_title, estimate, threshold)
        measurements.append(estimate)
        measured = dict(child)
        if isinstance(child.get("body"), str):
            measured["body"] = stamp_budget_estimate(child["body"], estimate, threshold)
        stamped_children.append(measured)

    # ONE txn so the fan-out is atomic; helpers that open their own write_txn
    # (create_task, link_tasks, add_comment) must not be called in here.
    now = int(time.time())
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if root_row is None or root_row["status"] != "triage":
            return None
        # Dependency links alone do not imply lineage. The completion event is
        # committed with the graph, and survives re-triage or unlinking.
        if conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'decomposed' LIMIT 1",
            (task_id,),
        ).fetchone():
            return None
        child_ids = [
            _insert_decomposed_child(conn, task_id, root_row, child, author, now)
            for child in stamped_children
        ]
        # Record each child's budget estimate in the same txn as its row: a child
        # that exists without an estimate is the hole this pass closes.
        for child_id, estimate in zip(child_ids, measurements):
            record_size_estimate(
                conn, child_id, estimate,
                decision="allowed", path="decompose", threshold=threshold,
            )
        # Sibling edges within the decomposed graph.
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id, child_id = child_ids[p_idx], child_ids[idx]
                _link(conn, parent_id, child_id)
                _append_event(conn, child_id, "linked", {"parent": parent_id, "child": child_id})
        # Root waits for the whole graph: link it as the *child* of every leaf
        # (simpler than computing leaves; cycle-free since the root is only
        # ever a child in the link table). recompute_ready then keeps the root
        # in 'todo' until every child is done, at which point it promotes the
        # root and the orchestrator wakes up to judge completion.
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
    with no lock; leaving it unset makes dispatch acquire a workspace per
    child from the board anchor (a treehouse pool slot when the repo has a
    ``treehouse.toml``, else ``<repo>/.worktrees/<child-id>``).
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


def _event_payload(raw: object) -> dict:
    """Parse a stored event payload into a dict (empty on anything unexpected)."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def latest_decomposed_payload(conn: sqlite3.Connection, task_id: str) -> Optional[dict]:
    """Newest ``decomposed`` event payload for ``task_id``, or None when absent.

    A corrupted payload yields ``{}`` rather than raising: the audit row must
    never be able to crash a caller or strand the card it describes.
    """
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'decomposed' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    return None if row is None else _event_payload(row["payload"])


def decomposed_child_ids(payload: Optional[dict]) -> list[str]:
    """The ``child_ids`` a ``decomposed`` event recorded, defensively.

    Lineage comes from this payload, never from ``task_links``: a decomposed
    child and an ordinary parent are both stored as ``parent_id -> root_id``,
    so the link table cannot tell them apart. Missing or malformed entries are
    dropped instead of raising — an unreadable payload must not crash the
    caller (the card then lands conservatively on ``todo``).
    """
    raw = (payload or {}).get("child_ids")
    if not isinstance(raw, list):
        return []
    return [cid for cid in raw if isinstance(cid, str) and cid]


def triage_park_cause(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Kind of the newest event that parked ``task_id`` in ``triage``.

    Two surfaces park a card there and the newest marker names which one:

    - ``size_estimate`` — the size gate measured the card and recorded
      ``decision=triage`` (creation of an oversized ``ready`` card, or
      ``divert_promotion_to_triage`` on a promotion). This is the dead-end
      :func:`repair_stuck_decomposed_root` exists to undo.
    - ``block_loop_detected`` — ``block_task``'s loop brake fired at
      ``BLOCK_RECURRENCE_LIMIT`` and routed the card to ``triage`` for a human.
      Promoting that trip back to ``ready`` respawns the card every tick, which
      is exactly what ``kanban_db``'s ``recompute_ready`` docs forbid.

    Newest wins, so a card the brake parked earlier and the size gate diverted
    later is repairable again. ``None`` when neither record exists.
    """
    row = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? "
        "AND kind IN ('block_loop_detected', 'size_estimate') "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    return None if row is None else row["kind"]


def repair_stuck_decomposed_root(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Repair a triage card that already has a ``decomposed`` event.

    A decomposed root can be forced back into ``triage`` by the size gate (it
    measures the root's oversized epic body), and ``triage`` is then a terminal
    dead-end — every kernel verb refuses it and the auto-decompose sweep retries
    it forever. This function repairs such a card:

    - If every child is ``done`` or ``archived``, move the root to ``ready``.
    - Otherwise, move it to ``todo`` so it waits on its children.

    Two properties keep the repair from becoming worse than the dead-end:

    1. Cause. Only a size-gate park is repairable. When
       :func:`triage_park_cause` reports ``block_loop_detected`` the card was
       parked by the block-loop brake FOR A HUMAN: writing ``ready`` would
       bypass ``recompute_ready``'s sticky-block check and failure breaker and
       respawn the card on every dispatcher tick, forever
       (``kanban_db.BLOCK_RECURRENCE_LIMIT`` / ``recompute_ready`` document the
       same hazard for the bare promotion). Such a card is left in ``triage``.
    2. Lineage. The child set comes from the ``decomposed`` event payload's
       ``child_ids``, not ``task_links``: links do not imply lineage, so the
       link table also matches an unrelated open parent of the root and would
       land the root on ``todo`` even when its only child is done. A missing or
       malformed payload lands the root on ``todo`` — never ``ready``, and never
       stranded in ``triage``.

    The status write is a compare-and-set evaluated inside the write txn
    (``AND status = 'triage'`` + ``rowcount``): a concurrent archive or move
    between the read above and the txn loses nothing and is never resurrected.

    Returns the new status (``"ready"`` or ``"todo"``), or ``None`` when the card
    is not a repairable stuck decomposed root (not in triage, no decomposed
    event, brake-parked, or a concurrent writer moved it first). Does NOT create
    new children or duplicate the ``decomposed`` event.
    """
    from hermes_cli.kanban_db import write_txn, _append_event

    row = conn.execute(
        "SELECT id, status FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None or row["status"] != "triage":
        return None
    payload = latest_decomposed_payload(conn, task_id)
    if payload is None:
        return None
    if triage_park_cause(conn, task_id) == "block_loop_detected":
        # The loop brake parked this card for a human; repairing it would
        # respawn it every tick with no human in the loop.
        return None
    child_ids = decomposed_child_ids(payload)
    children = []
    if child_ids:
        children = conn.execute(
            "SELECT id, status FROM tasks WHERE id IN "
            "(" + ",".join("?" * len(child_ids)) + ")", tuple(child_ids),
        ).fetchall()
    all_done = bool(children) and all(
        c["status"] in ("done", "archived") for c in children
    )
    new_status = "ready" if all_done else "todo"
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ? AND status = 'triage'",
            (new_status, task_id),
        )
        if cur.rowcount != 1:
            # A concurrent writer (archive_task accepts triage) won the race.
            # Never overwrite a status this call did not verify in the txn.
            return None
        _append_event(conn, task_id, "repaired", {
            "reason": "stuck_decomposed_root",
            "new_status": new_status,
            "children_count": len(children),
            "all_children_done": all_done,
        })
    return new_status
