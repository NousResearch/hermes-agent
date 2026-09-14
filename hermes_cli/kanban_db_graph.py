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


def _explicit_board_db_path(slug: str) -> Path:
    """DB path for ``slug`` IGNORING ``HERMES_KANBAN_DB`` — that env pins the
    calling process's own board (the dispatcher injects it into workers), so an
    explicit in-code board target must resolve through the board layout instead."""
    from hermes_cli.kanban_db import kanban_home, board_dir, DEFAULT_BOARD
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban.db"
    return board_dir(slug) / "kanban.db"


def _require_target_board(board: str) -> str:
    """Validate a decompose target-board slug; ValueError when it is empty,
    malformed, or (for non-``default``) the board does not exist."""
    from hermes_cli.kanban_db import _normalize_board_slug, board_exists, DEFAULT_BOARD
    slug = _normalize_board_slug(board)
    if slug is None:
        raise ValueError("board must be a non-empty board slug")
    if slug != DEFAULT_BOARD and not board_exists(slug):
        raise ValueError(f"board {slug!r} does not exist")
    return slug


def _has_decompose_event(conn: sqlite3.Connection, task_id: str) -> bool:
    """Whether ``task_id`` already carries a ``decomposed`` event — the
    idempotence marker for re-decomposing a cross-board root (which stays in
    ``triage`` and would otherwise be fanned out again on every ``--all``)."""
    row = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'decomposed' LIMIT 1",
        (task_id,),
    ).fetchone()
    return row is not None


def decompose_triage_task(
    conn: sqlite3.Connection, task_id: str, *, root_assignee: Optional[str], children: list[dict],
    author: Optional[str] = None, auto_promote: bool = True, board: Optional[str] = None,
) -> Optional[list[str]]:
    """Fan a triage task out into children and move the root to ``todo``; the root
    waits on every child and wakes (``ready``) when all are done.

    ``children``: dicts of ``title`` (required), ``body``, ``assignee``,
    ``parents`` (indices into this list), optional workspace overrides.
    ``board``: optional target board slug — children and their sibling links
    (including the root-wait edges) are created on that board's DB instead of
    the root's, so a decomposition can place its work where it will be watched
    while the root stays put. The root lookup and the audit comment/event
    always stay on ``conn``'s board. ``None`` keeps the historical
    everything-on-one-board behavior.

    Cross-board semantics (``board`` given): the root deliberately STAYS in
    ``triage`` on its own board. It cannot be flipped to ``todo``: its gating
    links live on the child board, whose promotion sweep cannot see the root's
    row (boards are separate DBs) — and the parent board's sweep would see an
    ungated ``todo`` root and vacuously promote it mid-run, dispatching a
    zombie orchestrator while the children still execute (both failure modes
    proven live during development of this change). A ``triage`` root is not
    dispatchable and not sweep-eligible, so the hazard is closed by
    construction; completion judgment happens where the children are visible.
    Because a cross-board root never leaves ``triage``, a repeat decompose
    (``--all`` sweeps re-find it) is an idempotent no-op returning ``None``
    once a ``decomposed`` event exists — no duplicate child batches.

    Returns child ids in input order, or None when the root is missing / not
    in triage / already cross-board-decomposed. Atomic: a malformed entry
    aborts the whole fan-out.
    """
    from hermes_cli.kanban_db import (
        write_txn, recompute_ready, _canonical_assignee, _link,
        _append_event, _insert_comment, _new_task_id,
    )
    from hermes_cli.kanban_db import (
        kanban_home, board_dir, DEFAULT_BOARD, _normalize_board_slug,
        board_exists, contextlib,
    )
    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)
    target_conn = conn
    close_target = False
    if board is not None:
        slug = _require_target_board(board)
        from hermes_cli import kanban_db_connect as _kbc  # deferred: split module (imported at this module's tail)

        target_conn = _kbc.connect(db_path=_explicit_board_db_path(slug))
        close_target = True

    # ONE txn so the fan-out is atomic; helpers that open their own write_txn
    # (create_task, link_tasks, add_comment) must not be called in here.
    now = int(time.time())
    try:
        with write_txn(conn):
            root_row = conn.execute(
                "SELECT id, status, tenant, workspace_kind, workspace_path "
                "FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            if root_row is None or root_row["status"] != "triage":
                return None
            cross_board = target_conn is not conn
            # Dependency links alone do not imply lineage: the ``decomposed`` event is
            # the idempotence marker for ANY repeat decompose (upstream 258fa9741c),
            # not just cross-board ones — a cross-board root additionally stays in
            # ``triage`` (see docstring), so ``--all`` sweeps would re-find it forever.
            if _has_decompose_event(conn, task_id):
                return None
            if cross_board:
                # Children + their edges in one txn on the target board. The
                # root board's audit comment/event run inside that txn (their
                # INSERTs commit through SQLite's implicit txn; there is no
                # cross-DB transaction), so they are written only after the
                # child rows exist and can never name children that do not.
                with write_txn(target_conn):
                    child_ids = _fanout_children(
                        target_conn, task_id, root_row, children, author, now, read_conn=conn,
                    )
                    if author and author.strip():
                        _insert_comment(
                            conn, task_id, author.strip(),
                            "Decomposed into " + ", ".join(child_ids)
                            + " (children on target board).",
                            now,
                        )
                    _append_event(
                        conn, task_id, "decomposed",
                        {"child_ids": child_ids, "root_assignee": root_assignee},
                    )
                # Root stays in ``triage`` on its own board — it is not a
                # dispatchable card there (see docstring). ``author`` gets the
                # trail; nobody else needs to look at this board to know.
            else:
                child_ids = _fanout_children(conn, task_id, root_row, children, author, now)
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
            recompute_ready(target_conn)
        # Cross-board: no sweep on ``conn``'s board — the root's gating links
        # live on the target board, so a parent-board sweep would see an
        # ungated ``todo`` root and vacuously promote it mid-run (a zombie
        # orchestrator dispatch while the children still execute). Proven
        # live in the smoke run for this change: with the sweep, the root
        # flipped triage -> todo -> ready instantly.
    finally:
        if close_target:
            with contextlib.suppress(Exception):
                target_conn.close()
    return child_ids


def _insert_decomposed_child(
    root_conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, child: dict,
    author: Optional[str], now: int, write_conn: Optional[sqlite3.Connection] = None,
) -> str:
    """Insert one decomposed child as ``todo`` (linked under the root later so
    the dispatcher only ever sees a coherent graph); returns its id.

    Workspace: per-child override wins, else inherit the root's kind. Path
    inherits only when kinds match (a 'dir' child must not point at the
    root's worktree) and NEVER for worktrees — siblings dispatch concurrently
    and one shared checkout would put them all on the first sibling's branch
    with no lock; leaving it unset makes dispatch materialize a fresh
    ``<repo>/.worktrees/<child-id>`` per child from the board anchor.

    ``write_conn``: the DB the child row is written to. It equals ``root_conn``
    (single-board fan-out) except for ``decompose_triage_task(board=...)``,
    where the child + its ``created`` event are written to ``write_conn`` while
    the root's notify subscriptions are read from ``root_conn``.
    """
    from hermes_cli.kanban_db import (
        _new_task_id, _canonical_assignee, _append_event, _inherit_notify_subs,
    )
    write_conn = write_conn if write_conn is not None else root_conn
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
    write_conn.execute(
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
        write_conn, new_id, "created", {"by": author or "decomposer", "from_decompose_of": root_id},
    )
    if write_conn is root_conn:
        # Single-board path: unchanged behavior, including notify-sub inheritance.
        _inherit_notify_subs(root_conn, new_id, (root_id,), created_at=now)
    else:
        # Cross-board path: read-only SELECTs against the root's DB (safe to
        # nest inside its open write txn — they never see uncommitted writes
        # from other writers, which is exactly what must be inherited).
        subs = root_conn.execute(
            "SELECT platform, chat_id, thread_id, user_id, user_id_alt, chat_type, "
            "notifier_profile, delivery_mode, delivery_metadata FROM kanban_notify_subs "
            "WHERE task_id = ?", (root_id,),
        ).fetchall()
        row = write_conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS cursor FROM task_events WHERE task_id = ?",
            (new_id,),
        ).fetchone()
        cursor = int(row["cursor"] if row is not None else 0)
        created_at = int(now)
        write_conn.executemany(
            "INSERT OR IGNORE INTO kanban_notify_subs "
            "(task_id, platform, chat_id, thread_id, user_id, user_id_alt, chat_type, "
            " notifier_profile, delivery_mode, delivery_metadata, created_at, last_event_id) "
            "VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, 'dm'), ?, COALESCE(?, 'notify'), ?, ?, ?)",
            [
                (new_id, s["platform"], s["chat_id"], s["thread_id"], s["user_id"],
                 s["user_id_alt"], s["chat_type"], s["notifier_profile"], s["delivery_mode"],
                 s["delivery_metadata"], created_at, cursor)
                for s in subs
            ],
        )
    return new_id


def _fanout_children(
    write_conn: sqlite3.Connection, task_id: str, root_row: sqlite3.Row, children: list[dict],
    author: Optional[str], now: int, read_conn: Optional[sqlite3.Connection] = None,
) -> list[str]:
    """Insert the children and their sibling + root edges on ``write_conn``.

    ``read_conn`` is the board the root row lives on; it differs from
    ``write_conn`` only for ``decompose_triage_task(board=...)`` (children on
    the target board, root on the caller's). ``_insert_decomposed_child`` and
    ``_link`` run on ``write_conn``; the ``linked`` events ride ``write_conn``
    (they describe the child-side edge)."""
    from hermes_cli.kanban_db import _link, _append_event
    source = read_conn if read_conn is not None else write_conn
    child_ids = [
        _insert_decomposed_child(
            source, task_id, root_row, child, author, now,
            None if write_conn is source else write_conn,
        )
        for child in children
    ]
    # Sibling edges within the decomposed graph.
    for idx, child in enumerate(children):
        for p_idx in child.get("parents") or []:
            parent_id, child_id = child_ids[p_idx], child_ids[idx]
            _link(write_conn, parent_id, child_id)
            _append_event(write_conn, child_id, "linked", {"parent": parent_id, "child": child_id})
    # Root waits for the whole graph: link it under EVERY child (simpler
    # than computing leaves; cycle-free since the root is only ever a child).
    for cid in child_ids:
        _link(write_conn, cid, task_id)
    return child_ids
