"""Shared Kanban operator semantics.

This module holds mutation behavior that is needed by both the built-in
Kanban dashboard plugin and external operator surfaces.  It intentionally
stays close to :mod:`hermes_cli.kanban_db`: callers still own authentication,
RBAC, request validation, and response serialization; this layer centralizes
state-machine/event semantics that previously lived only in the dashboard
plugin.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import sqlite3

from hermes_cli import kanban_db


def set_status_direct(
    conn: sqlite3.Connection,
    task_id: str,
    new_status: str,
) -> bool:
    """Direct status write for operator drag/drop moves.

    This is for transitions not covered by structured verbs such as
    ``complete_task``/``block_task``/``archive_task``.  It preserves the
    historical dashboard semantics:

    * moving to ``ready`` is refused while any parent is not ``done``;
    * leaving ``running`` clears claim fields and closes the active run as
      ``reclaimed`` with the legacy dashboard summary;
    * a ``status`` task event is appended for the live feed;
    * ``done``/``ready`` moves trigger dependency recomputation.

    The dashboard's single-task route rejects direct ``running`` transitions
    before calling this helper.  Bulk dashboard semantics historically allowed
    direct ``running`` writes, so this helper does not add a new guard.
    """
    with kanban_db.write_txn(conn):
        prev = conn.execute(
            "SELECT status, current_run_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if prev is None:
            return False

        if new_status == "ready":
            parent_statuses = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?",
                (task_id,),
            ).fetchall()
            if parent_statuses and not all(
                p["status"] == "done" for p in parent_statuses
            ):
                return False

        was_running = prev["status"] == "running"
        cur = conn.execute(
            "UPDATE tasks SET status = ?, "
            "  claim_lock = CASE WHEN ? = 'running' THEN claim_lock ELSE NULL END, "
            "  claim_expires = CASE WHEN ? = 'running' THEN claim_expires ELSE NULL END, "
            "  worker_pid = CASE WHEN ? = 'running' THEN worker_pid ELSE NULL END "
            "WHERE id = ?",
            (new_status, new_status, new_status, new_status, task_id),
        )
        if cur.rowcount != 1:
            return False

        run_id = None
        if was_running and new_status != "running" and prev["current_run_id"]:
            run_id = kanban_db._end_run(
                conn,
                task_id,
                outcome="reclaimed",
                status="reclaimed",
                summary=f"status changed to {new_status} (dashboard/direct)",
            )

        kanban_db._append_event(
            conn,
            task_id,
            "status",
            {"status": new_status},
            run_id=run_id,
        )

    if new_status in ("done", "ready"):
        kanban_db.recompute_ready(conn)
    return True


def update_priority(
    conn: sqlite3.Connection,
    task_id: str,
    priority: int,
) -> bool:
    """Set task priority and emit the dashboard-parity ``reprioritized`` event."""
    value = int(priority)
    with kanban_db.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET priority = ? WHERE id = ?",
            (value, task_id),
        )
        if cur.rowcount != 1:
            return False
        kanban_db._append_event(conn, task_id, "reprioritized", {"priority": value})
    return True


def edit_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
) -> bool:
    """Update task title/body fields and emit the dashboard-parity ``edited`` event.

    ``None`` means the field was omitted and should not be changed.  An empty
    or whitespace-only title is rejected with ``ValueError`` so HTTP callers can
    translate it to a 400 without duplicating the validation.
    """
    if title is None and body is None:
        return True

    sets: list[str] = []
    vals: list[Any] = []
    if title is not None:
        cleaned = title.strip()
        if not cleaned:
            raise ValueError("title cannot be empty")
        sets.append("title = ?")
        vals.append(cleaned)
    if body is not None:
        sets.append("body = ?")
        vals.append(body)

    vals.append(task_id)
    with kanban_db.write_txn(conn):
        cur = conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?",
            vals,
        )
        if cur.rowcount != 1:
            return False
        kanban_db._append_event(conn, task_id, "edited", None)
    return True


def bulk_update(
    conn: sqlite3.Connection,
    *,
    ids: list[str],
    status: Optional[str] = None,
    assignee: Optional[str] = None,
    priority: Optional[int] = None,
    archive: bool = False,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Apply dashboard bulk-update semantics and return per-id outcomes.

    Each id is independent: failures are captured on that row and do not abort
    sibling updates.  Later requested fields for the same id continue to run
    even if an earlier requested field failed, matching the original dashboard
    route behavior.
    """
    results: list[dict[str, Any]] = []
    for tid in [i for i in (ids or []) if i]:
        entry: dict[str, Any] = {"id": tid, "ok": True}
        try:
            task = kanban_db.get_task(conn, tid)
            if task is None:
                entry.update(ok=False, error="not found")
                results.append(entry)
                continue

            if archive:
                if not kanban_db.archive_task(conn, tid):
                    entry.update(ok=False, error="archive refused")

            if status is not None and not archive:
                s = status
                if s == "done":
                    ok = kanban_db.complete_task(
                        conn,
                        tid,
                        result=result,
                        summary=summary,
                        metadata=metadata,
                    )
                elif s == "blocked":
                    ok = kanban_db.block_task(conn, tid)
                elif s == "ready":
                    cur = kanban_db.get_task(conn, tid)
                    if cur and cur.status == "blocked":
                        ok = kanban_db.unblock_task(conn, tid)
                    else:
                        ok = set_status_direct(conn, tid, "ready")
                elif s in ("todo", "running", "triage"):
                    ok = set_status_direct(conn, tid, s)
                else:
                    entry.update(ok=False, error=f"unknown status {s!r}")
                    results.append(entry)
                    continue
                if not ok:
                    entry.update(ok=False, error=f"transition to {s!r} refused")

            if assignee is not None:
                try:
                    if not kanban_db.assign_task(conn, tid, assignee or None):
                        entry.update(ok=False, error="assign refused")
                except RuntimeError as exc:
                    entry.update(ok=False, error=str(exc))

            if priority is not None:
                if not update_priority(conn, tid, int(priority)):
                    entry.update(ok=False, error="priority refused")
        except Exception as exc:  # defensive — one bad id shouldn't kill the batch
            entry.update(ok=False, error=str(exc))
        results.append(entry)
    return {"results": results}


def reclaim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
) -> bool:
    """Thin wrapper around the kernel reclaim primitive."""
    return kanban_db.reclaim_task(conn, task_id, reason=reason)


def reassign_task(
    conn: sqlite3.Connection,
    task_id: str,
    profile: Optional[str],
    *,
    reclaim_first: bool = False,
    reason: Optional[str] = None,
) -> bool:
    """Thin wrapper around the kernel reassign primitive."""
    return kanban_db.reassign_task(
        conn,
        task_id,
        profile,
        reclaim_first=reclaim_first,
        reason=reason,
    )


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    dry_run: bool = False,
    max_spawn: int = 8,
    board: Optional[str] = None,
) -> dict[str, Any]:
    """Run one dispatcher tick and serialize its result for operator APIs."""
    result = kanban_db.dispatch_once(
        conn,
        dry_run=dry_run,
        max_spawn=max_spawn,
        board=board,
    )
    try:
        return asdict(result)
    except TypeError:
        return {"result": str(result)}


def board_counts(slug: str) -> dict[str, int]:
    """Return ``{status: count}`` for a board. Safe on an empty DB."""
    try:
        path = kanban_db.kanban_db_path(board=slug)
        if not path.exists():
            return {}
        conn = kanban_db.connect(board=slug)
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
            ).fetchall()
            return {r["status"]: int(r["n"]) for r in rows}
        finally:
            conn.close()
    except Exception:
        return {}


def list_boards(*, include_archived: bool = False) -> dict[str, Any]:
    """Return all boards with task counts and the active slug."""
    boards = kanban_db.list_boards(include_archived=include_archived)
    current = kanban_db.get_current_board()
    for board in boards:
        board["is_current"] = board["slug"] == current
        board["counts"] = board_counts(board["slug"])
        board["total"] = sum(board["counts"].values())
    return {"boards": boards, "current": current}


def create_board(
    slug: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    switch: bool = False,
) -> dict[str, Any]:
    """Create a board and optionally persist it as current."""
    meta = kanban_db.create_board(
        slug,
        name=name,
        description=description,
        icon=icon,
        color=color,
    )
    if switch:
        kanban_db.set_current_board(meta["slug"])
    return {"board": meta, "current": kanban_db.get_current_board()}


def update_board_metadata(
    slug: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
) -> dict[str, Any]:
    """Update display metadata for an existing board."""
    normed = kanban_db._normalize_board_slug(slug)
    if not normed or not kanban_db.board_exists(normed):
        raise FileNotFoundError(f"board {slug!r} does not exist")
    meta = kanban_db.write_board_metadata(
        normed,
        name=name,
        description=description,
        icon=icon,
        color=color,
    )
    return {"board": meta}


def remove_board(slug: str, *, delete: bool = False) -> dict[str, Any]:
    """Archive or hard-delete a board and return the current board pointer."""
    result = kanban_db.remove_board(slug, archive=not delete)
    return {"result": result, "current": kanban_db.get_current_board()}


def switch_board(slug: str) -> dict[str, str]:
    """Persist an existing board as current."""
    normed = kanban_db._normalize_board_slug(slug)
    if not normed or not kanban_db.board_exists(normed):
        raise FileNotFoundError(f"board {slug!r} does not exist")
    kanban_db.set_current_board(normed)
    return {"current": normed}
