"""Pause and quiesce kanban workers for an in-place Desktop update.

The shared update marker is the coordination primitive: every dispatcher stops
claiming new work while its live owner is replacing the install.  The Desktop
then calls :func:`quiesce_all_workers` before testing the Windows venv lock so
already-running workers are settled without changing their crash budget.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any


def update_dispatch_paused() -> bool:
    """Fail closed unless the shared update marker is proven absent or stale."""
    try:
        from hermes_cli.update_lock import read_update_marker_state

        return read_update_marker_state() != "absent"
    except Exception:
        return True


def _quiesce_task_for_update(conn, task_id: str) -> bool:
    """Release one local claim only after whole-tree termination is proven."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as dispatch

    row = conn.execute(
        "SELECT status, claim_lock, worker_pid, worker_started_at FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None or (row["status"] != "running" and row["claim_lock"] is None):
        return False
    prev_lock = row["claim_lock"]
    if not str(prev_lock or "").startswith(kb._host_prefix()):
        return False
    if row["worker_pid"] is None:
        return False

    termination = dispatch._terminate_reclaimed_worker(
        row["worker_pid"], prev_lock, started_at=row["worker_started_at"]
    )
    if not (
        termination.get("host_local")
        and termination.get("terminated")
        and termination.get("tree_terminated")
    ):
        return False
    termination["update_handoff_neutral"] = True

    with kb.write_txn(conn):
        retry_status = kb._retry_status_for_run(conn, task_id)
        cur = conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, worker_started_at = NULL WHERE id = ? "
            "AND status IN ('running', 'ready', 'blocked') AND claim_lock IS ? "
            "AND worker_pid IS ?",
            (retry_status, task_id, prev_lock, row["worker_pid"]),
        )
        if cur.rowcount != 1:
            return False
        kb._record_reclaim(
            conn,
            task_id,
            termination,
            error="update_handoff",
            payload={
                "manual": False,
                "reason": "Windows desktop update hand-off",
                "prev_lock": prev_lock,
                "retry_status": retry_status,
                "neutral": True,
            },
        )
    return True


def quiesce_all_workers() -> dict[str, Any]:
    """Reclaim host-local running workers on every board under dispatch locks.

    The caller must write the live update marker first.  Holding each board's
    dispatch lock closes the last race with a tick that started before the
    marker appeared; the updater-specific settlement restores the task's source
    phase only after whole-tree termination and preserves its failure budget.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    if not update_dispatch_paused():
        return {"ok": False, "error": "no live update marker", "reclaimed": [], "failed": []}

    reclaimed: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for meta in kb.list_boards(include_archived=True):
        board = str(meta.get("slug") or kb.DEFAULT_BOARD)
        db_path = kb.kanban_db_path(board=board).expanduser()
        try:
            resolved = str(db_path.resolve())
        except OSError:
            resolved = str(db_path)
        if resolved in seen_paths or not db_path.exists():
            continue
        seen_paths.add(resolved)

        try:
            with kbc._dispatch_tick_lock(db_path) as held:
                if not held:
                    failed.append({"board": board, "error": "dispatch lock busy"})
                    continue
                with contextlib.closing(kbc.connect(board=board)) as conn:
                    host_prefix = kb._host_prefix()
                    task_ids = [
                        str(row["id"])
                        for row in conn.execute(
                            "SELECT id, claim_lock FROM tasks "
                            "WHERE status = 'running' OR claim_lock IS NOT NULL"
                        ).fetchall()
                        if str(row["claim_lock"] or "").startswith(host_prefix)
                    ]
                    for task_id in task_ids:
                        if _quiesce_task_for_update(conn, task_id):
                            reclaimed.append({"board": board, "task_id": task_id})
                        else:
                            row = conn.execute(
                                "SELECT status, claim_lock FROM tasks WHERE id = ?", (task_id,)
                            ).fetchone()
                            if row is None or (row["status"] != "running" and row["claim_lock"] is None):
                                continue
                            failed.append({
                                "board": board,
                                "task_id": task_id,
                                "error": "worker did not quiesce or reclaim raced",
                            })
        except Exception as exc:
            failed.append({"board": board, "error": f"{type(exc).__name__}: {exc}"})

    return {"ok": not failed, "reclaimed": reclaimed, "failed": failed}


def main() -> None:
    result = quiesce_all_workers()
    print(json.dumps(result))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
