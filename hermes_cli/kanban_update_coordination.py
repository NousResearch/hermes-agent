"""Pause and quiesce kanban workers for an in-place Desktop update.

The shared update marker is the coordination primitive: every dispatcher stops
claiming new work while its live owner is replacing the install.  The Desktop
then calls :func:`quiesce_all_workers` before testing the Windows venv lock so
already-running workers are reclaimed without consuming their crash budget.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any


def update_dispatch_paused() -> bool:
    """Return whether a live updater owns the shared install marker."""
    try:
        from hermes_cli.update_lock import read_live_update

        return read_live_update() is not None
    except Exception:
        # The update lock is an availability gate, not a reason to brick normal
        # dispatch when an old/broken install cannot import the helper.
        return False


def quiesce_all_workers() -> dict[str, Any]:
    """Reclaim host-local running workers on every board under dispatch locks.

    The caller must write the live update marker first.  Holding each board's
    dispatch lock closes the last race with a tick that started before the
    marker appeared; reclaim restores the task's source phase and resets its
    failure budget instead of misclassifying an updater-requested stop as a
    crash.
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
                    task_ids = [
                        str(row["id"])
                        for row in conn.execute(
                            "SELECT id FROM tasks WHERE status = 'running' OR claim_lock IS NOT NULL"
                        ).fetchall()
                    ]
                    for task_id in task_ids:
                        if kb.reclaim_task(conn, task_id, reason="Windows desktop update hand-off"):
                            reclaimed.append({"board": board, "task_id": task_id})
                        else:
                            failed.append({"board": board, "task_id": task_id, "error": "reclaim raced"})
        except Exception as exc:
            failed.append({"board": board, "error": f"{type(exc).__name__}: {exc}"})

    return {"ok": not failed, "reclaimed": reclaimed, "failed": failed}


def main() -> None:
    result = quiesce_all_workers()
    print(json.dumps(result))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
