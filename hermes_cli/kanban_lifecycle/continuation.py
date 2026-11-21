"""Finite-batch continuation: end the owner's run as ``continued`` and requeue the same card.

Not completion: no children are released, no artifacts staged, no workspace removed, and no
completion/goal/PR acceptance gate is bypassed. Only the current run's owner may yield, and
parent gating is re-checked inside the same write transaction (a parent linked mid-run keeps
the card where it is).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import write_txn


def continue_task(
    conn: sqlite3.Connection, task_id: str, *, summary: str,
    metadata: Optional[dict] = None, expected_run_id: Optional[int] = None,
) -> bool:
    """Close run ``expected_run_id`` of ``task_id`` as ``continued`` and set the card ``ready``.

    False (nothing written) when the run is not the task's open current run, the task is not
    ``running``, or a parent is still open. ``summary`` is the next worker's handoff.
    """
    handoff = str(summary or "").strip()
    if not handoff:
        raise ValueError("continuation summary cannot be blank")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("continuation metadata must be an object/dict")
    if expected_run_id is None:
        return False
    with write_txn(conn):
        if not kb._parents_satisfied(conn, task_id):
            return False
        cur = conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
            "last_heartbeat_at = NULL, consecutive_failures = 0, "
            "last_failure_error = NULL "
            "WHERE id = ? AND status = 'running' AND current_run_id = ? "
            "AND EXISTS (SELECT 1 FROM task_runs r WHERE r.id = tasks.current_run_id "
            "AND r.task_id = tasks.id AND r.ended_at IS NULL)",
            (task_id, int(expected_run_id)),
        )
        if cur.rowcount != 1:
            return False
        run_id = kb._end_run(
            conn, task_id, outcome="continued", status="continued",
            summary=handoff, metadata=metadata,
        )
        payload: dict[str, Any] = {"summary": handoff.splitlines()[0][:400]}
        if metadata:
            payload["metadata"] = metadata
        kb._append_event(conn, task_id, "continued", payload, run_id=run_id)
    return True
