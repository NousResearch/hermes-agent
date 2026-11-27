"""Narrow task-lifecycle normalization helpers for the Kanban kernel."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LifecycleNormalization:
    disposition: str
    reason: Optional[str] = None
    previous_worker_started_at: Optional[int] = None


def normalize_ready_worker_start_residue(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    resume_authorization_id: str,
) -> LifecycleNormalization:
    """Clear an isolated terminal worker-start fingerprint atomically.

    A start fingerprint is not a process handle without ``worker_pid``.  It may
    be cleared only when the task is READY and every claim/run/PID/session
    handle is absent, including any open run row.  All other states fail closed.
    """
    from hermes_cli import kanban_db as kb

    with kb.write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, claim_expires, current_run_id, worker_pid, "
            "worker_started_at, session_id, "
            "EXISTS(SELECT 1 FROM task_runs WHERE task_id = tasks.id "
            "AND ended_at IS NULL) AS has_open_run "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return LifecycleNormalization("refused", "task_missing")
        previous = row["worker_started_at"]
        if previous is None:
            return LifecycleNormalization("unchanged")
        if row["status"] != "ready":
            return LifecycleNormalization("refused", "task_not_ready", int(previous))
        if any(
            row[column] is not None
            for column in (
                "claim_lock",
                "claim_expires",
                "current_run_id",
                "worker_pid",
                "session_id",
            )
        ):
            return LifecycleNormalization("refused", "active_task_handle", int(previous))
        if row["has_open_run"]:
            return LifecycleNormalization("refused", "active_run", int(previous))

        cur = conn.execute(
            "UPDATE tasks SET worker_started_at = NULL "
            "WHERE id = ? AND status = 'ready' "
            "AND claim_lock IS NULL AND claim_expires IS NULL "
            "AND current_run_id IS NULL AND worker_pid IS NULL "
            "AND worker_started_at = ? AND session_id IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM task_runs "
            "WHERE task_id = tasks.id AND ended_at IS NULL)",
            (task_id, previous),
        )
        if cur.rowcount != 1:
            return LifecycleNormalization("refused", "lifecycle_changed", int(previous))
        kb._append_event(
            conn,
            task_id,
            "lifecycle_normalized",
            {
                "field": "worker_started_at",
                "previous_value": int(previous),
                "reason": "terminal_residue",
                "resume_authorization_id": resume_authorization_id,
                "same_lane_resume": True,
            },
            run_id=None,
        )
        return LifecycleNormalization("normalized", previous_worker_started_at=int(previous))
