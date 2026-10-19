"""Atomic claim restoration for restart reconciliation."""
from __future__ import annotations

import sqlite3


def restore_orphaned_claim(conn: sqlite3.Connection, row: sqlite3.Row) -> tuple[bool, str, str]:
    """Re-gate a broken claim with CAS; caller holds the board write transaction.

    Return (restored, resume phase, landing status). The dispatcher records the
    landing status separately from review intent so later promotion restores it.
    """
    from hermes_cli import kanban_db as kb

    task_id = row["id"]
    resume_status = kb._retry_status_for_run(conn, task_id)
    new_status = kb._landing_status_after_parents(conn, task_id, resume_status)
    cur = conn.execute(
        "UPDATE tasks SET status = ?, claim_lock = NULL, "
        "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
        "last_heartbeat_at = NULL "
        "WHERE id = ? AND status = 'running' "
        "  AND claim_lock IS ? AND claim_expires IS ?",
        (new_status, task_id, row["claim_lock"], row["claim_expires"]),
    )
    return cur.rowcount == 1, resume_status, new_status
