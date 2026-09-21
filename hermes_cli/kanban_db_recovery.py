"""Owned-run pause and bounded legacy intake recovery."""
from __future__ import annotations
import sqlite3
from typing import Any, Optional
from hermes_cli import kanban_db as kb

def _is_machine_recoverable_pr_feedback_triage(*, title: Optional[str], idempotency_key: Optional[str], block_kind: Optional[str]) -> bool:
    """Identify legacy PR-feedback intake loops that should return to workers."""
    key = (idempotency_key or '').strip().casefold()
    if not key.startswith(kb._GITHUB_PR_FEEDBACK_IDEMPOTENCY_PREFIX):
        return False
    if key.startswith(kb._GITHUB_PR_INTENT_REVIEW_PREFIX):
        return False
    if block_kind != 'needs_input':
        return False
    return (title or '').startswith('GitHub PR feedback:')

def pause_task(conn: sqlite3.Connection, task_id: str, *, expected_run_id: Optional[int]=None, claimer: Optional[str]=None, reason: str='dispatcher shutdown') -> bool:
    """Cooperatively pause one owned running task and release its claim.

    This is deliberately a run outcome, not a new card status. The task
    returns to the lane from which its run was claimed (``ready`` or
    ``review``), so the normal dispatcher can resume it later. The CAS checks
    bind the transition to the worker's run and claim, preventing an old
    worker from pausing a successor that already reclaimed the card.
    """
    with kb.write_txn(conn):
        row = conn.execute('SELECT status, current_run_id, claim_lock FROM tasks WHERE id = ?', (task_id,)).fetchone()
        if row is None or row['status'] != 'running':
            return False
        run_id = row['current_run_id']
        if run_id is None:
            return False
        if expected_run_id is not None and int(run_id) != int(expected_run_id):
            return False
        if claimer is not None and row['claim_lock'] != claimer:
            return False
        retry_status = kb._retry_status_for_run(conn, task_id, int(run_id))
        where = "WHERE id = ? AND status = 'running' AND current_run_id = ?"
        params: tuple[Any, ...] = (retry_status, task_id, int(run_id))
        if claimer is not None:
            where += ' AND claim_lock = ?'
            params += (claimer,)
        cur = conn.execute('UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL ' + where, params)
        if cur.rowcount != 1:
            return False
        closed_run_id = kb._end_run(conn, task_id, outcome='paused', status='paused', error=None, metadata={'reason': str(reason or 'dispatcher shutdown')})
        kb._append_event(conn, task_id, 'paused', {'reason': str(reason or 'dispatcher shutdown'), 'run_id': closed_run_id, 'retry_status': retry_status}, run_id=closed_run_id)
        return True
