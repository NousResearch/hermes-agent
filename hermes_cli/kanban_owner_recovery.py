"""Read-only canonical admission fences for dispatcher reclaim paths."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import psutil


def owner_reclaim_paused(conn, task_id):
    """Missing/ambiguous owner receipts never authorize another attempt."""
    marker = conn.execute(
        "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
        "WHERE t.id=? AND e.task_id=t.id AND e.kind='owner_admitted' ORDER BY e.id DESC LIMIT 1",
        (task_id,)).fetchone()
    if marker is None:
        return False
    try:
        binding = json.loads(marker['payload'])
        with closing(sqlite3.connect(Path(binding['db']).as_uri() + '?mode=ro', uri=True)) as owner:
            row = owner.execute('SELECT status,outcome FROM session_admissions WHERE target_session_id=? AND request_id=?',
                                (binding['session_id'], binding['request_id'])).fetchone()
        if row is not None and row[0] == 'terminal':
            # Discard acknowledges uncertainty, not permission to repeat effects.
            return row[1] == 'interrupted'
        if row is None or row[0] in {'queued', 'unknown'}:
            return True
        return psutil.Process(binding['pid']).create_time() != binding['birth']
    except (OSError, ValueError, KeyError, sqlite3.Error):
        return True
    except psutil.Error:
        return True



def bound_interpreter_gone(conn, task_id):
    """True when the owner bound an interpreter to the task's current run and it is dead."""
    from hermes_cli.kanban_db_dispatch import _worker_alive
    row = conn.execute(
        "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
        "WHERE t.id=? AND e.task_id=t.id AND e.kind='worker_bound' ORDER BY e.id DESC LIMIT 1",
        (task_id,)).fetchone()
    if row is None:
        return False
    bound = json.loads(row['payload'])
    return not _worker_alive(bound.get('pid'), bound.get('started_at'))


def managed_run_exit_code(conn, task_id, worker_pid, claim_lock):
    """The exit code a managed run's authority recorded for the task's CURRENT run, else None.

    Managed interpreters are children of the authority, not the dispatcher, so their result is
    read from the run's ``worker_result`` event (it survives a dispatcher restart and cannot be
    reaped here). It speaks for the run only under the same claim and from a pid of this run:
    the dispatcher's own worker, or the owner-side interpreter ``worker_bound`` recorded under
    that claim — ``session_kanban._record_worker_result`` writes the interpreter's pid, never the
    spawned submitter's ``tasks.worker_pid``."""
    def latest(kind):
        row = conn.execute(
            "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
            "WHERE t.id=? AND e.task_id=t.id AND e.kind=? ORDER BY e.id DESC LIMIT 1", (task_id, kind)).fetchone()
        try:
            payload = json.loads(row['payload']) if row else {}
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    result, bound = latest('worker_result'), latest('worker_bound')
    pids = {worker_pid, *([bound.get('pid')] if bound.get('claim_lock') == claim_lock else [])}
    return result.get('exit_code') if result.get('claim_lock') == claim_lock and result.get('pid') in pids else None
