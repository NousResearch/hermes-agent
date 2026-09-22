"""Durable ownership of a Linux child-subreaper execution, distinct from root exit.

A lost supervisor never proves cleanup. Only its observation of ECHILD after
reaping every descendant can settle this receipt. Legacy root receipts do not
provide this stronger guarantee.
"""
import json
import math
import secrets
import os
import signal
import time

CONTRACT = 'linux-child-subreaper-v1'


def read(conn, run_id):
    row = conn.execute('SELECT execution_scope FROM task_runs WHERE id=?', (run_id,)).fetchone()
    return json.loads(row['execution_scope']) if row and row['execution_scope'] else None


def settled(conn, run_id):
    scope = read(conn, run_id)
    return bool(scope and scope.get('contract') == CONTRACT and scope.get('state') == 'settled'
                and scope.get('children_reaped') is True and scope.get('supervisor_fingerprint'))


def request_stop(conn, run_id):
    from hermes_cli.kanban_db_dispatch import _worker_identity
    scope = read(conn, run_id)
    if scope and scope.get('state') == 'active':
        pid, fingerprint = scope.get('supervisor_pid'), scope.get('supervisor_fingerprint')
        if not pid or pid == os.getpid() or not hasattr(os, 'pidfd_open'):
            return
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            return
        try:
            if _worker_identity(pid, fingerprint) == 'owned':
                signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            pass
        finally:
            os.close(fd)


def request_task_stop(conn, task_id):
    for row in conn.execute('SELECT id FROM task_runs WHERE task_id=? AND execution_scope IS NOT NULL', (task_id,)):
        if not settled(conn, row['id']):
            request_stop(conn, row['id'])


def prepare(conn, task, *, deadline=None):
    from hermes_cli import kanban_db as kb
    with kb.write_txn(conn):
        row = conn.execute('''SELECT r.*,t.current_run_id,t.claim_lock task_lock,t.status task_status
            FROM task_runs r JOIN tasks t ON t.id=r.task_id WHERE r.id=? AND r.task_id=?''',
            (task.current_run_id, task.id)).fetchone()
        if (not row or row['current_run_id'] != task.current_run_id or row['task_lock'] != task.claim_lock
                or row['claim_lock'] != task.claim_lock or row['task_status'] != 'running'
                or row['spawn_state'] != 'attempted' or row['execution_scope'] is not None):
            raise RuntimeError('fresh original spawn ownership required for execution scope')
        if row['max_runtime_seconds'] is not None:
            runtime_cutoff = row['started_at'] + row['max_runtime_seconds']
            deadline = runtime_cutoff if deadline is None else min(deadline, runtime_cutoff)
        if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
            raise ValueError('finite absolute execution deadline required')
        scope = {'contract':CONTRACT, 'id':secrets.token_hex(24), 'state':'prepared',
                 'deadline':deadline, 'prepared_at':time.time()}
        conn.execute('UPDATE task_runs SET execution_scope=? WHERE id=?', (json.dumps(scope), task.current_run_id))
        return scope


def activate(conn, task_id, run_id, claim_lock, scope_id, pid, fingerprint):
    from hermes_cli import kanban_db as kb
    with kb.write_txn(conn):
        row = conn.execute('SELECT task_id,claim_lock,spawn_state FROM task_runs WHERE id=?', (run_id,)).fetchone()
        scope = read(conn, run_id)
        if (not row or row['task_id'] != task_id or row['claim_lock'] != claim_lock
                or row['spawn_state'] not in {'attempted','receipted','uncertain'}
                or not scope or scope['id'] != scope_id or scope['state'] != 'prepared' or not fingerprint):
            raise RuntimeError('original execution scope cannot be replayed')
        scope.update(state='active', supervisor_pid=pid, supervisor_fingerprint=fingerprint)
        conn.execute('UPDATE task_runs SET execution_scope=? WHERE id=?', (json.dumps(scope), run_id))
        return scope


def bind_worker(conn, run_id, scope_id, supervisor_pid, supervisor_fingerprint, worker_pid, worker_fingerprint):
    """Persist the direct child's identity before its launch gate opens."""
    from hermes_cli import kanban_db as kb
    with kb.write_txn(conn):
        scope = read(conn, run_id)
        if (not scope or scope['id'] != scope_id or scope['state'] != 'active'
                or scope['supervisor_pid'] != supervisor_pid
                or scope['supervisor_fingerprint'] != supervisor_fingerprint
                or scope.get('worker_pid') is not None or not worker_fingerprint):
            raise RuntimeError('original execution worker cannot be replaced')
        scope.update(worker_pid=worker_pid, worker_fingerprint=worker_fingerprint)
        conn.execute('UPDATE task_runs SET execution_scope=? WHERE id=?', (json.dumps(scope), run_id))


def finish(conn, run_id, scope_id, pid, fingerprint, *, reason, returncode):
    """Called by the still-live owning supervisor after kernel ECHILD proof."""
    from hermes_cli import kanban_db as kb
    with kb.write_txn(conn):
        scope = read(conn, run_id)
        if (not scope or scope['id'] != scope_id or scope['state'] != 'active'
                or scope['supervisor_pid'] != pid or scope['supervisor_fingerprint'] != fingerprint):
            raise RuntimeError('execution cleanup owner changed')
        scope.update(state='settled', children_reaped=True, finished_at=time.time(),
                     reason=reason, returncode=returncode)
        conn.execute('UPDATE task_runs SET execution_scope=? WHERE id=?', (json.dumps(scope), run_id))
