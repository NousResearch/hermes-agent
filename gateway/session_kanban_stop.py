"""Retire earlier native attempts before another attempt may execute the same task."""
import asyncio
from contextlib import closing
from functools import partial
import json
from pathlib import Path
import sqlite3

from hermes_state_runtime import RuntimeStoreError


async def worker_control_frame(authority, ref, row, policy, worker, scope, frame):
    if frame == {'type': 'ready', 'pid': scope['pid']}:
        return True
    if frame == {'type': 'kanban_stop'} and policy.kanban_json is not None:
        from gateway.session_ancillary import record_process_stop
        from gateway.session_runtime_workers import tracked_write
        await tracked_write(authority, partial(record_process_stop, authority, ref, row['generation']))
        worker.control({'type': 'stop', 'processes': True})
        return True
    return False


def attempt_aborted(context):
    with closing(sqlite3.connect(Path(context['db']).as_uri() + '?mode=ro', uri=True)) as conn:
        row = conn.execute('SELECT status FROM task_runs WHERE id=?', (context['run_id'],)).fetchone()
    return row is None or row[0] not in {'done', 'review', 'scheduled'}


async def retain_failed_stop(authority, ref, row, policy):
    if policy.kanban_json is not None and await asyncio.to_thread(attempt_aborted, json.loads(policy.kanban_json)):
        from gateway.session_ancillary import record_process_stop
        from gateway.session_runtime_workers import tracked_write
        await tracked_write(authority, partial(record_process_stop, authority, ref, row['generation']))


def require_current_claim(conn, context):
    row = conn.execute(
        'SELECT t.status,t.current_run_id,t.claim_lock,r.status,r.claim_lock '
        'FROM tasks t JOIN task_runs r ON r.task_id=t.id WHERE t.id=? AND r.id=?',
        (context['task_id'], context['run_id'])).fetchone()
    if row is None or tuple(row) != ('running', context['run_id'], context['claim_lock'], 'running', context['claim_lock']):
        raise RuntimeStoreError('stale_kanban_claim')


def previous_attempts(context):
    with closing(sqlite3.connect(Path(context['db']).as_uri() + '?mode=ro', uri=True)) as conn:
        require_current_claim(conn, context)
        rows = conn.execute(
            "SELECT e.run_id,e.payload FROM task_events e LEFT JOIN task_runs r ON r.id=e.run_id "
            "WHERE e.task_id=? AND e.kind='owner_admitted' AND e.run_id<? "
            "AND (r.status IS NULL OR r.status NOT IN ('done','review','scheduled') OR EXISTS ("
            "SELECT 1 FROM task_events stop WHERE stop.task_id=e.task_id AND stop.id>e.id "
            "AND stop.kind IN ('blocked','block_loop_detected'))) ORDER BY e.id",
            (context['task_id'], context['run_id'])).fetchall()
    return [{**json.loads(row[1]), 'run_id': row[0]} for row in rows]


def check_previous(authority, context, binding):
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    if Path(binding['db']).resolve() != Path(authority.db.db_path).resolve():
        raise RuntimeStoreError('kanban_previous_owner_unavailable')
    saved = local_receipt(authority.db, binding['session_id'])
    policy = restore_policy(saved['policy'])
    old = json.loads(policy.kanban_json or '{}')
    if (policy.source != 'kanban' or old.get('run_id') != binding['run_id']
            or old.get('task_id') != context['task_id'] or old.get('board') != context['board']
            or Path(old.get('db', '')).resolve() != Path(context['db']).resolve()):
        raise RuntimeStoreError('invalid_kanban_claim')


def previous_workers_gone(context, binding):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _poll_worker_exit, _process_fingerprint, _worker_alive
    with closing(sqlite3.connect(Path(context['db']).as_uri() + '?mode=ro', uri=True)) as conn:
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
            "AND kind IN ('spawned','worker_registered','worker_bound')", (context['task_id'], binding['run_id'])).fetchall()
    workers = [json.loads(row[0]) for row in rows]
    for worker in workers:
        pid = worker.get('pid')
        if type(pid) is not int or pid <= 0:
            return False
        if kb._pid_alive(pid):
            if _process_fingerprint(pid) is None or not _poll_worker_exit(pid, worker.get('started_at')):
                return False
            if kb._pid_alive(pid) and (_process_fingerprint(pid) is None or _worker_alive(pid, worker.get('started_at'))):
                return False
    return True


async def retire_previous_attempts(connection, context):
    """An authorized new claim can stop its aborted predecessor, never repeat its effects.

    The existing board events identify the native owner session. Its ordinary Stop,
    execution task and process checkpoint establish quiescence, including after reload;
    another claim or a dead interpreter by itself cannot establish that.
    """
    from gateway.session_ancillary import process_stop_pending, record_process_stop, retry_process_stop
    from gateway.session_contract import SessionRef
    from gateway.session_managed_worker import STOP_ACK_SECONDS, ManagedExecutionUnknown
    from gateway.session_runtime_workers import tracked_write

    authority = connection.authority
    prior = await asyncio.to_thread(previous_attempts, context)
    seen = set()
    for binding in prior:
        sid = binding['session_id']
        if sid in seen:
            continue
        seen.add(sid)
        await asyncio.to_thread(check_previous, authority, context, binding)
        ref = SessionRef(authority.profile_id, sid)
        await connection.resume(ref, {})
        handle = await authority.resolve(connection.actor, ref)
        if sid in authority.executing:
            try:
                await connection.interrupt(ref, {'execution_generation': handle.execution_generation})
            except RuntimeStoreError as exc:
                if exc.reason != 'managed_worker_lost':
                    raise
        task = authority.sessions[sid].task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), STOP_ACK_SECONDS + 10)
            except TimeoutError as exc:
                raise RuntimeStoreError('kanban_previous_execution_unknown') from exc
            except ManagedExecutionUnknown:
                pass  # retain its unknown result; physical cleanup below does not replay that turn
        async with authority.sessions[sid].claim_gate:
            if sid in authority.executing:
                raise RuntimeStoreError('kanban_previous_execution_unknown')
            handle = await authority.resolve(connection.actor, ref)
            await tracked_write(authority, partial(record_process_stop, authority, ref, handle.execution_generation))
            if (not await asyncio.to_thread(previous_workers_gone, context, binding)
                    or not await retry_process_stop(authority, ref)
                    or process_stop_pending(authority.db, sid) is not None):
                raise RuntimeStoreError('kanban_previous_execution_unknown')
