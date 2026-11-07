"""Owner-side monitor preparation before allocating a conversation for a cron fire.

The v2 operations leave legacy submit receipts unchanged. A tracked observation
survives its caller; interrupted preparation is unknown, never sampled again.
Changed data enters the frozen policy separately from strict-scanned user input.
Sessionless terminal payloads compact after the existing seven-day retry horizon
on session maintenance; compact identity fences never authorize reexecution.
"""
import asyncio
from contextvars import ContextVar
import json
import threading
import time

from hermes_state_runtime import RuntimeStoreError, _epoch, _json
from hermes_state_cron_monitor import PREFIX as _PREFIX, fingerprint

_PREPARED = ContextVar('cron_prepared_monitor', default=None)


def prepared_context(job):
    from gateway.session_cron import current_execution
    data, execution = _PREPARED.get(), current_execution()
    if (data is not None and execution is not None
            and data['profile_id'] == execution[0].profile_id
            and data['job_id'] == execution[2] == job.get('id')
            and data['request_id'] == execution[3] == job.get('execution_id')):
        return data
    return None


def bind_context(data):
    return _PREPARED.set(data)


def reset_context(token):
    _PREPARED.reset(token)


def identity(authority, actor, params):
    from gateway.session_local_recovery import local_identity
    from hermes_state_keys import cron_request_id
    if (set(params) != {'job_id', 'request_id', 'extra_prompt'}
            or any(not isinstance(params[k], str) or not params[k] for k in ('job_id', 'request_id'))
            or len(params['request_id']) > 256
            or (params['extra_prompt'] is not None and not isinstance(params['extra_prompt'], str))):
        raise RuntimeStoreError('invalid_params')
    return local_identity(authority.profile_id, actor.subject,
                          cron_request_id(params['job_id'], params['request_id']))


def lookup(authority, actor, params):
    sid = identity(authority, actor, params)
    value = authority.db.get_meta(_PREFIX + sid)
    if value is None:
        return None
    record = json.loads(value)
    saved = record.get('fingerprint') or fingerprint(sid, record['params'])
    if saved != fingerprint(sid, params):
        raise RuntimeStoreError('admission_conflict')
    return record


def _save(authority, sid, record):
    if record['status'] == 'terminal':
        record.setdefault('settled_at', time.time())
    encoded = _json(record)
    def write(conn):
        _epoch(conn, authority.epoch)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                     'ON CONFLICT(key) DO UPDATE SET value=excluded.value', (_PREFIX + sid, encoded))
    authority.db._execute_write(write)


def terminal_result(record):
    if record is None or record['status'] == 'prepared':
        return None
    if record['status'] in {'preparing', 'expired'}:
        return {'status': 'unknown', 'result': None}
    return {'status': 'terminal', 'result': record['result'], 'job': record['job']}


def _prepare(authority, actor, params):
    from cron import jobs
    from cron.monitor import job_has_monitor
    from cron.scheduler import _apply_monitor_gate
    from gateway.session_authorities import owner_scope
    from hermes_cli.config import require_parseable_user_config
    from hermes_state_mutation_retirement import RETIRED_PREFIX

    sid = identity(authority, actor, params)
    old = lookup(authority, actor, params)
    if old is not None:
        return old
    # An admitted or retired identity must never sample a new observation on retry.
    if authority.db.get_session(sid) is not None or authority.db.get_meta(RETIRED_PREFIX + sid):
        return None
    if ':' in params['request_id']:
        raise RuntimeStoreError('invalid_params')
    with owner_scope(authority):
        job = jobs.get_job(params['job_id'])
        if job is None or job.get('no_agent') or not job_has_monitor(job):
            return None
        require_parseable_user_config()
        if jobs.job_payload_is_empty(job):
            return None
        record = {'params': dict(params), 'job': job, 'status': 'preparing'}
        # A crash after starting a script cannot prove it had no effects. Keep this fire
        # unknown rather than sampling again or concealing an already-observed change.
        _save(authority, sid, record)
        early, _, context = _apply_monitor_gate(
            job, job['id'], str(job.get('name') or job.get('prompt') or job['id']), params['extra_prompt'])
        record.update(status='terminal' if early is not None else 'prepared',
                      result=list(early) if early is not None else None, context=context)
        _save(authority, sid, record)
        return record


async def submit_legacy(authority, actor, params):
    """Serialize protocol versions before either can start source side effects."""
    from gateway.session_cron import _submit
    from gateway.session_runtime_workers import track_mutation
    sid = identity(authority, actor, params)
    if (sid in getattr(authority, '_cron_monitor_fires', {})
            or lookup(authority, actor, params) is not None):
        raise RuntimeStoreError('admission_conflict')
    pending = getattr(authority, '_cron_legacy_submits', None)
    if pending is None:
        pending = authority._cron_legacy_submits = {}
    active = pending.get(sid)
    if active is None:
        task = track_mutation(authority, _submit(authority, actor, dict(params)))
        active = pending[sid] = (dict(params), task)
        task.add_done_callback(lambda done: pending.pop(sid, None))
    if active[0] != params:
        raise RuntimeStoreError('admission_conflict')
    return await asyncio.shield(active[1])


async def operation(authority, name, params, actor):
    """Poll an owner-owned handoff; no source execution waits on an RPC lifetime."""
    from gateway.session_cron import operation as legacy_operation
    from gateway.session_runtime_workers import track_mutation

    sid = identity(authority, actor, params)
    legacy = getattr(authority, '_cron_legacy_submits', {}).get(sid)
    if legacy is not None:
        if legacy[0] != params:
            raise RuntimeStoreError('admission_conflict')
        if name == 'cancel_v2':
            receipt = await asyncio.shield(legacy[1])
            return await legacy_operation(authority, 'cancel', receipt, actor)
        return {'status': 'preparing', 'result': None}
    pending = getattr(authority, '_cron_monitor_fires', None)
    if pending is None:
        pending = authority._cron_monitor_fires = {}
    active = pending.get(sid)
    if active is not None:
        if active[0] != params:
            raise RuntimeStoreError('admission_conflict')
        if name == 'cancel_v2':
            active[2].set()
            return {'ok': True}
        return {'status': 'preparing', 'result': None}
    record = lookup(authority, actor, params)
    if name == 'cancel_v2':
        existing = await legacy_operation(authority, 'recover', params, actor)
        if 'admission_id' in existing:
            return await legacy_operation(authority, 'cancel', {
                'session_id': existing['session_id'], 'admission_id': existing['admission_id']}, actor)
        return {'ok': True}
    terminal = terminal_result(record)
    if terminal is not None:
        return terminal
    existing = await legacy_operation(authority, 'recover', params, actor)
    if existing['status'] != 'missing' or name == 'recover_v2':
        return existing
    authority._require_admission_open()
    cancel = threading.Event()
    task = track_mutation(authority, _handoff(authority, actor, dict(params), cancel))
    pending[sid] = (dict(params), task, cancel)
    task.add_done_callback(lambda done: pending.pop(sid, None))
    return {'status': 'preparing', 'result': None}


async def _handoff(authority, actor, params, cancel):
    from gateway.session_cron import _submit, operation as legacy_operation
    prepared = await prepare(authority, actor, params)
    if terminal_result(prepared) is None:
        if cancel.is_set() and prepared is not None:
            prepared.update(status='terminal', context=None,
                            result=[False, '', '', 'Cron execution cancelled before admission'])
            await asyncio.to_thread(_save, authority, identity(authority, actor, params), prepared)
            return
        receipt = await _submit(authority, actor, params, prepared)
        if cancel.is_set():
            await legacy_operation(authority, 'cancel', receipt, actor)


async def prepare(authority, actor, params):
    """One tracked preparation per identity; disconnect cannot abandon its durable result."""
    from gateway.session_runtime_workers import track_mutation
    sid = identity(authority, actor, params)
    pending = getattr(authority, '_cron_monitor_preparations', None)
    if pending is None:
        pending = authority._cron_monitor_preparations = {}
    task = pending.get(sid)
    if task is None:
        task = track_mutation(authority, asyncio.to_thread(_prepare, authority, actor, dict(params)))
        pending[sid] = task
        task.add_done_callback(lambda done: pending.pop(sid, None))
    record = await asyncio.shield(task)
    if record is not None and record.get('params', params) != params:
        raise RuntimeStoreError('admission_conflict')
    return record
