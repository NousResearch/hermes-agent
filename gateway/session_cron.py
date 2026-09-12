"""Trusted cron producer and owner execution of the existing scheduler contract.

Public callers name a persisted job, never supply agent policy or arbitrary jobs.
The ordinary admission ledger owns execution; the scheduler owns final delivery.
"""
import asyncio
from contextvars import ContextVar
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import threading

from gateway.session_contract import Principal, SessionRef
from hermes_state_runtime import RuntimeStoreError, admit_session_input, get_session_admission

_execution: ContextVar[tuple | None] = ContextVar('cron_owner_execution', default=None)
_owners = {}


def bind_owner(authority):
    home = Path(authority.db.db_path).resolve().parent
    _owners[home] = (os.getpid(), authority, asyncio.get_running_loop())


def unbind_owner(authority):
    home = Path(authority.db.db_path).resolve().parent
    bound = _owners.get(home)
    if bound is not None and bound[1] is authority:
        del _owners[home]


def owner_for_home(home):
    bound = _owners.get(Path(home).resolve())
    if bound is not None and bound[0] == os.getpid() and not bound[2].is_closed():
        return bound[1:]
    return None


def current_execution():
    return _execution.get()


def _actor(authority, actor):
    if actor is None:
        return Principal('cron-owner', authority.profile_id,
                         frozenset({'session:create', 'session:submit', 'session:read', 'session:control'}), 'cron-ticker')
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if not {'session:create', 'session:submit', 'session:control'} <= actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    # Cron fires belong to the profile scheduler, not the transport that observes
    # them. The same execution ID from ticker and CLI must dedupe together.
    return Principal('cron-owner', authority.profile_id, actor.capabilities, actor.transport_id)


def _create(authority, actor, params):
    from cron.jobs import get_job
    from gateway.session_local_recovery import local_identity, restore_local_session
    from gateway.session_policy import build_policy, bind_launch_key
    from gateway.run import _load_gateway_config
    from cron.scheduler import _load_cron_job_config
    from hermes_state_local import commit_local_session
    from gateway.config import Platform
    from gateway.session import SessionSource, SessionEntry
    from gateway.session_lifecycle import _now

    if set(params) != {'job_id', 'request_id', 'extra_prompt'}:
        raise RuntimeStoreError('invalid_params')
    jid, request_id, extra = params['job_id'], params['request_id'], params['extra_prompt']
    if (not isinstance(jid, str) or not jid or not isinstance(request_id, str)
            or not request_id or len(request_id) > 256 or (extra is not None and not isinstance(extra, str))):
        raise RuntimeStoreError('invalid_params')
    request_id = 'cron:' + jid + ':' + request_id
    sid = local_identity(authority.profile_id, actor.subject, request_id)
    if authority.db.get_session(sid) is not None:
        return restore_local_session(authority, sid), request_id
    job = get_job(jid)
    if job is None or job.get('no_agent'):
        raise RuntimeStoreError('not_found')
    config = _load_gateway_config()
    private = {}
    launch = {'source': 'cli', 'request_id': request_id, 'toolsets': []}
    if job.get('workdir'):
        launch['cwd'] = job['workdir']
    policy = build_policy(launch, config, private_secrets=private)
    policy = replace(policy, source='cron', platform='cron', model=_load_cron_job_config(job, jid, job.get('name') or jid).model,
                     request_json=json.dumps({'cron_job': job, 'extra_prompt': extra, 'request_id': request_id}))
    policy = bind_launch_key(authority, sid, policy, None, config_secrets=private)
    from gateway.session_local_recovery import local_source
    source = local_source(authority, sid, actor.subject)
    route = authority.runner.session_store._generate_session_key(source)
    now = _now()
    entry = SessionEntry(route, sid, now, now, origin=source, platform=Platform.LOCAL)
    commit_local_session(authority.db, epoch=authority.epoch, receipt={
        'profile_id': authority.profile_id, 'principal_id': actor.subject, 'request_id': request_id,
        'session_id': sid, 'route': route, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    return restore_local_session(authority, sid), request_id


async def operation(authority, name, params, actor=None):
    actor = _actor(authority, actor)
    authority._require_admission_open()
    if name == 'recover':
        from gateway.session_local_recovery import local_identity
        if (set(params) != {'job_id', 'request_id', 'extra_prompt'}
                or any(not isinstance(params[k], str) or not params[k] for k in ('job_id', 'request_id'))
                or (params['extra_prompt'] is not None and not isinstance(params['extra_prompt'], str))):
            raise RuntimeStoreError('invalid_params')
        request_id = 'cron:' + params['job_id'] + ':' + params['request_id']
        sid = local_identity(authority.profile_id, actor.subject, request_id)
        from hermes_state_runtime import list_session_admissions
        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        row = next((r for r in rows if r['request_id'] == request_id and r['principal_id'] == actor.subject), None)
        if row is None:
            return {'status': 'missing', 'result': None}
        state = await operation(authority, 'status', {'session_id': sid, 'admission_id': row['admission_id']}, actor)
        from gateway.session_local_recovery import restore_local_session
        restore_local_session(authority, sid)
        from gateway.config import Platform
        from gateway.session_local_recovery import local_adapter_map
        policy = local_adapter_map(authority)[Platform.LOCAL].policies[sid]
        frozen = json.loads(policy.request_json)
        if frozen['extra_prompt'] != params['extra_prompt']:
            raise RuntimeStoreError('admission_conflict')
        state['job'] = frozen['cron_job']
        return state
    if name == 'submit':
        from gateway.session_authorities import owner_scope
        # The job id, jobs file and gateway config belong to the OWNING profile.
        with owner_scope(authority):
            ref, request_id = _create(authority, actor, params)
        row = admit_session_input(authority.db, epoch=authority.epoch, principal_id=actor.subject,
                                  session_id=ref.session_id, request_id=request_id,
                                  payload={'text': params['extra_prompt'] or ''})
        authority._publish_pending(ref)
        authority._schedule(ref)
        return {'session_id': ref.session_id, 'admission_id': row['admission_id']}
    if set(params) != {'session_id', 'admission_id'} or name not in {'status', 'cancel'}:
        raise RuntimeStoreError('invalid_params')
    ref = SessionRef(authority.profile_id, params['session_id'])
    authority.authorize(actor, ref, 'session:submit')
    row = get_session_admission(authority.db, admission_id=params['admission_id'])
    if row is None or row['target_session_id'] != ref.session_id or row['principal_id'] != actor.subject:
        raise RuntimeStoreError('permission_denied')
    if name == 'cancel':
        if row['status'] == 'queued':
            await authority.cancel_queued(actor, ref, row['admission_id'])
        elif row['status'] == 'started':
            # The claim commits before execute() registers its event; latching the
            # cancellation on the admission here means that window cannot lose it.
            _cancellations(authority).setdefault(row['admission_id'], threading.Event()).set()
        return {'ok': True}
    result = None
    if row['status'] == 'terminal':
        from gateway.session_results import admission_result
        saved = admission_result(authority.db, row['admission_id'])
        result = saved['result'].get('cron_result') if saved else None
        if result is None:
            result = [False, '', '', row.get('outcome') or 'unknown_execution']
    return {'status': row['status'], 'result': result}


def _cancellations(authority):
    cancellations = getattr(authority, '_cron_cancellations', None)
    if cancellations is None:
        cancellations = authority._cron_cancellations = {}
    return cancellations


async def rpc(connection, name, params):
    return await operation(connection.authority, name, params, connection.actor)


async def execute(authority, ref, row, policy):
    from cron.scheduler import run_job
    from gateway.run import _profile_runtime_scope
    data = json.loads(policy.request_json)
    if (row['payload'] != {'text': data['extra_prompt'] or ''}
            or row['request_id'] != data['request_id']
            or row['principal_id'] != authority.sessions[ref.session_id].source.user_id):
        raise RuntimeStoreError('admission_conflict')
    cancellations = _cancellations(authority)
    cancel = cancellations.setdefault(row['admission_id'], threading.Event())
    token = _execution.set((authority, ref.session_id, data['cron_job']['id'], row['admission_id']))
    try:
        with _profile_runtime_scope(Path(authority.db.db_path).resolve().parent):
            result = await asyncio.to_thread(run_job, data['cron_job'], extra_prompt=data['extra_prompt'],
                                             execution_id=row['admission_id'], cancel_event=cancel)
        authority.pending_results[row['admission_id']] = {
            'result': {'final_response': result[2], 'cron_result': list(result),
                       'failed': not result[0], 'completed': result[0]}, 'usage': {}}
        if not result[0]:
            raise RuntimeError(result[3] or 'cron execution failed')
        return result[2]
    finally:
        _execution.reset(token)
        cancellations.pop(row['admission_id'], None)
