"""API run controls resolve durable claims, never adapter agent/task ownership."""
from dataclasses import asdict

from gateway.session_contract import Principal, SessionRef
from gateway.session_results import admission_result
from hermes_state_runtime import RuntimeStoreError, _row


def _authority(adapter):
    """The routed profile's authority: ``/p/<profile>/`` middleware already entered its scope."""
    from gateway.session_authorities import active_authority
    return active_authority(adapter.gateway_runner)


def raw_run_admission(adapter, run_id):
    """Unique canonical ownership, retaining the digest for integrity checks."""
    authority = _authority(adapter)
    if authority is None:
        return None
    with authority.db._read_ctx() as conn:
        rows = conn.execute("SELECT * FROM session_admissions WHERE principal_id='api' AND request_id=?", (run_id,)).fetchmany(2)
    if len(rows) > 1:
        raise RuntimeStoreError('admission_conflict')
    return (authority, dict(rows[0])) if rows else None


def run_admission(adapter, run_id):
    owned = raw_run_admission(adapter, run_id)
    return (owned[0], _row(owned[1])) if owned is not None else None


def run_projection(adapter, run_id, *, receipt_identity=None):
    owned = run_admission(adapter, run_id)
    if owned is None:
        return None
    authority, row = owned
    if receipt_identity is not None:
        scope, session_id = receipt_identity
        if (row['target_session_id'] != session_id or row['request_id'] != run_id
                or row['payload'].get('api_turn_v1', {}).get('run_owner_scope') != scope):
            raise RuntimeStoreError('admission_conflict')
    status = {'queued': 'queued', 'started': 'running', 'unknown': 'interrupted', 'terminal': row['outcome']}.get(row['status'])
    saved = admission_result(authority.db, row['admission_id'])
    result = saved.get('result', {}) if saved else {}
    if row['status'] == 'terminal':
        if result.get('interrupted') or row['outcome'] == 'interrupted':
            status = 'cancelled'
        elif result.get('failed') or result.get('error'):
            status = 'failed'
    if receipt_identity is not None:
        # POST replay observes the accepted run, not today's Output readiness.
        # Artifact reads, result projections and ACK keep the default full path.
        if status is None:
            raise RuntimeStoreError('storage_unavailable')
        return {'run_id': run_id, 'status': status, 'session_id': row['target_session_id'],
                'admission_id': row['admission_id'], 'execution_generation': row['generation']}
    pending = []
    live = authority.sessions.get(row['target_session_id'])
    if live is not None and row['status'] == 'started':
        pending = list(live.controls.snapshot(row['target_session_id'], row['generation']))
    return {'pending_controls': pending, 'run_id': run_id, 'status': status, 'session_id': row['target_session_id'],
            'admission_id': row['admission_id'], 'execution_generation': row['generation'],
            'output': result.get('final_response', ''), 'usage': saved.get('usage', {}) if saved else {}}


async def send_clarify(adapter, *, chat_id, **kwargs):
    from gateway.platforms.base import SendResult
    authority = _authority(adapter)
    if authority is None or chat_id not in authority.sessions:
        return SendResult(success=False, error='No canonical API session')
    handle = authority._handle(SessionRef(authority.profile_id, chat_id))
    if handle.execution_state != 'running':
        return SendResult(success=False, error='No active API execution')
    # TurnRunner registers the shared prompt after this ACK; HTTP polling and WS
    # subscribers consume that projection rather than an adapter-local message.
    return SendResult(success=True, message_id=kwargs['clarify_id'])


async def respond_run(adapter, run_id, body, *, kind):
    import uuid
    owned = run_admission(adapter, run_id)
    if owned is None:
        raise RuntimeStoreError('not_found')
    authority, row = owned
    generation = body.get('execution_generation')
    prompt_id = body.get('request_id')
    field = 'choice' if kind == 'approval' else 'answer'
    if set(body) != {'request_id', 'execution_generation', field}:
        raise RuntimeStoreError('invalid_params')
    if row['status'] != 'started' or type(generation) is not int or generation != row['generation']:
        raise RuntimeStoreError('stale_generation')
    ref = SessionRef(authority.profile_id, row['target_session_id'])
    actor = Principal('api', authority.profile_id,
        frozenset({'session:read', 'session:approve', 'session:respond'}), 'api-control:' + uuid.uuid4().hex)
    snapshot = await authority.attach(actor, ref)
    try:
        if not any(p['prompt_id'] == prompt_id and p['kind'] == kind for p in snapshot.prompts):
            raise RuntimeStoreError('approval_not_pending')
        return await authority.respond(actor, ref, generation, prompt_id, {field: body[field]}, kind=kind)
    finally:
        await authority.detach(actor, snapshot.subscription_id)


async def stop_run(adapter, run_id):
    owned = run_admission(adapter, run_id)
    if owned is None:
        raise RuntimeStoreError('not_found')
    authority, row = owned
    ref = SessionRef(authority.profile_id, row['target_session_id'])
    actor = Principal('api', authority.profile_id,
                      frozenset({'session:submit', 'session:control'}), 'api-run:' + run_id)
    if row['status'] == 'queued':
        await authority.cancel_queued(actor, ref, row['admission_id'])
        adapter._stopping_run_ids.add(run_id)
        waiter = authority.waiters.pop(row['admission_id'], None)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)
    elif row['status'] == 'started':
        await authority.interrupt(actor, ref, row['generation'])
        adapter._stopping_run_ids.add(run_id)
        return {'run_id': run_id, 'status': 'stopping', 'admission_id': row['admission_id']}
    elif row['status'] == 'unknown':
        raise RuntimeStoreError('unknown_execution')
    return run_projection(adapter, run_id)


async def resolve_unknown_run(adapter, run_id, body):
    """Resolve only the exact unknown admission durably bound to an owned API run."""
    owned = run_admission(adapter, run_id)
    if owned is None:
        raise RuntimeStoreError('not_found')
    authority, row = owned
    if not isinstance(body, dict) or set(body) != {'admission_id', 'execution_generation'}:
        raise RuntimeStoreError('invalid_params')
    if body['admission_id'] != row['admission_id']:
        raise RuntimeStoreError('not_found')
    generation = body['execution_generation']
    if row['status'] != 'unknown' or type(generation) is not int or generation != row['generation']:
        raise RuntimeStoreError('stale_generation')
    actor = Principal(
        'api', authority.profile_id, frozenset({'session:submit', 'session:control'}),
        'api-run:' + run_id)
    receipt = await authority.resolve_unknown(
        actor, SessionRef(authority.profile_id, row['target_session_id']),
        row['admission_id'], generation)
    return asdict(receipt)
