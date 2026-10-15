"""Process-local continuation of a proven, durable failed provider request.

The FIFO owns execution and crash recovery. A retry admission is never a fresh user
input: losing its owner checkpoint refuses it before the agent can run.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from uuid import uuid4

from hermes_state_runtime import RuntimeStoreError, admit_session_input
from tui_gateway.methods_prompt_retry import _retry_history_signature, _retry_tools_complete

continuation = ContextVar('retry_continuation', default=None)


def _history(authority, ref):
    target = authority.physical_target(ref)
    return target, authority.db.get_messages_as_conversation(
        target, include_row_ids=True, repair_alternation=False)


def prepare_checkpoint(authority, live, ref, row, settled, captured):
    """Called after terminal settlement commits, before its completion is published."""
    result = (captured or {}).get('result') or {}
    if settled['outcome'] != 'failed' or result.get('failure_reason') != 'rate_limit':
        return None
    from gateway.config import Platform
    from gateway.session_managed_worker import managed_policy
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, live.source)
    agent = authority.agent(ref)
    if (live.source.platform != Platform.LOCAL or policy is None or policy.platform != 'desktop'
            or managed_policy(authority, ref) is not None or agent is None
            or getattr(agent, 'api_mode', None) == 'codex_app_server'):
        return None
    target, history = _history(authority, ref)
    messages = result.get('messages') or []
    if not history or _retry_history_signature(history) != _retry_history_signature(messages):
        return None
    index = next((i for i in range(len(history) - 1, -1, -1) if history[i].get('role') == 'user'), None)
    if index is None or not _retry_tools_complete(history[index + 1:]):
        return None
    user_id = history[index].get('_row_id')
    if type(user_id) is not int or not isinstance(history[index].get('content'), str):
        return None
    return {
        'token': uuid4().hex, 'target': target, 'signature': _retry_history_signature(history),
        'user_row_id': user_id, 'history': deepcopy(history), 'agent': agent,
        'epoch': authority.epoch, 'generation': row['generation'], 'spent': False,
    }


def _validate(authority, ref, checkpoint):
    target, history = _history(authority, ref)
    if (checkpoint is None or checkpoint['epoch'] != authority.epoch
            or target != checkpoint['target'] or authority.agent(ref) is not checkpoint['agent']
            or _retry_history_signature(history) != checkpoint['signature']):
        raise RuntimeStoreError('retry_checkpoint_unavailable')
    return history


async def retry(connection, ref, params):
    authority = connection.authority
    authority.authorize(connection.actor, ref, 'session:submit')
    if ref.session_id not in connection.subscriptions:
        raise RuntimeStoreError('permission_denied')
    live = authority.sessions[ref.session_id]
    async with live.mutation_lock:
        authority.authorize(connection.actor, ref, 'session:submit')
        authority._require_admission_open()
        if authority._pending_rows(ref) or authority._classic_turn_holds(live.route):
            raise RuntimeStoreError('session_busy')
        checkpoint = live.retry_checkpoint
        if checkpoint is None or checkpoint['spent'] or checkpoint['token'] != params['retry_token']:
            raise RuntimeStoreError('retry_checkpoint_unavailable')
        history = _validate(authority, ref, checkpoint)
        checkpoint['spent'] = True  # lost admission acknowledgement must never resubmit fresh input
        user = next(row for row in history if row.get('_row_id') == checkpoint['user_row_id'])
        # This private envelope is not accepted by prompt.submit. It is re-authorized at claim
        # and needs this exact live owner again at execution; a restart fails closed.
        payload = {'text': user['content'], 'retry_continuation_v1': checkpoint['token']}
        if live.source.user_id != connection.actor.subject:
            payload['local_operator_v1'] = {
                'profile_id': authority.profile_id, 'session_id': ref.session_id,
                'principal_id': connection.actor.subject}
        from gateway.session_runtime_workers import tracked_write
        admit = partial(admit_session_input, authority.db, epoch=authority.epoch,
                        principal_id=connection.actor.subject, session_id=ref.session_id,
                        request_id='retry-' + checkpoint['token'], payload=payload,
                        _authorize_write=authority._admission_gate())
        receipt = await tracked_write(authority, admit, then=partial(authority._admitted, ref, None),
                                      ordered=True, after=partial(authority._schedule_admitted, ref))
        return asdict(receipt)


def check_retry_input(authority, ref, row):
    checkpoint = authority.sessions[ref.session_id].retry_checkpoint
    if (checkpoint is None or not checkpoint['spent']
            or row['payload'].get('retry_continuation_v1') != checkpoint['token']
            or row['request_id'] != 'retry-' + checkpoint['token']):
        raise RuntimeStoreError('retry_checkpoint_unavailable')
    _validate(authority, ref, checkpoint)
    return checkpoint


@contextmanager
def retry_scope(authority, ref, row):
    checkpoint = check_retry_input(authority, ref, row) if 'retry_continuation_v1' in row['payload'] else None
    token = continuation.set((authority, ref, checkpoint) if checkpoint else None)
    try:
        yield
    finally:
        continuation.reset(token)


def resumed_history(agent):
    scope = continuation.get()
    if scope is None:
        return None
    authority, ref, checkpoint = scope
    history = _validate(authority, ref, checkpoint)
    if agent is not checkpoint['agent']:
        raise RuntimeStoreError('retry_checkpoint_unavailable')
    return history


@contextmanager
def resume_agent(agent):
    scope = continuation.get()
    if scope is None:
        yield
        return
    resumed_history(agent)  # recheck immediately before invoking the model
    agent._resume_user_row_id = scope[2]['user_row_id']
    try:
        yield
    finally:
        agent._resume_user_row_id = None


def retry_inflight(live, pending=()):
    checkpoint = live.retry_checkpoint
    if checkpoint is None or checkpoint['spent'] or pending:
        return None
    return {'status': 'error', 'streaming': False, 'error': 'Provider rate limit',
            'retry_token': checkpoint['token'],
            'user': next(row['content'] for row in checkpoint['history']
                           if row.get('_row_id') == checkpoint['user_row_id']),
            'assistant': '', 'error_surface': {'layer': 'provider', 'code': 'rate_limit', 'retryable': True}}
