"""Local mutation replay and pause episodes respect the durable FIFO; ``local_session`` is shared."""
import asyncio
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest


async def local_session(tmp_path, monkeypatch, source='cli'):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': [], 'acp': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
        _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
        _evict_cached_agent=lambda route: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    ref = create_local_session(authority, connection.actor, {'request_id': 'create', 'source': source,
        'cwd': str(tmp_path), 'model': 'old', 'toolsets': []})
    return authority, connection, ref


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['model', 'reset'])
async def test_replayed_receipt_keeps_a_newer_turns_agent(tmp_path, monkeypatch, operation):
    """An exact retry of an OLD model/reset receipt after a newer admission claimed must not
    evict that turn's cached agent (the projection is fenced by execution generation)."""
    from gateway import session_mutations
    from gateway.session_policy import restore_policy
    from hermes_state_runtime import admit_session_input, claim_session_input
    owner, connection, ref = await local_session(tmp_path, monkeypatch)
    async def prepare(authority, live, payload, prepared):
        old = restore_policy(prepared['snapshot']['receipt']['policy'])
        config = old.config()
        config['model'] = {'default': payload['model'], 'provider': 'fixture'}
        return {**prepared, 'policy': asdict(replace(old, model=payload['model'], config_json=json.dumps(config)))}
    monkeypatch.setattr('gateway.session_mutation_model.prepare_model', prepare)
    live = owner.sessions[ref.session_id]
    agent, cached, evicted = object(), {}, []
    def evict(route):
        evicted.append(route)
        cached.pop(route, None)
    owner.runner._evict_cached_agent = owner.runner._evict_cached_agent_at_boundary = evict
    try:
        handle = owner._handle(ref)
        params = dict(session_id=ref.session_id, request_id='op', operation=operation,
                      payload={'model': 'new'} if operation == 'model' else {},
                      expected_revision=handle.revision, expected_generation=handle.execution_generation)
        first = await session_mutations.mutate_session(owner, connection.actor, ref, dict(params))
        if live.task is not None:
            await asyncio.gather(live.task, return_exceptions=True)
        admit_session_input(owner.db, epoch=owner.epoch, principal_id=connection.actor.subject,
                            session_id=ref.session_id, request_id='next', payload={'text': 'next'})
        assert claim_session_input(owner.db, epoch=owner.epoch, session_id=ref.session_id) is not None
        cached[live.route] = agent
        evicted.clear()
        again = await session_mutations.mutate_session(owner, connection.actor, ref, dict(params))
        assert again['revision'] == first['revision']
        assert cached.get(live.route) is agent, f'{operation} replay evicted the newer turn agent: {evicted}'
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_session_busy_pause_keeps_the_notice_episode(tmp_path, monkeypatch):
    """A head blocked behind running work is the same pause episode: the "saved, will be answered"
    notice is not repeated for every message queued behind it."""
    from hermes_state_runtime import admit_session_input, claim_session_input
    owner, connection, ref = await local_session(tmp_path, monkeypatch)
    live = owner.sessions[ref.session_id]
    try:
        for request_id in ('running', 'queued'):
            admit_session_input(owner.db, epoch=owner.epoch, principal_id=connection.actor.subject,
                                session_id=ref.session_id, request_id=request_id, payload={'text': request_id})
            if request_id == 'running':
                assert claim_session_input(owner.db, epoch=owner.epoch, session_id=ref.session_id) is not None
        live.pause_notified = True  # the user was already told about this busy episode
        await owner._drain(ref)
        assert live.pause_notified is True, 'session_busy pause reset the notice episode'
    finally:
        await connection.close()
        owner.db.close()
