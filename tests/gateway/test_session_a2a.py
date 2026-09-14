"""Forwarded profile identity is owner-bound, not a title or latest-row guess."""
import json
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection


@pytest.mark.asyncio
async def test_forwarded_identity_survives_reconnect_without_policy_override(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    import gateway.run as run
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'fixture')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.adapters = {}
    runner.session_store = store
    runner._session_db = store._db
    runner._draining = False
    authority = await initialize_session_authority(runner, profile_id='target', instance_id='test')
    db = authority.db
    authority._schedule = lambda ref: None
    transport = SimpleNamespace(write=lambda frame: None)
    conn = AuthorityConnection(authority, transport, {'user_id': 'producer'})
    params = dict(agent='dev', tenant='team', peer='alice', context_id='ctx/a', input_id='first', text='hello')
    response = await conn.dispatch({'id': 1, 'method': 'a2a.forward', 'params': params})
    assert 'result' in response, response
    first = response['result']
    sid = first['session_id']
    live = authority.sessions[sid]
    store._record_gateway_session_peer(sid, live.route, live.source)
    assert db.get_session(sid)['source'] == 'a2a'
    policy = runner.adapters[next(iter(runner.adapters))].policies[sid]
    assert policy.source == policy.platform == 'a2a'
    assert json.loads(policy.request_json)['a2a_identity'] == ['dev', 'team', 'alice', 'ctx/a']
    # Neither changed defaults nor a fresh transport may turn resume into a policy override.
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'changed'}})
    again = AuthorityConnection(authority, transport, {'user_id': 'producer'})
    repeated = await again.dispatch({'id': 2, 'method': 'a2a.forward', 'params': dict(params)})
    assert repeated['result']['admission_id'] == first['admission_id']
    second = await again.dispatch({'id': 3, 'method': 'a2a.forward', 'params': dict(params, input_id='second', text='again')})
    assert second['result']['session_id'] == sid
    assert second['result']['admission_id'] != first['admission_id']
    assert runner.adapters[next(iter(runner.adapters))].policies[sid] == policy
    info = await again.dispatch({'id': 4, 'method': 'session.info', 'params': {'session_id': sid}})
    assert info['result']['source'] == 'a2a'
    assert info['result']['model'] == 'fixture'
    assert info['result']['lazy'] is True
    # Source spelling alone must never grant producer policy.
    with pytest.raises(RuntimeStoreError):
        build_policy({'source': 'a2a'}, {})
    await conn.close()
    await again.close()
    db.close()


@pytest.mark.asyncio
async def test_forwarding_does_not_merge_lossy_context_or_peer_identity(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    import gateway.run as run

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'fixture')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.adapters = {}
    runner.session_store = store
    runner._session_db = store._db
    runner._draining = False
    authority = await initialize_session_authority(runner, profile_id='target', instance_id='test')
    db = authority.db
    authority._schedule = lambda ref: None
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'producer'})
    ids = []
    for peer, context in [('alice', 'ctx/a'), ('alice', 'ctx-a'), ('bob', 'ctx/a')]:
        result = await conn.dispatch({'id': 1, 'method': 'a2a.forward', 'params': dict(
            agent='dev', tenant='team', peer=peer, context_id=context, input_id='first', text='hello')})
        assert 'result' in result, result
        ids.append(result['result']['session_id'])
    assert len(set(ids)) == len(ids)
    conflict = await conn.dispatch({'id': 2, 'method': 'a2a.forward', 'params': dict(
        agent='dev', tenant='team', peer='alice', context_id='ctx/a', input_id='first', text='changed')})
    assert conflict['error']['message'] == 'admission_conflict'
    db.create_session('legacy', source='a2a')
    db.set_session_title('legacy', 'a2a-dev-old-context')
    legacy = await conn.dispatch({'id': 3, 'method': 'a2a.forward', 'params': dict(
        agent='dev', tenant='team', peer='alice', context_id='old/context', input_id='first', text='hello')})
    assert legacy['error']['message'] == 'runtime_coordination_required'
    assert db.get_session('legacy')['title'] == 'a2a-dev-old-context'
    denied = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
        {'user_id': 'reader', 'capabilities': ['session:read']})
    result = await denied.dispatch({'id': 1, 'method': 'a2a.forward', 'params': dict(
        agent='dev', tenant='team', peer='alice', context_id='ctx/a', input_id='first', text='hello')})
    assert result['error']['message'] == 'permission_denied'
    await conn.close()
    await denied.close()
    db.close()
