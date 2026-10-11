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
    from gateway import run
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
    from gateway import run

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
    # A colliding pre-authority title is neither adopted nor a permanent refusal: the exact
    # identity gets its own session and the old row keeps its title and history.
    assert 'result' in legacy, legacy
    assert legacy['result']['session_id'] not in {'legacy', *ids}
    assert db.get_session('legacy')['title'] == 'a2a-dev-old-context'
    assert db.get_messages('legacy') == []
    denied = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
        {'user_id': 'reader', 'capabilities': ['session:read']})
    result = await denied.dispatch({'id': 1, 'method': 'a2a.forward', 'params': dict(
        agent='dev', tenant='team', peer='alice', context_id='ctx/a', input_id='first', text='hello')})
    assert result['error']['message'] == 'permission_denied'
    await conn.close()
    await denied.close()
    db.close()


@pytest.mark.asyncio
async def test_forwarded_turns_are_authored_by_the_peer_for_memory_attribution(tmp_path, monkeypatch):
    """The owner binds the exact peer into the policy; memory providers must see that peer as the
    bot author of every forwarded turn, not the local principal that hosts the session."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_a2a import forward_author
    from gateway.session_ingress import row_turn_author
    from agent.turn_author import a2a_key
    from hermes_state_runtime import list_session_admissions
    from gateway import run

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
    authority._schedule = lambda ref: None
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'producer'})
    params = dict(agent='dev', tenant='team', peer='alice', context_id='ctx/a', input_id='first', text='hello')
    response = await conn.dispatch({'id': 1, 'method': 'a2a.forward', 'params': params})
    assert 'result' in response, response
    sid = response['result']['session_id']
    policy = runner.adapters[next(iter(runner.adapters))].policies[sid]
    author = forward_author(policy)
    assert author == {'id': 'alice', 'name': 'alice', 'is_bot': True}
    assert a2a_key(author) == 'a2a:alice'
    row = list_session_admissions(authority.db, session_id=sid)[0]
    assert row_turn_author(policy, row) == author
    # A producer's own stamp wins; an ordinary local policy names nobody.
    stamped = {**row, 'payload': {**row['payload'], 'local_automation_v1': {'turn_author': {'id': 'bot-7', 'is_bot': True}}}}
    assert row_turn_author(policy, stamped) == {'id': 'bot-7', 'is_bot': True}
    from gateway.session_policy import build_policy
    plain = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {'platform_toolsets': {'cli': []}})
    assert row_turn_author(plain, row) is None
    # An ordinary API session has no policy at all; it names nobody rather than failing admission.
    assert row_turn_author(None, row) is None
    await conn.close()
    authority.db.close()


@pytest.mark.asyncio
async def test_forward_wakes_on_the_completion_frame_instead_of_a_fixed_100ms_poll(tmp_path, monkeypatch):
    """A forwarded turn is observed through its subscribed completion frame; the exact durable
    receipt is re-read on a backed-off safety tick, not ten authority RPCs a second for the turn."""
    import asyncio
    import time
    from websockets.asyncio.server import serve
    from hermes_cli import gateway_client
    from hermes_cli import gateway_runtime
    from gateway.session_a2a import forward_to_owner

    calls = []

    async def owner(ws):
        async for raw in ws:
            request = json.loads(raw)
            calls.append(time.monotonic())
            done = time.monotonic() - calls[0] >= 1.0
            await ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': {
                'status': 'terminal' if done else 'queued', 'admission_id': 'adm', 'outcome': 'completed'}}))
            if len(calls) == 1:
                async def complete():
                    await asyncio.sleep(1.0)
                    await ws.send(json.dumps({'jsonrpc': '2.0', 'method': 'event', 'params': {
                        'type': 'message.complete', 'session_id': 's', 'payload': {'admission_id': 'adm'}}}))
                asyncio.ensure_future(complete())

    async with serve(owner, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1']) as server:
        port = server.sockets[0].getsockname()[1]
        endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{port}')
        monkeypatch.setattr(gateway_runtime, 'ensure_gateway_runtime',
                            lambda home: SimpleNamespace(state='ready', endpoint=endpoint, reason_code=None))
        monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, ep: 'fixture')
        started = time.monotonic()
        receipt = await forward_to_owner(tmp_path, agent='dev', tenant='', peer='alice', context_id='ctx',
                                         input_id='i1', text='hello', timeout=10)
        elapsed = time.monotonic() - started
    assert receipt['status'] == 'terminal'
    assert elapsed < 1.6, elapsed  # woken by the frame, not a late tick
    assert len(calls) <= 5, f'{len(calls)} a2a.forward RPCs for a 1 s turn'
