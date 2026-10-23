"""Single-query affinity crosses the canonical owner and both agent constructors (#136359)."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('value', [None, '', 'raw-label', 'cli_' + 'a' * 31, 'cli_' + 'g' * 32, 7])
def test_owner_refuses_unbounded_or_raw_cache_scope(tmp_path, value):
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        build_policy({'cwd': str(tmp_path), 'cache_scope': value}, {})


@pytest.mark.parametrize('source', ['tui', 'gui', 'acp'])
def test_owner_refuses_cache_scope_on_other_surfaces(tmp_path, source):
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        build_policy({'source': source, 'cwd': str(tmp_path), 'cache_scope': 'cli_' + 'a' * 32}, {})


def test_binding_is_owner_profile_and_workspace_scoped(tmp_path, monkeypatch):
    from gateway.session_cache_scope import bind_session_cache_scope
    from gateway.session_policy import build_policy, policy_scope
    from agent.prompt_cache_scope import resolve_prompt_cache_scope
    scopes = []
    for profile, workspace in [('a', 'x'), ('b', 'x'), ('a', 'y'), ('a', 'x')]:
        home, cwd = tmp_path / profile, tmp_path / workspace
        home.mkdir(exist_ok=True)
        cwd.mkdir(exist_ok=True)
        monkeypatch.setenv('HERMES_HOME', str(home))
        policy = build_policy({'cwd': str(cwd), 'cache_scope': 'cli_' + 'a' * 32}, {})
        parent = bind_session_cache_scope(SimpleNamespace(session_id='parent'), policy)
        with policy_scope(policy):
            child = SimpleNamespace(session_id='child', _session_db=None)
            assert resolve_prompt_cache_scope(child) != resolve_prompt_cache_scope(parent)
        scopes.append(parent._cli_prompt_cache_scope)
    assert scopes[0] == scopes[3]
    assert len(set(scopes[:3])) == 3
    plain = build_policy({'cwd': str(tmp_path)}, {})
    agent = bind_session_cache_scope(SimpleNamespace(session_id='plain'), plain)
    assert not hasattr(agent, '_cli_prompt_cache_scope')


def _inprocess_parent(runner, live, sid):
    from gateway.session_execution import GatewaySessionAgentMixin
    from run_agent import AIAgent
    turn = GatewaySessionAgentMixin()
    turn._runner = runner
    turn._ctx = SimpleNamespace(
        source=live.source, AIAgent=AIAgent, session_id=sid, session_key=live.route,
        enabled_toolsets=[], disabled_toolsets=['all'], user_config={},
    )
    return turn._build_fresh_agent(
        {'model': 'gpt-5.4', 'runtime': {
            'provider': 'custom', 'api_mode': 'codex_responses',
            'base_url': 'https://chatgpt.com/backend-api/codex', 'api_key': 'synthetic-test-key'}},
        'cli', None, 1, None, {}, True,
    )


def _managed_parent(policy, sid, route):
    from agent.managed_worker import _construct_agent
    from run_agent import AIAgent
    return _construct_agent({}, AIAgent, policy,
        model='gpt-5.4', provider='custom', api_mode='codex_responses',
        base_url='https://chatgpt.com/backend-api/codex', api_key='synthetic-test-key',
        session_id=sid, gateway_session_key=route, enabled_toolsets=[], disabled_toolsets=['all'],
        quiet_mode=True, skip_memory=True, skip_context_files=True, skip_background_review=True,
        save_trajectories=False,
    )


@pytest.mark.asyncio
async def test_canonical_create_reaches_real_parent_requests_without_sharing_history(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    from hermes_cli.cli_cache_scope import cli_cache_scope

    home = tmp_path / 'owner'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    cfg = {'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: cfg)
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'gpt-5.4')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _refresh_fallback_model=lambda: None,
                             _service_tier=None, _prefill_messages=None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='cache-test')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    requests, sessions, routes, agents = [], [], [], []
    digest = cli_cache_scope('private-local-label', tmp_path)
    try:
        for index in range(2):
            reply = await connection.dispatch({'id': index, 'method': 'session.create', 'params': {
                'request_id': f'cache-{index}', 'source': 'oneshot', 'cwd': str(tmp_path),
                'cache_scope': digest, 'ignore_rules': True, 'toolsets': [],
            }})
            assert 'result' in reply, reply
            sid = reply['result']['session_id']
            sessions.append(sid)
            live = authority.sessions[sid]
            routes.append(live.route)
            policy = restore_policy(local_receipt(authority.db, sid)['policy'])
            assert json.loads(policy.request_json)['cache_scope'] == digest
            assert 'private-local-label' not in json.dumps(reply)
            parent = _inprocess_parent(runner, live, sid)
            # A managed worker restores the same frozen receipt in its fresh interpreter.
            worker = _managed_parent(restore_policy(asdict(policy)), sid, live.route)
            agents.extend([parent, worker])
            messages = [{'role': 'system', 'content': 'Stable instructions'},
                        {'role': 'user', 'content': f'query-{index}'}]
            pair = [agent._build_api_kwargs(messages, []) for agent in (parent, worker)]
            assert pair[0]['prompt_cache_key'] == pair[1]['prompt_cache_key']
            assert pair[0]['extra_headers']['session_id'] == pair[1]['extra_headers']['session_id']
            assert parent.session_id == worker.session_id == sid
            assert parent._gateway_session_key == worker._gateway_session_key == live.route
            requests.append(pair[0])
            authority.db.append_message(sid, 'user', content=f'query-{index}')
            assert [m['content'] for m in authority.db.get_messages(sid)] == [f'query-{index}']
        assert sessions[0] != sessions[1] and routes[0] != routes[1]
        assert requests[0]['prompt_cache_key'] == requests[1]['prompt_cache_key']
        assert requests[0]['extra_headers']['session_id'] == requests[1]['extra_headers']['session_id']
        assert requests[0]['extra_headers']['session_id'] not in sessions
    finally:
        for agent in agents:
            agent.close()
