"""A resume from a new local shell rebinds the session to the attaching client's environment.

Regression for #144587: the attaching client's ``client_env`` only rode ``session.create``, so a
session resumed from another terminal kept running with the creating shell's values.
"""
import pytest

_CONFIG = ('model: {default: m1, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n'
           'platform_toolsets: {cli: []}\nterminal: {backend: local}\n')
_A = {'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'PROMPT-A'}
_B = {'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'PROMPT-B'}


def _home(monkeypatch, tmp_path):
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(_CONFIG)
    monkeypatch.setattr(run, '_hermes_home', home)
    return home


def _runner(store, handle=None):
    from types import SimpleNamespace
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None,
                             _set_session_reasoning_override=lambda route, value: None,
                             _resolve_session_agent_runtime=lambda **k: ('m1', {}), _handle_message=handle)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return runner


async def _authority(monkeypatch, home, store):
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    first = await initialize_session_authority(_runner(store), profile_id=str(home), instance_id='first')
    # A shell client attached through the gateway's own ticket: the proof create trusts for host values.
    return first, AuthorityConnection(first, object(), {'user_id': 'human', 'native_bootstrap': True})


async def _call(connection, method, **params):
    reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return reply['result']


def _stored_env(authority, sid):
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    return restore_policy(local_receipt(authority.db, sid)['policy']).client_env()


def _turn_prompt(authority, sid):
    """The ephemeral prompt a turn of this session resolves inside its policy scope."""
    from gateway.run_config_loaders import GatewayConfigLoadersMixin
    from gateway.session_policy import policy_for_source, policy_scope
    policy = policy_for_source(authority.runner, authority.sessions[sid].source)
    with policy_scope(policy, authority=authority):
        return GatewayConfigLoadersMixin._load_ephemeral_system_prompt()


@pytest.mark.asyncio
async def test_resume_adopts_the_attaching_shells_client_env(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    home = _home(monkeypatch, tmp_path)
    store = SessionStore(home / 'sessions', GatewayConfig())
    first, owner = await _authority(monkeypatch, home, store)
    try:
        sid = (await _call(owner, 'session.create', request_id='env', source='cli',
                           cwd=str(tmp_path), client_env=dict(_A)))['session_id']
        assert _stored_env(first, sid) == _A
        resumed = await _call(owner, 'session.resume', session_id=sid, client_env=dict(_B))
        assert resumed['session_id'] == sid
        assert _stored_env(first, sid) == _B
        assert _turn_prompt(first, sid) == 'PROMPT-B'
        # Same shell resumes again: nothing moves, no new policy generation.
        generation = store._db.get_session(sid)['runtime_generation']
        await _call(owner, 'session.resume', session_id=sid, client_env=dict(_B))
        assert store._db.get_session(sid)['runtime_generation'] == generation
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_resume_without_client_env_keeps_the_frozen_map(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    home = _home(monkeypatch, tmp_path)
    store = SessionStore(home / 'sessions', GatewayConfig())
    first, owner = await _authority(monkeypatch, home, store)
    try:
        sid = (await _call(owner, 'session.create', request_id='env', source='cli',
                           cwd=str(tmp_path), client_env=dict(_A)))['session_id']
        await _call(owner, 'session.resume', session_id=sid)
        assert _stored_env(first, sid) == _A
        assert _turn_prompt(first, sid) == 'PROMPT-A'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_resume_with_an_invalid_map_is_refused_and_keeps_the_frozen_map(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    home = _home(monkeypatch, tmp_path)
    store = SessionStore(home / 'sessions', GatewayConfig())
    first, owner = await _authority(monkeypatch, home, store)
    try:
        sid = (await _call(owner, 'session.create', request_id='env', source='cli',
                           cwd=str(tmp_path), client_env=dict(_A)))['session_id']
        reply = await owner.dispatch({'id': 2, 'method': 'session.resume',
                                      'params': {'session_id': sid, 'client_env': {'HERMES_EPHEMERAL_SYSTEM_PROMPT': 1}}})
        assert 'error' in reply, reply
        assert _stored_env(first, sid) == _A
    finally:
        await owner.close()
        store.close_all_db_handles()
