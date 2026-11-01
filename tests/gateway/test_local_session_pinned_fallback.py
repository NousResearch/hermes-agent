"""A local session whose creator recorded an explicit route choice (``pin_route``: a typed ``hermes chat -m``,
a Kanban task's model override) never borrows the profile's ``fallback_providers``: the rule a pinned cron
job or delegate child follows (``hermes_cli.fallback_config.scoped_fallback_chain``). A chain entry is
another provider and usually another model, so a caller that pins one route per call (evals, cost
control, a scheduler walking its own chain) silently got a different model when the pinned one failed.

Every place the gateway hands a local session's agent a chain is checked: a fresh in-process agent,
a cached agent's per-turn refresh, a ``/bg`` side agent and a managed worker's ``construct_v1``, plus
the resolution-time walk on an AuthError. A ``model`` nobody pinned (a Desktop composer pick, the TUI's
``HERMES_MODEL``, a legacy row's last model, a ``/new`` copy of one, every receipt from before the
field), a provider or endpoint alone, and sessions following their profile keep the profile chain,
read from a real ``config.yaml``.
"""
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_local_create_env_refs import _owner

BACKUP = {'provider': 'custom', 'model': 'backup-model', 'base_url': 'http://127.0.0.1:8/v1'}
CONFIG = ('platform_toolsets: {cli: []}\n'
          'model: {provider: custom, default: m, base_url: "http://127.0.0.1:9/v1"}\n'
          'fallback_providers:\n'
          '  - {provider: custom, model: backup-model, base_url: "http://127.0.0.1:8/v1"}\n')
LAUNCHES = {
    'plain': ({}, [BACKUP]),
    'endpoint only': ({'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'}, [BACKUP]),
    'model not pinned': ({'model': 'm', 'provider': 'custom'}, [BACKUP]),
    'tui model': ({'source': 'tui', 'model': 'm'}, [BACKUP]),
    'desktop composer pick': ({'source': 'gui', 'model': 'm', 'provider': 'custom'}, [BACKUP]),
    'follows profile': ({'model': 'm', 'follow_profile_config': True}, [BACKUP]),
    'model pinned': ({'model': 'm', 'pin_route': True}, None),
    'model and provider pinned': ({'model': 'm', 'provider': 'custom', 'pin_route': True}, None),
}


async def _session(tmp_path, monkeypatch, launch):
    from gateway.run import GatewayRunner
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(CONFIG)
    owner, connection = await _owner(home, monkeypatch)
    runner = owner.runner
    # The real per-turn profile read (#60955), against this home's config.yaml.
    runner._fallback_model = None
    runner._refresh_fallback_model = GatewayRunner._refresh_fallback_model.__get__(runner)
    runner._apply_fallback_chain_to_agent = GatewayRunner._apply_fallback_chain_to_agent
    runner._peek_session_state = lambda key: None
    ref = create_local_session(owner, connection.actor, {'request_id': 'r', 'source': 'cli', 'cwd': str(home),
                                                         'toolsets': [], **launch})
    policy = restore_policy(local_receipt(owner.db, ref.session_id)['policy'])
    return owner, connection, ref, policy


def _turn(owner, policy, ref, monkeypatch, **runner_attrs):
    from gateway import session_policy
    from gateway.session_execution import GatewaySessionAgentMixin
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    for name, value in runner_attrs.items():
        setattr(owner.runner, name, value)
    turn = GatewaySessionAgentMixin.__new__(GatewaySessionAgentMixin)
    turn._runner = owner.runner
    source = SimpleNamespace(chat_id=ref.session_id, user_id='u', user_id_alt=None, user_name=None,
                             chat_name=None, chat_type='dm', thread_id=None)
    turn._ctx = SimpleNamespace(source=source, user_config={}, enabled_toolsets=[], disabled_toolsets=None,
                                session_id=ref.session_id, session_key='k', _interrupt_depth=0,
                                AIAgent=lambda **kw: SimpleNamespace(**kw))
    return turn


def _fresh_agent_chain(owner, policy, ref, monkeypatch):
    turn = _turn(owner, policy, ref, monkeypatch, _prefill_messages=None, _service_tier=None)
    agent = turn._build_fresh_agent({'model': 'm', 'runtime': {}}, 'local', None, 1, None, {}, False)
    return agent.fallback_model


def _cached_agent_chain(owner, policy, ref, monkeypatch):
    """A reused agent built with a stale chain: the per-turn refresh replaces it (or empties it)."""
    stale = [{'provider': 'custom', 'model': 'stale', 'base_url': 'http://127.0.0.1:7/v1'}]
    agent = SimpleNamespace(_fallback_chain=stale, _fallback_model=stale[0], _fallback_index=0,
                            _fallback_activated=False, _rate_limited_until=0, max_iterations=0)
    turn = _turn(owner, policy, ref, monkeypatch, _session_db=None,
                 _agent_cache_lock=threading.Lock(), _agent_cache={'k': (agent, 'sig', None, ref.session_id)},
                 _agent_config_signature=lambda *a, **kw: 'sig', _extract_cache_busting_config=lambda cfg: (),
                 _init_cached_agent_for_turn=lambda agent, depth: None)
    reused, was_cached = turn._resolve_turn_agent({'model': 'm', 'runtime': {}}, 'local', None, 1, None, {})
    assert was_cached and reused is agent
    return agent._fallback_chain or None


def _side_agent_chain(owner, policy, ref, monkeypatch):
    import run_agent
    from gateway import session_tui_side
    monkeypatch.setattr(run_agent, 'AIAgent', lambda **kw: SimpleNamespace(**kw))
    agent = session_tui_side._side_agent(owner, policy, owner.sessions[ref.session_id], 'bg_pin')
    return agent.fallback_model


def _worker_chain(owner, policy, ref, monkeypatch):
    from gateway.session_worker_construct import construct_inputs
    return construct_inputs(owner, policy, ref.session_id)['fallback_model']


@pytest.mark.asyncio
@pytest.mark.parametrize('site', [_fresh_agent_chain, _cached_agent_chain, _side_agent_chain, _worker_chain],
                         ids=['fresh', 'cached', 'side', 'worker'])
@pytest.mark.parametrize('shape', list(LAUNCHES))
async def test_only_a_pinned_launch_goes_without_the_profile_chain(tmp_path, monkeypatch, shape, site):
    launch, expected = LAUNCHES[shape]
    owner, connection, ref, policy = await _session(tmp_path, monkeypatch, launch)
    try:
        assert site(owner, policy, ref, monkeypatch) == expected
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['plain', 'model pinned'])
async def test_a_pinned_launch_fails_on_a_primary_auth_error_instead_of_resolving_the_chain(
        tmp_path, monkeypatch, shape):
    """Resolution time: an AuthError from the primary walks the frozen chain only for an unpinned
    session (a managed worker already follows ``construct_v1``'s empty chain for a pinned one)."""
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from hermes_cli import runtime_provider, runtime_provider_custom
    from hermes_cli.auth import AuthError
    owner, connection, _ref, policy = await _session(tmp_path, monkeypatch, LAUNCHES[shape][0])

    def resolve(*, target_model=None, **kwargs):
        if target_model != 'backup-model':
            raise AuthError('primary credentials exhausted')
        return {'provider': 'custom', 'api_key': 'backup-key', 'base_url': BACKUP['base_url']}
    monkeypatch.setattr(runtime_provider_custom, '_resolve_named_custom_runtime', lambda **kw: None)
    monkeypatch.setattr(runtime_provider, 'resolve_runtime_provider', resolve)
    try:
        if shape == 'plain':
            assert _resolve_policy_agent_runtime(owner.runner, policy)[0] == 'backup-model'
        else:
            with pytest.raises(AuthError):
                _resolve_policy_agent_runtime(owner.runner, policy)
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.parametrize('flag', ['safe_mode', 'ignore_user_config'])
@pytest.mark.parametrize('source', ['cli', 'gui'])
def test_a_bypass_session_never_reads_the_profile_chain(tmp_path, source, flag):
    """``--safe-mode`` / ``--ignore-user-config`` promise never to read the profile, Desktop included
    (exempt from the pin): a side agent or worker gets no profile chain and no profile read."""
    from gateway.session_local import _bypass_policy
    from gateway.session_worker_construct import session_fallback_chain
    reads = []
    runner = SimpleNamespace(_refresh_fallback_model=lambda: reads.append('profile') or [BACKUP])
    policy = _bypass_policy({'source': source, 'cwd': str(tmp_path), 'model': 'm', flag: True}, private_secrets={})
    assert (session_fallback_chain(runner, policy), reads) == (None, [])


def test_a_pinned_session_never_reads_the_profile_chain(tmp_path):
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import session_fallback_chain
    reads = []
    runner = SimpleNamespace(_refresh_fallback_model=lambda: reads.append('profile') or [BACKUP])
    pinned = build_policy({'cwd': str(tmp_path), 'model': 'm', 'pin_route': True}, {})
    unpinned = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {})
    assert (session_fallback_chain(runner, pinned), reads) == (None, [])
    assert (session_fallback_chain(runner, unpinned), reads) == ([BACKUP], ['profile'])


def test_pin_route_needs_a_model_and_a_boolean(tmp_path):
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError
    for params in ({'pin_route': True}, {'model': 'm', 'pin_route': 'yes'},
                   {'model': 'm', 'pin_route': True, 'follow_profile_config': True},
                   {'model': 'm', 'pin_route': True, 'source': 'gui'}):
        with pytest.raises(RuntimeStoreError) as refused:
            build_policy({'cwd': str(tmp_path), **params}, {})
        assert refused.value.field == 'pin_route'


@pytest.mark.asyncio
@pytest.mark.parametrize('launch, pinned', [({'model': 'm', 'pin_route': True}, True), ({'model': 'm'}, False)])
async def test_the_thin_clients_new_keeps_the_sessions_pin(tmp_path, monkeypatch, launch, pinned):
    """``/new`` copies the frozen launch request: a pinned chat's ``/new`` stays pinned on the same model,
    and a ``model`` nobody pinned stays unpinned."""
    import json
    from gateway.session_local import create_local_session
    from gateway.session_policy import CREATE_FIELDS, restore_policy
    from hermes_cli.gateway_chat_commands import _new
    from hermes_state_local import local_receipt
    owner, connection, ref, policy = await _session(tmp_path, monkeypatch, launch)
    created = []

    async def rpc(method, **params):
        if method == 'session.info':
            return {'launch_request': json.loads(policy.request_json), 'cwd': policy.cwd}
        if method == 'runtime.describe':
            return {'session_create': {'parameters': sorted(CREATE_FIELDS)}}
        created.append(create_local_session(owner, connection.actor, params))
        return {'session_id': created[-1].session_id}

    async def adopt(snapshot):
        pass
    monkeypatch.setattr('hermes_cli.gateway_chat.client_env_policy', lambda fields: {})
    view = SimpleNamespace(client=SimpleNamespace(rpc=rpc), session_id=ref.session_id, launch_api_key=None,
                           adopt=adopt)
    try:
        await _new(view, '')
        fresh = restore_policy(local_receipt(owner.db, created[-1].session_id)['policy'])
        assert (fresh.pins_route, fresh.model) == (pinned, 'm')
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_a_pinned_create_retries_exactly(tmp_path, monkeypatch):
    """``pin_route`` rides the frozen request: the same create again is the same session, and a retry
    that drops the pin is a different request."""
    from gateway.session_local import create_local_session
    from hermes_state_runtime import RuntimeStoreError
    owner, connection, ref, _policy = await _session(tmp_path, monkeypatch, {'model': 'm', 'pin_route': True})
    request = {'request_id': 'r', 'source': 'cli', 'cwd': str(tmp_path / 'home'), 'toolsets': [], 'model': 'm'}
    try:
        assert create_local_session(owner, connection.actor, {**request, 'pin_route': True}).session_id == ref.session_id
        with pytest.raises(RuntimeStoreError):
            create_local_session(owner, connection.actor, request)
    finally:
        await connection.close()
        owner.db.close()


async def _adopted(tmp_path, monkeypatch):
    """A pre-authority CLI row adopted through the real titled resume (its last model is copied into
    the creation request), on an owner whose runner reads the profile chain."""
    from gateway import run
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local_title import resolve_titled_session
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    config = {'model': {'default': 'profile-default', 'provider': 'custom'}, 'platform_toolsets': {'cli': []},
              'fallback_providers': [BACKUP]}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: config)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db.create_session('old-cli', source='cli', model='profile-default', cwd=str(tmp_path))
    store._db.end_session('old-cli', 'cli_exit')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _peek_session_state=lambda route: None,
                             _refresh_fallback_model=lambda: [BACKUP])
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='legacy')
    actor = Principal('native', str(tmp_path), frozenset({'session:create', 'session:read', 'session:submit',
                                                         'session:control'}), 'native-socket')
    owner._native_legacy_transports = {actor.transport_id: actor}
    ref = resolve_titled_session(owner, actor, 'old-cli')
    return owner, actor, ref, runner.adapters[Platform.LOCAL].policies[ref.session_id]


@pytest.mark.asyncio
async def test_a_resumed_pre_authority_session_keeps_the_profile_chain_after_a_cold_restore(tmp_path, monkeypatch):
    """The adopted row's last model is in its creation request but nobody pinned it; the receipt
    restored by a restarted owner (no re-adoption) still keeps the chain."""
    from gateway.config import Platform
    from gateway.session_local_title import resolve_titled_session
    from gateway.session_worker_construct import session_fallback_chain
    owner, actor, _ref, policy = await _adopted(tmp_path, monkeypatch)
    try:
        assert policy.model == 'profile-default' and not policy.pins_route
        assert session_fallback_chain(owner.runner, policy, owner) == [BACKUP]
        owner.sessions.clear()
        resumed = resolve_titled_session(owner, actor, 'old-cli')
        restored = owner.runner.adapters[Platform.LOCAL].policies[resumed.session_id]
        assert session_fallback_chain(owner.runner, restored, owner) == [BACKUP]
    finally:
        owner.db.close()


@pytest.mark.asyncio
async def test_new_from_an_unpinned_session_with_a_model_keeps_the_profile_chain(tmp_path, monkeypatch):
    """The thin client's ``/new`` copies the frozen launch request, ``model`` included: a copy of a model
    nobody pinned stays unpinned."""
    from gateway.config import Platform
    from gateway.session_local import create_local_session, local_session_info
    from gateway.session_policy import CREATE_FIELDS
    from gateway.session_worker_construct import session_fallback_chain
    from hermes_cli.gateway_chat_commands import _new
    owner, actor, ref, _policy = await _adopted(tmp_path, monkeypatch)
    created = []

    async def rpc(method, **params):
        if method == 'session.info':
            return local_session_info(owner, ref)
        if method == 'runtime.describe':
            return {'session_create': {'parameters': sorted(CREATE_FIELDS)}}
        assert method == 'session.create' and params.get('model') == 'profile-default'
        created.append(create_local_session(owner, actor, params))
        return {'session_id': created[-1].session_id}

    async def adopt(snapshot):
        pass
    monkeypatch.setattr('hermes_cli.gateway_chat.client_env_policy', lambda fields: {})
    view = SimpleNamespace(client=SimpleNamespace(rpc=rpc), session_id=ref.session_id, launch_api_key=None,
                           adopt=adopt)
    try:
        await _new(view, '')
        fresh = owner.runner.adapters[Platform.LOCAL].policies[created[-1].session_id]
        assert session_fallback_chain(owner.runner, fresh, owner) == [BACKUP]
    finally:
        owner.db.close()


def test_a_config_only_worker_keeps_its_frozen_chain_without_reading_the_profile(tmp_path, monkeypatch):
    """``--ignore-user-config``: the frozen snapshot's own chain serves the worker, at construction and in
    its key walk, even with a pinned model; the profile is never read. The chain here is synthetic: a
    real ``--ignore-user-config`` snapshot freezes code defaults, whose ``fallback_providers`` is empty."""
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import construct_inputs, worker_api_key
    from hermes_cli import runtime_provider, runtime_provider_custom
    from hermes_cli.auth import AuthError
    policy = build_policy({'cwd': str(tmp_path), 'model': 'primary', 'pin_route': True, 'ignore_user_config': True},
                          {'fallback_providers': [BACKUP]})

    def no_read():
        pytest.fail('profile read')
    owner = SimpleNamespace(profile_id=str(tmp_path), runner=SimpleNamespace(_refresh_fallback_model=no_read))
    construct = construct_inputs(owner, policy, 'c')
    assert construct['fallback_model'] == [BACKUP]
    calls = []

    def resolve(*, target_model=None, **kwargs):
        calls.append(target_model)
        if target_model == 'primary':
            raise AuthError('exhausted')
        return {'provider': 'custom', 'api_key': 'test-key'}
    monkeypatch.setattr(runtime_provider_custom, '_resolve_named_custom_runtime', lambda **kw: None)
    monkeypatch.setattr(runtime_provider, 'resolve_runtime_provider', resolve)
    assert worker_api_key(policy, construct) is None
    assert calls == ['primary', 'backup-model']
