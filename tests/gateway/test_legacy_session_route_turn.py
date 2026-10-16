"""#138912: a session a pre-0.22.1 surface wrote resumes on ITS route, with only that route's credential.

Rows are shaped exactly as v0.21.x persisted them: the TUI's ``_persist_live_session_runtime`` ->
``_runtime_model_config`` (model, provider, base_url, api_mode, reasoning_config, service_tier in
``model_config``), and the CLI's ``AIAgent._session_init_model_config`` (max_iterations, reasoning_config,
max_tokens) whose route lives only in the ``billing_*`` columns ``update_token_counts`` COALESCE-wrote.
The profile now targets another endpoint with its own key. Two loopback providers record what each
received on the adopted session's next turn.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

SID = '20261012_122515_721bf6'


def _adopt_and_resolve(home, legacy_route, source='tui', billing=None):
    from gateway.config import GatewayConfig
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local_migration import bind_native_transport, resolve_local_target
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt

    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = asyncio.run(initialize_session_authority(runner, profile_id=str(home), instance_id='fixture'))
    actor = Principal('uid:1000', str(home), frozenset({'session:read', 'session:create'}), 'native')
    bind_native_transport(authority, actor, {'provider': 'local', 'profile_id': str(home), 'instance_id': 'fixture'})
    reasoning = {'enabled': True, 'effort': 'xhigh'}
    authority.db.create_session(SID, source=source, model='legacy-model', model_config={
        'model': 'legacy-model', **legacy_route, 'api_mode': 'chat_completions',
        'reasoning_config': reasoning, 'service_tier': 'priority'} if source in ('tui', 'desktop') else {
        'max_iterations': 90, 'reasoning_config': reasoning, 'max_tokens': None})
    if billing:
        authority.db.update_token_counts(SID, input_tokens=1, model='legacy-model', **billing)
    try:
        resolve_local_target(authority, actor, SID)
        policy = restore_policy(local_receipt(authority.db, SID)['policy'])
        return policy, _resolve_policy_agent_runtime(runner, policy)
    finally:
        store._db.close()


def _write_profile(home, profile_url, extra=''):
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(
        f'model:\n  default: profile-model\n  provider: custom\n  base_url: {profile_url}\n'
        '  api_key: sk-profile-only\n  context_length: 128000\n'
        'agent:\n  api_max_retries: 1\n  reasoning_effort: low\nplatform_toolsets:\n  tui: []\n' + extra)


@pytest.mark.parametrize('shape', ['tui-custom', 'tui-named', 'tui-provider-only', 'cli-billing'])
def test_adopted_legacy_session_turn_goes_to_its_route_with_its_own_key(monkeypatch, shape):
    from run_agent import AIAgent
    with FakeLLMServer([Text('profile')]) as profile, FakeLLMServer([Text('legacy ok')]) as legacy:
        home = Path(os.environ['HERMES_HOME'])  # the per-test home conftest isolates
        monkeypatch.setattr('gateway.run._hermes_home', home)
        _write_profile(home, profile.base_url, extra=(
            f'custom_providers:\n  - name: legacy-box\n    base_url: {legacy.base_url}\n    api_key: sk-legacy-own\n'))
        if shape == 'tui-provider-only':
            # A registry provider's route is its name: the endpoint and key are that provider's own.
            monkeypatch.setenv('OPENROUTER_BASE_URL', legacy.base_url)
            monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-legacy-own')
            policy, (model, runtime) = _adopt_and_resolve(home, {'provider': 'openrouter'})
        elif shape == 'cli-billing':
            policy, (model, runtime) = _adopt_and_resolve(home, {}, source='cli', billing={
                'billing_provider': 'custom', 'billing_base_url': legacy.base_url})
        else:
            policy, (model, runtime) = _adopt_and_resolve(home, {
                'provider': 'custom' if shape == 'tui-custom' else 'custom:legacy-box', 'base_url': legacy.base_url})
        agent = AIAgent(model=model, **{k: runtime[k] for k in ('api_key', 'base_url', 'provider', 'api_mode')},
                        reasoning_config=policy.reasoning_config, quiet_mode=True, skip_context_files=True,
                        skip_memory=True, enabled_toolsets=['memory'])
        agent.run_conversation('hello again')

        assert profile.requests == []  # the profile's endpoint is never contacted ...
        sent = legacy.main_requests()
        assert [r['model'] for r in sent] == ['legacy-model']
        assert policy.reasoning_config == {'enabled': True, 'effort': 'xhigh'}
        if shape != 'tui-provider-only':  # OpenRouter sends effort only for models it knows reason
            assert sent[0]['reasoning_effort'] == 'xhigh'
        # ... and the profile's key never reaches the legacy endpoint: it gets the credential its own
        # provider resolves (the entry serving that URL, also for a bare ``custom`` route).
        assert {r['auth'] for r in legacy.requests} == {'Bearer sk-legacy-own'}


def test_adopted_cli_custom_session_keeps_custom_route_when_profile_moved_to_another_provider(monkeypatch):
    """A v0.21 CLI row's route is ``billing_provider='custom'`` + ``billing_base_url``. With the profile now
    on Anthropic, adoption must not pair ``anthropic`` (its wire mode and key) with the legacy endpoint."""
    from run_agent import AIAgent
    with FakeLLMServer([Text('legacy ok')]) as legacy:
        home = Path(os.environ['HERMES_HOME'])
        monkeypatch.setattr('gateway.run._hermes_home', home)
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-profile-only')
        home.mkdir(parents=True, exist_ok=True)
        (home / 'config.yaml').write_text(
            'model:\n  default: claude-profile\n  provider: anthropic\n'
            'agent:\n  api_max_retries: 1\nplatform_toolsets:\n  tui: []\n'
            f'custom_providers:\n  - name: legacy-box\n    base_url: {legacy.base_url}\n    api_key: sk-legacy-own\n')
        policy, (model, runtime) = _adopt_and_resolve(home, {}, source='cli', billing={
            'billing_provider': 'custom', 'billing_base_url': legacy.base_url})
        agent = AIAgent(model=model, **{k: runtime[k] for k in ('api_key', 'base_url', 'provider', 'api_mode')},
                        reasoning_config=policy.reasoning_config, quiet_mode=True, skip_context_files=True,
                        skip_memory=True, enabled_toolsets=['memory'])
        agent.run_conversation('hello again')

        assert [r['model'] for r in legacy.main_requests()] == ['legacy-model']
        assert legacy.requests and all('sk-ant-profile-only' not in str(r['headers']) for r in legacy.requests)
        assert {r['auth'] for r in legacy.requests} == {'Bearer sk-legacy-own'}


def test_adopted_legacy_session_whose_provider_lost_credentials_is_refused_by_name(monkeypatch):
    with FakeLLMServer([Text('profile')]) as profile:
        home = Path(os.environ['HERMES_HOME'])
        monkeypatch.setattr('gateway.run._hermes_home', home)
        _write_profile(home, profile.base_url)  # the ``legacy-box`` entry is gone from config.yaml
        with pytest.raises(Exception) as caught:
            _adopt_and_resolve(home, {'provider': 'custom:legacy-box', 'base_url': 'http://127.0.0.1:9/v1'})
        assert 'custom:legacy-box' in str(caught.value)
        assert profile.requests == []


def _run_turn(policy, model, runtime):
    from run_agent import AIAgent
    agent = AIAgent(model=model, **{k: runtime[k] for k in ('api_key', 'base_url', 'provider', 'api_mode')},
                    reasoning_config=policy.reasoning_config, quiet_mode=True, skip_context_files=True,
                    skip_memory=True, enabled_toolsets=['memory'])
    return agent.run_conversation('hello again')


@pytest.mark.parametrize('owner', ['recovered', 'gone'])
def test_named_provider_moved_never_sends_its_new_key_to_the_saved_endpoint(monkeypatch, owner):
    """The session saved ``custom:legacy-box`` at B; ``legacy-box`` now serves A with A's key. A's key
    belongs to A: B gets the credential of the entry that still serves B, or adoption is refused."""
    with FakeLLMServer([Text('new')]) as new, FakeLLMServer([Text('legacy ok')], api_key='sk-legacy-own') as legacy:
        home = Path(os.environ['HERMES_HOME'])
        monkeypatch.setattr('gateway.run._hermes_home', home)
        _write_profile(home, new.base_url, extra=(
            f'providers:\n  legacy-box:\n    base_url: {new.base_url}\n    api_key: sk-new-endpoint-only\n'
            + (f'custom_providers:\n  - name: old-box\n    base_url: {legacy.base_url}\n    api_key: sk-legacy-own\n'
               if owner == 'recovered' else '')))
        saved = {'provider': 'custom:legacy-box', 'base_url': legacy.base_url}
        if owner == 'recovered':
            policy, (model, runtime) = _adopt_and_resolve(home, saved)
            result = _run_turn(policy, model, runtime)
        else:
            with pytest.raises(Exception) as caught:
                _adopt_and_resolve(home, saved)
            assert getattr(caught.value, 'reason', None) == 'launch_credentials_unavailable'
            assert 'custom:legacy-box' in str(caught.value.detail)
        assert new.requests == []
        assert all('sk-new-endpoint-only' not in str(r['headers']) for r in legacy.requests)
        if owner == 'recovered':
            assert result['final_response'] == 'legacy ok'
            assert {r['auth'] for r in legacy.requests} == {'Bearer sk-legacy-own'}


@pytest.mark.parametrize('shape', ['inline', 'env-ref', 'cli-billing', 'ollama-key-env'])
def test_unchanged_route_keeps_the_profile_credential_for_that_endpoint(monkeypatch, shape):
    """The profile still selects exactly the saved endpoint, with a valid credential for it: the
    adopted session keeps using it (literal key, ``${ENV}`` key, CLI billing row, or an ``ollama``
    profile whose saved provider is its resolved ``custom``)."""
    with FakeLLMServer([Text('legacy ok')], api_key='sk-profile-only') as legacy:
        home = Path(os.environ['HERMES_HOME'])
        monkeypatch.setattr('gateway.run._hermes_home', home)
        monkeypatch.setenv('PROFILE_PROBE_KEY', 'sk-profile-only')
        credential = {'inline': 'api_key: sk-profile-only', 'cli-billing': 'api_key: sk-profile-only',
                      'env-ref': 'api_key: ${PROFILE_PROBE_KEY}', 'ollama-key-env': 'key_env: PROFILE_PROBE_KEY'}[shape]
        home.mkdir(parents=True, exist_ok=True)
        (home / 'config.yaml').write_text(
            f'model:\n  default: profile-model\n  provider: {"ollama" if shape == "ollama-key-env" else "custom"}\n'
            f'  base_url: {legacy.base_url}\n  {credential}\n'
            'agent:\n  api_max_retries: 1\nplatform_toolsets:\n  tui: []\n')
        if shape == 'cli-billing':
            resolved = _adopt_and_resolve(home, {}, source='cli', billing={
                'billing_provider': 'custom', 'billing_base_url': legacy.base_url})
        else:
            resolved = _adopt_and_resolve(home, {'provider': 'custom', 'base_url': legacy.base_url})
        policy, (model, runtime) = resolved
        result = _run_turn(policy, model, runtime)
        assert result['final_response'] == 'legacy ok'
        assert [r['model'] for r in legacy.main_requests()] == ['legacy-model']
        assert {r['auth'] for r in legacy.requests} == {'Bearer sk-profile-only'}


@pytest.mark.parametrize('shape', ['desktop-custom', 'auto-base-url', 'stored-api-key', 'cli-billing-auto'])
def test_v0_21_19_row_shapes_go_to_their_own_endpoint_with_its_own_key(monkeypatch, shape):
    """v0.21.19 row shapes beyond the TUI/CLI ones: a Desktop chat (tui_gateway with ``source='desktop'``,
    same ``model_config`` writer), a route whose provider is ``auto`` with a base_url (in model_config,
    or the CLI's ``billing_provider='auto'`` bucket), and a row whose model_config carries an
    ``api_key`` (no v0.21.19 writer stores one; a hand-edited or imported row may). Each turn reaches
    the row's endpoint with that endpoint's configured key; the profile endpoint gets nothing, and a
    stored row key is never sent anywhere."""
    with FakeLLMServer([Text('profile')], api_key='sk-profile-only') as profile, \
            FakeLLMServer([Text('legacy ok')], api_key='sk-legacy-own') as legacy:
        home = Path(os.environ['HERMES_HOME'])
        monkeypatch.setattr('gateway.run._hermes_home', home)
        _write_profile(home, profile.base_url, extra=(
            f'custom_providers:\n  - name: legacy-box\n    base_url: {legacy.base_url}\n    api_key: sk-legacy-own\n'))
        if shape == 'cli-billing-auto':
            resolved = _adopt_and_resolve(home, {}, source='cli', billing={
                'billing_provider': 'auto', 'billing_base_url': legacy.base_url})
        else:
            route = {'provider': 'auto' if shape == 'auto-base-url' else 'custom:legacy-box',
                     'base_url': legacy.base_url}
            if shape == 'stored-api-key':
                route['api_key'] = 'sk-row-stored'
            resolved = _adopt_and_resolve(home, route, source='desktop' if shape == 'desktop-custom' else 'tui')
        policy, (model, runtime) = resolved
        result = _run_turn(policy, model, runtime)
        assert result['final_response'] == 'legacy ok'
        assert profile.requests == []
        assert [r['model'] for r in legacy.main_requests()] == ['legacy-model']
        assert {r['auth'] for r in legacy.requests} == {'Bearer sk-legacy-own'}
        assert 'sk-row-stored' not in str(policy) + str(runtime)
