"""Offline API factory contract: fail closed on auth and preserve explicit effort."""
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason
from agent.fallback_policy import fallback_reason_allowed, reresolve_fallback_reasoning_config
from gateway.platforms.api_server_model_policy import (
    constructor_fallback, create_api_agent, scoped_fallback_chain)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def config(chain=None):
    return {'gateway': {'api_server': {'fallback_providers': chain if chain is not None else
            [{'provider': 'openrouter', 'model': 'example/fallback-model'}]}},
            'fallback_providers': [{'provider': 'nous', 'model': 'example/global-fallback'}]}


def factory(**kwargs):
    return SimpleNamespace(**kwargs)


def test_scoped_chain_installed_only_after_constructor_and_reasoning_survives(monkeypatch):
    from agent import agent_init_fallback
    monkeypatch.setattr('agent.agent_runtime_helpers.sync_credential_pool_entry_id', lambda agent: None)
    seen = []
    def construct(**kwargs):
        seen.append(kwargs)
        return factory(**kwargs)
    explicit = {'enabled': True, 'effort': 'xhigh'}
    agent = create_api_agent(construct, {'quiet_mode': True, 'model': 'example/primary-model',
                             'fallback_model': {'provider': 'nous', 'model': 'example/global-fallback'}},
                             config(), explicit_reasoning=explicit)
    assert seen[0]['fallback_model'] is None
    assert agent._fallback_chain == [{'provider': 'openrouter', 'model': 'example/fallback-model'}]
    explicit['effort'] = 'low'
    agent.model = 'example/fallback-model'
    reresolve_fallback_reasoning_config(agent)
    assert agent.reasoning_config == {'enabled': True, 'effort': 'xhigh'}


@pytest.mark.parametrize('reason', list(FailoverReason) + [None])
def test_capacity_only_gate_never_advances_for_auth_approval_or_unknown(reason):
    from gateway.platforms.api_server_model_policy import CAPACITY_REASONS
    agent = SimpleNamespace(_request_fallback_reasons=CAPACITY_REASONS)
    assert fallback_reason_allowed(agent, reason) == (reason in {
        FailoverReason.rate_limit, FailoverReason.upstream_rate_limit, FailoverReason.overloaded})
    assert fallback_reason_allowed(SimpleNamespace(), reason)  # Other platforms unchanged.


def test_constructor_auth_error_never_tries_fallback():
    calls = []
    def denied(**kwargs):
        calls.append(kwargs)
        raise PermissionError('approval required')
    with pytest.raises(PermissionError):
        create_api_agent(denied, {'fallback_model': config()['fallback_providers']}, config())
    assert calls == [{'fallback_model': None}]


@pytest.mark.parametrize('bad', [{}, 'fallback', [{'provider': 'openrouter'}],
                                     [{'provider': 'openrouter', 'model': 'example/fallback-model', 'api_key': 'forbidden'}]])
def test_malformed_scoped_policy_fails_before_construct(bad):
    cfg = {'gateway': {'api_server': {'fallback_providers': bad}}}
    with pytest.raises(ValueError):
        create_api_agent(lambda **kwargs: pytest.fail('constructed'), {}, cfg)


def test_locked_runtime_and_unconfigured_platform_keep_existing_semantics():
    original = {'fallback_model': [{'provider': 'legacy', 'model': 'old'}]}
    assert create_api_agent(factory, original, {}).fallback_model == original['fallback_model']
    assert create_api_agent(factory, original, config(), locked=True).fallback_model is None


def test_null_chain_inherits_the_global_chain_and_empty_list_disables_fallback(monkeypatch):
    global_chain = [{'provider': 'legacy', 'model': 'old'}]
    monkeypatch.setattr('gateway.run.GatewayRunner._load_fallback_model', lambda: global_chain)
    monkeypatch.setattr('agent.agent_runtime_helpers.sync_credential_pool_entry_id', lambda agent: None)
    unset ={'gateway': {'api_server': {'fallback_providers': None}}}
    assert constructor_fallback(unset, False) == global_chain
    agent = create_api_agent(factory, {'fallback_model': global_chain}, unset)
    assert agent.fallback_model == global_chain
    assert not hasattr(agent, '_request_fallback_reasons')
    disabled = {'gateway': {'api_server': {'fallback_providers': []}}}
    assert constructor_fallback(disabled, False) is None
    agent = create_api_agent(factory, {'fallback_model': global_chain}, disabled)
    assert agent.fallback_model is None and agent._request_fallback_reasons


def test_scoped_chain_is_read_from_config_yaml_through_the_gateway_loader(tmp_path):
    from gateway.run import _load_gateway_config
    (tmp_path / 'config.yaml').write_text(
        'gateway:\n  api_server:\n    fallback_providers:\n'
        '      - {provider: openrouter, model: example/fallback-model}\n', encoding='utf-8')
    token = set_hermes_home_override(tmp_path)
    try:
        chain = scoped_fallback_chain(_load_gateway_config())
    finally:
        reset_hermes_home_override(token)
    assert chain == [{'provider': 'openrouter', 'model': 'example/fallback-model'}]


@pytest.mark.parametrize('explicit', [{'enabled': False}, {'enabled': True, 'effort': 'xhigh'}])
def test_explicit_reasoning_overrides_fallback_model_defaults(explicit, tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('agent:\n  reasoning_effort: low\n')
    agent = SimpleNamespace(model='example/fallback-model', _request_reasoning_config=explicit)
    reresolve_fallback_reasoning_config(agent)
    assert agent.reasoning_config == explicit
    del agent._request_reasoning_config
    reresolve_fallback_reasoning_config(agent)
    assert agent.reasoning_config == {'enabled': True, 'effort': 'low'}


def test_live_agent_capacity_swap_preserves_turn_and_explicit_reasoning(monkeypatch):
    from unittest.mock import MagicMock, patch
    from run_agent import AIAgent
    with (patch('model_tools.get_tool_definitions', return_value=[]),
          patch('model_tools.check_toolset_requirements', return_value={}),
          patch('agent.process_bootstrap.OpenAI')):
        agent = create_api_agent(AIAgent, dict(
            model='example/primary-model', api_key='offline-test-key',
            base_url='https://openrouter.ai/api/v1', quiet_mode=True,
            skip_context_files=True, skip_memory=True, reasoning_config={'enabled': True, 'effort': 'xhigh'}),
            config(), explicit_reasoning={'enabled': True, 'effort': 'xhigh'})
    conversation = [{'role': 'user', 'content': 'Continue existing work'}]
    agent.conversation_history = conversation
    index = agent._fallback_index
    assert not agent._try_activate_fallback(reason=FailoverReason.auth)
    assert agent._fallback_index == index
    client = MagicMock(base_url='https://example.invalid/v1', api_key='offline-test-key')
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(client, 'example/fallback-model')) as resolve:
        assert agent._try_activate_fallback(reason=FailoverReason.overloaded)
        assert resolve.call_count == 1
    assert agent.model == 'example/fallback-model'
    assert agent.reasoning_config == {'enabled': True, 'effort': 'xhigh'}
    assert agent.conversation_history is conversation
    assert agent._fallback_index == 1


def test_actual_api_factory_resolves_explicit_route_without_global_fallback(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from unittest.mock import MagicMock
    captured = []
    def construct(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(**kwargs)
    monkeypatch.setattr('run_agent.AIAgent', construct)
    monkeypatch.setattr('gateway.run._load_gateway_config', config)
    monkeypatch.setattr('gateway.run._resolve_runtime_agent_kwargs', lambda: pytest.fail('global auth route used'))
    monkeypatch.setattr('gateway.run._resolve_gateway_model', lambda: 'example/global-fallback')
    monkeypatch.setattr('gateway.run._checkpoint_agent_kwargs', lambda cfg: {})
    monkeypatch.setattr('gateway.run._current_max_iterations', lambda: 10)
    monkeypatch.setattr('gateway.run.GatewayRunner._load_fallback_model', lambda: pytest.fail('global fallback used'))
    monkeypatch.setattr('hermes_cli.tools_config._get_platform_tools', lambda *_: {'web'})
    resolved = {'provider': 'anthropic', 'api_key': 'offline-test', 'base_url': 'https://example.invalid', 'api_mode': 'anthropic_messages'}
    monkeypatch.setattr('gateway.platforms.api_server._resolve_request_runtime_agent_kwargs', lambda provider, target_model=None: dict(resolved))
    monkeypatch.setattr('gateway.run._resolve_runtime_agent_kwargs_for_provider', lambda provider, target_model=None: dict(resolved))
    monkeypatch.setattr('agent.agent_runtime_helpers.sync_credential_pool_entry_id', lambda agent: None)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, '_ensure_session_db', lambda: None)
    monkeypatch.setattr(adapter, '_session_model_override_for', lambda *_: None)
    adapter._memory_sessions = MagicMock()
    agent = adapter._create_agent(session_id='api-owner', requested_provider='anthropic',
                                  requested_model='example/primary-model',
                                  model_options={'reasoning': {'enabled': True, 'effort': 'xhigh'}})
    assert agent.provider == 'anthropic' and agent.model == 'example/primary-model'
    assert captured[0]['fallback_model'] is None
    assert agent._fallback_chain == [{'provider': 'openrouter', 'model': 'example/fallback-model'}]
    agent.model = 'example/fallback-model'
    reresolve_fallback_reasoning_config(agent)
    assert agent.reasoning_config == {'enabled': True, 'effort': 'xhigh'}
