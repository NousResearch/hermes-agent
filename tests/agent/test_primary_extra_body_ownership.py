"""Primary runtime restoration must preserve model-scoped request provenance."""
from copy import deepcopy
import socket
from agent.agent_runtime_helpers import _apply_primary_runtime_fields, switch_model, try_recover_primary_transport
from types import SimpleNamespace
from pathlib import Path
import inspect
from unittest.mock import patch
import pytest
import hermes_yaml as yaml
from run_agent import AIAgent
from agent.chat_completion_helpers import build_api_kwargs

M = 'huihui-qwen3.8-27b-abliterated'
U = 'http://127.0.0.1:1234/v1'
E = dict(name='LM Studio LAN', provider='lmstudio', model=M, base_url=U, extra_body={'temperature': 0})

@pytest.mark.parametrize('primary', ['custom', 'lmstudio'])
@pytest.mark.parametrize('destination', ['lmstudio', 'openrouter'])
def test_primary_restore_preserves_config_ownership(tmp_path, monkeypatch, primary, destination):
    assert Path(inspect.getfile(AIAgent)).resolve() == Path.cwd() / 'run_agent.py'
    assert Path(inspect.getfile(build_api_kwargs)).resolve() == Path.cwd() / 'agent/chat_completion_helpers.py'
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(yaml.safe_dump({'custom_providers': [E]}))
    monkeypatch.setenv('HERMES_HOME', str(home))
    client = SimpleNamespace(base_url=U, api_key='test')
    with (patch('httpx.Client.send', side_effect=AssertionError('network forbidden')),
          # urllib probes connect through Hermes' happy-eyeballs racer, i.e. connect_ex.
          patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network forbidden')),
          patch('model_tools.get_tool_definitions', return_value=[]),
          patch('model_tools.check_toolset_requirements', return_value={}),
          patch('agent.process_bootstrap.OpenAI'),
          patch('agent.auxiliary_client.resolve_provider_client', side_effect=lambda provider, model, **kw: (client, model)),
          patch('agent.model_metadata.get_model_context_length', return_value=128_000),
          patch.object(AIAgent, '_ensure_lmstudio_runtime_loaded', return_value=None)):
        agent = AIAgent(model=M, provider=primary, api_key='test', base_url=U,
                        enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                        skip_memory=True, skip_background_review=True,
                        fallback_model=[{'provider': destination, 'model': 'OTHER', 'base_url': U}])
        assert agent.request_overrides == {'extra_body': {'temperature': 0}}
        # Ownership representation is implementation-specific; wire behavior is the invariant.
        assert agent._try_activate_fallback()
        first = build_api_kwargs(agent, [{'role': 'user', 'content': 'offline'}], [])
        assert 'temperature' not in first.get('extra_body', {})
        with patch('agent.agent_runtime_helpers._rebuild_primary_client'):
            assert agent._restore_primary_runtime()
        assert agent.request_overrides == {'extra_body': {'temperature': 0}}
        print('restored ledger:', getattr(agent, '_custom_provider_injected_extra_body', '<not recorded>'))
        assert agent._try_activate_fallback()
        second = build_api_kwargs(agent, [{'role': 'user', 'content': 'offline'}], [])
        print('first:', first.get('extra_body'), 'second:', second.get('extra_body'))
        assert 'temperature' not in second.get('extra_body', {}), 'primary config-owned temperature leaked after turn-start restore'


MSGS = [{'role': 'user', 'content': 'offline'}]


def wire(agent):
    return build_api_kwargs(agent, MSGS, []).get('extra_body') or {}


@pytest.fixture
def agent_factory(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    client = SimpleNamespace(base_url=U, api_key='test')
    with (patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')),
          # Hermes' happy-eyeballs racer connects via connect_ex, which the line above misses.
          patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network forbidden')),
          patch('model_tools.get_tool_definitions', return_value=[]),
          patch('model_tools.check_toolset_requirements', return_value={}),
          patch('agent.process_bootstrap.OpenAI'),
          patch('agent.auxiliary_client.resolve_provider_client', side_effect=lambda provider, model, **kw: (client, model)),
          patch('agent.model_metadata.get_model_context_length', return_value=128_000),
          patch('agent.agent_runtime_helpers._rebuild_primary_client'),
          patch('agent.agent_runtime_helpers.time.sleep'),
          patch.object(AIAgent, '_ensure_lmstudio_runtime_loaded', return_value=None),
          patch('agent.credential_pool.load_pool', return_value=None)):
        def make(entries=(E,), primary='lmstudio', destination='custom', body=None, config=None):
            """``body`` is the constructor's extra_body, the slot runtime resolution fills."""
            (home / 'config.yaml').write_text(yaml.safe_dump(config or {'custom_providers': list(entries)}))
            overrides = {'seed': 7, **({'extra_body': deepcopy(body)} if body else {})}
            return AIAgent(model=M, provider=primary, api_key='test', base_url=U,
                           enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                           skip_memory=True, skip_background_review=True, request_overrides=overrides,
                           fallback_model=[{'provider': destination, 'model': 'OTHER', 'base_url': U}])
        yield make


@pytest.mark.parametrize('primary', ['lmstudio', 'custom'])
@pytest.mark.parametrize('destination', ['lmstudio', 'openrouter'])
def test_repeated_restore_and_direct_switch(agent_factory, primary, destination):
    agent = agent_factory(primary=primary, destination=destination)
    for _ in range(3):
        assert wire(agent).get('temperature') == 0
        assert agent._custom_provider_injected_extra_body == {'temperature': 0}
        assert agent._try_activate_fallback()
        assert 'temperature' not in wire(agent)
        assert agent._restore_primary_runtime()
    assert agent.request_overrides == {'extra_body': {'temperature': 0}, 'seed': 7}
    switch_model(agent, 'OTHER', destination, api_key='test', base_url=U, api_mode='chat_completions')
    assert 'temperature' not in wire(agent)
    # Deliberate switches must snapshot the destination, not pre-rescope primary overrides.
    _apply_primary_runtime_fields(agent, agent._primary_runtime)
    assert 'temperature' not in wire(agent)
    assert agent.request_overrides == {'seed': 7}


@pytest.mark.parametrize('provenance', ['constructor', 'after_init'])
def test_switch_snapshot_carries_only_caller_owned_values(agent_factory, provenance):
    """The one way an old route's value outlives a /model switch: upstream's switch snapshot kept the
    whole pre-switch body for the next restore; here only its positively caller-owned part."""
    second = {**E, 'model': 'SECOND', 'extra_body': {'temperature': 0.8, 'top_k': 5}}
    caller = {'caller': {'nested': [1]}}
    agent = agent_factory(entries=(E, second), body=caller if provenance == 'constructor' else None)
    if provenance == 'after_init':
        agent.request_overrides['extra_body'] = {**agent.request_overrides['extra_body'], **caller}
    switch_model(agent, 'SECOND', 'lmstudio', api_key='test', base_url=U, api_mode='chat_completions')
    assert agent.request_overrides == {'extra_body': second['extra_body'], 'seed': 7}  # live: upstream's switch
    assert agent._try_activate_fallback()
    assert not {'temperature', 'top_k', 'caller'} & wire(agent).keys()
    carried = caller if provenance == 'after_init' else {}
    for _ in range(3):
        assert agent._restore_primary_runtime()
        assert agent.request_overrides == {'extra_body': {**second['extra_body'], **carried}, 'seed': 7}
        assert agent._try_activate_fallback()
        assert wire(agent).get('caller') == carried.get('caller')
        assert not {'temperature', 'top_k'} & wire(agent).keys()


def test_constructor_body_without_config_extra_body_survives_like_upstream(agent_factory):
    body = {'reasoning': {'effort': 'high'}}
    agent = agent_factory(primary='custom', body=body, config={'model': {'default': M}})
    for _ in range(3):
        assert agent._try_activate_fallback()
        assert wire(agent).get('reasoning') == body['reasoning']
        assert agent._restore_primary_runtime()
    switch_model(agent, 'THIRD', 'custom', api_key='test', base_url=U, api_mode='chat_completions')
    assert agent.request_overrides == {'seed': 7}
    assert agent._try_activate_fallback()
    assert agent._restore_primary_runtime()
    assert agent.request_overrides == {'extra_body': body, 'seed': 7}


def test_primary_recovery_restores_body_and_ownership(agent_factory):
    agent = agent_factory()
    agent.request_overrides = {}
    agent._custom_provider_injected_extra_body = {}
    assert try_recover_primary_transport(agent, type('ReadTimeout', (Exception,), {})(), retry_count=1, max_retries=1)
    assert agent.request_overrides == {'extra_body': {'temperature': 0}, 'seed': 7}
    assert agent._custom_provider_injected_extra_body == {'temperature': 0}
    assert agent._try_activate_fallback()
    assert 'temperature' not in wire(agent)


@pytest.mark.parametrize('snapshot', ['key_missing', 'ledger_none', 'captured_without_ledger'])
def test_restore_from_a_snapshot_without_ledger_fails_closed(agent_factory, snapshot):
    agent = agent_factory(primary='custom', config={'model': {'default': M}})
    if snapshot == 'captured_without_ledger':
        from agent.agent_runtime_helpers import _build_primary_runtime_snapshot
        agent.request_overrides = {'extra_body': {'unknown': 1}}
        del agent._custom_provider_injected_extra_body
        rt = _build_primary_runtime_snapshot(agent, agent.api_mode)
    else:
        rt = deepcopy(agent._primary_runtime)
        rt['request_overrides'] = {'extra_body': {'unknown': 1}}
        if snapshot == 'key_missing':
            rt.pop('custom_provider_injected_extra_body', None)
        else:
            rt['custom_provider_injected_extra_body'] = None
    _apply_primary_runtime_fields(agent, rt)
    assert agent.request_overrides['extra_body'] == {'unknown': 1}
    assert agent._try_activate_fallback()
    assert 'unknown' not in wire(agent)


def test_nested_mutation_does_not_corrupt_registration_ledger_or_snapshot(agent_factory):
    entry = {**E, 'extra_body': {'temperature': 0, 'chat_template_kwargs': {'enable_thinking': True}}}
    agent = agent_factory(entries=(entry,))
    state = lambda: deepcopy((agent._custom_providers, agent._custom_provider_injected_extra_body, agent._primary_runtime))
    before = state()
    agent.request_overrides['extra_body']['chat_template_kwargs']['enable_thinking'] = False
    assert state() == before
    # Rewritten by the caller, the nested value is no longer the registration's and moves on.
    assert agent._try_activate_fallback()
    assert wire(agent).get('chat_template_kwargs') == {'enable_thinking': False}
    assert 'temperature' not in wire(agent)
    assert agent._restore_primary_runtime()
    assert agent.request_overrides['extra_body'] == entry['extra_body']
    # Partly the registration's, partly the caller's: ambiguous, so it stays behind.
    agent.request_overrides['extra_body']['chat_template_kwargs']['mine'] = 1
    assert state() == before
    assert agent._try_activate_fallback()
    assert 'chat_template_kwargs' not in wire(agent)


def test_voice_route_restores_body_and_ownership(agent_factory):
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    agent = agent_factory(destination='openrouter', config={
        'custom_providers': [E], 'auxiliary': {'voice_chat': {'provider': 'zai', 'model': 'glm-voice'}}})
    agent.request_overrides['extra_body'] = {**agent.request_overrides['extra_body'], 'mine': 1}
    agent._voice_turn_pending = True
    with patch('agent.chat_completion_helpers.rewrite_prompt_model_identity'):
        begin_voice_turn_route(agent, MSGS, 'sys')
        assert (agent.provider, agent.model) == ('zai', 'glm-voice')
        assert 'temperature' not in wire(agent) and wire(agent).get('mine') == 1
        end_voice_turn_route(agent)
    assert (agent.provider, agent.model) == ('lmstudio', M)
    assert wire(agent).get('temperature') == 0 and wire(agent).get('mine') == 1
    assert agent._custom_provider_injected_extra_body == {'temperature': 0}
    assert agent._try_activate_fallback()
    assert 'temperature' not in wire(agent) and wire(agent).get('mine') == 1
