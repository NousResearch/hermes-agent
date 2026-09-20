"""Interactive runs add clarify only to defaults, never to restricted tool lists."""
import copy
import pytest
from gateway.platforms.api_server_clarify import interactive_run_toolsets

@pytest.mark.parametrize('config,expected', [
    ({}, True),
    ({'platform_toolsets': {'telegram': ['terminal']}}, True),
    ({'platform_toolsets': {'api_server': []}}, False),
    ({'platform_toolsets': {'api_server': ['terminal']}}, False),
    ({'platform_toolsets': {'api_server': ['terminal', 'clarify']}}, True),
    ({'agent': {'disabled_toolsets': ['clarify']}}, False),
    ({'agent': {'disabled_toolsets': '["clarify"]'}}, False),
])
def test_respects_defaults_explicit_lists_and_global_disable(config, expected):
    before = copy.deepcopy(config)
    tools = interactive_run_toolsets(config)
    assert ('clarify' in tools) == expected
    assert config == before

@pytest.mark.parametrize('interactive,room,expected', [
    (False, False, False), (True, False, True), (True, True, False),
])
def test_agent_constructor_receives_tools_and_room_policy_wins(monkeypatch, interactive, room, expected):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    captured = {}
    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr('run_agent.AIAgent', FakeAgent)
    monkeypatch.setattr('gateway.run._resolve_runtime_agent_kwargs', lambda: {'provider': 'openai'})
    monkeypatch.setattr('gateway.run._resolve_gateway_model', lambda: 'test-model')
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {})
    monkeypatch.setattr('gateway.run.GatewayRunner._load_reasoning_config', staticmethod(lambda model='': {}))
    monkeypatch.setattr('gateway.run.GatewayRunner._load_fallback_model', staticmethod(lambda: None))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, '_ensure_session_db', lambda: None)
    from gateway.hosted_room_execution_policy import execution_policy_mapping
    policy = execution_policy_mapping(target_profile='default', config={
        'platform_toolsets': {'api_server': ['terminal']}}) if room else None
    try:
        adapter._create_agent(session_id='toolset-test', interactive_run=interactive,
                              room_dispatch={} if room else None,
                              room_execution_policy=policy)
        assert ('clarify' in captured['enabled_toolsets']) == expected
    finally:
        adapter._run_idempotency_store.close()
        adapter._close_cached_session_dbs()


def test_legacy_opt_in_is_not_removed():
    from hermes_cli.tools_config import _get_platform_tools
    config = {'toolsets': ['kanban']}
    baseline = _get_platform_tools(config, 'api_server')
    assert 'kanban' in baseline
    assert interactive_run_toolsets(config) == baseline | {'clarify'}


def test_noninteractive_default_still_has_no_clarify():
    from hermes_cli.tools_config import _get_platform_tools
    baseline = _get_platform_tools({}, 'api_server')
    assert 'clarify' not in baseline
    assert interactive_run_toolsets({}) == baseline | {'clarify'}
