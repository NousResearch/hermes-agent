"""Profile routing and repeated churn controls for signature retirement."""
from types import MethodType
from unittest.mock import Mock
import pytest
from gateway.run_agent_cache import GatewayAgentCacheMixin
from tests.gateway.test_signature_change_releases_memory import execution, ResourceAgent, no_source_updater


def test_displaced_release_preserves_route_owner():
    old = Mock()
    turn, runner, _ = execution(old)
    runner._spawn_release_thread = Mock()
    turn._release_evicted_agent(old)
    assert runner._spawn_release_thread.call_args.kwargs.get('session_key') == 'route'


def test_default_eviction_routes_release_to_session_owner():
    from tests.gateway.test_agent_cache import _make_runner
    runner = _make_runner()
    key = 'agent:main:telegram:dm:scope-test'
    agent = Mock()
    runner._agent_cache[key] = (agent, 'sig')
    runner._spawn_release_thread = Mock()
    runner._evict_cached_agent(key)
    assert runner._spawn_release_thread.call_args.kwargs.get('session_key') == key


@pytest.mark.parametrize('changes', [3, 10])
def test_real_successive_replacements_release_every_generation(monkeypatch, changes):
    monkeypatch.setattr('gateway.session_execution.api_runtime_locked', lambda: None)
    agents = [ResourceAgent() for _ in range(changes + 1)]
    try:
        turn, runner, released = execution(agents[0])
        for i in range(changes):
            runner._agent_config_signature = lambda *a, i=i, **kw: f'sig-{i}'
            turn._build_fresh_agent.return_value = agents[i+1]
            fresh, reused = turn._resolve_turn_agent({'model':'new','runtime':{}}, 'telegram', '', 20, None, {})
            assert fresh is agents[i+1] and not reused
            assert not agents[i].provider.writer.is_alive()
        assert released == agents[:-1]
        assert agents[-1].provider.writer.is_alive()
        assert runner._agent_cache['route'][0] is agents[-1]
    finally:
        for agent in agents:
            agent.cleanup()
