"""Cache maintenance must respect a currently supervised native Goal, not pin idle goals forever."""
from collections import OrderedDict
import threading
from types import SimpleNamespace

from gateway.agent_cache_pressure import AgentCacheBounds
from gateway.run import GatewayRunner


def test_idle_and_lru_eviction_protect_native_consumer_until_it_releases():
    runner = GatewayRunner.__new__(GatewayRunner)
    session = SimpleNamespace(_native_goal_running=True)
    agent = SimpleNamespace(_codex_session=session, _last_activity_ts=0)
    other = SimpleNamespace(_last_activity_ts=0)
    runner._agent_cache = OrderedDict([('native', (agent, 'sig')), ('idle', (other, 'sig'))])
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents = {}
    runner._agent_cache_bounds_cache = AgentCacheBounds(max_size=1, idle_ttl_secs=1)
    released = []
    runner._spawn_release_thread = lambda fn, args, name, **kwargs: released.append(args[0])
    runner._enforce_agent_cache_cap()
    assert 'native' in runner._agent_cache and released == []
    assert runner._sweep_idle_cached_agents() == 1
    assert list(runner._agent_cache) == ['native'] and released == [other]
    session._native_goal_running = False
    assert runner._sweep_idle_cached_agents() == 1
    assert not runner._agent_cache and released == [other, agent]
