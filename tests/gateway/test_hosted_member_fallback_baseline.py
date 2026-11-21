"""A hosted room member switched by a profile edit keeps its cached agent after a successful turn.

The member resolves its route from the owning profile only inside the turn's policy scope; the
post-turn fallback check runs after that scope exits. Reading the frozen model there made the new
route look like a provider fallback, so every reply after a switch rebuilt the agent.
"""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from persistence_assembly import profile_scope


def _write_profile(home, model):
    (home / 'config.yaml').write_text(json.dumps({'model': {
        'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1', 'default': model}}), encoding='utf-8')


def test_switched_room_member_is_not_evicted_as_a_fallback(tmp_path, monkeypatch):
    from gateway import session_policy
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    frozen = {'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1', 'default': 'old-model'}}
    policy = replace(session_policy.build_policy({'source': 'gui', 'cwd': str(tmp_path), 'model': 'old-model'}, frozen),
                     source='bot_room', platform='bot_room')
    _write_profile(tmp_path, 'new-model')
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._session_model_overrides = {}
    runner.session_authority = SimpleNamespace(persistence=SimpleNamespace(scope=profile_scope(home=tmp_path)))  # type: ignore[attr-defined]
    runner._evict_cached_agent = MagicMock()
    source = SessionSource(platform=Platform.LOCAL, user_id='u', chat_id='member', chat_type='dm')
    ctx = TurnContext(source=source, session_key='agent:main:local:dm:member')
    ctx.agent_holder[0] = SimpleNamespace(model='new-model', provider='custom')
    ctx.result_holder[0] = {'final_response': 'ok'}

    runner._run_agent_evict_on_fallback(ctx)
    assert not runner._evict_cached_agent.called

    # A real fallback off the profile's current model still evicts.
    ctx.agent_holder[0] = SimpleNamespace(model='fallback-model', provider='custom')
    runner._run_agent_evict_on_fallback(ctx)
    assert runner._evict_cached_agent.called
    runner._evict_cached_agent.assert_called_once_with(
        'agent:main:local:dm:member', release_after_turn=True
    )


def test_fallback_evicted_running_agent_is_released_when_turn_slot_clears():
    """Issue #158456: ``_run_agent_evict_on_fallback`` runs while the agent still holds its
    ``_running_agents`` slot, so ``_evict_cached_agent`` defers teardown to ``retired_agents`` and
    ``_release_running_agent_state`` soft-releases the agent (and its memory provider writer thread)
    when the turn ends."""
    from collections import OrderedDict
    import threading
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext
    from tests.waiting import wait_until

    route = 'agent:main:local:dm:fallback'
    released = threading.Event()
    agent = MagicMock()
    agent.model = 'fallback-model'
    agent.provider = 'custom'
    agent._provider_fallback_active = True
    agent.release_clients.side_effect = released.set

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._agent_cache = OrderedDict({route: (agent, 'sig-fb', 0, 's1')})
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents[route] = agent
    runner._persist_active_agents = lambda: None

    source = SessionSource(platform=Platform.LOCAL, user_id='u', chat_id='fallback', chat_type='dm')
    ctx = TurnContext(source=source, session_key=route)
    ctx.agent_holder[0] = agent
    ctx.result_holder[0] = {'final_response': 'ok'}

    runner._run_agent_evict_on_fallback(ctx)

    # Evicted from the cache immediately, but not torn down while the turn slot is still active.
    assert route not in runner._agent_cache
    assert not agent.release_clients.called

    # Clearing the running-agent slot at turn end triggers the deferred soft-release.
    assert runner._release_running_agent_state(route) is True
    assert wait_until(released.is_set, timeout=5, what='fallback-evicted agent was soft-released')
    assert agent.release_clients.call_count == 1


def test_in_turn_evict_cached_agent_defers_release_by_default():
    """Issue #158456: any in-turn ``_evict_cached_agent(session_key)`` call while the evicted agent
    still holds its running slot retires the agent for post-turn soft-release by default (without
    duplicating entries), while ``release_after_turn=False`` still opts out."""
    from collections import OrderedDict
    import threading
    from gateway.run import GatewayRunner
    from tests.waiting import wait_until

    route = 'agent:main:local:dm:in-turn'
    released = threading.Event()
    agent = MagicMock()
    agent.release_clients.side_effect = released.set

    runner = object.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict({route: (agent, 'sig-1', 0, 's1')})
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents[route] = agent
    runner._persist_active_agents = lambda: None

    # Opt-out path: release_after_turn=False pops the cache without scheduling post-turn release.
    runner._evict_cached_agent(route, release_after_turn=False)
    assert route not in runner._agent_cache
    assert runner._session_state(route).turn.retired_agents == []

    # Default path: evicting a running agent schedules it once in retired_agents and releases on slot clear.
    runner._agent_cache[route] = (agent, 'sig-1', 0, 's1')
    runner._evict_cached_agent(route)
    runner._agent_cache[route] = (agent, 'sig-1', 0, 's1')
    runner._evict_cached_agent(route)
    assert runner._session_state(route).turn.retired_agents == [agent]
    assert not agent.release_clients.called

    assert runner._release_running_agent_state(route) is True
    assert wait_until(released.is_set, timeout=5, what='in-turn evicted agent was soft-released')
    assert agent.release_clients.call_count == 1


