"""Regression for #158726: an agent evicted by a new turn's cache lookup must not be released while
its own turn is still streaming.

``_lookup_cached_agent`` drops a cached agent (a changed message count, a new config signature) and
``_release_evicted_agent`` soft-released it at once. ``release_clients()`` closes the in-flight HTTP
stream, so the turn that still ran the agent saw a ReadError and retried. ``_evict_cached_agent``
already retires a mid-turn agent into ``state.turn.retired_agents``; the lookup path now does the
same, and ``_release_running_agent_state`` releases it when that turn ends.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.run import GatewayRunner
from gateway.session_execution import GatewaySessionAgentMixin

SESSION_KEY = "telegram:12345"


def _setup():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._sessions = {}
    runner._persist_active_agents = lambda: None
    spawned: list = []
    runner._spawn_release_thread = lambda target, args, name, **kw: spawned.append(args[0])
    mixin = GatewaySessionAgentMixin()
    mixin._runner = runner
    mixin._ctx = SimpleNamespace(session_key=SESSION_KEY)
    return runner, mixin, spawned


def test_an_idle_evicted_agent_is_released_at_once():
    _runner, mixin, spawned = _setup()
    agent = MagicMock()

    mixin._release_evicted_agent(agent)

    assert spawned == [agent]


def test_a_mid_turn_evicted_agent_is_not_released_mid_stream():
    runner, mixin, spawned = _setup()
    agent = MagicMock()
    state = runner._session_state(SESSION_KEY)
    state.turn.agent = agent

    mixin._release_evicted_agent(agent)

    assert spawned == []
    assert state.turn.retired_agents == [agent]


def test_the_retired_agent_is_released_when_its_turn_ends():
    runner, mixin, spawned = _setup()
    agent = MagicMock()
    runner._session_state(SESSION_KEY).turn.agent = agent
    mixin._release_evicted_agent(agent)
    assert spawned == []

    runner._release_running_agent_state(SESSION_KEY)

    assert spawned == [agent]


def test_a_mid_turn_agent_running_under_another_key_is_retired_there():
    # The cache key and the running slot can differ (a routing-key self-heal), so the owning
    # turn is found by the agent, not by this turn's session key.
    runner, mixin, spawned = _setup()
    agent = MagicMock()
    other = runner._session_state("telegram:other")
    other.turn.agent = agent

    mixin._release_evicted_agent(agent)

    assert spawned == []
    assert other.turn.retired_agents == [agent]
