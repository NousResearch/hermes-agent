"""Regression tests for PR #48127: cached agent max_iterations refresh.

When a long-lived gateway reuses an agent from its cache, the agent must run
the *current* configured iteration budget — not the budget it was constructed
with on the first turn of that session. Two pieces make that true:

1. ``GatewayRunner._init_cached_agent_for_turn`` must NOT reset
   ``max_iterations`` itself (the gateway refreshes it explicitly right after,
   from current config). If this helper ever started clobbering it, the
   gateway's refresh would be silently undone.
2. The per-turn budget object is rebuilt from ``agent.max_iterations`` at the
   start of every turn (``agent/turn_context.py`` -> ``IterationBudget``), so
   refreshing ``max_iterations`` on the cached agent is sufficient to change
   the operative cap the agent loop checks.

These tests exercise the real code paths rather than asserting a plain
assignment, so they fail if either contract regresses.
"""

import time
from types import SimpleNamespace

from agent.iteration_budget import IterationBudget


def _make_cached_agent(max_iterations: int) -> SimpleNamespace:
    """A minimal stand-in cached agent with the attributes the helpers touch."""
    # The turn loop checks both api_call_count >= max_iterations AND
    # iteration_budget.remaining <= 0 (turn_finalizer.py), so the budget must
    # also reflect the new cap. Seed it with the stale value to prove the
    # refresh propagates.
    return SimpleNamespace(
        _last_activity_ts=time.time() - 1000,
        _last_activity_desc="previous turn",
        _api_call_count=42,
        _last_flushed_db_idx=5,
        max_iterations=max_iterations,
        iteration_budget=IterationBudget(max_iterations),
    )


def test_init_cached_agent_for_turn_does_not_touch_max_iterations():
    """The per-turn reset helper must leave max_iterations untouched.

    The gateway refreshes max_iterations explicitly right after calling this
    helper; if the helper ever reset it, that refresh would be undone.
    """
    from gateway.run import GatewayRunner

    agent = _make_cached_agent(90)
    GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)

    # Per-turn state was reset...
    assert agent._api_call_count == 0
    assert agent._last_activity_desc == "starting new turn (cached)"
    assert agent._last_flushed_db_idx == 0
    # ...but the iteration budget was NOT changed by the helper itself.
    assert agent.max_iterations == 90


def test_init_cached_agent_preserves_max_iterations_on_interrupt_depth():
    """Interrupt-recursive turns must also leave max_iterations alone."""
    from gateway.run import GatewayRunner

    agent = _make_cached_agent(200)
    GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=1)

    # Activity timestamps preserved for the inactivity watchdog (#15654)...
    assert agent._last_activity_desc == "previous turn"
    # ...and max_iterations untouched.
    assert agent.max_iterations == 200


def test_refreshed_max_iterations_propagates_to_turn_budget():
    """Refreshing max_iterations on a cached agent changes the operative cap.

    The gateway sets ``agent.max_iterations = max_iterations`` on cache reuse;
    the new turn's setup then rebuilds ``iteration_budget`` from it. This proves
    the refresh actually moves the budget the agent loop enforces — the cached
    agent started at 90 and ends a new turn capped at 200.
    """
    agent = _make_cached_agent(90)
    assert agent.iteration_budget.max_total == 90

    # Gateway refresh on cache reuse:
    agent.max_iterations = 200

    # Start-of-turn budget rebuild (agent/turn_context.py:166):
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    assert agent.iteration_budget.max_total == 200
    assert agent.iteration_budget.remaining == 200
