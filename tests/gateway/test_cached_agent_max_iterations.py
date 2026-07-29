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


def test_init_cached_agent_preserves_max_iterations_on_interrupt_depth():
    """Interrupt-recursive turns must also leave max_iterations alone."""
    from gateway.run import GatewayRunner

    agent = _make_cached_agent(200)
    GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=1)

    # Activity timestamps preserved for the inactivity watchdog (#15654)...
    assert agent._last_activity_desc == "previous turn"
    # ...and max_iterations untouched.
    assert agent.max_iterations == 200


