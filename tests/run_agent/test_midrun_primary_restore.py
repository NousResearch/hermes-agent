"""Tests for periodic primary-runtime restoration during fallback turns."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.conversation_loop import _maybe_restore_primary_midrun
from agent.verify_hooks import (
    DEFAULT_MIDRUN_PRIMARY_RESTORE_INTERVAL,
    midrun_primary_restore_interval,
)


def _fallback_agent(*, fallback_iterations: int = 0) -> SimpleNamespace:
    agent = SimpleNamespace(
        _fallback_activated=True,
        _iters_since_fallback=fallback_iterations,
        model="fallback-model",
        provider="fallback-provider",
        quiet_mode=False,
    )
    agent._restore_primary_runtime = MagicMock()
    agent._buffer_status = MagicMock()
    return agent


def test_restores_primary_at_interval_and_resets_counter():
    agent = _fallback_agent(fallback_iterations=1)

    def restore_primary() -> bool:
        agent.model = "primary-model"
        agent.provider = "primary-provider"
        return True

    agent._restore_primary_runtime.side_effect = restore_primary

    assert _maybe_restore_primary_midrun(agent, interval=2) is True
    agent._restore_primary_runtime.assert_called_once_with()
    assert agent._iters_since_fallback == 0
    agent._buffer_status.assert_called_once_with(
        "🔄 Primary model restored: primary-model (primary-provider)"
    )


def test_cooldown_gated_restore_retries_at_next_interval():
    agent = _fallback_agent(fallback_iterations=1)
    # restore_primary_runtime owns the cooldown gate and returns False while it
    # remains armed.
    agent._restore_primary_runtime.return_value = False

    assert _maybe_restore_primary_midrun(agent, interval=2) is False
    assert agent._iters_since_fallback == 0
    agent._restore_primary_runtime.assert_called_once_with()

    assert _maybe_restore_primary_midrun(agent, interval=2) is False
    agent._restore_primary_runtime.assert_called_once_with()
    assert agent._iters_since_fallback == 1

    assert _maybe_restore_primary_midrun(agent, interval=2) is False
    assert agent._iters_since_fallback == 0
    assert agent._restore_primary_runtime.call_count == 2
    agent._buffer_status.assert_not_called()


def test_zero_interval_disables_midrun_restore():
    agent = _fallback_agent(fallback_iterations=4)

    assert _maybe_restore_primary_midrun(agent, interval=0) is False
    agent._restore_primary_runtime.assert_not_called()
    assert agent._iters_since_fallback == 4


def test_no_fallback_does_not_restore_or_grow_counter():
    agent = _fallback_agent(fallback_iterations=3)
    agent._fallback_activated = False

    assert _maybe_restore_primary_midrun(agent, interval=1) is False
    agent._restore_primary_runtime.assert_not_called()
    assert agent._iters_since_fallback == 3


def test_interval_config_defaults_coerces_and_allows_disable():
    assert midrun_primary_restore_interval({}) == DEFAULT_MIDRUN_PRIMARY_RESTORE_INTERVAL
    assert midrun_primary_restore_interval(
        {"agent": {"midrun_primary_restore_interval": "2"}}
    ) == 2
    assert midrun_primary_restore_interval(
        {"agent": {"midrun_primary_restore_interval": 0}}
    ) == 0
    assert midrun_primary_restore_interval(
        {"agent": {"midrun_primary_restore_interval": -1}}
    ) == 0
