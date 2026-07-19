"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

Tests for gateway stale-running-agent eviction helpers.
"""
from __future__ import annotations
import pytest

pytestmark = pytest.mark.dead_runtime_service


from unittest.mock import MagicMock

from gateway.running_agent_staleness_runtime_service import (
    evict_stale_gateway_running_agent,
)


class _FakeActivityAgent:
    def __init__(
        self,
        *,
        idle_seconds: float,
        last_activity_desc: str = "api_call_streaming",
        api_call_count: int = 0,
        max_iterations: int = 0,
    ) -> None:
        self._activity = {
            "seconds_since_activity": idle_seconds,
            "last_activity_desc": last_activity_desc,
            "api_call_count": api_call_count,
            "max_iterations": max_iterations,
        }

    def get_activity_summary(self):
        return dict(self._activity)


def test_evicts_agent_idle_past_timeout():
    logger = MagicMock()
    running_agents = {
        "session-1": _FakeActivityAgent(
            idle_seconds=45.0,
            last_activity_desc="delegate_task",
            api_call_count=4,
            max_iterations=60,
        )
    }
    running_agents_ts = {"session-1": 955.0}

    evicted = evict_stale_gateway_running_agent(
        session_key="session-1",
        running_agents=running_agents,
        running_agents_ts=running_agents_ts,
        pending_agent_sentinel=object(),
        logger=logger,
        stale_timeout_seconds=30.0,
        now=1000.0,
    )

    assert evicted is True
    assert "session-1" not in running_agents
    assert "session-1" not in running_agents_ts
    logger.warning.assert_called_once()


def test_does_not_evict_pending_sentinel_even_when_age_is_high():
    sentinel = object()
    logger = MagicMock()
    running_agents = {"session-1": sentinel}
    running_agents_ts = {"session-1": 0.5}

    evicted = evict_stale_gateway_running_agent(
        session_key="session-1",
        running_agents=running_agents,
        running_agents_ts=running_agents_ts,
        pending_agent_sentinel=sentinel,
        logger=logger,
        stale_timeout_seconds=30.0,
        now=10000.0,
    )

    assert evicted is False
    assert running_agents["session-1"] is sentinel
    assert running_agents_ts["session-1"] == 0.5
    logger.warning.assert_not_called()


def test_does_not_evict_active_agent_below_idle_and_wall_thresholds():
    logger = MagicMock()
    agent = _FakeActivityAgent(idle_seconds=5.0, last_activity_desc="streaming")
    running_agents = {"session-1": agent}
    running_agents_ts = {"session-1": 950.0}

    evicted = evict_stale_gateway_running_agent(
        session_key="session-1",
        running_agents=running_agents,
        running_agents_ts=running_agents_ts,
        pending_agent_sentinel=object(),
        logger=logger,
        stale_timeout_seconds=30.0,
        now=1000.0,
    )

    assert evicted is False
    assert running_agents["session-1"] is agent
    assert running_agents_ts["session-1"] == 950.0
    logger.warning.assert_not_called()


def test_evicts_ancient_entry_without_activity_summary():
    logger = MagicMock()
    running_agents = {"session-1": object()}
    running_agents_ts = {"session-1": 1.0}

    evicted = evict_stale_gateway_running_agent(
        session_key="session-1",
        running_agents=running_agents,
        running_agents_ts=running_agents_ts,
        pending_agent_sentinel=object(),
        logger=logger,
        stale_timeout_seconds=30.0,
        now=8000.0,
    )

    assert evicted is True
    assert "session-1" not in running_agents
    assert "session-1" not in running_agents_ts
    logger.warning.assert_called_once()
