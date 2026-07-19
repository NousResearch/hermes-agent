"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

Unit tests for gateway agent lifecycle runtime helpers.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.agent_lifecycle_runtime_service import (
    GatewayAgentRuntimeTasks,
    cleanup_gateway_agent_runtime_tasks,
    mark_gateway_streaming_delivery_state,
    resolve_gateway_effective_model_state,
    wait_for_gateway_agent_result,
)

class _FakeActivityAgent:
    def __init__(
        self,
        *,
        idle_seconds: float,
        last_activity_desc: str = "api_call_streaming",
        current_tool: str | None = None,
        api_call_count: int = 0,
        max_iterations: int = 0,
    ) -> None:
        self._activity = {
            "seconds_since_activity": idle_seconds,
            "last_activity_desc": last_activity_desc,
            "current_tool": current_tool,
            "api_call_count": api_call_count,
            "max_iterations": max_iterations,
        }
        self.interrupt = MagicMock()

    def get_activity_summary(self):
        return dict(self._activity)

@pytest.mark.asyncio
async def test_wait_for_gateway_agent_result_returns_executor_response(monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "10")
    agent = _FakeActivityAgent(idle_seconds=0.0)

    result = await wait_for_gateway_agent_result(
        run_sync=lambda: {"final_response": "done", "messages": [], "api_calls": 1},
        agent_holder=[agent],
        result_holder=[None],
        tools_holder=[[]],
        session_key="session-1",
        logger=MagicMock(),
        poll_interval=0.01,
    )

    assert result["final_response"] == "done"
    agent.interrupt.assert_not_called()

@pytest.mark.asyncio
async def test_wait_for_gateway_agent_result_builds_timeout_response(monkeypatch):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0.1")
    agent = _FakeActivityAgent(
        idle_seconds=0.2,
        current_tool="web_search",
        api_call_count=3,
        max_iterations=50,
    )

    def _run_sync():
        time.sleep(0.3)
        return {"final_response": "late", "messages": [], "api_calls": 1}

    result = await wait_for_gateway_agent_result(
        run_sync=_run_sync,
        agent_holder=[agent],
        result_holder=[None],
        tools_holder=[["web_search"]],
        session_key="session-timeout",
        logger=MagicMock(),
        poll_interval=0.01,
    )

    assert result["failed"] is True
    assert "web_search" in result["final_response"]
    assert result["api_calls"] == 3
    agent.interrupt.assert_called_once_with("Execution timed out (inactivity)")

@pytest.mark.asyncio
async def test_cleanup_gateway_agent_runtime_tasks_clears_tracking():
    async def _sleep_forever():
        await asyncio.sleep(3600)

    tasks = GatewayAgentRuntimeTasks(
        progress_task=asyncio.create_task(_sleep_forever()),
        stream_task=asyncio.create_task(asyncio.sleep(0)),
        tracking_task=asyncio.create_task(_sleep_forever()),
        interrupt_monitor_task=asyncio.create_task(_sleep_forever()),
        long_running_notify_task=asyncio.create_task(_sleep_forever()),
    )
    running_agents = {"session-1": object()}
    running_agents_ts = {"session-1": 123.0}

    await cleanup_gateway_agent_runtime_tasks(
        tasks=tasks,
        session_key="session-1",
        running_agents=running_agents,
        running_agents_ts=running_agents_ts,
    )

    assert "session-1" not in running_agents
    assert "session-1" not in running_agents_ts
    assert tasks.progress_task.cancelled()
    assert tasks.tracking_task.cancelled()
    assert tasks.interrupt_monitor_task.cancelled()
    assert tasks.long_running_notify_task.cancelled()

def test_mark_gateway_streaming_delivery_state_sets_already_sent():
    response = {"final_response": "done"}

    marked = mark_gateway_streaming_delivery_state(
        response=response,
        stream_consumer=SimpleNamespace(already_sent=True),
    )

    assert marked["already_sent"] is True

def test_resolve_gateway_effective_model_state_tracks_fallback_and_eviction():
    agent = SimpleNamespace(model="gpt-fallback", provider="custom")

    state = resolve_gateway_effective_model_state(
        agent=agent,
        configured_model="gpt-primary",
        should_evict_cached_agent_after_turn=lambda current, configured: (
            current is agent and configured == "gpt-primary"
        ),
    )

    assert state.effective_model == "gpt-fallback"
    assert state.effective_provider == "custom"
    assert state.should_evict_cached_agent is True

def test_resolve_gateway_effective_model_state_clears_when_primary_model_used():
    agent = SimpleNamespace(model="gpt-primary", provider="custom")

    state = resolve_gateway_effective_model_state(
        agent=agent,
        configured_model="gpt-primary",
        should_evict_cached_agent_after_turn=lambda *_: True,
    )

    assert state.effective_model is None
    assert state.effective_provider is None
    assert state.should_evict_cached_agent is False
