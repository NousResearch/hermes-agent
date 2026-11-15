"""Transport-failure failover patience (``agent.transport_failover_retries``).

A connect/read timeout or an overloaded provider classifies as a *transport* failure. Stock code
switched to the fallback chain once that happened twice (``retry_count >= 2``) while the retry
backoff is 2s -> 4s -> 8s ..., so roughly seven seconds of network blip was enough to finish the
turn on another model: the whole prompt is re-read uncached and the model changes mid-task.

These tests pin the configurable budget:

  1. ``transport_failover_retries`` (default 2) is what gates the eager fallback.
  2. A raised budget also lifts this turn's retry ceiling, so the loop can reach it instead of
     ending the turn on the last retry.
  3. The default keeps the historical behaviour (switch on the second transport failure).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent import turn_recovery
from agent.error_classifier import FailoverReason, classify_api_error
from run_agent import AIAgent


def _conn_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid/v1/chat"))


def _make_agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "openai", "model": "gpt-4o"}],
        )
        agent.client = MagicMock()
        # The decision under test is "does the loop walk the chain now", not whether the chain's
        # entries happen to be configured in this process.
        agent._try_activate_fallback = MagicMock(return_value=True)
        return agent


def _route(agent, retry_count: int, max_retries: int = 3):
    error = _conn_error()
    classified = classify_api_error(error, provider=agent.provider, model=agent.model)
    return turn_recovery.route_classified_error(
        agent, error, classified, MagicMock(),
        error_msg=str(error), error_context={}, recovered_with_pool=False,
        base_url=agent.base_url, model=agent.model, messages=[], api_messages=[],
        system_message="", active_system_prompt="", conversation_history=[],
        retry_count=retry_count, max_retries=max_retries, compression_attempts=0,
        max_compression_attempts=2, api_call_count=1, effective_task_id=None,
    ), classified


def test_connection_error_is_a_transport_failure():
    assert classify_api_error(_conn_error()).reason is FailoverReason.timeout


@pytest.mark.parametrize("value,expected", [(None, 2), (7, 7), ("5", 5), (0, 0), (-3, 0), ("junk", 2)])
def test_budget_reads_config(monkeypatch, value, expected):
    cfg = {} if value is None else {"agent": {"transport_failover_retries": value}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    assert turn_recovery.transport_failover_retry_budget() == expected


def test_default_budget_switches_on_second_transport_failure():
    """2 = historical behaviour: the second transport failure walks the fallback chain."""
    agent = _make_agent()
    with patch.object(turn_recovery, "transport_failover_retry_budget", lambda: 2):
        first, _ = _route(agent, retry_count=1)
        second, _ = _route(agent, retry_count=2)
    assert first.action == "fallthrough"
    assert second.action == "break"


def test_raised_budget_keeps_retrying_on_the_primary():
    """With patience configured, an early blip stays on the primary and the retry ceiling rises."""
    agent = _make_agent()
    with patch.object(turn_recovery, "transport_failover_retry_budget", lambda: 7):
        early, _ = _route(agent, retry_count=2, max_retries=3)
        late, _ = _route(agent, retry_count=7, max_retries=3)
    assert early.action == "fallthrough", "a 7-attempt budget must not switch on failure #2"
    assert early.max_retries >= 8, "the loop has to be able to reach the configured budget"
    assert late.action == "break", "the budget still switches once it is spent"
