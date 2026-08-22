"""Regression coverage for bounded post-response error retries (#92450)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from run_agent import AIAgent


def _make_agent() -> AIAgent:
    tool_definition = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    with (
        patch("run_agent.get_tool_definitions", return_value=[tool_definition]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test-model",
            max_iterations=sys.maxsize,
            enabled_toolsets=[],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


def _mock_response(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test-model", usage=None)


def _mock_tool_response():
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[tool_call],
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test-model", usage=None)


def test_persistent_outer_loop_error_is_bounded_and_backed_off():
    """Unlimited useful turns must not imply unlimited post-response failures."""
    agent = _make_agent()
    response = _mock_response("eventually recovered")
    transport = agent._get_transport()
    original_normalize = transport.normalize_response
    normalize_calls = 0
    outer_failures = 0

    def fail_four_outer_normalizations(raw_response, **kwargs):
        nonlocal normalize_calls, outer_failures
        normalize_calls += 1
        # The retry layer first normalizes once to inspect finish_reason. The
        # outer post-response block normalizes the same response again. Fail
        # only that second call so the exception reaches the handler under
        # test instead of the separately bounded provider retry path.
        if normalize_calls % 2 == 0 and outer_failures < 4:
            outer_failures += 1
            raise RuntimeError("permanent normalization failure")
        return original_normalize(raw_response, **kwargs)

    with (
        patch.object(agent, "_interruptible_api_call", return_value=response) as api_call,
        patch.object(
            transport,
            "normalize_response",
            side_effect=fail_four_outer_normalizations,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop._sleep_outer_loop_error_backoff") as sleep,
    ):
        result = agent.run_conversation("hello")

    assert api_call.call_count == 3
    assert normalize_calls == 6
    assert outer_failures == 3
    assert "encountered repeated errors" in result["final_response"]
    assert "permanent normalization failure" in result["final_response"]
    assert result["failed"] is True
    assert result["completed"] is False
    assert sleep.call_args_list == [call(1), call(2)]


def test_successful_tool_iteration_resets_outer_loop_error_streak():
    """A successful useful iteration starts a fresh failure episode."""
    agent = _make_agent()
    transport = agent._get_transport()
    original_normalize = transport.normalize_response
    response_by_attempt = [
        _mock_response("ignored"),
        _mock_tool_response(),
        _mock_response("ignored"),
        _mock_response("ignored"),
        _mock_response("recovered after progress"),
    ]
    normalize_calls_by_attempt: dict[int, int] = {}

    def fail_selected_outer_normalizations(raw_response, **kwargs):
        attempt = agent._api_call_count
        normalize_calls_by_attempt[attempt] = normalize_calls_by_attempt.get(attempt, 0) + 1
        if attempt in {1, 3, 4} and normalize_calls_by_attempt[attempt] == 2:
            raise RuntimeError("episode-scoped normalization failure")
        return original_normalize(raw_response, **kwargs)

    def append_tool_result(_assistant_message, messages, *_args):
        messages.append(
            {
                "role": "tool",
                "name": "web_search",
                "tool_call_id": "call_1",
                "content": "ok",
            }
        )

    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=response_by_attempt,
        ) as api_call,
        patch.object(
            transport,
            "normalize_response",
            side_effect=fail_selected_outer_normalizations,
        ),
        patch.object(agent, "_execute_tool_calls", side_effect=append_tool_result),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop._sleep_outer_loop_error_backoff") as sleep,
    ):
        result = agent.run_conversation("hello")

    assert api_call.call_count == 5
    assert result["final_response"] == "recovered after progress"
    assert sleep.call_args_list == [call(1), call(1), call(2)]


def test_outer_loop_error_near_iteration_limit_is_failed_not_completed():
    """A terminal outer-loop error must not be reported as a successful turn."""
    agent = _make_agent()
    agent.max_iterations = 2
    response = _mock_response("ignored")
    transport = agent._get_transport()
    original_normalize = transport.normalize_response
    normalize_calls = 0

    def fail_outer_normalization(raw_response, **kwargs):
        nonlocal normalize_calls
        normalize_calls += 1
        if normalize_calls == 2:
            raise RuntimeError("normalization failed near iteration limit")
        return original_normalize(raw_response, **kwargs)

    with (
        patch.object(agent, "_interruptible_api_call", return_value=response),
        patch.object(
            transport,
            "normalize_response",
            side_effect=fail_outer_normalization,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop._sleep_outer_loop_error_backoff") as sleep,
    ):
        result = agent.run_conversation("hello")

    assert result["failed"] is True
    assert result["completed"] is False
    assert result["turn_exit_reason"].startswith("error_near_max_iterations(")
    sleep.assert_not_called()
