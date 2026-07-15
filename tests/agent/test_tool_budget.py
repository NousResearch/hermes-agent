"""Focused tests for optional per-agent tool call/output budgets."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


ToolBudget = getattr(run_agent, "ToolBudget", None)


def _budget(**kwargs):
    """Construct the wished-for public budget primitive."""
    assert ToolBudget is not None, "run_agent.ToolBudget is not implemented"
    return ToolBudget(**kwargs)


def _tool_call(call_id: str, name: str = "web_search"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _assistant_message(*calls):
    return SimpleNamespace(tool_calls=list(calls))


def _response_with_tool_call(tool_call):
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(
        choices=[choice],
        model="test/model",
        usage=None,
    )


@pytest.fixture()
def agent():
    """Minimal real AIAgent with network and tool discovery isolated."""
    tool_def = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "web_search tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    with (
        patch("run_agent.get_tool_definitions", return_value=[tool_def]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        value = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    value.client = MagicMock()
    value.tool_delay = 0
    value._persist_session = lambda *args, **kwargs: None
    value._save_trajectory = lambda *args, **kwargs: None
    return value


def test_aia_agent_accepts_and_stores_optional_tool_budget():
    budget = _budget(max_calls=2, max_output_chars=11)
    tool_def = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "web_search tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with (
        patch("run_agent.get_tool_definitions", return_value=[tool_def]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            tool_budget=budget,
        )

    assert agent.tool_budget is budget


def test_tool_budget_reserves_max_calls_atomically_under_concurrency():
    budget = _budget(max_calls=3)

    with ThreadPoolExecutor(max_workers=24) as pool:
        allowed = list(pool.map(lambda _: budget.reserve_call(), range(64)))

    assert sum(allowed) == 3
    assert budget.calls_used == 3
    assert budget.remaining_calls == 0


def test_tool_budget_allocates_output_chars_atomically_under_concurrency():
    budget = _budget(max_output_chars=7)

    with ThreadPoolExecutor(max_workers=24) as pool:
        outputs = list(pool.map(lambda _: budget.truncate_output("abcd"), range(32)))

    assert sum(len(output) for output in outputs) == 7
    assert budget.output_chars_used == 7
    assert budget.remaining_output_chars == 0


def test_concurrent_tool_calls_over_call_budget_are_not_executed_but_get_tool_results(agent):
    agent.tool_budget = _budget(max_calls=1)
    calls = [_tool_call("c1"), _tool_call("c2"), _tool_call("c3")]
    messages = []

    with patch("run_agent.handle_function_call", return_value="ok") as dispatch:
        agent._execute_tool_calls_concurrent(
            _assistant_message(*calls), messages, "task-1"
        )

    assert dispatch.call_count == 1
    assert [message["role"] for message in messages] == ["tool", "tool", "tool"]
    assert [message["tool_call_id"] for message in messages] == ["c1", "c2", "c3"]
    assert "budget" in messages[1]["content"].lower()
    assert "budget" in messages[2]["content"].lower()


def test_budget_skip_results_respect_remaining_output_chars(agent):
    agent.tool_budget = _budget(max_calls=1, max_output_chars=5)
    calls = [_tool_call("c1"), _tool_call("c2"), _tool_call("c3")]
    messages = []

    with patch("run_agent.handle_function_call", return_value="abcd") as dispatch:
        agent._execute_tool_calls_concurrent(
            _assistant_message(*calls), messages, "task-1"
        )

    assert dispatch.call_count == 1
    assert [message["tool_call_id"] for message in messages] == ["c1", "c2", "c3"]
    assert sum(len(message["content"]) for message in messages) == 5
    assert agent.tool_budget.output_chars_used == 5


def test_concurrent_tool_outputs_are_truncated_to_exact_remaining_chars(agent):
    agent.tool_budget = _budget(max_output_chars=7)
    calls = [_tool_call("c1"), _tool_call("c2"), _tool_call("c3")]
    messages = []

    with patch("run_agent.handle_function_call", return_value="abcd"):
        agent._execute_tool_calls_concurrent(
            _assistant_message(*calls), messages, "task-1"
        )

    assert [message["role"] for message in messages] == ["tool", "tool", "tool"]
    assert sum(len(message["content"]) for message in messages) == 7
    assert agent.tool_budget.output_chars_used == 7


def test_sequential_tool_calls_enforce_call_budget_and_preserve_protocol_results(agent):
    agent.tool_budget = _budget(max_calls=1)
    calls = [_tool_call("c1"), _tool_call("c2")]
    messages = []

    with patch("run_agent.handle_function_call", return_value="ok") as dispatch:
        agent._execute_tool_calls_sequential(
            _assistant_message(*calls), messages, "task-1"
        )

    assert dispatch.call_count == 1
    assert [message["role"] for message in messages] == ["tool", "tool"]
    assert [message["tool_call_id"] for message in messages] == ["c1", "c2"]
    assert "budget" in messages[1]["content"].lower()


def test_omitted_tool_budget_preserves_existing_tool_execution(agent):
    assert getattr(agent, "tool_budget", None) is None
    messages = []
    call = _tool_call("c1")

    with patch("run_agent.handle_function_call", return_value="unlimited result") as dispatch:
        agent._execute_tool_calls_sequential(
            _assistant_message(call), messages, "task-1"
        )

    dispatch.assert_called_once()
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["name"] == "web_search"
    assert messages[0]["tool_call_id"] == "c1"
    assert messages[0]["content"] == "unlimited result"


def test_tool_budget_exhaustion_stops_before_another_provider_call(agent):
    tool_call = _tool_call("c1")
    provider_calls = []

    def fake_api_call(api_kwargs):
        provider_calls.append(api_kwargs)
        return _response_with_tool_call(tool_call)

    agent.tool_budget = _budget(max_calls=1)
    agent._interruptible_api_call = fake_api_call

    with patch("run_agent.handle_function_call", return_value="tool result"):
        result = agent.run_conversation("hello")

    assert len(provider_calls) == 1
    assert result["tool_budget"]["calls_used"] == 1
    assert result["tool_budget"]["exhausted"] is True
    assert result["tool_budget"]["exhaustion_reason"] == "max_calls"
    assert "tool_budget" in result["turn_exit_reason"]


def test_tool_output_budget_exhaustion_stops_before_another_provider_call(agent):
    tool_call = _tool_call("c1")
    provider_calls = []

    def fake_api_call(api_kwargs):
        provider_calls.append(api_kwargs)
        return _response_with_tool_call(tool_call)

    agent.tool_budget = _budget(max_output_chars=3)
    agent._interruptible_api_call = fake_api_call

    with patch("run_agent.handle_function_call", return_value="four chars"):
        result = agent.run_conversation("hello")

    assert len(provider_calls) == 1
    assert result["tool_budget"]["output_chars_used"] == 3
    assert result["tool_budget"]["exhausted"] is True
    assert result["tool_budget"]["exhaustion_reason"] == "max_output_chars"
    assert "tool_budget" in result["turn_exit_reason"]
