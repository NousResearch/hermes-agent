from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=4,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _mock_response(*, content="", finish_reason="stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _mock_tool_call(call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )


def _run_without_persistence(agent, prompt="work"):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(prompt)


def test_budget_checkpoint_disabled_does_not_call_checkpoint_helper(agent, monkeypatch):
    agent._budget_checkpointing_enabled = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(finish_reason="tool_calls", tool_calls=[_mock_tool_call("c1")]),
        _mock_response(finish_reason="tool_calls", tool_calls=[_mock_tool_call("c2")]),
        _mock_response(content="Done", finish_reason="stop"),
    ]
    checkpoint_calls = []

    def fake_checkpoint(*args, **kwargs):
        checkpoint_calls.append((args, kwargs))
        return "SHOULD NOT RUN"

    monkeypatch.setattr("agent.conversation_loop.handle_budget_checkpoint", fake_checkpoint)

    with patch("run_agent.handle_function_call", return_value="search result"):
        result = _run_without_persistence(agent)

    assert result["final_response"] == "Done"
    assert checkpoint_calls == []


def test_budget_checkpoint_enabled_calls_helper_before_exhaustion(agent, monkeypatch):
    agent._budget_checkpointing_enabled = True
    agent._budget_checkpoint_warning_ratio = 0.25
    agent._budget_checkpoint_checkpoint_ratio = 0.50
    agent._budget_checkpoint_mode = "continuation_packet"
    agent.client.chat.completions.create.side_effect = [
        _mock_response(finish_reason="tool_calls", tool_calls=[_mock_tool_call("c1")]),
        _mock_response(content="SHOULD NOT REACH SECOND API CALL", finish_reason="stop"),
    ]
    checkpoint_calls = []

    def fake_checkpoint(agent_arg, messages, api_call_count, **kwargs):
        checkpoint_calls.append((api_call_count, kwargs))
        return "CONTINUATION PACKET"

    monkeypatch.setattr("agent.conversation_loop.handle_budget_checkpoint", fake_checkpoint)

    with patch("run_agent.handle_function_call", return_value="search result"):
        result = _run_without_persistence(agent)

    assert result["final_response"] == "CONTINUATION PACKET"
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "budget_checkpoint"
    assert checkpoint_calls == [(2, {"budget_state": "checkpoint", "task_metadata": {}})]
    assert agent.client.chat.completions.create.call_count == 1


def test_budget_checkpoint_disabled_preserves_max_iteration_summary(agent):
    agent.max_iterations = 1
    agent.iteration_budget.max_total = 1
    agent._budget_checkpointing_enabled = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(finish_reason="tool_calls", tool_calls=[_mock_tool_call("c1")]),
        _mock_response(content="Could not finish — budget exhausted.", finish_reason="stop"),
    ]

    with patch("run_agent.handle_function_call", return_value="search result"):
        result = _run_without_persistence(agent)

    assert result["completed"] is False
    assert result["turn_exit_reason"].startswith("max_iterations_reached")
    assert result["final_response"] == "Could not finish — budget exhausted."
