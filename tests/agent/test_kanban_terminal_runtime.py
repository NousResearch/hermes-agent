"""Runtime contracts for successful terminal kanban tools."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import execute_tool_calls_segmented
from agent.tool_guardrails import ToolGuardrailDecision
from run_agent import AIAgent


def _tool_defs(*names: str) -> list[dict[str, Any]]:
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


def _agent(tmp_path: Path) -> Any:
    (tmp_path / "logs").mkdir(exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_tool_defs("kanban_complete", "kanban_block", "terminal"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent: Any = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = 3
    return agent


def _tool_call(name: str, call_id: str) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(*, content: str = "", tool_calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(
        message=message,
        finish_reason="tool_calls" if tool_calls else "stop",
    )
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _run(agent: Any, message: str) -> dict[str, Any]:
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        return agent.run_conversation(message)


@pytest.mark.parametrize("terminal_name", ["kanban_complete", "kanban_block"])
def test_successful_terminal_tool_stops_provider_and_drains_batch(
    tmp_path,
    terminal_name,
):
    agent = _agent(tmp_path)
    agent.client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call(terminal_name, "complete"),
            _tool_call("terminal", "must-not-run"),
        ]
    )
    dispatched: list[str] = []

    def dispatch(name: str, *_args: Any, **_kwargs: Any) -> str:
        dispatched.append(name)
        return '{"ok": true}'

    with patch("run_agent.handle_function_call", side_effect=dispatch):
        result = _run(agent, "finish the card")

    assert agent.client.chat.completions.create.call_count == 1
    assert dispatched == [terminal_name]
    assert result["completed"] is True
    assert result["failed"] is False
    tool_rows = [row for row in result["messages"] if row.get("role") == "tool"]
    assert [row["tool_call_id"] for row in tool_rows] == ["complete", "must-not-run"]
    assert "skipped" in tool_rows[-1]["content"].lower()


def test_failed_terminal_tool_continues_provider_loop(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent = _agent(tmp_path)
    agent.client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("kanban_complete", "failed")]),
        _response(content="retry finished"),
    ]

    with patch("run_agent.handle_function_call", return_value='{"ok": false}'):
        result = _run(agent, "finish the card")

    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "retry finished"


def test_segmented_executor_skips_every_later_segment(tmp_path):
    agent = _agent(tmp_path)
    agent._tool_guardrail_halt_decision = ToolGuardrailDecision(
        action="halt",
        code="earlier_guardrail",
    )
    complete = _tool_call("kanban_complete", "complete")
    later = _tool_call("terminal", "later")
    assistant = SimpleNamespace(tool_calls=[complete, later])
    messages: list[dict[str, Any]] = []
    dispatched: list[str] = []

    def dispatch(name: str, *_args: Any, **_kwargs: Any) -> str:
        dispatched.append(name)
        return '{"ok": true}'

    with (
        patch("run_agent.handle_function_call", side_effect=dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        execute_tool_calls_segmented(
            agent,
            assistant,
            messages,
            "task-1",
            segments=[("sequential", [complete]), ("sequential", [later])],
        )

    assert dispatched == ["kanban_complete"]
    assert [row["tool_call_id"] for row in messages] == ["complete", "later"]
    assert "skipped" in messages[-1]["content"].lower()
