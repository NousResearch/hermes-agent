"""Behavior contracts for ``finish`` dispatch on the active execution paths.

``finish`` is a plain registry tool (tools/finish_tool.py) with no agent-loop
state, so it must dispatch through the standard ``handle_function_call``
surface used by both real executors in ``agent/tool_executor.py``:

    * sequential -> ``execute_tool_calls_sequential`` fallback branch
    * concurrent -> ``execute_tool_calls_concurrent`` -> ``agent._invoke_tool``

These tests exercise those production surfaces end-to-end (no mocked
dispatch) and pin that ``finish`` is NOT stubbed out as an agent-loop tool.
"""

import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

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


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("finish"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
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
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _finish_tool_call(arguments: dict, call_id: str = "call_finish_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="finish", arguments=json.dumps(arguments)),
    )


def _tool_result_payload(messages: list, call_id: str) -> dict:
    result = next(
        m for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    )
    return json.loads(result["content"])


def test_finish_is_not_stubbed_as_agent_loop_tool():
    from model_tools import _AGENT_LOOP_TOOLS, handle_function_call

    assert "finish" not in _AGENT_LOOP_TOOLS

    payload = json.loads(
        handle_function_call(
            "finish",
            {"status": "done", "summary": "ok", "evidence": ["tests pass"]},
            "task-1",
        )
    )
    assert payload == {
        "success": True,
        "status": "done",
        "summary": "ok",
        "evidence": ["tests pass"],
    }


def test_sequential_executor_dispatches_finish_to_registry_handler():
    agent = _make_agent()
    tool_call = _finish_tool_call(
        {"status": "done", "summary": "All steps complete", "evidence": ["pytest: 4 passed"]},
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []

    agent._execute_tool_calls_sequential(assistant_message, messages, "task-seq")

    payload = _tool_result_payload(messages, tool_call.id)
    assert payload == {
        "success": True,
        "status": "done",
        "summary": "All steps complete",
        "evidence": ["pytest: 4 passed"],
    }


def test_sequential_executor_surfaces_invalid_finish_status():
    agent = _make_agent()
    tool_call = _finish_tool_call(
        {"status": "continue", "summary": "not done", "evidence": []},
        call_id="call_finish_bad",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []

    agent._execute_tool_calls_sequential(assistant_message, messages, "task-seq")

    payload = _tool_result_payload(messages, tool_call.id)
    assert payload["success"] is False
    assert "status" in payload["error"]


def test_concurrent_executor_dispatches_finish_to_registry_handler():
    agent = _make_agent()
    tool_call = _finish_tool_call(
        {"status": "blocked", "summary": "Need credentials", "evidence": ["login failed"]},
        call_id="call_finish_conc",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []

    agent._execute_tool_calls_concurrent(assistant_message, messages, "task-conc")

    payload = _tool_result_payload(messages, tool_call.id)
    assert payload == {
        "success": True,
        "status": "blocked",
        "summary": "Need credentials",
        "evidence": ["login failed"],
    }
