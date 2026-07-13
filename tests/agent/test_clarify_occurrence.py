import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import invoke_tool
from run_agent import AIAgent


def test_invoke_tool_propagates_clarify_tool_call_id_to_callback():
    received = {}

    def clarify_callback(question, choices, *, tool_call_id=None):
        received.update(question=question, choices=choices, tool_call_id=tool_call_id)
        return "B"

    agent = SimpleNamespace(
        clarify_callback=clarify_callback,
        session_id="session-1",
        _current_turn_id="turn-1",
        _current_api_request_id="request-1",
        _memory_manager=None,
    )

    result = json.loads(invoke_tool(
        agent,
        "clarify",
        {"question": "Choose", "choices": ["A", "B"]},
        effective_task_id="task-1",
        tool_call_id="call-concurrent-b",
        pre_tool_block_checked=True,
        skip_tool_request_middleware=True,
    ))

    assert result["user_response"] == "B"
    assert received == {
        "question": "Choose",
        "choices": ["A", "B"],
        "tool_call_id": "call-concurrent-b",
    }


def test_sequential_tool_path_propagates_clarify_tool_call_id_to_callback():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-clarify-test-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    tool_defs = [{
        "type": "function",
        "function": {
            "name": "clarify",
            "description": "ask",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    received = {}

    def clarify_callback(question, choices, *, tool_call_id=None):
        received.update(question=question, choices=choices, tool_call_id=tool_call_id)
        return "A"

    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
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
            clarify_callback=clarify_callback,
        )
    agent.client = MagicMock()
    setattr(agent, "tool_delay", 0)
    agent._flush_messages_to_session_db = MagicMock()
    tool_call = SimpleNamespace(
        id="call-sequential",
        type="function",
        function=SimpleNamespace(
            name="clarify",
            arguments=json.dumps({"question": "Choose", "choices": ["A", "B"]}),
        ),
    )

    messages = []
    agent._execute_tool_calls_sequential(
        SimpleNamespace(content="", tool_calls=[tool_call]),
        messages,
        "task-1",
    )

    assert received == {
        "question": "Choose",
        "choices": ["A", "B"],
        "tool_call_id": "call-sequential",
    }
