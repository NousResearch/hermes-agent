import json
from types import SimpleNamespace

from agent.agent_runtime_helpers import invoke_tool


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
