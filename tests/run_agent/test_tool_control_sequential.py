import json
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_call(name: str, arguments: str = "{}", tool_call_id: str = "tc_1"):
    tc = MagicMock()
    tc.id = tool_call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def test_sequential_agent_loop_tool_respects_pre_tool_deny():
    agent = AIAgent(
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assistant_msg = MagicMock()
    assistant_msg.tool_calls = [_make_tool_call("todo", '{"todos": []}')]
    messages = []

    with (
        patch.object(agent, "_evaluate_tool_control", return_value={"action": "deny", "reason": "blocked"}),
        patch("tools.todo_tool.todo_tool", side_effect=AssertionError("todo tool should not run")),
    ):
        agent._execute_tool_calls_sequential(assistant_msg, messages, "default")

    assert len(messages) == 1
    payload = json.loads(messages[0]["content"])
    assert payload["error"] == "操作被拒绝: blocked"


def test_sequential_agent_loop_tool_applies_post_tool_result_control():
    agent = AIAgent(
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assistant_msg = MagicMock()
    assistant_msg.tool_calls = [_make_tool_call("todo", '{"todos": []}')]
    messages = []

    with (
        patch.object(agent, "_evaluate_tool_control", return_value={"action": "allow"}),
        patch.object(agent, "_finalize_tool_result", return_value='{"ok": "post-processed"}'),
        patch("tools.todo_tool.todo_tool", return_value='{"ok": "raw"}'),
    ):
        agent._execute_tool_calls_sequential(assistant_msg, messages, "default")

    assert len(messages) == 1
    payload = json.loads(messages[0]["content"])
    assert payload == {"ok": "post-processed"}
