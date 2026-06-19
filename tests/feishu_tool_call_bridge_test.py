import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.chat_completion_helpers import build_assistant_message
from gateway.platforms.feishu import FeishuAdapter
from model_tools import handle_function_call


class _DummyAgent:
    verbose_logging = False
    reasoning_callback = None
    stream_delta_callback = None
    _stream_callback = None

    @staticmethod
    def _extract_reasoning(_assistant_message):
        return None

    @staticmethod
    def _strip_think_blocks(content: str) -> str:
        return content

    @staticmethod
    def _needs_thinking_reasoning_pad() -> bool:
        return False

    @staticmethod
    def _split_responses_tool_id(_raw_id):
        return None, None

    @staticmethod
    def _derive_responses_function_call_id(call_id, _response_item_id=None):
        return f"fc_{call_id}"

    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        return f"{fn_name}:{index}:{arguments}"


def test_tool_call_brackets_become_terminal_tool_calls():
    agent = _DummyAgent()
    assistant_message = SimpleNamespace(
        content="[TOOL_CALL]whoami[/TOOL_CALL]",
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )

    msg = build_assistant_message(agent, assistant_message, "stop")

    assert msg["content"] == ""
    assert len(msg["tool_calls"]) == 1
    tool_call = msg["tool_calls"][0]
    assert tool_call["function"]["name"] == "terminal"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "whoami"}


def test_terminal_bridge_executes_and_surfaces_explicit_errors():
    with patch("tools.terminal_tool.terminal_tool", return_value="ubuntu") as mock_terminal:
        ok_result = handle_function_call("terminal", {"command": "whoami"}, task_id="bridge-test")

    assert ok_result == "ubuntu"
    mock_terminal.assert_called_once()

    with patch("tools.terminal_tool.terminal_tool", side_effect=RuntimeError("terminal unavailable")):
        err_result = handle_function_call("terminal", {"command": "whoami"}, task_id="bridge-test")

    payload = json.loads(err_result)
    assert "terminal unavailable" in payload["error"]


def test_feishu_format_message_does_not_echo_tool_call_tags():
    adapter = FeishuAdapter.__new__(FeishuAdapter)

    formatted = adapter.format_message("[TOOL_CALL]whoami[/TOOL_CALL]")

    assert "[TOOL_CALL]" not in formatted
    assert formatted == "⚠️ Unresolved tool call: whoami"
