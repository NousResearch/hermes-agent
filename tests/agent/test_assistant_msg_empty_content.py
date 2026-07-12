"""Regression test for #63200.

``build_assistant_message`` must normalize empty-string ``content`` to ``None``
when ``tool_calls`` are present. The OpenAI API spec requires assistant messages
with tool_calls to use ``content: null`` (or omit it), not an empty string.
Strict validators like DeepSeek reject ``content: ""`` with HTTP 400.
"""

from unittest.mock import MagicMock

from agent.chat_completion_helpers import build_assistant_message


class _FakeToolCall:
    def __init__(self, tc_id, name, arguments):
        self.id = tc_id
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments
        self.extra_content = None

    def __getattr__(self, _name):
        return None


class _FakeAssistantMsg:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls
        self.function_call = None
        self.reasoning_content = None
        self.model_extra = None
        self.reasoning_details = None

    def __getattr__(self, _name):
        return None


class _FakeAgent:
    stream_delta_callback = None
    _stream_callback = None
    reasoning_callback = None
    verbose_logging = False

    def _extract_reasoning(self, _msg):
        return None

    def _strip_think_blocks(self, text):
        return text

    def _needs_thinking_reasoning_pad(self):
        return False

    def _split_responses_tool_id(self, _raw):
        return (None, None)

    def _derive_responses_function_call_id(self, _call_id, _resp_id):
        return None

    def _deterministic_call_id(self, _name, _args, idx):
        return f"det_{idx}"


def test_empty_content_normalized_to_none_with_tool_calls():
    """Assistant message with tool_calls and empty content should have content=None."""
    tc = _FakeToolCall("call_1", "terminal", "{}")
    msg = build_assistant_message(
        _FakeAgent(), _FakeAssistantMsg("", [tc]), "tool_calls"
    )
    assert msg["tool_calls"] is not None
    assert msg["content"] is None


def test_nonempty_content_preserved_with_tool_calls():
    """Assistant message with tool_calls and real content should keep it."""
    tc = _FakeToolCall("call_1", "terminal", "{}")
    msg = build_assistant_message(
        _FakeAgent(), _FakeAssistantMsg("Running command...", [tc]), "tool_calls"
    )
    assert msg["tool_calls"] is not None
    assert msg["content"] == "Running command..."


def test_empty_content_without_tool_calls_stays_empty():
    """Assistant message with no tool_calls should keep empty string content."""
    msg = build_assistant_message(
        _FakeAgent(), _FakeAssistantMsg("", None), "stop"
    )
    assert msg.get("tool_calls") is None
    assert msg["content"] == ""


def test_none_content_with_tool_calls_stays_none():
    """Assistant message with tool_calls and None content should stay None."""
    tc = _FakeToolCall("call_1", "terminal", "{}")
    msg = build_assistant_message(
        _FakeAgent(), _FakeAssistantMsg(None, [tc]), "tool_calls"
    )
    assert msg["tool_calls"] is not None
    assert msg["content"] is None
