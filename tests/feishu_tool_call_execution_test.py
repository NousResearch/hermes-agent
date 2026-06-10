"""Regression tests for the [TOOL_CALL] bridge → _execute_tool_calls execution path.

PR #22 follow-up to PR #21.
PR #21 parsed [TOOL_CALL]...[/TOOL_CALL] into structured tool_calls and
removed them from assistant content, but bridge tool_calls were only added
to the returned message dict — not written back to assistant_message.tool_calls.
The caller (_execute_tool_calls) reads assistant_message.tool_calls directly,
so bracket-derived calls were never executed.

Fix: write the merged tool_calls list back to assistant_message.tool_calls
inside build_assistant_message() before returning.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.chat_completion_helpers import build_assistant_message
from gateway.platforms.feishu import FeishuAdapter


class _DummyAgent:
    """Minimal agent stub with the methods build_assistant_message needs."""
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


def _make_assistant(content: str, tool_calls=None):
    """Build a minimal assistant_message SimpleNamespace."""
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )


# ── Core fix verification ──────────────────────────────────────────────────────

def test_bridge_tool_calls_written_back_to_assistant_message():
    """Regression: bridge tool_calls must be written to assistant_message.tool_calls.

    The fix ensures assistant_message.tool_calls reflects the merged list after
    build_assistant_message returns, so _execute_tool_calls sees them.
    """
    agent = _DummyAgent()
    assistant_message = _make_assistant("[TOOL_CALL]whoami[/TOOL_CALL]", tool_calls=None)

    result_msg = build_assistant_message(agent, assistant_message, "stop")

    # Content must be stripped of bracket tags
    assert result_msg["content"] == "", f"Expected empty content, got {result_msg!r}"

    # The returned dict must have tool_calls
    assert "tool_calls" in result_msg
    assert len(result_msg["tool_calls"]) == 1

    # KEY ASSERTION: assistant_message.tool_calls must also be updated
    # so that the caller (_execute_tool_calls) sees the bridge calls
    assert assistant_message.tool_calls is not None, (
        "assistant_message.tool_calls must be written back — "
        "_execute_tool_calls reads this field, not the returned dict"
    )
    assert len(assistant_message.tool_calls) == 1
    assert assistant_message.tool_calls[0].function.name == "terminal"
    assert json.loads(assistant_message.tool_calls[0].function.arguments)["command"] == "whoami"


def test_bridge_plus_native_tool_calls_both_written_back():
    """When the model returns both native tool_calls and bracket tags,
    both must appear in assistant_message.tool_calls after build_assistant_message.
    """
    agent = _DummyAgent()
    native_tc = SimpleNamespace(
        id="native_call_1",
        call_id="native_call_1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path":"/etc/hosts"}'),
    )
    assistant_message = _make_assistant(
        "[TOOL_CALL]pwd[/TOOL_CALL]",
        tool_calls=[native_tc],
    )

    build_assistant_message(agent, assistant_message, "tool_calls")

    result_tc_names = {tc.function.name for tc in assistant_message.tool_calls}
    assert "read_file" in result_tc_names, "native tool_call must survive"
    assert "terminal" in result_tc_names, "bridge tool_call must be present"
    assert len(assistant_message.tool_calls) == 2


def test_bridge_with_json_payload_written_back():
    """JSON [TOOL_CALL] payloads (with explicit name/arguments) also
    must be written back to assistant_message.tool_calls.
    """
    agent = _DummyAgent()
    payload = json.dumps({"name": "terminal", "arguments": {"command": "ls /tmp"}})
    assistant_message = _make_assistant(f"[TOOL_CALL]{payload}[/TOOL_CALL]", tool_calls=None)

    build_assistant_message(agent, assistant_message, "stop")

    assert assistant_message.tool_calls is not None
    assert len(assistant_message.tool_calls) == 1
    assert assistant_message.tool_calls[0].function.name == "terminal"
    assert json.loads(assistant_message.tool_calls[0].function.arguments)["command"] == "ls /tmp"


def test_no_bridge_no_mutation():
    """When there are no bracket tags, assistant_message.tool_calls
    must not be mutated.
    """
    agent = _DummyAgent()
    original_tc = SimpleNamespace(
        id="existing",
        call_id="existing",
        type="function",
        function=SimpleNamespace(name="memory", arguments="{}"),
    )
    assistant_message = _make_assistant("Just a plain text response.", tool_calls=[original_tc])
    original_tc_ref = assistant_message.tool_calls[0]

    build_assistant_message(agent, assistant_message, "stop")

    # Must not have been replaced
    assert assistant_message.tool_calls[0] is original_tc_ref
    assert len(assistant_message.tool_calls) == 1


# ── Feishu format_message regression ──────────────────────────────────────────

def test_feishu_format_message_no_echo():
    """Feishu's format_message must never echo [TOOL_CALL] tags verbatim."""
    adapter = FeishuAdapter.__new__(FeishuAdapter)

    result = adapter.format_message("[TOOL_CALL]whoami[/TOOL_CALL]")
    assert "[TOOL_CALL]" not in result
    assert "⚠️" in result  # explicit error, not the raw tags

    # Clean text should pass through unchanged
    assert adapter.format_message("Hello world") == "Hello world"


# ── Execution smoke test ─────────────────────────────────────────────────────

def test_bridge_terminal_command_written_back():
    """A [TOOL_CALL]ls[/TOOL_CALL] must write back a callable
    terminal tool invocation to assistant_message.tool_calls.
    """
    agent = _DummyAgent()
    assistant_message = _make_assistant("[TOOL_CALL]ls /tmp[/TOOL_CALL]", tool_calls=None)

    build_assistant_message(agent, assistant_message, "stop")

    assert assistant_message.tool_calls is not None
    assert len(assistant_message.tool_calls) == 1
    tc = assistant_message.tool_calls[0]
    assert tc.function.name == "terminal"
    raw_args = json.loads(tc.function.arguments)
    assert raw_args["command"] == "ls /tmp"
