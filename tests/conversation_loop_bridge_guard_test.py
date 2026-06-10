"""Regression tests: [TOOL_CALL] must be extracted at conversation_loop guard level.

PR #22 (P1 follow-up): build_assistant_message() write-back is not enough.
The normal execution path in conversation_loop.py:3608 checks
`if assistant_message.tool_calls:` BEFORE calling _build_assistant_message.
For bracket-only tool calls (no native tool_calls), this guard is false,
so the tool execution branch is completely skipped.

Fix: extract [TOOL_CALL] blocks from content BEFORE the guard check,
directly in conversation_loop.py, so bracket-only calls also reach
_execute_tool_calls.

These tests verify the guard-level extraction logic in isolation.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import _extract_bracket_tool_calls


class _DummyAgent:
    """Minimal stub matching what _extract_bracket_tool_calls needs."""
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
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )


# ── Guard-level extraction logic (mirrors conversation_loop.py) ────────────────

def _guard_level_extract(agent, assistant_message):
    """Mirrors the extraction logic added to conversation_loop.py:3605.
    
    Returns (assistant_message with mutated tool_calls, stripped content).
    """
    _raw_content = assistant_message.content or ""
    if isinstance(_raw_content, str) and "[TOOL_CALL]" in _raw_content:
        _bridge_calls, _stripped = _extract_bracket_tool_calls(agent, _raw_content)
        if _bridge_calls:
            assistant_message.content = _stripped
            existing = list(assistant_message.tool_calls) if assistant_message.tool_calls else []
            assistant_message.tool_calls = existing + _bridge_calls
    return assistant_message


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_bracket_only_reaches_tool_calls_before_guard():
    """Bracket-only responses must populate assistant_message.tool_calls.

    This is the core regression: before the fix, if the model returns only
    [TOOL_CALL]whoami[/TOOL_CALL] with no native tool_calls, the
    conversation_loop guard `if assistant_message.tool_calls:` at line 3608
    evaluates to False and tools are never executed.

    After the fix: guard-level extraction populates tool_calls first.
    """
    agent = _DummyAgent()
    assistant_message = _make_assistant("[TOOL_CALL]whoami[/TOOL_CALL]", tool_calls=None)

    # Pre-condition: no native tool_calls
    assert assistant_message.tool_calls is None

    # Apply guard-level extraction (mirrors conversation_loop.py fix)
    result = _guard_level_extract(agent, assistant_message)

    # Post-condition: tool_calls now populated
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "terminal"
    assert json.loads(result.tool_calls[0].function.arguments)["command"] == "whoami"
    # Content stripped of bracket tags
    assert result.content == ""


def test_bracket_plus_native_both_survive():
    """When model provides both native tool_calls AND bracket tags,
    guard-level extraction must preserve both.
    """
    agent = _DummyAgent()
    native_tc = SimpleNamespace(
        id="native_1",
        call_id="native_1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path":"/etc/hosts"}'),
    )
    assistant_message = _make_assistant(
        "[TOOL_CALL]pwd[/TOOL_CALL]",
        tool_calls=[native_tc],
    )

    result = _guard_level_extract(agent, assistant_message)

    names = {tc.function.name for tc in result.tool_calls}
    assert "read_file" in names, "native tool_call must survive"
    assert "terminal" in names, "bridge tool_call must be added"
    assert len(result.tool_calls) == 2


def test_no_bracket_no_mutation():
    """Plain text response with no bracket tags must not be mutated."""
    agent = _DummyAgent()
    original_tc = SimpleNamespace(
        id="existing",
        call_id="existing",
        type="function",
        function=SimpleNamespace(name="memory", arguments="{}"),
    )
    assistant_message = _make_assistant("Hello world.", tool_calls=[original_tc])
    original_ref = assistant_message.tool_calls[0]

    result = _guard_level_extract(agent, assistant_message)

    assert result.tool_calls[0] is original_ref
    assert result.content == "Hello world."


def test_bracket_with_json_payload():
    """JSON [TOOL_CALL] payloads also reach tool_calls."""
    agent = _DummyAgent()
    payload = json.dumps({"name": "terminal", "arguments": {"command": "ls /tmp"}})
    assistant_message = _make_assistant(f"[TOOL_CALL]{payload}[/TOOL_CALL]", tool_calls=None)

    result = _guard_level_extract(agent, assistant_message)

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "terminal"
    assert json.loads(result.tool_calls[0].function.arguments)["command"] == "ls /tmp"
    assert result.content == ""


def test_guard_now_true_for_bracket_only():
    """Verify the actual guard condition evaluates to True after extraction."""
    agent = _DummyAgent()
    assistant_message = _make_assistant("[TOOL_CALL]ls /[/TOOL_CALL]", tool_calls=None)

    # Guard condition BEFORE fix (would be False)
    assert not assistant_message.tool_calls

    # Apply fix
    _guard_level_extract(agent, assistant_message)

    # Guard condition AFTER fix (now True)
    assert assistant_message.tool_calls  # this is `if assistant_message.tool_calls:`
