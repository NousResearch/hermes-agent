"""Tests for ``agent/replay_cleanup.py`` — the shared resume cleanup.

Covers the new ``strip_trailing_orphan_tool_messages`` function and the
``sanitize_replay_history`` entry point that all three resume paths (CLI,
gateway, TUI) share.
"""

from agent.replay_cleanup import (
    is_interrupted_tool_result,
    sanitize_replay_history,
    strip_trailing_orphan_tool_messages,
    strip_dangling_tool_call_tail,
    strip_interrupted_tool_tails,
)


def _user(text):
    return {"role": "user", "content": text}


def _assistant_tc(name):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _tool(content):
    return {"role": "tool", "tool_call_id": "c1", "content": content}


def test_is_interrupted_tool_result_markers():
    assert is_interrupted_tool_result("[Command interrupted]")
    assert is_interrupted_tool_result("foo\nexit_code: 130 (interrupt)\nbar")
    assert not is_interrupted_tool_result("exit_code: 0\nclean output")
    assert not is_interrupted_tool_result("ordinary tool output")
    assert not is_interrupted_tool_result(None)


def test_strip_dangling_tool_call_tail_removes_unanswered_read_only_tail():
    history = [_user("hi"), _assistant_tc("read_file")]
    out = strip_dangling_tool_call_tail(history)
    assert out == [_user("hi")]


def test_dangling_side_effect_is_recovered_as_unknown_not_erased():
    history = [_user("hi"), _assistant_tc("write_file")]

    out = strip_dangling_tool_call_tail(history)

    assert out[:-1] == history
    assert out[-1]["role"] == "tool"
    assert out[-1]["tool_call_id"] == "c1"
    assert out[-1]["effect_disposition"] == "unknown"
    assert "may have executed" in out[-1]["content"].lower()


def test_dangling_session_mutation_is_recovered_as_unknown():
    history = [_user("hi"), _assistant_tc("todo")]

    out = strip_dangling_tool_call_tail(history)

    assert out[:-1] == history
    assert out[-1]["effect_disposition"] == "unknown"
    assert "may have executed" in out[-1]["content"].lower()


def test_mixed_dangling_batch_uses_truthful_per_call_wording():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "read", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "write", "function": {"name": "write_file", "arguments": "{}"}},
        ],
    }
    out = strip_dangling_tool_call_tail([_user("hi"), assistant])

    read_result, write_result = out[-2:]
    assert read_result["effect_disposition"] == "none"
    assert "no effect" in read_result["content"].lower()
    assert "unknown" not in read_result["content"].lower()
    assert write_result["effect_disposition"] == "unknown"
    assert "unknown" in write_result["content"].lower()


def test_strip_dangling_tool_call_tail_preserves_answered_pair():
    history = [_user("hi"), _assistant_tc("read_file"), _tool("contents")]
    out = strip_dangling_tool_call_tail(history)
    assert out == history  # answered -> untouched


def test_strip_interrupted_tool_tails_removes_interrupted_read_only_block():
    history = [_user("hi"), _assistant_tc("read_file"), _tool("[Command interrupted]")]
    out = strip_interrupted_tool_tails(history)
    assert out == [_user("hi")]


def test_interrupted_side_effect_is_preserved_as_unknown():
    history = [_user("hi"), _assistant_tc("terminal"), _tool("[Command interrupted]")]

    out = strip_interrupted_tool_tails(history)

    assert out[:-1] == history[:-1]
    assert out[-1]["role"] == "tool"
    assert out[-1]["effect_disposition"] == "unknown"


def test_strip_interrupted_tool_tails_preserves_successful_block():
    history = [_user("hi"), _assistant_tc("read_file"), _tool("ok"),
               {"role": "assistant", "content": "done"}]
    out = strip_interrupted_tool_tails(history)
    assert out == history


def test_strip_interrupted_tool_tails_removes_orphan_interrupted_tool():
    history = [_user("hi"), _tool("[Command interrupted] exit_code: 130 interrupt")]
    out = strip_interrupted_tool_tails(history)
    assert out == [_user("hi")]


def test_sanitize_replay_history_combines_both():
    # interrupted block is removed; a dangling read-only call is safe to erase
    history = [
        _user("first"),
        _assistant_tc("terminal"), _tool("[Command interrupted]"),
        _user("second"),
        _assistant_tc("read_file"),  # dangling
    ]
    out = sanitize_replay_history(history)
    assert out[:2] == [
        _user("first"),
        _assistant_tc("terminal"),
    ]
    assert out[2]["effect_disposition"] == "unknown"
    assert out[-1] == _user("second")


def test_sanitize_replay_history_noop_on_clean_history():
    history = [_user("hi"), {"role": "assistant", "content": "hello"}]
    assert sanitize_replay_history(history) == history


def test_sanitize_replay_history_empty():
    assert sanitize_replay_history([]) == []


# ── strip_trailing_orphan_tool_messages ──────────────────────────────────


def test_strips_single_trailing_tool_message():
    """A single trailing tool message (orphaned, session died after tool
    result was written but before the assistant continuation) should be
    stripped."""
    history = [
        {"role": "user", "content": "run a command"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "command output"},
    ]
    result = strip_trailing_orphan_tool_messages(history)
    assert len(result) == 2
    assert result[-1]["role"] == "assistant"
    assert result[-1].get("tool_calls")  # assistant(tool_calls) stays


def test_strips_multiple_trailing_tool_messages():
    """Multiple trailing tool messages (from concurrent tool calls) should
    all be stripped."""
    history = [
        {"role": "user", "content": "run 3 commands"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}},
            {"id": "tc2", "function": {"name": "terminal", "arguments": "{}"}},
            {"id": "tc3", "function": {"name": "terminal", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "output1"},
        {"role": "tool", "tool_call_id": "tc2", "content": "output2"},
        {"role": "tool", "tool_call_id": "tc3", "content": "output3"},
    ]
    result = strip_trailing_orphan_tool_messages(history)
    assert len(result) == 2
    assert result[-1]["role"] == "assistant"


def test_does_not_strip_when_tail_is_user():
    """``assistant(tool_calls) → tool → user`` is a VALID mid-dialog pattern.
    The user tail means the conversation is ongoing, not dead. The trailing
    tool messages must NOT be stripped."""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "out"},
        {"role": "user", "content": "Q2"},
    ]
    original = [dict(m) for m in history]
    result = strip_trailing_orphan_tool_messages(history)
    assert result == original


def test_does_not_strip_when_tail_is_assistant():
    """A trailing assistant message means the conversation ended cleanly.
    No trailing tool messages to strip."""
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = strip_trailing_orphan_tool_messages(history)
    assert result == history


def test_empty_history_no_op():
    """An empty history should be a no-op."""
    result = strip_trailing_orphan_tool_messages([])
    assert result == []


def test_no_trailing_tool_no_op():
    """A history with no trailing tool message should be unchanged."""
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "response"},
    ]
    result = strip_trailing_orphan_tool_messages(history)
    assert result == history


# ── sanitize_replay_history integration ──────────────────────────────────


def test_sanitize_strips_trailing_tool_message():
    """``sanitize_replay_history`` should strip a trailing orphan tool
    message — this is the case the new ``strip_trailing_orphan_tool_messages``
    handles that the existing strippers don't."""
    history = [
        {"role": "user", "content": "run a command"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "command output"},
    ]
    result = sanitize_replay_history(history)
    # The trailing tool is stripped by strip_trailing_orphan_tool_messages,
    # then the dangling assistant(tool_calls) is stripped by
    # strip_dangling_tool_call_tail (no content → popped entirely).
    assert result == [{"role": "user", "content": "run a command"}]


def test_sanitize_preserves_mid_dialog_tool_pair():
    """``sanitize_replay_history`` with ``assistant(tool_calls) → tool → user``
    should NOT strip anything — this is a valid mid-dialog pattern."""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "out"},
        {"role": "user", "content": "Q2"},
    ]
    original = [dict(m) for m in history]
    result = sanitize_replay_history(history)
    assert result == original


def test_sanitize_strips_dangling_assistant_tool_calls_no_content():
    """When the tail is a dangling ``assistant(tool_calls)`` with NO content
    and NO tool results, both ``strip_trailing_orphan_tool_messages`` (no-op,
    tail is not tool) and ``strip_dangling_tool_call_tail`` (pops the
    assistant) work together to clean it up."""
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}]},
    ]
    result = sanitize_replay_history(history)
    assert result == [{"role": "user", "content": "hello"}]


def test_sanitize_preserves_content_in_dangling_assistant():
    """When the trailing ``assistant(tool_calls)`` has BOTH content and
    tool_calls, the content is preserved and only the tool_calls are
    stripped."""
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Let me check...", "tool_calls": [
            {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}]},
    ]
    result = sanitize_replay_history(history)
    assert len(result) == 2
    assert result[-1]["role"] == "assistant"
    assert result[-1]["content"] == "Let me check..."
    assert "tool_calls" not in result[-1]


# ── Gateway resume simulation ────────────────────────────────────────────


def test_gateway_resume_strips_trailing_tool_after_tool_results():
    """Simulate the gateway resume path: session dies after
    ``assistant(tool_calls) → tool → [dies]``.

    The gateway calls ``sanitize_replay_history`` which should strip the
    trailing tool message and the dangling assistant(tool_calls)."""
    # This is what the gateway would see on resume after a session died
    # mid-tool-loop (tool results written, but assistant never responded)
    history = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "file1.txt\nfile2.txt"},
    ]
    result = sanitize_replay_history(history)
    # After stripping trailing tool + dangling assistant(tool_calls):
    # Only the user message remains
    assert result == [{"role": "user", "content": "list files"}]


def test_gateway_resume_with_concurrent_tool_calls():
    """Gateway resume with multiple concurrent tool calls — all trailing
    tool results stripped, dangling assistant popped."""
    history = [
        {"role": "user", "content": "run 3 commands"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}},
            {"id": "tc2", "function": {"name": "terminal", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "output1"},
        {"role": "tool", "tool_call_id": "tc2", "content": "output2"},
    ]
    result = sanitize_replay_history(history)
    assert result == [{"role": "user", "content": "run 3 commands"}]


def test_gateway_resume_preserves_active_dialog():
    """Gateway resume where the user already sent a follow-up message
    after the tool result — the dialog is active, nothing should be
    stripped."""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
        {"role": "user", "content": "follow-up question"},
    ]
    original = [dict(m) for m in history]
    result = sanitize_replay_history(history)
    assert result == original