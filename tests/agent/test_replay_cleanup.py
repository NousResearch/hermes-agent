"""Tests for agent.replay_cleanup — shared replay-tail sanitizers.

These functions were extracted from gateway/run.py so every resume surface
(messaging gateway AND TUI/WebUI gateway) strips poisoned tool-call tails the
same way. Regression coverage for #29086 (WebUI session permanently stuck
because the dangling tool-call tail was replayed on every resume).
"""

from agent.replay_cleanup import (
    is_interrupted_tool_result,
    strip_dangling_tool_call_tail,
    strip_interrupted_tool_tails,
    sanitize_replay_history,
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
    envelope = '{"exit_code": 130, "output": "partial\\n[Command interrupted]"}'

    assert is_interrupted_tool_result("[Command interrupted]", "terminal")
    assert is_interrupted_tool_result(envelope, "terminal")

    # Result content has no authority until its call id is bound to the exact
    # terminal tool name.
    assert not is_interrupted_tool_result("[Command interrupted]")
    assert not is_interrupted_tool_result(envelope, "read_file")

    # Generic fields, coercible values, natural exit 130, and marker
    # substrings are all ordinary opaque tool data.
    assert not is_interrupted_tool_result(
        '{"exit_code": -1, "status": "interrupted"}', "terminal"
    )
    assert not is_interrupted_tool_result(
        '{"exit_code": "130", "output": "[Command interrupted]"}', "terminal"
    )
    assert not is_interrupted_tool_result(
        '{"exit_code": 130, "output": "natural child exit"}', "terminal"
    )
    assert not is_interrupted_tool_result(
        "prefix [Command interrupted] suffix", "terminal"
    )
    assert not is_interrupted_tool_result(None, "terminal")


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


def test_terminal_marker_from_non_terminal_tool_stays_opaque():
    history = [_user("hi"), _assistant_tc("read_file"), _tool("[Command interrupted]")]
    out = strip_interrupted_tool_tails(history)
    assert out == history


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


def test_strip_interrupted_tool_tails_preserves_orphan_unknown_result():
    history = [_user("hi"), _tool("[Command interrupted]")]
    out = strip_interrupted_tool_tails(history)
    assert out == history


def test_terminal_envelope_with_unknown_call_id_stays_opaque():
    history = [
        _user("hi"),
        _assistant_tc("terminal"),
        {
            "role": "tool",
            "tool_call_id": "not-c1",
            "content": '{"exit_code": 130, "output": "[Command interrupted]"}',
        },
    ]

    assert strip_interrupted_tool_tails(history) == history


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
