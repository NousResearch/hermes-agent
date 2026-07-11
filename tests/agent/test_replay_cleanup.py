"""Tests for agent.replay_cleanup — shared replay-tail sanitizers.

These functions were extracted from gateway/run.py so every resume surface
(messaging gateway AND TUI/WebUI gateway) strips poisoned tool-call tails the
same way. Regression coverage for #29086 (WebUI session permanently stuck
because the dangling tool-call tail was replayed on every resume).
"""

from agent.replay_cleanup import (
    is_interrupted_tool_result,
    is_dangerous_confirmation,
    strip_stale_dangerous_confirmations,
    strip_dangling_tool_call_tail,
    strip_interrupted_tool_tails,
    sanitize_replay_history,
)


def _user(text):
    return {"role": "user", "content": text}


def _user_at(text, timestamp):
    return {"role": "user", "content": text, "timestamp": timestamp}


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


def test_is_dangerous_confirmation_matches_known_high_risk_phrases():
    assert is_dangerous_confirmation("confirm forced restart")
    assert is_dangerous_confirmation("Please CONFIRM shutdown now.")
    assert is_dangerous_confirmation("確認重啟")
    assert not is_dangerous_confirmation("confirm")
    assert not is_dangerous_confirmation("yes")
    assert not is_dangerous_confirmation("ordinary follow-up")


def test_strip_stale_dangerous_confirmations_redacts_old_confirmation_in_place():
    history = [
        _user("start"),
        {"role": "assistant", "content": "ready"},
        _user_at("confirm forced restart", 100.0),
        {"role": "assistant", "content": "restarting"},
    ]

    out = strip_stale_dangerous_confirmations(history, now=500.0)

    assert len(out) == len(history)
    assert [msg["role"] for msg in out] == ["user", "assistant", "user", "assistant"]
    assert out[2]["timestamp"] == 100.0
    assert out[2]["content"] != "confirm forced restart"
    assert "EXPIRED" in out[2]["content"]
    assert "confirm forced restart" not in out[2]["content"]


def test_strip_stale_dangerous_confirmations_redacts_iso_timestamp():
    history = [_user_at("confirm forced restart", "1970-01-01T00:01:40+00:00")]

    out = strip_stale_dangerous_confirmations(history, now=500.0)

    assert out[0]["timestamp"] == "1970-01-01T00:01:40+00:00"
    assert "EXPIRED" in out[0]["content"]


def test_strip_stale_dangerous_confirmations_preserves_fresh_confirmation():
    history = [_user_at("confirm forced restart", 100.0)]

    assert strip_stale_dangerous_confirmations(history, now=150.0) == history


def test_strip_stale_dangerous_confirmations_preserves_old_ordinary_text():
    history = [_user_at("ordinary follow-up", 100.0)]

    assert strip_stale_dangerous_confirmations(history, now=500.0) == history


def test_strip_stale_dangerous_confirmations_preserves_untimestamped_messages():
    history = [_user("confirm forced restart")]

    assert strip_stale_dangerous_confirmations(history, now=500.0) == history


def test_strip_dangling_tool_call_tail_removes_unanswered_tail():
    history = [_user("hi"), _assistant_tc("write_file")]
    out = strip_dangling_tool_call_tail(history)
    assert out == [_user("hi")]


def test_strip_dangling_tool_call_tail_preserves_answered_pair():
    history = [_user("hi"), _assistant_tc("read_file"), _tool("contents")]
    out = strip_dangling_tool_call_tail(history)
    assert out == history  # answered -> untouched


def test_strip_interrupted_tool_tails_removes_interrupted_block():
    history = [_user("hi"), _assistant_tc("terminal"), _tool("[Command interrupted]")]
    out = strip_interrupted_tool_tails(history)
    assert out == [_user("hi")]


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
    # interrupted block in the middle + dangling tail at the end
    history = [
        _user("first"),
        _assistant_tc("terminal"), _tool("[Command interrupted]"),
        _user("second"),
        _assistant_tc("write_file"),  # dangling
    ]
    out = sanitize_replay_history(history)
    assert out == [_user("first"), _user("second")]


def test_sanitize_replay_history_noop_on_clean_history():
    history = [_user("hi"), {"role": "assistant", "content": "hello"}]
    assert sanitize_replay_history(history) == history


def test_sanitize_replay_history_empty():
    assert sanitize_replay_history([]) == []
