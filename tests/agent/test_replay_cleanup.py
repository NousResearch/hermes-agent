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


def _assistant_tc(name, *, call_id="c1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool(content, *, call_id="c1"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


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


def test_partial_side_effecting_batch_is_completed_without_retrying_missing_call():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "read", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "write", "function": {"name": "write_file", "arguments": "{}"}},
        ],
    }
    interrupted_read = _tool("[Command interrupted]", call_id="read")
    following_user = _user("continue")

    out = sanitize_replay_history([assistant, interrupted_read, following_user])

    assert out[0] is assistant
    assert [result["tool_call_id"] for result in out[1:3]] == ["read", "write"]
    assert out[1]["effect_disposition"] == "none"
    assert out[2]["effect_disposition"] == "unknown"
    assert "inspect current state before retrying" in out[2]["content"].lower()
    assert out[3] is following_user


def test_partial_read_only_batch_is_removed_so_it_can_be_retried_safely():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "b", "function": {"name": "web_search", "arguments": "{}"}},
        ],
    }
    history = [_user("inspect"), assistant, _tool("done", call_id="a"), _user("next")]

    assert sanitize_replay_history(history) == [_user("inspect"), _user("next")]


def test_completed_multi_call_batch_is_unchanged():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "b", "function": {"name": "write_file", "arguments": "{}"}},
        ],
    }
    history = [assistant, _tool("read", call_id="a"), _tool("wrote", call_id="b")]

    assert sanitize_replay_history(history) == history


def test_surplus_malformed_result_is_dropped_from_completed_batch():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "b", "function": {"name": "write_file", "arguments": "{}"}},
        ],
    }
    read_result = _tool("read", call_id="a")
    write_result = _tool("wrote", call_id="b")
    malformed_result = {"role": "tool", "content": "missing id"}
    following_user = _user("next")

    assert sanitize_replay_history(
        [assistant, read_result, write_result, malformed_result, following_user]
    ) == [assistant, read_result, write_result, following_user]


def test_surplus_duplicate_result_is_dropped_from_completed_batch():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "b", "function": {"name": "write_file", "arguments": "{}"}},
        ],
    }
    first_a = _tool("first", call_id="a")
    duplicate_a = _tool("duplicate", call_id="a")
    result_b = _tool("wrote", call_id="b")

    assert sanitize_replay_history(
        [assistant, first_a, duplicate_a, result_b]
    ) == [assistant, first_a, result_b]


def test_sanitize_replay_history_combines_both():
    # interrupted block is removed; a dangling read-only call is safe to erase
    history = [
        _user("first"),
        _assistant_tc("terminal"),
        _tool("[Command interrupted]"),
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
