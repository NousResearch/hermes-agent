"""Tests for agent.replay_cleanup — shared replay-tail sanitizers.

These functions were extracted from gateway/run.py so every resume surface
(messaging gateway AND TUI/WebUI gateway) strips poisoned tool-call tails the
same way. Regression coverage for #29086 (WebUI session permanently stuck
because the dangling tool-call tail was replayed on every resume).
"""

import json

from agent.replay_cleanup import (
    is_interrupted_tool_result,
    strip_dangling_tool_call_tail,
    strip_interrupted_tool_tails,
    sanitize_replay_history,
)
from agent.tool_dispatch_helpers import make_tool_result_message


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


def _quoted_interrupt_history(tool_name, text, call_id="c1"):
    result = make_tool_result_message(tool_name, text, call_id)
    history = [
        {"role": "user", "content": "Read the documentation."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        result,
        {"role": "user", "content": "Explain the result; do not rerun the command."},
    ]
    return result, history


def test_quoted_interrupt_marker_in_successful_terminal_is_not_interrupted():
    text = json.dumps({"output": "Documentation quotes [Command interrupted].", "exit_code": 0})
    assert is_interrupted_tool_result(text) is False
    result, history = _quoted_interrupt_history("terminal", text)
    replay = sanitize_replay_history(history)
    assert any(m.get("role") == "tool" and m.get("content") == result["content"] for m in replay)
    assert replay[-1]["content"] == "Explain the result; do not rerun the command."


def test_quoted_interrupt_marker_in_successful_read_file_keeps_block():
    text = json.dumps({"content": "Documentation quotes [Command interrupted]."})
    assert is_interrupted_tool_result(text) is False
    result, history = _quoted_interrupt_history("read_file", text)
    replay = sanitize_replay_history(history)
    assert len(replay) == 4
    assert any(m.get("role") == "tool" and m.get("content") == result["content"] for m in replay)
    assert replay[-1]["content"] == "Explain the result; do not rerun the command."


def test_unstructured_command_interrupted_still_classifies():
    assert is_interrupted_tool_result("[Command interrupted]") is True


def test_structured_genuine_interrupt_still_classifies():
    assert is_interrupted_tool_result(json.dumps({
        "output": "[Command interrupted]", "exit_code": 130
    })) is True


def test_ordinary_failure_is_not_interrupt():
    assert is_interrupted_tool_result(json.dumps({
        "output": "boom", "exit_code": 1
    })) is False


def test_exit_code_zero_discussing_interrupt_130_is_not_interrupt():
    assert is_interrupted_tool_result(json.dumps({
        "output": "docs mention interrupt exit_code 130", "exit_code": 0
    })) is False


def test_structured_exit_code_1300_is_not_interrupt():
    assert is_interrupted_tool_result(json.dumps({
        "output": "notes discuss interrupt", "exit_code": 1300
    })) is False


def test_bare_structured_exit_code_130_without_indication_is_not_interrupt():
    assert is_interrupted_tool_result(json.dumps({
        "output": "killed", "exit_code": 130
    })) is False
