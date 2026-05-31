"""Unit tests for hermes_cli.session_recap."""
from __future__ import annotations

import json

import pytest

from hermes_cli.session_recap import build_recap


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text=None, tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(name, args):
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _tool_result(content="ok"):
    return {"role": "tool", "content": content}


def test_empty_history():
    out = build_recap([])
    assert "Session recap" in out
    assert "nothing to recap" in out


def test_header_shows_title_when_provided():
    out = build_recap([_user("hello")], session_title="Refactor the adapter")
    assert "Refactor the adapter" in out.splitlines()[0]


def test_header_shows_short_id_when_no_title():
    out = build_recap([_user("hello")], session_id="abcdef1234567890")
    assert "abcdef12" in out.splitlines()[0]


def test_counts_recent_turns():
    msgs = [
        _user("one"),
        _assistant("first reply"),
        _user("two"),
        _assistant("second reply"),
    ]
    out = build_recap(msgs)
    assert "2 user turn" in out
    assert "assistant repl" in out


def test_last_ask_and_reply_are_surfaced():
    msgs = [
        _user("old question"),
        _assistant("old answer"),
        _user("summarise the docs"),
        _assistant("here is the summary of the docs you asked for"),
    ]
    out = build_recap(msgs)
    assert "summarise the docs" in out
    assert "summary of the docs" in out


def test_tool_counts_and_files():
    msgs = [
        _user("edit the readme and run tests"),
        _assistant(
            tool_calls=[
                _tool_call("read_file", {"path": "README.md"}),
                _tool_call("patch", {"path": "README.md"}),
            ]
        ),
        _tool_result(),
        _tool_result(),
        _assistant(
            tool_calls=[
                _tool_call("terminal", {"command": "pytest"}),
            ]
        ),
        _tool_result("tests ok"),
        _assistant("All green."),
    ]
    out = build_recap(msgs)
    assert "patch×1" in out
    assert "terminal×1" in out
    assert "read_file×1" in out
    # README.md should appear (may include cwd-relative prefix stripping).
    assert "README.md" in out


def test_tool_preview_length_truncates_long_user_prompt():
    long = "x " * 500
    out = build_recap([_user(long)])
    ask_line = [l for l in out.splitlines() if "Last ask" in l][0]
    assert len(ask_line) < 300  # truncated with ellipsis
    assert "…" in ask_line


def test_respects_recent_window():
    # 30 turns of user+assistant; only the most recent 20 should be summarised.
    msgs = []
    for i in range(30):
        msgs.append(_user(f"question {i}"))
        msgs.append(_assistant(f"answer {i}"))
    out = build_recap(msgs)
    # We scoped to the 20-turn window but show "of 30/30 total".
    assert "of 30/30 total" in out


def test_multimodal_content_blocks_flattened():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "check this file"},
                {"type": "image_url", "image_url": {"url": "..."}},
            ],
        },
        _assistant("Looked at your image."),
    ]
    out = build_recap(msgs)
    assert "check this file" in out
    assert "Looked at your image" in out


def test_handles_arguments_as_dict_not_string():
    # Some providers return arguments already as a dict.
    msgs = [
        _user("go"),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "patch",
                        "arguments": {"path": "foo.py"},
                    },
                }
            ],
        },
    ]
    out = build_recap(msgs)
    assert "patch×1" in out
    assert "foo.py" in out


def test_no_assistant_activity_hint():
    out = build_recap([_user("just sent my first message")])
    assert "no assistant activity" in out or "Last ask" in out


def test_tool_message_count_reported():
    msgs = [
        _user("go"),
        _assistant(tool_calls=[_tool_call("read_file", {"path": "a"})]),
        _tool_result(),
        _tool_result(),
        _assistant("done"),
    ]
    out = build_recap(msgs)
    assert "2 tool result" in out


def test_ignores_non_mapping_entries_gracefully():
    msgs = [None, "stray", _user("hi"), _assistant("hello")]
    # Should not raise.
    out = build_recap(msgs)
    assert "Session recap" in out
