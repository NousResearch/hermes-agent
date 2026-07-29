"""Unit tests for hermes_cli.session_recap."""
from __future__ import annotations

import json


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


def test_ignores_non_mapping_entries_gracefully():
    msgs = [None, "stray", _user("hi"), _assistant("hello")]
    # Should not raise.
    out = build_recap(msgs)
    assert "Session recap" in out


def test_escape_sequences_sanitized_in_previews():
    """Recap previews must not carry raw terminal escapes (codex#31494 class)."""
    msgs = [
        _user("please \x1b[2J\x1b]0;pwned\x07 do the thing"),
        _assistant("done \x9b31m with it\x07"),
    ]
    out = build_recap(msgs)
    assert "\x1b" not in out
    assert "\x9b" not in out
    assert "\x07" not in out
    assert "do the thing" in out
    assert "with it" in out
