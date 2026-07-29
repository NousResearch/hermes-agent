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










def test_tool_preview_length_truncates_long_user_prompt():
    long = "x " * 500
    out = build_recap([_user(long)])
    ask_line = [l for l in out.splitlines() if "Last ask" in l][0]
    assert len(ask_line) < 300  # truncated with ellipsis
    assert "…" in ask_line




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
