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


def test_recent_line_reports_windowed_tool_count_not_whole_session():
    # 30 [user, assistant, tool] turns (90 messages). The recent 20-turn window
    # (user+assistant) covers ~10 turns, so ~10 tool results — but the "Recent:"
    # line used to print the whole-session count (30), overstating activity by
    # the entire out-of-window history.
    msgs = []
    for i in range(30):
        msgs.append(_user(f"question {i}"))
        msgs.append(_assistant(f"answer {i}"))
        msgs.append(_tool_result(f"result {i}"))
    out = build_recap(msgs)
    recent_line = next(line for line in out.splitlines() if "Recent:" in line)
    # Windowed count is reported; the global total rides along as an annotation.
    assert "10 tool results" in recent_line
    assert "(of 30 total)" in recent_line
    assert "30 tool results" not in recent_line
