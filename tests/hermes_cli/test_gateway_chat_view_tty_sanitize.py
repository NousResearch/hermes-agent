"""`chat -Q` / `-z` print model text through GatewayChatView: on a real terminal only SGR styling may
reach it (no OSC 52 clipboard write, OSC 0 title, screen clear); a pipe gets the bytes unchanged."""
import os
import sys

import pytest

ESC = "\x1b"
HOSTILE = (f"start {ESC}]0;PWNED{ESC}\\{ESC}]52;c;UFdORUQ={chr(7)}"
           f"{ESC}[31mred{ESC}[0m {ESC}[2J{ESC}[H\x07end")


def _settle(text, stdout):
    from hermes_cli.gateway_chat_view import GatewayChatView
    view = GatewayChatView(None, {"stored_session_id": "s"})
    old, sys.stdout = sys.stdout, stdout
    try:
        view._delta("a1", {"text": text[:20]})
        view._complete("a1", {"text": text})
    finally:
        sys.stdout = old


def _read_all(fd):
    out = b""
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            return out
        if not chunk:
            return out
        out += chunk


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a pty")
def test_reply_on_a_tty_keeps_only_sgr():
    master, slave = os.openpty()
    os.set_blocking(master, False)
    with os.fdopen(slave, "w", encoding="utf-8", closefd=True) as tty:
        assert tty.isatty()
        _settle(HOSTILE, tty)
    shown = _read_all(master).decode()
    os.close(master)
    assert "start" in shown and "end" in shown
    assert f"{ESC}[31mred{ESC}[0m" in shown, "SGR styling must survive"
    for bad in (f"{ESC}]", f"{ESC}[2J", f"{ESC}[H", "\x07", "PWNED"):
        assert bad not in shown, f"{bad!r} reached the terminal: {shown!r}"


def test_reply_to_a_pipe_is_unchanged(tmp_path):
    path = tmp_path / "out.txt"
    with open(path, "w", encoding="utf-8") as pipe:
        _settle(HOSTILE, pipe)
    assert path.read_text(encoding="utf-8").strip("\n") == HOSTILE
