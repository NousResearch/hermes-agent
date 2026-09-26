"""display.bell_on_prompt / bell_on_complete also drive OSC 9 + Warp OSC 777 via _ring_bell."""

import json

import pytest

from cli import HermesCLI
from hermes_cli import terminal_notify

_WARP_OK = {
    "TERM_PROGRAM": "WarpTerminal",
    "WARP_CLI_AGENT_PROTOCOL_VERSION": "1",
    "WARP_CLIENT_VERSION": "v0.2026.08.01.00.00.stable_01",
}


def _ring(monkeypatch, *, flag_on, env, **kwargs):
    for key in _WARP_OK:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    cli = HermesCLI.__new__(HermesCLI)
    cli.bell_on_prompt = flag_on
    cli.session_id = "sess-1"
    cli._ring_bell(prompt=True, **kwargs)
    return "".join(written)


def test_osc9_body_emitted_and_sanitized_only_when_flag_on(monkeypatch):
    out = _ring(monkeypatch, flag_on=True, env={}, context="approval\x1b\x07\x00\x7f!")
    assert out == "\a\x1b]9;Hermes: approval!\x07"
    assert _ring(monkeypatch, flag_on=False, env={}, context="approval") == ""


def test_ghostty_gets_the_titled_osc777_instead_of_osc9(monkeypatch):
    """Ghostty's OSC 9 has no title field, so it gets rxvt OSC 777 — and NOT both (it would notify twice)."""
    out = _ring(monkeypatch, flag_on=True, env={"TERM_PROGRAM": "ghostty"}, context="approval",
                detail="rm -rf build")
    assert out == "\a\x1b]777;notify;Hermes;Hermes: approval\x07"
    assert "\x1b]9;" not in out
    # TERM=xterm-ghostty is the same terminal (TERM_PROGRAM is masked under tmux/screen).
    assert _ring(monkeypatch, flag_on=True, env={"TERM": "xterm-ghostty"},
                 context="approval") == "\a\x1b]777;notify;Hermes;Hermes: approval\x07"
    # Semicolons are the field separator: they must not shift the body into the title.
    assert _ring(monkeypatch, flag_on=True, env={"TERM_PROGRAM": "ghostty"},
                 context="a;b") == "\a\x1b]777;notify;Hermes;Hermes: a,b\x07"


def test_warp_osc777_only_under_supported_warp_build(monkeypatch):
    out = _ring(monkeypatch, flag_on=True, env=_WARP_OK, context="approval", detail="rm -rf build")
    prefix = "\x1b]777;notify;warp://cli-agent;"
    assert out.count(prefix) == 1
    payload = json.loads(out.split(prefix, 1)[1].rstrip("\x07"))
    assert payload["agent"] == "hermes"
    assert payload["event"] == "permission_request"
    assert payload["summary"] == "rm -rf build"
    assert payload["session_id"] == "sess-1"
    assert payload["v"] == 1
    # Broken build (advertises the protocol var but can't render) → OSC 9 only.
    broken = dict(_WARP_OK, WARP_CLIENT_VERSION="v0.2026.03.25.08.24.stable_05")
    assert prefix not in _ring(monkeypatch, flag_on=True, env=broken, context="approval")
    # Not Warp at all → OSC 9 only.
    not_warp = dict(_WARP_OK, TERM_PROGRAM="ghostty")
    assert prefix not in _ring(monkeypatch, flag_on=True, env=not_warp, context="approval")


def test_running_app_gets_bell_and_osc9_on_its_loop_never_a_second_tty_writer(monkeypatch):
    """With the prompt_toolkit app live, the bell + notification must reach the tty through the
    app's output ON THE APP LOOP, never via a second writer from the calling thread."""
    for key in _WARP_OK:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(terminal_notify, "write_tty", lambda seq: pytest.fail(f"stray tty write: {seq!r}"))

    class _Output:
        raw = []

        def write_raw(self, data):
            self.raw.append(data)

        def flush(self):
            self.raw.append("<flush>")

    class _Loop:
        queued = []

        def call_soon_threadsafe(self, fn):
            self.queued.append(fn)

    class _App:
        _is_running = True
        loop = _Loop()
        output = _Output()

    cli = HermesCLI.__new__(HermesCLI)
    cli.bell_on_complete = True
    cli.session_id = "sess-1"
    cli._app = _App()
    cli._ring_bell(context="turn complete")
    # Nothing touched the tty from the calling thread; the write is queued for the loop.
    assert _Output.raw == []
    assert len(_Loop.queued) == 1
    _Loop.queued[0]()
    assert _Output.raw == ["\a\x1b]9;Hermes: turn complete\x07", "<flush>"]
