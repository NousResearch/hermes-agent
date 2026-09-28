"""display.bell_on_prompt / bell_on_complete also drive OSC 9 + Warp OSC 777 via _ring_bell.

display.notify_on_interact additionally rings blocking prompts (independently of
bell_on_prompt) and fires the paplay fallback sound (#25022)."""

import json

import pytest

from cli import HermesCLI
from hermes_cli import terminal_notify

_WARP_OK = {
    "TERM_PROGRAM": "WarpTerminal",
    "WARP_CLI_AGENT_PROTOCOL_VERSION": "1",
    "WARP_CLIENT_VERSION": "v0.2026.08.01.00.00.stable_01",
}


def _ring(monkeypatch, *, flag_on, env, sound_calls=None, **kwargs):
    for key in _WARP_OK:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)
    if sound_calls is not None:
        monkeypatch.setattr(terminal_notify, "play_attention_sound", lambda: sound_calls.append(True))
    cli = HermesCLI.__new__(HermesCLI)
    cli.bell_on_prompt = flag_on
    cli.session_id = "sess-1"
    cli._ring_bell(prompt=True, **kwargs)
    return "".join(written)


def test_osc9_body_emitted_and_sanitized_only_when_flag_on(monkeypatch):
    out = _ring(monkeypatch, flag_on=True, env={}, context="approval\x1b\x07\x00\x7f!")
    assert out == "\a\x1b]9;Hermes: approval!\x07"
    assert _ring(monkeypatch, flag_on=False, env={}, context="approval") == ""


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


def test_notify_on_interact_rings_prompt_bell_and_sound_without_bell_on_prompt(monkeypatch):
    """display.notify_on_interact (#25022): a blocking prompt rings (BEL + OSC 9) and fires
    the paplay sound even when bell_on_prompt is off — and stays silent when both are off."""
    for key in _WARP_OK:
        monkeypatch.delenv(key, raising=False)
    sounds = []

    def _ring_notify(*, notify_on_interact, bell_on_prompt):
        written = []
        monkeypatch.setattr(terminal_notify, "write_tty", written.append)
        monkeypatch.setattr(terminal_notify, "play_attention_sound", lambda: sounds.append(True))
        cli = HermesCLI.__new__(HermesCLI)
        cli.bell_on_prompt = bell_on_prompt
        cli.notify_on_interact = notify_on_interact
        cli.session_id = "sess-1"
        cli._ring_bell(prompt=True, context="approval")
        return "".join(written)

    # notify_on_interact alone is enough for the prompt bell + sound.
    out = _ring_notify(notify_on_interact=True, bell_on_prompt=False)
    assert out.startswith("\a\x1b]9;Hermes: approval")
    assert sounds == [True]

    # Both off → silent, no sound.
    sounds.clear()
    assert _ring_notify(notify_on_interact=False, bell_on_prompt=False) == ""
    assert sounds == []

    # bell_on_prompt alone keeps its existing behavior and also gets the sound.
    sounds.clear()
    out = _ring_notify(notify_on_interact=False, bell_on_prompt=True)
    assert out.startswith("\a")
    assert sounds == [True]


def test_play_attention_sound_linux_only(monkeypatch):
    """The paplay fallback (#25022) only spawns on Linux and absorbs spawn errors."""
    calls = []

    def _fake_popen(args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(terminal_notify.subprocess, "Popen", _fake_popen)
    terminal_notify.play_attention_sound("linux")
    assert len(calls) == 1
    assert calls[0][0][0] == "paplay"
    assert calls[0][0][1] == terminal_notify.ATTENTION_SOUND
    # start_new_session detaches: a dead TUI must not kill the sound mid-play.
    assert calls[0][1]["start_new_session"] is True

    # Non-Linux → no spawn at all.
    calls.clear()
    terminal_notify.play_attention_sound("darwin")
    assert calls == []

    # Spawn failure (no paplay on the box) is absorbed.
    def _boom(args, **kwargs):
        raise FileNotFoundError("paplay")

    monkeypatch.setattr(terminal_notify.subprocess, "Popen", _boom)
    terminal_notify.play_attention_sound("linux")  # must not raise
