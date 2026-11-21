"""display.bell_on_prompt / bell_on_complete also drive OSC 9 + Warp OSC 777 via _ring_bell."""

import json
import threading

import pytest

from cli import HermesCLI
from hermes_cli import cli_terminal_mixin as terminal_mixin
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


def test_program_status_protocol_is_herdr_terminal_and_tty_gated():
    supported = terminal_notify.program_status_supported
    assert supported({"HC_TERMINAL_ID": "t4"}, is_tty=True)
    assert not supported({}, is_tty=True)
    assert not supported({"HC_TERMINAL_ID": "t4"}, is_tty=False)


def test_program_status_sequence_uses_only_escape_safe_fixed_fields():
    encode = terminal_notify.program_status_sequence
    assert encode("working") == "\x1b]7501;state=working:app=hermes\x1b\\"
    assert encode("blocked", kind="permission") == (
        "\x1b]7501;state=blocked:kind=permission:app=hermes\x1b\\"
    )
    assert encode("clear") == "\x1b]7501;state=clear:app=hermes\x1b\\"
    with pytest.raises(ValueError):
        encode("bogus")


def test_running_app_gets_program_status_on_its_loop(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_ID", "t4")
    monkeypatch.setattr(terminal_notify.os, "isatty", lambda fd: True)
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
    cli._app = _App()
    cli._terminal_io_broken = False
    cli._report_program_status("blocked", kind="question")
    assert _Output.raw == []
    assert len(_Loop.queued) == 1
    _Loop.queued[0]()
    assert _Output.raw == [
        "\x1b]7501;state=blocked:kind=question:app=hermes\x1b\\",
        "<flush>",
    ]


def test_program_status_is_silent_outside_herdr(monkeypatch):
    monkeypatch.delenv("HC_TERMINAL_ID", raising=False)
    monkeypatch.setattr(terminal_notify, "write_tty", lambda seq: pytest.fail(f"unexpected write: {seq!r}"))

    cli = HermesCLI.__new__(HermesCLI)
    cli._terminal_io_broken = False
    cli._report_program_status("working")


def test_program_status_drops_when_app_loop_rejects_scheduling(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_ID", "t4")
    monkeypatch.setattr(terminal_notify.os, "isatty", lambda fd: True)
    monkeypatch.setattr(terminal_notify, "write_tty", lambda seq: pytest.fail(f"stray tty write: {seq!r}"))

    class _Loop:
        def call_soon_threadsafe(self, fn):
            raise RuntimeError("loop closed")

    class _Output:
        def write_raw(self, data):
            pytest.fail(f"worker-thread write: {data!r}")

    cli = HermesCLI.__new__(HermesCLI)
    cli._app = type("App", (), {"_is_running": True, "loop": _Loop(), "output": _Output()})()
    cli._terminal_io_broken = False
    cli._report_program_status("working")


def test_clear_latches_and_suppresses_late_status(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_ID", "t4")
    monkeypatch.setattr(terminal_notify.os, "isatty", lambda fd: True)
    written = []
    monkeypatch.setattr(terminal_notify, "write_tty", written.append)

    cli = HermesCLI.__new__(HermesCLI)
    cli._app = None
    cli._terminal_io_broken = False
    cli._report_program_status("clear")
    cli._report_program_status("working")

    assert written == ["\x1b]7501;state=clear:app=hermes\x1b\\"]


def test_direct_status_writes_hold_the_shutdown_serialization_lock(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_ID", "t4")
    monkeypatch.setattr(terminal_notify.os, "isatty", lambda fd: True)
    written = []

    def write_while_locked(seq):
        assert terminal_mixin._PROGRAM_STATUS_WRITE_LOCK.locked()
        written.append(seq)

    monkeypatch.setattr(terminal_notify, "write_tty", write_while_locked)

    cli = HermesCLI.__new__(HermesCLI)
    cli._app = None
    cli._terminal_io_broken = False
    cli._report_program_status("working")
    cli._report_program_status("clear")

    assert written == [
        "\x1b]7501;state=working:app=hermes\x1b\\",
        "\x1b]7501;state=clear:app=hermes\x1b\\",
    ]


def test_clear_suppresses_status_that_was_already_racing_to_the_app_loop(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_ID", "t4")
    monkeypatch.setattr(terminal_notify.os, "isatty", lambda fd: True)
    worker_scheduling = threading.Event()
    release_worker = threading.Event()
    queued = []
    written = []

    class _Loop:
        def call_soon_threadsafe(self, fn):
            if threading.current_thread() is worker:
                worker_scheduling.set()
                assert release_worker.wait(timeout=2)
            queued.append(fn)

    class _Output:
        def write_raw(self, data):
            written.append(data)

        def flush(self):
            pass

    cli = HermesCLI.__new__(HermesCLI)
    cli._app = type("App", (), {"_is_running": True, "loop": _Loop(), "output": _Output()})()
    cli._terminal_io_broken = False

    worker = threading.Thread(target=lambda: cli._report_program_status("working"))
    worker.start()
    assert worker_scheduling.wait(timeout=2)
    cli._report_program_status("clear")
    release_worker.set()
    worker.join(timeout=2)
    assert not worker.is_alive()

    for fn in queued:
        fn()

    assert written == ["\x1b]7501;state=clear:app=hermes\x1b\\"]
