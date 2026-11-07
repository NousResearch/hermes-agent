"""Program Status Protocol (OSC 7501) reports from the classic CLI."""
import os

import pytest

from hermes_cli import program_status


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(program_status, "_last", None)
    monkeypatch.delenv(program_status._ENV, raising=False)
    monkeypatch.delenv("HERDR_ENV", raising=False)
    monkeypatch.delenv("HC_TERMINAL_SOCKET", raising=False)
    yield


def _capture(monkeypatch):
    written = []

    class _Fd:
        def __init__(self, path, flags):
            assert path == "/dev/tty"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(program_status.os, "open", lambda path, flags: 7)
    monkeypatch.setattr(program_status.os, "close", lambda fd: None)
    monkeypatch.setattr(program_status.os, "write", lambda fd, data: written.append(data) or len(data))
    return written


def test_quiet_outside_herdr(monkeypatch):
    written = _capture(monkeypatch)
    program_status.working()
    assert written == []


def test_reports_in_herdr_app(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/tmp/herdr.sock")
    written = _capture(monkeypatch)
    program_status.working()
    assert written == [b"\x1b]7501;state=working:app=hermes\x1b\\"]


def test_reports_in_herdr_tui(monkeypatch):
    monkeypatch.setenv("HERDR_ENV", "1")
    written = _capture(monkeypatch)
    program_status.idle()
    assert written == [b"\x1b]7501;state=idle:app=hermes\x1b\\"]


def test_blocked_carries_kind(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/tmp/herdr.sock")
    written = _capture(monkeypatch)
    program_status.blocked("permission")
    assert written == [b"\x1b]7501;state=blocked:app=hermes:kind=permission\x1b\\"]


def test_repeated_state_is_skipped(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/tmp/herdr.sock")
    written = _capture(monkeypatch)
    program_status.working()
    program_status.working()
    program_status.done()
    assert len(written) == 2


def test_env_opt_out(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/tmp/herdr.sock")
    monkeypatch.setenv(program_status._ENV, "0")
    written = _capture(monkeypatch)
    program_status.working()
    assert written == []


def test_env_opt_in_anywhere(monkeypatch):
    monkeypatch.setenv(program_status._ENV, "1")
    written = _capture(monkeypatch)
    program_status.done()
    assert written == [b"\x1b]7501;state=done:app=hermes\x1b\\"]


def test_write_failure_is_swallowed(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/tmp/herdr.sock")

    def _boom(path, flags):
        raise OSError("no tty")

    monkeypatch.setattr(program_status.os, "open", _boom)
    program_status.working()  # must not raise
