"""`pm cli` progress renders in place on a terminal and line-per-tick off one."""

from __future__ import annotations

import io

import pytest

from pm import cli


def _capture(monkeypatch, *, tty: bool) -> io.StringIO:
    """Return the stream progress writes to: the live stream on a TTY, or
    stdout (via print) off one."""
    if tty:
        stream = io.StringIO()
        monkeypatch.setattr(cli, "_progress_stream", lambda: stream)
        return stream
    stdout = io.StringIO()
    monkeypatch.setattr(cli, "_progress_stream", lambda: None)
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    return stdout


def test_progress_redraws_in_place_on_a_tty(monkeypatch):
    stream = _capture(monkeypatch, tty=True)
    progress = cli._live_progress("node")
    progress("download", 0, 100 * 1024 * 1024, "")
    progress("download", 100 * 1024 * 1024, 100 * 1024 * 1024, "")
    progress.finish()
    out = stream.getvalue()
    assert "\r\x1b[2K" in out
    assert "\n" not in out  # redraws, never wraps until the caller prints
    assert "100.0%" in out


def test_progress_off_a_tty_prints_throttled_newline_lines(monkeypatch):
    stream = _capture(monkeypatch, tty=False)
    progress = cli._live_progress("node")
    progress("download", 0, 100 * 1024 * 1024, "")
    progress("download", 1 * 1024 * 1024, 100 * 1024 * 1024, "")  # < 4 MiB: dropped
    progress("download", 8 * 1024 * 1024, 100 * 1024 * 1024, "")
    progress("download", 100 * 1024 * 1024, 100 * 1024 * 1024, "")  # done always prints
    lines = [line for line in stream.getvalue().split("\n") if line]
    assert len(lines) == 2
    assert all("\r" not in line for line in lines)


def test_unpack_phase_is_a_phase_line_in_both_modes(monkeypatch):
    stream = _capture(monkeypatch, tty=False)
    progress = cli._live_progress("node")
    progress("unpack", 0, 0, "1/2")
    assert "unpacking 1/2" in stream.getvalue()


def test_progress_stream_prefers_stdout_then_stderr(monkeypatch):
    monkeypatch.setattr(cli, "_enable_vt", lambda stream: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: False)
    assert cli._progress_stream() is None
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True)
    assert cli._progress_stream() is cli.sys.stderr
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert cli._progress_stream() is cli.sys.stdout


def test_interactive_tracks_any_terminal(monkeypatch):
    monkeypatch.setattr(cli, "_enable_vt", lambda stream: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: False)
    assert cli._interactive() is False
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True)
    assert cli._interactive() is True


@pytest.mark.platforms("windows")
def test_windows_tty_without_a_console_gets_plain_lines(monkeypatch):
    """NUL is a character device (isatty() is True) with no console mode: an
    escape sequence there would be garbage, so progress falls back to lines."""
    with open("NUL", "w") as nul:
        assert nul.isatty()
        monkeypatch.setattr(cli.sys, "stdout", nul)
        monkeypatch.setattr(cli.sys, "stderr", nul)
        assert cli._progress_stream() is None
