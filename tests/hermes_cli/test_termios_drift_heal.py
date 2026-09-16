"""Regression tests for cooked-mode termios drift healing (cli._heal_cooked_mode_drift).

Failure class: prompt_toolkit's ``run_in_terminal`` flips the tty to cooked
mode for every "print above the prompt" window and restores raw mode after.
If that restore is lost (cancelled coroutine, racing chained windows, an
external actor), the terminal stays in cooked mode while the prompt_toolkit
Application still believes it owns raw mode — the kernel line-buffers
keystrokes and the CLI appears to stop taking input even though the process
is healthy. Observed live on 2026-08-13 (session 20260813_113539_0f5b00):
pts left in ``icanon echo`` after a background-review + bg-notification
sequence, healed externally with stty.

These tests exercise the healer against a REAL pty pair — no mocks.
"""

import os
import sys

import pytest

if sys.platform == "win32":  # pragma: no cover
    pytest.skip("termios drift healing is POSIX-only", allow_module_level=True)

import termios
import tty as _tty

from cli import _heal_cooked_mode_drift


@pytest.fixture()
def pty_fd():
    master, slave = os.openpty()
    try:
        yield slave
    finally:
        os.close(master)
        os.close(slave)


def _is_cooked(fd: int) -> bool:
    lflag = termios.tcgetattr(fd)[3]
    return bool(lflag & (termios.ICANON | termios.ECHO))


def test_heals_cooked_drift_back_to_raw(pty_fd):
    # ptys start in cooked mode — that IS the drifted state.
    assert _is_cooked(pty_fd), "precondition: fresh pty must be cooked"

    healed = _heal_cooked_mode_drift(pty_fd)

    assert healed is True
    attrs = termios.tcgetattr(pty_fd)
    lflag, iflag, cc = attrs[3], attrs[0], attrs[6]
    assert not (lflag & termios.ICANON)
    assert not (lflag & termios.ECHO)
    assert not (lflag & termios.ISIG)
    assert not (lflag & termios.IEXTEN)
    # iflag surgery matches prompt_toolkit raw_mode._patch_iflag
    for flag in (termios.IXON, termios.IXOFF, termios.ICRNL,
                 termios.INLCR, termios.IGNCR):
        assert not (iflag & flag)
    assert cc[termios.VMIN] == 1


def test_noop_when_already_raw(pty_fd):
    _tty.setraw(pty_fd)
    before = termios.tcgetattr(pty_fd)

    healed = _heal_cooked_mode_drift(pty_fd)

    assert healed is False, "already-raw tty must not be treated as drifted"
    assert termios.tcgetattr(pty_fd) == before, "raw tty attrs must be untouched"


def test_returns_false_on_non_tty():
    r, w = os.pipe()
    try:
        assert _heal_cooked_mode_drift(r) is False
        assert _heal_cooked_mode_drift(w) is False
    finally:
        os.close(r)
        os.close(w)


def test_heal_preserves_unrelated_flags(pty_fd):
    # Set a flag the healer must not clobber (OPOST in oflag).
    attrs = termios.tcgetattr(pty_fd)
    attrs[1] |= termios.OPOST
    termios.tcsetattr(pty_fd, termios.TCSANOW, attrs)

    assert _heal_cooked_mode_drift(pty_fd) is True

    after = termios.tcgetattr(pty_fd)
    assert after[1] & termios.OPOST, "oflag must be preserved by the heal"
