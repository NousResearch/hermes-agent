"""Tests for cli._cprint's console-less stdout fallback (#65558).

Background: prompt_toolkit's ``create_output()`` falls back to a plain-text
output on POSIX when stdout is not a tty, but its win32 branch
unconditionally builds ``Win32Output``, which raises
``NoConsoleScreenBufferError`` when stdout is a pipe or file — the normal
state for any automation wrapper capturing ``hermes chat -q`` output on
Windows.  ``_cprint`` now short-circuits that case to ``_plain_print``,
which strips ANSI styling (parity with the POSIX plain-text output) and
degrades encoding failures (cp1252 pipes without ``PYTHONIOENCODING``)
via ``errors="replace"`` instead of crashing or dropping the line.

These tests are hermetic: no real console, no Windows host required.
"""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import pytest

import cli


@pytest.fixture(autouse=True)
def reset_output_history():
    cli._configure_output_history(False, 200)
    yield
    cli._configure_output_history(True, 200)


# ---------------------------------------------------------------------------
# _win32_stdout_lacks_console detection
# ---------------------------------------------------------------------------


def test_detection_false_on_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli._win32_stdout_lacks_console() is False


def test_detection_true_on_win32_pipe(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: False))
    assert cli._win32_stdout_lacks_console() is True


def test_detection_false_on_win32_console(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    assert cli._win32_stdout_lacks_console() is False


def test_detection_treats_isatty_failure_as_console_less(monkeypatch):
    def _boom():
        raise ValueError("closed stdout")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=_boom))
    assert cli._win32_stdout_lacks_console() is True


# ---------------------------------------------------------------------------
# _cprint routing
# ---------------------------------------------------------------------------


def test_cprint_win32_piped_stdout_short_circuits_to_plain_print(
    monkeypatch, capsys
):
    """Windows + non-tty stdout → plain ANSI-stripped print, no prompt_toolkit."""
    pt_calls = []
    monkeypatch.setattr(cli, "_pt_print", lambda x: pt_calls.append(x))
    monkeypatch.setattr(sys, "platform", "win32")

    cli._cprint("\x1b[2;3mInitializing agent...\x1b[0m")

    assert pt_calls == []
    out = capsys.readouterr().out
    assert out == "Initializing agent...\n"
    assert "\x1b[" not in out


def test_cprint_posix_piped_stdout_keeps_prompt_toolkit_path(monkeypatch):
    """POSIX behavior is unchanged: prompt_toolkit handles non-tty itself."""
    pt_calls = []
    monkeypatch.setattr(cli, "_pt_print", lambda x: pt_calls.append(x))
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)
    monkeypatch.setattr(sys, "platform", "linux")

    cli._cprint("hello")

    assert pt_calls == ["hello"]


def test_cprint_win32_console_stdout_keeps_prompt_toolkit_path(monkeypatch):
    """Windows with a real console keeps the styled prompt_toolkit path."""
    pt_calls = []
    monkeypatch.setattr(cli, "_pt_print", lambda x: pt_calls.append(x))
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        sys, "stdout", SimpleNamespace(isatty=lambda: True, encoding="utf-8")
    )

    cli._cprint("styled")

    assert pt_calls == ["styled"]


def test_cprint_pt_print_failure_falls_back_to_stripped_plain_print(
    monkeypatch, capsys
):
    """Detection miss (isatty True, console gone) → exception fallback strips ANSI.

    Before, the fallback printed the raw payload, leaking escape sequences
    into piped stdout.
    """

    def _no_console(_):
        raise OSError("No Windows console found. Are you running cmd.exe?")

    monkeypatch.setattr(cli, "_pt_print", _no_console)

    cli._cprint("\x1b[1mbold status\x1b[0m")

    out = capsys.readouterr().out
    assert out == "bold status\n"
    assert "\x1b[" not in out


# ---------------------------------------------------------------------------
# _plain_print encoding degradation
# ---------------------------------------------------------------------------


def test_plain_print_replaces_unencodable_chars_instead_of_raising(monkeypatch):
    """cp1252 pipe + braille spinner char → replaced output, line not dropped."""
    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", wrapper)

    cli._plain_print("spinner ⠋ frame")

    wrapper.flush()
    assert buf.getvalue() == b"spinner ? frame\n"


def test_plain_print_keep_ansi_preserves_payload(monkeypatch, capsys):
    """keep_ansi=True (the -Q final-response path) must not alter the text."""
    cli._plain_print("exact model text", keep_ansi=True)

    assert capsys.readouterr().out == "exact model text\n"


def test_plain_print_strips_osc_and_csi_sequences(capsys):
    cli._plain_print("\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\ \x1b[31mred\x1b[0m")

    assert capsys.readouterr().out == "link red\n"
