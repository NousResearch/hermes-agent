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
import types
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
# _emit_pt_or_plain: the single guarded emission point
# ---------------------------------------------------------------------------


def test_emit_pt_or_plain_uses_prompt_toolkit_on_success(monkeypatch):
    pt_calls = []
    monkeypatch.setattr(cli, "_pt_print", lambda x: pt_calls.append(x))
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: ("ANSI", t))

    cli._emit_pt_or_plain("hi")

    assert pt_calls == [("ANSI", "hi")]


def test_emit_pt_or_plain_degrades_to_stripped_plain(monkeypatch, capsys):
    def _no_console(_):
        raise OSError("No Windows console found. Are you running cmd.exe?")

    monkeypatch.setattr(cli, "_pt_print", _no_console)

    cli._emit_pt_or_plain("\x1b[1mstatus\x1b[0m")

    out = capsys.readouterr().out
    assert out == "status\n"
    assert "\x1b[" not in out


# ---------------------------------------------------------------------------
# Active-application renderer failures (detection miss: isatty True, no console)
# ---------------------------------------------------------------------------


def _install_active_app(monkeypatch, loop, *, running_loop):
    """Wire up a running prompt_toolkit app whose loop is ``loop``.

    ``running_loop`` is what ``asyncio.get_running_loop()`` returns inside
    ``_cprint`` — pass ``loop`` to take the same-thread branch, or a
    different object to take the cross-thread scheduling branch.
    """
    fake_asyncio = types.ModuleType("asyncio")
    fake_asyncio.get_running_loop = lambda: running_loop
    fake_asyncio.ensure_future = lambda coro: None
    monkeypatch.setitem(sys.modules, "asyncio", fake_asyncio)

    fake_app = SimpleNamespace(_is_running=True, loop=loop)
    fake_pt_app = types.ModuleType("prompt_toolkit.application")
    fake_pt_app.get_app_or_none = lambda: fake_app

    def _run_in_terminal(func, **kw):
        func()  # PT executes the emission synchronously in this fake
        return None

    fake_pt_app.run_in_terminal = _run_in_terminal
    monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)


def test_cprint_active_app_same_loop_renderer_failure_degrades(monkeypatch, capsys):
    """Same-thread active-app arm: renderer failure falls back to plain print."""

    def _no_console(_):
        raise OSError("No Windows console found. Are you running cmd.exe?")

    monkeypatch.setattr(cli, "_pt_print", _no_console)

    class FakeLoop:
        def is_running(self):
            return True

        def call_soon_threadsafe(self, cb, *a):
            raise AssertionError("same-thread path must not schedule")

    loop = FakeLoop()
    _install_active_app(monkeypatch, loop, running_loop=loop)

    cli._cprint("\x1b[31msame-loop status\x1b[0m")

    out = capsys.readouterr().out
    assert out == "same-loop status\n"
    assert "\x1b[" not in out


def test_cprint_active_app_cross_thread_renderer_failure_degrades(monkeypatch, capsys):
    """Cross-thread arm: renderer failure inside run_in_terminal degrades."""

    def _no_console(_):
        raise OSError("No Windows console found. Are you running cmd.exe?")

    monkeypatch.setattr(cli, "_pt_print", _no_console)

    scheduled = []

    class FakeLoop:
        def is_running(self):
            return True

        def call_soon_threadsafe(self, cb, *a):
            scheduled.append(cb)

    loop = FakeLoop()
    # running_loop differs from the app loop → cross-thread branch.
    other = SimpleNamespace(is_running=lambda: True)
    _install_active_app(monkeypatch, loop, running_loop=other)

    cli._cprint("\x1b[32mcross-thread status\x1b[0m")

    assert len(scheduled) == 1
    scheduled[0]()  # run the scheduled callback → run_in_terminal → emit

    out = capsys.readouterr().out
    assert out == "cross-thread status\n"
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
