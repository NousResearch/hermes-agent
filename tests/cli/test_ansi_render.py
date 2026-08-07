"""Tests for ANSI color rendering, focused on issue #59397.

Issue: ``hermes setup`` and ``hermes model`` printed literal ``[35m`` /
``[0m`` text fragments in Windows CMD / PowerShell instead of rendering
the intended magenta-colored box-drawing banner.

Root cause: Windows console hosts default to "literal mode" — the ESC
byte (``\x1b``) is silently consumed and only the bracket-form of the
escape sequence is visible.  The fix is in :mod:`hermes_cli.stdio`:
``configure_windows_stdio()`` now flips the console into
``ENABLE_VIRTUAL_TERMINAL_PROCESSING`` mode so ANSI escapes are
interpreted.

This test file exercises the contract end-to-end:

  1. Round-trip of the existing ``hermes_cli.colors.color()`` helper —
     when NOT a TTY (CI mode), codes are stripped; when "is" a TTY
     (mocked), codes are preserved.
  2. The specific broken-render pattern from #59397: ``print()`` of a
     colored string never produces a literal ``[Nm`` fragment in any
     configuration we test.
  3. The Windows stdio bootstrap actually flips the console-mode flag
     that turns literal escapes into rendering, with a simulated kernel32.
"""

from __future__ import annotations

import ctypes
import io
import os
import sys

import pytest

import hermes_cli.colors as colors
import hermes_cli.stdio as hermes_stdio


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


class _FakeKernel32:
    """In-memory stand-in for ``ctypes.windll.kernel32`` console APIs.

    Records every ``SetConsoleMode`` call and pretends stdout / stderr
    are attached console handles.  Used by the VT-processing tests so we
    can run them on any platform (the real ``kernel32`` doesn't exist
    off-Windows, and even on Windows it requires a real console session).
    """

    def __init__(
        self,
        *,
        stdout_mode: int = 0x0001,
        stderr_mode: int = 0x0001,
        stdout_get_console_mode: bool = True,
        stderr_get_console_mode: bool = True,
        stdout_handle: int | None = -11,
        stderr_handle: int | None = -12,
    ) -> None:
        self._modes = {
            hermes_stdio._STD_OUTPUT_HANDLE: stdout_mode,
            hermes_stdio._STD_ERROR_HANDLE: stderr_mode,
        }
        self._get_console_mode_results = {
            hermes_stdio._STD_OUTPUT_HANDLE: stdout_get_console_mode,
            hermes_stdio._STD_ERROR_HANDLE: stderr_get_console_mode,
        }
        self._handles = {
            hermes_stdio._STD_OUTPUT_HANDLE: stdout_handle,
            hermes_stdio._STD_ERROR_HANDLE: stderr_handle,
        }
        self.set_modes: list[tuple[int, int]] = []

    # kernel32 surface used by the helper.  Each method matches the real
    # kernel32 signature closely enough to drive the helper without
    # dragging ctypes into the test.
    def GetStdHandle(self, std_handle: int):  # noqa: N802 — Win32 casing
        return self._handles[std_handle]

    def GetConsoleMode(self, handle, mode_ptr):  # noqa: N802
        if not self._get_console_results_for(handle):
            return 0
        mode_ptr._obj.value = self._modes[self._handle_key(handle)]
        return 1

    def SetConsoleMode(self, handle, mode):  # noqa: N802
        self.set_modes.append((handle, mode))
        if self._get_console_results_for(handle):
            self._modes[self._handle_key(handle)] = mode
        return 1

    # ── plumbing ──
    def _handle_key(self, handle) -> int:
        for k, v in self._handles.items():
            if v == handle:
                return k
        raise KeyError(handle)

    def _get_console_results_for(self, handle) -> bool:
        for k, v in self._handles.items():
            if v == handle:
                return self._get_console_mode_results[k]
        return False


class _FakeWindll:
    """ctypes.windll replacement exposing a ``.kernel32`` attribute."""

    def __init__(self, kernel32: _FakeKernel32) -> None:
        self.kernel32 = kernel32


class _BrokenWindll:
    """A windll-like object whose ``.kernel32`` raises on access."""

    @property
    def kernel32(self):
        raise AttributeError("no kernel32 here")


@pytest.fixture
def reset_stdio_module(monkeypatch):
    """Reset ``hermes_cli.stdio`` state between tests.

    The module caches a ``_CONFIGURED`` sentinel and a ``_default_editor``
    side-effect via ``os.environ``; we want each test to start from a
    fresh slate so the helpers under test see a clean module.
    """
    monkeypatch.delenv("HERMES_DISABLE_WINDOWS_UTF8", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    hermes_stdio._CONFIGURED = False
    yield


# ──────────────────────────────────────────────────────────────────────
# 1. Round-trip of color() — TTY vs non-TTY
# ──────────────────────────────────────────────────────────────────────


class TestColorRoundTrip:
    """``hermes_cli.colors.color()`` must drop codes when not a TTY and
    preserve them when stdout is a TTY.

    This is the foundational invariant that ``setup.py`` /
    ``cli_output.py`` rely on; if it breaks, every colored line in the
    setup wizard leaks ANSI codes into piped output.
    """

    def test_non_tty_strips_color_codes(self, monkeypatch):
        """In CI / piped-output mode, ``color()`` must return plain text."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        assert not colors.should_use_color()
        assert colors.color("hello", colors.Colors.MAGENTA) == "hello"
        assert colors.color("box", colors.Colors.GREEN, colors.Colors.BOLD) == "box"

    def test_tty_preserves_color_codes(self, monkeypatch):
        """When stdout is a TTY, ``color()`` must wrap text in ANSI codes
        with a reset suffix."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")

        assert colors.should_use_color()
        out = colors.color("hello", colors.Colors.MAGENTA)
        assert out.startswith(colors.Colors.MAGENTA)
        assert out.endswith(colors.Colors.RESET)
        assert "hello" in out

    def test_tty_bold_and_color_compose(self, monkeypatch):
        """Stacking BOLD + a color code must produce both codes around
        the text, then a single RESET suffix."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")

        out = colors.color("hi", colors.Colors.CYAN, colors.Colors.BOLD)
        assert out.startswith(colors.Colors.CYAN + colors.Colors.BOLD)
        assert out.endswith(colors.Colors.RESET)
        # Exactly one reset at the end — no double-wrap leakage.
        assert out.count(colors.Colors.RESET) == 1

    def test_no_color_env_disables_codes_even_on_tty(self, monkeypatch):
        """``NO_COLOR`` (https://no-color.org/) wins over TTY detection."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setenv("NO_COLOR", "1")

        assert not colors.should_use_color()
        assert colors.color("hello", colors.Colors.MAGENTA) == "hello"

    def test_dumb_terminal_disables_codes(self, monkeypatch):
        """``TERM=dumb`` disables colors even on a real TTY."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")

        assert not colors.should_use_color()
        assert colors.color("hello", colors.Colors.GREEN) == "hello"


# ──────────────────────────────────────────────────────────────────────
# 2. Specific broken-render case from the issue
# ──────────────────────────────────────────────────────────────────────


class TestBrokenRenderCase:
    """The exact failure mode reported in issue #59397.

    The reporter saw ``[35m`` and ``[0m`` printed as literal text in the
    setup wizard's banner — i.e. the ``\x1b`` (ESC) byte was lost while
    the bracket-form of the SGR sequence survived.

    On a properly configured console (TTY + ``ENABLE_VIRTUAL_TERMINAL_
    PROCESSING``), the byte sequence must be preserved verbatim so the
    terminal can interpret it.  The bracket-form should never appear in
    our output without a leading ESC.
    """

    def test_color_output_starts_with_esc_when_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")

        out = colors.color("⚕ Hermes Setup", colors.Colors.MAGENTA)

        # The full ESC byte must precede the bracket-form SGR.
        assert out.startswith("\x1b["), (
            "Color output lost its ESC byte — Windows console would "
            "render this as literal `[35m` text. See issue #59397."
        )
        # And the SGR sequence must be the expected magenta code.
        assert "\x1b[35m" in out
        assert out.endswith("\x1b[0m")
        # The output must be exactly ESC[35m <text> ESC[0m — no naked
        # bracket-form SGR fragments that would leak through if a
        # future refactor accidentally split the ESC off the SGR.
        assert out == f"\x1b[35m⚕ Hermes Setup\x1b[0m"

    def test_setup_banner_color_round_trip_preserves_esc(self, monkeypatch):
        """Simulate the setup wizard's banner code: ``color()`` of a
        57-char box-drawing border in magenta must keep its ESC byte
        intact and end with a reset."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")

        banner = "│  Press Ctrl+C at any time to exit.                     │"
        out = colors.color(banner, colors.Colors.MAGENTA)

        # Whole sequence: ESC[35m <banner> ESC[0m
        assert out == f"\x1b[35m{banner}\x1b[0m"

    def test_non_tty_strips_codes_no_literal_bracket_leak(self, monkeypatch):
        """Even when codes are stripped (non-TTY), there must be no
        stray ``[35m`` / ``[0m`` fragments left over — that would mean
        somebody mangled the ESC byte but forgot to drop the rest."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

        out = colors.color("hello", colors.Colors.MAGENTA)
        assert out == "hello"
        assert "[35m" not in out
        assert "[0m" not in out
        assert "\x1b" not in out

    def test_printing_colored_string_does_not_leak_brackets(self, monkeypatch, capsys):
        """End-to-end: piping a colored banner through ``print()`` in
        CI mode must never emit ``[Nm`` fragments."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        print(colors.color("⚕ Hermes Setup", colors.Colors.MAGENTA))

        captured = capsys.readouterr()
        # The output must be the bare text with no ANSI leakage.
        assert "⚕ Hermes Setup" in captured.out
        assert "[35m" not in captured.out
        assert "[0m" not in captured.out
        assert "\x1b" not in captured.out


# ──────────────────────────────────────────────────────────────────────
# 3. Windows VT processing — the underlying fix
# ──────────────────────────────────────────────────────────────────────


class TestWindowsVirtualTerminalProcessing:
    """``configure_windows_stdio()`` flips ``ENABLE_VIRTUAL_TERMINAL_
    PROCESSING`` on attached console handles.  These tests simulate the
    kernel32 surface so they run on any platform."""

    def test_stdout_and_stderr_both_get_vt_flag(self, monkeypatch, reset_stdio_module):
        """With both stdout and stderr backed by a console handle, both
        must have ``ENABLE_VIRTUAL_TERMINAL_PROCESSING`` (0x4) OR-ed in."""
        monkeypatch.setattr(hermes_stdio, "is_windows", lambda: True)

        fake = _FakeKernel32(stdout_mode=0x0001, stderr_mode=0x0001)
        monkeypatch.setattr(ctypes, "windll", _FakeWindll(fake), raising=False)

        # configure_windows_stdio() re-runs the whole bootstrap, which
        # calls _enable_windows_virtual_terminal_processing() internally.
        assert hermes_stdio.configure_windows_stdio() is True
        assert fake.set_modes == [
            (hermes_stdio._STD_OUTPUT_HANDLE, 0x0005),
            (hermes_stdio._STD_ERROR_HANDLE, 0x0005),
        ]

    def test_redirected_stderr_is_skipped(self, monkeypatch, reset_stdio_module):
        """When stderr is redirected (no console attached), only stdout
        gets the VT flag.  Touching stderr's mode would be a no-op at
        best and could disturb the redirected stream."""
        monkeypatch.setattr(hermes_stdio, "is_windows", lambda: True)

        fake = _FakeKernel32(
            stdout_mode=0x0001,
            stderr_mode=0x0001,
            # GetConsoleMode fails on stderr → it has been redirected
            stderr_get_console_mode=False,
        )
        monkeypatch.setattr(ctypes, "windll", _FakeWindll(fake), raising=False)

        hermes_stdio.configure_windows_stdio()
        assert fake.set_modes == [
            (hermes_stdio._STD_OUTPUT_HANDLE, 0x0005),
        ]

    def test_no_attached_console_does_not_call_set_mode(
        self, monkeypatch, reset_stdio_module
    ):
        """When both stdout and stderr lack attached consoles (CI /
        piped output), ``SetConsoleMode`` must not be called at all."""
        monkeypatch.setattr(hermes_stdio, "is_windows", lambda: True)

        fake = _FakeKernel32(
            stdout_get_console_mode=False,
            stderr_get_console_mode=False,
        )
        monkeypatch.setattr(ctypes, "windll", _FakeWindll(fake), raising=False)

        hermes_stdio.configure_windows_stdio()
        assert fake.set_modes == []

    def test_invalid_handle_value_is_skipped(self, monkeypatch, reset_stdio_module):
        """``GetStdHandle`` returning the sentinel ``INVALID_HANDLE_VALUE``
        means there's no console — skip the handle instead of calling
        ``GetConsoleMode`` on a junk pointer."""
        monkeypatch.setattr(hermes_stdio, "is_windows", lambda: True)

        fake = _FakeKernel32(
            stdout_handle=hermes_stdio._INVALID_HANDLE_VALUE,
            stderr_mode=0x0001,
        )
        monkeypatch.setattr(ctypes, "windll", _FakeWindll(fake), raising=False)

        hermes_stdio.configure_windows_stdio()
        # Only stderr should be touched — stdout is the invalid sentinel.
        assert fake.set_modes == [
            (hermes_stdio._STD_ERROR_HANDLE, 0x0005),
        ]

    def test_existing_vt_flag_is_preserved(self, monkeypatch, reset_stdio_module):
        """If VT processing is already enabled (``mode & 0x4 != 0``), the
        resulting mode should still be valid and idempotent — no
        double-bit-shifting, no spurious changes."""
        monkeypatch.setattr(hermes_stdio, "is_windows", lambda: True)

        # Console already has VT processing on (0x0005).
        fake = _FakeKernel32(stdout_mode=0x0005, stderr_mode=0x0005)
        monkeypatch.setattr(ctypes, "windll", _FakeWindll(fake), raising=False)

        hermes_stdio.configure_windows_stdio()
        # 0x0005 | 0x0004 = 0x0005 — bit is already set, no change.
        assert fake.set_modes == [
            (hermes_stdio._STD_OUTPUT_HANDLE, 0x0005),
            (hermes_stdio._STD_ERROR_HANDLE, 0x0005),
        ]

    def test_missing_ctypes_windll_returns_silently(
        self, monkeypatch, reset_stdio_module
    ):
        """If ``ctypes.windll`` is unavailable (rare embedded case),
        the helper must not raise — best-effort VT processing only."""
        monkeypatch.setattr(hermes_stdio, "is_windows", lambda: True)

        monkeypatch.setattr(ctypes, "windll", _BrokenWindll(), raising=False)

        # configure_windows_stdio() must complete without raising.
        hermes_stdio.configure_windows_stdio()


# ──────────────────────────────────────────────────────────────────────
# 4. Opt-out path (HERMES_DISABLE_WINDOWS_UTF8)
# ──────────────────────────────────────────────────────────────────────


class TestOptOut:
    """``HERMES_DISABLE_WINDOWS_UTF8=1`` short-circuits the entire
    bootstrap, including the new VT-processing call."""

    def test_disable_flag_skips_everything(
        self, monkeypatch, reset_stdio_module
    ):
        monkeypatch.setattr(hermes_stdio, "is_windows", lambda: True)
        monkeypatch.setenv("HERMES_DISABLE_WINDOWS_UTF8", "1")

        fake = _FakeKernel32()
        # If our flag is honored, the helper must never even consult
        # the kernel32 mock.
        monkeypatch.setattr(ctypes, "windll", _FakeWindll(fake), raising=False)

        # Even on a "Windows" run with the opt-out flag, configure must
        # return False and never call into the kernel32 mock.
        assert hermes_stdio.configure_windows_stdio() is False
        assert fake.set_modes == []