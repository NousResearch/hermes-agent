"""Regression tests for #33271: terminal recovery after interrupt.

When the user interrupts a running agent turn by typing a new message,
prompt_toolkit may have an in-flight ``CSI 6n`` cursor-position query whose
reply (``ESC[<row>;<col>R``) arrives on stdin after the input parser has torn
down. The reply then leaks as literal text (``^[[19;1R``) and the VT100 parser
can stall, accepting no further keystrokes — the terminal appears frozen.

The recovery path lives in ``HermesCLI._recover_terminal_after_interrupt()``,
which is invoked from ``process_loop``'s ``finally`` block only when
``self._last_turn_interrupted`` is set. It must:
  1. Drain stray escape bytes from the OS input buffer (``flush_stdin``).
  2. Force a clean prompt_toolkit renderer redraw (``_force_full_redraw``).

These tests exercise the real method (not a re-implementation of its logic).
"""

from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI


@pytest.fixture
def bare_cli():
    """A HermesCLI with no __init__ — we only exercise the recovery helper."""
    return object.__new__(HermesCLI)


class TestRecoverTerminalAfterInterrupt:
    """Directly exercise HermesCLI._recover_terminal_after_interrupt()."""


    def test_redraw_still_runs_when_flush_fails(self, bare_cli):
        """A flush_stdin failure (no TTY, non-POSIX) must not skip the redraw.

        The two recovery steps are independent — losing the stdin drain must
        never leave the renderer un-repainted.
        """
        bare_cli._force_full_redraw = MagicMock()
        with patch(
            "hermes_cli.curses_ui.flush_stdin", side_effect=OSError("no tty")
        ):
            bare_cli._recover_terminal_after_interrupt()  # must not raise

        bare_cli._force_full_redraw.assert_called_once()

    def test_flush_runs_before_redraw(self, bare_cli):
        """Order matters: drain stray bytes first so they don't arrive mid-redraw."""
        events = []
        bare_cli._force_full_redraw = MagicMock(
            side_effect=lambda: events.append("redraw")
        )
        with patch(
            "hermes_cli.curses_ui.flush_stdin",
            side_effect=lambda: events.append("flush"),
        ):
            bare_cli._recover_terminal_after_interrupt()

        assert events == ["flush", "redraw"]

    def test_flush_stdin_is_tty_gated(self):
        """The real flush_stdin is a no-op on non-TTY stdin (piped/redirected).

        Under pytest stdin is not a TTY, so this must return cleanly without
        touching termios.
        """
        from hermes_cli.curses_ui import flush_stdin

        flush_stdin()  # must not raise in a non-TTY test environment


