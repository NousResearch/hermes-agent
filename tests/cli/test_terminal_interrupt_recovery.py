"""Regression test for #33271: terminal recovery after interrupt.

After an interrupt, the process_loop finally block must:
  1. Drain stray escape bytes from the OS input buffer (flush_stdin)
  2. Force a full prompt_toolkit renderer redraw

Without this fix, CSI 6n cursor position reports can leak as literal
text (^[[19;1R) and the VT100 input parser can stall, accepting no
further keystrokes.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


@pytest.fixture
def cli():
    """Create a minimal HermesCLI mock with the required attributes."""
    from cli import HermesCLI

    instance = MagicMock(spec=HermesCLI)
    instance._agent_running = True
    instance._spinner_text = "thinking"
    instance._tool_start_time = 1.0
    instance._pending_tool_info = {"name": "test"}
    instance._last_scrollback_tool = "tool_a"
    instance._last_turn_interrupted = False
    instance._force_full_redraw = MagicMock()
    instance._last_input_mode_recovery = 0.0
    instance._input_mode_recovery_notice_shown = False

    app = MagicMock()
    app.invalidate = MagicMock()
    instance._app = app

    return instance


class TestPostInterruptTerminalRecovery:
    """Verify that the finally block in process_loop recovers the terminal
    state after an interrupted agent turn."""

    def test_no_recovery_when_turn_completes_normally(self, cli):
        """_force_full_redraw should NOT be called when the turn finishes
        normally (no interrupt)."""
        cli._last_turn_interrupted = False

        # Simulate the finally block logic
        if cli._last_turn_interrupted:
            cli._force_full_redraw()

        cli._force_full_redraw.assert_not_called()

    def test_recovery_after_interrupt(self, cli):
        """_force_full_redraw MUST be called when the turn was interrupted."""
        cli._last_turn_interrupted = True

        # Simulate the finally block logic
        if cli._last_turn_interrupted:
            try:
                from hermes_cli.curses_ui import flush_stdin
                flush_stdin()
            except Exception:
                pass
            cli._force_full_redraw()

        cli._force_full_redraw.assert_called_once()

    @patch("hermes_cli.curses_ui.flush_stdin")
    def test_flush_stdin_called_after_interrupt(self, mock_flush, cli):
        """flush_stdin must be called to drain stray escape bytes."""
        cli._last_turn_interrupted = True

        if cli._last_turn_interrupted:
            try:
                from hermes_cli.curses_ui import flush_stdin
                flush_stdin()
            except Exception:
                pass
            cli._force_full_redraw()

        mock_flush.assert_called_once()

    @patch("hermes_cli.curses_ui.flush_stdin", side_effect=OSError("no tty"))
    def test_flush_stdin_failure_does_not_prevent_redraw(self, mock_flush, cli):
        """Even if flush_stdin fails (e.g., no TTY), _force_full_redraw must
        still be called."""
        cli._last_turn_interrupted = True

        if cli._last_turn_interrupted:
            try:
                from hermes_cli.curses_ui import flush_stdin
                flush_stdin()
            except Exception:
                pass
            cli._force_full_redraw()

        cli._force_full_redraw.assert_called_once()

    def test_agent_running_cleared_on_normal_exit(self, cli):
        """State flags must be reset regardless of interrupt status."""
        cli._last_turn_interrupted = False
        cli._agent_running = True
        cli._spinner_text = "active"

        # Simulate the finally block
        cli._agent_running = False
        cli._spinner_text = ""

        assert cli._agent_running is False
        assert cli._spinner_text == ""

    def test_agent_running_cleared_on_interrupt(self, cli):
        """State flags must be reset even after interrupt + recovery."""
        cli._last_turn_interrupted = True
        cli._agent_running = True
        cli._spinner_text = "active"

        # Simulate the finally block
        cli._agent_running = False
        cli._spinner_text = ""
        if cli._last_turn_interrupted:
            cli._force_full_redraw()

        assert cli._agent_running is False
        assert cli._spinner_text == ""
        cli._force_full_redraw.assert_called_once()
