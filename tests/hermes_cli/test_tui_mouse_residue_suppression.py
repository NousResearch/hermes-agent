"""Tests for the TUI-hot-path mouse-residue suppression.

The Python launcher (`hermes --tui …`) has a ~100–300ms cold-start window
where stdin is still in cooked + echo mode. If a previous Hermes session
left DEC mouse-tracking asserted, any mouse motion during that window
echoes literal ``^[[<…M`` text into the user's scrollback.

`_suppress_mouse_residue_early()` writes the disable sequence to stdout
before the heavy imports so the terminal stops emitting events ASAP.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

# Importing the module triggers `_suppress_mouse_residue_early()` at module
# scope. Under the test runner argv (`pytest …`) it's a no-op, but we import
# at file scope so individual tests don't race the import side-effect with
# their `patch("os.write")` context.
from hermes_cli.main import _suppress_mouse_residue_early

EXPECTED = (
    b"\x1b[?1003l\x1b[?1002l\x1b[?1001l\x1b[?1000l\x1b[?9l"
    b"\x1b[?1006l\x1b[?1005l\x1b[?1015l\x1b[?1016l\x1b[?2029l"
)


class TestEarlyMouseDisable:
    def test_writes_disable_sequence_when_tui_flag_in_argv(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes", "--tui", "-c", "abc"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        with patch("os.isatty", return_value=True), patch("os.write") as mock_write:
            _suppress_mouse_residue_early()

        mock_write.assert_called_once_with(1, EXPECTED)

    def test_writes_disable_sequence_when_hermes_tui_env_set(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes"])
        monkeypatch.setenv("HERMES_TUI", "1")
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        with patch("os.isatty", return_value=True), patch("os.write") as mock_write:
            _suppress_mouse_residue_early()

        mock_write.assert_called_once_with(1, EXPECTED)

    def test_no_op_on_non_tui_invocation(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes", "--version"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        with patch("os.write") as mock_write:
            _suppress_mouse_residue_early()

        mock_write.assert_not_called()

    def test_respects_diagnostic_escape_hatch(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes", "--tui"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.setenv("HERMES_TUI_NO_EARLY_DISABLE", "1")

        with patch("os.write") as mock_write:
            _suppress_mouse_residue_early()

        mock_write.assert_not_called()

    def test_skips_when_stdout_is_not_a_tty(self, monkeypatch):
        # `hermes --tui … >log` or CI capture: pipe is fd 1, not a TTY. The
        # bytes can't reach a terminal and would just pollute the log.
        monkeypatch.setattr(sys, "argv", ["hermes", "--tui"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        with patch("os.isatty", return_value=False), patch("os.write") as mock_write:
            _suppress_mouse_residue_early()

        mock_write.assert_not_called()

    def test_oserror_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hermes", "--tui"])
        monkeypatch.delenv("HERMES_TUI", raising=False)
        monkeypatch.delenv("HERMES_TUI_NO_EARLY_DISABLE", raising=False)

        def boom(*_a, **_k):
            raise OSError("stdout closed")

        with patch("os.isatty", return_value=True), patch("os.write", side_effect=boom):
            # Must not propagate — startup hot path can never break.
            _suppress_mouse_residue_early()
