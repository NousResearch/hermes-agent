"""Tests for the TUI-hot-path mouse-residue suppression.

The Python launcher (`rabbit --tui …`) has a ~100–300ms cold-start window
where stdin is still in cooked + echo mode. If a previous Rabbit session
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
from rabbit_cli.main import _suppress_mouse_residue_early

class TestEarlyMouseDisable:


    def test_respects_diagnostic_escape_hatch(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["rabbit", "--tui"])
        monkeypatch.delenv("RABBIT_TUI", raising=False)
        monkeypatch.setenv("RABBIT_TUI_NO_EARLY_DISABLE", "1")

        with patch("os.write") as mock_write:
            _suppress_mouse_residue_early()

        mock_write.assert_not_called()


    def test_oserror_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["rabbit", "--tui"])
        monkeypatch.delenv("RABBIT_TUI", raising=False)
        monkeypatch.delenv("RABBIT_TUI_NO_EARLY_DISABLE", raising=False)

        def boom(*_a, **_k):
            raise OSError("stdout closed")

        with patch("os.isatty", return_value=True), patch("os.write", side_effect=boom):
            # Must not propagate — startup hot path can never break.
            _suppress_mouse_residue_early()
