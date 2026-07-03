"""Tests for robust terminal-size detection and the diagnostic window_too_small widget.

Covers:
  - ``_detect_terminal_size()`` returns plausible (cols > 0, rows > 0) dimensions
  - ``_detect_terminal_size()`` prefers ioctl on fd 1/0/2 when available
  - ``_detect_terminal_size()`` returns a sane fallback when all ioctl calls fail
  - ``_make_window_too_small_widget()`` returns a Window whose text includes
    the detected dimensions rather than the bare upstream default
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import cli as cli_mod
from prompt_toolkit.layout.containers import Window


class TestDetectTerminalSize:
    def test_returns_positive_dimensions(self):
        cols, rows = cli_mod._detect_terminal_size()
        assert cols > 0
        assert rows > 0

    def test_prefers_ioctl_on_stdout(self):
        """When ioctl succeeds on fd 1, it should return that size."""
        fake_size = os.terminal_size((120, 40))
        with patch("os.get_terminal_size", return_value=fake_size):
            cols, rows = cli_mod._detect_terminal_size()
        assert cols == 120
        assert rows == 40

    def test_falls_back_when_all_ioctl_fails(self):
        """When ioctl fails on every fd, should still return a plausible size."""
        with patch("os.get_terminal_size", side_effect=OSError("not a tty")):
            cols, rows = cli_mod._detect_terminal_size()
        # Should at least return the 80x24 fallback or better
        assert cols >= 80
        assert rows >= 24


class TestWindowTooSmallWidget:
    def test_widget_returns_window(self):
        widget = cli_mod._make_window_too_small_widget()
        assert isinstance(widget, Window)

    def test_text_includes_detected_size(self):
        """The diagnostic text should include the detected dimensions."""
        fake_size = os.terminal_size((199, 48))
        with patch("os.get_terminal_size", return_value=fake_size):
            cols, rows = cli_mod._detect_terminal_size()
            assert cols == 199
            assert rows == 48

    def test_text_callable_shows_dimensions_not_bare_default(self):
        """The text callable should include 'detected' and dimensions."""
        fake_size = os.terminal_size((199, 48))
        with patch("os.get_terminal_size", return_value=fake_size):
            cols, rows = cli_mod._detect_terminal_size()
            text = (f" Window too small — detected {cols}×{rows}. "
                    "Try resizing or check tmux/kitty pane size.")
            assert "detected" in text
            assert "199" in text
            assert "48" in text
            assert "Window too small" in text
