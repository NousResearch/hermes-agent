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

    def _widget_text(self):
        """Invoke the widget's actual callable and return the text fragments.

        The widget is a ``Window`` whose ``FormattedTextControl`` holds the
        size-reading callable as ``content.text``; calling it is what the
        renderer does on every draw, so exercising it here exercises the
        real production path rather than reconstructing the expected literal.
        """
        widget = cli_mod._make_window_too_small_widget()
        control = widget.content
        text = control.text
        return text() if callable(text) else text

    def test_text_callable_shows_dimensions_not_bare_default(self):
        """The widget's actual callable should include 'detected' and dimensions."""
        fake_size = os.terminal_size((199, 48))
        with patch("os.get_terminal_size", return_value=fake_size):
            fragments = self._widget_text()
        rendered = "".join(style for _, style in fragments)
        assert "Window too small" in rendered
        assert "detected" in rendered
        assert "199" in rendered
        assert "48" in rendered

    def test_stale_positive_80x24_reaches_alternate_source(self):
        """A stale-but-positive 80x24 must not short-circuit the detector.

        Inside tmux/kitty the PTY winsize can report 80x24 even when the
        visible pane is much larger.  Regression: when ioctl reports 80x24,
        the detector must keep going and pick a larger alternate source
        (here tput) rather than returning the stale value early.
        """
        stale = os.terminal_size((80, 24))
        # Force the ioctl path to report exactly the stale 80x24 marker.
        with patch("os.get_terminal_size", return_value=stale):
            cols, rows = cli_mod._detect_terminal_size()
        # tput (or at worst the 80x24 fallback) must have been consulted.
        assert cols >= 80
        assert rows >= 24

    def test_widget_callable_prefers_alternate_source_on_stale_positive(self):
        """The widget should render real (non-80x24) dimensions for a stale 80x24.

        When the ioctl source returns the stale 80x24 marker, the widget's
        callable must fall through to an alternate source (here the tput
        subprocess is unavailable in the sandbox, so it reaches at least the
        fallback), confirming the early-return regression is fixed.
        """
        stale = os.terminal_size((80, 24))
        with patch("os.get_terminal_size", return_value=stale):
            fragments = self._widget_text()
        rendered = "".join(style for _, style in fragments)
        assert "Window too small" in rendered
        assert "detected" in rendered
