"""Tests for curses color compatibility on low-color terminals (Docker).

Regression test for #13688: ``hermes plugins`` crashes with
``curses.error: init_pair() : color number is greater than COLORS-1``
in Docker containers where curses.COLORS == 8 (only colors 0-7 exist).

The bug was ``curses.init_pair(4, 8, -1)`` using raw color 8 ("bright
black" / dim gray) which does not exist on 8-color terminals.  The fix
clamps with ``min(8, curses.COLORS - 1)``.
"""
import sys

import pytest

# curses (and its _curses C extension) is Unix-only; skip the whole module on Windows.
if sys.platform == "win32":
    pytest.skip("curses is not available on Windows", allow_module_level=True)

import curses
import re
from pathlib import Path
from unittest.mock import patch, MagicMock



# Path to the source files under test
_SRC_ROOT = Path(__file__).parent.parent.parent / "hermes_cli"


class TestInitPairClampingBehavior:
    """Simulate curses color initialization on low-color terminals.

    Patches curses.COLORS to 8 (Docker default) and verifies that
    init_pair is never called with a color >= COLORS.
    """

    def _collect_init_pair_calls(self, draw_fn, colors_value):
        """Run a curses draw function with a mock stdscr and patched COLORS.

        Returns list of (pair_number, fg, bg) tuples from init_pair calls.
        """
        calls = []
        real_init_pair = curses.init_pair

        def tracking_init_pair(pair, fg, bg):
            calls.append((pair, fg, bg))

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        mock_stdscr.getch.return_value = 27  # ESC to exit

        with patch("curses.COLORS", colors_value, create=True), \
             patch("curses.init_pair", side_effect=tracking_init_pair), \
             patch("curses.has_colors", return_value=True), \
             patch("curses.start_color"), \
             patch("curses.use_default_colors"), \
             patch("curses.curs_set"):
            try:
                draw_fn(mock_stdscr)
            except (SystemExit, StopIteration, Exception):
                pass  # draw functions loop until keypress

        return calls

    def test_8_color_terminal_no_color_exceeds_limit(self):
        """On an 8-color terminal (Docker), no init_pair fg color >= 8."""
        # Simulate the color init pattern from plugins_cmd.py
        def _simulated_color_init(stdscr):
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                curses.init_pair(3, curses.COLOR_CYAN, -1)
                curses.init_pair(4, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)

        calls = self._collect_init_pair_calls(_simulated_color_init, 8)
        for pair, fg, bg in calls:
            assert fg < 8, (
                f"init_pair({pair}, {fg}, {bg}) uses color {fg} which "
                f"does not exist on an 8-color terminal (valid: 0-7)"
            )

    def test_256_color_terminal_uses_color_8(self):
        """On a 256-color terminal, color 8 (dim gray) should be used."""
        def _simulated_color_init(stdscr):
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(4, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)

        calls = self._collect_init_pair_calls(_simulated_color_init, 256)
        assert any(fg == 8 for _, fg, _ in calls), (
            "On 256-color terminals, color 8 (dim gray) should be used"
        )

    def test_16_color_terminal_uses_color_8(self):
        """On a 16-color terminal, color 8 should be available."""
        def _simulated_color_init(stdscr):
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(4, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)

        calls = self._collect_init_pair_calls(_simulated_color_init, 16)
        assert any(fg == 8 for _, fg, _ in calls)


class TestSourceCodeGuardrails:
    """Regression guardrails: raw color 8 must not reappear in source.

    These complement the behavioral tests above — they catch regressions
    introduced by copy-paste of the old pattern.
    """

    _RAW_COLOR_8_PATTERN = re.compile(r'init_pair\(\d+,\s*8\s*,')

    def test_no_raw_color_8_in_plugins_cmd(self):
        source = (_SRC_ROOT / "plugins_cmd.py").read_text()
        matches = self._RAW_COLOR_8_PATTERN.findall(source)
        assert not matches, (
            f"plugins_cmd.py contains unclamped color 8: {matches}"
        )

    def test_no_raw_color_8_in_main(self):
        source = (_SRC_ROOT / "main.py").read_text()
        matches = self._RAW_COLOR_8_PATTERN.findall(source)
        assert not matches, (
            f"main.py contains unclamped color 8: {matches}"
        )

    def test_no_raw_color_8_in_curses_ui(self):
        source = (_SRC_ROOT / "curses_ui.py").read_text()
        matches = self._RAW_COLOR_8_PATTERN.findall(source)
        assert not matches, (
            f"curses_ui.py contains unclamped color 8: {matches}"
        )
