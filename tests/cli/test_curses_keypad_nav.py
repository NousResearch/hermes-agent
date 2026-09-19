"""kitty keypad keys must navigate the curses menus (#103875)."""

from __future__ import annotations

import pytest

from hermes_cli import curses_ui


@pytest.mark.parametrize("codepoint,expected", [
    (57419, curses_ui.NAV_UP),     # KP_Up
    (57420, curses_ui.NAV_DOWN),   # KP_Down
    (57417, curses_ui.NAV_BACK),   # KP_Left, matching curses.KEY_LEFT
])
def test_keypad_nav_mirrors_the_arrow_cluster(codepoint, expected):
    """kitty reports the keypad as PUA codepoints, which reached no action (#103875)."""
    assert curses_ui._enhanced_key_action(codepoint) == expected


@pytest.mark.parametrize("codepoint", [57418, 57421, 57422, 57423, 57424, 57414])
def test_keypad_keys_whose_twin_does_nothing_stay_inert(codepoint):
    """Right/PageUp/Home do nothing here, so the keypad must not gain what they lack."""
    assert curses_ui._enhanced_key_action(codepoint) == curses_ui.NAV_NONE
