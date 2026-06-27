"""Regression tests for #53301: TUI COLORTERM injection for truecolor terminals.

Tests the _apply_tui_truecolor_env() helper extracted from _launch_tui().
"""

from __future__ import annotations

import hermes_cli.main


def test_injects_colorterm_when_term_is_256color():
    """When TERM ends in 256color and COLORTERM is unset, inject truecolor."""
    env: dict[str, str] = {"TERM": "xterm-256color"}
    hermes_cli.main._apply_tui_truecolor_env(env)
    assert env.get("COLORTERM") == "truecolor"


def test_injects_for_screen_256color():
    """'screen-256color' is also a >=256-color TERM."""
    env: dict[str, str] = {"TERM": "screen-256color"}
    hermes_cli.main._apply_tui_truecolor_env(env)
    assert env.get("COLORTERM") == "truecolor"


def test_injects_for_tmux_256color():
    """'tmux-256color' is a >=256-color TERM."""
    env: dict[str, str] = {"TERM": "tmux-256color"}
    hermes_cli.main._apply_tui_truecolor_env(env)
    assert env.get("COLORTERM") == "truecolor"


def test_noop_when_colorterm_already_truecolor():
    """If COLORTERM is already 'truecolor', do nothing."""
    env: dict[str, str] = {"TERM": "xterm-256color", "COLORTERM": "truecolor"}
    hermes_cli.main._apply_tui_truecolor_env(env)
    assert env.get("COLORTERM") == "truecolor"
    assert "FORCE_COLOR" not in env


def test_noop_when_colorterm_is_24bit():
    """'24bit' is also a valid truecolor COLORTERM value."""
    env: dict[str, str] = {"TERM": "xterm-256color", "COLORTERM": "24bit"}
    hermes_cli.main._apply_tui_truecolor_env(env)
    assert env.get("COLORTERM") == "24bit"


def test_noop_when_term_is_not_256color():
    """'xterm' without -256color does NOT trigger injection."""
    env: dict[str, str] = {"TERM": "xterm"}
    hermes_cli.main._apply_tui_truecolor_env(env)
    assert "COLORTERM" not in env


def test_noop_when_term_is_empty():
    """Missing TERM: no injection."""
    env: dict[str, str] = {}
    hermes_cli.main._apply_tui_truecolor_env(env)
    assert "COLORTERM" not in env


def test_colorterm_case_insensitive():
    """COLORTERM=TrueColor (mixed case) is recognized and not overwritten."""
    env: dict[str, str] = {"TERM": "xterm-256color", "COLORTERM": "TrueColor"}
    hermes_cli.main._apply_tui_truecolor_env(env)
    assert env.get("COLORTERM") == "TrueColor"  # preserved as-is
