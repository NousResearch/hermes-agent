"""Tests for hermes_cli.colors — ANSI color utilities."""

from __future__ import annotations

import pytest

from hermes_cli.colors import Colors, color, should_use_color


# ============================================================================
# should_use_color
# ============================================================================
class TestShouldUseColor:
    def test_no_color_env_set_returns_false(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        # Also ensure TERM isn't dumb and stdout is TTY-like
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        assert should_use_color() is False

    def test_no_color_env_empty_string_returns_false(self, monkeypatch):
        """Any value (including empty) for NO_COLOR disables color."""
        monkeypatch.setenv("NO_COLOR", "")
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        assert should_use_color() is False

    def test_term_dumb_returns_false(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        assert should_use_color() is False

    def test_not_tty_returns_false(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        assert should_use_color() is False

    def test_all_conditions_met_returns_true(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        assert should_use_color() is True

    def test_no_color_takes_priority_over_term(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        # NO_COLOR is checked first → False regardless of TERM/tty
        assert should_use_color() is False

    def test_term_checked_before_tty(self, monkeypatch):
        """TERM=dumb short-circuits before TTY check."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        # Leave stdout.isatty as-is — shouldn't matter
        assert should_use_color() is False


# ============================================================================
# color
# ============================================================================
class TestColor:
    def test_color_enabled_wraps_text(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.colors.should_use_color", lambda: True)
        result = color("hello", Colors.RED, Colors.BOLD)
        assert result == "\033[31m\033[1mhello\033[0m"

    def test_color_disabled_returns_plain_text(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.colors.should_use_color", lambda: False)
        result = color("hello", Colors.RED)
        assert result == "hello"

    def test_single_code(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.colors.should_use_color", lambda: True)
        result = color("test", Colors.GREEN)
        assert result == "\033[32mtest\033[0m"

    def test_no_codes(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.colors.should_use_color", lambda: True)
        result = color("plain")
        assert result == "plain\033[0m"

    def test_multiple_codes(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.colors.should_use_color", lambda: True)
        result = color("warn", Colors.YELLOW, Colors.BOLD, Colors.DIM)
        assert result == "\033[33m\033[1m\033[2mwarn\033[0m"


# ============================================================================
# Colors class
# ============================================================================
class TestColorsClass:
    def test_reset_is_correct(self):
        assert Colors.RESET == "\033[0m"

    def test_all_colors_are_distinct(self):
        values = [Colors.RED, Colors.GREEN, Colors.YELLOW,
                  Colors.BLUE, Colors.MAGENTA, Colors.CYAN]
        assert len(values) == len(set(values))

    def test_bold_and_dim_are_distinct_from_colors(self):
        assert Colors.BOLD not in [Colors.RED, Colors.GREEN, Colors.BLUE]
        assert Colors.DIM not in [Colors.RED, Colors.GREEN, Colors.BLUE]
