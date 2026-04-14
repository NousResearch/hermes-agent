"""Tests for hermes_cli.colors theme handling (issue #8526).

These tests cover:
  * ``NO_COLOR`` still wins over any theme configuration.
  * ``HERMES_THEME=light|dark`` takes priority over ``COLORFGBG``.
  * ``COLORFGBG``-based auto-detection is only consulted when
    ``HERMES_THEME`` is unset or set to ``auto``.
  * In light mode, yellow/dim text is remapped to codes that are
    readable on a white background; every other ANSI code is left
    untouched so semantic meaning (error=red, ok=green, info=cyan)
    is preserved.
"""

import sys

import pytest

from hermes_cli.colors import (
    Colors,
    _resolve_theme,
    color,
    should_use_color,
)


class _FakeTTY:
    """Minimal stdout stub whose ``isatty`` returns True."""

    def isatty(self) -> bool:
        return True


@pytest.fixture
def tty(monkeypatch):
    """Pretend stdout is a TTY so ``color()`` does not short-circuit."""
    monkeypatch.setattr(sys, "stdout", _FakeTTY())


@pytest.fixture
def clean_env(monkeypatch):
    """Clear every env var that influences theme resolution."""
    for var in ("NO_COLOR", "TERM", "HERMES_THEME", "COLORFGBG"):
        monkeypatch.delenv(var, raising=False)


class TestShouldUseColor:
    def test_no_color_env_disables_color(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("NO_COLOR", "1")
        assert should_use_color() is False

    def test_no_color_empty_string_still_disables_color(self, monkeypatch, tty, clean_env):
        # Per https://no-color.org/ the presence of NO_COLOR — even empty — disables color.
        monkeypatch.setenv("NO_COLOR", "")
        assert should_use_color() is False

    def test_no_color_wins_over_hermes_theme(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("HERMES_THEME", "light")
        # Even with light theme explicitly set, NO_COLOR still disables color.
        assert color("x", Colors.YELLOW) == "x"

    def test_term_dumb_disables_color(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("TERM", "dumb")
        assert should_use_color() is False


class TestResolveTheme:
    def test_hermes_theme_light(self, monkeypatch, clean_env):
        monkeypatch.setenv("HERMES_THEME", "light")
        assert _resolve_theme() == "light"

    def test_hermes_theme_dark(self, monkeypatch, clean_env):
        monkeypatch.setenv("HERMES_THEME", "dark")
        assert _resolve_theme() == "dark"

    def test_hermes_theme_is_case_insensitive(self, monkeypatch, clean_env):
        monkeypatch.setenv("HERMES_THEME", "LIGHT")
        assert _resolve_theme() == "light"

    def test_hermes_theme_with_whitespace(self, monkeypatch, clean_env):
        monkeypatch.setenv("HERMES_THEME", "  dark  ")
        assert _resolve_theme() == "dark"

    def test_auto_falls_back_to_colorfgbg_light(self, monkeypatch, clean_env):
        monkeypatch.setenv("HERMES_THEME", "auto")
        monkeypatch.setenv("COLORFGBG", "0;15")
        assert _resolve_theme() == "light"

    def test_unset_falls_back_to_colorfgbg_light(self, monkeypatch, clean_env):
        monkeypatch.setenv("COLORFGBG", "0;15")
        assert _resolve_theme() == "light"

    def test_colorfgbg_dark_bg(self, monkeypatch, clean_env):
        monkeypatch.setenv("COLORFGBG", "15;0")
        assert _resolve_theme() == "dark"

    def test_colorfgbg_light_grey_bg(self, monkeypatch, clean_env):
        monkeypatch.setenv("COLORFGBG", "0;7")
        assert _resolve_theme() == "light"

    def test_colorfgbg_default_segment_returns_none(self, monkeypatch, clean_env):
        # Some terminals report "default" instead of a numeric background.
        # We don't guess — fall through so callers keep their current defaults.
        monkeypatch.setenv("COLORFGBG", "0;default")
        assert _resolve_theme() is None

    def test_colorfgbg_three_segments_uses_last(self, monkeypatch, clean_env):
        # rxvt reports ``fg;?;bg`` with a middle marker.
        monkeypatch.setenv("COLORFGBG", "15;default;0")
        assert _resolve_theme() == "dark"

    def test_hermes_theme_overrides_colorfgbg(self, monkeypatch, clean_env):
        # User's explicit choice must win over COLORFGBG auto-detection.
        monkeypatch.setenv("HERMES_THEME", "dark")
        monkeypatch.setenv("COLORFGBG", "0;15")
        assert _resolve_theme() == "dark"

    def test_unrecognized_hermes_theme_value_returns_none(self, monkeypatch, clean_env):
        monkeypatch.setenv("HERMES_THEME", "sepia")
        assert _resolve_theme() is None

    def test_no_env_returns_none(self, clean_env):
        assert _resolve_theme() is None


class TestColorRemapLightTheme:
    def test_remaps_yellow(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("HERMES_THEME", "light")
        out = color("warn", Colors.YELLOW)
        assert Colors.YELLOW not in out
        assert Colors.MAGENTA in out
        assert "warn" in out
        assert out.endswith(Colors.RESET)

    def test_remaps_dim_to_default_foreground(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("HERMES_THEME", "light")
        out = color("hint", Colors.DIM)
        # DIM remap is empty, so the output is just the text + RESET.
        assert Colors.DIM not in out
        assert out == "hint" + Colors.RESET

    def test_preserves_unmapped_colors(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("HERMES_THEME", "light")
        for code in (
            Colors.RED,
            Colors.GREEN,
            Colors.CYAN,
            Colors.BLUE,
            Colors.MAGENTA,
            Colors.BOLD,
        ):
            out = color("x", code)
            assert code in out, f"expected {code!r} preserved in light mode"

    def test_remap_applies_to_combined_codes(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("HERMES_THEME", "light")
        out = color("warn", Colors.BOLD, Colors.YELLOW)
        # BOLD stays, YELLOW is swapped to MAGENTA.
        assert Colors.BOLD in out
        assert Colors.MAGENTA in out
        assert Colors.YELLOW not in out


class TestColorRemapDarkAndDefault:
    def test_dark_theme_preserves_yellow(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("HERMES_THEME", "dark")
        out = color("warn", Colors.YELLOW)
        assert Colors.YELLOW in out
        assert Colors.MAGENTA not in out

    def test_dark_theme_preserves_dim(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("HERMES_THEME", "dark")
        out = color("hint", Colors.DIM)
        assert Colors.DIM in out

    def test_default_no_env_preserves_yellow(self, monkeypatch, tty, clean_env):
        # No HERMES_THEME, no COLORFGBG → behave exactly as before.
        out = color("warn", Colors.YELLOW)
        assert Colors.YELLOW in out


class TestColorRemapAutoDetection:
    def test_auto_with_light_colorfgbg_remaps_yellow(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("HERMES_THEME", "auto")
        monkeypatch.setenv("COLORFGBG", "0;15")
        assert Colors.YELLOW not in color("warn", Colors.YELLOW)

    def test_unset_with_light_colorfgbg_remaps_yellow(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("COLORFGBG", "0;15")
        assert Colors.YELLOW not in color("warn", Colors.YELLOW)

    def test_unset_with_dark_colorfgbg_preserves_yellow(self, monkeypatch, tty, clean_env):
        monkeypatch.setenv("COLORFGBG", "15;0")
        assert Colors.YELLOW in color("warn", Colors.YELLOW)
