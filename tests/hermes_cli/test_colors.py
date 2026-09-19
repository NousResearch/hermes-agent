"""Behavioral tests for hermes_cli.colors — ANSI color gating + application.

`colors.py` is imported by ~20 modules but had no direct unit coverage. These
pin the *behavior* (when color is on/off and how it's applied), not values, so
they don't break on cosmetic changes.
"""

import io

import pytest

from hermes_cli.colors import Colors, color, should_use_color


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)


def test_should_use_color_no_color_var_disables(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("hermes_cli.colors.sys.stdout", io.StringIO())
    assert should_use_color() is False


def test_should_use_color_term_dumb_disables(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setattr("hermes_cli.colors.sys.stdout", io.StringIO())
    assert should_use_color() is False


def test_should_use_color_non_tty_disables(monkeypatch):
    # A StringIO is not a TTY, so with no NO_COLOR/TERM override color must be off.
    monkeypatch.setattr("hermes_cli.colors.sys.stdout", io.StringIO())
    assert should_use_color() is False


def test_should_use_color_no_color_takes_precedence_over_tty(monkeypatch):
    # A real TTY would otherwise enable color, but NO_COLOR must override it.
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("hermes_cli.colors.sys.stdout", _Tty())
    assert should_use_color() is False


def test_color_returns_text_unchanged_when_color_disabled(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("hermes_cli.colors.sys.stdout", io.StringIO())
    assert color("hello", Colors.RED) == "hello"


def test_color_applies_codes_when_color_enabled(monkeypatch):
    # Simulate a TTY so color is allowed.
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("hermes_cli.colors.sys.stdout", _Tty())
    assert color("hello", Colors.RED) == f"{Colors.RED}hello{Colors.RESET}"


def test_color_multiple_codes_concatenate(monkeypatch):
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("hermes_cli.colors.sys.stdout", _Tty())
    assert color("x", Colors.BOLD, Colors.GREEN) == f"{Colors.BOLD}{Colors.GREEN}x{Colors.RESET}"


def test_colors_constants_are_ansi_escapes():
    # These are the stable ANSI SGR sequences the app relies on; a change
    # here would break every colored surface, so pin them as a contract.
    assert Colors.RESET == "\x1b[0m"
    assert Colors.RED == "\x1b[31m"
    assert Colors.GREEN == "\x1b[32m"
    assert Colors.YELLOW == "\x1b[33m"
