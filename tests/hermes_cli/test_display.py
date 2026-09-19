"""Tests for hermes_cli/display.py — display module import."""


def test_display_module_imports():
    from hermes_cli.display import COLORS, RESET
    assert COLORS is not None
    assert RESET is not None
