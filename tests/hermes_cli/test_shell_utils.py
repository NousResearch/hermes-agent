"""Tests for hermes_cli/shell_utils.py — shell utility helpers."""


def test_is_windows_returns_bool():
    from hermes_cli.shell_utils import _is_windows
    assert isinstance(_is_windows(), bool)
