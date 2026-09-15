"""Tests for hermes_cli/path_utils.py — path utility helpers."""


def test_expand_home():
    from hermes_cli.path_utils import _expand_home
    result = _expand_home("~/.hermes")
    assert result is not None
    assert isinstance(result, str)
