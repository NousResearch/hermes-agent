"""Tests for hermes_cli/_install_repair.py — install repair helpers."""


def test_has_no_pip_issue_returns_bool():
    from hermes_cli._install_repair import _has_no_pip_issue
    result = _has_no_pip_issue()
    assert isinstance(result, bool)
