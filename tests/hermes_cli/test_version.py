"""Tests for hermes_cli/version.py — version info export."""


def test_version_string_non_empty():
    from hermes_cli.version import VERSION
    assert isinstance(VERSION, str)
    assert len(VERSION) > 0
