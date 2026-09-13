"""Tests for hermes_cli/config_update.py — config format migration helpers."""


def test_config_version_present():
    from hermes_cli.config import DEFAULT_CONFIG
    assert "_config_version" in DEFAULT_CONFIG


def test_config_version_is_integer():
    from hermes_cli.config import DEFAULT_CONFIG
    v = DEFAULT_CONFIG.get("_config_version")
    assert isinstance(v, int)
    assert v >= 1
