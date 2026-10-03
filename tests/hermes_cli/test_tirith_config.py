"""hermes_cli.tirith_config: every reader of the tirith flags honours the env overrides."""

from unittest.mock import patch

import pytest

from hermes_cli import tirith_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TIRITH_ENABLED", raising=False)
    monkeypatch.delenv("TIRITH_FAIL_OPEN", raising=False)


def test_defaults_when_config_is_empty():
    assert tirith_config.tirith_enabled({}) is True
    assert tirith_config.tirith_fail_open({}) is True
    assert tirith_config.fail_open_when_scanner_unavailable({}) is True


def test_config_values_are_read():
    cfg = {"security": {"tirith_enabled": True, "tirith_fail_open": False}}
    assert tirith_config.tirith_fail_open(cfg) is False
    assert tirith_config.fail_open_when_scanner_unavailable(cfg) is False


def test_env_overrides_config(monkeypatch):
    cfg = {"security": {"tirith_enabled": True, "tirith_fail_open": True}}
    monkeypatch.setenv("TIRITH_FAIL_OPEN", "false")
    assert tirith_config.fail_open_when_scanner_unavailable(cfg) is False
    monkeypatch.setenv("TIRITH_ENABLED", "0")
    assert tirith_config.tirith_enabled(cfg) is False
    assert tirith_config.fail_open_when_scanner_unavailable(cfg) is True


def test_unreadable_config_falls_back_to_defaults():
    with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")):
        assert tirith_config.tirith_fail_open() is True


def test_approval_import_error_branch_sees_env_override(monkeypatch):
    """tools.approval's un-importable-scanner branch reads TIRITH_FAIL_OPEN, as tips.py says."""
    from tools.approval_context import _tirith_fail_open

    cfg = {"security": {"tirith_enabled": True, "tirith_fail_open": True}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert _tirith_fail_open() is True
        monkeypatch.setenv("TIRITH_FAIL_OPEN", "false")
        assert _tirith_fail_open() is False


def test_quoted_config_strings_parse_like_the_env_values():
    """A quoted ``"false"`` in config must not read truthy while ``TIRITH_*=false`` reads false."""
    cfg = {"security": {"tirith_enabled": "false", "tirith_fail_open": "false"}}
    assert tirith_config.tirith_enabled(cfg) is False
    assert tirith_config.tirith_fail_open(cfg) is False
    cfg = {"security": {"tirith_enabled": "true", "tirith_fail_open": "false"}}
    assert tirith_config.tirith_enabled(cfg) is True
    assert tirith_config.fail_open_when_scanner_unavailable(cfg) is False


@pytest.mark.parametrize("raw", ["on", "ON", " yes ", " true ", "1"])
def test_shared_truthy_spellings_keep_the_scanner_on(monkeypatch, raw):
    """``on`` and padded values read True from config and env alike (``utils.is_truthy_value``)."""
    monkeypatch.delenv("TIRITH_ENABLED", raising=False)
    monkeypatch.delenv("TIRITH_FAIL_OPEN", raising=False)
    cfg = {"security": {"tirith_enabled": raw, "tirith_fail_open": raw}}
    assert tirith_config.tirith_enabled(cfg) is True
    assert tirith_config.tirith_fail_open(cfg) is True
    monkeypatch.setenv("TIRITH_ENABLED", raw)
    assert tirith_config.tirith_enabled({"security": {"tirith_enabled": False}}) is True
