"""MCP enabled flags must accept shared truthy aliases like 'on'."""

from __future__ import annotations

from hermes_cli.mcp_catalog import is_enabled, parse_mcp_enabled_flag


def test_parse_mcp_enabled_flag_accepts_on_alias():
    assert parse_mcp_enabled_flag("on") is True
    assert parse_mcp_enabled_flag("ON") is True
    assert parse_mcp_enabled_flag("1") is True
    assert parse_mcp_enabled_flag("yes") is True
    assert parse_mcp_enabled_flag(" true ") is True


def test_parse_mcp_enabled_flag_rejects_falsy():
    assert parse_mcp_enabled_flag("off") is False
    assert parse_mcp_enabled_flag("false") is False
    assert parse_mcp_enabled_flag("0") is False
    assert parse_mcp_enabled_flag("no") is False


def test_parse_mcp_enabled_flag_bool_and_default():
    assert parse_mcp_enabled_flag(True) is True
    assert parse_mcp_enabled_flag(False) is False
    assert parse_mcp_enabled_flag(None, default=True) is True
    assert parse_mcp_enabled_flag(None, default=False) is False


def test_is_enabled_accepts_on_alias(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.mcp_catalog.installed_servers",
        lambda: {"demo": {"enabled": "on"}},
    )
    assert is_enabled("demo") is True


def test_is_enabled_rejects_off_alias(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.mcp_catalog.installed_servers",
        lambda: {"demo": {"enabled": "off"}},
    )
    assert is_enabled("demo") is False
