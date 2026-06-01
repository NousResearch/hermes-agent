"""Tests for the second LINE admin channel registration."""

from __future__ import annotations

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_line = load_plugin_adapter("line")

LineAdapter = _line.LineAdapter
check_requirements = _line.check_requirements
check_requirements_admin = getattr(_line, "check_requirements_admin", lambda: "missing")


def test_customer_defaults_remain_unchanged_without_admin_env(monkeypatch):
    from gateway.config import Platform, PlatformConfig

    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "customer-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "customer-secret")
    for name in (
        "LINE_PORT",
        "LINE_ADMIN_CHANNEL_ACCESS_TOKEN",
        "LINE_ADMIN_CHANNEL_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    adapter = LineAdapter(PlatformConfig(enabled=True))

    assert adapter.channel_access_token == "customer-token"
    assert adapter.channel_secret == "customer-secret"
    assert adapter.platform == Platform("line")
    assert adapter.webhook_port == 8646
    assert adapter.webhook_path == "/line/webhook"
    assert check_requirements() is True
    assert check_requirements_admin() is False
