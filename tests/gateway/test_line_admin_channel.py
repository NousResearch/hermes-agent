"""Tests for the second LINE admin channel registration."""

from __future__ import annotations

import base64
import hashlib
import hmac

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_line = load_plugin_adapter("line")

LineAdapter = _line.LineAdapter
check_requirements = _line.check_requirements
check_requirements_admin = getattr(_line, "check_requirements_admin", lambda: "missing")
register = _line.register
verify_line_signature = _line.verify_line_signature


def _line_signature(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


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


def test_admin_constructor_reads_admin_env_and_defaults(monkeypatch):
    from gateway.config import Platform, PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "customer-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "customer-secret")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    monkeypatch.delenv("LINE_ADMIN_CHANNEL_PORT", raising=False)
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        adapter = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
        )
    finally:
        platform_registry.unregister("line_admin")

    assert adapter.channel_access_token == "admin-token"
    assert adapter.channel_secret == "admin-secret"
    assert adapter.platform == Platform("line_admin")
    assert adapter.webhook_port == 8647
    assert adapter.webhook_path == "/line-admin/webhook"


def test_register_adds_customer_and_admin_platforms():
    class _FakeCtx:
        def __init__(self):
            self.calls = []

        def register_platform(self, **kwargs):
            self.calls.append(kwargs)

    ctx = _FakeCtx()

    register(ctx)

    by_name = {call["name"]: call for call in ctx.calls}
    assert set(by_name) == {"line", "line_admin"}
    assert by_name["line"]["check_fn"] is check_requirements
    assert by_name["line_admin"]["check_fn"] is check_requirements_admin


def test_admin_requirements_gate_needs_token_and_secret(monkeypatch):
    monkeypatch.delenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINE_ADMIN_CHANNEL_SECRET", raising=False)
    assert check_requirements_admin() is False

    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    assert check_requirements_admin() is False

    monkeypatch.delenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    assert check_requirements_admin() is False

    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    assert check_requirements_admin() is True


def test_hmac_signatures_are_isolated_by_channel_secret(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    body = b'{"events": []}'
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "customer-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "customer-secret")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        customer = LineAdapter(PlatformConfig(enabled=True))
        admin = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
            default_media_prefix="/line-admin/media",
        )
    finally:
        platform_registry.unregister("line_admin")

    customer_sig = _line_signature(body, "customer-secret")
    admin_sig = _line_signature(body, "admin-secret")

    assert verify_line_signature(body, customer_sig, customer.channel_secret)
    assert not verify_line_signature(body, admin_sig, customer.channel_secret)
    assert verify_line_signature(body, admin_sig, admin.channel_secret)
    assert not verify_line_signature(body, customer_sig, admin.channel_secret)
