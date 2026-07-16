from types import SimpleNamespace
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry
from gateway.run import GatewayRunner, MultiplexConfigError
from plugins.platforms.webex.adapter import (
    WebexAdapter,
    _is_port_binding,
    _validate_config,
    register,
)


def test_register_exposes_complete_platform_hooks():
    ctx = SimpleNamespace(register_platform=lambda **kwargs: setattr(ctx, "entry", kwargs))
    register(ctx)
    assert ctx.entry["name"] == "webex"
    assert ctx.entry["cron_deliver_env_var"] == "WEBEX_HOME_CHANNEL"
    assert ctx.entry["standalone_sender_fn"] is not None
    assert ctx.entry["is_connected"] is _validate_config
    assert ctx.entry["allowed_users_env"] == "WEBEX_ALLOWED_USERS"
    assert ctx.entry["platform_hint"]


def test_port_binding_is_mode_aware(monkeypatch):
    monkeypatch.delenv("WEBEX_CONNECTION_MODE", raising=False)
    assert not _is_port_binding(PlatformConfig(extra={"connection_mode": "websocket"}))
    assert _is_port_binding(PlatformConfig(extra={"connection_mode": "webhook"}))


def test_connection_validation_is_mode_aware(monkeypatch):
    monkeypatch.delenv("WEBEX_BOT_TOKEN", raising=False)
    assert _validate_config(PlatformConfig(token="token", extra={}))
    assert not _validate_config(
        PlatformConfig(token="token", extra={"connection_mode": "webhook"})
    )
    assert _validate_config(
        PlatformConfig(
            token="token",
            extra={
                "connection_mode": "webhook",
                "public_url": "https://bot.example.com",
            },
        )
    )


@pytest.mark.asyncio
async def test_secondary_webhook_mode_is_rejected(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._profile_adapters = {}
    cfg = GatewayConfig(multiplex_profiles=True)
    cfg.platforms = {
        Platform("webex"): PlatformConfig(
            enabled=True, extra={"connection_mode": "webhook"}
        )
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)
    entry = PlatformEntry(
        name="webex",
        label="Webex",
        adapter_factory=WebexAdapter,
        check_fn=lambda: True,
        is_port_binding_fn=_is_port_binding,
    )
    monkeypatch.setattr("gateway.platform_registry.platform_registry.get", lambda _: entry)

    with pytest.raises(MultiplexConfigError, match="webex"):
        await runner._start_one_profile_adapters("secondary", "/tmp/profile", {})


@pytest.mark.asyncio
async def test_secondary_websocket_mode_can_start(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._profile_adapters = {}
    cfg = GatewayConfig(multiplex_profiles=True)
    cfg.platforms = {
        Platform("webex"): PlatformConfig(
            enabled=True, extra={"connection_mode": "websocket"}
        )
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)
    entry = PlatformEntry(
        name="webex",
        label="Webex",
        adapter_factory=WebexAdapter,
        check_fn=lambda: True,
        is_port_binding_fn=_is_port_binding,
    )
    monkeypatch.setattr("gateway.platform_registry.platform_registry.get", lambda _: entry)
    monkeypatch.setattr(runner, "_create_adapter", lambda *_: None)

    assert await runner._start_one_profile_adapters("secondary", "/tmp/profile", {}) == 0
