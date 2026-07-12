from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, platform_binds_port
from gateway.platform_registry import PlatformEntry
from gateway.run import GatewayRunner, MultiplexConfigError
from plugins.platforms.webex.adapter import (
    WebexAdapter,
    _is_port_binding,
    _validate_config,
    register,
)


def test_register_exposes_complete_platform_hooks():
    ctx = SimpleNamespace(
        register_platform=lambda **kwargs: setattr(ctx, "entry", kwargs)
    )
    register(ctx)
    assert ctx.entry["name"] == "webex"
    assert ctx.entry["cron_deliver_env_var"] == "WEBEX_HOME_CHANNEL"
    assert ctx.entry["standalone_sender_fn"] is not None
    assert ctx.entry["is_connected"] is _validate_config
    assert ctx.entry["allowed_users_env"] == "WEBEX_ALLOWED_USERS"
    assert ctx.entry["platform_hint"]
    assert ctx.entry["pii_safe"] is True


def test_port_binding_is_mode_aware(monkeypatch):
    monkeypatch.delenv("WEBEX_CONNECTION_MODE", raising=False)
    assert not _is_port_binding(PlatformConfig(extra={"connection_mode": "websocket"}))
    assert _is_port_binding(PlatformConfig(extra={"connection_mode": "webhook"}))


def test_port_binding_predicate_failure_fails_closed(monkeypatch):
    entry = PlatformEntry(
        name="webex",
        label="Webex",
        adapter_factory=WebexAdapter,
        check_fn=lambda: True,
        is_port_binding_fn=lambda _config: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "gateway.platform_registry.platform_registry.get", lambda _name: entry
    )

    assert platform_binds_port(
        "webex",
        {"connection_mode": "webhook"},
        platform_config=PlatformConfig(extra={"connection_mode": "webhook"}),
    )


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
                "secret": "signing-secret",
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
    scope_active = False

    @contextmanager
    def _profile_scope(*_args, **_kwargs):
        nonlocal scope_active
        scope_active = True
        try:
            yield
        finally:
            scope_active = False

    def _get_entry(_name):
        assert scope_active, (
            "plugin metadata must resolve in the secondary profile scope"
        )
        return entry

    monkeypatch.setattr("gateway.run._profile_runtime_scope", _profile_scope)
    monkeypatch.setattr("gateway.platform_registry.platform_registry.get", _get_entry)

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
    monkeypatch.setattr(
        "gateway.platform_registry.platform_registry.get", lambda _: entry
    )
    monkeypatch.setattr(runner, "_create_adapter", lambda *_: None)

    assert (
        await runner._start_one_profile_adapters("secondary", "/tmp/profile", {}) == 0
    )
