"""Telegram runtime receipt contracts."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from hermes_cli import env_loader
from plugins.platforms.telegram import adapter as telegram_module
from plugins.platforms.telegram.adapter import TelegramAdapter


def test_mark_connected_publishes_sanitized_degraded_polling_receipt(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))
    adapter._bot = SimpleNamespace(id=123456, username="fleet_bot")
    status_writer = MagicMock()
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", status_writer)
    monkeypatch.setattr(
        env_loader,
        "get_effective_credential_source",
        lambda *args, **kwargs: "onepassword",
    )

    adapter._mark_telegram_connected(transport_ready=False)

    status_writer.assert_called_once()
    runtime_call = status_writer.call_args
    assert runtime_call.args == ("connected",)
    receipt = runtime_call.kwargs["platform_runtime"]
    assert receipt == {
        "credential_source": "onepassword",
        "authenticated": True,
        "bot_id": "123456",
        "bot_username": "fleet_bot",
        "transport_mode": "polling",
        "transport_ready": False,
        "verified_at": receipt["verified_at"],
    }
    assert datetime.fromisoformat(receipt["verified_at"]).tzinfo is not None
    assert "secret-token" not in repr(receipt)


def test_mark_disconnected_clears_runtime_receipt(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))
    status_writer = MagicMock()
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", status_writer)

    adapter._mark_disconnected()

    status_writer.assert_called_once_with(
        "disconnected",
        platform_state="disconnected",
        error_code=None,
        error_message=None,
        platform_runtime=None,
    )


def test_connect_records_polling_not_ready_when_bootstrap_degrades(monkeypatch):
    fake_app = MagicMock()
    fake_app.bot = SimpleNamespace(id=123456, username="fleet_bot")
    fake_app.initialize = AsyncMock(return_value=None)
    fake_app.start = AsyncMock(return_value=None)
    fake_app.add_handler = MagicMock()

    chainable = MagicMock()
    chainable.token.return_value = chainable
    chainable.request.return_value = chainable
    chainable.get_updates_request.return_value = chainable
    chainable.build.return_value = fake_app
    builder_root = MagicMock()
    builder_root.builder.return_value = chainable

    monkeypatch.setattr(telegram_module, "Application", builder_root)
    monkeypatch.setattr(telegram_module, "HTTPXRequest", MagicMock)
    monkeypatch.setattr(
        telegram_module,
        "discover_fallback_ips",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(telegram_module, "resolve_proxy_url", lambda *a, **k: None)
    monkeypatch.setattr(
        env_loader,
        "get_effective_credential_source",
        lambda *args, **kwargs: "profile_env",
    )

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))
    status_writer = MagicMock()
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", status_writer)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])
    monkeypatch.setattr(adapter, "_delete_webhook_best_effort", AsyncMock())
    monkeypatch.setattr(
        adapter,
        "_start_polling_resilient",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(adapter, "_polling_heartbeat_loop", AsyncMock())
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", MagicMock())

    assert asyncio.run(adapter.connect()) is True

    runtime_calls = [
        call
        for call in status_writer.call_args_list
        if "platform_runtime" in call.kwargs
    ]
    assert runtime_calls[-1].kwargs["platform_runtime"]["transport_ready"] is False


def test_polling_network_recovery_updates_runtime_readiness(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))
    adapter._bot = SimpleNamespace(id=123456, username="fleet_bot")
    adapter._app = SimpleNamespace(
        bot=adapter._bot,
        updater=SimpleNamespace(
            running=False,
            start_polling=AsyncMock(return_value=None),
        ),
    )
    status_writer = MagicMock()
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", status_writer)
    monkeypatch.setattr(adapter, "_drain_polling_connections", AsyncMock())
    monkeypatch.setattr(adapter, "_verify_polling_after_reconnect", AsyncMock())
    monkeypatch.setattr(telegram_module.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        env_loader,
        "get_effective_credential_source",
        lambda *args, **kwargs: "profile_env",
    )

    asyncio.run(adapter._handle_polling_network_error(OSError("network down")))

    readiness = [
        call.kwargs["platform_runtime"]["transport_ready"]
        for call in status_writer.call_args_list
        if "platform_runtime" in call.kwargs
    ]
    assert readiness == [False, True]
