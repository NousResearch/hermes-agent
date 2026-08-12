"""Tests for the Telegram adapter's handler-registration extension point."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_module
from plugins.platforms.telegram.adapter import TelegramAdapter


def test_subclass_can_extend_telegram_handler_registration():
    custom_handler = object()

    class ExtendedTelegramAdapter(TelegramAdapter):
        def _register_handlers(self, application):
            super()._register_handlers(application)
            application.add_handler(custom_handler)

    application = MagicMock()
    adapter = ExtendedTelegramAdapter(PlatformConfig(enabled=True, token="test-token"))

    adapter._register_handlers(application)

    assert application.add_handler.call_count >= 2
    assert application.add_handler.call_args_list[-1].args == (custom_handler,)


@pytest.mark.asyncio
async def test_connect_restores_overridden_handlers_after_application_rebuild(
    monkeypatch,
):
    class ExtendedTelegramAdapter(TelegramAdapter):
        def __init__(self, config):
            super().__init__(config)
            self.handler_applications = []

        def _register_handlers(self, application):
            self.handler_applications.append(application)

    adapter = ExtendedTelegramAdapter(PlatformConfig(enabled=True, token="test-token"))

    first_app = SimpleNamespace(
        bot=SimpleNamespace(username="test_bot"),
        initialize=AsyncMock(side_effect=OSError("temporary reset")),
        shutdown=AsyncMock(),
    )
    rebuilt_app = SimpleNamespace(
        bot=SimpleNamespace(username="test_bot"),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.side_effect = [first_app, rebuilt_app]

    monkeypatch.setattr(
        telegram_module,
        "Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )
    monkeypatch.setattr(telegram_module, "HTTPXRequest", MagicMock)
    monkeypatch.setattr(
        telegram_module, "discover_fallback_ips", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(telegram_module.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock", lambda scope, identity: None
    )
    monkeypatch.setattr(adapter, "_delete_webhook_best_effort", AsyncMock())
    monkeypatch.setattr(
        adapter, "_start_polling_resilient", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(adapter, "_polling_heartbeat_loop", AsyncMock())
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", MagicMock())

    assert await adapter.connect() is True
    assert adapter.handler_applications == [first_app, rebuilt_app]
