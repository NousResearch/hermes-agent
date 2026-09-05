"""The startup attempt and success reach the same terminal-visible logging level."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as module


@pytest.mark.asyncio
async def test_connect_success_line_matches_attempt_line_visibility(monkeypatch, caplog):
    adapter = module.TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    app = SimpleNamespace(
        bot=SimpleNamespace(username="test_bot"), initialize=AsyncMock(), start=AsyncMock()
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(module, "Application", SimpleNamespace(builder=lambda: builder))
    monkeypatch.setattr(module, "TELEGRAM_AVAILABLE", True)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    monkeypatch.setattr(adapter, "_build_ptb_requests", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda app: None)
    monkeypatch.setattr(adapter, "_register_handlers", lambda app: None)
    monkeypatch.setattr(adapter, "_start_polling_mode", AsyncMock())
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", lambda: None)
    monkeypatch.setattr(adapter, "_restart_task_attr", lambda name, coro: coro.close())
    with caplog.at_level(logging.INFO, logger=module.logger.name):
        assert await adapter.connect()
    attempts = [r for r in caplog.records if "Connecting to Telegram (attempt" in r.getMessage()]
    successes = [r for r in caplog.records if "Connected to Telegram (" in r.getMessage()]
    assert len(attempts) == len(successes) == 1
    assert attempts[0].levelno == successes[0].levelno == logging.WARNING
