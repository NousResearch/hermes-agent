"""Telegram webhook startup requires its scoped secret; polling does not."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", ["", "   ", " webhook-secret "])
async def test_webhook_branch_checks_secret(monkeypatch, secret):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    start = AsyncMock()
    adapter._app = SimpleNamespace(updater=SimpleNamespace(start_webhook=start))
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda name: secret)
    if not secret.strip():
        with pytest.raises(RuntimeError, match="TELEGRAM_WEBHOOK_SECRET is required"):
            await adapter._start_webhook_mode("https://example.org/hooks", is_reconnect=False)
        start.assert_not_awaited()
    else:
        await adapter._start_webhook_mode("https://example.org/hooks", is_reconnect=False)
        assert start.await_args.kwargs["secret_token"] == secret.strip()
        assert start.await_args.kwargs["url_path"] == "/hooks"


@pytest.mark.asyncio
async def test_polling_branch_has_no_secret_guard(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._delete_webhook_best_effort = AsyncMock()
    adapter._start_polling_resilient = AsyncMock(return_value=True)

    def no_webhook_secret(name):
        raise AssertionError("Polling must not read the webhook secret")

    monkeypatch.setattr("agent.secret_scope.get_secret", no_webhook_secret)
    await adapter._start_polling_mode(is_reconnect=False)
    adapter._start_polling_resilient.assert_awaited_once()
