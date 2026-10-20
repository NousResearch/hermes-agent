"""Tests for GHSA-3vpc-7q5r-276h — Telegram webhook secret required.

Previously, when TELEGRAM_WEBHOOK_URL was set but TELEGRAM_WEBHOOK_SECRET
was not, python-telegram-bot received secret_token=None and the webhook
endpoint accepted any HTTP POST.

The fix refuses to start the adapter in webhook mode without the secret.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = MagicMock()
    adapter._app.updater.start_webhook = AsyncMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", [None, "", "   "], ids=["unset", "empty", "blank"])
async def test_webhook_mode_refuses_to_start_without_secret(monkeypatch, secret):
    if secret is None:
        monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", secret)
    adapter = _adapter()

    with pytest.raises(RuntimeError):
        await adapter._start_webhook_mode("https://example.test/telegram", is_reconnect=False)

    adapter._app.updater.start_webhook.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_mode_registers_secret_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "  s3cret  ")
    adapter = _adapter()

    await adapter._start_webhook_mode("https://example.test/telegram", is_reconnect=False)

    adapter._app.updater.start_webhook.assert_awaited_once()
    assert adapter._app.updater.start_webhook.await_args.kwargs["secret_token"] == "s3cret"
