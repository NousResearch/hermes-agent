"""Cold-boot pending-queue knob for the Telegram adapter.

Contract: a cold boot drops Telegram's server-side pending updates unless
``extra.drop_pending_on_cold_boot`` is false; a watcher reconnect always
preserves them. Conflict recovery is a separate path and is not covered here.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(extra=None) -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))


async def _capture_drop_pending(adapter: TelegramAdapter, *, is_reconnect: bool):
    """Run _start_polling_mode with staging mocked; return the forwarded flag."""
    captured = {}
    adapter._delete_webhook_best_effort = AsyncMock()

    async def _fake_resilient(*, drop_pending_updates, error_callback, require_progress=False):
        captured["drop_pending_updates"] = drop_pending_updates
        captured["require_progress"] = require_progress
        return True

    adapter._start_polling_resilient = _fake_resilient
    await adapter._start_polling_mode(is_reconnect=is_reconnect)
    return captured


@pytest.mark.asyncio
async def test_cold_boot_drops_queue_by_default():
    """Default config: cold boot drops, reconnect preserves."""
    adapter = _make_adapter()
    assert adapter._drop_pending_on_cold_boot is True

    cold = await _capture_drop_pending(adapter, is_reconnect=False)
    assert cold["drop_pending_updates"] is True

    warm = await _capture_drop_pending(adapter, is_reconnect=True)
    assert warm["drop_pending_updates"] is False


@pytest.mark.asyncio
async def test_cold_boot_preserves_queue_when_opted_out():
    """extra.drop_pending_on_cold_boot=false: cold boot preserves the backlog."""
    adapter = _make_adapter(extra={"drop_pending_on_cold_boot": False})
    assert adapter._drop_pending_on_cold_boot is False

    cold = await _capture_drop_pending(adapter, is_reconnect=False)
    assert cold["drop_pending_updates"] is False

    warm = await _capture_drop_pending(adapter, is_reconnect=True)
    assert warm["drop_pending_updates"] is False


async def _capture_webhook_drop_pending(
    adapter: TelegramAdapter, monkeypatch, *, is_reconnect: bool
):
    """Run _start_webhook_mode against a stub updater; return the forwarded flag."""
    captured = {}

    async def _fake_start_webhook(**kwargs):
        captured.update(kwargs)

    adapter._app = SimpleNamespace(
        updater=SimpleNamespace(start_webhook=AsyncMock(side_effect=_fake_start_webhook))
    )
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.test/telegram")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    await adapter._start_webhook_mode(
        "https://example.test/telegram", is_reconnect=is_reconnect
    )
    return captured


@pytest.mark.asyncio
async def test_webhook_cold_boot_honors_the_knob_and_logs_the_decision(monkeypatch, caplog):
    """The webhook start path forwards the same policy as polling, and says which one
    it applied. The flag is not a no-op there: PTB feeds it to delete_webhook /
    set_webhook, which drop Telegram's stored updates."""
    adapter = _make_adapter(extra={"drop_pending_on_cold_boot": False})

    with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
        captured = await _capture_webhook_drop_pending(
            adapter, monkeypatch, is_reconnect=False
        )

    assert captured["drop_pending_updates"] is False
    assert any("preserving" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_webhook_reconnect_preserves_queue_regardless_of_the_knob(monkeypatch):
    """A watcher reconnect keeps #46621's guarantee even when the knob asks cold
    boots to drop."""
    adapter = _make_adapter()

    captured = await _capture_webhook_drop_pending(adapter, monkeypatch, is_reconnect=True)

    assert captured["drop_pending_updates"] is False
