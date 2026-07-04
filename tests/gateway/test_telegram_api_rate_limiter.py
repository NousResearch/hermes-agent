from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as adapter_mod
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(interval: float) -> TelegramAdapter:
    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"api_min_interval_seconds": interval},
    )
    adapter = TelegramAdapter(config)
    adapter._bot = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_outbound_bot_api_calls_are_paced_per_chat(monkeypatch):
    adapter = _adapter(0.5)
    msg1 = MagicMock(message_id=101)
    msg2 = MagicMock(message_id=102)
    adapter._bot.send_message = AsyncMock(side_effect=[msg1, msg2])

    now = {"value": 100.0}
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now["value"]

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now["value"] += delay

    monkeypatch.setattr(adapter_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(adapter_mod.asyncio, "sleep", fake_sleep)

    await adapter.send("12345", "first", metadata={"notify": True})
    await adapter.send("12345", "second", metadata={"notify": True})

    assert adapter._bot.send_message.await_count == 2
    assert sleeps == [pytest.approx(0.5)]


@pytest.mark.asyncio
async def test_outbound_bot_api_limiter_is_per_chat(monkeypatch):
    adapter = _adapter(0.5)
    adapter._bot.send_message = AsyncMock(
        side_effect=[MagicMock(message_id=101), MagicMock(message_id=102)]
    )

    now = {"value": 100.0}
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now["value"]

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now["value"] += delay

    monkeypatch.setattr(adapter_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(adapter_mod.asyncio, "sleep", fake_sleep)

    await adapter.send("111", "first", metadata={"notify": True})
    await adapter.send("222", "second", metadata={"notify": True})

    assert adapter._bot.send_message.await_count == 2
    assert sleeps == []


def test_retry_after_is_parsed_from_ptb_attribute_and_text():
    adapter = _adapter(0.5)

    class RetryAfterError(Exception):
        retry_after = 3

    assert adapter._telegram_api_retry_after_seconds(RetryAfterError("ignored")) == 3
    assert (
        adapter._telegram_api_retry_after_seconds(
            Exception("Flood control exceeded. Retry in 14 seconds")
        )
        == 14
    )
