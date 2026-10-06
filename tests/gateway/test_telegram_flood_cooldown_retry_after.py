"""The per-chat flood cooldown honours Telegram's full retry_after, not a five-minute cap.

Measured 2026-09-27 on ai-hub (chat 1335137548): Telegram answered ``Retry in 27156 seconds``.
The adapter armed a local window of ``min(wait, 300)``; five minutes later the next queued reply
went to the API again, Telegram answered with a fresh multi-hour penalty, and the chat stayed
banned for the rest of the evening: 19:26 -> 27156 s, 19:31 -> 26853 s, 23:39 -> 11929 s.
Every re-probe inside the penalty extended it.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import (
    FLOOD_COOLDOWN_MAX_SECONDS,
    TelegramAdapter,
)


class _FloodError(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _adapter(send_message: AsyncMock) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = send_message
    return adapter


@pytest.mark.asyncio
async def test_multi_hour_retry_after_keeps_the_chat_closed_past_five_minutes(monkeypatch):
    calls = {"n": 0}

    async def fake_send_message(text: str, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FloodError(27156.0)
        return MagicMock(message_id=1000 + calls["n"])

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())

    first = await adapter.send("1335137548", "hello")
    assert first.success is False and first.error == "flood_control:27156.0"
    assert calls["n"] == 1

    # Jump the loop clock six minutes ahead: past the old 300 s cap, far inside the real penalty.
    loop = asyncio.get_running_loop()
    real_time = loop.time
    monkeypatch.setattr(loop, "time", lambda: real_time() + 360.0)

    again = await adapter.send("1335137548", "hello again")
    assert again.success is False and again.error.startswith("flood_control:")
    assert calls["n"] == 1, "a send inside Telegram's penalty must not reach the API"

    remaining = adapter._send_flood_cooldown_remaining("1335137548")
    assert remaining is not None and remaining > 3600, remaining


@pytest.mark.asyncio
async def test_cooldown_is_bounded_so_a_bogus_retry_after_cannot_park_a_chat_forever(monkeypatch):
    calls = {"n": 0}

    async def fake_send_message(text: str, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FloodError(10 ** 9)
        return MagicMock(message_id=1)

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())

    await adapter.send("4242", "hello")
    remaining = adapter._send_flood_cooldown_remaining("4242")
    assert remaining is not None
    assert remaining <= FLOOD_COOLDOWN_MAX_SECONDS + 1.0


@pytest.mark.asyncio
async def test_short_retry_after_still_reopens_the_chat(monkeypatch):
    calls = {"n": 0}

    async def fake_send_message(text: str, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FloodError(30.0)
        return MagicMock(message_id=1000 + calls["n"])

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())

    await adapter.send("4242", "hello")
    loop = asyncio.get_running_loop()
    real_time = loop.time
    monkeypatch.setattr(loop, "time", lambda: real_time() + 31.0)

    later = await adapter.send("4242", "hello later")
    assert later.success is True and calls["n"] == 2
