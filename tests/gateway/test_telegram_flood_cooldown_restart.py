"""A Telegram flood window survives a gateway restart (#89962).

A 429 ``retry_after`` arms a per-chat window inside which ``send()`` fails closed locally. The window
lived in memory only, so the next process fired straight into the penalty. These drive the real
``TelegramAdapter.send`` on two adapter instances sharing one profile home (old process / new process).
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _FloodError(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _adapter(home, token="7001:AAA-test") -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=token))
    adapter._update_receipt_dir = home  # the profile home the adapter was constructed in
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=7))
    return adapter


async def _flush(adapter):
    task = adapter.__dict__.get("_telegram_send_cooldown_flush")
    if task is not None:
        await task


@pytest.mark.asyncio
async def test_a_restart_inside_a_flood_window_keeps_the_chat_quiet(tmp_path):
    old = _adapter(tmp_path)
    old._bot.send_message = AsyncMock(side_effect=_FloodError(200.0))
    refused = await old.send("4242", "hello")
    assert refused.success is False and refused.error.startswith("flood_control:")
    await _flush(old)

    new = _adapter(tmp_path)  # the next process, same profile home and bot
    again = await new.send("4242", "after restart")
    assert again.success is False and again.error.startswith("flood_control:")
    new._bot.send_message.assert_not_called()  # no request fired into the penalty
    # Another chat of the same bot is unaffected.
    assert (await new.send("999", "other chat")).success is True
    new._bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_persisted_windows_are_scoped_to_their_bot_and_expire(tmp_path):
    old = _adapter(tmp_path)
    old._bot.send_message = AsyncMock(side_effect=_FloodError(200.0))
    await old.send("4242", "hello")
    await _flush(old)

    other_bot = _adapter(tmp_path, token="7002:BBB-test")  # another bot in the same home
    assert (await other_bot.send("4242", "hi")).success is True

    # An expired entry (the penalty ran out while the gateway was down) is not re-armed, and a
    # deadline far in the future (wall-clock jump, restored copy) is clamped to the window cap.
    store = tmp_path / "telegram_flood_cooldown_7001.json"
    store.write_text(json.dumps({"until": {"4242": 1.0, "555": 4.0e12}}), encoding="utf-8")
    fresh = _adapter(tmp_path)
    assert (await fresh.send("4242", "after expiry")).success is True
    assert 0 < fresh._send_flood_cooldown_remaining("555") <= 300.0
