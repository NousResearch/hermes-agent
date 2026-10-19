"""Shared per-chat send+edit pacing budget (#116312).

sendMessage and editMessageText count against the same per-chat Telegram allowance; a stream that edits
too often while the assistant's own backend sends were also in flight tripped FloodWait (23 events).
A per-chat 1/s slot shared by both: a SEND waits for its slot (never dropped), an INTERIM edit is
skipped (the next edit shows the same text anyway), and the FINAL edit is never gated (the completed
answer is always delivered).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(send_message: AsyncMock) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = send_message
    adapter._bot.edit_message_text = AsyncMock(return_value=MagicMock())
    return adapter


def _now() -> float:
    return asyncio.get_running_loop().time()


@pytest.mark.asyncio
async def test_interim_edit_skipped_when_slot_busy_but_final_edit_never_gated():
    """An interim edit while the shared slot is held returns success with the SAME message id and makes
    no API call; a finalize edit to the same chat still fires even though the slot is still held."""
    adapter = _adapter(AsyncMock())

    result = await adapter.edit_message("c1", "900", "part one", finalize=False)
    assert result.success is True  # consumed a slot on the first real edit

    busy = adapter._chat_outbound_slot_remaining( "c1")
    assert busy > 0

    skipped = await adapter.edit_message("c1", "900", "part two", finalize=False)
    assert skipped.success is True and skipped.message_id == "900"
    assert adapter._bot.edit_message_text.await_count == 1

    final = await adapter.edit_message("c1", "900", "part two final", finalize=True)
    assert final.success is True
    assert adapter._bot.edit_message_text.await_count == 2  # the final edit is never gated


@pytest.mark.asyncio
async def test_skipped_interim_edit_leaves_visible_prefix_untouched():
    """A slot-busy skip shows nothing new, so the stream consumer must not record the skipped text as
    the on-screen prefix: a later turn-final flood must not treat the unseen tail as delivered."""
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = _adapter(AsyncMock())
    consumer = GatewayStreamConsumer(adapter, "c1", StreamConsumerConfig(cursor="▌"))
    consumer._message_id = "900"
    consumer._already_sent = True
    consumer._flood_strikes = 1

    assert await consumer._edit_existing("The answer▌", finalize=False, is_turn_final=False)
    assert consumer._last_sent_text == "The answer▌"
    assert consumer._flood_strikes == 0
    consumer._flood_strikes = 1

    assert adapter._chat_outbound_slot_remaining("c1") > 0  # slot held: next interim edit is skipped
    assert await consumer._edit_existing("The answer is 42▌", finalize=False, is_turn_final=False)
    assert adapter._bot.edit_message_text.await_count == 1
    assert consumer._last_sent_text == "The answer▌"  # screen still shows the older preview
    assert consumer._flood_strikes == 1

    from gateway.platforms.base import SendResult

    adapter.edit_message = AsyncMock(return_value=SendResult(
        success=False, error="Flood control exceeded. Retry in 9 seconds", retry_after=9))
    assert not await consumer._edit_existing("The answer is 42", finalize=True, is_turn_final=True)
    assert not getattr(consumer, "_final_content_delivered", False)
    assert consumer._fallback_prefix == "The answer"  # fallback re-sends " is 42", not nothing


@pytest.mark.asyncio
async def test_send_waits_for_held_slot():
    """A send to a chat whose slot is held defers until the slot opens and then fires exactly once."""
    adapter = _adapter(AsyncMock())
    await adapter.edit_message("c1", "900", "one", finalize=False)  # hold the shared slot for c1

    sleeps: list[float] = []

    async def fake_sleep(seconds: float):
        sleeps.append(seconds)

    original_sleep = asyncio.sleep
    asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        result = await adapter.send("c1", "hello")
    finally:
        asyncio.sleep = original_sleep  # type: ignore[assignment]

    assert result.success is True
    assert adapter._bot.send_message.await_count == 1
    assert sleeps, "send should have awaited asyncio.sleep for the pending slot"
