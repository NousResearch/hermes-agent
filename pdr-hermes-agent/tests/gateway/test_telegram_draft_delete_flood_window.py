"""Draft frames and message deletes stay out of a chat's Telegram flood window.

A 429 ``retry_after`` arms the chat's send flood window; ``send_draft`` and ``delete_message``
must not fire Bot API calls into it (Telegram lengthens a penalty that keeps getting hit), and a
429 on either arms the window for the other paths. SDK-free: the refusal is any exception that
carries ``retry_after``, as PTB's ``RetryAfter`` does.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Flood(Exception):
    def __init__(self, seconds: float) -> None:
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    adapter._bot.delete_message = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_draft_frames_skip_the_flood_window_a_refused_draft_armed():
    adapter = _adapter()
    adapter._bot.send_message_draft = AsyncMock(side_effect=[_Flood(30), True])

    refused = await adapter.send_draft("4242", 7, "frame 1")
    later = [await adapter.send_draft("4242", 7, f"frame {i}") for i in (2, 3)]

    # The refusal keeps the draft transport (a skipped frame, not a failure) and fires nothing more.
    assert refused.success and refused.raw_response == {"skipped": True}
    assert all(r.success and r.raw_response == {"skipped": True} for r in later)
    assert adapter._bot.send_message_draft.await_count == 1
    # The window is the chat's: a send refuses locally too, and another chat still drafts.
    assert (await adapter.send("4242", "hello")).error.startswith("flood_control:")
    assert (await adapter.send_draft("777", 1, "other chat")).raw_response is None
    assert adapter._bot.send_message_draft.await_count == 2


@pytest.mark.asyncio
async def test_delete_waits_out_the_flood_window_then_deletes():
    adapter = _adapter()
    adapter._record_send_flood_cooldown("4242", 1.0)  # a refused send armed a short window

    task = asyncio.create_task(adapter.delete_message("4242", "55"))
    for _ in range(5):
        await asyncio.sleep(0)  # let the delete reach its wait
    assert adapter._bot.delete_message.await_count == 0  # nothing fired into the window
    assert await task is True
    adapter._bot.delete_message.assert_awaited_once_with(chat_id=4242, message_id=55)


@pytest.mark.asyncio
async def test_long_flood_window_defers_the_delete_and_a_refused_delete_arms_it():
    adapter = _adapter()
    adapter._bot.delete_message = AsyncMock(side_effect=_Flood(60))
    assert await adapter.delete_message("4242", "55") is False  # the 429 itself
    assert adapter._send_flood_cooldown_remaining("4242") is not None

    adapter._bot.delete_message = AsyncMock(return_value=True)
    # Past the inline cap the delete is owned by a tracked background task: the caller is not parked.
    assert await asyncio.wait_for(adapter.delete_message("4242", "56"), 0.5) is True
    assert adapter._bot.delete_message.await_count == 0
    pending = [t for t in adapter._background_tasks if not t.done()]
    assert len(pending) == 1
    await adapter.cancel_background_tasks()
