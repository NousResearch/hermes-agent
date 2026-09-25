"""Telegram post-send typing re-arm is scheduled off the send path and collapsed per chat (#111727).

Awaiting ``sendChatAction`` after every intermediate send ran its TLS round-trip on the same event
loop as the ``getUpdates`` long-polls; under concurrent streaming the polls starved until they
rotted into CLOSE-WAIT while the adapter still reported ``connected``.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._rich_messages_enabled = False
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    adapter._bot.send_chat_action = AsyncMock(return_value=None)
    # This test models a burst of streamed chunks landing back-to-back; the shared send+edit
    # pacing slot (#116312) would space those sends out to ~1/s, so real time elapses between them
    # and typing re-arms — exactly its collapse cadence. Shut the slot off to keep the burst
    # instantaneous so only the per-chat collapse is measured.
    adapter._telegram_chat_outbound_slot_secs = 0.0
    return adapter


async def _drain(adapter):
    pending = [t for t in adapter._background_tasks if not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=2)


@pytest.mark.asyncio
async def test_send_returns_without_waiting_on_typing():
    """An intermediate send completes while sendChatAction is still stalled."""
    adapter = _make_adapter()
    released = asyncio.Event()

    async def blocking_action(**kwargs):
        await released.wait()

    adapter._bot.send_chat_action = AsyncMock(side_effect=blocking_action)

    result = await asyncio.wait_for(adapter.send("123", "chunk"), timeout=1.0)

    assert result.success is True
    released.set()
    await _drain(adapter)
    assert adapter._bot.send_chat_action.await_count == 1


@pytest.mark.asyncio
async def test_streaming_burst_collapses_to_one_chat_action_per_chat():
    """Every streamed chunk re-arms typing; within the interval that is one API call per chat."""
    adapter = _make_adapter()

    for i in range(20):
        assert (await adapter.send("123", f"chunk {i}")).success
        assert (await adapter.send("456", f"chunk {i}")).success
    await _drain(adapter)

    assert adapter._bot.send_chat_action.await_count == 2
