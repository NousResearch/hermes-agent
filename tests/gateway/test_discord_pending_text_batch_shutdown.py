"""Regression guard for Discord text-batch flush during gateway shutdown."""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


from tests.discord_mock import ensure_discord_module as _ensure_discord_mock

_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_cancel_background_tasks_awaits_pending_text_batch_before_clearing():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    flushed = asyncio.Event()

    async def pending_flush():
        await asyncio.sleep(0)
        flushed.set()

    task = asyncio.create_task(pending_flush())
    adapter._pending_text_batch_tasks["chat"] = task
    adapter._pending_text_batches["chat"] = MessageEvent(
        text="pending",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="chat", chat_type="group"),
    )

    await adapter.cancel_background_tasks()

    assert flushed.is_set()
    assert task.done()
    assert adapter._pending_text_batch_tasks == {}
    assert adapter._pending_text_batches == {}
