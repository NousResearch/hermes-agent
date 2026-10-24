"""Regression for #123660: private-topic activity must stay in the reply lane."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


TOPIC = {"thread_id": "99", "telegram_dm_topic_reply_fallback": True}


def _adapter(draft):
    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="fake-token", extra={"rich_messages": False},
    ))
    adapter._bot = SimpleNamespace(
        send_message_draft=AsyncMock(side_effect=draft),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        send_chat_action=AsyncMock(),
        edit_message_text=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        delete_message=AsyncMock(return_value=True),
    )
    return adapter


def _consumer(adapter, *, chat_type="dm", metadata=TOPIC, transport="auto"):
    return GatewayStreamConsumer(
        adapter, "123", StreamConsumerConfig(
            transport=transport, chat_type=chat_type, edit_interval=0.01,
            buffer_threshold=1, cursor="",
        ), metadata=dict(metadata), initial_reply_to_id="7",
    )


@pytest.mark.asyncio
async def test_topic_activity_stays_ephemeral_and_yields_to_the_answer(monkeypatch):
    frames = asyncio.Queue()

    async def draft(**kwargs):
        frames.put_nowait(kwargs)
        return True

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("gateway.stream_consumer_transport.time", SimpleNamespace(
        monotonic=lambda: clock.now,
    ))
    adapter = _adapter(draft)
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    try:
        first = await asyncio.wait_for(frames.get(), timeout=2)
        assert first["text"] == ""
        assert first["message_thread_id"] == int(TOPIC["thread_id"])
        assert first["draft_id"] != 0
        assert not consumer.final_response_sent
        adapter._bot.send_message.assert_not_awaited()

        clock.now += 19
        await consumer._refresh_draft_activity()
        assert adapter._bot.send_message_draft.await_count == 1
        clock.now += 2
        refreshed = await asyncio.wait_for(frames.get(), timeout=2)
        assert refreshed == first

        consumer.on_delta("The answer")
        answer = await asyncio.wait_for(frames.get(), timeout=2)
        assert answer["text"] == "The answer"
        assert answer["draft_id"] == first["draft_id"]
        assert answer["message_thread_id"] == first["message_thread_id"]
        clock.now += 21
        await consumer._refresh_draft_activity()
        assert adapter._bot.send_message_draft.await_count == 3

        consumer.finish("The answer")
        await asyncio.wait_for(task, timeout=2)
        adapter._bot.send_message.assert_awaited_once()
        final = adapter._bot.send_message.call_args.kwargs
        assert final["text"] == answer["text"]
        assert final["message_thread_id"] == first["message_thread_id"]
        assert consumer.final_response_sent
    finally:
        task.cancel()
        await task

    # The empty preview belongs only to opted-in, non-general private topics.
    for chat_type, metadata, transport in (
        ("dm", {}, "auto"),
        ("dm", {**TOPIC, "thread_id": "1"}, "auto"),
        ("group", TOPIC, "auto"),
        ("dm", TOPIC, "edit"),
        ("dm", TOPIC, "off"),
    ):
        adapter = _adapter(draft)
        consumer = _consumer(adapter, chat_type=chat_type, metadata=metadata, transport=transport)
        await consumer._start_transports()
        await consumer._refresh_draft_activity()
        adapter._bot.send_message_draft.assert_not_awaited()

    adapter = _adapter(draft)
    consumer = _consumer(adapter)
    consumer.stream_deltas_enabled = False  # Created only for interim commentary.
    await consumer._start_transports()
    await consumer._refresh_draft_activity()
    adapter._bot.send_message_draft.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_topic_activity_falls_back_without_losing_the_reply():
    attempted = asyncio.Event()

    async def reject_draft(**kwargs):
        assert kwargs["text"] == ""
        assert kwargs["message_thread_id"] == int(TOPIC["thread_id"])
        attempted.set()
        return False

    adapter = _adapter(reject_draft)
    consumer = _consumer(adapter)
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(attempted.wait(), timeout=2)
        consumer.on_delta("The answer")
        consumer.finish("The answer")
        await asyncio.wait_for(task, timeout=2)
        adapter._bot.send_message_draft.assert_awaited_once()
        adapter._bot.send_message.assert_awaited_once()
        final = adapter._bot.send_message.call_args.kwargs
        assert final["text"] == "The answer"
        assert final["message_thread_id"] == int(TOPIC["thread_id"])
        assert consumer.final_response_sent
    finally:
        task.cancel()
        await task
