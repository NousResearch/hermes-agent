"""Text-debounce batching for the WhatsApp adapter (issue #35301).

WhatsApp delivers rapid multi-message bursts (forwarded batches, paste-splits)
individually.  Without debounce each fragment triggers a separate agent
invocation, wasting tokens and flooding the user with reply fragments.  This
mirrors the Telegram/WeCom/Feishu pattern.

Batch delays are read from ``config.extra`` (config.yaml), not env vars.
"""

import asyncio
from unittest.mock import AsyncMock, call

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from gateway.session import SessionSource


def _make_adapter(**extra):
    base = {"session_name": "test"}
    base.update(extra)
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=base))


def _event(
    text,
    *,
    chat_id="chat123",
    chat_type="dm",
    user_id="user1",
    user_name="tester",
):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=src)


def test_batch_delays_overridden_via_config_extra():
    adapter = _make_adapter(
        text_batch_delay_seconds="2.5",
        text_batch_split_delay_seconds=7,
    )
    assert adapter._text_batch_delay_seconds == 2.5
    assert adapter._text_batch_split_delay_seconds == 7.0


def test_invalid_config_value_falls_back_to_default():
    adapter = _make_adapter(
        text_batch_delay_seconds="garbage",
        text_batch_split_delay_seconds=-3,
    )
    assert adapter._text_batch_delay_seconds == 5.0
    assert adapter._text_batch_split_delay_seconds == 10.0


def test_same_sender_messages_share_one_batch():
    async def run():
        adapter = _make_adapter(text_batch_delay_seconds=60)
        adapter.handle_message = AsyncMock()

        await adapter._enqueue_text_event(_event("first"))
        await adapter._enqueue_text_event(_event("second"))

        assert adapter.handle_message.await_count == 0
        pending = next(iter(adapter._pending_text_batches.values()))
        assert pending.text == "first\nsecond"
        await adapter._flush_text_batch_now(adapter._text_batch_key(pending))
        adapter.handle_message.assert_awaited_once_with(pending)
        assert not adapter._pending_text_batches
        assert not adapter._pending_text_batch_tasks

    asyncio.run(run())


def test_sender_change_flushes_before_starting_next_batch():
    async def run():
        adapter = _make_adapter(
            group_sessions_per_user=False,
            text_batch_delay_seconds=60,
        )
        adapter.handle_message = AsyncMock()
        first = _event(
            "from Alice",
            chat_id="shared@g.us",
            chat_type="group",
            user_id="15550000001@s.whatsapp.net",
            user_name="Alice",
        )
        second = _event(
            "from Bob",
            chat_id="shared@g.us",
            chat_type="group",
            user_id="15550000002@s.whatsapp.net",
            user_name="Bob",
        )

        assert adapter._text_batch_key(first) == adapter._text_batch_key(second)
        await adapter._enqueue_text_event(first)
        await adapter._enqueue_text_event(second)

        adapter.handle_message.assert_awaited_once_with(first)
        pending = next(iter(adapter._pending_text_batches.values()))
        assert pending is second
        assert pending.source.user_name == "Bob"
        await adapter._flush_text_batch_now(adapter._text_batch_key(second))
        assert adapter.handle_message.await_args_list == [call(first), call(second)]
        assert not adapter._pending_text_batches
        assert not adapter._pending_text_batch_tasks

    asyncio.run(run())


def test_sender_change_waits_for_in_flight_flush_without_cancelling_it():
    async def run():
        adapter = _make_adapter(
            group_sessions_per_user=False,
            text_batch_delay_seconds=0,
        )
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        handled = []

        async def handle(event):
            handled.append(event)
            if len(handled) == 1:
                first_started.set()
                await release_first.wait()

        adapter.handle_message = AsyncMock(side_effect=handle)
        first = _event(
            "from Alice",
            chat_id="shared@g.us",
            chat_type="group",
            user_id="15550000001@s.whatsapp.net",
            user_name="Alice",
        )
        second = _event(
            "from Bob",
            chat_id="shared@g.us",
            chat_type="group",
            user_id="15550000002@s.whatsapp.net",
            user_name="Bob",
        )

        await adapter._enqueue_text_event(first)
        await asyncio.wait_for(first_started.wait(), timeout=1)
        adapter._text_batch_delay_seconds = 60
        enqueue_second = asyncio.create_task(adapter._enqueue_text_event(second))
        await asyncio.sleep(0)
        assert not enqueue_second.done()

        release_first.set()
        await asyncio.wait_for(enqueue_second, timeout=1)
        assert handled == [first]
        assert next(iter(adapter._pending_text_batches.values())) is second
        await adapter._flush_text_batch_now(adapter._text_batch_key(second))
        assert handled == [first, second]

    asyncio.run(run())


def test_unknown_group_senders_are_not_merged():
    async def run():
        adapter = _make_adapter(
            group_sessions_per_user=False,
            text_batch_delay_seconds=60,
        )
        adapter.handle_message = AsyncMock()
        first = _event(
            "first",
            chat_id="shared@g.us",
            chat_type="group",
            user_id=None,
            user_name=None,
        )
        second = _event(
            "second",
            chat_id="shared@g.us",
            chat_type="group",
            user_id=None,
            user_name=None,
        )

        await adapter._enqueue_text_event(first)
        await adapter._enqueue_text_event(second)

        adapter.handle_message.assert_awaited_once_with(first)
        assert next(iter(adapter._pending_text_batches.values())) is second
        await adapter._flush_text_batch_now(adapter._text_batch_key(second))
        assert adapter.handle_message.await_args_list == [call(first), call(second)]
        assert not adapter._pending_text_batches
        assert not adapter._pending_text_batch_tasks

    asyncio.run(run())
