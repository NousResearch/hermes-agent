"""Regressions for late source routing and physical item order."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import Bot

from gateway.platforms.base import MessageType
from tests.test_telegram_forward_comment_ingress import adapter, make_update


@pytest.mark.asyncio
async def test_guest_routing_can_override_source_after_event_construction(adapter):
    # Guest Mode deliberately rewrites the source after _build_message_event.
    bot = Bot("123:test")
    event = adapter._build_message_event(make_update(bot, 1, text="guest question").message, MessageType.TEXT)
    event.guest_mode_invocation = True
    event.session_key_override = "guest:query:1"
    event.source.session_key_override = event.session_key_override
    event.source.chat_id = "guest:1"
    await adapter.handle_message(event)
    assert len(adapter.seen) == 1
    from gateway.session import build_session_key
    assert adapter.seen[0][1] == build_session_key(event.source)


@pytest.mark.asyncio
async def test_two_forwarded_album_documents_keep_source_order_when_downloads_reverse(adapter, monkeypatch):
    bot = Bot("123:test")
    first, second_ready = asyncio.Event(), asyncio.Event()
    async def get_file(_self, file_id, **kw):
        if file_id == "2":
            await first.wait()
        else:
            second_ready.set()
        return SimpleNamespace(file_path="/file.json", download_as_bytearray=AsyncMock(return_value=f'{{"id":{file_id}}}'.encode()))
    monkeypatch.setattr(Bot, "get_file", get_file)
    await adapter._handle_text_message(make_update(bot, 1, text="question"), None)
    tasks = [asyncio.create_task(adapter._handle_media_message(make_update(bot, i, media="document", album="same"), None)) for i in (2, 3)]
    await second_ready.wait()
    await asyncio.sleep(0)  # both finish before the normal album flush
    first.set()
    await asyncio.gather(*tasks)
    await asyncio.sleep(0.3)
    assert len(adapter.seen) == 1
    event = adapter.seen[0][0]
    assert event.text.index("Source 2") < event.text.index("Source 3")
    assert len(event.media_urls) == 2
    await adapter._cancel_pending_delivery_tasks()


@pytest.mark.asyncio
async def test_forward_later_in_text_batch_keeps_media_batch_eligibility(adapter):
    bot = Bot("123:test")
    ordinary = adapter._build_message_event(make_update(bot, 1, text="question").message, MessageType.TEXT)
    forwarded = adapter._build_message_event(make_update(bot, 2, text="forward").message, MessageType.TEXT)
    forwarded.forward_origin = {"sender_name": "Author"}
    adapter._enqueue_text_event(ordinary)
    adapter._enqueue_text_event(forwarded)
    await asyncio.sleep(0.025)
    # Still inside the known forward admission window, not ordinary fast path.
    assert not adapter.seen
    await asyncio.sleep(0.15)
    assert len(adapter.seen) == 1
    assert "Author" in adapter.seen[0][0].text
    await adapter._cancel_pending_delivery_tasks()
