"""Startup ownership, admission cutoff and reconnect ready-event retention."""
import asyncio

import pytest
from telegram import Bot

from gateway.platforms.base import MessageType
from tests.test_telegram_forward_comment_ingress import adapter, make_update
from tests.test_telegram_forward_comment_lifecycle import install_download


@pytest.mark.asyncio
async def test_later_ordinary_message_is_not_swallowed_during_download_drain(adapter, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    install_download(monkeypatch, release, started)
    bot = Bot("123:test")
    await adapter._handle_text_message(make_update(bot, 1, text="first question"), None)
    download = asyncio.create_task(adapter._handle_media_message(make_update(bot, 2, media="video"), None))
    await started.wait()
    await asyncio.sleep(0.2)
    await adapter._handle_text_message(make_update(bot, 3, text="independent next question"), None)
    await asyncio.sleep(0.04)
    release.set()
    await download
    await asyncio.sleep(0.15)
    assert len(adapter.seen) == 2
    first, later = [e for e, _ in adapter.seen]
    assert "first question" in first.text and len(first.media_urls) == 1
    assert "independent next question" not in first.text
    assert later.text == "independent next question" and not later.media_urls
    await adapter._cancel_pending_delivery_tasks()


@pytest.mark.asyncio
async def test_retryable_disconnect_keeps_ready_startup_event(adapter):
    bot = Bot("123:test")
    event = adapter._build_message_event(make_update(bot, 1, text="ready forward").message, MessageType.TEXT)
    event.forward_origin = {"sender_name": "Source"}
    await adapter._dispatch_ingress_event(event)
    assert adapter._ingress_coordinator().batches
    adapter._drop_delayed_deliveries = True
    await adapter._cancel_pending_delivery_tasks()
    held = adapter._held_inbound_events
    assert held == [event]
    assert adapter._ingress_coordinator().current(event)
    adapter._held_inbound_events = []
    adapter._drop_delayed_deliveries = False
    await adapter.handle_message(event)
    assert [e for e, _ in adapter.seen] == [event]


@pytest.mark.asyncio
async def test_forward_origins_stay_with_each_item_in_text_debounce(adapter):
    bot = Bot("123:test")
    for i in (1, 2):
        event = adapter._build_message_event(make_update(bot, i, text=f"post {i}").message, MessageType.TEXT)
        event.forward_origin = {"sender_name": f"Author {i}"}
        adapter._enqueue_text_event(event)
    await asyncio.sleep(0.2)
    assert len(adapter.seen) == 1
    text = adapter.seen[0][0].text
    assert text.index("Author 1") < text.index("post 1") < text.index("Author 2") < text.index("post 2")
    await adapter._cancel_pending_delivery_tasks()


@pytest.mark.asyncio
async def test_input_during_final_dispatch_is_not_lost(adapter):
    bot = Bot("123:test")
    event = adapter._build_message_event(make_update(bot, 1, text="post").message, MessageType.TEXT)
    event.forward_origin = {"sender_name": "Author"}
    entered, release = asyncio.Event(), asyncio.Event()
    delivered = []
    async def dispatch(event):
        if not delivered:
            entered.set()
            await release.wait()
        delivered.append(event)
    adapter.handle_message = dispatch
    await adapter._dispatch_ingress_event(event)
    await asyncio.wait_for(entered.wait(), 1)
    other = adapter._build_message_event(make_update(bot, 2, text="later").message, MessageType.TEXT)
    await adapter._dispatch_ingress_event(other)
    release.set()
    await asyncio.sleep(0.05)
    assert len(delivered) == 2
    assert delivered[1].text == "later"
    await adapter._cancel_pending_delivery_tasks()
