"""Lifecycle and queue isolation probes for the real Telegram ingress path."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import Bot

from tests.test_telegram_forward_comment_ingress import adapter, make_update
from gateway.platforms.base import MessageEvent, MessageType


def install_download(monkeypatch, release, started=None):
    async def get_file(_self, file_id, **kw):
        if started:
            started.set()
        await release.wait()
        return SimpleNamespace(file_path="/video.mp4", download_as_bytearray=AsyncMock(return_value=b"{}"))
    monkeypatch.setattr(Bot, "get_file", get_file)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["stop", "new", "reset"])
async def test_boundary_drops_inflight_media_and_preserves_new_input(adapter, monkeypatch, command):
    release, started = asyncio.Event(), asyncio.Event()
    install_download(monkeypatch, release, started)
    bot = Bot("123:test")
    download = asyncio.create_task(adapter._handle_media_message(make_update(bot, 2, media="video"), None))
    await started.wait()
    # No agent/active-session guard exists at the boundary.
    await adapter._handle_command(make_update(bot, 3, text=f"/{command}"), None)
    await adapter._handle_text_message(make_update(bot, 4, text="fresh input"), None)
    release.set()
    await download
    await asyncio.sleep(0.2)
    assert [e.text for e, _ in adapter.seen] == [f"/{command}", "fresh input"]
    assert not adapter._ingress_coordinator().downloads
    assert not adapter._ingress_coordinator().batches
    await adapter._cancel_pending_delivery_tasks()


@pytest.mark.asyncio
async def test_reserved_callback_cannot_reenter_after_stop(adapter, monkeypatch):
    bot = Bot("123:test")
    update = make_update(bot, 2, media="video")
    await adapter._reserve_media_ingress(update, None)
    await adapter._handle_command(make_update(bot, 3, text="/stop"), None)
    get_file = AsyncMock(side_effect=AssertionError("stale callback downloaded"))
    monkeypatch.setattr(Bot, "get_file", get_file)
    await adapter._handle_media_message(update, None)
    get_file.assert_not_called()
    assert [e.text for e, _ in adapter.seen] == ["/stop"]
    assert not adapter._ingress_coordinator().downloads


@pytest.mark.asyncio
async def test_final_base_seam_rechecks_epoch_after_topic_recovery(adapter):
    bot = Bot("123:test")
    event = adapter._build_message_event(make_update(bot, 1, text="stale").message, MessageType.TEXT)
    key = adapter._text_batch_key(event)
    adapter._ingress_coordinator().stamp(event)
    entered, release = threading.Event(), threading.Event()
    def recover(e):
        entered.set()
        assert release.wait(2)
    adapter._topic_recovery_fn = recover
    task = asyncio.create_task(adapter.handle_message(event))
    await asyncio.to_thread(entered.wait, 2)
    adapter._ingress_coordinator().invalidate(key)
    release.set()
    await task
    assert adapter.seen == []


@pytest.mark.asyncio
async def test_slow_second_album_document_and_fifo_are_preserved(adapter, monkeypatch):
    bot = Bot("123:test")
    slow, entered = asyncio.Event(), asyncio.Event()
    async def get_file(_self, file_id, **kw):
        if file_id == "3":
            entered.set()
            await slow.wait()
        return SimpleNamespace(file_path="/document.json", download_as_bytearray=AsyncMock(return_value=b"{}"))
    monkeypatch.setattr(Bot, "get_file", get_file)
    comment = adapter._build_message_event(make_update(bot, 1, text="comment").message, MessageType.TEXT)
    key = adapter._text_batch_key(comment)
    head = MessageEvent(text="explicit queued task", source=comment.source)
    adapter._pending_messages[key] = head
    adapter._enqueue_text_event(comment)
    downloads = [asyncio.create_task(adapter._handle_media_message(make_update(bot, i, media="document", album="album"), None)) for i in (2, 3)]
    await entered.wait()
    await asyncio.sleep(0.2)  # beyond album debounce and admission, inside drain
    assert not adapter.seen
    slow.set()
    await asyncio.gather(*downloads)
    await asyncio.sleep(0.15)
    assert len(adapter.seen) == 1
    event = adapter.seen[0][0]
    assert len(event.media_urls) == 2
    assert event.text.index("Source 2") < event.text.index("Source 3")
    assert "caption 2" in event.text and "caption 3" in event.text
    assert adapter._pending_messages[key] is head
    assert not adapter._media_group_events
    await adapter._cancel_pending_delivery_tasks()


@pytest.mark.asyncio
async def test_ordinary_text_keeps_fast_path(adapter):
    bot = Bot("123:test")
    await adapter._handle_text_message(make_update(bot, 1, text="ordinary"), None)
    await asyncio.sleep(0.04)
    assert len(adapter.seen) == 1
    assert not adapter._ingress_coordinator().batches


@pytest.mark.asyncio
async def test_secondary_profile_uses_same_real_base_guard(adapter):
    adapter._owner_profile = "secondary"
    bot = Bot("123:test")
    event = adapter._build_message_event(make_update(bot, 1, text="hello").message, MessageType.TEXT)
    key = adapter._text_batch_key(event)
    assert key.startswith("agent:secondary:")
    adapter._active_sessions[key] = asyncio.Event()
    adapter._heal_stale_session_lock = lambda _: None
    adapter._busy_session_handler = AsyncMock(return_value=True)
    await adapter._dispatch_ingress_event(event)
    adapter._busy_session_handler.assert_awaited_once_with(event, key)
    assert adapter.seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["video", "audio", "voice", "document"])
async def test_rejected_media_does_not_reserve_or_download(adapter, monkeypatch, kind):
    bot = Bot("123:test")
    u = make_update(bot, 2, media=kind)
    obj = u.to_dict()
    obj["message"][kind]["file_size"] = 10**12
    from telegram import Update
    u = Update.de_json(obj, bot)
    download = AsyncMock(side_effect=AssertionError("oversized media downloaded"))
    monkeypatch.setattr(Bot, "get_file", download)
    await adapter._reserve_media_ingress(u, None)
    assert not adapter._ingress_coordinator().downloads
    await adapter._handle_media_message(u, None)
    download.assert_not_called()
    assert not adapter._ingress_coordinator().downloads
    await adapter._cancel_pending_delivery_tasks()


@pytest.mark.asyncio
async def test_hung_download_is_bounded_and_late_media_stays_deliverable(adapter, monkeypatch):
    release, started = asyncio.Event(), asyncio.Event()
    install_download(monkeypatch, release, started)
    c = adapter._ingress_coordinator()
    c.DRAIN_SECONDS = 0.16
    bot = Bot("123:test")
    await adapter._handle_text_message(make_update(bot, 1, text="comment"), None)
    task = asyncio.create_task(adapter._handle_media_message(make_update(bot, 2, media="video"), None))
    await started.wait()
    await asyncio.sleep(0.25)
    assert len(adapter.seen) == 1 and not adapter.seen[0][0].media_urls
    release.set()
    await task
    await asyncio.sleep(0.2)
    assert len(adapter.seen) == 2 and len(adapter.seen[1][0].media_urls) == 1
    await adapter._cancel_pending_delivery_tasks()


@pytest.mark.asyncio
async def test_disconnect_epoch_cannot_be_reused_by_old_download(adapter, monkeypatch):
    release, started = asyncio.Event(), asyncio.Event()
    install_download(monkeypatch, release, started)
    bot = Bot("123:test")
    task = asyncio.create_task(adapter._handle_media_message(make_update(bot, 1, media="video"), None))
    await started.wait()
    c = adapter._ingress_coordinator()
    old = c.downloads[1]
    await adapter._cancel_pending_delivery_tasks()
    fresh = adapter._build_message_event(make_update(bot, 2, text="fresh").message, MessageType.TEXT)
    assert c.stamp(fresh).epoch != old.epoch
    release.set()
    await task
    await asyncio.sleep(0.15)
    assert not adapter.seen
    assert not c.downloads
    await adapter._cancel_pending_delivery_tasks()


@pytest.mark.asyncio
async def test_real_ptb_dispatch_keeps_comment_runnable_during_media_download(adapter, monkeypatch):
    from telegram import User
    from telegram.ext import Application

    release, started = asyncio.Event(), asyncio.Event()
    install_download(monkeypatch, release, started)
    adapter._handle_guest_update = AsyncMock()
    adapter._on_platform_update = AsyncMock()
    app = Application.builder().token("123:test").build()
    # No network: exercise PTB's real process_update/filter/task machinery.
    app._initialized = True
    app._running = True
    app.bot._bot_user = User(987, "test", True, username="testbot")
    adapter._register_handlers(app)
    try:
        await app.process_update(make_update(app.bot, 2, media="video"))
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(app.process_update(make_update(app.bot, 3, text="comment")), 0.1)
        await asyncio.sleep(0.05)
        assert not adapter.seen
        assert adapter._ingress_coordinator().downloads
        release.set()
        await asyncio.gather(*list(app._Application__create_task_tasks))
        await asyncio.sleep(0.2)
        assert len(adapter.seen) == 1
        event = adapter.seen[0][0]
        assert "comment" in event.text and "caption 2" in event.text
        assert len(event.media_urls) == 1
    finally:
        release.set()
        await asyncio.gather(*list(app._Application__create_task_tasks), return_exceptions=True)
        app._running = False
        await adapter._cancel_pending_delivery_tasks()
