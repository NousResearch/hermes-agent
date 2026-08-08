"""Regression tests for stale Telegram ingress fragments across session boundaries.

Production evidence: a forwarded video update stayed in Telegram startup
buffers after /stop and merged into an unrelated later ordinary text message
(gateway logged 'Merged 1 Telegram startup attachment(s)').
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="273403055",
        chat_type="dm",
        user_id="273403055",
    )


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    return adapter


def _photo_event(source: SessionSource) -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/stale-video.jpg"],
        media_types=["image/jpeg"],
    )


def test_invalidate_session_ingress_clears_startup_slot():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()

    adapter.queue_startup_batch_event(session_key, _photo_event(source))
    assert adapter.has_startup_media_pending(session_key) is True

    adapter.invalidate_session_ingress(session_key)

    assert adapter._startup_batch_events == {}
    assert adapter.has_startup_media_pending(session_key) is False
    assert adapter.pop_startup_media_event(session_key) is None


def test_invalidate_session_ingress_clears_text_and_photo_buffers():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()

    text_event = MessageEvent(
        text="[Forwarded message | Chat: X]\n\nbody",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter._pending_text_batches[session_key] = text_event
    adapter._pending_photo_batches[f"{session_key}:photo-burst"] = _photo_event(source)
    adapter._media_downloads_in_progress_by_session[session_key] = 1

    adapter.invalidate_session_ingress(session_key)

    assert session_key not in adapter._pending_text_batches
    assert f"{session_key}:photo-burst" not in adapter._pending_photo_batches
    assert adapter._media_downloads_in_progress_by_session.get(session_key, 0) == 0
    assert adapter.has_startup_media_pending(session_key) is False


@pytest.mark.asyncio
async def test_late_photo_completion_after_invalidate_is_dropped_not_enqueued():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()

    event = _photo_event(source)
    adapter._stamp_ingress_generation(event)
    adapter.invalidate_session_ingress(session_key)

    batch_key = f"{session_key}:photo-burst"
    adapter._enqueue_photo_event(batch_key, event)

    assert adapter._pending_photo_batches == {}


@pytest.mark.asyncio
async def test_text_stamped_before_boundary_is_not_enqueued_afterward():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()
    event = MessageEvent(text="old text", message_type=MessageType.TEXT, source=source)
    adapter._stamp_ingress_generation(event)
    adapter.invalidate_session_ingress(session_key)

    adapter._enqueue_text_event(event)

    assert adapter._pending_text_batches == {}
    assert adapter._pending_text_batch_tasks == {}


@pytest.mark.asyncio
async def test_new_events_after_invalidate_are_accepted():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()

    adapter.invalidate_session_ingress(session_key)
    event = _photo_event(source)
    adapter._stamp_ingress_generation(event)
    adapter._enqueue_photo_event(f"{session_key}:photo-burst", event)

    assert f"{session_key}:photo-burst" in adapter._pending_photo_batches


@pytest.mark.asyncio
async def test_old_download_finally_does_not_decrement_new_epoch_counter():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()

    old_event = _photo_event(source)
    old_token = adapter._track_media_download_start(old_event)
    adapter.invalidate_session_ingress(session_key)
    assert old_token not in adapter._media_downloads_in_progress_by_token
    new_event = _photo_event(source)
    new_token = adapter._track_media_download_start(new_event)

    adapter._track_media_download_done(old_token)

    assert old_token != new_token
    assert adapter._media_downloads_in_progress_by_session[session_key] == 1
    assert adapter.has_startup_media_pending(session_key) is True


@pytest.mark.asyncio
async def test_teardown_keeps_epochs_monotonic_against_old_finally():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()
    old_token = adapter._track_media_download_start(_photo_event(source))

    await adapter._cancel_pending_delivery_tasks()
    new_token = adapter._track_media_download_start(_photo_event(source))
    adapter._track_media_download_done(old_token)

    assert old_token != new_token
    assert new_token == (session_key, old_token[1] + 1)
    assert adapter._media_downloads_in_progress_by_session[session_key] == 1


def test_stale_media_group_event_is_not_enqueued():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()
    event = _photo_event(source)
    adapter._stamp_ingress_generation(event)
    adapter.invalidate_session_ingress(session_key)

    asyncio.run(adapter._queue_media_group_event("stale-album", event))

    assert adapter._media_group_events == {}
    assert adapter._media_group_tasks == {}


@pytest.mark.asyncio
async def test_boundary_during_startup_grace_aborts_old_token_merge(monkeypatch):
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._adapter_for_source = lambda _source: adapter
    monkeypatch.setattr(runner, "_startup_media_grace_seconds", lambda: 0.2)

    old_event = _photo_event(source)
    adapter._track_media_download_start(old_event)

    async def cross_boundary():
        await asyncio.sleep(0.02)
        adapter.invalidate_session_ingress(session_key)
        fresh = _photo_event(source)
        adapter.queue_startup_batch_event(session_key, fresh)

    boundary = asyncio.create_task(cross_boundary())
    event = await runner._merge_startup_media_followups(
        MessageEvent(text="ordinary", message_type=MessageType.TEXT, source=source),
        source,
        session_key,
    )
    await boundary

    assert event.message_type is MessageType.TEXT
    assert event.media_urls == []
    assert adapter.pop_startup_media_event(session_key) is not None


@pytest.mark.asyncio
async def test_stop_boundary_invalidates_telegram_ingress():
    source = _source()
    session_key = build_session_key(source)
    adapter = _adapter()
    adapter.queue_startup_batch_event(session_key, _photo_event(source))

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._adapter_for_source = lambda _source: adapter
    runner._peek_session_state = lambda _key: None
    order = []
    real_invalidate = adapter.invalidate_session_ingress

    def invalidate_ingress(key):
        order.append("ingress")
        real_invalidate(key)

    adapter.invalidate_session_ingress = invalidate_ingress

    def invalidate_run(_key, reason=""):
        order.append("run")
        return 1

    runner._invalidate_session_run_generation = invalidate_run
    runner._release_running_agent_state = lambda _key: None
    runner._evict_cached_agent = lambda _key: None
    runner._thread_metadata_for_source = lambda _source: {}

    await runner._interrupt_and_clear_session(
        session_key, source, interrupt_reason="/stop", invalidation_reason="stop"
    )

    assert order[:2] == ["ingress", "run"]
    assert adapter._startup_batch_events == {}
    assert adapter.has_startup_media_pending(session_key) is False
