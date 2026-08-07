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
    runner._invalidate_session_run_generation = lambda _key, reason="": 1
    runner._release_running_agent_state = lambda _key: None
    runner._evict_cached_agent = lambda _key: None
    runner._thread_metadata_for_source = lambda _source: {}

    await runner._interrupt_and_clear_session(
        session_key, source, interrupt_reason="/stop", invalidation_reason="stop"
    )

    assert adapter._startup_batch_events == {}
    assert adapter.has_startup_media_pending(session_key) is False
