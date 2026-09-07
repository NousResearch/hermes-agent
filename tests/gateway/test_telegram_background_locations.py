"""RAM-only contract tests for opted-in Telegram live locations."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_background_locations import TelegramLiveLocationRef


def _message(*, latitude=37.7749, longitude=-122.4194, live_period=None, message_id=50, date=None):
    return SimpleNamespace(
        message_id=message_id, text=None, caption=None, entities=[], caption_entities=[],
        message_thread_id=None, is_topic_message=False,
        chat=SimpleNamespace(id=111, type="private", title=None, full_name="Alice", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Alice", first_name="Alice", is_bot=False),
        sender_chat=None, reply_to_message=None, date=date or datetime.now(timezone.utc), edit_date=None,
        business_connection_id=None,
        location=SimpleNamespace(latitude=latitude, longitude=longitude, live_period=live_period),
        venue=None, forum_topic_created=None,
    )


def _update(message, *, update_id=1, edited=False):
    return SimpleNamespace(
        update_id=update_id, message=None if edited else message, effective_message=message,
        edited_message=message if edited else None, edited_channel_post=None,
        business_message=None, edited_business_message=None,
    )


def _adapter(monkeypatch, tmp_path) -> TelegramAdapter:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:secret", extra={"background_locations": True, "allowed_chats": [], "allowed_topics": [], "group_allowed_chats": []}))
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._should_process_message = lambda *_args, **_kwargs: True
    adapter._should_observe_unmentioned_group_message = lambda *_args, **_kwargs: False
    adapter._observe_unmentioned_group_message = Mock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = Mock()
    return adapter


async def _reference(adapter: TelegramAdapter, message) -> MessageEvent:
    source = adapter._source_from_message_for_auth(message)
    event = MessageEvent(text="where am I?", source=source)
    return await adapter._attach_background_location_context(event, message)


@pytest.mark.asyncio
async def test_live_location_is_ram_only_and_event_contains_only_a_reference(monkeypatch, tmp_path):
    adapter, live = _adapter(monkeypatch, tmp_path), _message(live_period=3600)
    await adapter._handle_location_message(_update(live), SimpleNamespace())

    adapter.handle_message.assert_not_awaited()
    event = await _reference(adapter, live)
    assert isinstance(event.ephemeral_context_ref, TelegramLiveLocationRef)
    assert event.ephemeral_user_context is None
    assert "37.7749" not in repr(event.ephemeral_context_ref)
    context = adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)
    assert context and "Latitude: 37.7749" in context
    assert not list(tmp_path.rglob("*telegram_background_locations*"))


@pytest.mark.asyncio
async def test_foreground_snapshot_is_fixed_when_live_location_moves(monkeypatch, tmp_path):
    adapter, live = _adapter(monkeypatch, tmp_path), _message(live_period=3600)
    await adapter._persist_background_location(_update(live), live)
    event = await _reference(adapter, live)
    snapshot = adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)
    moved = _message(latitude=40.0, longitude=-73.0, live_period=3600)
    moved.edit_date = datetime.now(timezone.utc)
    await adapter._persist_background_location(_update(moved, update_id=2, edited=True), moved)

    assert "Latitude: 37.7749" in snapshot
    assert "Latitude: 40.0" in adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)


@pytest.mark.asyncio
async def test_stop_removes_future_turn_context_but_not_an_existing_snapshot(monkeypatch, tmp_path):
    adapter, live = _adapter(monkeypatch, tmp_path), _message(live_period=3600)
    await adapter._persist_background_location(_update(live), live)
    event = await _reference(adapter, live)
    snapshot = adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)
    stop = _message(live_period=None)
    await adapter._persist_background_location(_update(stop, update_id=2, edited=True), stop)

    assert snapshot and "Latitude: 37.7749" in snapshot
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event) is None


@pytest.mark.asyncio
async def test_delayed_edit_cannot_resurrect_a_stopped_share(monkeypatch, tmp_path):
    adapter, live = _adapter(monkeypatch, tmp_path), _message(live_period=3600)
    await adapter._persist_background_location(_update(live, update_id=10), live)
    stop = _message(live_period=None)
    await adapter._persist_background_location(_update(stop, update_id=11, edited=True), stop)
    delayed = _message(latitude=40.0, longitude=-73.0, live_period=3600)
    await adapter._persist_background_location(_update(delayed, update_id=9, edited=True), delayed)
    assert not adapter._background_location_records


@pytest.mark.asyncio
async def test_real_edited_handler_path_stops_the_live_share(monkeypatch, tmp_path):
    adapter, live = _adapter(monkeypatch, tmp_path), _message(live_period=3600)
    await adapter._handle_location_message(_update(live), SimpleNamespace())
    event = await _reference(adapter, live)
    stop = _message(live_period=None)
    await adapter._handle_background_location_lifecycle(_update(stop, update_id=2, edited=True), SimpleNamespace())
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event) is None


@pytest.mark.asyncio
async def test_one_time_pin_remains_an_ordinary_user_turn(monkeypatch, tmp_path):
    adapter, pin = _adapter(monkeypatch, tmp_path), _message(live_period=None)
    await adapter._handle_location_message(_update(pin), SimpleNamespace())
    adapter._enqueue_text_event.assert_called_once()
    event = adapter._enqueue_text_event.call_args.args[0]
    assert "one-time location pin" in event.text
    assert event.ephemeral_context_ref is None
