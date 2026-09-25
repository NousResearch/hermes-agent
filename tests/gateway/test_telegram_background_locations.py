"""RAM-only contract tests for opted-in Telegram live locations."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import merge_ephemeral_context_ref
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter, _apply_yaml_config
from plugins.platforms.telegram.telegram_background_locations import (
    TelegramLiveLocationRef,
)


def _message(
    *,
    latitude=37.7749,
    longitude=-122.4194,
    live_period=None,
    message_id=50,
    date=None,
    chat_id=111,
    user_id=111,
    chat_type="private",
    thread_id=None,
    venue=None,
    business=False,
    sender_chat=None,
):
    location = SimpleNamespace(
        latitude=latitude,
        longitude=longitude,
        live_period=live_period,
    )
    return SimpleNamespace(
        message_id=message_id,
        text=None,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        chat=SimpleNamespace(
            id=chat_id,
            type=chat_type,
            title="Group" if chat_type != "private" else None,
            full_name="Alice",
            is_forum=thread_id is not None,
        ),
        from_user=SimpleNamespace(
            id=user_id,
            username="alice",
            full_name="Alice",
            first_name="Alice",
            is_bot=False,
        ),
        sender_chat=sender_chat,
        reply_to_message=None,
        date=date or datetime.now(timezone.utc),
        edit_date=None,
        business_connection_id="business" if business else None,
        location=location,
        venue=venue,
        forum_topic_created=None,
    )


def _update(message, *, update_id=1, edited=False, business=False):
    return SimpleNamespace(
        update_id=update_id,
        message=None if edited else message,
        effective_message=message,
        edited_message=message if edited and not business else None,
        edited_channel_post=None,
        business_message=message if business and not edited else None,
        edited_business_message=message if business and edited else None,
    )


def _adapter(monkeypatch, tmp_path, **extra) -> TelegramAdapter:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_extra = {
        "background_locations": True,
        "allowed_chats": [],
        "allowed_topics": [],
        "group_allowed_chats": [],
        **extra,
    }
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="123:secret", extra=config_extra)
    )
    adapter._bot = SimpleNamespace(id=999, username="hermes")
    adapter._send_path_degraded = False
    adapter._polling_teardown_started = False
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._should_process_message = lambda *_args, **_kwargs: True
    adapter._should_observe_unmentioned_group_message = lambda *_args, **_kwargs: False
    adapter._observe_unmentioned_group_message = Mock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = Mock()
    return adapter


def test_timedelta_live_period_is_accepted(monkeypatch, tmp_path):
    """PTB 22.2+ can expose Location.live_period as datetime.timedelta."""
    adapter = _adapter(monkeypatch, tmp_path)
    location = SimpleNamespace(live_period=timedelta(seconds=3600))

    assert adapter._active_live_location_period(location) == 3600


async def _record_live(adapter: TelegramAdapter, message=None, *, update_id=1):
    live = message or _message(live_period=3600)
    await adapter._record_background_location(_update(live, update_id=update_id), live)
    return live


async def _reference(
    adapter: TelegramAdapter, message, *, text="where am I?", message_id="question-1",
) -> MessageEvent:
    # A foreground text update has the same sender/chat identity as the live share,
    # but Telegram does not copy the earlier location object onto that message.
    foreground = SimpleNamespace(**vars(message))
    foreground.message_id = message_id
    foreground.text = text
    foreground.location = None
    source = adapter.build_source(
        chat_id=str(message.chat.id),
        chat_type="dm" if message.chat.type == "private" else "group",
        user_id=str(message.from_user.id),
        thread_id=adapter._effective_message_thread_id(foreground),
        message_id=message_id,
    )
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=foreground,
        message_id=message_id,
    )
    return await adapter._attach_background_location_context(event, foreground)


@pytest.mark.asyncio
async def test_live_update_is_silent_ram_only_and_event_is_coordinate_free(
    monkeypatch, tmp_path,
):
    adapter = _adapter(monkeypatch, tmp_path)
    live = _message(live_period=3600)

    await adapter._handle_location_message(_update(live), SimpleNamespace())

    adapter.handle_message.assert_not_awaited()
    adapter._enqueue_text_event.assert_not_called()
    event = await _reference(adapter, live)
    assert isinstance(event.ephemeral_context_ref, TelegramLiveLocationRef)
    assert "37.7749" not in repr(event)
    assert "37.7749" not in json.dumps(vars(event.ephemeral_context_ref))
    context = adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)
    assert context and "Latitude: 37.7749" in context
    assert not list(tmp_path.rglob("*telegram_background_locations*"))


@pytest.mark.asyncio
async def test_snapshot_is_immutable_and_stop_affects_only_future_resolution(
    monkeypatch, tmp_path,
):
    adapter = _adapter(monkeypatch, tmp_path)
    live = await _record_live(adapter, _message(live_period=3600, message_id=51))
    event = await _reference(adapter, live)
    first_snapshot = adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)

    moved = _message(latitude=40.0, longitude=-73.0, live_period=3600)
    moved.edit_date = datetime.now(timezone.utc) + timedelta(seconds=1)
    await adapter._record_background_location(
        _update(moved, update_id=2, edited=True), moved
    )
    second_snapshot = adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)
    assert first_snapshot and "Latitude: 37.7749" in first_snapshot
    assert second_snapshot and "Latitude: 40.0" in second_snapshot
    assert "Latitude: 40.0" not in first_snapshot

    stop = _message(live_period=None)
    stop.edit_date = datetime.now(timezone.utc) + timedelta(seconds=2)
    await adapter._handle_background_location_lifecycle(
        _update(stop, update_id=3, edited=True), SimpleNamespace()
    )
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event) is None
    assert "Latitude: 37.7749" in first_snapshot


@pytest.mark.asyncio
async def test_stale_and_duplicate_edits_cannot_overwrite_or_resurrect(
    monkeypatch, tmp_path,
):
    adapter = _adapter(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    live = _message(live_period=3600, date=now)
    await _record_live(adapter, live, update_id=10)
    subject = adapter._background_location_subject_key(live)

    duplicate = _message(latitude=9.0, longitude=9.0, live_period=3600, date=now)
    await adapter._record_background_location(
        _update(duplicate, update_id=10, edited=True), duplicate
    )
    assert adapter._background_location_records[subject]["latitude"] == 37.7749

    stop = _message(live_period=None, date=now)
    stop.edit_date = now + timedelta(seconds=2)
    await adapter._record_background_location(
        _update(stop, update_id=12, edited=True), stop
    )
    delayed = _message(latitude=8.0, longitude=8.0, live_period=3600, date=now)
    delayed.edit_date = now + timedelta(seconds=1)
    await adapter._record_background_location(
        _update(delayed, update_id=11, edited=True), delayed
    )
    assert not adapter._background_location_records


@pytest.mark.asyncio
async def test_stop_purges_even_when_profile_resolution_fails(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)
    live = _message(
        live_period=adapter._BACKGROUND_LOCATION_INDEFINITE_LIVE_PERIOD,
        message_id=51,
    )
    await _record_live(adapter, live)
    assert adapter._background_location_records

    adapter.gateway_runner = SimpleNamespace(
        _profile_name_for_source=Mock(side_effect=RuntimeError("resolver unavailable"))
    )
    stop = _message(live_period=None, message_id=51)
    stop.edit_date = datetime.now(timezone.utc) + timedelta(seconds=1)

    assert await adapter._record_background_location(
        _update(stop, update_id=2, edited=True), stop
    )
    assert not adapter._background_location_records


@pytest.mark.asyncio
async def test_expiry_polling_fence_and_disconnect_purge_records(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)
    expired = _message(
        live_period=30, date=datetime.now(timezone.utc) - timedelta(minutes=2)
    )
    await _record_live(adapter, expired)
    event = await _reference(adapter, expired)
    assert event.ephemeral_context_ref is None
    assert not adapter._background_location_records

    # A fresh Telegram live-share has its own lifecycle message id.
    live = await _record_live(adapter, _message(live_period=3600, message_id=51))
    event = await _reference(adapter, live)
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)
    adapter._fence_polling()
    assert not adapter._background_location_records
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event) is None

    await _record_live(adapter)
    adapter._app = None
    adapter._release_platform_lock = Mock()
    await adapter.disconnect()
    assert not adapter._background_location_records


@pytest.mark.asyncio
async def test_finite_share_expires_without_another_inbound_event(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)
    live = await _record_live(adapter, _message(live_period=1, message_id=51))
    event = await _reference(adapter, live)

    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)
    assert adapter._background_location_expiry_handle is not None

    await asyncio.sleep(1.1)

    assert not adapter._background_location_records
    assert adapter._background_location_expiry_handle is None
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event) is None


@pytest.mark.asyncio
async def test_polling_generation_rejects_backlog_and_invalidates_old_refs(
    monkeypatch, tmp_path,
):
    adapter = _adapter(monkeypatch, tmp_path)
    original = await _record_live(adapter)
    old_event = await _reference(adapter, original)
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(old_event)

    generation, _ = adapter._begin_polling_generation()
    adapter._send_path_degraded = False
    backlog = _message(
        latitude=9.0,
        longitude=9.0,
        live_period=3600,
        message_id=51,
    )
    adapter._observe_background_location_poll_result(
        generation, {"result": [{"update_id": 10}]}
    )
    await adapter._handle_location_message(
        _update(backlog, update_id=10), SimpleNamespace()
    )
    assert not adapter._background_location_records

    # Empty getUpdates proves the restart backlog was drained, but an update that
    # arrived before that proof stays ineligible even if its handler runs late.
    adapter._observe_background_location_poll_result(generation, {"result": []})
    await adapter._handle_location_message(
        _update(backlog, update_id=10), SimpleNamespace()
    )
    assert not adapter._background_location_records

    fresh = _message(
        latitude=40.0,
        longitude=-73.0,
        live_period=3600,
        message_id=52,
        date=datetime.now(timezone.utc),
    )
    adapter._observe_background_location_poll_result(
        generation, {"result": [{"update_id": 11}]}
    )
    await adapter._handle_location_message(
        _update(fresh, update_id=11), SimpleNamespace()
    )
    assert adapter._background_location_records
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(old_event) is None
    fresh_event = await _reference(adapter, fresh, message_id="question-2")
    assert "Latitude: 40.0" in (
        adapter._resolve_ephemeral_user_context_for_dispatch_sync(fresh_event) or ""
    )


@pytest.mark.asyncio
async def test_authorization_routes_and_unsupported_identities_fail_closed(
    monkeypatch, tmp_path,
):
    adapter = _adapter(monkeypatch, tmp_path)
    live = _message(live_period=3600)
    adapter.set_authorization_check(lambda *_args, **_kwargs: False)
    await adapter._handle_location_message(_update(live), SimpleNamespace())
    assert not adapter._background_location_records

    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._should_accept_background_location = lambda _message: False
    await adapter._handle_location_message(_update(live, update_id=2), SimpleNamespace())
    assert not adapter._background_location_records

    business = _message(live_period=3600, business=True)
    await adapter._handle_location_message(
        _update(business, update_id=3, business=True), SimpleNamespace()
    )
    assert not adapter._background_location_records

    sender_chat = _message(
        live_period=3600, sender_chat=SimpleNamespace(id=-100, title="Anonymous")
    )
    await adapter._record_background_location(_update(sender_chat), sender_chat)
    assert not adapter._background_location_records


@pytest.mark.asyncio
async def test_explicit_adapter_allowlist_cannot_be_bypassed_by_runner_auth(
    monkeypatch, tmp_path,
):
    adapter = _adapter(monkeypatch, tmp_path, allow_from=["222"])
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)

    live = _message(live_period=3600, user_id=111)
    await adapter._handle_location_message(_update(live), SimpleNamespace())

    assert not adapter._background_location_records
    adapter.handle_message.assert_not_awaited()
    adapter._enqueue_text_event.assert_not_called()


@pytest.mark.asyncio
async def test_shared_session_and_copied_reference_never_resolve(monkeypatch, tmp_path):
    adapter = _adapter(
        monkeypatch,
        tmp_path,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    live = _message(
        live_period=3600, chat_id=-100, chat_type="supergroup", thread_id=7
    )
    await _record_live(adapter, live)
    shared_event = await _reference(adapter, live)
    assert shared_event.ephemeral_context_ref is None

    dm = await _record_live(adapter, _message(live_period=3600, message_id=51), update_id=2)
    owner_event = await _reference(adapter, dm)
    assert owner_event.ephemeral_context_ref is not None
    copied = replace(
        owner_event,
        source=adapter.build_source(
            chat_id="111", chat_type="dm", user_id="222", message_id="question-2"
        ),
        message_id="question-2",
    )
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(copied) is None

    other_adapter = _adapter(monkeypatch, tmp_path)
    assert other_adapter._resolve_ephemeral_user_context_for_dispatch_sync(owner_event) is None


@pytest.mark.asyncio
async def test_private_chat_topics_are_isolated(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)
    topic_seven = await _record_live(
        adapter,
        _message(live_period=3600, chat_type="private", thread_id=7),
    )
    same_topic = await _reference(adapter, topic_seven)
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(same_topic)

    other_topic_message = _message(
        live_period=None,
        chat_type="private",
        thread_id=8,
        message_id=52,
    )
    other_topic = await _reference(adapter, other_topic_message)
    assert other_topic.ephemeral_context_ref is None


@pytest.mark.asyncio
async def test_forum_general_topic_uses_the_same_canonical_subject_for_live_and_text(
    monkeypatch, tmp_path,
):
    adapter = _adapter(
        monkeypatch,
        tmp_path,
        group_sessions_per_user=True,
        thread_sessions_per_user=True,
    )
    general = _message(
        live_period=3600,
        chat_id=-100,
        chat_type="supergroup",
        thread_id=None,
    )
    general.chat.is_forum = True

    await _record_live(adapter, general)
    event = await _reference(adapter, general)

    assert event.source.thread_id == adapter._GENERAL_TOPIC_THREAD_ID
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event)


def test_yaml_config_preserves_explicit_background_location_boolean(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BACKGROUND_LOCATIONS", raising=False)

    assert _apply_yaml_config({}, {"background_locations": True}) == {
        "background_locations": True
    }
    assert _apply_yaml_config({}, {"background_locations": False}) == {
        "background_locations": False
    }


def test_explicit_empty_scope_allowlists_beat_legacy_nested_values(
    monkeypatch, tmp_path,
):
    """A stale legacy extra cannot re-authorize live telemetry collection."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  enabled: true\n"
        "  background_locations: true\n"
        "  group_allow_from: []\n"
        "  allowed_chats: ['-100']\n"
        "  allowed_topics: []\n"
        "  group_allowed_chats: []\n"
        "  extra:\n"
        "    group_allow_from: ['attacker']\n"
        "    allowed_chats: ['-999']\n"
        "    allowed_topics: [99]\n"
        "    group_allowed_chats: ['-999']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test-token")
    for name in (
        "TELEGRAM_ALLOWED_CHATS",
        "TELEGRAM_ALLOWED_TOPICS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_gateway_config()
    telegram = config.platforms[Platform.TELEGRAM]

    assert telegram.extra["group_allow_from"] == []
    assert telegram.extra["allowed_chats"] == ["-100"]
    assert telegram.extra["allowed_topics"] == []
    assert telegram.extra["group_allowed_chats"] == []
    assert "attacker" not in json.dumps(telegram.extra)

    adapter = TelegramAdapter(telegram)
    attacker_share = _message(
        live_period=3600,
        chat_id=-100,
        user_id="attacker",
        chat_type="supergroup",
    )
    assert adapter._is_background_location_authorized(attacker_share) is False


@pytest.mark.asyncio
async def test_gateway_resolves_once_only_for_an_identified_foreground_turn(
    monkeypatch, tmp_path,
):
    from gateway.run import GatewayRunner

    adapter = _adapter(monkeypatch, tmp_path)
    live = await _record_live(adapter)
    event = await _reference(adapter, live)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: adapter.config},
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}

    snapshot = runner._resolve_event_volatile_user_context(event)
    assert snapshot and "Latitude: 37.7749" in snapshot
    assert runner._resolve_event_volatile_user_context(replace(event, internal=True)) is None
    assert runner._resolve_event_volatile_user_context(replace(event, message_id=None)) is None
    assert runner._resolve_event_volatile_user_context(
        replace(event, media_urls=["/tmp/photo.jpg"])
    ) is None

    shared_source = replace(
        event.source,
        chat_id="-100",
        chat_type="group",
        thread_id="7",
    )
    assert runner._resolve_event_volatile_user_context(
        replace(event, source=shared_source)
    ) is None


@pytest.mark.asyncio
async def test_fixed_pin_is_ordinary_text_and_blocks_ambient_context_in_both_orders(
    monkeypatch, tmp_path,
):
    adapter = _adapter(monkeypatch, tmp_path)
    live = await _record_live(adapter)
    ambient = await _reference(adapter, live)
    assert ambient.ephemeral_context_ref is not None

    pin = _message(latitude=10.0, longitude=20.0, live_period=None, message_id=60)
    await adapter._handle_location_message(_update(pin, update_id=3), SimpleNamespace())
    pin_event = adapter._enqueue_text_event.call_args.args[0]
    assert pin_event.message_type == MessageType.TEXT
    assert "one-time location pin" in pin_event.text
    assert pin_event.ephemeral_context_ref is None
    assert pin_event._ephemeral_context_blocked is True

    first = replace(ambient)
    merge_ephemeral_context_ref(first, pin_event)
    assert first.ephemeral_context_ref is None
    assert first._ephemeral_context_blocked is True

    second = replace(pin_event)
    merge_ephemeral_context_ref(second, ambient)
    assert second.ephemeral_context_ref is None
    assert second._ephemeral_context_blocked is True


@pytest.mark.asyncio
async def test_runtime_disabled_live_updates_stay_silent_but_fixed_pins_dispatch(
    monkeypatch, tmp_path,
):
    adapter = _adapter(monkeypatch, tmp_path)
    adapter._background_locations_enabled = False  # webhook-style fail-closed state
    live = _message(live_period=3600)
    await adapter._handle_location_message(_update(live), SimpleNamespace())
    adapter.handle_message.assert_not_awaited()
    adapter._enqueue_text_event.assert_not_called()

    pin = _message(live_period=None, message_id=61)
    await adapter._handle_location_message(_update(pin, update_id=2), SimpleNamespace())
    adapter._enqueue_text_event.assert_called_once()
