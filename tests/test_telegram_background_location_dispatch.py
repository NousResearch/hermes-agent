"""Real python-telegram-bot dispatch coverage for RAM-only live locations."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

telegram_module = pytest.importorskip(
    "telegram", reason="python-telegram-bot is an optional messaging dependency"
)
if not isinstance(getattr(telegram_module, "__file__", None), str):
    pytest.skip(
        "requires the real python-telegram-bot SDK, not the gateway test mock",
        allow_module_level=True,
    )

from telegram import Chat, Location, Message, Update, User

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_background_locations import (
    TelegramLiveLocationRef,
)


@pytest.mark.asyncio
async def test_registered_handlers_keep_live_updates_silent_and_ram_only(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="123:test-token",
            extra={"background_locations": True},
        )
    )
    adapter._send_path_degraded = False
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter._should_process_message = lambda _message, **_kwargs: True
    adapter._enqueue_text_event = Mock()

    class _RecordingApp:
        def __init__(self):
            self.handlers = []

        def add_handler(self, handler, **kwargs):
            self.handlers.append((kwargs.get("group", 0), handler))

    app = _RecordingApp()
    adapter._register_background_location_lifecycle_handler(app)
    adapter._register_handlers(app)

    def _handler(callback):
        matches = [
            (group, handler)
            for group, handler in app.handlers
            if getattr(handler, "callback", None) == callback
        ]
        assert len(matches) == 1
        return matches[0]

    lifecycle_group, lifecycle = _handler(
        adapter._handle_background_location_lifecycle
    )
    ordinary_group, ordinary = _handler(adapter._handle_one_time_location_message)
    _text_group, text_handler = _handler(adapter._handle_text_message)
    assert lifecycle_group == adapter._BACKGROUND_LOCATION_LIFECYCLE_HANDLER_GROUP
    assert lifecycle_group != ordinary_group == 0

    chat = Chat(id=111, type="private")
    user = User(id=111, first_name="Alice", is_bot=False)
    fixed = Message(
        message_id=49,
        date=datetime(2026, 7, 17, 11, 59, tzinfo=timezone.utc),
        chat=chat,
        from_user=user,
        location=Location(latitude=48.8584, longitude=2.2945),
    )
    fixed_update = Update(update_id=1, message=fixed)
    checked = ordinary.check_update(fixed_update)
    assert checked
    await ordinary.handle_update(fixed_update, app, checked, SimpleNamespace())
    fixed_event = adapter._enqueue_text_event.call_args.args[0]
    assert fixed_event.message_type == MessageType.TEXT
    assert fixed_event._ephemeral_context_blocked is True
    assert "one-time location pin" in fixed_event.text

    started_at = datetime.now(timezone.utc)
    live = Message(
        message_id=50,
        date=started_at,
        edit_date=started_at,
        chat=chat,
        from_user=user,
        location=Location(
            latitude=51.5015,
            longitude=-0.1419,
            live_period=timedelta(seconds=3600),
        ),
    )
    live_update = Update(update_id=2, edited_message=live)
    checked = lifecycle.check_update(live_update)
    assert checked
    await lifecycle.handle_update(live_update, app, checked, SimpleNamespace())
    subject = adapter._background_location_subject_key(live)
    assert adapter._background_location_records[subject]["longitude"] == -0.1419

    adapter._enqueue_text_event.reset_mock()
    question = Message(
        message_id=51,
        date=datetime.now(timezone.utc),
        chat=chat,
        from_user=user,
        text="what is nearby?",
    )
    question_update = Update(update_id=3, message=question)
    checked = text_handler.check_update(question_update)
    assert checked
    await text_handler.handle_update(
        question_update, app, checked, SimpleNamespace()
    )
    event = adapter._enqueue_text_event.call_args.args[0]
    assert isinstance(event.ephemeral_context_ref, TelegramLiveLocationRef)
    assert "51.5015" not in json.dumps(vars(event.ephemeral_context_ref))
    assert "Latitude: 51.5015" in (
        adapter._resolve_ephemeral_user_context_for_dispatch_sync(event) or ""
    )

    stop = Message(
        message_id=50,
        date=started_at,
        edit_date=datetime.now(timezone.utc),
        chat=chat,
        from_user=user,
        location=Location(latitude=51.5015, longitude=-0.1419),
    )
    stop_update = Update(update_id=4, edited_message=stop)
    checked = lifecycle.check_update(stop_update)
    assert checked
    await lifecycle.handle_update(stop_update, app, checked, SimpleNamespace())
    assert adapter._resolve_ephemeral_user_context_for_dispatch_sync(event) is None
    assert not adapter._background_location_records
    assert not list(tmp_path.rglob("*telegram_background_locations*"))
