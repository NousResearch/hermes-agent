"""Real python-telegram-bot dispatch coverage for background locations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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


@pytest.mark.asyncio
async def test_registered_handler_distinguishes_one_time_and_edited_live_updates(
    monkeypatch, tmp_path
):
    """Exercise real PTB Location shapes through the production handler."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"background_locations": True},
        )
    )
    adapter.set_authorization_check(
        lambda _user_id, _chat_type=None, _chat_id=None: True
    )
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter._should_process_message = lambda _message, **_kwargs: True
    adapter._enqueue_text_event = Mock()

    class _RecordingApp:
        def __init__(self):
            self.handlers = []

        def add_handler(self, handler, **kwargs):
            self.handlers.append((kwargs.get("group", 0), handler))

    app = _RecordingApp()
    adapter._register_handlers(app)
    lifecycle_handlers = [
        (group, handler)
        for group, handler in app.handlers
        if getattr(handler, "callback", None)
        == adapter._handle_background_location_lifecycle
    ]
    ordinary_handlers = [
        (group, handler)
        for group, handler in app.handlers
        if getattr(handler, "callback", None)
        == adapter._handle_one_time_location_message
    ]
    assert len(lifecycle_handlers) == 1
    assert len(ordinary_handlers) == 1
    lifecycle_group, lifecycle_handler = lifecycle_handlers[0]
    ordinary_group, ordinary_handler = ordinary_handlers[0]
    assert lifecycle_group == adapter._BACKGROUND_LOCATION_LIFECYCLE_HANDLER_GROUP
    assert lifecycle_group != ordinary_group == 0

    fixed_message = Message(
        message_id=49,
        date=datetime(2026, 7, 17, 11, 59, tzinfo=timezone.utc),
        chat=Chat(id=111, type="private"),
        from_user=User(id=111, first_name="Alice", is_bot=False),
        location=Location(
            latitude=48.8584,
            longitude=2.2945,
            horizontal_accuracy=8.5,
        ),
    )
    fixed_update = Update(update_id=1, message=fixed_message)
    fixed_check_result = ordinary_handler.check_update(fixed_update)
    assert fixed_check_result
    await ordinary_handler.handle_update(
        fixed_update,
        app,
        fixed_check_result,
        SimpleNamespace(),
    )

    adapter._enqueue_text_event.assert_called_once()
    fixed_event = adapter._enqueue_text_event.call_args.args[0]
    assert "[The user shared a one-time location pin.]" in fixed_event.text
    assert fixed_event.message_type == MessageType.TEXT
    assert fixed_event.ephemeral_user_context is None
    assert not adapter._background_location_state_path.exists()

    started_at = datetime.now(timezone.utc)
    message = Message(
        message_id=50,
        date=started_at,
        edit_date=started_at,
        chat=Chat(id=111, type="private"),
        from_user=User(id=111, first_name="Alice", is_bot=False),
        location=Location(
            latitude=51.5015,
            longitude=-0.1419,
            live_period=3600,
            horizontal_accuracy=8.5,
        ),
    )
    update = Update(update_id=2, edited_message=message)

    check_result = lifecycle_handler.check_update(update)
    assert check_result
    await lifecycle_handler.handle_update(
        update,
        app,
        check_result,
        SimpleNamespace(),
    )

    payload = json.loads(
        adapter._background_location_state_path.read_text()
    )
    subject_key = adapter._background_location_subject_key(message)
    assert subject_key is not None
    record = payload["locations"][subject_key]
    assert record["longitude"] == -0.1419
    assert record["source"] == "live_location"
    assert record["is_edited_update"] is True
    adapter._enqueue_text_event.assert_called_once()

    # A native plugin may consume the matching update in PTB's default group.
    # The reserved group must still observe the edited stop independently.
    stop_message = Message(
        message_id=50,
        date=started_at,
        edit_date=datetime.now(timezone.utc),
        chat=Chat(id=111, type="private"),
        from_user=User(id=111, first_name="Alice", is_bot=False),
        location=Location(latitude=51.5015, longitude=-0.1419),
    )
    stop_update = Update(update_id=3, edited_message=stop_message)
    stop_check_result = lifecycle_handler.check_update(stop_update)
    assert stop_check_result
    await lifecycle_handler.handle_update(
        stop_update,
        app,
        stop_check_result,
        SimpleNamespace(),
    )

    stopped_payload = json.loads(adapter._background_location_state_path.read_text())
    stopped_record = stopped_payload["locations"][subject_key]
    assert stopped_record["source"] == "live_location_stop"
    assert "latitude" not in stopped_record
    assert "longitude" not in stopped_record
