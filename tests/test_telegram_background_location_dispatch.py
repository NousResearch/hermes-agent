"""Real python-telegram-bot dispatch coverage for background locations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "telegram", reason="python-telegram-bot is an optional messaging dependency"
)

from telegram import Chat, Location, Message, Update, User

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_registered_location_handler_accepts_edited_updates(
    monkeypatch, tmp_path
):
    """Exercise the production MessageHandler and PTB edited-update filter."""
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

    class _RecordingApp:
        def __init__(self):
            self.handlers = []

        def add_handler(self, handler):
            self.handlers.append(handler)

    app = _RecordingApp()
    adapter._register_update_handlers(app)
    location_handler = next(
        handler
        for handler in app.handlers
        if getattr(getattr(handler, "callback", None), "__name__", "")
        == "_handle_location_message"
    )
    message = Message(
        message_id=50,
        date=datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
        edit_date=datetime(2026, 7, 17, 12, 1, tzinfo=timezone.utc),
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

    check_result = location_handler.check_update(update)
    assert check_result
    await location_handler.handle_update(
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
