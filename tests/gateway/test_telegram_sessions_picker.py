"""Tests for Telegram /sessions inline picker UX."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.telegram import TelegramAdapter
from gateway.session import SessionSource, build_session_key


def _make_source(chat_id="67890", user_id="12345"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )


def _make_event(text="/sessions"):
    return MessageEvent(text=text, source=_make_source())


def _make_runner(session_db, event=None, current_session_id="current_session_001"):
    from gateway.run import GatewayRunner

    event = event or _make_event()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._session_db = session_db
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._session_reasoning_overrides = {}
    runner._update_prompt_pending = {}
    runner._slash_confirm_state = {}
    runner._approval_state = {}

    session_key = build_session_key(event.source)
    current_entry = MagicMock()
    current_entry.session_id = current_session_id
    current_entry.session_key = session_key

    store = MagicMock()
    store.get_or_create_session.return_value = current_entry
    store.switch_session.return_value = current_entry
    store.load_transcript.return_value = []
    runner.session_store = store
    return runner


@pytest.mark.asyncio
async def test_gateway_sessions_command_uses_telegram_picker_when_available(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess_001", "telegram")
    db.set_session_title("sess_001", "Research")
    db.create_session("sess_002", "telegram")
    db.set_session_title("sess_002", "Coding")
    db.create_session("current_session_001", "telegram")

    event = _make_event("/sessions")
    runner = _make_runner(db, event=event)
    adapter = MagicMock()
    adapter.send_sessions_picker = AsyncMock(return_value=SendResult(success=True))
    runner.adapters[Platform.TELEGRAM] = adapter

    result = await runner._handle_sessions_command(event)

    assert result is None
    adapter.send_sessions_picker.assert_awaited_once()
    kwargs = adapter.send_sessions_picker.await_args.kwargs
    assert kwargs["chat_id"] == "67890"
    assert kwargs["sessions"][0]["title"] in {"Research", "Coding"}
    assert kwargs["current_session_id"] == "current_session_001"
    db.close()


@pytest.mark.asyncio
async def test_telegram_sessions_picker_builds_inline_buttons(monkeypatch):
    import gateway.platforms.telegram as tg

    built = []

    class _Button:
        def __init__(self, text, callback_data=None, **kwargs):
            self.text = text
            self.callback_data = callback_data
            built.append((text, callback_data))

    class _Markup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    monkeypatch.setattr(tg, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", _Markup)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()

    async def fake_send(**kwargs):
        return SimpleNamespace(message_id=101)

    adapter._send_message_with_thread_fallback = AsyncMock(side_effect=fake_send)

    result = await adapter.send_sessions_picker(
        chat_id="67890",
        sessions=[
            {"id": "sess_001", "title": "Research", "preview": "notes", "last_active": "2026-06-11T10:00:00"},
            {"id": "sess_002", "title": "Coding", "preview": "fix bug", "last_active": "2026-06-11T09:00:00"},
        ],
        current_session_id="current_session_001",
        on_session_selected=AsyncMock(),
        metadata=None,
    )

    assert result.success is True
    assert ("Research", "sr:0") in built
    assert ("Coding", "sr:1") in built
    assert ("➕ New session", "sn") in built
    assert ("✗ Cancel", "sx") in built
    assert adapter._sessions_picker_state["67890"]["sessions"][0]["id"] == "sess_001"


@pytest.mark.asyncio
async def test_telegram_sessions_callback_resumes_selected_session(monkeypatch):
    import gateway.platforms.telegram as tg

    class _Button:
        def __init__(self, text, callback_data=None, **kwargs):
            self.text = text
            self.callback_data = callback_data

    class _Markup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    monkeypatch.setattr(tg, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", _Markup)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    callback = AsyncMock(return_value="Resumed session **Research**.")
    adapter._sessions_picker_state["67890"] = {
        "sessions": [{"id": "sess_001", "title": "Research"}],
        "current_session_id": "current_session_001",
        "on_session_selected": callback,
    }

    query = AsyncMock()
    query.message = MagicMock()
    query.message.chat_id = 67890
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    await adapter._handle_sessions_picker_callback(query, "sr:0", "67890")

    callback.assert_awaited_once_with("sess_001")
    query.edit_message_text.assert_awaited()
    assert "67890" not in adapter._sessions_picker_state


@pytest.mark.asyncio
async def test_telegram_sessions_callback_new_session_runs_new_callback():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    callback = AsyncMock(return_value="Started a new session.")
    adapter._sessions_picker_state["67890"] = {
        "sessions": [],
        "current_session_id": "current_session_001",
        "on_new_session": callback,
    }

    query = AsyncMock()
    query.message = MagicMock()
    query.message.chat_id = 67890
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    await adapter._handle_sessions_picker_callback(query, "sn", "67890")

    callback.assert_awaited_once()
    query.edit_message_text.assert_awaited()
    assert "67890" not in adapter._sessions_picker_state
