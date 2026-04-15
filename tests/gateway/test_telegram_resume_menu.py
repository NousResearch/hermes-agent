"""Tests for /resume inline keyboard menu on Telegram."""

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    class _FakeInlineKeyboardButton:
        def __init__(self, text, callback_data=None):
            self.text = text
            self.callback_data = callback_data

    class _FakeInlineKeyboardMarkup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    mod = MagicMock()
    mod.InlineKeyboardButton = _FakeInlineKeyboardButton
    mod.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter():
    import gateway.platforms.telegram as telegram_mod

    class _FakeInlineKeyboardButton:
        def __init__(self, text, callback_data=None):
            self.text = text
            self.callback_data = callback_data

    class _FakeInlineKeyboardMarkup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    telegram_mod.InlineKeyboardButton = _FakeInlineKeyboardButton
    telegram_mod.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 99
    adapter._bot.send_message = AsyncMock(return_value=mock_msg)
    return adapter


class TestRelativeTime:
    def test_just_now(self):
        ts = datetime.now(timezone.utc).isoformat()
        assert TelegramAdapter._relative_time(ts) == "just now"

    def test_hours_ago(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        assert TelegramAdapter._relative_time(ts) == "3h ago"

    def test_days_ago(self):
        ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        assert TelegramAdapter._relative_time(ts) == "5d ago"

    def test_empty_string(self):
        assert TelegramAdapter._relative_time("") == ""

    def test_none(self):
        assert TelegramAdapter._relative_time(None) == ""


class TestBuildResumeKeyboard:
    def test_session_buttons_with_relative_time(self):
        adapter = _make_adapter()
        sessions = [
            {
                "id": "s1",
                "title": "My Project",
                "last_active": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            },
            {
                "id": "s2",
                "title": "Debug Session",
                "last_active": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            },
        ]

        keyboard = adapter._build_resume_keyboard(sessions)

        assert len(keyboard.inline_keyboard) == 3  # 2 sessions + close
        button = keyboard.inline_keyboard[0][0]
        assert "My Project" in button.text
        assert "2h ago" in button.text
        assert button.callback_data == "rs:0"

    def test_close_button_present(self):
        adapter = _make_adapter()
        keyboard = adapter._build_resume_keyboard([{"id": "s1", "title": "Test"}])
        assert any(button.callback_data == "rx" for button in keyboard.inline_keyboard[-1])

    def test_back_button_present_for_chain_menu(self):
        adapter = _make_adapter()
        keyboard = adapter._build_resume_keyboard([{"id": "s1", "title": "Test"}], show_back=True)
        assert any(button.callback_data == "rb" for button in keyboard.inline_keyboard[-2])

    def test_empty_preview_falls_back_to_session_id_label(self):
        adapter = _make_adapter()
        keyboard = adapter._build_resume_keyboard([{"id": "s1", "title": None, "preview": ""}])
        assert keyboard.inline_keyboard[0][0].text == "s1"


class TestSendResumeMenu:
    @pytest.mark.asyncio
    async def test_sends_menu_and_stores_state(self):
        adapter = _make_adapter()
        sessions = [{"id": "s1", "title": "Test", "last_active": "2026-01-01T00:00:00+00:00"}]
        on_resume = AsyncMock(return_value="Resumed!")

        result = await adapter.send_resume_menu(
            chat_id="12345",
            sessions=sessions,
            session_key="agent:main:telegram:dm",
            on_session_resumed=on_resume,
        )

        assert result.success
        state = adapter._resume_menu_state["12345"]
        assert state["sessions"] == sessions
        assert state["session_key"] == "agent:main:telegram:dm"
        assert state["on_session_resumed"] is on_resume
        assert state["msg_id"] == 99

    @pytest.mark.asyncio
    async def test_retries_without_thread_when_thread_not_found(self):
        adapter = _make_adapter()
        sessions = [{"id": "s1", "title": "Test", "last_active": "2026-01-01T00:00:00+00:00"}]
        call_log = []

        class FakeBadRequest(Exception):
            pass

        async def mock_send_message(**kwargs):
            call_log.append(dict(kwargs))
            if kwargs.get("message_thread_id") is not None:
                raise FakeBadRequest("Message thread not found")
            return SimpleNamespace(message_id=99)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_resume_menu(
            chat_id="12345",
            sessions=sessions,
            session_key="agent:main:telegram:dm",
            on_session_resumed=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success
        assert len(call_log) == 2
        assert call_log[0]["message_thread_id"] == 99999
        assert "message_thread_id" not in call_log[1] or call_log[1]["message_thread_id"] is None


class TestResumeCallback:
    @pytest.mark.asyncio
    async def test_select_session_invokes_callback(self):
        adapter = _make_adapter()
        sessions = [
            {"id": "s1", "title": "Project Alpha", "last_active": "2026-01-01T00:00:00+00:00"},
            {"id": "s2", "title": "Project Beta", "last_active": "2026-01-02T00:00:00+00:00"},
        ]
        on_resume = AsyncMock(return_value="Resumed!")
        adapter._resume_menu_state["12345"] = {
            "sessions": sessions,
            "msg_id": 99,
            "session_key": "k",
            "on_session_resumed": on_resume,
        }

        query = MagicMock()
        query.message = SimpleNamespace(message_id=99, message_thread_id=None)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._handle_resume_callback(query, "rs:0", "12345")

        on_resume.assert_called_once_with("s1", "Project Alpha")
        assert "12345" not in adapter._resume_menu_state
        query.edit_message_text.assert_called_once()
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_empty_untitled_session_uses_id_as_callback_label(self):
        adapter = _make_adapter()
        sessions = [{"id": "sess_empty", "title": None, "preview": ""}]
        on_resume = AsyncMock(return_value="Resumed!")
        adapter._resume_menu_state["12345"] = {
            "sessions": sessions,
            "msg_id": 99,
            "session_key": "k",
            "on_session_resumed": on_resume,
        }

        query = MagicMock()
        query.message = SimpleNamespace(message_id=99, message_thread_id=None)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._handle_resume_callback(query, "rs:0", "12345")

        on_resume.assert_called_once_with("sess_empty", "sess_empty")
        kwargs = query.edit_message_text.call_args.kwargs
        assert kwargs["text"] == "↻ Resuming **sess_empty**…"
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_root_with_resume_chain_opens_nested_menu(self):
        adapter = _make_adapter()
        roots = [{
            "id": "root",
            "title": "Project Alpha",
            "last_active": "2026-01-02T00:00:00+00:00",
            "resume_chain": [
                {"id": "root", "title": "Project Alpha", "last_active": "2026-01-01T00:00:00+00:00"},
                {"id": "child", "title": "Project Alpha #2", "last_active": "2026-01-02T00:00:00+00:00"},
            ],
        }]
        on_resume = AsyncMock(return_value="Resumed!")
        adapter._resume_menu_state["12345"] = {
            "sessions": roots,
            "root_sessions": roots,
            "mode": "root",
            "msg_id": 99,
            "session_key": "k",
            "on_session_resumed": on_resume,
        }

        query = MagicMock()
        query.message = SimpleNamespace(message_id=99, message_thread_id=None)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._handle_resume_callback(query, "rs:0", "12345")

        on_resume.assert_not_called()
        state = adapter._resume_menu_state["12345"]
        assert state["mode"] == "chain"
        assert [s["id"] for s in state["sessions"]] == ["root", "child"]
        kwargs = query.edit_message_text.call_args.kwargs
        assert kwargs["text"] == adapter._render_resume_menu_text("chain")
        assert kwargs["reply_markup"].inline_keyboard[-2][0].callback_data == "rb"
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_back_restores_root_resume_menu(self):
        adapter = _make_adapter()
        roots = [{"id": "root", "title": "Project Alpha", "last_active": "2026-01-02T00:00:00+00:00"}]
        chain = [
            {"id": "root", "title": "Project Alpha", "last_active": "2026-01-01T00:00:00+00:00"},
            {"id": "child", "title": "Project Alpha #2", "last_active": "2026-01-02T00:00:00+00:00"},
        ]
        adapter._resume_menu_state["12345"] = {
            "sessions": chain,
            "root_sessions": roots,
            "mode": "chain",
            "msg_id": 99,
            "session_key": "k",
            "on_session_resumed": AsyncMock(),
        }

        query = MagicMock()
        query.message = SimpleNamespace(message_id=99, message_thread_id=None)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._handle_resume_callback(query, "rb", "12345")

        state = adapter._resume_menu_state["12345"]
        assert state["mode"] == "root"
        assert state["sessions"] == roots
        kwargs = query.edit_message_text.call_args.kwargs
        assert kwargs["text"] == adapter._render_resume_menu_text("root")
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_router_routes_back_button_to_resume_handler(self):
        adapter = _make_adapter()
        query = MagicMock()
        query.data = "rb"
        query.message = SimpleNamespace(chat_id=12345)
        update = SimpleNamespace(callback_query=query)
        adapter._handle_resume_callback = AsyncMock()

        await adapter._handle_callback_query(update, None)

        adapter._handle_resume_callback.assert_awaited_once_with(query, "rb", "12345")

    @pytest.mark.asyncio
    async def test_resume_followup_retries_without_thread_when_needed(self):
        adapter = _make_adapter()
        sessions = [{"id": "s1", "title": "Project Alpha", "last_active": "2026-01-01T00:00:00+00:00"}]
        on_resume = AsyncMock(return_value="Resumed!")
        adapter._resume_menu_state["12345"] = {
            "sessions": sessions,
            "msg_id": 99,
            "session_key": "k",
            "on_session_resumed": on_resume,
        }

        query = MagicMock()
        query.message = SimpleNamespace(message_id=99, message_thread_id=777)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        call_log = []

        class FakeBadRequest(Exception):
            pass

        async def mock_send_message(**kwargs):
            call_log.append(dict(kwargs))
            if kwargs.get("message_thread_id") is not None:
                raise FakeBadRequest("Message thread not found")
            return SimpleNamespace(message_id=123)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        await adapter._handle_resume_callback(query, "rs:0", "12345")

        assert len(call_log) == 2
        assert call_log[0]["message_thread_id"] == 777
        assert "message_thread_id" not in call_log[1] or call_log[1]["message_thread_id"] is None

    @pytest.mark.asyncio
    async def test_close_dismisses_menu(self):
        adapter = _make_adapter()
        adapter._resume_menu_state["12345"] = {
            "sessions": [],
            "msg_id": 99,
            "session_key": "k",
            "on_session_resumed": AsyncMock(),
        }
        query = MagicMock()
        query.message = SimpleNamespace(message_id=99)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._handle_resume_callback(query, "rx", "12345")

        assert "12345" not in adapter._resume_menu_state
        query.edit_message_text.assert_called_once()
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_expired_menu_answers(self):
        adapter = _make_adapter()
        query = MagicMock()
        query.message = SimpleNamespace(message_id=99)
        query.answer = AsyncMock()

        await adapter._handle_resume_callback(query, "rs:0", "12345")

        query.answer.assert_called_once()
        assert "expired" in query.answer.call_args.kwargs["text"].lower()
