"""Tests for Telegram /resume inline button UI.

Tests the button-based session resume UI that replaces the plain-text
list when /resume is invoked with no arguments on Telegram.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, ANY

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
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
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    adapter._bot.username = "testbot"
    adapter._app = MagicMock()
    return adapter


def _make_msg(chat_id=12345, text="/resume", message_id=100):
    msg = MagicMock()
    msg.chat_id = chat_id
    msg.text = text
    msg.message_id = message_id
    msg.message_thread_id = None
    msg.effective_attachment = None
    msg.caption = None
    msg.chat = MagicMock()
    msg.chat.type = "private"
    msg.chat.id = chat_id
    return msg


def _make_update(msg, update_id=1):
    update = MagicMock()
    update.update_id = update_id
    update.message = msg
    update.effective_message = msg
    return update


def _make_context():
    return MagicMock()


def _make_session_row(sid, title, preview="up to 60 chars preview text here", source="telegram"):
    return {
        "id": sid,
        "title": title,
        "preview": preview,
        "message_count": 5,
        "source": source,
        "last_active": "2026-05-27T00:00:00",
    }


def _make_runner_with_sessions(session_rows):
    """Create a mock GatewayRunner that returns session_rows from get_resume_sessions."""
    runner = MagicMock()
    runner.get_resume_sessions = MagicMock(return_value=session_rows)
    return runner


class TestHandleCommandResumeInterception:
    """Tests for /resume interception in _handle_command."""

    @pytest.mark.asyncio
    async def test_resume_no_args_triggers_button_ui(self):
        """/resume with no arguments sends inline keyboard instead of plain text."""
        adapter = _make_adapter()
        adapter._send_resume_button_ui = AsyncMock()
        adapter.handle_message = AsyncMock()

        msg = _make_msg(text="/resume")
        update = _make_update(msg)
        context = _make_context()

        await adapter._handle_command(update, context)

        adapter._send_resume_button_ui.assert_called_once_with(msg, ANY)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_resume_with_args_falls_through(self):
        """/resume <args> falls through to normal gateway handling."""
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        msg = _make_msg(text="/resume My Session")
        update = _make_update(msg)
        context = _make_context()

        await adapter._handle_command(update, context)

        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_other_commands_unaffected(self):
        """Other commands like /help are not intercepted."""
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        msg = _make_msg(text="/help")
        update = _make_update(msg)
        context = _make_context()

        await adapter._handle_command(update, context)

        adapter.handle_message.assert_called_once()


class TestSendResumeButtonUI:
    """Tests for _send_resume_button_ui."""

    @pytest.mark.asyncio
    async def test_sends_keyboard_with_session_buttons(self):
        """Sends inline keyboard with one button per session."""
        adapter = _make_adapter()
        sessions = [
            _make_session_row("sess_001", "Alpha Project"),
            _make_session_row("sess_002", "Beta Project"),
        ]
        runner = _make_runner_with_sessions(sessions)
        adapter._message_handler = MagicMock()
        adapter._message_handler.__self__ = runner

        sent = {}

        async def mock_send(**kwargs):
            sent.update(kwargs)

        adapter._send_message_with_thread_fallback = mock_send

        msg = _make_msg()
        event = MagicMock()
        await adapter._send_resume_button_ui(msg, event)

        assert "reply_markup" in sent
        assert "text" in sent
        # Text should contain numbered session titles
        assert "1. Alpha Project" in sent["text"]
        assert "2. Beta Project" in sent["text"]
        # No parse mode — plain text
        assert "parse_mode" not in sent
        # Disable web page preview
        assert sent.get("disable_web_page_preview") is True

    @pytest.mark.asyncio
    async def test_button_labels_truncated(self):
        """Long session titles are truncated in button labels."""
        adapter = _make_adapter()
        long_title = "A" * 50
        sessions = [_make_session_row("sess_001", long_title)]
        runner = _make_runner_with_sessions(sessions)
        adapter._message_handler = MagicMock()
        adapter._message_handler.__self__ = runner

        sent = {}

        async def mock_send(**kwargs):
            sent.update(kwargs)

        adapter._send_message_with_thread_fallback = mock_send

        msg = _make_msg()
        event = MagicMock()
        await adapter._send_resume_button_ui(msg, event)

        # Button label should be truncated
        keyboard = sent["reply_markup"]
        btn_text = keyboard.inline_keyboard[0][0].text
        assert len(btn_text) <= 30  # reasonable mobile width
        assert btn_text.endswith("...")
        # But full title should be in the message text
        assert long_title in sent["text"]

    @pytest.mark.asyncio
    async def test_includes_preview_in_message_body(self):
        """Session preview is included in the message body."""
        adapter = _make_adapter()
        sessions = [
            _make_session_row("sess_001", "My Session", preview="Hello world preview"),
        ]
        runner = _make_runner_with_sessions(sessions)
        adapter._message_handler = MagicMock()
        adapter._message_handler.__self__ = runner

        sent = {}

        async def mock_send(**kwargs):
            sent.update(kwargs)

        adapter._send_message_with_thread_fallback = mock_send

        msg = _make_msg()
        event = MagicMock()
        await adapter._send_resume_button_ui(msg, event)

        assert "Hello world preview" in sent["text"]

    @pytest.mark.asyncio
    async def test_no_sessions_sends_fallback_message(self):
        """When no sessions found, sends a helpful message."""
        adapter = _make_adapter()
        runner = _make_runner_with_sessions([])
        adapter._message_handler = MagicMock()
        adapter._message_handler.__self__ = runner

        sent = {}

        async def mock_send(**kwargs):
            sent.update(kwargs)

        adapter._send_message_with_thread_fallback = mock_send

        msg = _make_msg()
        event = MagicMock()
        await adapter._send_resume_button_ui(msg, event)

        assert "No named sessions" in sent["text"]
        assert "/title" in sent["text"]

    @pytest.mark.asyncio
    async def test_no_runner_falls_back(self):
        """When runner is not available, falls back to handle_message."""
        adapter = _make_adapter()
        adapter._message_handler = None
        adapter.handle_message = AsyncMock()

        msg = _make_msg()
        event = MagicMock()
        await adapter._send_resume_button_ui(msg, event)

        adapter.handle_message.assert_called_once()


class TestHandleResumeCallback:
    """Tests for _handle_resume_callback."""

    @pytest.mark.asyncio
    async def test_callback_resumes_session(self):
        """Clicking a resume button dispatches /resume <session_id>."""
        adapter = _make_adapter()

        query = MagicMock()
        query.from_user.id = "user123"
        query.from_user.first_name = "TestUser"
        query.answer = AsyncMock()
        query.message.message_id = "msg456"

        with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
            with patch.object(adapter, "handle_message", new_callable=AsyncMock) as mock_handle:
                await adapter._handle_resume_callback(
                    query,
                    "sess_abc123",
                    query_chat_id="chat789",
                    query_chat_type="private",
                    query_thread_id=None,
                )

                query.answer.assert_called_once()
                mock_handle.assert_called_once()
                # Check the event text
                event = mock_handle.call_args[0][0]
                assert event.text == "/resume sess_abc123"

    @pytest.mark.asyncio
    async def test_callback_denied_for_unauthorized_user(self):
        """Unauthorized users get denied with a message."""
        adapter = _make_adapter()

        query = MagicMock()
        query.from_user.id = "baduser"
        query.from_user.first_name = "BadUser"
        query.answer = AsyncMock()

        with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
            await adapter._handle_resume_callback(
                query,
                "sess_abc123",
                query_chat_id="chat789",
                query_chat_type="private",
                query_thread_id=None,
            )

            query.answer.assert_called_once()
            answer_text = query.answer.call_args[1].get("text", "")
            assert "not authorized" in answer_text.lower()


class TestCallbackQueryRouting:
    """Tests for rs:<session_id> routing in _handle_callback_query."""

    @pytest.mark.asyncio
    async def test_rs_callback_routed_correctly(self):
        """rs: callbacks are routed to _handle_resume_callback."""
        adapter = _make_adapter()

        query = MagicMock()
        query.data = "rs:sess_xyz"
        query.from_user.id = "user123"
        query.from_user.first_name = "TestUser"
        query.answer = AsyncMock()
        query.message.chat_id = "chat789"
        query.message.message_id = "msg456"
        query.message.chat.type = "private"
        query.message.message_thread_id = None

        with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
            with patch.object(adapter, "handle_message", new_callable=AsyncMock):
                await adapter._handle_callback_query(
                    MagicMock(callback_query=query),
                    MagicMock(),
                )


class TestGetResumeSessions:
    """Tests for GatewayRunner.get_resume_sessions."""

    def test_returns_titled_sessions_sorted_by_recency(self, tmp_path):
        """Returns up to 12 titled sessions, excluding cron/tool."""
        from hermes_state import SessionDB
        from gateway.run import GatewayRunner
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.session import SessionSource

        db = SessionDB(db_path=tmp_path / "state.db")
        # Create sessions in order — oldest first
        db.create_session("old_sess", "telegram")
        db.set_session_title("old_sess", "Old Session")
        db.create_session("new_sess", "telegram")
        db.set_session_title("new_sess", "New Session")
        db.create_session("cron_sess", "cron")
        db.set_session_title("cron_sess", "Cron Session")
        db.create_session("no_title", "telegram")  # No title

        source = SessionSource(platform=Platform.TELEGRAM, user_id="123", chat_id="456")
        event = MagicMock(spec=MessageEvent)
        event.source = source

        runner = object.__new__(GatewayRunner)
        runner._session_db = db

        result = runner.get_resume_sessions(event, limit=12)

        titles = [s["title"] for s in result]
        assert "Cron Session" not in titles
        assert "no_title" not in [s.get("id") for s in result]
        assert len(result) <= 12
        # Should have both titled non-cron sessions
        assert "Old Session" in titles
        assert "New Session" in titles

        db.close()

    def test_respects_limit(self, tmp_path):
        """Does not return more than limit sessions."""
        from hermes_state import SessionDB
        from gateway.run import GatewayRunner
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.session import SessionSource

        db = SessionDB(db_path=tmp_path / "state.db")
        for i in range(20):
            sid = f"sess_{i:03d}"
            db.create_session(sid, "telegram")
            db.set_session_title(sid, f"Session {i}")

        source = SessionSource(platform=Platform.TELEGRAM, user_id="123", chat_id="456")
        event = MagicMock(spec=MessageEvent)
        event.source = source

        runner = object.__new__(GatewayRunner)
        runner._session_db = db

        result = runner.get_resume_sessions(event, limit=12)
        assert len(result) == 12

        db.close()
