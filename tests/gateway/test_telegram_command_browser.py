"""Tests for Telegram interactive command browser buttons."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported
# ---------------------------------------------------------------------------
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
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramCommandBrowser:
    @pytest.mark.asyncio
    async def test_send_command_browser_renders_buttons_and_stores_state(self):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=200))

        result = await adapter.send_command_browser(
            chat_id="12345",
            entries=[
                {"command_text": "/status", "button_text": "status", "description": "Show status"},
                {"command_text": "/model", "button_text": "model", "description": "Switch model"},
            ],
            page=1,
            page_size=1,
            title="Command Browser",
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert kwargs["reply_markup"] is not None
        assert "Command Browser" in kwargs["text"]
        assert 200 in adapter._command_browser_state
        state = adapter._command_browser_state[200]
        assert state["page"] == 1
        assert len(state["entries"]) == 2

    @pytest.mark.asyncio
    async def test_command_browser_page_navigation_edits_same_message(self):
        adapter = _make_adapter()
        adapter._command_browser_state[42] = {
            "entries": [
                {"command_text": "/status", "button_text": "status", "description": "Show status"},
                {"command_text": "/model", "button_text": "model", "description": "Switch model"},
            ],
            "page": 1,
            "page_size": 1,
            "title": "Command Browser",
        }

        query = AsyncMock()
        query.data = "gc:p:2"
        query.message = MagicMock()
        query.message.message_id = 42
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "/model" in edit_kwargs["text"]
        assert adapter._command_browser_state[42]["page"] == 2
        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_command_browser_run_dispatches_selected_command(self):
        adapter = _make_adapter()
        adapter._command_browser_state[42] = {
            "entries": [
                {"command_text": "/status", "button_text": "status", "description": "Show status"},
            ],
            "page": 1,
            "page_size": 10,
            "title": "Command Browser",
        }
        adapter._message_handler = AsyncMock(return_value="Current status")
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="9"))

        query = AsyncMock()
        query.data = "gc:r:0"
        query.message = MagicMock()
        query.message.message_id = 42
        query.message.chat_id = 12345
        query.message.chat = MagicMock()
        query.message.chat.id = 12345
        query.message.chat.type = "private"
        query.message.chat.title = None
        query.message.chat.full_name = "Tester Chat"
        query.message.message_thread_id = None
        query.message.date = None
        query.message.from_user = None
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.full_name = "Tester"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.update_id = 123

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        await_args = adapter._message_handler.await_args
        assert await_args is not None
        dispatched_event = await_args.args[0]
        assert dispatched_event.text == "/status"
        adapter.send.assert_awaited_once()
        query.edit_message_text.assert_awaited()
