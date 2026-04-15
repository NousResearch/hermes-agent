"""Tests for /commands inline keyboard menu on Telegram."""

import sys
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
    mock_msg.message_id = 42
    adapter._bot.send_message = AsyncMock(return_value=mock_msg)
    return adapter


class TestBuildCommandsKeyboard:
    def test_single_page_no_pagination(self):
        adapter = _make_adapter()
        entries = ["`/help` — Show help", "`/new` — New session"]
        keyboard, page_info = adapter._build_commands_keyboard(entries, page=0, page_size=12)
        assert len(keyboard.inline_keyboard) == 1  # close only
        assert page_info == ""

    def test_filters_non_command_entries(self):
        adapter = _make_adapter()
        entries = [
            "`/help` — Show help",
            "",
            "⚡ **Skill Commands**:",
            "`/study` — Review cards",
        ]
        keyboard, page_info = adapter._build_commands_keyboard(entries, page=0, page_size=12)
        assert len(keyboard.inline_keyboard) == 1  # close only
        assert page_info == ""

    def test_multi_page_has_pagination(self):
        adapter = _make_adapter()
        entries = [f"`/cmd{i}` — Command {i}" for i in range(30)]
        keyboard, page_info = adapter._build_commands_keyboard(entries, page=0, page_size=12)
        assert len(keyboard.inline_keyboard) == 2  # nav + close
        assert "1–12" in page_info

    def test_close_button_always_present(self):
        adapter = _make_adapter()
        entries = ["`/help` — Show help"]
        keyboard, _ = adapter._build_commands_keyboard(entries, page=0)
        last_row = keyboard.inline_keyboard[-1]
        assert any(button.callback_data == "cx" for button in last_row)


class TestSendCommandsMenu:
    @pytest.mark.asyncio
    async def test_sends_menu_and_stores_state(self):
        adapter = _make_adapter()
        entries = ["`/help` — Show help", "`/new` — New session"]

        result = await adapter.send_commands_menu(chat_id="12345", entries=entries, page=0)

        assert result.success
        assert adapter._commands_menu_state["12345"]["entries"] == entries
        assert adapter._commands_menu_state["12345"]["msg_id"] == 42
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert kwargs.get("parse_mode") is None
        assert "`/help` — Show help" in kwargs["text"]
        assert "`/new` — New session" in kwargs["text"]
        assert "Select a command" not in kwargs["text"]

    @pytest.mark.asyncio
    async def test_retries_without_thread_when_thread_not_found(self):
        adapter = _make_adapter()
        entries = ["`/help` — Show help"]
        call_log = []

        class FakeBadRequest(Exception):
            pass

        async def mock_send_message(**kwargs):
            call_log.append(dict(kwargs))
            if kwargs.get("message_thread_id") is not None:
                raise FakeBadRequest("Message thread not found")
            return SimpleNamespace(message_id=42)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_commands_menu(
            chat_id="12345",
            entries=entries,
            page=0,
            metadata={"thread_id": "99999"},
        )

        assert result.success
        assert len(call_log) == 2
        assert call_log[0]["message_thread_id"] == 99999
        assert "message_thread_id" not in call_log[1] or call_log[1]["message_thread_id"] is None

    @pytest.mark.asyncio
    async def test_sends_plain_text_for_markdown_unsafe_entries(self):
        adapter = _make_adapter()
        entries = [
            "`/help` — Show help",
            "⚡ **Skill Commands**:",
            "`/broken-skill` — Diagnose cases where `hermes -c \"<title>\"` picked `title",
        ]

        result = await adapter.send_commands_menu(chat_id="12345", entries=entries, page=0)

        assert result.success
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert kwargs.get("parse_mode") is None
        assert "`/help` — Show help" in kwargs["text"]
        assert "⚡ **Skill Commands**:" in kwargs["text"]
        assert "`/broken-skill` — Diagnose cases where `hermes -c \"<title>\"` picked `title" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_preserves_plain_text_underscores(self):
        adapter = _make_adapter()
        entries = ["`/env` — Explain HERMES_API_KEY and __init__.py in foo_bar_baz"]

        result = await adapter.send_commands_menu(chat_id="12345", entries=entries, page=0)

        assert result.success
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert "`/env` — Explain HERMES_API_KEY and __init__.py in foo_bar_baz" in kwargs["text"]


class TestCommandsCallback:
    @pytest.mark.asyncio
    async def test_page_navigation_edits_message(self):
        adapter = _make_adapter()
        entries = [f"`/cmd{i}` — Command {i}" for i in range(20)]
        adapter._commands_menu_state["12345"] = {"entries": entries, "msg_id": 42}
        query = MagicMock()
        query.message = SimpleNamespace(message_id=42)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._handle_commands_callback(query, "cp:1", "12345")

        query.edit_message_text.assert_called_once()
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_dismisses_menu(self):
        adapter = _make_adapter()
        adapter._commands_menu_state["12345"] = {"entries": ["`/help` — Show help"], "msg_id": 42}
        query = MagicMock()
        query.message = SimpleNamespace(message_id=42)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._handle_commands_callback(query, "cx", "12345")

        assert "12345" not in adapter._commands_menu_state
        query.edit_message_text.assert_called_once()
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_expired_menu_answers(self):
        adapter = _make_adapter()
        query = MagicMock()
        query.message = SimpleNamespace(message_id=42)
        query.answer = AsyncMock()

        await adapter._handle_commands_callback(query, "cp:0", "12345")

        query.answer.assert_called_once()
        assert "expired" in query.answer.call_args.kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_stale_message_id_answers_expired(self):
        adapter = _make_adapter()
        adapter._commands_menu_state["12345"] = {"entries": ["`/help` — Show help"], "msg_id": 42}
        query = MagicMock()
        query.message = SimpleNamespace(message_id=99)
        query.answer = AsyncMock()

        await adapter._handle_commands_callback(query, "cp:0", "12345")

        query.answer.assert_called_once()
        assert "expired" in query.answer.call_args.kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_page_navigation_edits_plain_text_for_markdown_unsafe_entries(self):
        adapter = _make_adapter()
        entries = [f"`/cmd{i}` — Command {i}" for i in range(12)] + [
            "`/broken-skill` — Diagnose cases where `hermes -c \"<title>\"` picked `title",
            "⚡ **Skill Commands**:",
        ]
        adapter._commands_menu_state["12345"] = {"entries": entries, "msg_id": 42}
        query = MagicMock()
        query.message = SimpleNamespace(message_id=42)
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._handle_commands_callback(query, "cp:1", "12345")

        kwargs = query.edit_message_text.call_args.kwargs
        assert kwargs.get("parse_mode") is None
        assert "`/broken-skill` — Diagnose cases where `hermes -c \"<title>\"` picked `title" in kwargs["text"]
        assert "⚡ **Skill Commands**:" in kwargs["text"]
        query.answer.assert_called_once()
