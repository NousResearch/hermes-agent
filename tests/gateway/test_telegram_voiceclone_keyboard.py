"""Tests for Telegram voice-clone keyboard helpers."""

import sys
from unittest.mock import AsyncMock, MagicMock

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
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


@pytest.fixture
def voice_payload():
    return {
        "success": True,
        "default_voice": "sagan-clone-jackson",
        "voices": [
            {"voice_id": "sagan-clone-jackson", "display_name": "Jackson"},
            {"voice_id": "sagan-clone-sagan", "display_name": "Sagan"},
        ],
    }


class TestTelegramVoicecloneKeyboard:
    def test_build_keyboard_marks_current_voice(self, monkeypatch, voice_payload):
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

        adapter = _make_adapter()
        monkeypatch.setattr(adapter, "_voiceclone_current_voice", lambda payload=None: "sagan-clone-jackson")

        keyboard = adapter._build_voiceclone_keyboard(voice_payload)

        first_row = keyboard.inline_keyboard[0]
        assert first_row[0].text == "✓ Jackson"
        assert first_row[0].callback_data == "vc:pick:sagan-clone-jackson"
        assert first_row[1].callback_data == "vc:pick:sagan-clone-sagan"
        assert keyboard.inline_keyboard[-1][0].callback_data == "vc:menu"
        assert keyboard.inline_keyboard[-1][1].callback_data == "vc:close"

    @pytest.mark.asyncio
    async def test_pick_callback_edits_to_action_keyboard(self, monkeypatch):
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

        adapter = _make_adapter()
        monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *args, **kwargs: True)

        query = AsyncMock()
        query.data = "vc:pick:sagan-clone-sagan"
        query.from_user = MagicMock(id=123, first_name="Tester")
        query.message = MagicMock(chat_id=456, message_thread_id=None)
        query.message.chat = MagicMock(type="private")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_voiceclone_callback(query, "vc:pick:sagan-clone-sagan")

        query.edit_message_text.assert_awaited_once()
        kwargs = query.edit_message_text.call_args.kwargs
        assert "sagan-clone-sagan" in kwargs["text"]
        rows = kwargs["reply_markup"].inline_keyboard
        assert rows[0][0].callback_data == "vc:set:sagan-clone-sagan"
        assert rows[0][1].callback_data == "vc:test:sagan-clone-sagan"
        query.answer.assert_awaited_once()
