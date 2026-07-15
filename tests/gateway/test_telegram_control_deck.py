"""Tests for Telegram Control Deck /menu inline keyboard."""

import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure the repo root is importable.
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


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
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _control_deck_query(data="cd:code", user_id="777"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_thread_id = None
    query.message.chat.type = "private"
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    return update, query


def _outside_inline_code(text: str) -> str:
    """Return text segments outside Telegram MarkdownV2 inline code spans."""
    return "".join(re.split(r"`[^`]*`", text))


def _unescaped_positions(text: str, char: str) -> list[int]:
    return [
        idx
        for idx, value in enumerate(text)
        if value == char and (idx == 0 or text[idx - 1] != "\\")
    ]


def test_control_deck_payload_sections_render_as_markdownv2_safe_text():
    adapter = _make_adapter()

    for section in ["home", "code", "review", "memory", "system", "commands", "unknown"]:
        text, keyboard = adapter._control_deck_payload(section)
        non_code = _outside_inline_code(text)

        assert text.strip()
        assert "Hermes" in text or section != "home"
        for char in [".", "-", "(", ")"]:
            assert _unescaped_positions(non_code, char) == []
        assert keyboard is not None
        assert getattr(keyboard, "inline_keyboard", None)


@pytest.mark.asyncio
async def test_control_deck_callback_denies_unauthorized_user_without_editing():
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=False)
    update, query = _control_deck_query(user_id="999")

    await adapter._handle_callback_query(update, MagicMock())

    adapter._is_callback_user_authorized.assert_called_once()
    query.answer.assert_called_once()
    assert "not authorized" in query.answer.call_args.kwargs["text"].lower()
    query.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_control_deck_callback_authorized_navigation_edits_message_only():
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    update, query = _control_deck_query(data="cd:review")

    await adapter._handle_callback_query(update, MagicMock())

    adapter._is_callback_user_authorized.assert_called_once()
    query.edit_message_text.assert_called_once()
    kwargs = query.edit_message_text.call_args.kwargs
    assert "Ревью" in kwargs["text"]
    assert kwargs["reply_markup"] is not None
    query.answer.assert_called_once_with()
