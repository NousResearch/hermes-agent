"""Telegram callback-query plugin hook tests."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_update(data="spu:a:123"):
    query = AsyncMock()
    query.data = data
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Sasha"
    query.message = MagicMock()
    query.message.chat_id = -1003819053296
    query.message.message_thread_id = 3852
    query.message.text = "old text"
    query.message.chat = MagicMock()
    query.message.chat.type = "supergroup"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    return update, query


@pytest.mark.asyncio
async def test_unknown_callback_can_be_handled_by_plugin_hook(monkeypatch):
    adapter = _make_adapter()
    update, query = _make_update()
    calls = []

    def fake_invoke_hook(name, **kwargs):
        calls.append((name, kwargs))
        return [
            {
                "handled": True,
                "answer_text": "approved",
                "edit_text": "<b>approved</b>",
                "parse_mode": "HTML",
                "remove_keyboard": True,
            }
        ]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "telegram_callback_query")

    await adapter._handle_callback_query(update, MagicMock())

    assert calls
    assert calls[0][0] == "telegram_callback_query"
    payload = calls[0][1]
    assert payload["callback_data"] == "spu:a:123"
    assert payload["user_id"] == "777"
    assert payload["chat_id"] == "-1003819053296"
    assert payload["thread_id"] == "3852"

    query.answer.assert_awaited_once_with(text="approved", show_alert=False)
    query.edit_message_text.assert_awaited_once_with(
        text="<b>approved</b>",
        parse_mode="HTML",
        reply_markup=None,
    )


@pytest.mark.asyncio
async def test_plugin_hook_can_decline_callback(monkeypatch):
    adapter = _make_adapter()
    update, query = _make_update(data="unknown:noop")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [{"action": "allow"}])
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "telegram_callback_query")

    await adapter._handle_callback_query(update, MagicMock())

    query.answer.assert_not_awaited()
    query.edit_message_text.assert_not_awaited()
