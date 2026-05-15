"""Tests for Telegram generic KB action-card callbacks."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
from gateway.platforms.telegram import TelegramAdapter
from tools import kb_callback_registry as kb_callbacks


class FakeButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data


class FakeMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


@pytest.fixture(autouse=True)
def _clean_registry():
    kb_callbacks._pending.clear()
    yield
    kb_callbacks._pending.clear()


@pytest.mark.asyncio
async def test_send_kb_actions_renders_short_opaque_buttons_in_thread():
    adapter = _make_adapter(extra={"disable_link_previews": True})
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=55))

    async def noop(ctx):
        return None

    actions = [
        kb_callbacks.KbAction(label="Archive", action_id="archive", handler=noop),
        kb_callbacks.KbAction(label="Open Note", action_id="open", handler=noop),
    ]

    with patch("gateway.platforms.telegram.InlineKeyboardButton", FakeButton), \
            patch("gateway.platforms.telegram.InlineKeyboardMarkup", FakeMarkup):
        result = await adapter.send_kb_actions(
            chat_id="12345",
            text="*KB actions*",
            actions=actions,
            metadata={"thread_id": "999"},
            reply_to="44",
        )

    assert result.success is True
    assert result.message_id == "55"

    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["chat_id"] == 12345
    assert kwargs["message_thread_id"] == 999
    assert kwargs["reply_to_message_id"] == 44
    assert "MARKDOWN" in repr(kwargs["parse_mode"])
    assert kwargs.get("disable_web_page_preview") is True or kwargs.get("link_preview_options") is not None
    assert kwargs.get("disable_notification") is True

    rows = kwargs["reply_markup"].inline_keyboard
    assert [[button.text for button in row] for row in rows] == [["Archive"], ["Open Note"]]
    callback_data = [row[0].callback_data for row in rows]
    assert all(data.startswith("kb:") for data in callback_data)
    assert all(len(data.encode("utf-8")) <= 64 for data in callback_data)

    callback_ids = [data.split(":", 1)[1] for data in callback_data]
    assert {kb_callbacks.get_pending(cid)["action_id"] for cid in callback_ids} == {"archive", "open"}


@pytest.mark.asyncio
async def test_kb_callback_resolves_removes_buttons_and_sends_followup_in_topic():
    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=77))
    seen = []

    async def handler(ctx):
        seen.append(ctx)
        return "Action **complete**"

    callback_id = kb_callbacks.register(
        "archive",
        handler,
        chat_id="12345",
        thread_id="999",
        metadata={"entity": "accounts/acme"},
    )

    query = AsyncMock()
    query.data = f"kb:{callback_id}"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_thread_id = 999
    query.message.message_id = 44
    query.message.chat.type = "supergroup"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Ada"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    query.answer.assert_called_once()
    assert "opening" in query.answer.call_args[1]["text"].lower()
    query.edit_message_reply_markup.assert_called_once_with(reply_markup=None)
    assert kb_callbacks.get_pending(callback_id) is None

    assert len(seen) == 1
    assert seen[0].actor_id == "777"
    assert seen[0].actor_name == "Ada"
    assert seen[0].chat_id == "12345"
    assert seen[0].thread_id == "999"
    assert seen[0].metadata == {"entity": "accounts/acme"}

    followup = adapter._bot.send_message.call_args[1]
    assert followup["chat_id"] == 12345
    assert followup["message_thread_id"] == 999
    assert "Action" in followup["text"]
    assert "MARKDOWN_V2" in repr(followup["parse_mode"])


@pytest.mark.asyncio
async def test_kb_callback_rejects_unauthorized_user_without_resolving():
    adapter = _make_adapter()

    async def handler(ctx):
        return "should not run"

    callback_id = kb_callbacks.register("archive", handler)

    query = AsyncMock()
    query.data = f"kb:{callback_id}"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.chat.type = "private"
    query.from_user = MagicMock()
    query.from_user.id = "222"
    query.from_user.first_name = "Mallory"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111"}):
        await adapter._handle_callback_query(update, MagicMock())

    query.answer.assert_called_once()
    assert "not authorized" in query.answer.call_args[1]["text"].lower()
    query.edit_message_reply_markup.assert_not_called()
    assert kb_callbacks.get_pending(callback_id) is not None


@pytest.mark.asyncio
async def test_kb_callback_dispatch_errors_are_redacted_to_user():
    adapter = _make_adapter()
    adapter.send_kb_actions = AsyncMock(side_effect=RuntimeError("secret /tmp/path"))

    async def handler(ctx):
        return {
            "text": "Preview ready",
            "actions": [kb_callbacks.KbAction(label="Confirm", action_id="confirm", handler=lambda _ctx: "ok")],
        }

    callback_id = kb_callbacks.register("preview", handler, chat_id="12345")

    query = AsyncMock()
    query.data = f"kb:{callback_id}"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_id = 44
    query.message.message_thread_id = None
    query.message.chat.type = "private"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Ada"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    query.answer.assert_any_call(text="Opening KB action...")
    query.answer.assert_any_call(text="KB action failed. Check gateway logs for details.")
    assert "secret" not in query.answer.call_args[1]["text"]


@pytest.mark.asyncio
async def test_kb_callback_expired_button_sends_visible_refresh_hint():
    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=88))

    query = AsyncMock()
    query.data = "kb:missing-token"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_id = 44
    query.message.message_thread_id = None
    query.message.chat.type = "private"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Ada"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    query.answer.assert_called_once()
    assert "expired" in query.answer.call_args[1]["text"].lower()
    query.edit_message_reply_markup.assert_called_once_with(reply_markup=None)
    sent = adapter._bot.send_message.call_args[1]
    assert "expired" in sent["text"].lower()
    assert "/kbqueue" in sent["text"]
