"""Tests for Telegram protected sensitive-input prompts."""

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
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter(extra=None):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_send_secret_capture_renders_precise_prompt_and_cancel_button():
    adapter = _make_adapter()
    mock_msg = MagicMock()
    mock_msg.message_id = 321
    assert adapter._bot is not None
    adapter._bot.send_message = AsyncMock(return_value=mock_msg)

    result = await adapter.send_secret_capture(
        chat_id="12345",
        prompt="Send <token>",
        env_var="MY_SECRET",
        secret_id="sid1",
        session_key="session1",
        metadata={"thread_id": "777", "reply_to_message_id": "555"},
        timeout_seconds=600,
    )

    assert result.success is True
    assert result.message_id == "321"
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 12345
    assert "Protected sensitive input" in kwargs["text"]
    assert "Send &lt;token&gt;" in kwargs["text"]
    assert "including slash commands" in kwargs["text"]
    assert "Use the Cancel button" in kwargs["text"]
    assert "agent model will not receive" in kwargs["text"]
    assert kwargs["reply_markup"] is not None
    assert adapter._secret_capture_state["sid1"]["session_key"] == "session1"
    assert adapter._secret_capture_prompt_messages["sid1"] == 321


@pytest.mark.asyncio
async def test_finalize_secret_capture_edits_prompt_and_clears_state():
    adapter = _make_adapter()
    assert adapter._bot is not None
    adapter._bot.edit_message_text = AsyncMock()
    adapter._secret_capture_state["sid2"] = {"session_key": "session2", "env_var": "MY_SECRET", "chat_id": "12345"}
    adapter._secret_capture_prompt_messages["sid2"] = 654

    result = await adapter.finalize_secret_capture(
        chat_id="12345",
        secret_id="sid2",
        status="timeout",
        env_var="MY_SECRET",
    )

    assert result.success is True
    assert "sid2" not in adapter._secret_capture_state
    assert "sid2" not in adapter._secret_capture_prompt_messages
    kwargs = adapter._bot.edit_message_text.call_args.kwargs
    assert kwargs["message_id"] == 654
    assert "expired" in kwargs["text"]
    assert kwargs["reply_markup"] is None


@pytest.mark.asyncio
async def test_secret_capture_cancel_callback_resolves_and_edits(monkeypatch):
    from tools import secret_capture_gateway

    adapter = _make_adapter()
    adapter._secret_capture_state["sid3"] = {"session_key": "session3", "env_var": "MY_SECRET", "chat_id": "12345"}
    adapter._secret_capture_prompt_messages["sid3"] = 111

    entry = secret_capture_gateway.register("sid3", "session3", "MY_SECRET", "Send it")
    from gateway.session import SessionSource
    from gateway.config import Platform
    entry.bind_source(SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="42"))
    try:
        query = MagicMock()
        query.data = "sgc:cancel:sid3"
        query.message.chat_id = 12345
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user.id = "42"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: True)

        await adapter._handle_callback_query(update, MagicMock())

        assert entry.cancelled is True
        assert entry.reason == "button_cancelled"
        assert "sid3" not in adapter._secret_capture_state
        query.edit_message_text.assert_awaited()
    finally:
        secret_capture_gateway.clear_session("session3")
