"""Regression guard: send_slash_confirm must use format_message + MARKDOWN_V2."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace
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

from gateway.platforms.telegram import TelegramAdapter
from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestSendSlashConfirm:

    @pytest.mark.asyncio
    async def test_uses_markdown_v2_and_escapes_special_chars(self):
        """send_slash_confirm must pass preview through format_message and use
        MARKDOWN_V2 — so commands with underscores, dots, or brackets don't
        raise BadRequest: Can't parse entities."""
        adapter = _make_adapter()
        sent = {}

        async def mock_send(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=7)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send)

        result = await adapter.send_slash_confirm(
            chat_id="100",
            title="Confirm",
            message="/run script_name.sh --flag=value [option]",
            session_key="sk",
            confirm_id="cid1",
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        # Underscores and dots must be escaped by format_message
        assert "script\\_name" in sent["text"]
        assert "\\." in sent["text"]

    @pytest.mark.asyncio
    async def test_stores_slash_confirm_state(self):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(
            return_value=SimpleNamespace(message_id=8)
        )

        await adapter.send_slash_confirm(
            chat_id="100",
            title="Confirm",
            message="reload-mcp",
            session_key="my-session",
            confirm_id="cid2",
        )

        assert adapter._slash_confirm_state["cid2"] == "my-session"

    @pytest.mark.asyncio
    async def test_not_connected_returns_failure(self):
        adapter = _make_adapter()
        adapter._bot = None

        result = await adapter.send_slash_confirm(
            chat_id="100",
            title="Confirm",
            message="reload-mcp",
            session_key="sk",
            confirm_id="cid3",
        )

        assert result.success is False


def _make_palette_update(data="qa:usage", *, thread_id="99"):
    chat = SimpleNamespace(id=-100, type="supergroup", title="Ops", is_forum=True)
    message = SimpleNamespace(
        chat_id=-100,
        chat=chat,
        message_id=42,
        message_thread_id=thread_id,
        is_topic_message=True,
    )
    query = SimpleNamespace(
        data=data,
        message=message,
        from_user=SimpleNamespace(id=123, first_name="Merlin"),
        answer=AsyncMock(),
    )
    return SimpleNamespace(callback_query=query), query


class TestTelegramCommandPalette:
    """Regression tests for Telegram palette callback dispatch and safety gates."""

    @pytest.mark.asyncio
    async def test_palette_callback_dispatches_shared_command_event_with_topic_context(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
        monkeypatch.delenv("TELEGRAM_ALLOWED_TOPICS", raising=False)
        monkeypatch.delenv("TELEGRAM_IGNORED_THREADS", raising=False)
        adapter = _make_adapter()
        adapter.config.extra["group_topics"] = [
            {"chat_id": "-100", "topics": [{"thread_id": "99", "name": "Ops", "skill": "ops-skill"}]}
        ]
        adapter.handle_message = AsyncMock()
        update, query = _make_palette_update("qa:usage")

        await adapter._handle_callback_query(update, SimpleNamespace())

        query.answer.assert_awaited_once_with(text="Running /usage…")
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/usage"
        assert event.message_type == MessageType.COMMAND
        assert event.source.chat_id == "-100"
        assert event.source.chat_type == "group"
        assert event.source.thread_id == "99"
        assert event.source.chat_topic == "Ops"
        assert event.auto_skill == "ops-skill"

    @pytest.mark.asyncio
    async def test_sensitive_palette_callback_requires_confirmation(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
        monkeypatch.delenv("TELEGRAM_ALLOWED_TOPICS", raising=False)
        monkeypatch.delenv("TELEGRAM_IGNORED_THREADS", raising=False)
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._send_palette_confirmation = AsyncMock()
        update, query = _make_palette_update("qa:yolo")

        await adapter._handle_callback_query(update, SimpleNamespace())

        query.answer.assert_awaited_once_with(text="Confirm /yolo to continue.")
        adapter._send_palette_confirmation.assert_awaited_once_with(query, "yolo", "-100", "99")
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirmed_sensitive_palette_callback_sets_preconfirmed_flag(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
        monkeypatch.delenv("TELEGRAM_ALLOWED_TOPICS", raising=False)
        monkeypatch.delenv("TELEGRAM_IGNORED_THREADS", raising=False)
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._palette_confirmations["nonce1"] = {
            "command": "new",
            "chat_id": "-100",
            "user_id": "123",
            "thread_id": "99",
            "ts": time.monotonic(),
        }
        update, _query = _make_palette_update("qa:confirm:new:nonce1")

        await adapter._handle_callback_query(update, SimpleNamespace())

        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/new"
        assert getattr(event, "preconfirmed_destructive", False) is True

    @pytest.mark.asyncio
    async def test_stateless_or_expired_palette_confirmation_is_rejected(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
        monkeypatch.delenv("TELEGRAM_ALLOWED_TOPICS", raising=False)
        monkeypatch.delenv("TELEGRAM_IGNORED_THREADS", raising=False)
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        update, query = _make_palette_update("qa:confirm:new")
        await adapter._handle_callback_query(update, SimpleNamespace())
        query.answer.assert_awaited_with(text="Confirmation expired. Please try again.")
        adapter.handle_message.assert_not_called()

        adapter._palette_confirmations["expired"] = {
            "command": "new",
            "chat_id": "-100",
            "user_id": "123",
            "thread_id": "99",
            "ts": time.monotonic() - adapter._PALETTE_CONFIRM_TTL - 1,
        }
        update, query = _make_palette_update("qa:confirm:new:expired")
        await adapter._handle_callback_query(update, SimpleNamespace())
        query.answer.assert_awaited_with(text="Confirmation expired. Please try again.")
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_palette_callback_respects_allowed_topic_gate(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
        monkeypatch.setenv("TELEGRAM_ALLOWED_TOPICS", "100")
        monkeypatch.delenv("TELEGRAM_IGNORED_THREADS", raising=False)
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        update, query = _make_palette_update("qa:usage", thread_id="99")

        await adapter._handle_callback_query(update, SimpleNamespace())

        query.answer.assert_awaited_once_with(text="This palette is not available in this chat/topic.")
        adapter.handle_message.assert_not_called()
