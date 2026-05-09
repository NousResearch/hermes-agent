"""Tests for Telegram assistant-response Quick Actions."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)
_ensure_telegram_mock()

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.telegram import TelegramAdapter
from gateway.session import SessionSource, build_session_key


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _quick_action_metadata():
    return {
        "thread_id": "3220",
        "quick_actions": {
            "content": "오빠, 다음으로 quick action을 붙였어.",
            "source": {
                "platform": "telegram",
                "chat_id": "12345",
                "chat_type": "supergroup",
                "chat_name": "Hermes Agent - Mina",
                "thread_id": "3220",
                "user_id": "111",
                "user_name": "Joohyun",
            },
        },
    }


class TestTelegramQuickActionsSend:
    @pytest.mark.asyncio
    async def test_send_attaches_quick_action_keyboard_to_final_chunk_and_persists_payload(self, tmp_path):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 888
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            result = await adapter.send(
                chat_id="12345",
                content="오빠, 다음으로 quick action을 붙였어.",
                metadata=_quick_action_metadata(),
            )

        assert result.success is True
        assert result.message_id == "888"
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert kwargs["chat_id"] == 12345
        assert kwargs["message_thread_id"] == 3220
        assert kwargs["reply_markup"] is not None

        assert len(adapter._quick_action_state) == 1
        token, payload = next(iter(adapter._quick_action_state.items()))
        assert len(token) == 10
        assert payload["message_id"] == "888"
        assert payload["chat_id"] == "12345"
        assert payload["thread_id"] == "3220"
        assert payload["content"] == "오빠, 다음으로 quick action을 붙였어."
        assert payload["source"]["user_name"] == "Joohyun"

        active_path = tmp_path / "telegram_quick_actions" / "active_actions.json"
        active = json.loads(active_path.read_text())
        assert token in active

    @pytest.mark.asyncio
    async def test_send_without_quick_action_metadata_has_no_keyboard(self, tmp_path):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 889
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            result = await adapter.send(
                chat_id="12345",
                content="plain response",
                metadata={"thread_id": "3220"},
            )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert "reply_markup" in kwargs
        assert kwargs["reply_markup"] is None
        assert adapter._quick_action_state == {}

    @pytest.mark.asyncio
    async def test_background_assistant_reply_injects_quick_action_metadata_for_telegram(self, tmp_path):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 890
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        async def handler(_event):
            return "assistant result"

        adapter.set_message_handler(handler)
        event = MessageEvent(
            text="do a thing",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="supergroup",
                chat_name="Hermes Agent - Mina",
                user_id="111",
                user_name="Joohyun",
                thread_id="3220",
            ),
            message_id="777",
        )

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            await adapter._process_message_background(event, build_session_key(event.source))

        kwargs = adapter._bot.send_message.call_args.kwargs
        assert kwargs["reply_markup"] is not None
        assert len(adapter._quick_action_state) == 1
        payload = next(iter(adapter._quick_action_state.values()))
        assert payload["content"] == "assistant result"
        assert payload["source"]["user_name"] == "Joohyun"


class TestTelegramQuickActionsCallback:
    @pytest.mark.asyncio
    async def test_save_callback_records_ledger_removes_buttons_and_active_state(self, tmp_path):
        adapter = _make_adapter()
        payload = {
            "token": "abc123def4",
            "chat_id": "12345",
            "thread_id": "3220",
            "message_id": "888",
            "content": "saved response",
            "source": {"user_name": "Joohyun"},
        }
        adapter._quick_action_state["abc123def4"] = payload

        query = MagicMock()
        query.data = "qa:abc123def4:save"
        query.from_user.id = 111
        query.from_user.first_name = "Joohyun"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await adapter._handle_callback_query(update, MagicMock())

        query.answer.assert_called_once_with(text="💾 Saved")
        query.edit_message_reply_markup.assert_called_once_with(reply_markup=None)
        adapter._bot.send_message.assert_called_once()
        confirm_kwargs = adapter._bot.send_message.call_args.kwargs
        assert confirm_kwargs["chat_id"] == 12345
        assert confirm_kwargs["message_thread_id"] == 3220
        assert confirm_kwargs["reply_to_message_id"] == 888
        assert "Saved as routing candidate" in confirm_kwargs["text"]
        assert "cortex_memory" in confirm_kwargs["text"]
        assert "abc123def4" not in adapter._quick_action_state

        records_path = tmp_path / "telegram_quick_actions" / "actions.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record["token"] == "abc123def4"
        assert record["action"] == "save"
        assert record["user_id"] == "111"
        assert record["user_name"] == "Joohyun"
        assert record["chat_id"] == "12345"
        assert record["thread_id"] == "3220"
        assert record["message_id"] == "888"
        assert record["content"] == "saved response"

        saved_path = tmp_path / "telegram_quick_actions" / "saved_responses.jsonl"
        assert saved_path.exists()
        routing_path = tmp_path / "telegram_quick_actions" / "routing_candidates.jsonl"
        routing = [json.loads(line) for line in routing_path.read_text().splitlines()]
        assert len(routing) == 1
        assert routing[0]["action"] == "save"
        assert routing[0]["status"] == "candidate"
        assert routing[0]["recommended_targets"] == ["cortex_memory"]
        assert routing[0]["memory"]["project"] == "hermes"

    @pytest.mark.asyncio
    async def test_callback_loads_persisted_payload_after_restart(self, tmp_path):
        sending_adapter = _make_adapter()
        payload = {
            "token": "abc123def4",
            "chat_id": "12345",
            "thread_id": "3220",
            "message_id": "888",
            "content": "todo response",
            "source": {},
        }
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            sending_adapter._save_active_quick_action("abc123def4", payload)

        restarted_adapter = _make_adapter()
        query = MagicMock()
        query.data = "qa:abc123def4:todo"
        query.from_user.id = 111
        query.from_user.first_name = "Joohyun"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await restarted_adapter._handle_callback_query(update, MagicMock())

        query.answer.assert_called_once_with(text="☑️ Todo captured")
        restarted_adapter._bot.send_message.assert_called_once()
        confirm_kwargs = restarted_adapter._bot.send_message.call_args.kwargs
        assert confirm_kwargs["chat_id"] == 12345
        assert confirm_kwargs["message_thread_id"] == 3220
        assert confirm_kwargs["reply_to_message_id"] == 888
        assert "Todo captured as routing candidate" in confirm_kwargs["text"]
        assert "cortex_todo" in confirm_kwargs["text"]
        active_path = tmp_path / "telegram_quick_actions" / "active_actions.json"
        active = json.loads(active_path.read_text())
        assert "abc123def4" not in active
        assert (tmp_path / "telegram_quick_actions" / "todos.jsonl").exists()
        routing = [
            json.loads(line)
            for line in (tmp_path / "telegram_quick_actions" / "routing_candidates.jsonl").read_text().splitlines()
        ]
        assert len(routing) == 1
        assert routing[0]["action"] == "todo"
        assert routing[0]["recommended_targets"] == ["cortex_todo", "kanban_candidate"]
        assert routing[0]["todo"]["project"] == "hermes"
        assert routing[0]["todo"]["category"] == "dev"
        assert routing[0]["todo"]["source_type"] == "manual"

    @pytest.mark.asyncio
    async def test_callback_rejects_unauthorized_user(self, tmp_path):
        adapter = _make_adapter()
        adapter._quick_action_state["abc123def4"] = {"token": "abc123def4", "content": "secret"}

        query = MagicMock()
        query.data = "qa:abc123def4:save"
        query.from_user.id = 222
        query.from_user.first_name = "Other"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111"}):
                await adapter._handle_callback_query(update, MagicMock())

        query.answer.assert_called_once_with(text="⛔ You are not authorized to use this quick action.")
        query.edit_message_reply_markup.assert_not_called()
        assert not (tmp_path / "telegram_quick_actions" / "actions.jsonl").exists()
