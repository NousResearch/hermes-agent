"""Tests for Telegram inline keyboard suggested-action buttons.

Mirrors test_telegram_clarify_buttons.py for ``send_suggested_actions`` and
the ``sa:`` callback dispatch (feat/guards-and-actions, salvaged as a split
of #77905).
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import PlatformConfig


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _clear_sa_state():
    from tools import suggested_actions_gateway as sa
    with sa._lock:
        sa._sets.clear()
        sa._session_index.clear()


class TestTelegramSendSuggestedActions:
    """Verify the rendered prompt has one button per action."""

    def setup_method(self):
        _clear_sa_state()

    @pytest.mark.asyncio
    async def test_renders_one_button_per_action(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 200
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_suggested_actions(
            chat_id="12345",
            message="Want me to do X or Y?",
            actions=["Do X", {"label": "Do Y", "payload": "please do y"}],
            set_id="setA",
            session_key="sk1",
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert "Want me to do X or Y?" in kwargs["text"]
        assert kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_html_escapes_message(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 201
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send_suggested_actions(
            chat_id="12345", message="<script>alert(1)</script>", actions=["ok"],
            set_id="setB", session_key="sk2",
        )
        kwargs = adapter._bot.send_message.call_args[1]
        assert "<script>" not in kwargs["text"]
        assert "&lt;script&gt;" in kwargs["text"]


class TestTelegramSuggestedActionCallback:
    """Verify tapping a button injects a fresh user turn with the action's payload."""

    def setup_method(self):
        _clear_sa_state()

    @pytest.mark.asyncio
    async def test_tap_injects_payload_as_new_turn(self):
        from tools import suggested_actions_gateway as sa

        adapter = _make_adapter()
        sa.register(set_id="setC", session_key="sk-cb", message="Pick one",
                    actions=["Do X", {"label": "Do Y", "payload": "please do y"}])

        received = {}

        async def _fake_handle_message(event):
            received["text"] = event.text
            received["chat_id"] = event.source.chat_id

        adapter.handle_message = _fake_handle_message

        query = AsyncMock()
        query.data = "sa:setC:1"  # "Do Y" -> payload "please do y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.text = "Pick one"
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, context)

        assert received.get("text") == "please do y"
        assert received.get("chat_id") == "12345"
        query.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_expired_set_answers_gracefully(self):
        adapter = _make_adapter()
        # No set registered for "setD" — resolve() returns None.

        query = AsyncMock()
        query.data = "sa:setD:0"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.answer = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "expired" in query.answer.call_args[1]["text"].lower()

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected(self):
        from tools import suggested_actions_gateway as sa

        adapter = _make_adapter()
        sa.register(set_id="setE", session_key="sk-auth", message="Pick one", actions=["a", "b"])

        class _DenyRunner:
            async def _handle_message(self, event):
                return None
            def _is_user_authorized(self, source):
                return False

        adapter._message_handler = _DenyRunner()._handle_message

        received = {"called": False}

        async def _fake_handle_message(event):
            received["called"] = True

        adapter.handle_message = _fake_handle_message

        query = AsyncMock()
        query.data = "sa:setE:0"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat.type = "private"
        query.from_user = MagicMock()
        query.from_user.id = "999"
        query.from_user.first_name = "Mallory"
        query.answer = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        await adapter._handle_callback_query(update, context)

        assert received["called"] is False
        query.answer.assert_called_once()
        assert "not authorized" in query.answer.call_args[1]["text"].lower()


class TestBaseAdapterSuggestedActionsFallback:
    """Adapters without button overrides render a numbered text list."""

    @pytest.mark.asyncio
    async def test_default_numbered_list_fallback(self):
        from gateway.platforms.base import BasePlatformAdapter

        class _PlainAdapter(BasePlatformAdapter):
            def __init__(self):
                self.sent = None
            async def connect(self):
                return True
            async def disconnect(self):
                pass
            async def get_chat_info(self, chat_id):
                return {}
            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.sent = content
                from gateway.platforms.base import SendResult
                return SendResult(success=True, message_id="1")

        adapter = _PlainAdapter()
        await adapter.send_suggested_actions(
            chat_id="c1", message="Next steps?", actions=["Book it", {"label": "See more"}],
            set_id="setF", session_key="sk3",
        )
        assert "Next steps?" in adapter.sent
        assert "1. Book it" in adapter.sent
        assert "2. See more" in adapter.sent
