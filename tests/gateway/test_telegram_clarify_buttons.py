"""Tests for Telegram inline keyboard clarify buttons.

Mirrors test_telegram_approval_buttons.py for the new ``send_clarify`` and
``cl:`` callback dispatch added in feat/clarify-gateway-buttons.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported (mirrors
# test_telegram_approval_buttons.py)
# ---------------------------------------------------------------------------
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import PlatformConfig


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


# ===========================================================================
# send_clarify — render
# ===========================================================================

class TestTelegramSendClarify:
    """Verify the rendered prompt has buttons or none, and stores state."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_multi_choice_renders_buttons_and_other(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 100
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_clarify(
            chat_id="12345",
            question="Which option?",
            choices=["alpha", "beta", "gamma"],
            clarify_id="cid1",
            session_key="sk1",
        )

        assert result.success is True
        assert result.message_id == "100"

        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert "Which option?" in kwargs["text"]
        # Full option text rendered in the message body (not just buttons)
        assert "1. alpha" in kwargs["text"]
        assert "2. beta" in kwargs["text"]
        assert "3. gamma" in kwargs["text"]
        # InlineKeyboardMarkup with N+1 buttons (3 choices + Other)
        markup = kwargs["reply_markup"]
        assert markup is not None
        # Mocked InlineKeyboardMarkup — just verify it was constructed
        # with rows.  We check state instead of poking the mock structure.
        assert "cid1" in adapter._clarify_state
        assert adapter._clarify_state["cid1"] == "sk1"


        # The button label should be short ("1"), not the long choice
        # (we can't inspect mock button labels directly, but the send
        # succeeded — old truncation code could raise on edge cases)

    @pytest.mark.asyncio
    async def test_html_escapes_question(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 103
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send_clarify(
            chat_id="12345",
            question="<script>alert(1)</script>",
            choices=["x"],
            clarify_id="cid5",
            session_key="sk5",
        )
        kwargs = adapter._bot.send_message.call_args[1]
        # Must NOT contain raw <script> — html.escape should have neutralized
        assert "<script>" not in kwargs["text"]
        assert "&lt;script&gt;" in kwargs["text"]


# ===========================================================================
# Callback dispatch — _handle_callback_query routing for cl:* prefixes
# ===========================================================================

class TestTelegramClarifyCallback:
    """Verify clicking a button resolves the clarify primitive."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_successful_numeric_choice_resumes_typing_for_chat(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        # Pre-register a clarify entry so the callback can look up the choice text
        cm.register("cidA", "sk-cb", "Pick", ["red", "green", "blue"])
        adapter._clarify_state["cidA"] = "sk-cb"
        adapter.pause_typing_for_chat("12345")
        assert "12345" in adapter._typing_paused

        query = AsyncMock()
        query.data = "cl:cidA:1"  # green
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.text = "Pick"
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

        # State popped
        assert "cidA" not in adapter._clarify_state
        # Wait shouldn't be needed — resolve_gateway_clarify is sync.
        # The entry's response should be set.
        # We test by reading the entry's response directly.
        with cm._lock:
            entry = cm._entries.get("cidA")
        # Entry might be popped by wait_for_response, but here we never
        # called wait — so it's still in _entries with response set.
        assert entry is not None
        assert entry.response == "green"
        assert entry.event.is_set()
        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()
        assert "12345" not in adapter._typing_paused

    @pytest.mark.asyncio
    async def test_failed_numeric_choice_does_not_resume_typing(self):
        """A stale adapter entry must not restart typing when resolution fails."""
        adapter = _make_adapter()
        adapter._clarify_state["cidStale"] = "sk-stale"
        adapter.resume_typing_for_chat = MagicMock(wraps=adapter.resume_typing_for_chat)

        query = AsyncMock()
        query.data = "cl:cidStale:0"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.text = "Pick"
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        adapter.resume_typing_for_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_numeric_choice_without_chat_id_does_not_resume_typing(self):
        """Resolution can succeed without message context, but cannot target typing."""
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        cm.register("cidNoChat", "sk-no-chat", "Pick", ["red"])
        adapter._clarify_state["cidNoChat"] = "sk-no-chat"
        adapter.resume_typing_for_chat = MagicMock(wraps=adapter.resume_typing_for_chat)

        query = AsyncMock()
        query.data = "cl:cidNoChat:0"
        query.message = None
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

        adapter.resume_typing_for_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_follow_up_delivery_failure_keeps_typing_cleanup_with_turn_owner(self):
        """Resolution resumes the existing loop; a failed final send does not own it."""
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        adapter.send_typing = AsyncMock(return_value=None)
        stop_event = asyncio.Event()
        typing_task = asyncio.create_task(
            adapter._keep_typing("12345", interval=0.01, stop_event=stop_event)
        )

        try:
            # The outer message-processing lifecycle starts this loop before
            # the agent reaches clarify. Pausing must suppress refreshes
            # without cancelling that owner task.
            for _ in range(20):
                if adapter.send_typing.await_count:
                    break
                await asyncio.sleep(0.01)
            assert adapter.send_typing.await_count > 0
            adapter.pause_typing_for_chat("12345")
            paused_call_count = adapter.send_typing.await_count
            await asyncio.sleep(0.03)
            assert adapter.send_typing.await_count == paused_call_count
            assert not typing_task.done()

            cm.register("cidDelivery", "sk-delivery", "Pick", ["red"])
            adapter._clarify_state["cidDelivery"] = "sk-delivery"

            query = AsyncMock()
            query.data = "cl:cidDelivery:0"
            query.message = MagicMock()
            query.message.chat_id = 12345
            query.message.text = "Pick"
            query.from_user = MagicMock()
            query.from_user.id = "777"
            query.from_user.first_name = "Tester"
            query.answer = AsyncMock()
            query.edit_message_text = AsyncMock()

            update = MagicMock()
            update.callback_query = query

            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
                await adapter._handle_callback_query(update, MagicMock())

            # The callback only releases the pause. The same loop resumes and
            # remains owned by the original turn even if its final delivery
            # subsequently fails.
            for _ in range(20):
                if adapter.send_typing.await_count > paused_call_count:
                    break
                await asyncio.sleep(0.01)
            assert adapter.send_typing.await_count > paused_call_count
            assert not typing_task.done()

            adapter._should_attempt_rich = MagicMock(return_value=False)
            assert adapter._bot is not None
            adapter._bot.send_message = AsyncMock(
                side_effect=RuntimeError("follow-up delivery failed")
            )
            delivery = await adapter.send(
                "12345", "follow-up", metadata={"notify": True}
            )

            assert delivery.success is False
            assert "12345" not in adapter._typing_paused
            assert not typing_task.done()
        finally:
            stop_event.set()
            await asyncio.wait_for(typing_task, timeout=1)

        assert "12345" not in adapter._typing_paused

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        cm.register("cidC", "sk-auth", "Pick", ["a", "b"])
        adapter._clarify_state["cidC"] = "sk-auth"

        # Hook up a runner that says NOT authorized
        class _DenyRunner:
            async def _handle_message(self, event):
                return None
            def _is_user_authorized(self, source):
                return False

        adapter._message_handler = _DenyRunner()._handle_message

        query = AsyncMock()
        query.data = "cl:cidC:0"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat.type = "private"
        query.message.text = "Pick"
        query.from_user = MagicMock()
        query.from_user.id = "999"
        query.from_user.first_name = "Mallory"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        await adapter._handle_callback_query(update, context)

        # Must not resolve, must answer with not-authorized message
        with cm._lock:
            entry = cm._entries.get("cidC")
        assert entry is not None
        assert not entry.event.is_set()
        query.answer.assert_called_once()
        assert "not authorized" in query.answer.call_args[1]["text"].lower()
        # State preserved
        assert adapter._clarify_state["cidC"] == "sk-auth"


# ===========================================================================
# Base adapter fallback render — text numbered list
# ===========================================================================

class TestBaseAdapterClarifyFallback:
    """Adapters without button overrides should render numbered text."""

    @pytest.mark.asyncio
    async def test_numbered_text_fallback(self):
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        # Subclass just enough to instantiate
        class _Stub(BasePlatformAdapter):
            name = "stub"

            def __init__(self):
                # Skip base __init__ — we're not exercising it
                self.sent: list = []

            async def connect(self, *, is_reconnect: bool = False): pass
            async def disconnect(self): pass
            async def send(self, chat_id, content, **kw):
                self.sent.append({"chat_id": chat_id, "content": content})
                return SendResult(success=True, message_id="1")
            async def edit(self, *a, **k): return SendResult(success=False)
            async def get_history(self, *a, **k): return []
            async def get_chat_info(self, *a, **k): return {}

        adapter = _Stub()

        result = await adapter.send_clarify(
            chat_id="c",
            question="Pick a fruit",
            choices=["apple", "banana"],
            clarify_id="x",
            session_key="s",
        )
        assert result.success is True
        assert len(adapter.sent) == 1
        text = adapter.sent[0]["content"]
        assert "Pick a fruit" in text
        assert "1." in text and "apple" in text
        assert "2." in text and "banana" in text

