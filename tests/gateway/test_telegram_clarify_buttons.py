"""Tests for Telegram inline keyboard clarify buttons.

Mirrors test_telegram_approval_buttons.py for the new ``send_clarify`` and
``cl:`` callback dispatch added in feat/clarify-gateway-buttons.
"""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
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
from gateway.config import Platform, PlatformConfig


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
        # InlineKeyboardMarkup with N+1 buttons (3 choices + Other)
        markup = kwargs["reply_markup"]
        assert markup is not None
        # Mocked InlineKeyboardMarkup — just verify it was constructed
        # with rows.  We check state instead of poking the mock structure.
        assert "cid1" in adapter._clarify_state
        assert adapter._clarify_state["cid1"] == "sk1"

    @pytest.mark.asyncio
    async def test_open_ended_no_keyboard(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 101
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_clarify(
            chat_id="12345",
            question="What is your name?",
            choices=None,
            clarify_id="cid2",
            session_key="sk2",
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        # No reply_markup means no buttons — open-ended path
        assert "reply_markup" not in kwargs
        assert "What is your name?" in kwargs["text"]
        assert adapter._clarify_state["cid2"] == "sk2"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._bot = None
        result = await adapter.send_clarify(
            chat_id="12345",
            question="?",
            choices=["a"],
            clarify_id="cid3",
            session_key="sk3",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_truncates_long_choice_label(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 102
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        long_choice = "x" * 200  # > 60 char cap
        result = await adapter.send_clarify(
            chat_id="12345",
            question="?",
            choices=[long_choice],
            clarify_id="cid4",
            session_key="sk4",
        )
        assert result.success is True
        # The truncation logic replaces with "..." past 57 chars; we don't
        # inspect the mock's button labels directly (auto-MagicMock), but
        # we can verify the call didn't raise on absurdly long input.

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
    async def test_numeric_choice_resolves_with_choice_text(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        # Pre-register a clarify entry so the callback can look up the choice text
        cm.register("cidA", "sk-cb", "Pick", ["red", "green", "blue"])
        adapter._clarify_state["cidA"] = "sk-cb"

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

    @pytest.mark.asyncio
    async def test_other_button_flips_to_text_mode(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        cm.register("cidB", "sk-cb-other", "Pick", ["x", "y"])
        adapter._clarify_state["cidB"] = "sk-cb-other"

        query = AsyncMock()
        query.data = "cl:cidB:other"
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

        # Entry should now be in text-capture mode
        pending = cm.get_pending_for_session("sk-cb-other")
        assert pending is not None
        assert pending.clarify_id == "cidB"
        assert pending.awaiting_text is True
        # State NOT popped — the user still needs to type their answer
        assert "cidB" in adapter._clarify_state
        # Entry NOT yet resolved
        with cm._lock:
            entry = cm._entries.get("cidB")
        assert entry is not None
        assert not entry.event.is_set()

    @pytest.mark.asyncio
    async def test_already_resolved(self):
        adapter = _make_adapter()
        # No state for cidGone

        query = AsyncMock()
        query.data = "cl:cidGone:0"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        # Should NOT resolve anything
        assert "already" in query.answer.call_args[1]["text"].lower()

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

    @pytest.mark.asyncio
    async def test_invalid_choice_token(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        cm.register("cidD", "sk-inv", "Q?", ["a"])
        adapter._clarify_state["cidD"] = "sk-inv"

        query = AsyncMock()
        query.data = "cl:cidD:not-a-number"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.text = "Q?"
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, context)

        with cm._lock:
            entry = cm._entries.get("cidD")
        assert entry is not None
        assert not entry.event.is_set()
        query.answer.assert_called_once()
        assert "invalid" in query.answer.call_args[1]["text"].lower()


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

            async def connect(self): pass
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

    @pytest.mark.asyncio
    async def test_open_ended_fallback_renders_question_only(self):
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        class _Stub(BasePlatformAdapter):
            name = "stub"
            def __init__(self):
                self.sent: list = []
            async def connect(self): pass
            async def disconnect(self): pass
            async def send(self, chat_id, content, **kw):
                self.sent.append(content)
                return SendResult(success=True, message_id="1")
            async def edit(self, *a, **k): return SendResult(success=False)
            async def get_history(self, *a, **k): return []
            async def get_chat_info(self, *a, **k): return {}

        adapter = _Stub()
        await adapter.send_clarify(
            chat_id="c",
            question="Free form?",
            choices=None,
            clarify_id="x",
            session_key="s",
        )
        assert "Free form?" in adapter.sent[0]
        # No numbered list — choices were empty
        assert "1." not in adapter.sent[0]
