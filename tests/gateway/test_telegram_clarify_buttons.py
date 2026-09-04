"""Tests for Telegram inline keyboard clarify buttons.

Mirrors test_telegram_approval_buttons.py for the new ``send_clarify`` and
``cl:`` callback dispatch added in feat/clarify-gateway-buttons.
"""

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
from plugins.platforms.telegram.adapter import TelegramAdapter, _ClarifyPrompt
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
        mock_msg.chat_id = 12345
        mock_msg.message_thread_id = None
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
        # State carries the session key *and* the identity of the message that rendered the
        # buttons, so a callback can be matched against it (#102957).
        assert adapter._clarify_state["cid1"] == _ClarifyPrompt(
            session_key="sk1", chat_id="12345", message_id="100", thread_id=None)


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
    async def test_numeric_choice_resolves_with_choice_text(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        # Pre-register a clarify entry so the callback can look up the choice text
        cm.register("cidA", "sk-cb", "Pick", ["red", "green", "blue"])
        adapter._clarify_state["cidA"] = _ClarifyPrompt(
            session_key="sk-cb", chat_id="12345", message_id="100", thread_id=None)

        query = AsyncMock()
        query.data = "cl:cidA:1"  # green
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.message_id = 100
        query.message.message_thread_id = None
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
    async def test_unauthorized_user_rejected(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        cm.register("cidC", "sk-auth", "Pick", ["a", "b"])
        adapter._clarify_state["cidC"] = _ClarifyPrompt(
            session_key="sk-auth", chat_id="12345", message_id="100", thread_id=None)

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
        assert adapter._clarify_state["cidC"].session_key == "sk-auth"


# ===========================================================================
# Prompt binding — a callback may only resolve the clarify it was rendered on
# ===========================================================================

class TestTelegramClarifyPromptBinding:
    """#102957 — chat-level authorization cannot tell a deliberate tap from a callback
    that arrived on some other message, so the clarify is bound to its prompt message."""

    ASK_CHAT_ID = 12345
    ASK_MESSAGE_ID = 100

    def setup_method(self):
        _clear_clarify_state()

    def _ask(self, adapter, clarify_id="cid-bind", session_key="sk-bind"):
        """Register a pending clarify bound to (ASK_CHAT_ID, ASK_MESSAGE_ID)."""
        from tools import clarify_gateway as cm
        cm.register(clarify_id, session_key, "Recover the worker?", ["keep running", "stop the worker"])
        adapter._clarify_state[clarify_id] = _ClarifyPrompt(
            session_key=session_key, chat_id=str(self.ASK_CHAT_ID),
            message_id=str(self.ASK_MESSAGE_ID), thread_id=None)

    def _tap(self, chat_id, message_id, *, clarify_id="cid-bind", token="1"):
        query = AsyncMock()
        query.id = "cbq-1"
        query.data = f"cl:{clarify_id}:{token}"
        query.message = MagicMock()
        query.message.chat_id = chat_id
        query.message.message_id = message_id
        query.message.message_thread_id = None
        query.message.chat.type = "private"
        query.message.text = "Recover the worker?"
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Owner"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        return update, query

    async def _deliver(self, adapter, update):
        # The caller is allowed to answer prompts — authorization is not what is under test.
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())

    @pytest.mark.asyncio
    async def test_callback_from_another_message_does_not_resolve(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        self._ask(adapter)
        update, query = self._tap(self.ASK_CHAT_ID, 999)

        await self._deliver(adapter, update)

        with cm._lock:
            entry = cm._entries.get("cid-bind")
        assert not entry.event.is_set()
        assert entry.response is None
        # The real prompt stays answerable.
        assert "cid-bind" in adapter._clarify_state
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_from_another_chat_does_not_resolve(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        self._ask(adapter)
        update, query = self._tap(67890, 555)

        await self._deliver(adapter, update)

        with cm._lock:
            entry = cm._entries.get("cid-bind")
        assert not entry.event.is_set()
        assert "cid-bind" in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_mismatched_callback_does_not_flip_to_text_capture(self):
        """The "Other" branch mutates state too, so it fails closed on the same check."""
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        self._ask(adapter)
        update, query = self._tap(self.ASK_CHAT_ID, 999, token="other")

        await self._deliver(adapter, update)

        with cm._lock:
            entry = cm._entries.get("cid-bind")
        assert entry.awaiting_text is False
        assert "cid-bind" in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_genuine_tap_on_the_prompt_still_resolves(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        self._ask(adapter)
        update, query = self._tap(self.ASK_CHAT_ID, self.ASK_MESSAGE_ID)

        await self._deliver(adapter, update)

        with cm._lock:
            entry = cm._entries.get("cid-bind")
        assert entry.event.is_set()
        assert entry.response == "stop the worker"
        assert "cid-bind" not in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_prompt_without_a_bound_message_stays_answerable(self):
        """A send that reported no message id leaves nothing to match — resolve rather than
        strand the prompt, since no caller-controlled input can reach that state."""
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        cm.register("cid-unbound", "sk-unbound", "Pick", ["a", "b"])
        adapter._clarify_state["cid-unbound"] = _ClarifyPrompt(session_key="sk-unbound")
        update, query = self._tap(self.ASK_CHAT_ID, self.ASK_MESSAGE_ID, clarify_id="cid-unbound", token="0")

        await self._deliver(adapter, update)

        with cm._lock:
            entry = cm._entries.get("cid-unbound")
        assert entry.event.is_set()
        assert entry.response == "a"


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

