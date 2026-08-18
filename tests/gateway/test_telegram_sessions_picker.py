"""Tests for the Telegram /sessions inline-keyboard picker.

Mirrors `test_telegram_model_picker.py` and `test_telegram_clarify_buttons.py`
for the new ``send_sessions_picker`` and ``se:/sg:/sx`` callback dispatch
added in feat/telegram-sessions-picker.

The runner-side IDOR guard (``_resume_target_allowed``) and the full session
switch path (``_resume_session_by_id``) are exercised in
``test_sessions_command_picker_integration.py``; this file covers the
adapter-only behavior:

- keyboard construction (buttons, pagination, cancel)
- state stored per (chat_id, msg_id, thread_id) — collision-safe
- callback rejects stale msg_id (click on an old keyboard after a new
  /sessions has replaced it)
- callback rejects unauthorized taps (co-member in a shared group)
- callback pops state on success / cancel / completion
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    # Default: every callback tester is authorized, so the test focuses on the
    # picker-specific state machine (collision, IDOR, pagination). Tests that
    # need to verify the unauthorized path explicitly override this.
    adapter._is_callback_user_authorized = lambda *_a, **_kw: True
    return adapter


def _make_adapter_unauthorized():
    """Adapter variant where every callback is rejected at the auth gate."""
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = lambda *_a, **_kw: False
    return adapter


def _make_query(*, chat_id=12345, msg_id=101, thread_id=None, user_id="777"):
    """Build a minimal callback_query mock with the attrs the adapter reads."""
    query = AsyncMock()
    query.message = MagicMock()
    query.message.chat_id = chat_id
    query.message.message_id = msg_id
    query.message.message_thread_id = thread_id
    query.message.chat = MagicMock()
    query.message.chat.type = "private"
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


class TestSessionsPickerState:
    """State-key contract: (chat_id, msg_id, thread_id) — collision-safe."""

    def test_composite_key_includes_thread_id_for_forum_safety(self):
        """Two pickers in the same chat, different topics, must not collide."""
        adapter = _make_adapter()
        k1 = TelegramAdapter._sessions_picker_key("12345", 100, "777")
        k2 = TelegramAdapter._sessions_picker_key("12345", 100, "888")
        # Different thread → different key, no overwrite
        assert k1 != k2
        # None thread normalizes to "" so the key is hashable and uniform
        k3 = TelegramAdapter._sessions_picker_key("12345", 100, None)
        k4 = TelegramAdapter._sessions_picker_key("12345", 100, "")
        assert k3 == k4

    def test_same_chat_different_msg_id_does_not_collide(self):
        """A second picker opened in the same chat/thread must not overwrite
        the first picker's state — the sweeper flagged this as the stale
        picker bug class."""
        adapter = _make_adapter()
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 100, "")
        ] = {"sessions": [{"id": "a"}], "on_session_selected": AsyncMock()}
        # Second picker arrives in the same chat/thread (different msg_id)
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 200, "")
        ] = {"sessions": [{"id": "b"}], "on_session_selected": AsyncMock()}
        assert len(adapter._sessions_picker_state) == 2
        # The first picker's state is still intact
        first = adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 100, "")
        ]
        assert first["sessions"][0]["id"] == "a"


class TestSendSessionsPicker:
    """send_sessions_picker — render + state."""

    @pytest.mark.asyncio
    async def test_send_sessions_picker_renders_buttons_and_cancel(self):
        adapter = _make_adapter()
        msg = SimpleNamespace(message_id=101)
        adapter._bot.send_message = AsyncMock(return_value=msg)

        callback = AsyncMock()
        result = await adapter.send_sessions_picker(
            chat_id="12345",
            sessions=[
                {"id": "s1", "title": "Research", "preview": "notes"},
                {"id": "s2", "title": "Coding", "preview": "fix bug"},
            ],
            current_session_id="current",
            on_session_selected=callback,
            metadata=None,
        )

        assert result.success is True
        assert result.message_id == "101"
        # State stored under composite key
        state = adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 101, "")
        ]
        assert [s["id"] for s in state["sessions"]] == ["s1", "s2"]
        assert state["on_session_selected"] is callback
        assert state["current_session_id"] == "current"

    @pytest.mark.asyncio
    async def test_session_picker_label_includes_preview(self):
        """The button label must include the preview (first message) so the
        user can disambiguate titled-but-similar sessions — the text picker
        already shows both. Telegram button text supports newlines.
        """
        adapter = _make_adapter()

        # Title + preview → both shown, title on its own line
        label = adapter._session_picker_label(
            {"id": "s1", "title": "Research", "preview": "need to investigate MCP"}
        )
        assert "Research" in label
        assert "need to investigate MCP" in label
        # First line is the title (Telegram buttons render \n)
        assert label.split("\n", 1)[0] == "Research"

        # Preview only (no title) → preview is the label
        label = adapter._session_picker_label(
            {"id": "s2", "title": "", "preview": "fix login bug"}
        )
        assert label == "fix login bug"

        # Title that matches the preview → no duplicate
        label = adapter._session_picker_label(
            {"id": "s3", "title": "Same", "preview": "Same"}
        )
        assert label == "Same"

        # Long preview → truncated to 40 chars + "..."
        long_preview = "x" * 100
        label = adapter._session_picker_label(
            {"id": "s4", "title": "T", "preview": long_preview}
        )
        # "T\n— x*40..." → 1 + 2 + 40 + 3 = 46 chars
        assert label.endswith("...")
        assert len(label) <= 64  # Telegram button text cap

        # Button label exceeds Telegram cap → truncated
        label = adapter._session_picker_label(
            {"id": "s5", "title": "A" * 200, "preview": "p"}
        )
        assert len(label) <= 64
        assert label.endswith("...")

    @pytest.mark.asyncio
    async def test_send_sessions_picker_returns_error_on_empty_sessions(self):
        adapter = _make_adapter()
        result = await adapter.send_sessions_picker(
            chat_id="12345",
            sessions=[],
            current_session_id="c",
            on_session_selected=AsyncMock(),
        )
        assert result.success is False
        assert "No sessions" in result.error

    @pytest.mark.asyncio
    async def test_send_sessions_picker_paginates_past_page_size(self):
        """More than _SESSIONS_PAGE_SIZE sessions → Prev/Next nav row appears."""
        adapter = _make_adapter()
        msg = SimpleNamespace(message_id=200)
        adapter._bot.send_message = AsyncMock(return_value=msg)

        # 15 sessions → 2 pages
        sessions = [
            {"id": f"s{i}", "title": f"S{i}", "preview": ""}
            for i in range(15)
        ]
        await adapter.send_sessions_picker(
            chat_id="12345",
            sessions=sessions,
            current_session_id="c",
            on_session_selected=AsyncMock(),
        )

        kwargs = adapter._bot.send_message.call_args[1]
        text = kwargs["text"]
        # Page info suffix (N–M of T) — the picker is on page 1 of 2, so the
        # first 8 entries are listed.
        assert "1–8" in text or "1-8" in text
        assert "15" in text  # "of 15"
        # Markup is built (one-button-per-row + nav + cancel)
        assert kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_send_sessions_picker_keeps_state_per_topic(self):
        """Two /sessions in the same chat but different topics → separate state."""
        adapter = _make_adapter()
        msg1 = SimpleNamespace(message_id=101)
        msg2 = SimpleNamespace(message_id=202)
        adapter._bot.send_message = AsyncMock(side_effect=[msg1, msg2])

        await adapter.send_sessions_picker(
            chat_id="12345",
            sessions=[{"id": "s1", "title": "T1", "preview": ""}],
            current_session_id="c",
            on_session_selected=AsyncMock(),
            metadata={"thread_id": "777"},
        )
        await adapter.send_sessions_picker(
            chat_id="12345",
            sessions=[{"id": "s2", "title": "T2", "preview": ""}],
            current_session_id="c",
            on_session_selected=AsyncMock(),
            metadata={"thread_id": "888"},
        )

        # Two distinct entries — neither overwrites the other
        k1 = TelegramAdapter._sessions_picker_key("12345", 101, "777")
        k2 = TelegramAdapter._sessions_picker_key("12345", 202, "888")
        assert k1 in adapter._sessions_picker_state
        assert k2 in adapter._sessions_picker_state
        assert adapter._sessions_picker_state[k1]["sessions"][0]["id"] == "s1"
        assert adapter._sessions_picker_state[k2]["sessions"][0]["id"] == "s2"

    @pytest.mark.asyncio
    async def test_picker_filters_current_session(self, monkeypatch):
        """The session the user is already on doesn't need a button — it
        would just self-resume. The adapter filters it out before building
        the keyboard; the on-disk sessions list is preserved so the
        callback still indexes correctly."""
        adapter = _make_adapter()
        msg = SimpleNamespace(message_id=101)
        adapter._bot.send_message = AsyncMock(return_value=msg)

        captured_rows = []
        captured_buttons = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: captured_buttons.append((text, callback_data)) or text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            lambda rows: captured_rows.extend(rows) or rows,
        )

        await adapter.send_sessions_picker(
            chat_id="12345",
            sessions=[
                {"id": "current", "title": "Current", "preview": ""},
                {"id": "s1", "title": "S1", "preview": ""},
                {"id": "s2", "title": "S2", "preview": ""},
            ],
            current_session_id="current",
            on_session_selected=AsyncMock(),
        )

        # The state still has the full list so the callback can index it.
        state = adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 101, "")
        ]
        assert len(state["sessions"]) == 3
        # The rendered keyboard only has the 2 non-current rows + Cancel.
        rendered_labels = [b[0] for b in captured_buttons]
        assert "Current" not in rendered_labels
        assert "S1" in rendered_labels
        assert "S2" in rendered_labels
        assert "✗ Cancel" in rendered_labels
        # The captured rows are 2 single-button rows + 1 cancel row, no
        # pagination row (only 2 items fits in one page).
        assert len(captured_rows) == 3


class TestSessionsPickerCallback:
    """_handle_sessions_picker_callback — IDOR, stash, navigation."""

    @pytest.mark.asyncio
    async def test_select_invokes_runner_callback_with_session_id(self):
        adapter = _make_adapter()
        cb = AsyncMock(return_value="Resumed **Research** (12 messages).")
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 101, "")
        ] = {
            "sessions": [{"id": "s1", "title": "Research"}],
            "current_session_id": "c",
            "on_session_selected": cb,
            "thread_id": "",
        }

        query = _make_query(chat_id=12345, msg_id=101, user_id="777")
        await adapter._handle_sessions_picker_callback(
            query, "se:0", "12345", 101, ""
        )

        # Runner callback got the picked session id
        cb.assert_awaited_once_with("s1")
        # State popped — session-switch is one-shot
        assert (
            TelegramAdapter._sessions_picker_key("12345", 101, "")
            not in adapter._sessions_picker_state
        )
        # Message got replaced with the runner's result
        query.edit_message_text.assert_awaited()

    @pytest.mark.asyncio
    async def test_stale_msg_id_is_rejected_without_invoking_callback(self):
        """A click on an old keyboard after a new /sessions has replaced it
        must NOT invoke the runner — the sweeper flagged this as the stale
        picker bug class."""
        adapter = _make_adapter()
        cb = AsyncMock()
        # Current state is for msg_id=200, NOT 101
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 200, "")
        ] = {
            "sessions": [{"id": "s1", "title": "Research"}],
            "current_session_id": "c",
            "on_session_selected": cb,
            "thread_id": "",
        }

        # Query claims msg_id=101 (stale)
        query = _make_query(chat_id=12345, msg_id=101, user_id="777")
        await adapter._handle_sessions_picker_callback(
            query, "se:0", "12345", 101, ""
        )

        cb.assert_not_awaited()
        # Picker answers "expired"
        answer_text = (
            query.answer.call_args[1].get("text", "")
            if query.answer.call_args
            else ""
        )
        assert "expired" in answer_text.lower()
        # The current state (msg_id=200) is untouched
        assert (
            TelegramAdapter._sessions_picker_key("12345", 200, "")
            in adapter._sessions_picker_state
        )

    @pytest.mark.asyncio
    async def test_unauthorized_caller_rejected_without_invoking_callback(self):
        """Co-member in a shared group tapping someone else's session picker
        must be rejected at the adapter gate (mirrors approval / choice
        picker pattern). The runner-side IDOR guard is a second line of
        defense but the adapter should reject cheaply first."""
        adapter = _make_adapter_unauthorized()
        cb = AsyncMock()
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 101, "")
        ] = {
            "sessions": [{"id": "s1", "title": "Research"}],
            "current_session_id": "c",
            "on_session_selected": cb,
            "thread_id": "",
        }

        query = _make_query(chat_id=12345, msg_id=101, user_id="999")
        await adapter._handle_sessions_picker_callback(
            query, "se:0", "12345", 101, ""
        )

        cb.assert_not_awaited()
        answer_text = (
            query.answer.call_args[1].get("text", "")
            if query.answer.call_args
            else ""
        )
        assert "not authorized" in answer_text.lower()
        # State is preserved for the authorized caller
        assert (
            TelegramAdapter._sessions_picker_key("12345", 101, "")
            in adapter._sessions_picker_state
        )

    @pytest.mark.asyncio
    async def test_out_of_range_session_index_rejected(self):
        adapter = _make_adapter()
        cb = AsyncMock()
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 101, "")
        ] = {
            "sessions": [{"id": "s1", "title": "Only"}],
            "current_session_id": "c",
            "on_session_selected": cb,
            "thread_id": "",
        }

        query = _make_query(chat_id=12345, msg_id=101, user_id="777")
        await adapter._handle_sessions_picker_callback(
            query, "se:99", "12345", 101, ""
        )

        cb.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pagination_updates_message_with_next_page(self):
        adapter = _make_adapter()
        sessions = [
            {"id": f"s{i}", "title": f"S{i}", "preview": ""}
            for i in range(15)
        ]
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 101, "")
        ] = {
            "sessions": sessions,
            "current_session_id": "c",
            "on_session_selected": AsyncMock(),
            "thread_id": "",
        }

        query = _make_query(chat_id=12345, msg_id=101, user_id="777")
        await adapter._handle_sessions_picker_callback(
            query, "sg:1", "12345", 101, ""
        )

        # Message re-rendered with the second page
        query.edit_message_text.assert_awaited_once()
        edit_kwargs = query.edit_message_text.call_args[1]
        # Page info reflects page 2 (entries 9-15)
        assert "9–15" in edit_kwargs["text"] or "9-15" in edit_kwargs["text"]
        # reply_markup is the new keyboard
        assert edit_kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_cancel_pops_state_and_edits_message(self):
        adapter = _make_adapter()
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 101, "")
        ] = {
            "sessions": [{"id": "s1", "title": "Research"}],
            "current_session_id": "c",
            "on_session_selected": AsyncMock(),
            "thread_id": "",
        }

        query = _make_query(chat_id=12345, msg_id=101, user_id="777")
        await adapter._handle_sessions_picker_callback(
            query, "sx", "12345", 101, ""
        )

        assert (
            TelegramAdapter._sessions_picker_key("12345", 101, "")
            not in adapter._sessions_picker_state
        )
        query.edit_message_text.assert_awaited()

    @pytest.mark.asyncio
    async def test_noop_page_indicator_is_silent(self):
        adapter = _make_adapter()
        adapter._sessions_picker_state[
            TelegramAdapter._sessions_picker_key("12345", 101, "")
        ] = {
            "sessions": [{"id": "s1", "title": "Research"}],
            "current_session_id": "c",
            "on_session_selected": AsyncMock(),
            "thread_id": "",
        }

        query = _make_query(chat_id=12345, msg_id=101, user_id="777")
        await adapter._handle_sessions_picker_callback(
            query, "sx:noop", "12345", 101, ""
        )

        # No edit — just an empty answer
        query.edit_message_text.assert_not_awaited()
        query.answer.assert_awaited()
        # State preserved (the picker is still active)
        assert (
            TelegramAdapter._sessions_picker_key("12345", 101, "")
            in adapter._sessions_picker_state
        )

    @pytest.mark.asyncio
    async def test_no_state_means_expired_picker(self):
        adapter = _make_adapter()
        query = _make_query(chat_id=12345, msg_id=999, user_id="777")
        await adapter._handle_sessions_picker_callback(
            query, "se:0", "12345", 999, ""
        )

        query.answer.assert_awaited()
        answer_text = (
            query.answer.call_args[1].get("text", "")
            if query.answer.call_args
            else ""
        )
        assert "expired" in answer_text.lower()
        query.edit_message_text.assert_not_awaited()
