"""Tests for Telegram rich-options clarify rendering (T3 / ticket #4).

The Telegram adapter used to silently swallow the rich ``options`` parameter
(absorbed via ``**kwargs``), so a rich clarify rendered as a blank open-ended
prompt. These tests pin the new behaviour: ``options`` is honoured as a real
keyword, renders the same numbered body + inline-button rows the simple
choices path renders, and a button tap resolves to the option's ``value``
(not its label) — mirroring what Discord returns to the agent.

Mirrors the harness in ``test_telegram_clarify_buttons.py``.
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

import plugins.platforms.telegram.adapter as _tga  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


# Capturing stand-ins for the telegram widget classes so we can assert on the
# rendered button rows and their callback_data payloads.
class _CapturedButton:
    def __init__(self, *args, **kwargs):
        self.text = args[0] if args else kwargs.get("text")
        self.callback_data = kwargs.get("callback_data")


class _CapturedMarkup:
    def __init__(self, rows):
        self.rows = rows


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
# send_clarify — rich options rendering
# ===========================================================================

class TestTelegramRichClarifyRender:
    """Rich ``options`` render the numbered body + inline buttons."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_rich_options_renders_numbered_body(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 200
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        options = [
            {"label": "Ship it", "value": "deploy"},
            {"label": "Hold", "value": "wait"},
        ]
        with patch.object(_tga, "InlineKeyboardButton", _CapturedButton), \
             patch.object(_tga, "InlineKeyboardMarkup", _CapturedMarkup):
            result = await adapter.send_clarify(
                chat_id="12345",
                question="Deploy now?",
                choices=None,
                clarify_id="rid1",
                session_key="sk1",
                options=options,
            )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        # Numbered body block built from option labels (same shape as choices)
        assert "1. Ship it" in kwargs["text"]
        assert "2. Hold" in kwargs["text"]
        assert "Deploy now?" in kwargs["text"]
        # State registered like the simple path
        assert adapter._clarify_state["rid1"] == "sk1"

    @pytest.mark.asyncio
    async def test_rich_options_renders_inline_button_rows(self):
        """N option buttons (cl:<id>:0..N-1) plus the 'Other' row = N+1 rows."""
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 201
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        options = [
            {"label": "A", "value": "a"},
            {"label": "B", "value": "b"},
            {"label": "C", "value": "c"},
        ]
        with patch.object(_tga, "InlineKeyboardButton", _CapturedButton), \
             patch.object(_tga, "InlineKeyboardMarkup", _CapturedMarkup):
            await adapter.send_clarify(
                chat_id="12345",
                question="?",
                choices=None,
                clarify_id="rid2",
                session_key="sk2",
                options=options,
            )

        kwargs = adapter._bot.send_message.call_args[1]
        markup = kwargs["reply_markup"]
        assert markup is not None
        # 3 option rows + 1 "Other" row
        assert len(markup.rows) == len(options) + 1
        # Each option row carries the index callback_data convention
        for idx in range(len(options)):
            btn = markup.rows[idx][0]
            assert btn.callback_data == f"cl:rid2:{idx}"
        # Trailing "Other" row
        other_btn = markup.rows[len(options)][0]
        assert other_btn.callback_data == "cl:rid2:other"

    @pytest.mark.asyncio
    async def test_rich_options_label_falls_back_to_value(self):
        """An option with an empty label renders its value in the body."""
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 202
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        options = [
            {"label": "", "value": "fallback-value"},
        ]
        with patch.object(_tga, "InlineKeyboardButton", _CapturedButton), \
             patch.object(_tga, "InlineKeyboardMarkup", _CapturedMarkup):
            await adapter.send_clarify(
                chat_id="12345",
                question="?",
                choices=None,
                clarify_id="rid3",
                session_key="sk3",
                options=options,
            )

        kwargs = adapter._bot.send_message.call_args[1]
        assert "1. fallback-value" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_unknown_kwarg_does_not_swallow_options(self):
        """Regression guard: an extra unknown kwarg is tolerated and does NOT
        cause ``options`` to be silently dropped (the old **kwargs bug)."""
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 203
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        options = [{"label": "Go", "value": "go"}]
        with patch.object(_tga, "InlineKeyboardButton", _CapturedButton), \
             patch.object(_tga, "InlineKeyboardMarkup", _CapturedMarkup):
            result = await adapter.send_clarify(
                chat_id="12345",
                question="?",
                choices=None,
                clarify_id="rid4",
                session_key="sk4",
                options=options,
                display_type="buttons",  # ignored on Telegram, must not break
                auth_policy="session_owner_only",  # ignored on Telegram
                some_future_kwarg=123,  # unknown — tolerated
            )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        # options were NOT swallowed — body + buttons rendered
        assert "1. Go" in kwargs["text"]
        markup = kwargs["reply_markup"]
        assert markup is not None
        assert len(markup.rows) == 2  # 1 option + Other


# ===========================================================================
# Callback dispatch — rich option tap resolves to the option's value
# ===========================================================================

class TestTelegramRichClarifyCallback:
    """Tapping a rich-option button resolves to the option's ``value``."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_tap_index_zero_resolves_to_option_value(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        options = [
            {"label": "Ship it", "value": "deploy_prod"},
            {"label": "Hold", "value": "wait"},
        ]
        # Rich entry: choices is None, options is set.
        cm.register("ridA", "sk-rich", "Deploy?", choices=None, options=options)
        adapter._clarify_state["ridA"] = "sk-rich"

        query = AsyncMock()
        query.data = "cl:ridA:0"  # first option
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.text = "Deploy?"
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

        # Agent receives the option's *value*, not its label.
        with cm._lock:
            entry = cm._entries.get("ridA")
        assert entry is not None
        assert entry.response == "deploy_prod"
        assert entry.event.is_set()
        assert "ridA" not in adapter._clarify_state
        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_tap_other_index_resolves_to_correct_value(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        options = [
            {"label": "A", "value": "v_a"},
            {"label": "B", "value": "v_b"},
            {"label": "C", "value": "v_c"},
        ]
        cm.register("ridB", "sk-rich2", "Pick", choices=None, options=options)
        adapter._clarify_state["ridB"] = "sk-rich2"

        query = AsyncMock()
        query.data = "cl:ridB:2"  # third option
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

        with cm._lock:
            entry = cm._entries.get("ridB")
        assert entry is not None
        assert entry.response == "v_c"


# ===========================================================================
# Regression — simple-choices & open-ended paths must be unchanged
# ===========================================================================

class TestTelegramRichClarifyRegression:
    """The rich-options change must not perturb the existing paths."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_simple_choices_path_unchanged(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 300
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        with patch.object(_tga, "InlineKeyboardButton", _CapturedButton), \
             patch.object(_tga, "InlineKeyboardMarkup", _CapturedMarkup):
            result = await adapter.send_clarify(
                chat_id="12345",
                question="Which?",
                choices=["alpha", "beta"],
                clarify_id="sid1",
                session_key="sk1",
            )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        assert "1. alpha" in kwargs["text"]
        assert "2. beta" in kwargs["text"]
        markup = kwargs["reply_markup"]
        assert len(markup.rows) == 3  # 2 choices + Other
        assert markup.rows[0][0].callback_data == "cl:sid1:0"
        assert markup.rows[2][0].callback_data == "cl:sid1:other"

    @pytest.mark.asyncio
    async def test_open_ended_path_unchanged(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 301
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_clarify(
            chat_id="12345",
            question="Free form?",
            choices=None,
            clarify_id="sid2",
            session_key="sk2",
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        assert "reply_markup" not in kwargs
        assert "Free form?" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_simple_choice_tap_resolves_to_choice_text(self):
        """Simple-path callback still resolves to choices[idx] (not options)."""
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        cm.register("sidA", "sk-simple", "Pick", ["red", "green", "blue"])
        adapter._clarify_state["sidA"] = "sk-simple"

        query = AsyncMock()
        query.data = "cl:sidA:1"  # green
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

        with cm._lock:
            entry = cm._entries.get("sidA")
        assert entry is not None
        assert entry.response == "green"
