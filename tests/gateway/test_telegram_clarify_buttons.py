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
    mod.error.Forbidden = type("Forbidden", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter
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
        assert kwargs["parse_mode"] == telegram_adapter.ParseMode.MARKDOWN_V2
        assert "Which option?" in kwargs["text"]
        # Full option text rendered in the message body (not just buttons)
        assert "1\\. alpha" in kwargs["text"]
        assert "2\\. beta" in kwargs["text"]
        assert "3\\. gamma" in kwargs["text"]
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
        assert kwargs["parse_mode"] == telegram_adapter.ParseMode.MARKDOWN_V2
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
    async def test_choice_labels_use_full_text_with_compact_oversize_fallback(
        self,
        monkeypatch,
    ):
        """Readable choices label buttons directly unless the label is oversized."""
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 102
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        buttons = []

        class _RecordingButton:
            def __init__(self, text, *, callback_data, style=None):
                self.text = text
                self.callback_data = callback_data
                self.style = style
                buttons.append(self)

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _RecordingButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        short_choice = "Use the existing configuration"
        boundary_choice = "y" * 64
        long_choice = "x" * 65
        result = await adapter.send_clarify(
            chat_id="12345",
            question="?",
            choices=[short_choice, boundary_choice, long_choice, ""],
            clarify_id="cid4",
            session_key="sk4",
        )
        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        assert short_choice in kwargs["text"]
        assert boundary_choice in kwargs["text"]
        assert long_choice in kwargs["text"]
        assert [button.text for button in buttons] == [
            short_choice,
            boundary_choice,
            "3",
            "4",
            "✏️ Other (type answer)",
        ]

    @pytest.mark.asyncio
    async def test_choice_label_limit_uses_utf16_units_for_astral_emoji(
        self,
        monkeypatch,
    ):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 108
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        class _RecordingButton:
            def __init__(self, text, *, callback_data, style=None):
                self.text = text
                self.callback_data = callback_data
                self.style = style

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _RecordingButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        boundary_choice = "😀" * 32
        over_limit_choice = f"{boundary_choice}a"
        assert telegram_adapter.utf16_len(boundary_choice) == 64
        assert telegram_adapter.utf16_len(over_limit_choice) == 65

        result = await adapter.send_clarify(
            chat_id="12345",
            question="Choose an option",
            choices=[boundary_choice, over_limit_choice],
            clarify_id="cid-utf16-label",
            session_key="sk-utf16-label",
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args.kwargs
        buttons = [row[0] for row in kwargs["reply_markup"]]
        assert [(button.text, button.callback_data) for button in buttons] == [
            (boundary_choice, "cl:cid-utf16-label:0"),
            ("2", "cl:cid-utf16-label:1"),
            ("✏️ Other (type answer)", "cl:cid-utf16-label:other"),
        ]

    @pytest.mark.asyncio
    async def test_choice_button_styles_use_native_style_when_supported(
        self,
        monkeypatch,
    ):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 103
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        buttons = []

        class _StyleAwareButton:
            def __init__(self, text, *, callback_data, style=None):
                self.text = text
                self.callback_data = callback_data
                self.style = style
                buttons.append(self)

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _StyleAwareButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        choices = [
            "✅ Apply the change",
            "❌ Cancel the change",
            "✏️ Revise the change",
            "Keep reviewing",
        ]
        result = await adapter.send_clarify(
            chat_id="12345",
            question="Choose an action",
            choices=choices,
            clarify_id="cid-style",
            session_key="sk-style",
        )

        assert result.success is True
        assert [(button.text, button.style) for button in buttons] == [
            (choices[0], "success"),
            (choices[1], "danger"),
            (choices[2], "primary"),
            (choices[3], None),
            ("✏️ Other (type answer)", "primary"),
        ]

    @pytest.mark.asyncio
    async def test_choice_button_styles_use_api_kwargs_without_native_style(
        self,
        monkeypatch,
    ):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 104
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        buttons = []

        class _ApiKwargsButton:
            def __init__(self, text, *, callback_data, api_kwargs=None):
                self.text = text
                self.callback_data = callback_data
                self.api_kwargs = api_kwargs
                buttons.append(self)

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _ApiKwargsButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        choices = ["✅ Apply the change", "Keep reviewing"]
        result = await adapter.send_clarify(
            chat_id="12345",
            question="Choose an action",
            choices=choices,
            clarify_id="cid-api-kwargs",
            session_key="sk-api-kwargs",
        )

        assert result.success is True
        assert [(button.text, button.api_kwargs) for button in buttons] == [
            (choices[0], {"style": "success"}),
            (choices[1], None),
            ("✏️ Other (type answer)", {"style": "primary"}),
        ]

    @pytest.mark.asyncio
    async def test_style_rejection_retries_once_with_equivalent_unstyled_keyboard(
        self,
        monkeypatch,
    ):
        from telegram.error import BadRequest

        adapter = _make_adapter()
        attempts = []

        class _ApiKwargsButton:
            def __init__(self, text, *, callback_data, api_kwargs=None):
                self.text = text
                self.callback_data = callback_data
                self.api_kwargs = api_kwargs

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _ApiKwargsButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        async def _send_message(**kwargs):
            attempts.append(dict(kwargs))
            if len(attempts) == 1:
                raise BadRequest('unknown field "style"')
            mock_msg = MagicMock()
            mock_msg.message_id = 107
            return mock_msg

        adapter._bot.send_message = AsyncMock(side_effect=_send_message)
        choices = ["✅ Apply the change", "Keep reviewing"]

        result = await adapter.send_clarify(
            chat_id="12345",
            question="Choose an action",
            choices=choices,
            clarify_id="cid-style-fallback",
            session_key="sk-style-fallback",
            metadata={"thread_id": "321"},
        )

        assert result.success is True
        assert result.message_id == "107"
        assert len(attempts) == 2
        assert {
            key: value for key, value in attempts[0].items() if key != "reply_markup"
        } == {key: value for key, value in attempts[1].items() if key != "reply_markup"}
        assert attempts[0]["message_thread_id"] == 321
        assert attempts[1]["message_thread_id"] == 321

        first_buttons = [row[0] for row in attempts[0]["reply_markup"]]
        retry_buttons = [row[0] for row in attempts[1]["reply_markup"]]
        expected_buttons = [
            (choices[0], "cl:cid-style-fallback:0"),
            (choices[1], "cl:cid-style-fallback:1"),
            ("✏️ Other (type answer)", "cl:cid-style-fallback:other"),
        ]
        assert [
            (button.text, button.callback_data) for button in first_buttons
        ] == expected_buttons
        assert [
            (button.text, button.callback_data) for button in retry_buttons
        ] == expected_buttons
        assert [button.api_kwargs for button in first_buttons] == [
            {"style": "success"},
            None,
            {"style": "primary"},
        ]
        assert all(button.api_kwargs is None for button in retry_buttons)
        assert adapter._clarify_state["cid-style-fallback"] == "sk-style-fallback"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error_name", "message"),
        [
            ("BadRequest", "chat not found"),
            ("NetworkError", "temporary connection failure"),
            ("Forbidden", "bot is not authorized"),
        ],
    )
    async def test_unrelated_send_failures_do_not_retry(
        self,
        monkeypatch,
        error_name,
        message,
    ):
        from telegram import error as telegram_error

        adapter = _make_adapter()

        class _ApiKwargsButton:
            def __init__(self, text, *, callback_data, api_kwargs=None):
                self.text = text
                self.callback_data = callback_data
                self.api_kwargs = api_kwargs

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _ApiKwargsButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)
        error_type = getattr(telegram_error, error_name)
        adapter._bot.send_message = AsyncMock(side_effect=error_type(message))

        result = await adapter.send_clarify(
            chat_id="12345",
            question="Choose an action",
            choices=["✅ Apply the change"],
            clarify_id="cid-no-retry",
            session_key="sk-no-retry",
        )

        assert result.success is False
        assert message in result.error
        adapter._bot.send_message.assert_awaited_once()
        assert "cid-no-retry" not in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_choice_button_styles_fall_back_without_style_support(
        self,
        monkeypatch,
    ):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 105
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        buttons = []

        class _LegacyButton:
            def __init__(self, text, *, callback_data):
                self.text = text
                self.callback_data = callback_data
                buttons.append(self)

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _LegacyButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        choice = "✅ Apply the change"
        result = await adapter.send_clarify(
            chat_id="12345",
            question="Choose an action",
            choices=[choice],
            clarify_id="cid-legacy",
            session_key="sk-legacy",
        )

        assert result.success is True
        assert [button.text for button in buttons] == [
            choice,
            "✏️ Other (type answer)",
        ]

    @pytest.mark.asyncio
    async def test_complete_card_uses_standard_markdownv2_formatting(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 106
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        question = (
            "**Choose carefully** and *review* `config.yaml` with ||preview||.\n\n"
            "> Read the [guide](https://example.com/guide) first"
        )
        choices = [
            "✅ **Apply** `safe_mode`",
            "Keep [reviewing](https://example.com/review)",
        ]
        complete_card = f"❓ {question}\n\n" + "\n".join(
            f"{index + 1}. {choice}" for index, choice in enumerate(choices)
        )
        expected_text = adapter.format_message(complete_card)

        with patch.object(
            adapter,
            "format_message",
            wraps=adapter.format_message,
        ) as format_message:
            result = await adapter.send_clarify(
                chat_id="12345",
                question=question,
                choices=choices,
                clarify_id="cid5",
                session_key="sk5",
            )

        assert result.success is True
        format_message.assert_called_once_with(complete_card)
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["parse_mode"] == telegram_adapter.ParseMode.MARKDOWN_V2
        assert kwargs["text"] == expected_text
        assert "*Choose carefully*" in kwargs["text"]
        assert "_review_" in kwargs["text"]
        assert "`config.yaml`" in kwargs["text"]
        assert "||preview||" in kwargs["text"]
        assert "> Read the [guide](https://example.com/guide) first" in kwargs["text"]
        assert "1\\. ✅ *Apply* `safe_mode`" in kwargs["text"]
        assert "2\\. Keep [reviewing](https://example.com/review)" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_markdownv2_expansion_keeps_one_button_prompt_within_limit(
        self,
        monkeypatch,
    ):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 109
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        class _RecordingButton:
            def __init__(self, text, *, callback_data, style=None):
                self.text = text
                self.callback_data = callback_data
                self.style = style

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _RecordingButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        question = "." * 2500
        choices = ["Keep reviewing"]
        complete_card = f"❓ {question}\n\n1. {choices[0]}"
        formatted_card = adapter.format_message(complete_card)
        assert telegram_adapter.utf16_len(complete_card) < adapter.MAX_MESSAGE_LENGTH
        assert telegram_adapter.utf16_len(formatted_card) > adapter.MAX_MESSAGE_LENGTH

        with patch.object(
            adapter,
            "truncate_message",
            wraps=adapter.truncate_message,
        ) as truncate_message:
            result = await adapter.send_clarify(
                chat_id="12345",
                question=question,
                choices=choices,
                clarify_id="cid-expanded-card",
                session_key="sk-expanded-card",
            )

        assert result.success is True
        truncate_message.assert_called_once_with(
            complete_card,
            adapter.MAX_MESSAGE_LENGTH,
            len_fn=telegram_adapter.utf16_len,
        )
        adapter._bot.send_message.assert_awaited_once()
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert telegram_adapter.utf16_len(kwargs["text"]) <= adapter.MAX_MESSAGE_LENGTH
        assert kwargs["text"] == complete_card
        assert "parse_mode" not in kwargs
        buttons = [row[0] for row in kwargs["reply_markup"]]
        assert [(button.text, button.callback_data) for button in buttons] == [
            (choices[0], "cl:cid-expanded-card:0"),
            ("✏️ Other (type answer)", "cl:cid-expanded-card:other"),
        ]

    @pytest.mark.asyncio
    async def test_oversized_raw_card_sends_every_plain_text_chunk_in_order(
        self,
        monkeypatch,
    ):
        adapter = _make_adapter()
        sent = []

        class _RecordingButton:
            def __init__(self, text, *, callback_data, style=None):
                self.text = text
                self.callback_data = callback_data
                self.style = style

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _RecordingButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        async def _send_message(**kwargs):
            assert "cid-oversized-card" not in adapter._clarify_state
            sent.append(dict(kwargs))
            message = MagicMock()
            message.message_id = 200 + len(sent)
            return message

        adapter._bot.send_message = AsyncMock(side_effect=_send_message)
        long_choice = "A complete choice explanation " + "detail " * 900
        choices = [long_choice, "Keep reviewing"]
        complete_card = "❓ Choose an option\n\n" + "\n".join(
            f"{index + 1}. {choice}" for index, choice in enumerate(choices)
        )
        expected_chunks = adapter.truncate_message(
            complete_card,
            adapter.MAX_MESSAGE_LENGTH,
            len_fn=telegram_adapter.utf16_len,
        )
        assert len(expected_chunks) > 1

        with patch.object(
            adapter,
            "truncate_message",
            wraps=adapter.truncate_message,
        ) as truncate_message:
            result = await adapter.send_clarify(
                chat_id="12345",
                question="Choose an option",
                choices=choices,
                clarify_id="cid-oversized-card",
                session_key="sk-oversized-card",
                metadata={
                    "thread_id": "321",
                    "telegram_dm_topic_reply_fallback": True,
                    "telegram_reply_to_message_id": "654",
                },
            )

        assert result.success is True
        assert result.message_id == str(200 + len(expected_chunks))
        truncate_message.assert_called_once_with(
            complete_card,
            adapter.MAX_MESSAGE_LENGTH,
            len_fn=telegram_adapter.utf16_len,
        )
        assert [attempt["text"] for attempt in sent] == expected_chunks
        assert all("parse_mode" not in attempt for attempt in sent)
        assert all(attempt["message_thread_id"] == 321 for attempt in sent)
        assert all(attempt["reply_to_message_id"] == 654 for attempt in sent)
        assert all("reply_markup" not in attempt for attempt in sent[:-1])
        buttons = [row[0] for row in sent[-1]["reply_markup"]]
        assert [(button.text, button.callback_data) for button in buttons] == [
            ("1", "cl:cid-oversized-card:0"),
            (choices[1], "cl:cid-oversized-card:1"),
            ("✏️ Other (type answer)", "cl:cid-oversized-card:other"),
        ]
        assert long_choice in complete_card
        assert "1. A complete choice explanation" in "".join(expected_chunks)
        assert adapter._clarify_state["cid-oversized-card"] == "sk-oversized-card"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("first_error", ["markdown", "style"])
    async def test_markdown_and_style_retries_compose_with_keyboard(
        self,
        monkeypatch,
        first_error,
    ):
        from telegram.error import BadRequest

        adapter = _make_adapter()
        attempts = []

        class _ApiKwargsButton:
            def __init__(self, text, *, callback_data, api_kwargs=None):
                self.text = text
                self.callback_data = callback_data
                self.api_kwargs = api_kwargs

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _ApiKwargsButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        errors = {
            "markdown": BadRequest("can't parse entities"),
            "style": BadRequest('unknown field "style"'),
        }
        error_order = [first_error, "style" if first_error == "markdown" else "markdown"]

        async def _send_message(**kwargs):
            attempts.append(dict(kwargs))
            if len(attempts) <= len(error_order):
                raise errors[error_order[len(attempts) - 1]]
            message = MagicMock()
            message.message_id = 210
            return message

        adapter._bot.send_message = AsyncMock(side_effect=_send_message)
        choice = "✅ **Apply** the change"

        result = await adapter.send_clarify(
            chat_id="12345",
            question="**Choose** an action",
            choices=[choice],
            clarify_id=f"cid-{first_error}-first",
            session_key=f"sk-{first_error}-first",
            metadata={"thread_id": "321"},
        )

        assert result.success is True
        assert result.message_id == "210"
        assert len(attempts) == 3
        assert attempts[0]["parse_mode"] == telegram_adapter.ParseMode.MARKDOWN_V2
        assert "parse_mode" not in attempts[-1]
        assert attempts[-1]["text"] == telegram_adapter._strip_mdv2(
            attempts[0]["text"]
        )
        assert all(attempt["message_thread_id"] == 321 for attempt in attempts)
        expected_buttons = [
            (choice, f"cl:cid-{first_error}-first:0"),
            ("✏️ Other (type answer)", f"cl:cid-{first_error}-first:other"),
        ]
        for attempt in attempts:
            buttons = [row[0] for row in attempt["reply_markup"]]
            assert [
                (button.text, button.callback_data) for button in buttons
            ] == expected_buttons
        first_styles = [
            button.api_kwargs for row in attempts[0]["reply_markup"] for button in row
        ]
        final_styles = [
            button.api_kwargs for row in attempts[-1]["reply_markup"] for button in row
        ]
        assert first_styles == [{"style": "success"}, {"style": "primary"}]
        assert final_styles == [None, None]


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
    async def test_numeric_choice_expired_notifies_user(self):
        """Late tap after the entry was evicted (timeout) or the gateway
        restarted must surface an expiry notice, not a misleading ✓."""
        adapter = _make_adapter()
        # _clarify_state still maps the id (timeout eviction does not pop it),
        # but the clarify primitive entry is gone → resolve returns False.
        adapter._clarify_state["cidExpired"] = "sk-expired"

        query = AsyncMock()
        query.data = "cl:cidExpired:0"
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

        # User is told the prompt expired — not a misleading checkmark.
        answer_text = query.answer.call_args[1]["text"].lower()
        assert "expired" in answer_text
        edit_text = query.edit_message_text.call_args[1]["text"].lower()
        assert "expired" in edit_text or "session reset" in edit_text
        assert "/retry" in edit_text

    @pytest.mark.asyncio
    async def test_other_button_expired_notifies_user(self):
        """Tapping 'Other' after the entry was evicted must tell the user the
        prompt expired instead of silently entering text-capture mode."""
        adapter = _make_adapter()
        # No clarify primitive entry → mark_awaiting_text returns False.
        adapter._clarify_state["cidOtherExpired"] = "sk-other-expired"

        query = AsyncMock()
        query.data = "cl:cidOtherExpired:other"
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

        answer_text = query.answer.call_args[1]["text"].lower()
        assert "expired" in answer_text
        # State popped so a subsequent typed message is not mis-captured.
        assert "cidOtherExpired" not in adapter._clarify_state

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

    @pytest.mark.asyncio
    async def test_open_ended_fallback_renders_question_only(self):
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        class _Stub(BasePlatformAdapter):
            name = "stub"
            def __init__(self):
                self.sent: list = []
            async def connect(self, *, is_reconnect: bool = False): pass
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
        assert "1." not in adapter.sent[0]


# ===========================================================================
# Deferred cron decisions — nonblocking render and durable callback dispatch
# ===========================================================================


def _register_deferred_card(
    decisions,
    *,
    chat_id="1200",
    thread_id=None,
    callback_user_id="700",
    session_chat_type="dm",
    session_user_id="700",
    message_id="301",
    profile=None,
):
    from gateway.session import SessionSource, build_session_key

    session_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=session_chat_type,
        user_id=session_user_id,
        thread_id=thread_id,
        profile=profile,
    )
    session_key = build_session_key(session_source)
    record = decisions.register_cards(
        job={"id": "a1b2c3d4e5f6", "name": "Release review"},
        cards=[
            decisions.DeferredDecisionCard("Proceed?", ("Proceed", "Pause"))
        ],
        platform="telegram",
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=callback_user_id,
        context_ready=True,
        session_source=session_source.to_dict(),
        session_key=session_key,
    )[0]
    assert decisions.bind_message(record, message_id)
    return record, session_source, session_key


def _deferred_query(record, *, chat_type, message_id="301", user_id=700):
    query = AsyncMock()
    query.data = f"cd:a1b2c3d4e5f6:{record.decision_id}:0:1"
    query.message = MagicMock()
    query.message.chat_id = int(record.chat_id)
    query.message.message_id = int(message_id)
    query.message.message_thread_id = (
        int(record.thread_id) if record.thread_id is not None else None
    )
    query.message.chat.type = chat_type
    query.message.text = "Proceed?"
    query.from_user = MagicMock(id=user_id, first_name="Operator")
    return query


class TestTelegramDeferredDecision:
    @pytest.mark.asyncio
    async def test_render_reuses_rich_choice_card_styles_without_other(self, monkeypatch):
        adapter = _make_adapter()
        message = MagicMock(message_id=301)
        adapter._bot.send_message = AsyncMock(return_value=message)
        buttons = []

        class _Button:
            def __init__(self, text, *, callback_data, style=None):
                self.text = text
                self.callback_data = callback_data
                self.style = style
                buttons.append(self)

        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _Button)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", lambda rows: rows)

        result = await adapter.send_deferred_decision(
            chat_id="1200",
            question="Choose the rollout.",
            choices=["✅ Canary", "❌ Pause"],
            job_id="a1b2c3d4e5f6",
            decision_id="1122334455667788",
            card_index=1,
            metadata={"thread_id": "44"},
        )

        assert result.success is True
        assert [(b.text, b.callback_data, b.style) for b in buttons] == [
            ("✅ Canary", "cd:a1b2c3d4e5f6:1122334455667788:1:0", "success"),
            ("❌ Pause", "cd:a1b2c3d4e5f6:1122334455667788:1:1", "danger"),
        ]
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert kwargs["parse_mode"] == telegram_adapter.ParseMode.MARKDOWN_V2
        assert kwargs["message_thread_id"] == 44
        assert "Choose the rollout" in kwargs["text"]
        assert all(len(b.callback_data.encode("utf-8")) <= 64 for b in buttons)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "chat_id",
            "thread_id",
            "session_chat_type",
            "session_user_id",
            "query_chat_type",
        ),
        [
            ("-1001200", "44", "group", "700", "supergroup"),
            ("1200", None, "dm", "700", "private"),
            ("1200", "55", "thread", "system:cron", "private"),
        ],
        ids=["group-forum-topic", "dm-mirror", "new-private-continuation-topic"],
    )
    async def test_callback_routes_to_exact_persisted_session_source(
        self,
        tmp_path,
        monkeypatch,
        chat_id,
        thread_id,
        session_chat_type,
        session_user_id,
        query_chat_type,
    ):
        from cron import deferred_decisions as decisions
        from gateway.session import build_session_key

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        record, persisted_source, session_key = _register_deferred_card(
            decisions,
            chat_id=chat_id,
            thread_id=thread_id,
            session_chat_type=session_chat_type,
            session_user_id=session_user_id,
        )
        decisions._reset_for_tests()

        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter.__dict__["_is_callback_user_authorized"] = lambda *a, **k: True
        query = _deferred_query(record, chat_type=query_chat_type)
        update = MagicMock(callback_query=query)

        await adapter._handle_callback_query(update, MagicMock())

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.internal is True
        assert event.source.to_dict() == persisted_source.to_dict()
        assert build_session_key(event.source) == session_key
        assert event.metadata == {
            "trusted_deferred_cron_decision": True,
            "session_key": session_key,
        }
        assert event.text.startswith("Deferred cron decision response.\n")
        assert "untrusted user data" in event.text
        assert "job_id=a1b2c3d4e5f6" in event.text
        assert "card_index=0" in event.text
        assert "choice_index=1" in event.text
        assert 'selected_label_json="Pause"' in event.text
        assert "Release review" not in event.text
        assert not event.text.startswith("/")
        query.answer.assert_awaited_once()
        query.edit_message_text.assert_awaited_once()
        assert decisions.claim_choice(
            job_id=record.job_id,
            decision_id=record.decision_id,
            card_index=0,
            choice_index=1,
            platform="telegram",
            chat_id=chat_id,
            thread_id=thread_id,
            user_id="700",
            message_id="301",
        ) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("profile_allowlist", "global_allow_all", "authorized"),
        [
            ("700", False, True),
            ("999", True, False),
        ],
        ids=["profile-owner-global-deny", "profile-revoked-global-allow"],
    )
    async def test_multiplex_callback_uses_owning_profile_authorization(
        self,
        tmp_path,
        monkeypatch,
        profile_allowlist,
        global_allow_all,
        authorized,
    ):
        from cron import deferred_decisions as decisions
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        if global_allow_all:
            monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        else:
            monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

        profile_home = tmp_path / "profiles" / "owner"
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text(
            f"TELEGRAM_ALLOWED_USERS={profile_allowlist}\n",
            encoding="utf-8",
        )

        record, _, _ = _register_deferred_card(
            decisions,
            profile="owner",
        )
        adapter = _make_adapter()
        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        runner._profile_adapters = {"owner": {Platform.TELEGRAM: adapter}}
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        runner.pairing_stores = {"owner": runner.pairing_store}
        runner._handle_message = AsyncMock()
        adapter.set_message_handler(runner._make_profile_message_handler("owner"))
        adapter.handle_message = AsyncMock()
        adapter.set_authorization_check(
            runner._make_adapter_auth_check(
                Platform.TELEGRAM,
                profile="owner",
                profile_home=profile_home,
            )
        )
        query = _deferred_query(record, chat_type="private")

        await adapter._handle_callback_query(
            MagicMock(callback_query=query), MagicMock()
        )

        if authorized:
            adapter.handle_message.assert_awaited_once()
            assert adapter.handle_message.await_args.args[0].source.profile == "owner"
        else:
            adapter.handle_message.assert_not_awaited()
            assert "not authorized" in query.answer.await_args.kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_callback_rejects_unauthorized_without_consuming_choice(
        self, tmp_path, monkeypatch
    ):
        from cron import deferred_decisions as decisions

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        record, _, _ = _register_deferred_card(decisions)
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter.__dict__["_is_callback_user_authorized"] = lambda *a, **k: False
        query = AsyncMock()
        query.data = f"cd:a1b2c3d4e5f6:{record.decision_id}:0:0"
        query.message = MagicMock(chat_id=1200, message_thread_id=None)
        query.message.message_id = 301
        query.message.chat.type = "private"
        query.from_user = MagicMock(id=700, first_name="Operator")

        await adapter._handle_callback_query(
            MagicMock(callback_query=query), MagicMock()
        )

        adapter.handle_message.assert_not_awaited()
        assert "not authorized" in query.answer.await_args.kwargs["text"].lower()
        assert decisions.claim_choice(
            job_id="a1b2c3d4e5f6",
            decision_id=record.decision_id,
            card_index=0,
            choice_index=0,
            platform="telegram",
            chat_id="1200",
            thread_id=None,
            user_id="700",
            message_id="301",
        ) is not None

    @pytest.mark.asyncio
    async def test_callback_is_bound_to_the_exact_card_message(
        self, tmp_path, monkeypatch
    ):
        from cron import deferred_decisions as decisions

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        record, _, _ = _register_deferred_card(decisions, message_id="301")
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter.__dict__["_is_callback_user_authorized"] = lambda *a, **k: True
        query = _deferred_query(record, chat_type="private", message_id="999")

        await adapter._handle_callback_query(
            MagicMock(callback_query=query), MagicMock()
        )

        adapter.handle_message.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()
        assert decisions.claim_choice(
            job_id=record.job_id,
            decision_id=record.decision_id,
            card_index=0,
            choice_index=1,
            platform="telegram",
            chat_id="1200",
            thread_id=None,
            user_id="700",
            message_id="301",
        ) is not None

    @pytest.mark.asyncio
    async def test_dispatch_failure_releases_claim_without_marking_selected(
        self, tmp_path, monkeypatch
    ):
        from cron import deferred_decisions as decisions

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        record, _, _ = _register_deferred_card(decisions)
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock(side_effect=RuntimeError("enqueue failed"))
        adapter.__dict__["_is_callback_user_authorized"] = lambda *a, **k: True
        query = _deferred_query(record, chat_type="private")

        await adapter._handle_callback_query(
            MagicMock(callback_query=query), MagicMock()
        )

        query.edit_message_text.assert_not_awaited()
        assert "try again" in query.answer.await_args.kwargs["text"].lower()
        reclaimed = decisions.claim_choice(
            job_id=record.job_id,
            decision_id=record.decision_id,
            card_index=0,
            choice_index=1,
            platform="telegram",
            chat_id="1200",
            thread_id=None,
            user_id="700",
            message_id="301",
        )
        assert reclaimed is not None
        assert decisions.release_choice(reclaimed)

    @pytest.mark.asyncio
    async def test_cancelled_dispatch_releases_claim_without_marking_selected(
        self, tmp_path, monkeypatch
    ):
        from cron import deferred_decisions as decisions

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        record, _, _ = _register_deferred_card(decisions)
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock(side_effect=asyncio.CancelledError)
        adapter.__dict__["_is_callback_user_authorized"] = lambda *a, **k: True
        query = _deferred_query(record, chat_type="private")

        with pytest.raises(asyncio.CancelledError):
            await adapter._handle_callback_query(
                MagicMock(callback_query=query), MagicMock()
            )

        query.edit_message_text.assert_not_awaited()
        reclaimed = decisions.claim_choice(
            job_id=record.job_id,
            decision_id=record.decision_id,
            card_index=0,
            choice_index=1,
            platform="telegram",
            chat_id="1200",
            thread_id=None,
            user_id="700",
            message_id="301",
        )
        assert reclaimed is not None
        assert decisions.release_choice(reclaimed)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "callback",
        [
            "cd:bad:1122334455667788:0:0",
            "cd:a1b2c3d4e5f6:bad:0:0",
            "cd:a1b2c3d4e5f6:1122334455667788:x:0",
            "cd:a1b2c3d4e5f6:1122334455667788:0:9",
            "cd:a1b2c3d4e5f6:1122334455667788:0:0:extra",
        ],
    )
    async def test_malformed_or_stale_callback_fails_closed(self, callback):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter.__dict__["_is_callback_user_authorized"] = lambda *a, **k: True
        query = AsyncMock()
        query.data = callback
        query.message = MagicMock(chat_id=1200, message_thread_id=None)
        query.message.chat.type = "private"
        query.from_user = MagicMock(id=700, first_name="Operator")

        await adapter._handle_callback_query(
            MagicMock(callback_query=query), MagicMock()
        )

        adapter.handle_message.assert_not_awaited()
        assert any(
            word in query.answer.await_args.kwargs["text"].lower()
            for word in ("invalid", "expired", "resolved")
        )
