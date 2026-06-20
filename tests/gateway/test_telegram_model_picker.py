"""Tests for Telegram model picker thread fallback."""

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
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramModelPicker:
    @pytest.mark.asyncio
    async def test_send_model_picker_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=101)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}
            ],
            current_model="model_1",
            current_provider="provider_one",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "provider\\_one" in sent["text"]
        assert "`model_1`" in sent["text"]

    @pytest.mark.asyncio
    async def test_back_button_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        adapter._model_picker_state["12345"] = {
            "providers": [{"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}],
            "current_model": "model_1",
            "current_provider": "provider_one",
            "session_key": "s",
            "on_model_selected": AsyncMock(),
            "msg_id": 42,
        }

        query = AsyncMock()
        query.data = "mb"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mb", "12345")

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "provider\\_one" in edit_kwargs["text"]
        assert "`model_1`" in edit_kwargs["text"]

    @pytest.mark.asyncio
    async def test_model_selected_edits_message_on_success(self):
        """Regression: the mm: (model selected → switch) success path must
        edit the picker message to show the confirmation and remove the
        buttons.  An earlier revision of this PR over-indented the
        edit_message_text block so it lived inside the except branch and
        only fired when the callback raised."""
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Switched to `gpt-5`")
        adapter._model_picker_state["12345"] = {
            "providers": [
                {"slug": "openai", "name": "OpenAI", "total_models": 1, "is_current": True}
            ],
            "current_model": "model_1",
            "current_provider": "openai",
            "session_key": "s",
            "on_model_selected": callback,
            "selected_provider": "openai",
            "model_list": ["gpt-5"],
            "msg_id": 42,
        }

        query = AsyncMock()
        query.data = "mm:0"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mm:0", "12345")

        callback.assert_awaited_once()
        query.edit_message_text.assert_awaited()
        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "`gpt-5`" in edit_kwargs["text"]
        assert "12345" not in adapter._model_picker_state

    def test_provider_keyboard_preserves_mixed_case_custom_provider_slug(self, monkeypatch):
        """Custom providers with mixed-case slugs still render in /model.

        group_providers() lowercases ungrouped slug rows for canonical matching;
        the Telegram keyboard must still resolve that row back to the original
        provider entry so the picker shows the provider and preserves the
        configured slug in callback_data.
        """
        import gateway.platforms.telegram as tg

        built: list = []

        class _RecordingButton:
            def __init__(self, text, callback_data=None, **kw):
                self.text = text
                self.callback_data = callback_data
                built.append(callback_data)

        class _RecordingMarkup:
            def __init__(self, rows):
                self.inline_keyboard = rows

        monkeypatch.setattr(tg, "InlineKeyboardButton", _RecordingButton)
        monkeypatch.setattr(tg, "InlineKeyboardMarkup", _RecordingMarkup)

        adapter = _make_adapter()

        adapter._build_provider_keyboard([
            {"slug": "BitFun", "name": "BitFun", "total_models": 2},
        ])

        assert "mp:BitFun" in built
        assert "mx" in built

    @pytest.mark.asyncio
    async def test_provider_group_folds_and_drills_down(self, monkeypatch):
        """A provider family (e.g. MiniMax) collapses to one mpg: button at
        the top level; tapping it expands to its authenticated members as
        mp: buttons. A group reduced to a single authenticated member shows
        no submenu (direct mp: button).

        Inspects callback_data by recording every InlineKeyboardButton built,
        which is robust to whether `telegram` is the real SDK or the module
        mock (the SDK markup objects don't expose a plain iterable under the
        mock)."""
        import gateway.platforms.telegram as tg

        built: list = []

        class _RecordingButton:
            def __init__(self, text, callback_data=None, **kw):
                self.text = text
                self.callback_data = callback_data
                built.append(callback_data)

        class _RecordingMarkup:
            def __init__(self, rows):
                self.inline_keyboard = rows

        monkeypatch.setattr(tg, "InlineKeyboardButton", _RecordingButton)
        monkeypatch.setattr(tg, "InlineKeyboardMarkup", _RecordingMarkup)

        adapter = _make_adapter()

        async def mock_send_message(**kwargs):
            return SimpleNamespace(message_id=101)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        providers = [
            {"slug": "minimax", "name": "MiniMax", "total_models": 2},
            {"slug": "minimax-cn", "name": "MiniMax (China)", "total_models": 3},
            {"slug": "xai", "name": "xAI", "total_models": 1},
        ]

        await adapter.send_model_picker(
            chat_id="12345",
            providers=providers,
            current_model="m",
            current_provider="minimax",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata=None,
        )

        assert "mpg:minimax" in built
        assert "mp:xai" in built
        assert "mp:minimax" not in built
        assert "mp:minimax-cn" not in built

        built.clear()
        query = AsyncMock()
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mpg:minimax", "12345")

        assert "mp:minimax" in built
        assert "mp:minimax-cn" in built
        assert "mb" in built

    @pytest.mark.asyncio
    async def test_expensive_model_requires_confirmation(self, monkeypatch):
        adapter = _make_adapter()
        callback = AsyncMock(return_value="Switched to `openai/gpt-5.5-pro`")
        adapter._model_picker_state["12345"] = {
            "providers": [
                {"slug": "openrouter", "name": "OpenRouter", "total_models": 1, "is_current": True}
            ],
            "current_model": "model_1",
            "current_provider": "openrouter",
            "session_key": "s",
            "on_model_selected": callback,
            "selected_provider": "openrouter",
            "model_list": ["openai/gpt-5.5-pro"],
            "msg_id": 42,
        }
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *_args, **_kwargs: SimpleNamespace(
                message="!!! EXPENSIVE MODEL WARNING !!!\ndid you mean to select openai/gpt-5.5?"
            ),
        )

        query = AsyncMock()
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mm:0", "12345")

        callback.assert_not_awaited()
        assert "12345" in adapter._model_picker_state
        first_edit = query.edit_message_text.call_args[1]
        assert "EXPENSIVE MODEL WARNING" in first_edit["text"]
        assert first_edit["reply_markup"] is not None

        await adapter._handle_model_picker_callback(query, "mc:0", "12345")

        callback.assert_awaited_once_with("12345", "openai/gpt-5.5-pro", "openrouter")
        assert "12345" not in adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_retries_without_thread_when_thread_not_found(self):
        adapter = _make_adapter()
        providers = [{"slug": "openai", "name": "OpenAI", "total_models": 2, "is_current": True}]
        call_log = []

        class FakeBadRequest(Exception):
            pass

        async def mock_send_message(**kwargs):
            call_log.append(dict(kwargs))
            if kwargs.get("message_thread_id") is not None:
                raise FakeBadRequest("Message thread not found")
            return SimpleNamespace(message_id=99)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=providers,
            current_model="gpt-5",
            current_provider="openai",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success is True
        assert len(call_log) == 2
        assert call_log[0]["message_thread_id"] == 99999
        assert "message_thread_id" not in call_log[1] or call_log[1]["message_thread_id"] is None
