"""Tests for Telegram model picker thread fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramModelPicker:
    def test_bedrock_model_buttons_show_distinguishing_suffix_and_keep_index_callback(self, monkeypatch):
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()
        model_id = "global.anthropic.claude-sonnet-4-6"

        keyboard, _ = adapter._build_model_keyboard([model_id], page=0)
        button = keyboard.inline_keyboard[0][0]

        assert button.text == "claude-sonnet-4-6"
        assert button.callback_data == "mm:0"

    @pytest.mark.parametrize("model_id,expected", [
        # Regional Bedrock routing namespaces carry the same redundancy as
        # ``global.`` (review feedback on PR #94990).
        ("us.anthropic.claude-opus-4-1-20250805-v1:0", "claude-opus-4-1-20250805-v1:0"),
        ("eu.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("apac.amazon.nova-2-lite-v1:0", "nova-2-lite-v1:0"),
        ("us-gov.anthropic.claude-haiku-4-5", "claude-haiku-4-5"),
        # Non-Bedrock IDs must pass through untouched.
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("mistral-large-latest", "mistral-large-latest"),
    ])
    def test_regional_bedrock_prefixes_are_stripped_from_labels(self, monkeypatch, model_id, expected):
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()

        keyboard, _ = adapter._build_model_keyboard([model_id], page=0)
        button = keyboard.inline_keyboard[0][0]

        assert button.text == expected
        assert button.callback_data == "mm:0"

    def test_degenerate_two_segment_id_never_yields_blank_button(self, monkeypatch):
        """``global.anthropic`` must not produce an empty label — Telegram
        rejects blank button text (BUTTON_TEXT_INVALID) and the whole picker
        reply would fail (review feedback on PR #94990)."""
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()

        keyboard, _ = adapter._build_model_keyboard(["global.anthropic"], page=0)
        button = keyboard.inline_keyboard[0][0]

        assert button.text == "global.anthropic"
        assert button.callback_data == "mm:0"

    def test_same_model_in_multiple_geo_profiles_keeps_geo_in_label(self, monkeypatch):
        """Bedrock exposes the same model behind several routing namespaces
        (``us.xai.grok-4.6`` and ``global.xai.grok-4.6`` coexist). Stripping
        the namespace from both would render two identical buttons, so when
        the stripped label collides within the list the geo segment must be
        kept as the differentiator."""
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()
        models = [
            "us.xai.grok-4.6",
            "global.xai.grok-4.6",
            "eu.anthropic.claude-sonnet-4-6",
        ]

        keyboard, _ = adapter._build_model_keyboard(models, page=0)
        flat = [b for row in keyboard.inline_keyboard for b in row]
        labels = [b.text for b in flat[:3]]

        assert labels == ["us: grok-4.6", "global: grok-4.6", "claude-sonnet-4-6"]
        # All labels on the page must be pairwise distinct.
        assert len(set(labels)) == len(labels)
        assert [b.callback_data for b in flat[:3]] == ["mm:0", "mm:1", "mm:2"]

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


