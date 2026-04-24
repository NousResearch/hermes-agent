from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


class _FakeBot:
    def __init__(self):
        self.sent_messages = []

    async def send_message(self, **kwargs):
        self.sent_messages.append(kwargs)
        return SimpleNamespace(message_id=123)


class _FakeQuery:
    def __init__(self):
        self.edits = []
        self.answers = []

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)

    async def answer(self, **kwargs):
        self.answers.append(kwargs)


class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


def _install_fake_telegram_keyboard(monkeypatch):
    import gateway.platforms.telegram as telegram

    monkeypatch.setattr(telegram, "InlineKeyboardButton", _FakeInlineKeyboardButton)
    monkeypatch.setattr(telegram, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)


def _make_adapter():
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="test-token")
    adapter._bot = _FakeBot()
    adapter._model_picker_state = {}
    return adapter


def _button_callback_data(markup):
    return [button.callback_data for row in markup.inline_keyboard for button in row]


@pytest.mark.asyncio
async def test_model_picker_provider_callbacks_distinguish_duplicate_provider_slugs(monkeypatch):
    """Telegram picker entries may share a provider slug for UX groupings.

    The OpenRouter free-model group intentionally uses slug ``openrouter`` so
    selection resolves through the real provider. Telegram callback data must
    therefore identify the picker row, not just the provider slug, otherwise the
    free group opens the regular OpenRouter model list.
    """
    _install_fake_telegram_keyboard(monkeypatch)
    adapter = _make_adapter()
    providers = [
        {
            "name": "OpenRouter",
            "slug": "openrouter",
            "models": ["anthropic/claude-sonnet-4"],
            "total_models": 1,
            "is_current": False,
        },
        {
            "name": "OpenRouter free models",
            "slug": "openrouter",
            "models": ["meta-llama/llama-3.3-8b-instruct:free"],
            "total_models": 1,
            "is_current": False,
        },
    ]

    selected = {}

    async def on_model_selected(chat_id, model_id, provider_slug):
        selected.update({"chat_id": chat_id, "model_id": model_id, "provider_slug": provider_slug})
        return "ok"

    result = await adapter.send_model_picker(
        chat_id="123",
        providers=providers,
        current_model="anthropic/claude-sonnet-4",
        current_provider="openrouter",
        session_key="telegram:123",
        on_model_selected=on_model_selected,
    )

    assert result.success
    markup = adapter._bot.sent_messages[0]["reply_markup"]
    callback_data = _button_callback_data(markup)
    assert callback_data[:3] == ["mp:0", "mp:1", "mx"], repr(markup)
    provider_callbacks = [value for value in callback_data if value and value.startswith("mp:")]
    assert provider_callbacks == ["mp:0", "mp:1"]

    query = _FakeQuery()
    await adapter._handle_model_picker_callback(query, "mp:1", "123")

    assert adapter._model_picker_state["123"]["selected_provider"] == "openrouter"
    assert adapter._model_picker_state["123"]["selected_provider_name"] == "OpenRouter free models"
    assert adapter._model_picker_state["123"]["model_list"] == [
        "meta-llama/llama-3.3-8b-instruct:free"
    ]

    await adapter._handle_model_picker_callback(_FakeQuery(), "mm:0", "123")

    assert selected == {
        "chat_id": "123",
        "model_id": "meta-llama/llama-3.3-8b-instruct:free",
        "provider_slug": "openrouter",
    }
