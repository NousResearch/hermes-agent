"""Tests for Telegram model picker thread fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


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



class _Button:
    """Stand-in for ``InlineKeyboardButton`` that records what was rendered.

    ``tests/gateway/conftest.py`` installs a MagicMock ``telegram`` package when
    the real library is absent, and a MagicMock accepts every attribute access —
    so asserting on ``markup.inline_keyboard`` without this substitution passes
    vacuously whether or not the keyboard is correct. Production reads these
    names off the adapter module, so that is the seam.
    """

    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data


class _Markup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


@pytest.fixture
def rendered_keyboards(monkeypatch):
    from plugins.platforms.telegram import adapter as telegram_adapter

    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Markup)


class TestTelegramBedrockPickerNavigation:
    """Selectability of a Bedrock catalog, walked through the real callback router.

    The reported failure is functional rather than cosmetic: with indistinguishable
    labels, a tap that reaches no handler, and a hidden model count, a user cannot
    reliably switch model from Telegram. Each test walks the taps a user actually
    makes through ``_handle_callback_query`` — the global dispatcher, so a callback
    prefix missing from its table is caught.
    """

    # Same model advertised bare, regionally and globally, across more than one
    # page — the shape that produced identical buttons in #94986.
    MODELS = [
        "global.anthropic.claude-opus-5",
        "us.anthropic.claude-opus-5",
        "anthropic.claude-opus-5",
        "global.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
        "anthropic.claude-sonnet-5",
        "global.anthropic.claude-haiku-4-5",
        "us.anthropic.claude-haiku-4-5",
        "anthropic.claude-haiku-4-5",
        "amazon.nova-lite-v1:0",
        "us.amazon.nova-lite-v1:0",
    ]

    def _picker(self, models, total_models=None, extra_providers=()):
        adapter = _make_adapter()
        adapter._model_picker_state["12345"] = {
            "providers": [{"slug": "bedrock", "name": "AWS Bedrock", "models": models,
                           "total_models": total_models if total_models is not None else len(models),
                           "is_current": True}, *extra_providers],
            "current_model": models[0] if models else "",
            "current_provider": "bedrock",
            "session_key": "s",
            "on_model_selected": AsyncMock(return_value="switched"),
            "msg_id": 42,
        }
        return adapter, adapter._model_picker_state["12345"]

    @staticmethod
    async def _tap(adapter, data):
        """Send one callback tap through the global dispatcher, as Telegram does."""
        query = AsyncMock()
        query.data = data
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
        return query

    @classmethod
    async def _render(cls, adapter, data):
        """``(buttons, text)`` of the message a tap re-rendered."""
        rows, text = await cls._render_rows(adapter, data)
        return [b for row in rows for b in row], text

    @classmethod
    async def _render_rows(cls, adapter, data):
        """``(rows, text)`` — keeps the row layout, which is what mobile width costs."""
        query = await cls._tap(adapter, data)
        assert query.edit_message_text.await_count == 1, f"tap {data!r} reached no handler"
        kwargs = query.edit_message_text.call_args[1]
        markup = kwargs.get("reply_markup")
        return (markup.inline_keyboard if markup is not None else []), kwargs["text"]

    @pytest.mark.asyncio
    async def test_user_can_walk_the_picker_and_select_one_exact_model_id(self, rendered_keyboards):
        adapter, state = self._picker(self.MODELS)
        callback = state["on_model_selected"]

        vendors, _ = await self._render(adapter, "mp:bedrock")
        assert [b.callback_data for b in vendors if str(b.callback_data).startswith("mvd:")] == [
            "mvd:amazon", "mvd:anthropic"]

        page0, text0 = await self._render(adapter, "mvd:anthropic")
        labels0 = [b.text for b in page0 if str(b.callback_data).startswith("mm:")]
        assert len(set(labels0)) == len(labels0), labels0
        assert "Anthropic" in text0
        # A 9-model vendor spans pages; paging must stay reachable and keep labels distinct.
        pages = [b.callback_data for b in page0 if str(b.callback_data).startswith("mg:")]
        assert pages, "pagination row missing"
        page1, _ = await self._render(adapter, pages[-1])
        labels1 = [b.text for b in page1 if str(b.callback_data).startswith("mm:")]
        assert len(set(labels0 + labels1)) == len(labels0 + labels1)

        # Tapping a label hands the switch the exact advertised ID, unrewritten.
        chosen = next(b for b in page0 + page1 if b.text == "us: opus-5")
        await self._tap(adapter, chosen.callback_data)
        assert callback.await_args[0][1] == "us.anthropic.claude-opus-5"

    @pytest.mark.asyncio
    async def test_back_and_cancel_keep_the_picker_navigable(self, rendered_keyboards):
        adapter, _ = self._picker(self.MODELS)
        await self._render(adapter, "mp:bedrock")
        await self._render(adapter, "mvd:anthropic")

        to_vendors, _ = await self._render(adapter, "mb")
        assert "mvd:anthropic" in [b.callback_data for b in to_vendors]
        to_providers, _ = await self._render(adapter, "mb")
        assert "mp:bedrock" in [b.callback_data for b in to_providers]

        await self._tap(adapter, "mx")
        assert "12345" not in adapter._model_picker_state

    @pytest.mark.asyncio
    async def test_a_plain_provider_list_keeps_the_original_two_step_flow(self, rendered_keyboards):
        """The drill-down is inserted only where it helps; a non-Bedrock catalog
        must still land straight on selectable, verbatim-labelled buttons."""
        adapter, _ = self._picker(["gpt-4o-mini", "o3"])
        buttons, _text = await self._render(adapter, "mp:bedrock")
        picks = [(b.text, b.callback_data) for b in buttons if str(b.callback_data).startswith("mm:")]
        assert picks == [("gpt-4o-mini", "mm:0"), ("o3", "mm:1")]

    @pytest.mark.asyncio
    async def test_provider_count_and_truncation_hint_survive_rendering(self, rendered_keyboards):
        """Two further #94986 symptoms: a provider label ellipsized past its model
        count in a two-column row, and the truncation hint arriving with literal
        underscores because MarkdownV2 escaping does not spare a bare ``_``."""
        adapter, _ = self._picker(
            self.MODELS, total_models=len(self.MODELS) + 98,
            extra_providers=[{"slug": "openai", "name": "OpenAI", "models": ["o3"], "total_models": 1}])
        rows, _ = await self._render_rows(adapter, "mb")
        provider_row = next(r for r in rows if any(b.callback_data == "mp:bedrock" for b in r))
        provider_button = next(b for b in provider_row if b.callback_data == "mp:bedrock")
        assert provider_button.text.endswith(f"({len(self.MODELS) + 98})")
        # A label this wide takes a row alone: sharing it is what ellipsized the count.
        assert len(provider_row) == 1, [b.text for b in provider_row]

        _buttons, text = await self._render(adapter, "mp:bedrock")
        assert "98 more available" in text
        assert "\\_98 more" not in text, f"literal underscores leaked into the hint: {text!r}"
