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

    def test_group_models_by_vendor_folds_bedrock_ids(self):
        adapter = _make_adapter()
        models = [
            "global.anthropic.claude-opus-5",
            "us.anthropic.claude-opus-5",
            "global.amazon.nova-2-lite-v1:0",
            "us.xai.grok-4.6",
        ]

        vendors = adapter._group_models_by_vendor(models)

        assert [v["vendor"] for v in vendors] == ["amazon", "anthropic", "xai"]
        assert [v["label"] for v in vendors] == ["Amazon", "Anthropic", "xAI"]
        assert [len(v["indices"]) for v in vendors] == [1, 2, 1]
        # Indices must point back into the original list, unmodified.
        anthropic = next(v for v in vendors if v["vendor"] == "anthropic")
        assert [models[i] for i in anthropic["indices"]] == [
            "global.anthropic.claude-opus-5",
            "us.anthropic.claude-opus-5",
        ]

    def test_vendor_aliases_fold_into_a_single_group(self):
        """``moonshot.`` and ``moonshotai.`` are the same vendor and must not
        produce two identically-labelled buttons."""
        adapter = _make_adapter()
        vendors = adapter._group_models_by_vendor([
            "moonshot.kimi-k2-thinking",
            "moonshotai.kimi-k2.5",
        ])
        assert len(vendors) == 1
        assert vendors[0]["label"] == "Moonshot AI"
        assert len(vendors[0]["indices"]) == 2

    def test_geoless_id_is_labelled_when_colliding_with_a_routed_twin(self):
        """``openai.gpt-5.6-terra`` and ``global.openai.gpt-5.6-terra`` are
        distinct profiles: the geoless one must not render as a bare name
        indistinguishable from its routed twin."""
        adapter = _make_adapter()
        models = ["openai.gpt-5.6-terra", "global.openai.gpt-5.6-terra"]
        labels = adapter._model_button_labels(models)
        assert labels == ["direct: gpt-5.6-terra", "global: gpt-5.6-terra"]
        assert len(set(labels)) == 2

    def test_group_models_by_vendor_returns_empty_for_non_bedrock(self):
        """Plain provider model lists must not gain a vendor step."""
        adapter = _make_adapter()
        assert adapter._group_models_by_vendor(["gpt-4o-mini", "o3"]) == []

    def test_vendor_grouping_handles_ids_without_geo_segment(self):
        """A real Bedrock listing mixes ``<geo>.<vendor>.<model>`` with plain
        ``<vendor>.<model>`` IDs. The version dot in ``openai.gpt-5.6-terra``
        must not be mistaken for a vendor boundary ("Gpt-5"), and such IDs
        must land under their real vendor."""
        adapter = _make_adapter()
        models = [
            "openai.gpt-5.6-terra",
            "us.openai.gpt-5.6-terra",
            "zai.glm-4.7",
            "xai.grok-4.6",
            "deepseek.v3.2",
        ]

        vendors = {v["vendor"]: v for v in adapter._group_models_by_vendor(models)}

        assert set(vendors) == {"openai", "zai", "xai", "deepseek"}
        assert [models[i] for i in vendors["openai"]["indices"]] == [
            "openai.gpt-5.6-terra",
            "us.openai.gpt-5.6-terra",
        ]
        assert [models[i] for i in vendors["zai"]["indices"]] == ["zai.glm-4.7"]
        assert [models[i] for i in vendors["deepseek"]["indices"]] == ["deepseek.v3.2"]

    def test_geoless_ids_keep_full_model_name_in_label(self, monkeypatch):
        """``openai.gpt-5.6-terra`` has no geo: it renders as a bare model
        name when unique, and gets the ``direct:`` marker when a routed twin
        exists (both are real, distinct profiles)."""
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()
        models = [
            "openai.gpt-5.6-terra",
            "us.openai.gpt-5.6-terra",
            "openai.gpt-5.5",
        ]

        keyboard, _ = adapter._build_model_keyboard(models, page=0)
        flat = [b for row in keyboard.inline_keyboard for b in row]

        assert [b.text for b in flat[:3]] == [
            "direct: gpt-5.6-terra",
            "us: gpt-5.6-terra",
            "gpt-5.5",
        ]

    def test_vendor_label_uses_known_display_names(self):
        adapter = _make_adapter()
        vendors = {
            v["vendor"]: v["label"]
            for v in adapter._group_models_by_vendor([
                "zai.glm-4.7", "xai.grok-4.6", "openai.gpt-5.5",
                "moonshotai.kimi-k2.5", "nvidia.nemotron-nano-9b-v2",
            ])
        }
        assert vendors["zai"] == "Z.ai"
        assert vendors["xai"] == "xAI"
        assert vendors["openai"] == "OpenAI"
        assert vendors["moonshotai"] == "Moonshot AI"
        assert vendors["nvidia"] == "NVIDIA"

    def test_picker_callback_router_accepts_vendor_prefix(self):
        """The dispatcher must route ``mvd:`` to the picker handler, otherwise
        tapping a vendor button does nothing at all."""
        import inspect
        adapter = _make_adapter()
        src = inspect.getsource(type(adapter)._handle_callback_query)
        assert "mvd:" in src

    def test_vendor_keyboard_lists_vendors_with_counts(self, monkeypatch):
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()
        models = [
            "global.anthropic.claude-opus-5",
            "us.anthropic.claude-opus-5",
            "global.amazon.nova-2-lite-v1:0",
        ]

        keyboard = adapter._build_vendor_keyboard(models)
        flat = [b for row in keyboard.inline_keyboard for b in row]
        labels = [b.text for b in flat]

        assert "Amazon (1)" in labels
        assert "Anthropic (2)" in labels
        assert any(b.callback_data == "mvd:anthropic" for b in flat)
        # Back/Cancel must stay reachable.
        assert any(b.callback_data == "mb" for b in flat)
        assert any(b.callback_data == "mx" for b in flat)

    def test_vendor_scoped_models_show_short_name_and_geo_only(self, monkeypatch):
        """Inside a vendor the vendor segment is redundant: show the short
        model name, plus the geo when the same model exists in several
        routing namespaces."""
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()
        models = [
            "global.anthropic.claude-opus-5",
            "us.anthropic.claude-opus-5",
            "global.anthropic.claude-fable-5",
        ]

        keyboard, _ = adapter._build_model_keyboard(models, page=0)
        flat = [b for row in keyboard.inline_keyboard for b in row]
        labels = [b.text for b in flat[:3]]

        assert labels == [
            "global: claude-opus-5",
            "us: claude-opus-5",
            "claude-fable-5",
        ]
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


