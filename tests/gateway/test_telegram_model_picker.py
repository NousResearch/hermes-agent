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
    def test_anthropic_vendor_keeps_vendor_name_and_omits_redundant_prefix(self, monkeypatch):
        """The Bedrock vendor picker calls this family Anthropic; inside it,
        the repeated ``claude-`` model prefix remains unnecessary. A global
        profile uses compact ``G:`` to preserve the discriminating suffix."""
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()
        model_id = "global.anthropic.claude-sonnet-4-6"

        keyboard, _ = adapter._build_model_keyboard([model_id], page=0)
        button = keyboard.inline_keyboard[0][0]
        vendors = adapter._group_models_by_vendor([model_id])

        assert button.text == "sonnet-4-6"
        assert button.callback_data == "mm:0"
        assert vendors == [{"vendor": "anthropic", "label": "Anthropic", "indices": [0]}]

    def test_anthropic_global_collision_uses_compact_g_prefix(self):
        adapter = _make_adapter()
        models = [
            "global.anthropic.claude-opus-4-5-20251101-v1:0",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
        ]

        assert adapter._model_button_labels(models) == [
            "G: opus-4-5-20251101-v1:0",
            "us: opus-4-5-20251101-v1:0",
        ]

    @pytest.mark.parametrize("model_id,expected", [
        # Claude is already selected in the vendor menu, so its repeated
        # prefix is removed; other vendors retain their short model ID.
        ("us.anthropic.claude-opus-4-1-20250805-v1:0", "opus-4-1-20250805-v1:0"),
        ("eu.anthropic.claude-sonnet-4-6", "sonnet-4-6"),
        ("apac.amazon.nova-2-lite-v1:0", "nova-2-lite-v1:0"),
        ("us-gov.anthropic.claude-haiku-4-5", "haiku-4-5"),
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
        """``global.anthropic`` must not produce an empty label -- Telegram
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

        assert labels == ["us: grok-4.6", "G: grok-4.6", "sonnet-4-6"]
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

    def test_geoless_id_uses_configured_short_geo_when_colliding(self, monkeypatch):
        """A bare model ID gets the short geography of the configured Bedrock
        region (us/eu/ap/...), never the implementation term ``direct`` or a
        full region name such as ``us-east-1``."""
        adapter = _make_adapter()
        monkeypatch.setattr(adapter, "_bedrock_region_scope", lambda: "us")
        models = ["openai.gpt-5.6-terra", "global.openai.gpt-5.6-terra"]
        labels = adapter._model_button_labels(models)
        assert labels == ["us: gpt-5.6-terra", "G: gpt-5.6-terra"]
        assert all("direct" not in label and "us-east-1" not in label for label in labels)

    @pytest.mark.parametrize(
        ("region", "scope"),
        [
            ("us-east-1", "us"),
            ("eu-west-3", "eu"),
            ("ap-southeast-2", "ap"),
            ("ca-central-1", "ca"),
        ],
    )
    def test_bedrock_region_scope_is_short(self, monkeypatch, region, scope):
        adapter = _make_adapter()
        monkeypatch.setattr(
            "agent.bedrock_adapter.resolve_bedrock_runtime_region",
            lambda: region,
        )
        assert adapter._bedrock_region_scope() == scope

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
        name when unique, and gets the configured short geography when a
        routed twin exists."""
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _FakeInlineKeyboardButton)
        monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
        adapter = _make_adapter()
        monkeypatch.setattr(adapter, "_bedrock_region_scope", lambda: "us")
        models = [
            "openai.gpt-5.6-terra",
            "us.openai.gpt-5.6-terra",
            "openai.gpt-5.5",
        ]

        keyboard, _ = adapter._build_model_keyboard(models, page=0)
        flat = [b for row in keyboard.inline_keyboard for b in row]

        assert [b.text for b in flat[:3]] == [
            "us: gpt-5.6-terra",
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
            "G: opus-5",
            "us: opus-5",
            "fable-5",
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
