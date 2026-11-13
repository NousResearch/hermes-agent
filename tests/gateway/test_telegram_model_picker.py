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


def _picker_providers(*slugs):
    return [{"slug": s, "name": s, "models": ["m"], "total_models": 1} for s in slugs]


def _keyboard_buttons(keyboard):
    return [b for row in keyboard.inline_keyboard for b in row]


@pytest.fixture
def plain_keyboards(monkeypatch):
    """Inspectable stand-ins for PTB's keyboard types (the module mock records nothing)."""
    import plugins.platforms.telegram.adapter as tg

    monkeypatch.setattr(tg, "InlineKeyboardButton",
                        lambda text, callback_data=None, **_: SimpleNamespace(text=text, callback_data=callback_data))
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", lambda rows: SimpleNamespace(inline_keyboard=rows))


def test_provider_keyboard_keeps_a_provider_whose_config_key_has_uppercase(plain_keyboards):
    """#149672: ``group_providers`` lowercases slugs, so a raw-cased lookup dropped ``Axonhub``."""
    keyboard, _ = _make_adapter()._build_provider_keyboard(_picker_providers("Axonhub", "cnb2api"), 0)
    callbacks = [b.callback_data for b in _keyboard_buttons(keyboard)]
    assert "mp:Axonhub" in callbacks
    assert "mp:cnb2api" in callbacks


def test_provider_keyboard_folds_a_group_member_written_with_uppercase(plain_keyboards):
    keyboard, _ = _make_adapter()._build_provider_keyboard(_picker_providers("Kimi-Coding", "kimi-coding-cn"), 0)
    buttons = _keyboard_buttons(keyboard)
    assert [b.callback_data for b in buttons if b.callback_data.startswith("mp")] == ["mpg:kimi"]
    assert "(2)" in buttons[0].text


@pytest.mark.asyncio
async def test_group_callback_lists_a_member_written_with_uppercase(plain_keyboards):
    adapter = _make_adapter()
    adapter._model_picker_state["12345"] = {
        "providers": _picker_providers("Kimi-Coding", "kimi-coding-cn"),
        "current_model": "m",
        "current_provider": "kimi-coding-cn",
        "session_key": "s",
        "on_model_selected": AsyncMock(),
        "msg_id": 42,
    }
    query = AsyncMock()
    query.data = "mpg:kimi"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.from_user = MagicMock()

    await adapter._handle_model_picker_callback(query, "mpg:kimi", "12345")

    markup = query.edit_message_text.call_args[1]["reply_markup"]
    callbacks = [b.callback_data for b in _keyboard_buttons(markup)]
    assert "mp:Kimi-Coding" in callbacks
    assert "mp:kimi-coding-cn" in callbacks


_BEDROCK_IDS = [
    "global.anthropic.claude-opus-4-5-20251101-v1:0", "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0", "global.anthropic.claude-opus-4-1-20250805-v1:0",
]


@pytest.mark.asyncio
async def test_tapping_a_provider_shows_distinguishable_model_buttons(plain_keyboards):
    """Ids sharing a long namespace (Bedrock ``global.anthropic.``) were right-truncated into identical
    buttons; each must stay readable and distinct while the tap still selects the full id."""
    adapter = _make_adapter()
    provider = {"slug": "bedrock", "name": "AWS Bedrock", "models": _BEDROCK_IDS, "total_models": 50}
    adapter._model_picker_state["12345"] = {
        "providers": [provider], "current_model": "m", "current_provider": "bedrock",
        "session_key": "s", "on_model_selected": AsyncMock(), "msg_id": 42,
    }
    query = AsyncMock()
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.from_user = MagicMock()

    await adapter._handle_model_picker_callback(query, "mp:bedrock", "12345")

    edit = query.edit_message_text.call_args[1]
    model_rows = [row for row in edit["reply_markup"].inline_keyboard if row[0].callback_data.startswith("mm:")]
    labels = [row[0].text for row in model_rows]
    assert all(len(row) == 1 for row in model_rows)
    assert labels == [m.removeprefix("global.anthropic.") for m in _BEDROCK_IDS]
    assert [adapter._model_picker_state["12345"]["model_list"][int(r[0].callback_data.rsplit(":", 1)[1])] for r in model_rows] == _BEDROCK_IDS
    # The "N more available" note renders italic, not as literal underscores.
    assert "\n_46 more available" in edit["text"]


async def _model_labels_for(provider: dict) -> list[str]:
    adapter = _make_adapter()
    adapter._model_picker_state["12345"] = {
        "providers": [provider], "current_model": "m", "current_provider": provider["slug"],
        "session_key": "s", "on_model_selected": AsyncMock(), "msg_id": 42,
    }
    query = AsyncMock()
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.from_user = MagicMock()
    await adapter._handle_model_picker_callback(query, f"mp:{provider['slug']}", "12345")
    rows = query.edit_message_text.call_args[1]["reply_markup"].inline_keyboard
    return [row[0].text for row in rows if row[0].callback_data.startswith("mm:")]


@pytest.mark.asyncio
async def test_model_buttons_still_over_budget_keep_head_and_tail_with_plan_word(plain_keyboards):
    """Ids still too long after the shared namespace goes lose their MIDDLE, so the family head and the
    distinguishing tail stay visible next to the ``Large`` plan word; a short catalog keeps full names."""
    from plugins.platforms.telegram.telegram_picker_labels import MODEL_LABEL_BUDGET

    ns = "ri.language-model-service..language-model."
    ids = [ns + "claude-4-5-sonnet-extended-thinking-20250929", ns + "claude-4-5-opus-extended-thinking-20251101"]
    labels = await _model_labels_for({"slug": "palantir", "name": "P", "models": ids, "total_models": 2,
                                      "model_labels": {ids[0]: "Large"}})
    assert len(labels) == 2 and len(set(labels)) == 2
    assert all(len(lbl) <= MODEL_LABEL_BUDGET for lbl in labels)
    assert labels[0].startswith("claude-4-5-s") and labels[0].endswith("-20250929 · Large")
    assert labels[1].startswith("claude-4-5-o") and labels[1].endswith("-20251101")
    assert "…" in labels[0] and ns not in labels[0]

    short = await _model_labels_for({"slug": "openai", "name": "O", "models": ["gpt-5.1", "gpt-5.2"], "total_models": 2})
    assert short == ["gpt-5.1", "gpt-5.2"]


def _gateway_wired_adapter():
    """Real adapter with the real gateway auth callback, as ``GatewayRunner`` wires it."""
    from gateway.config import GatewayConfig, Platform
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store}
    runner._primary_profile_name = "default"
    adapter = _make_adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("tapper_id, switches", [("222", False), ("111", True)])
async def test_group_model_picker_switch_follows_callback_allowlist(monkeypatch, tapper_id, switches):
    """Only an allowlisted tapper can switch the model from a group /model picker."""
    for key in ("GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS",
                "TELEGRAM_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_CHATS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")
    import hermes_cli.model_selection_guards as guards
    monkeypatch.setattr(guards, "combined_selection_warning", lambda *a, **k: None)

    adapter = _gateway_wired_adapter()
    on_model_selected = AsyncMock(return_value="Switched to `other-model`")
    adapter._model_picker_state["-100777"] = {
        "providers": [{"slug": "openai", "name": "OpenAI", "total_models": 1}],
        "current_model": "owner-model",
        "current_provider": "openai",
        "session_key": "s",
        "on_model_selected": on_model_selected,
        "selected_provider": "openai",
        "model_list": ["other-model"],
        "full_model_list": ["other-model"],
        "listing": 7,
        "msg_id": 42,
    }
    query = SimpleNamespace(
        data="mm:7:0",
        message=SimpleNamespace(
            chat_id=-100777, chat=SimpleNamespace(type="supergroup"), message_thread_id=None
        ),
        from_user=SimpleNamespace(id=int(tapper_id), first_name="Tapper"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )

    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())

    assert on_model_selected.await_count == (1 if switches else 0)
    assert ("-100777" in adapter._model_picker_state) is not switches




@pytest.mark.asyncio
async def test_long_model_ids_differing_mid_name_render_distinct_labels(plain_keyboards):
    """Head/tail shortening dropped the middle, so ids that differ only there (``3-1`` vs ``3-2``,
    or a variant word) rendered identical buttons that select different models."""
    from plugins.platforms.telegram.telegram_picker_labels import MODEL_LABEL_BUDGET

    tail = "-70b-instruct-awq-quantized-production-20261101"
    ids = [f"custom-llama-3-{v}{tail}" for v in (1, 2)] + [f"custom-llama-3-1-70b-{w}-awq-quantized-production-20261101"
                                                          for w in ("chat-distilled-reasoning", "chat-distilled-reasoner")]
    labels = await _model_labels_for({"slug": "custom", "name": "C", "models": ids, "total_models": 4,
                                      "model_labels": {ids[0]: "Large"}})
    assert len(set(labels)) == len(ids), labels
    assert all(len(lbl) <= MODEL_LABEL_BUDGET for lbl in labels), labels
    assert "3-1" in labels[0] and "3-2" in labels[1] and labels[0].endswith(" · Large")


@pytest.mark.asyncio
async def test_model_ids_differing_only_in_their_org_keep_it_in_the_labels(plain_keyboards):
    """``org/`` can be all that differs (OpenRouter ``meta-llama/`` vs ``nousresearch/``, Ollama ``hf.co/<org>/``);
    the label must show it rather than an ordinal, with or without a plan word."""
    ids = ["meta-llama/llama-3.1-8b-instruct", "nousresearch/llama-3.1-8b-instruct", "openai/gpt-5"]
    for model_labels in ({}, {ids[0]: "Large"}):
        labels = await _model_labels_for({"slug": "openrouter", "name": "O", "models": ids, "total_models": 3,
                                          "model_labels": model_labels})
        assert labels[0].startswith("meta-llama/") and labels[1].startswith("nousresearch/"), labels
        assert labels[2] == "gpt-5"


@pytest.mark.asyncio
async def test_long_look_alike_ids_never_share_a_label(plain_keyboards):
    """Two clash groups whose names share a long beginning must not fall back to the same ordinal label."""
    x = "x" * 60
    ids = [f"{x}{v}{x}-{t}" for t in ("one", "two") for v in (1, 2)]
    labels = await _model_labels_for({"slug": "custom", "name": "C", "models": ids, "total_models": 4})
    assert len(set(labels)) == len(ids), labels



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

    def _picker(self, models, total_models=None, extra_providers=(), bedrock=True):
        adapter = _make_adapter()
        adapter._is_callback_user_authorized = lambda *a, **k: True  # dispatcher auth is not under test here
        first = [{"slug": "bedrock", "name": "AWS Bedrock", "models": models,
                  "total_models": total_models if total_models is not None else len(models),
                  "is_current": True}] if bedrock else []
        adapter._model_picker_state["12345"] = {
            "providers": [*first, *extra_providers],
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
        query.message.message_id = 42  # the picker message the state is bound to
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
    async def test_a_drawn_button_never_switches_to_a_different_model(self, rendered_keyboards):
        """A button keeps the model it displayed, across a navigation step.

        Telegram leaves every inline keyboard tappable, while the picker rewrites
        its scoped model sub-list on each vendor drill-down and Back. A payload
        that numbered the *scoped* list therefore resolved against whichever list
        was current when the tap arrived: the Amazon page's ``nova-lite-v1:0``
        button switched the session to ``global.anthropic.claude-opus-5`` once the
        user had stepped back to the vendor list (#94990 review). Either the exact
        advertised ID or an explicit refusal is acceptable — another model is not.
        """
        adapter, state = self._picker(self.MODELS)
        callback = state["on_model_selected"]
        await self._render(adapter, "mp:bedrock")
        amazon, _ = await self._render(adapter, "mvd:amazon")
        nova = next(b for b in amazon if b.text == "nova-lite-v1:0")

        await self._render(adapter, "mb")  # back to the vendors: the scoped list is replaced
        query = await self._tap(adapter, nova.callback_data)

        switched = [call[0][1] for call in callback.await_args_list]
        assert switched in ([], ["amazon.nova-lite-v1:0"]), (
            f"{nova.text!r} switched to {switched}")
        if not switched:  # refusing is fine, but it has to be said out loud
            assert (query.answer.await_args and query.answer.await_args.kwargs.get("text")), \
                "a refused tap must tell the user, not answer silently"

    @pytest.mark.asyncio
    async def test_a_button_from_a_superseded_picker_is_refused(self, rendered_keyboards):
        """``/model`` twice leaves two tappable messages but one picker state.

        The older message's buttons number a catalog that is no longer loaded, so
        resolving them against the new provider's list hands the switch a model
        from a listing the user never opened. Such a tap must be refused.
        """
        adapter, state = self._picker(self.MODELS)
        stale_callback = state["on_model_selected"]
        await self._render(adapter, "mp:bedrock")
        amazon, _ = await self._render(adapter, "mvd:amazon")
        stale_tap = next(b for b in amazon if b.text == "nova-lite-v1:0").callback_data

        fresh_callback = AsyncMock(return_value="switched")
        adapter._model_picker_state["12345"] = {
            "providers": [{"slug": "openai", "name": "OpenAI", "models": ["o3", "gpt-4o-mini"],
                           "total_models": 2}],
            "current_model": "o3", "current_provider": "openai", "session_key": "s",
            "on_model_selected": fresh_callback, "msg_id": 42,
        }
        await self._render(adapter, "mp:openai")

        query = await self._tap(adapter, stale_tap)

        assert fresh_callback.await_count == 0, (
            f"a stale tap switched into the new catalog: {fresh_callback.await_args_list}")
        assert stale_callback.await_count == 0
        assert (query.answer.await_args and query.answer.await_args.kwargs.get("text")), \
            "a refused tap must tell the user, not answer silently"

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
        adapter, state = self._picker(["gpt-4o-mini", "o3"])
        buttons, _text = await self._render(adapter, "mp:bedrock")
        picks = [b for b in buttons if str(b.callback_data).startswith("mm:")]
        assert [b.text for b in picks] == ["gpt-4o-mini", "o3"]
        await self._tap(adapter, picks[1].callback_data)
        assert state["on_model_selected"].await_args[0][1] == "o3"

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

    @pytest.mark.asyncio
    async def test_provider_family_submenu_keeps_long_member_labels_readable(self, rendered_keyboards):
        """A provider FAMILY submenu (``mpg:``) must not ellipsize its members.

        The top-level list folds families behind one button, so the member names
        are only ever read here — and real ones are long (``Kimi / Kimi Coding
        Plan``, ``ChatGPT or Codex Subscription``). Sharing a two-column row is
        exactly what hid ``✓ AWS Bedrock (1…``'s count at the top level (#94986);
        this sibling path reaches the same layout helper, so it must clamp the
        same way instead of truncating the member's identity.
        """
        from hermes_cli.models_catalog_static import PROVIDER_GROUPS

        from plugins.platforms.telegram.model_picker_display import TWO_COLUMN_BUDGET

        _label, _desc, members = PROVIDER_GROUPS["kimi"]
        adapter, _ = self._picker(
            [], bedrock=False,
            extra_providers=[{"slug": slug, "name": name, "models": ["m"], "total_models": 1}
                             for slug, name in zip(members, ("Kimi / Kimi Coding Plan", "Kimi / Moonshot (China)"))])
        rows, _text = await self._render_rows(adapter, "mpg:kimi")
        for row in rows:
            for button in row:
                if not str(button.callback_data).startswith("mp:"):
                    continue
                assert len(row) == 1 or len(button.text) <= TWO_COLUMN_BUDGET, (
                    f"{button.text!r} shares a two-column row and loses its tail: {[b.text for b in row]}")
