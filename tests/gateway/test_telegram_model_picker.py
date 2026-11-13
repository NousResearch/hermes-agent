"""Tests for Telegram model picker thread fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from hermes_agent.gateway.config import PlatformConfig
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
    assert [adapter._model_picker_state["12345"]["model_list"][int(r[0].callback_data[3:])] for r in model_rows] == _BEDROCK_IDS
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
    from hermes_agent.gateway.config import GatewayConfig, Platform
    from hermes_agent.gateway.pairing import PairingStore
    from hermes_agent.gateway.run import GatewayRunner

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
    import hermes_agent.hermes_cli.model_selection_guards as guards
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
        "msg_id": 42,
    }
    query = SimpleNamespace(
        data="mm:0",
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
