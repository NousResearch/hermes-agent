"""Tests for Telegram model picker thread fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _provider_rows():
    # ``providers:`` config keys are user-chosen and may contain uppercase letters
    # (e.g. ``providers: {Magpie: ...}``); the picker row slug preserves that case,
    # while the shared group fold lowercases every slug.
    return [
        {"slug": "Magpie", "name": "Magpie", "total_models": 59,
         "models": ["m1", "m2"], "is_current": True, "is_user_defined": True,
         "api_url": "https://relay.example/v1"},
        {"slug": "copilot", "name": "GitHub Copilot", "total_models": 44,
         "models": ["c1"], "is_current": False},
    ]


def _keyboard_buttons(adapter, providers):
    keyboard, _page_info = adapter._build_provider_keyboard(providers, 0)
    return [b for row in keyboard.inline_keyboard for b in row]


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


def _install_fake_keyboard(monkeypatch):
    """Some gateway tests rebind/reload the telegram adapter module (the lazy-install
    tests trigger PTB re-imports), so patch the globals of the module object that the
    adapter CLASS actually resolves — not the name reached via ``import`` here. The
    mocked telegram package otherwise provides a bare MagicMock markup/button.
    """
    class FakeButton:
        def __init__(self, text=None, callback_data=None, **_kwargs):
            self.text, self.callback_data = text, callback_data

    class FakeMarkup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    adapter_globals = TelegramAdapter.__init__.__globals__
    monkeypatch.setitem(adapter_globals, "InlineKeyboardButton", FakeButton)
    monkeypatch.setitem(adapter_globals, "InlineKeyboardMarkup", FakeMarkup)


def test_provider_keyboard_renders_mixed_case_user_provider(monkeypatch):
    """A user-defined provider with an uppercase slug must not be dropped by the
    lowercase-keyed group fold, and its button keeps the original slug/case."""
    _install_fake_keyboard(monkeypatch)
    adapter = _make_adapter()
    buttons = _keyboard_buttons(adapter, _provider_rows())

    callbacks = {b.callback_data for b in buttons}
    assert "mp:Magpie" in callbacks
    assert "mp:copilot" in callbacks
    magpie = next(b for b in buttons if b.callback_data == "mp:Magpie")
    assert "Magpie (59)" in magpie.text


@pytest.mark.asyncio
async def test_provider_button_tap_opens_mixed_case_provider_models(monkeypatch):
    """Tapping ``mp:<MixedCase>`` selects that exact provider row and renders its models."""
    _install_fake_keyboard(monkeypatch)
    adapter = _make_adapter()
    adapter._model_picker_state["12345"] = {
        "providers": _provider_rows(),
        "current_model": "m1",
        "current_provider": "Magpie",
        "session_key": "s",
        "on_model_selected": AsyncMock(),
        "msg_id": 42,
    }
    query = AsyncMock()
    query.data = "mp:Magpie"
    query.message = MagicMock()
    query.message.chat_id = 12345

    await adapter._handle_model_picker_callback(query, "mp:Magpie", "12345")

    state = adapter._model_picker_state["12345"]
    assert state["selected_provider"] == "Magpie"
    assert state["selected_provider_name"] == "Magpie"
    assert state["model_list"] == ["m1", "m2"]
    assert query.edit_message_text.await_count == 1


