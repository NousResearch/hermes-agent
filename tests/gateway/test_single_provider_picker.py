"""Exercise the sent Telegram keyboard and its existing callback path."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture(autouse=True)
def keyboard_values(monkeypatch):
    # The gateway suite substitutes the optional Telegram SDK at collection.
    # Keep its wire-value objects concrete so keyboard assertions are meaningful.
    import plugins.platforms.telegram.adapter as telegram
    monkeypatch.setattr(telegram, "InlineKeyboardButton", lambda text, callback_data: SimpleNamespace(text=text, callback_data=callback_data))
    monkeypatch.setattr(telegram, "InlineKeyboardMarkup", lambda rows: SimpleNamespace(inline_keyboard=rows))


def make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture-token"))
    adapter._bot = AsyncMock()
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=101)
    adapter._app = MagicMock()
    return adapter


def callbacks(keyboard):
    return [button.callback_data for row in keyboard.inline_keyboard for button in row]


@pytest.mark.asyncio
async def test_single_provider_opens_models_and_keeps_pagination_and_selection(monkeypatch):
    import hermes_cli.model_selection_guards as guards
    monkeypatch.setattr(guards, "combined_selection_warning", lambda *a, **k: None)
    adapter = make_adapter()
    selected = AsyncMock(return_value="Switched")
    models = [f"fixture-{i}" for i in range(12)]
    result = await adapter.send_model_picker(
        "123", [{"slug": "custom:fixture", "name": "Fixture", "models": models}],
        models[0], "custom:fixture", "fixture-session", selected,
    )
    assert result.success
    sent = adapter._bot.send_message.call_args.kwargs
    buttons = callbacks(sent["reply_markup"])
    assert "mm:0" in buttons
    assert "mb" not in buttons
    assert "mx" in buttons
    selected.assert_not_awaited()
    query = AsyncMock()
    await adapter._handle_model_picker_callback(query, "mg:1", "123")
    buttons = callbacks(query.edit_message_text.call_args.kwargs["reply_markup"])
    assert "mm:8" in buttons
    assert "mb" not in buttons
    await adapter._handle_model_picker_callback(query, "mm:8", "123")
    selected.assert_awaited_once_with("123", models[8], "custom:fixture")
    assert "123" not in adapter._model_picker_state


@pytest.mark.asyncio
@pytest.mark.parametrize("slugs", [["custom:a", "custom:b"], ["minimax", "minimax-cn"], []])
async def test_provider_choices_and_group_rows_are_not_skipped(slugs):
    adapter = make_adapter()
    providers = [{"slug": s, "name": s, "models": ["fixture"]} for s in slugs]
    await adapter.send_model_picker("123", providers, "fixture", "custom:a", "s", AsyncMock())
    buttons = callbacks(adapter._bot.send_message.call_args.kwargs["reply_markup"])
    assert "mm:0" not in buttons
    if slugs == ["minimax", "minimax-cn"]:
        assert any(b.startswith("mpg:") for b in buttons)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "send", "callback", "confirm"])
async def test_single_provider_failure_cancel_and_confirmation(monkeypatch, failure):
    import hermes_cli.model_selection_guards as guards
    warning = SimpleNamespace(title="Confirm", message="Cost warning") if failure == "confirm" else None
    monkeypatch.setattr(guards, "combined_selection_warning", lambda *a, **k: warning)
    adapter = make_adapter()
    selected = AsyncMock(return_value="Switched")
    if failure == "callback":
        selected.side_effect = RuntimeError("fixture failure")
    if failure == "send":
        adapter._bot.send_message.side_effect = RuntimeError("fixture transport")
    result = await adapter.send_model_picker(
        "123", [{"slug": "custom:fixture", "name": "Fixture", "models": ["fixture"]}],
        "fixture", "custom:fixture", "s", selected,
    )
    if failure == "send":
        assert not result.success
        assert "123" not in adapter._model_picker_state
        return
    query = AsyncMock()
    await adapter._handle_model_picker_callback(query, "mx" if failure == "cancel" else "mm:0", "123")
    if failure == "confirm":
        selected.assert_not_awaited()
        await adapter._handle_model_picker_callback(query, "mb", "123")
        assert "mm:0" in callbacks(query.edit_message_text.call_args.kwargs["reply_markup"])
        assert "mb" not in callbacks(query.edit_message_text.call_args.kwargs["reply_markup"])
        await adapter._handle_model_picker_callback(query, "mc:0", "123")
    assert "123" not in adapter._model_picker_state
    if failure == "cancel":
        selected.assert_not_awaited()
    else:
        selected.assert_awaited_once_with("123", "fixture", "custom:fixture")


def test_shared_decision_preserves_empty_and_grouping_fallback(monkeypatch):
    import sys
    from gateway.platforms.model_picker import single_provider_for_picker
    assert single_provider_for_picker([]) is None
    assert single_provider_for_picker([{"slug": "fixture", "models": []}]) is None
    provider = {"slug": "fixture", "models": ["model"]}
    monkeypatch.setitem(sys.modules, "hermes_cli.models_catalog_static", None)
    assert single_provider_for_picker([provider], grouped=True) is provider
    assert single_provider_for_picker([provider, dict(provider, slug="other")], grouped=True) is None
