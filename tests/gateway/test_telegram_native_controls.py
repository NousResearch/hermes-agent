"""Native Telegram menus preserve actions, selected state, and private/public routing."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as module
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_rich_controls import rich_control_html


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(module, "InlineKeyboardButton", lambda text, callback_data: {
        "text": text, "callback_data": callback_data})
    monkeypatch.setattr(module, "InlineKeyboardMarkup", lambda rows: {"inline_keyboard": rows})
    host = TelegramAdapter(PlatformConfig(extra={"rich_controls": True}))
    host._bot = SimpleNamespace(do_api_request=AsyncMock(return_value={"message_id": 42}),
                                send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)))
    return host


@pytest.mark.asyncio
async def test_current_choice_is_disabled_without_losing_other_actions(adapter):
    result = await adapter.send_choice_picker("123", "Mode", [
        {"value": "on", "label": "Включено", "is_current": True},
        {"value": "off", "label": "Выключено"}], "session", AsyncMock())
    assert result.success
    payload = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]
    body = payload["rich_message"]["html"]
    assert 'type="disabled"' in body
    assert 'data="cp:0"' not in body
    assert 'data="cp:1"' in body
    # "off" in a translated option label must not turn a selection into a danger action.
    assert 'style="danger"' not in body


@pytest.mark.asyncio
async def test_clarify_options_have_adjacent_controls_and_escape_untrusted_text(adapter):
    await adapter.send_clarify("123", "Choose", ["<script>& alpha", "beta"], "c", "s")
    payload = adapter._bot.do_api_request.call_args.kwargs["api_kwargs"]
    body = payload["rich_message"]["html"]
    assert body.count("&lt;script&gt;&amp; alpha") == 1
    assert "<script>" not in body
    assert body.index("&lt;script&gt;&amp; alpha") < body.index('data="cl:c:0"') < body.index("beta")
    assert body.index("beta") < body.index('data="cl:c:1"') < body.index('data="cl:c:other"')


def test_model_navigation_counter_and_current_model_are_disabled(adapter):
    markup, _ = adapter._build_model_keyboard(["current", "other"] * 12, 0, current_model="current")
    body = rich_control_html("Models", None, markup)
    assert 'data="mm:0"' not in body
    assert 'data="mm:1"' in body
    assert 'data="mx:noop"' not in body
    assert 'align="center"' in body
    assert 'data="mg:1"' in body
    assert 'data="mb"' in body and 'data="mx"' in body


@pytest.mark.asyncio
async def test_rich_rejection_restores_full_legacy_clarify_content_and_keyboard(adapter):
    adapter._is_rich_fallback_error = lambda exc: True
    adapter._bot.do_api_request.side_effect = RuntimeError("unsupported rich messages")
    await adapter.send_clarify("123", "Choose", ["alpha", "beta"], "c", "s")
    legacy = adapter._bot.send_message.call_args.kwargs
    assert "alpha" in legacy["text"] and "beta" in legacy["text"]
    assert legacy["reply_markup"] == {"inline_keyboard": [
        [{"text": "1", "callback_data": "cl:c:0"}],
        [{"text": "2", "callback_data": "cl:c:1"}],
        [{"text": module.t("platform.telegram.prompt.other"), "callback_data": "cl:c:other"}]]}


@pytest.mark.asyncio
async def test_disabled_flat_choice_remains_usable_when_rich_is_off(adapter):
    adapter.config.extra["rich_controls"] = False
    await adapter.send_choice_picker("123", "Mode", [
        {"value": "on", "label": "On", "is_current": True}], "s", AsyncMock())
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["reply_markup"] == {"inline_keyboard": [[{"text": "✓ On", "callback_data": "cp:0"}]]}
    adapter._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_rich_rejection_keeps_legacy_keyboard_requester_only(adapter):
    adapter._private_controls = True
    adapter._is_rich_fallback_error = lambda exc: True
    adapter._bot.do_api_request.side_effect = RuntimeError("rich unsupported")
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=0, api_kwargs={
        "ephemeral_message_id": 7, "receiver_user": {"id": 111}})
    result = await adapter.send_choice_picker("-100", "Mode", [
        {"value": "on", "label": "On", "is_current": True},
        {"value": "off", "label": "Off"}], "s", AsyncMock(), metadata={
            "telegram_chat_type": "supergroup", "telegram_requester_user_id": "111"})
    assert result.success and result.message_id == "eph:111:7"
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["api_kwargs"]["ephemeral_message_parameters"]["receiver_user_id"] == 111
    assert set(kwargs["reply_markup"]) == {"inline_keyboard"}
    assert all(set(button) == {"text", "callback_data"}
               for row in kwargs["reply_markup"]["inline_keyboard"] for button in row)


@pytest.mark.asyncio
async def test_result_edit_does_not_retry_lost_rich_acknowledgement(adapter):
    adapter.register_rich_control_message(123, 42)
    adapter._is_rich_fallback_error = lambda exc: False
    adapter._bot.do_api_request.side_effect = TimeoutError("lost ack")
    raw_query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=42),
                                edit_message_text=AsyncMock())
    query = adapter.wrap_rich_control_query(raw_query)
    with pytest.raises(TimeoutError):
        await adapter._edit_result_text(query, "Changed")
    assert adapter._bot.do_api_request.await_count == 1
    raw_query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["0", "other", "expired"])
async def test_rich_clarify_resolution_keeps_question_and_option_context(adapter, choice):
    from tools.clarify_gateway import register, wait_for_response
    entry = register("native-context", "s", "Which <target>?", ["alpha", "beta"])
    await adapter.send_clarify("123", entry.question, entry.choices, entry.clarify_id, "s")
    query = SimpleNamespace(message=SimpleNamespace(chat_id=123, message_id=42, text=""),
                            from_user=SimpleNamespace(id=111, first_name="Member"),
                            answer=AsyncMock(), edit_message_text=AsyncMock())
    adapter._callback_authorized = AsyncMock(return_value=True)
    if choice == "expired":
        wait_for_response(entry.clarify_id, 0.001)
        choice = "0"
    await adapter._handle_clarify_callback(query, f"cl:{entry.clarify_id}:{choice}", {"chat_id": 123})
    text = query.edit_message_text.call_args.kwargs["text"]
    assert "Which &lt;target&gt;?" in text
    assert "alpha" in text and "beta" in text
    wait_for_response(entry.clarify_id, 0.001)


def test_compact_selectors_pair_short_actions_without_truncating_long_labels(adapter):
    labels = ["minimal", "low", "medium", "high", "ultra (sends max on this route)", "reset — clear session override"]
    buttons = [{"text": label, "callback_data": f"cp:{i}"} for i, label in enumerate(labels)]
    rows = adapter._selection_rows(buttons)
    assert [[b["callback_data"] for b in row] for row in rows] == [
        ["cp:0", "cp:1"], ["cp:2", "cp:3"], ["cp:4"], ["cp:5"]]
    assert [b["text"] for row in rows for b in row] == labels
