from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class SlottedUpdate:
    """Fake PTB-like update that rejects arbitrary marker attributes."""

    __slots__ = ("update_id", "message", "effective_message", "callback_query")

    def __init__(self, *, update_id, message=None, effective_message=None, callback_query=None):
        self.update_id = update_id
        self.message = message
        self.effective_message = effective_message
        self.callback_query = callback_query


@pytest.mark.asyncio
async def test_text_handler_raw_update_hook_can_handle_before_auth_and_dispatch(monkeypatch):
    from gateway.platforms import telegram as telegram_mod

    adapter = object.__new__(telegram_mod.TelegramAdapter)
    adapter._should_process_message = MagicMock(return_value=True)
    adapter._should_observe_unmentioned_group_message = MagicMock(return_value=False)
    adapter._observe_unmentioned_group_message = MagicMock()
    adapter._ensure_forum_commands = AsyncMock()
    adapter._build_message_event = MagicMock()
    adapter._clean_bot_trigger_text = MagicMock()
    adapter._apply_telegram_group_observe_attribution = MagicMock()
    adapter._enqueue_text_event = MagicMock()

    msg = SimpleNamespace(text="hello", chat_id=123)
    update = SimpleNamespace(update_id=42, message=msg, effective_message=msg)
    context = object()
    seen = {}

    async def fake_invoke_hook_async(hook_name, **kwargs):
        seen["hook_name"] = hook_name
        seen["kwargs"] = kwargs
        return [{"action": "handled", "reason": "business-inbox"}]

    monkeypatch.setattr(telegram_mod, "invoke_hook_async", fake_invoke_hook_async, raising=False)

    await adapter._handle_text_message(update, context)

    assert seen["hook_name"] == "telegram_raw_update"
    assert seen["kwargs"]["update"] is update
    assert seen["kwargs"]["context"] is context
    assert seen["kwargs"]["adapter"] is adapter
    assert seen["kwargs"]["handler"] == "text"
    adapter._should_process_message.assert_not_called()
    adapter._enqueue_text_event.assert_not_called()
    adapter._ensure_forum_commands.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_handler_continues_normally_when_raw_update_hook_allows(monkeypatch):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms import telegram as telegram_mod

    adapter = object.__new__(telegram_mod.TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.1
    adapter._should_process_message = MagicMock(return_value=True)
    adapter._should_observe_unmentioned_group_message = MagicMock(return_value=False)
    adapter._observe_unmentioned_group_message = MagicMock()
    adapter._ensure_forum_commands = AsyncMock()
    adapter._build_message_event = MagicMock(return_value=SimpleNamespace(text="hello", media_urls=[], media_types=[]))
    adapter._clean_bot_trigger_text = MagicMock(return_value="hello")
    adapter._apply_telegram_group_observe_attribution = MagicMock(side_effect=lambda event: event)
    adapter._enqueue_text_event = MagicMock()

    msg = SimpleNamespace(text="hello", chat_id=123)
    update = SimpleNamespace(update_id=42, message=msg, effective_message=msg)

    async def fake_invoke_hook_async(hook_name, **kwargs):
        return [{"action": "allow"}]

    monkeypatch.setattr(telegram_mod, "invoke_hook_async", fake_invoke_hook_async, raising=False)

    await adapter._handle_text_message(update, object())

    adapter._should_process_message.assert_called_once_with(msg)
    adapter._ensure_forum_commands.assert_awaited_once_with(msg)
    adapter._enqueue_text_event.assert_called_once()


@pytest.mark.asyncio
async def test_raw_update_hook_does_not_mutate_slotted_update(monkeypatch):
    from gateway.platforms import telegram as telegram_mod

    adapter = object.__new__(telegram_mod.TelegramAdapter)
    calls = []
    msg = SimpleNamespace(text="hello", chat_id=123)
    update = SlottedUpdate(update_id=4242, message=msg, effective_message=msg)

    async def fake_invoke_hook_async(hook_name, **kwargs):
        calls.append((hook_name, kwargs["handler"]))
        return [{"action": "allow"}]

    monkeypatch.setattr(telegram_mod, "invoke_hook_async", fake_invoke_hook_async, raising=False)

    handled = await adapter._invoke_raw_update_hooks(update, object(), handler="raw")
    handled_again = await adapter._invoke_raw_update_hooks(update, object(), handler="text")

    assert handled is False
    assert handled_again is False
    assert calls == [("telegram_raw_update", "raw")]


@pytest.mark.asyncio
async def test_callback_query_raw_update_hook_can_handle_before_builtin_prefixes(monkeypatch):
    from gateway.platforms import telegram as telegram_mod

    adapter = object.__new__(telegram_mod.TelegramAdapter)
    adapter._handle_model_picker_callback = AsyncMock()
    adapter._handle_gmail_triage_callback = AsyncMock()
    adapter._approval_state = {}
    adapter._slash_confirm_state = {}
    adapter._clarify_state = {}

    query = SimpleNamespace(
        data="mp:gpt-5",
        message=SimpleNamespace(chat_id=123, chat=SimpleNamespace(type="private"), message_thread_id=None),
        from_user=SimpleNamespace(id=602562, first_name="Alen"),
        answer=AsyncMock(),
    )
    update = SimpleNamespace(update_id=99, callback_query=query)
    context = object()

    async def fake_invoke_hook_async(hook_name, **kwargs):
        assert hook_name == "telegram_raw_update"
        assert kwargs["handler"] == "callback_query"
        assert kwargs["update"] is update
        assert kwargs["context"] is context
        assert kwargs["adapter"] is adapter
        return [{"action": "skip", "reason": "plugin-callback"}]

    monkeypatch.setattr(telegram_mod, "invoke_hook_async", fake_invoke_hook_async, raising=False)

    await adapter._handle_callback_query(update, context)

    adapter._handle_model_picker_callback.assert_not_awaited()
    adapter._handle_gmail_triage_callback.assert_not_awaited()
