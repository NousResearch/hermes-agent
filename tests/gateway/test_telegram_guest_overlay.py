from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.run import GatewayRunner
from plugins.platforms.telegram import adapter as telegram_mod


def _adapter():
    adapter = telegram_mod.TelegramAdapter(
        PlatformConfig(enabled=True, extra={"guest_mode": True})
    )
    adapter._bot = MagicMock()
    adapter._bot.username = "hermes_test_bot"
    adapter.handle_message = AsyncMock()
    return adapter


def test_allowed_update_types_adds_guest_message(monkeypatch):
    monkeypatch.setattr(telegram_mod, "Update", SimpleNamespace(ALL_TYPES=["message"]))
    assert telegram_mod.TelegramAdapter._allowed_update_types() == ["message", "guest_message"]


def test_gateway_guest_config_reads_top_level_telegram():
    assert GatewayRunner._telegram_guest_config(
        {"telegram": {"guest_mode_toolsets": ["web"]}}
    ) == {"guest_mode_toolsets": ["web"]}
    assert GatewayRunner._telegram_guest_config({"telegram": "invalid"}) == {}


def test_guest_context_from_raw_update():
    raw = {
        "guest_query_id": "q1",
        "chat": {"id": 11, "type": "private"},
        "from": {"id": 42, "first_name": "Max"},
        "text": "hello",
        "guest_bot_caller_user": {"id": 42, "first_name": "Max"},
        "guest_bot_caller_chat": {"id": -100, "title": "Group"},
    }
    update = SimpleNamespace(guest_message=None, api_kwargs={"guest_message": raw})

    ctx = telegram_mod.TelegramAdapter._guest_context_from_update(update)

    assert ctx is not None
    assert ctx.guest_query_id == "q1"
    assert ctx.caller_user_id == "42"
    assert ctx.caller_chat_id == "-100"
    assert ctx.message.text == "hello"


@pytest.mark.asyncio
async def test_guest_update_routes_to_answer_target():
    adapter = _adapter()
    raw = {
        "guest_query_id": "q1",
        "chat": {"id": 11, "type": "private"},
        "from": {"id": 42, "first_name": "Max"},
        "text": "hello",
        "guest_bot_caller_user": {"id": 42, "first_name": "Max"},
        "guest_bot_caller_chat": {"id": -100, "title": "Group"},
    }
    update = SimpleNamespace(
        update_id=7, guest_message=None, api_kwargs={"guest_message": raw}
    )

    await adapter._handle_guest_update(update, None)

    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "guest:q1"
    assert event.source.user_id == "42"
    assert event.guest_mode_invocation is True
    assert event.session_key_override.startswith("telegram:guest-session:")
    assert event.source.session_key_override == event.session_key_override


@pytest.mark.asyncio
async def test_guest_send_answers_query(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        adapter,
        "_raw_answer_guest_query",
        AsyncMock(return_value={"inline_message_id": "inline-1"}),
    )

    result = await adapter.send("guest:q1", "hello")

    assert result.success is True
    assert result.message_id == "inline-1"
    assert adapter._guest_inline_message_ids_cache()["q1"] == "inline-1"


@pytest.mark.asyncio
async def test_guest_send_edits_existing_inline(monkeypatch):
    adapter = _adapter()
    adapter._guest_inline_message_ids_cache()["q1"] = "inline-1"
    edit = AsyncMock(
        return_value=telegram_mod.SendResult(success=True, message_id="inline-1")
    )
    monkeypatch.setattr(adapter, "_edit_guest_inline_message", edit)

    result = await adapter.send("guest:q1", "final")

    assert result.success is True
    edit.assert_awaited_once_with("inline-1", "final", finalize=True)
