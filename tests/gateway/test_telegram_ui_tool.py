"""Agent UI effects require an authenticated, still-current Telegram turn."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_context import set_session_vars, clear_session_vars
from plugins.platforms.telegram.telegram_ui_tool import bind_telegram_ui, telegram_ui


@pytest.mark.asyncio
async def test_ui_tool_without_gateway_binding_cannot_choose_destination():
    result = json.loads(await telegram_ui({"action": "send_widget", "chat_id": "123"}))
    assert not result["success"]
    assert "active local Telegram" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stale,actor,authorized", [(True, "42", True), (False, "43", True), (False, "42", False)])
async def test_ui_tool_rejects_stale_foreign_or_revoked_actor(monkeypatch, stale, actor, authorized):
    from plugins.platforms.telegram import telegram_ui_actions
    effect = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(telegram_ui_actions, "execute_ui_action", effect)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42")
    adapter = SimpleNamespace(_bot=object(), _is_callback_user_authorized=lambda *a, **k: authorized)
    runner = SimpleNamespace(_gateway_loop=asyncio.get_running_loop(), _is_session_run_current=lambda *a: not stale,
                             _delivery_adapter_for=lambda s: adapter)
    tokens = set_session_vars(platform="telegram", user_id=actor)
    try:
        with bind_telegram_ui(runner, source, "session", "key", 1):
            result = json.loads(await telegram_ui({"action": "chat_info"}))
    finally:
        clear_session_vars(tokens)
    assert not result["success"]
    effect.assert_not_awaited()


@pytest.mark.asyncio
async def test_ui_tool_uses_gateway_route_not_model_destination(monkeypatch):
    from plugins.platforms.telegram import telegram_ui_actions
    effect = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(telegram_ui_actions, "execute_ui_action", effect)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", thread_id="7", profile="secondary")
    adapter = SimpleNamespace(_bot=object(), _is_callback_user_authorized=lambda *a, **k: True)
    runner = SimpleNamespace(_gateway_loop=asyncio.get_running_loop(), _is_session_run_current=lambda *a: True,
                             _delivery_adapter_for=lambda s: adapter)
    tokens = set_session_vars(platform="telegram", user_id="42")
    try:
        with bind_telegram_ui(runner, source, "session", "key", 1):
            result = json.loads(await telegram_ui({"action": "chat_info"}))
    finally:
        clear_session_vars(tokens)
    assert result["success"]
    owner = effect.call_args.args[1]
    assert (owner["chat_id"], owner["thread_id"], owner["profile"], owner["session_id"]) == ("42", "7", "secondary", "session")
