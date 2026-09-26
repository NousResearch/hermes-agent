"""One-shot MCP consent uses the real gateway/Telegram approval route, offline."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import approval, approval_context, approval_prompt
from tools.approval_gateway_wait import _ApprovalEntry, _await_gateway_decision
from tools.mcp_tool_handlers import _trust_gate_check
from tools import mcp_tool as _core


@pytest.mark.parametrize("one_shot", [True, False], ids=["mcp", "terminal"])
@pytest.mark.parametrize("choice", ["once", "session", "always", "deny"])
def test_gateway_scope_reaches_telegram_and_controls_resolution(monkeypatch, one_shot, choice):
    session = "test-mcp-consent-scope"
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_KEY", session)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 2)
    monkeypatch.setattr(_core, "_server_trust_levels", {"test": _core._TRUST_UNTRUSTED})
    monkeypatch.setattr(_core, "_tool_read_only_hints", {})
    monkeypatch.setattr(approval, "_gateway_queues", {})
    monkeypatch.setattr(approval, "_gateway_notify_cbs", {})
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_permanent_approved", set())

    # The gateway test harness may stub the optional Telegram SDK. Keep its
    # value objects observable while exercising the real adapter and callbacks.
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardButton",
                        lambda text, callback_data: SimpleNamespace(text=text, callback_data=callback_data))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
                        lambda rows: SimpleNamespace(inline_keyboard=rows))
    adapter: Any = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=42)
    adapter._app = MagicMock()
    runner: Any = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="12345", _status_thread_metadata={},
        session_key=session,
    )
    runner._close_native_stream_boundary = lambda _why: None

    def schedule(coro, _label):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    runner._schedule = schedule
    prompts = []
    confirmations = []

    def notify(data):
        prompts.append(data)
        runner._approval_notify_sync(data)
        markup = adapter._bot.send_message.call_args.kwargs["reply_markup"]
        buttons = {button.callback_data.split(":")[1]: button
                   for row in markup.inline_keyboard for button in row}
        expected = {"once", "deny"} if one_shot else {"once", "session", "always", "deny"}
        assert set(buttons) == expected
        selected = choice
        if one_shot and choice in {"session", "always"}:
            # Typed /approve always (including bulk or request-id clients) must
            # not consume the one-shot request or report a persistent approval.
            selectors: list[dict[str, Any]] = [{}, {"resolve_all": True}, {"request_id": data["request_id"]}]
            for kwargs in selectors:
                assert approval.resolve_gateway_approval(session, choice, **kwargs) == 0
                assert approval.get_pending_gateway_approval(session) == data
            # Bulk approval still resolves ordinary requests in a mixed queue,
            # without widening or consuming the MCP consent beside them.
            ordinary = _ApprovalEntry({"command": "terminal control"})
            with approval._lock:
                approval._gateway_queues[session].append(ordinary)
            assert approval.resolve_gateway_approval(session, choice, resolve_all=True) == 1
            assert ordinary.event.is_set() and ordinary.result == choice
            assert approval.get_pending_gateway_approval(session) == data
            selected = "deny"
        query = AsyncMock()
        query.data = buttons[selected].callback_data
        query.message = MagicMock(chat_id=12345)
        query.from_user = SimpleNamespace(id=12345, first_name="Tester")
        asyncio.run(adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock()))
        confirmations.append(query.answer.call_args.kwargs["text"])

    approval.register_gateway_notify(session, notify)
    try:
        if one_shot:
            # The trust gate itself re-enters the prompt on every invocation.
            for _ in range(2):
                result = _trust_gate_check("test", "write")
                assert (result is None) == (choice == "once")
            assert len(prompts) == 2
            assert not approval._session_approved
            assert not approval._permanent_approved
            expected_label = "Approved once" if choice == "once" else "Denied"
        else:
            result = _await_gateway_decision(session, notify, {
                "command": "test command", "description": "terminal control",
                "pattern_key": "test", "pattern_keys": ["test"],
            })
            assert result["resolved"] is True
            assert result["choice"] == choice
            expected_label = {"once": "Approved once", "session": "Approved for session",
                              "always": "Approved permanently", "deny": "Denied"}[choice]
        assert len(confirmations) == len(prompts)
        assert all(expected_label in text for text in confirmations)
        assert not approval.has_blocking_approval(session)
    finally:
        approval.unregister_gateway_notify(session)


@pytest.mark.parametrize("choice,typed", [
    ("once", "o"), ("session", "s"), ("always", "a"), ("deny", "d"), ("timeout", None),
])
def test_cli_consent_is_once_only_without_changing_terminal_scopes(monkeypatch, choice, typed):
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr("prompt_toolkit.application.current.get_app_or_none", lambda: None)
    monkeypatch.setattr(approval_prompt, "_read_choice", lambda *_: typed)
    expected = "accept" if choice == "once" else "cancel" if choice == "timeout" else "decline"
    assert approval_prompt.request_elicitation_consent("test", "test", timeout_seconds=2) == expected
    # A nonconforming prompt/legacy client cannot upgrade one-shot consent.
    assert approval_prompt._consent(choice, "cancel") == expected
    assert approval_prompt.prompt_dangerous_approval("test", "test", timeout_seconds=2) == choice
