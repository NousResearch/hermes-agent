"""Real approval/foreground/Matrix boundaries; transport only is substituted."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt
from tools import approval, approval_context
from tools.approval_gateway_wait import _await_gateway_decision


def adapter_for_test(monkeypatch):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@owner:example.org,@other:example.org")
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
    adapter._client = SimpleNamespace()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$card"))
    adapter._send_reaction = AsyncMock(return_value="$reaction")
    return adapter


def foreground(adapter, loop):
    runner = TurnRunner.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="!room:example.org", session_key="boundary", _status_thread_metadata={}, source=SimpleNamespace(user_id="@owner:example.org"))
    runner._close_native_stream_boundary = lambda _: None
    runner._schedule = lambda coro, _: asyncio.run_coroutine_threadsafe(coro, loop)
    return runner


@pytest.mark.asyncio
async def test_foreground_core_deadline_and_requester_reach_real_card(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 900)
    adapter = adapter_for_test(monkeypatch)
    adapter._approval_timeout_seconds = 1
    adapter._schedule_approval_resolution_watch = lambda _: None
    runner = foreground(adapter, asyncio.get_running_loop())
    seen = {}

    def notify(data):
        runner._approval_notify_sync(data)
        seen.update(data)
        assert approval.resolve_gateway_approval("boundary", "deny", approval_id=data["approval_id"]) == 1

    try:
        result = await asyncio.to_thread(_await_gateway_decision, "boundary", notify, {"command": "echo boundary"})
        assert result["choice"] == "deny"
        prompt = adapter._approval_prompts_by_event["$card"]
        assert prompt.requester_user_id == "@owner:example.org"
        assert prompt.expires_at == seen["expires_at"]
        assert prompt.expires_at > __import__("time").monotonic() + 800
        assert not await adapter._validate_matrix_prompt_reactor(prompt.chat_id, prompt.message_id, "@other:example.org", prompt, "approval")
        assert await adapter._validate_matrix_prompt_reactor(prompt.chat_id, prompt.message_id, "@owner:example.org", prompt, "approval")
    finally:
        approval.unregister_gateway_notify("boundary")


@pytest.mark.asyncio
async def test_failed_foreground_full_fallback_returns_notify_failed(monkeypatch):
    adapter = adapter_for_test(monkeypatch)
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="too large"))
    runner = foreground(adapter, asyncio.get_running_loop())
    # A broken notifier must not block the test for the default approval timeout.
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 0.01)
    result = await asyncio.to_thread(_await_gateway_decision, "boundary", runner._approval_notify_sync, {"command": "x" * 70000})
    assert result.get("notify_failed") is True
    assert not approval.has_blocking_approval("boundary")
    assert adapter.send.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_terminal_delivery_failure_or_cancellation_retains_retryable_card(monkeypatch, cancel):
    adapter = adapter_for_test(monkeypatch)
    prompt = _MatrixApprovalPrompt("boundary", "!room:example.org", "$card", command="echo boundary", resolved=True)
    adapter._approval_prompts_by_event["$card"] = prompt
    adapter._approval_prompt_by_session["boundary"] = {"$card"}
    adapter.edit_message = AsyncMock(side_effect=asyncio.CancelledError() if cancel else None, return_value=SendResult(success=False, error="offline"))
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="offline"))
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await adapter._finalize_matrix_approval_prompt(prompt.chat_id, "$card", prompt, choice="deny", max_attempts=1)
    else:
        await adapter._finalize_matrix_approval_prompt(prompt.chat_id, "$card", prompt, choice="deny", max_attempts=1)
    assert adapter._approval_prompts_by_event["$card"] is prompt
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="$edit"))
    await adapter._finalize_matrix_approval_prompt(prompt.chat_id, "$card", prompt, choice="deny", max_attempts=1)
    adapter.edit_message.assert_awaited_once()
    assert prompt.state == "terminal_deny"
    assert "$card" not in adapter._approval_prompts_by_event


def test_notification_latency_consumes_core_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("tools.approval_gateway_wait.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 900)
    seen = {}
    def notify(data):
        seen.update(data)
        clock[0] += 901
    class Event:
        def wait(self, timeout):
            pytest.fail("core restarted timeout after notification")
        def set(self):
            pass
    monkeypatch.setattr("tools.approval_gateway_wait.threading.Event", Event)
    try:
        result = _await_gateway_decision("deadline", notify, {"command": "echo deadline"})
        assert seen["expires_at"] == 1000
        assert not result["resolved"]
    finally:
        approval.unregister_gateway_notify("deadline")


@pytest.mark.parametrize("by_id", [False, True])
def test_core_rejects_expired_request_during_notification(monkeypatch, by_id):
    clock = [100.0]
    monkeypatch.setattr("tools.approval_gateway_wait.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 900)
    def notify(data):
        assert data["expires_at"] == 1000
        clock[0] = data["expires_at"]
        kwargs = {"approval_id": data["approval_id"]} if by_id else {}
        assert approval.resolve_gateway_approval("expired", "once", **kwargs) == 0
    result = _await_gateway_decision("expired", notify, {"command": "echo expired"})
    assert result["choice"] is None
    assert not approval.has_blocking_approval("expired")


@pytest.mark.asyncio
async def test_visible_failure_notice_allows_cleanup(monkeypatch):
    adapter = adapter_for_test(monkeypatch)
    prompt = _MatrixApprovalPrompt("notice", "!room:example.org", "$card", command="echo test", resolved=True)
    adapter._approval_prompts_by_event["$card"] = prompt
    adapter._approval_prompt_by_session["notice"] = {"$card"}
    adapter.edit_message = AsyncMock(return_value=SendResult(success=False, error="offline"))
    await adapter._finalize_matrix_approval_prompt(prompt.chat_id, "$card", prompt, choice="expired", max_attempts=1)
    assert prompt.terminal_visible
    assert "$card" not in adapter._approval_prompts_by_event
    assert "expired" in adapter.send.call_args.args[1]


@pytest.mark.asyncio
async def test_watcher_retries_cancelled_terminal_without_losing_decision(monkeypatch):
    adapter = adapter_for_test(monkeypatch)
    prompt = _MatrixApprovalPrompt("watch-retry", "!room:example.org", "$card", command="echo test", resolved=True)
    adapter._approval_prompts_by_event["$card"] = prompt
    adapter._approval_prompt_by_session["watch-retry"] = {"$card"}
    adapter.edit_message = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await adapter._finalize_matrix_approval_prompt(prompt.chat_id, "$card", prompt, choice="deny", actor="@owner:example.org", max_attempts=1)
    delivered = asyncio.Event()
    async def edit(*args, **kwargs):
        assert "Denied" in args[2]
        assert "@owner:example.org" in args[2]
        delivered.set()
        return SendResult(success=True, message_id="$edit")
    adapter.edit_message = edit
    adapter._schedule_approval_resolution_watch(prompt)
    await asyncio.wait_for(delivered.wait(), timeout=1)
    assert prompt.terminal_visible
    assert "$card" not in adapter._approval_prompts_by_event
