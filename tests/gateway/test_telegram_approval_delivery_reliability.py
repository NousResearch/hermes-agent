"""Approval cards must be actionable, including taps before the send acknowledgement."""
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


def adapter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    a._bot = AsyncMock()
    a._app = MagicMock()
    return a


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,authorized", [("once", True), ("deny", True), ("once", False)])
async def test_early_tap_resolves_only_authenticated_pending_request(monkeypatch, choice, authorized):
    a = adapter()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardButton",
        lambda label, callback_data: SimpleNamespace(text=label, callback_data=callback_data))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
        lambda rows: SimpleNamespace(inline_keyboard=rows))
    session = "agent:main:telegram:dm:12345"
    entry = _ApprovalEntry({"command": "test only", "pattern_key": "test-only"})
    monkeypatch.setattr(approval, "_gateway_queues", {session: [entry]})
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    async def send(**kw):
        data = next(b.callback_data for row in kw["reply_markup"].inline_keyboard
                    for b in row if b.callback_data.startswith(f"ea:{choice}:"))
        query = SimpleNamespace(data=data, from_user=SimpleNamespace(id=111 if authorized else 222, first_name="Tester"),
            message=SimpleNamespace(chat_id=12345, chat=SimpleNamespace(type="private"), message_thread_id=None),
            answer=AsyncMock(), edit_message_text=AsyncMock())
        # Telegram has rendered the card, but sendMessage's HTTP acknowledgement is still in flight.
        await a._handle_callback_query(SimpleNamespace(callback_query=query), None)
        assert entry.event.is_set() is authorized
        assert entry.result == (choice if authorized else None)
        return SimpleNamespace(message_id=42, reply_markup=kw["reply_markup"])

    a._bot.send_message.side_effect = send
    result = await a.send_exec_approval(chat_id="12345", command="test only", session_key=session)
    assert result.success
    assert bool(a._approval_state) is (not authorized), "late ACK must not resurrect a resolved card"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_ok", [True, False])
async def test_missing_keyboard_receipt_is_repaired_or_reported_as_failure(repair_ok):
    a = adapter()
    a._bot.send_message.return_value = SimpleNamespace(message_id=42, reply_markup=None)

    async def repair(**kw):
        if not repair_ok:
            raise RuntimeError("transport rejected keyboard")
        return SimpleNamespace(message_id=42, reply_markup=kw["reply_markup"])

    a._bot.edit_message_reply_markup.side_effect = repair
    result = await a.send_exec_approval(chat_id="12345", command="test only", session_key="s")
    assert result.success is repair_ok
    assert a._bot.edit_message_reply_markup.await_count == 1
    assert bool(a._approval_state) is repair_ok
    text = a._bot.send_message.call_args.kwargs["text"]
    assert "/approve" in text and "/deny" in text, "visible text remains usable when a client hides buttons"


@pytest.mark.asyncio
async def test_send_failure_does_not_leave_callback_state():
    a = adapter()
    a._bot.send_message.side_effect = RuntimeError("transport unavailable")
    result = await a.send_exec_approval(chat_id="12345", command="test only", session_key="s")
    assert not result.success
    assert not a._approval_state


@pytest.mark.parametrize("failure", ["error_result", "exception", "no_loop"])
def test_failed_text_delivery_unblocks_as_notify_failed_not_human_timeout(monkeypatch, failure):
    class TextAdapter:
        def pause_typing_for_chat(self, chat_id):
            pass
        async def send(self, *args, **kwargs):
            pass
    runner = TurnRunner(None, SimpleNamespace(_status_adapter=TextAdapter(), _status_chat_id="12345",
        _status_thread_metadata=None, session_key="s"))
    monkeypatch.setattr(runner, "_close_native_stream_boundary", lambda *a: None)

    flushed = []
    runner._ctx.stream_consumer_holder = [SimpleNamespace(flush_pending_sync=lambda **kw: flushed.append(True))]
    def schedule(coro, *args):
        assert flushed.pop(), "pending prose must be flushed before the approval send"
        coro.close()
        if failure == "no_loop":
            return None
        f = Future()
        if failure == "exception":
            f.set_exception(RuntimeError("send failed"))
        else:
            f.set_result(SendResult(success=False, error="send failed"))
        return f
    monkeypatch.setattr(runner, "_schedule", schedule)
    with pytest.raises(RuntimeError, match="approval.*undeliverable"):
        runner._approval_notify_sync({"command": "test only", "request_id": "test"})

    from tools.approval_gateway_wait import _await_gateway_decision
    def unexpected_wait(*args, **kwargs):
        pytest.fail("undelivered prompt must not start a human-response timer")
    monkeypatch.setattr("tools.approval_gateway_wait._poll_event", unexpected_wait)
    monkeypatch.setattr(approval, "_gateway_queues", {})
    decision = _await_gateway_decision("s", runner._approval_notify_sync,
        {"command": "test only", "pattern_key": "test-only"})
    assert decision.get("notify_failed") is True
    assert not decision["resolved"]
    assert not approval._gateway_queues


@pytest.mark.asyncio
async def test_expiry_preserves_request_and_removes_only_its_buttons():
    from gateway.run_turn_runner_approval_settle import _post_timeout_notice
    a = adapter()
    async def send(**kw):
        return SimpleNamespace(message_id=42, reply_markup=kw["reply_markup"])
    a._bot.send_message.side_effect = send
    await a.send_exec_approval(chat_id="12345", command="test only", session_key="s")
    a.send = AsyncMock(return_value=SendResult(success=True))
    a.edit_message = AsyncMock(return_value=SendResult(success=True))
    ctx = SimpleNamespace(_status_adapter=a, _status_chat_id="12345", _status_thread_metadata=None)
    await _post_timeout_notice(ctx, "test only", "42", 300)
    a.edit_message.assert_not_awaited()
    a._bot.edit_message_reply_markup.assert_awaited_once_with(chat_id=12345, message_id=42, reply_markup=None)
    assert not a._approval_state
    assert not a._approval_message_ids
    assert a.send.call_args.kwargs["metadata"]["notify"] is True
    assert a.send.call_args.kwargs["metadata"]["_interim_send"] is True


@pytest.mark.asyncio
async def test_cancelled_send_drops_preregistered_callback():
    import asyncio
    a = adapter()
    a._bot.send_message.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await a.send_exec_approval(chat_id="12345", command="test only", session_key="s")
    assert not a._approval_state
