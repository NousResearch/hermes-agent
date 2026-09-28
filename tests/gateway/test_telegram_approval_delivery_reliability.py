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


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_first", ["retire", "tap"])
async def test_same_message_id_in_two_chats_keeps_approvals_independent(monkeypatch, finish_first):
    a = adapter()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardButton",
        lambda label, callback_data: SimpleNamespace(text=label, callback_data=callback_data))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
        lambda rows: SimpleNamespace(inline_keyboard=rows))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111,222")
    entries = {chat: _ApprovalEntry({"command": "test only"}) for chat in ("111", "222")}
    monkeypatch.setattr(approval, "_gateway_queues", {chat: [entry] for chat, entry in entries.items()})
    cards = {}

    async def send(**kw):
        cards[str(kw["chat_id"])] = kw["reply_markup"]
        return SimpleNamespace(message_id=42, reply_markup=kw["reply_markup"])

    async def tap(chat):
        data = next(b.callback_data for row in cards[chat].inline_keyboard
                    for b in row if b.callback_data.startswith("ea:once:"))
        query = SimpleNamespace(data=data, from_user=SimpleNamespace(id=int(chat), first_name="Tester"),
            message=SimpleNamespace(chat_id=int(chat), message_id=42,
                chat=SimpleNamespace(type="private"), message_thread_id=None),
            answer=AsyncMock(), edit_message_text=AsyncMock())
        await a._handle_callback_query(SimpleNamespace(callback_query=query), None)

    a._bot.send_message.side_effect = send
    for chat in entries:
        await a.send_exec_approval(chat_id=chat, command="test only", session_key=chat)
    if finish_first == "tap":
        await tap("111")
        assert entries["111"].event.is_set()
        # Answering A must not lose B's retirement lookup.
        await a.retire_exec_approval_card(" 222 ", "42")
        await tap("222")
        assert not entries["222"].event.is_set()
    else:
        await a.retire_exec_approval_card(" 111 ", "42")
        assert not entries["222"].event.is_set()
        await tap("222")
        assert entries["222"].event.is_set()
        assert entries["222"].result == "once"
    assert not a._approval_state
    assert not a._approval_message_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("send_elapsed", [0.0, 8.0, 20.0])
async def test_keyboard_repair_uses_remaining_send_budget(monkeypatch, send_elapsed):
    import asyncio
    import plugins.platforms.telegram.adapter as telegram_mod

    a = adapter()
    now = [0.0]
    monkeypatch.setattr(telegram_mod, "time", SimpleNamespace(monotonic=lambda: now[0]))
    budgets = []

    async def send(*args, **kwargs):
        now[0] = send_elapsed
        return SimpleNamespace(message_id=42, reply_markup=None)

    async def bounded(awaitable, timeout, **kwargs):
        awaitable.close()
        budgets.append(timeout)
        raise asyncio.TimeoutError("synthetic slow keyboard repair")

    monkeypatch.setattr(a, "_send_control_message", send)
    monkeypatch.setattr(telegram_mod, "_await_with_thread_deadline", bounded)
    result = await a.send_exec_approval(chat_id="111", command="test only", session_key="s")
    assert not result.success
    assert not a._approval_state
    if send_elapsed >= 15:
        assert budgets == [], "an already-late send must not start a new repair window"
    else:
        assert len(budgets) == 1
        assert 0 < send_elapsed + budgets[0] < 15, "send and repair share the runner's budget"


@pytest.mark.parametrize("buttons", [True, False])
@pytest.mark.parametrize("late_result", ["success", "failure", "exception", "pending", "cancelled"])
def test_ambiguous_send_still_posts_timeout_notice(monkeypatch, buttons, late_result):
    import asyncio
    from gateway.run import _approval_send_outcome

    class TextAdapter:
        def __init__(self):
            self.send = AsyncMock(return_value=SendResult(success=True))
            self.retired = []
        def pause_typing_for_chat(self, chat_id):
            pass
        async def retire_exec_approval_card(self, chat_id, message_id):
            self.retired.append((chat_id, message_id))

    class ButtonAdapter(TextAdapter):
        async def send_exec_approval(self, **kwargs):
            pass

    a = ButtonAdapter() if buttons else TextAdapter()
    entry = _ApprovalEntry({"command": "test only"})
    monkeypatch.setattr(approval, "_gateway_queues", {"s": [entry]})
    runner = TurnRunner(None, SimpleNamespace(_status_adapter=a, _status_chat_id="111",
        _status_thread_metadata=None, session_key="s"))
    monkeypatch.setattr(runner, "_close_native_stream_boundary", lambda *args: None)
    # Real classifier, but exercise the timeout deterministically rather than sleeping 15s.
    monkeypatch.setattr("gateway.run._approval_send_outcome",
        lambda future, timeout: _approval_send_outcome(future, timeout=0))
    future = Future()
    sends = []
    def schedule(coro, label):
        coro.close()
        sends.append(label)
        return future
    monkeypatch.setattr(runner, "_schedule", schedule)
    runner._approval_notify_sync(dict(entry.data))
    assert len(sends) == 1, "possibly delivered prompts must not be sent twice"
    assert entry.settle is not None, "ambiguous delivery still needs a timeout notice"
    if late_result == "exception":
        future.set_exception(RuntimeError("late send failed"))
    elif late_result == "cancelled":
        future.cancel()
    elif late_result != "pending":
        future.set_result(SendResult(success=late_result == "success", message_id="42"))
    monkeypatch.setattr(runner, "_schedule", lambda coro, label: asyncio.run(coro))
    entry.settle("timeout")
    assert a.send.await_count == 1
    assert "NOT run" in a.send.call_args.args[1]
    assert a.retired == ([("111", "42")] if buttons and late_result == "success" else [])
