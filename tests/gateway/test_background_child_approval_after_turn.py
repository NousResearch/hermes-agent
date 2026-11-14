"""A background child's approval after its parent turn ended still reaches the chat (#133514).

The parent turn's ``finally`` unregisters the session's notifier. Before the fix, a ``delegate_task``
child that asked for approval after that got ``pending_approval``, no card was sent, and ``/approve``
found nothing. These tests drive the real ``TurnRunner._run_conversation_with_approval`` (which
registers and unregisters the notifier) and the real ``dispatch_async_delegation`` (which takes the
child's lease): the child asks only after the parent's run returned.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

# Module level on purpose: gateway.run's import side effects stat the checkout's git dir, which the
# fixture-scoped home-I/O guard would refuse inside a test.
import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tools import approval, async_delegation as ad

SESSION = "agent:main:telegram:dm:111"


class _Adapter:
    """Records every prompt; a card send returns the card's id."""

    typed_command_prefix = "/"

    def __init__(self, name: str) -> None:
        self.name, self.calls = name, []

    def max_message_length_for_chat(self, chat_id: str) -> int:
        return 4096

    def message_len_fn_for_chat(self, chat_id: str):
        return len

    def pause_typing_for_chat(self, chat_id: str) -> None:
        raise AssertionError("a background prompt must not touch the finished turn's typing state")

    async def send_exec_approval(self, **kw: Any) -> SendResult:
        self.calls.append(("card", kw["chat_id"], kw.get("request_id")))
        return SendResult(success=True, message_id=f"{self.name}-card")

    async def send(self, chat_id: str, text: str, **_kw: Any) -> SendResult:
        self.calls.append(("text", chat_id, None))
        return SendResult(success=True, message_id=f"{self.name}-text")

    async def edit_message(self, chat_id: str, message_id: str, content: str, **_kw: Any) -> SendResult:
        self.calls.append(("edit", chat_id, message_id))
        return SendResult(success=True, message_id=message_id)


class _CanonicalOwner:
    """The parent turn's canonical owner: refuses once the parent's generation settled, as
    ``SessionAuthority.check_approval_generation`` does, and records every use."""

    def __init__(self) -> None:
        self.settled, self.calls = False, []

    def check_approval_generation(self, session_id, generation):
        self.calls.append("check")
        if self.settled:
            raise RuntimeError("stale_generation")

    def register_approval(self, *a, **k):
        self.calls.append("register")
        self.check_approval_generation(None, None)


@pytest.fixture
def gate(monkeypatch):
    """Manual approvals on a messaging surface; one shared child-ask latch per test."""
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 10)
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, True, False))
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 10)
    ad._reset_for_tests()
    yield
    ad.interrupt_all()
    ad._reset_for_tests()
    approval.unregister_gateway_notify(SESSION)
    with approval._lock:
        approval._gateway_queues.pop(SESSION, None)
        approval._pending.pop(SESSION, None)


def _turn(platform: Platform, adapter: _Adapter, loop: asyncio.AbstractEventLoop) -> TurnRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {platform: adapter}
    runner.config = runner.hooks = runner.session_authority = None
    runner._consume_pending_native_image_paths = lambda key: []
    source = SessionSource(platform=platform, chat_id="111", chat_type="dm", user_id="111")
    ctx = TurnContext(source=source, session_key=SESSION, session_id="s1", user_config={},
                      message="go", _run_still_current=lambda: True)
    ctx.bind_status_adapter(lambda: runner._delivery_adapter_for(source))
    ctx._status_chat_id, ctx._loop_for_step = "111", loop
    return TurnRunner(runner, ctx)


def _parent_dispatches_child_then_ends(turn: TurnRunner, child) -> dict:
    """The parent's run dispatches a background child and returns; the notifier is unregistered."""
    dispatched = {}

    def run_conversation(message, **_kw):
        dispatched.update(ad.dispatch_async_delegation(
            goal="clean up", context=None, toolsets=None, role="leaf", model=None,
            session_key=SESSION, runner=child))
        return {"final_response": "started a helper"}

    turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run_conversation), [], None, None, None)
    assert dispatched["status"] == "dispatched"
    assert approval._gateway_notify_cb(SESSION) is None, "the parent turn ended: its notifier is gone"
    return dispatched


def _loop_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


@pytest.mark.parametrize("answer", ["once", "deny"])
def test_child_approval_after_the_parent_turn_reaches_the_chat_and_resolves(gate, answer):
    loop, thread = _loop_thread()
    first, live = _Adapter("first"), _Adapter("live")
    ask, done, decisions = threading.Event(), threading.Event(), []

    def child():
        assert ask.wait(5)
        decisions.append(approval.check_execute_code_guard("print('x')", "local"))
        done.set()
        return {"status": "completed"}

    try:
        turn = _turn(Platform.TELEGRAM, first, loop)
        _parent_dispatches_child_then_ends(turn, child)
        turn._runner.adapters[Platform.TELEGRAM] = live  # a reconnect after the turn: the live one sends
        ask.set()
        deadline = threading.Event()
        for _ in range(50):
            if approval.has_blocking_approval(SESSION) and live.calls:
                break
            deadline.wait(0.1)
        assert [c[0] for c in live.calls] == ["card"], "the child's approval card must reach the chat"
        assert first.calls == []
        pending = approval.list_gateway_approvals(SESSION)
        assert [p["request_id"] for p in pending] == [live.calls[0][2]], "/approve must find the request"
        assert approval.resolve_gateway_approval(SESSION, answer) == 1
        assert done.wait(5)
        assert decisions[0]["approved"] is (answer == "once")
        assert not approval.has_blocking_approval(SESSION)
    finally:
        ask.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


def test_child_protected_file_write_after_the_parent_turn_is_asked_not_refused(gate):
    """The AGENTS.md/SOUL.md write gate asks through the same lease as the command gates: a child
    writing one after the parent replied gets a card, and the tap decides the write."""
    from tools.file_tools_write_guards import _request_protected_instruction_approval

    loop, thread = _loop_thread()
    adapter = _Adapter("live")
    ask, done, results = threading.Event(), threading.Event(), []

    def child():
        assert ask.wait(5)
        results.append(_request_protected_instruction_approval(["AGENTS.md: agent instructions"]))
        done.set()
        return {"status": "completed"}

    try:
        _parent_dispatches_child_then_ends(_turn(Platform.TELEGRAM, adapter, loop), child)
        ask.set()
        _wait(lambda: approval.has_blocking_approval(SESSION) and adapter.calls, "the write's card was sent")
        assert [c[0] for c in adapter.calls] == ["card"]
        assert approval.resolve_gateway_approval(SESSION, "once") == 1
        assert done.wait(5)
        assert results == [None], "an approved protected write proceeds"
    finally:
        ask.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


def test_child_without_a_chat_fails_closed_instead_of_pending(gate):
    """An API-server parent leaves its child no chat once the run is over: fail closed, nothing queued."""
    loop, thread = _loop_thread()
    adapter = _Adapter("api")
    ask, done, decisions = threading.Event(), threading.Event(), []

    def child():
        assert ask.wait(5)
        decisions.append(approval.check_execute_code_guard("print('x')", "local"))
        done.set()
        return {"status": "completed"}

    try:
        _parent_dispatches_child_then_ends(_turn(Platform.API_SERVER, adapter, loop), child)
        ask.set()
        assert done.wait(5)
        assert decisions[0]["approved"] is False
        assert decisions[0].get("status") != "pending_approval"
        assert decisions[0]["outcome"] == "notify_failed"
        assert adapter.calls == []
        assert not approval.has_blocking_approval(SESSION)
        assert SESSION not in approval._pending
    finally:
        ask.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


def _wait(predicate, what):
    pause = threading.Event()
    for _ in range(80):
        if predicate():
            return
        pause.wait(0.1)
    raise AssertionError(what)


def test_child_card_raised_mid_turn_survives_the_parent_settling(gate, monkeypatch):
    """The child asks while the parent turn still runs; the parent then ends (its canonical generation
    settles) before anyone answers. The card must not touch the parent's generation, stays answerable,
    and its timeout notice still fires although a newer turn owns the chat by then."""
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 1)
    loop, thread = _loop_thread()
    adapter, owner = _Adapter("live"), _CanonicalOwner()
    ask, asked, done, decisions = threading.Event(), threading.Event(), threading.Event(), []

    def child():
        assert ask.wait(5)
        asked.set()
        decisions.append(approval.check_execute_code_guard("print('x')", "local"))
        done.set()
        return {"status": "completed"}

    try:
        turn = _turn(Platform.TELEGRAM, adapter, loop)
        turn._approval_owner = (owner, "s1", 1)
        turn._ctx._run_still_current = lambda: not owner.settled  # a successor turn takes the chat

        def run_conversation(message, **_kw):
            ad.dispatch_async_delegation(goal="clean up", context=None, toolsets=None, role="leaf",
                                         model=None, session_key=SESSION, runner=child)
            ask.set()
            _wait(lambda: adapter.calls, "the child's card was sent while the parent ran")
            return {"final_response": "started a helper"}

        turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run_conversation), [], None, None, None)
        owner.settled = True  # the parent's generation is gone
        assert owner.calls == [], "a child's prompt must never project onto the parent's generation"
        assert approval.has_blocking_approval(SESSION), "the parent ending must not withdraw the child's card"
        assert done.wait(5)
        assert decisions[0]["outcome"] == "timeout"
        # The card send was the first call; the timeout notice is a second one (edit or new message).
        _wait(lambda: any(c[0] == "edit" for c in adapter.calls), "the expired card was edited to say so")
    finally:
        ask.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
