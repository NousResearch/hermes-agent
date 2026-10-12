"""A definitive plain-text approval delivery failure must not park the session (#125950).

The text fallback in ``TurnRunner._approval_notify_sync`` called ``fut.result(timeout=15)``
without looking at the ``SendResult``, swallowed send exceptions, and returned quietly when
scheduling produced no future at all — so the central waiter kept the approval armed and the
guarded command blocked for the full approval timeout over a prompt nobody could see (observed
on Weixin with ``ret=-2`` / ``session not ready``). The button path above already classifies
every outcome through ``_approval_send_outcome``; the text path now shares that classification:

* ``sent`` keeps the existing timeout-notice behaviour;
* ``ambiguous`` (send timeout, lost ack) keeps the request armed — never a duplicate prompt;
* ``declined`` / ``failed`` raise, so ``_await_gateway_decision`` drops the queue entry and
  returns ``notify_failed``, unblocking the command without executing it.

These tests drive the real ``TurnRunner._approval_notify_sync`` through the real central queue.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from types import SimpleNamespace
from typing import Any, List

import pytest

from gateway.platforms.base import SendResult
from tools import approval as _approval
from tools import approval_gateway_wait as wait_mod

SESSION = "agent:main:weixin:dm:2"
APPROVAL = {
    "command": "rm -rf /tmp/x",
    "description": "recursive delete",
    "pattern_key": "k",
}


class _TextAdapter:
    """No ``send_exec_approval`` attribute: the runner must take the plain-text path."""

    typed_command_prefix = "/"

    def __init__(self, *, outcome: Any = None) -> None:
        self.sends: List[str] = []
        self._outcome = (
            outcome
            if outcome is not None
            else SendResult(success=True, message_id="m1")
        )

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send(self, chat_id: str, message: str, **k: Any):
        self.sends.append(message)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _CoroFut:
    """Stand-in for the scheduling future: the loop runs the coroutine eagerly and this future
    is already settled when ``result()`` inspects it (same verdict the real 15s deadline gives)."""

    def __init__(self, coro):
        self._fut = concurrent.futures.Future()
        try:
            self._fut.set_result(asyncio.run(coro))
        except BaseException as exc:  # the send raised before producing a SendResult
            self._fut.set_exception(exc)

    def result(self, timeout=None):
        return self._fut.result(timeout=timeout)


class _NoAckFut:
    """The platform call completes, but its ack never arrives within the send deadline."""

    def __init__(self, coro):
        self._coro = coro

    def result(self, timeout=None):
        asyncio.run(self._coro)
        raise concurrent.futures.TimeoutError()


def _runner(adapter, fut_factory):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter,
        _status_chat_id="C1",
        _status_thread_metadata={},
        session_key=SESSION,
    )
    runner._schedule = lambda coro, _label, loop=None: fut_factory(coro)
    runner._close_native_stream_boundary = lambda _why: None
    return runner


def _clear():
    with _approval._lock:
        _approval._gateway_queues.pop(SESSION, None)
        _approval._gateway_notify_cbs.pop(SESSION, None)


def _dropped_fut(coro):
    """Scheduling failed outright: no future comes back and the coroutine never reaches the loop."""
    coro.close()
    return None


@pytest.mark.parametrize(
    "no_future, send_outcome",
    [
        (False, SendResult(success=False, error="session not ready")),
        (False, RuntimeError("connection reset by peer")),
        (True, None),
        (
            False,
            SendResult(
                success=False,
                error=None,
                raw_response={"success": False, "code": "egress_declined"},
            ),
        ),
    ],
    ids=[
        "failed-send-result",
        "transport-exception",
        "no-future",
        "connector-declined",
    ],
)
def test_definitive_delivery_failure_resolves_notify_failed(
    no_future, send_outcome, monkeypatch
):
    adapter = _TextAdapter(outcome=send_outcome)
    runner = _runner(
        adapter, _dropped_fut if no_future else (lambda coro: _CoroFut(coro))
    )

    def _poll_never_reached(event, session_key, *, interrupt_log):
        pytest.fail(
            "the approval wait must not be entered when the prompt was never delivered"
        )

    monkeypatch.setattr(wait_mod, "_poll_event", _poll_never_reached)
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    _clear()
    try:
        decision = wait_mod._await_gateway_decision(
            SESSION, runner._approval_notify_sync, dict(APPROVAL)
        )
    finally:
        _clear()

    assert decision == {"resolved": False, "choice": None, "notify_failed": True}
    if not no_future:
        assert len(adapter.sends) == 1, (
            "the prompt was attempted once, never duplicated"
        )
    with _approval._lock:
        assert SESSION not in _approval._gateway_queues, (
            "no entry may stay queued for a late answer"
        )


def test_delivered_prompt_stays_armed_and_times_out_once(monkeypatch):
    adapter = _TextAdapter()  # SendResult(success=True)
    runner = _runner(adapter, lambda coro: _CoroFut(coro))
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(
        "gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300
    )
    monkeypatch.setattr(
        wait_mod, "_poll_event", lambda event, session_key, *, interrupt_log: "timeout"
    )
    _clear()
    try:
        decision = wait_mod._await_gateway_decision(
            SESSION, runner._approval_notify_sync, dict(APPROVAL)
        )
    finally:
        _clear()

    assert decision == {"resolved": False, "choice": None, "reason": None}
    # Exactly one prompt plus the timeout notice posted as a new message (no card to edit on text).
    assert len(adapter.sends) == 2, adapter.sends
    assert "/approve" in adapter.sends[0], (
        "the first message must be the approval prompt"
    )


def test_ambiguous_send_keeps_the_prompt_armed_without_resend(monkeypatch):
    adapter = _TextAdapter()
    runner = _runner(adapter, lambda coro: _NoAckFut(coro))
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(
        wait_mod, "_poll_event", lambda event, session_key, *, interrupt_log: "timeout"
    )
    _clear()
    try:
        decision = wait_mod._await_gateway_decision(
            SESSION, runner._approval_notify_sync, dict(APPROVAL)
        )
    finally:
        _clear()

    assert decision == {"resolved": False, "choice": None, "reason": None}
    # The prompt attempt stays armed for a late /approve: no duplicate send, and no timeout
    # notice (that is registered only for a confirmed delivery).
    assert len(adapter.sends) == 1, adapter.sends
