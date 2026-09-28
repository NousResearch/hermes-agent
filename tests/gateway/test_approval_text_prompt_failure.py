"""Approval text-prompt failures must be loud and fail closed (t_f6d13263).

``TurnRunner._approval_notify_sync``'s plain-text path (adapters without native
approval buttons) used to ignore both scheduling failures and failed
``SendResult``s: the queue entry then waited out the full ``approvals.timeout``
while the chat never showed a prompt and nothing was logged above DEBUG — four
silent 300s wedges on a WhatsApp himalaya send. These tests drive the real
``_approval_notify_sync`` with a text-only adapter through every verdict.

The queue-entry drop itself is ``_await_gateway_decision``'s notify-failure
contract (a raising notify_cb → ``notify_failed`` → entry dropped, tool
unblocked) and is covered end-to-end by ``test_notify_failed_unblocks_the_wait``
below, which drives the real wait machinery.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from types import SimpleNamespace
from typing import Any, List

import pytest

from gateway.platforms.base import SendResult
from gateway.relay.egress import EGRESS_DECLINE_CODE
from tools import approval as _approval
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:whatsapp:dm:31629030434"
APPROVAL = {"command": "himalaya message send 1 --send", "description": "mail",
            "pattern_key": "k"}


class _TextAdapter:
    """Text-only adapter (no native approval buttons), recording sends."""

    typed_command_prefix = "/"

    def __init__(self, result: SendResult = SendResult(success=True, message_id="m1")):
        self._result = result
        self.sends: List[str] = []

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send(self, chat_id: str, message: str, **k: Any) -> SendResult:
        self.sends.append(message)
        return self._result


class _Fut:
    def __init__(self, result=None, delay: float = 0.0, exc: Exception = None):
        self._result = result
        self._delay = delay
        self._exc = exc

    def result(self, timeout=None):
        if self._delay and timeout is not None and self._delay > timeout:
            raise concurrent.futures.TimeoutError()
        if self._exc is not None:
            raise self._exc
        return self._result


def _runner(adapter: _TextAdapter, schedule):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata={},
        session_key=SESSION,
        source=SimpleNamespace(chat_id="C1", platform="whatsapp", session_key=SESSION),
    )
    runner._schedule = schedule
    runner._close_native_stream_boundary = lambda _why: None
    return runner


def _pending_entry():
    entry = _ApprovalEntry(dict(APPROVAL))
    with _approval._lock:
        _approval._gateway_queues[SESSION] = [entry]
    return entry


def _clear_queue():
    with _approval._lock:
        _approval._gateway_queues.pop(SESSION, None)


def _direct(coro, _label):
    """The real ``_schedule`` runs the coroutine on the gateway loop and hands back a
    concurrent.futures.Future; here the coroutine runs to completion inline."""
    return _Fut(asyncio.run(coro))


def test_no_loop_raises():
    """The scheduling failure (fut is None) that silently wedged the wait for 300s."""
    from gateway.run_turn_runner import _ExecApprovalUndeliverable

    adapter = _TextAdapter()
    runner = _runner(adapter, lambda coro, _label: (coro.close(), None)[1])

    with pytest.raises(_ExecApprovalUndeliverable):
        runner._approval_notify_sync(dict(APPROVAL))

    assert adapter.sends == []


def test_failed_send_result_raises():
    """A completed-but-failed send (bridge 503, 'Not connected') is definitive: nobody
    can answer a prompt that never landed."""
    from gateway.run_turn_runner import _ExecApprovalUndeliverable

    adapter = _TextAdapter(SendResult(success=False, error="Not connected to WhatsApp"))
    runner = _runner(adapter, _direct)

    with pytest.raises(_ExecApprovalUndeliverable):
        runner._approval_notify_sync(dict(APPROVAL))

    assert len(adapter.sends) == 1, "the send was attempted (and classified failed)"


def test_send_exception_raises():
    from gateway.run_turn_runner import _ExecApprovalUndeliverable

    adapter = _TextAdapter()

    def _schedule(coro, _label):
        coro.close()
        return _Fut(exc=RuntimeError("aiohttp boom"))

    runner = _runner(adapter, _schedule)

    with pytest.raises(_ExecApprovalUndeliverable):
        runner._approval_notify_sync(dict(APPROVAL))


def test_ambiguous_timeout_keeps_the_entry_armed_and_arms_the_notice():
    """A send timeout is AMBIGUOUS (may have posted with a late ack): the registration
    stays armed and the timeout notice must fire when nobody answers."""
    entry = _pending_entry()
    adapter = _TextAdapter()

    runner = _runner(adapter, lambda coro, _label: _Fut(delay=60.0))

    runner._approval_notify_sync(dict(entry.data))  # must NOT raise

    with _approval._lock:
        assert SESSION in _approval._gateway_queues, "ambiguous send keeps the entry answerable"
    assert entry.settle is not None, "timeout notice armed"
    entry.settle("timeout")
    _clear_queue()


def test_successful_send_arms_the_notice():
    entry = _pending_entry()
    adapter = _TextAdapter()

    runner = _runner(adapter, _direct)
    runner._approval_notify_sync(dict(entry.data))

    assert len(adapter.sends) == 1
    assert "/approve" in adapter.sends[0]
    assert entry.settle is not None
    _clear_queue()


def test_declined_text_send_raises_declined():
    """A connector egress decline on the TEXT lane must not re-send into the refused chat."""
    from gateway.run_turn_runner import _ExecApprovalDeclined

    adapter = _TextAdapter(SendResult(success=False, error="declined",
                                      raw_response={"success": False, "code": EGRESS_DECLINE_CODE}))
    runner = _runner(adapter, _direct)

    with pytest.raises(_ExecApprovalDeclined):
        runner._approval_notify_sync(dict(APPROVAL))
    assert len(adapter.sends) == 1, "one text attempt, no re-send into the refused chat"


def test_notify_failed_unblocks_the_wait():
    """End-to-end: a raising notify_cb must drop the queue entry and return
    ``notify_failed`` so the blocked tool unblocks instead of wedging 300s."""
    from gateway.run_turn_runner import _ExecApprovalUndeliverable
    from tools.approval_gateway_wait import _await_gateway_decision

    def notify_cb(data):
        raise _ExecApprovalUndeliverable("exec approval text prompt could not be delivered")

    result = _await_gateway_decision(SESSION, notify_cb, dict(APPROVAL))

    assert result == {"resolved": False, "choice": None, "notify_failed": True}
    with _approval._lock:
        assert SESSION not in _approval._gateway_queues
