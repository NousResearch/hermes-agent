"""Text-fallback approval prompts classify their send like the button path does.

An adapter without native buttons gets the plain-text prompt with ``/approve`` steps. The send
outcome decides what the shared waiter (``_await_gateway_decision``) sees:

* sent       -> the timeout notice is armed (a new message: the text prompt has no card to edit)
* ambiguous  -> the send timed out but may have posted: keep the exact waiter, never re-send
* failed     -> the render raises, so the waiter drops its entry and returns ``notify_failed``
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.platforms.base import SendResult
from tools import approval as _approval
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:whatsapp:dm:1"
APPROVAL = {"command": "rm -rf /tmp/x", "description": "recursive delete", "pattern_key": "k", "request_id": "r1"}


class _TextAdapter:
    """No native buttons (``send_exec_approval is None`` per-class): the plain-text fallback."""

    typed_command_prefix = "/"
    send_exec_approval = None

    def __init__(self) -> None:
        self.sends: list[str] = []

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send(self, chat_id: str, message: str, **k: Any) -> SendResult:
        self.sends.append(message)
        return SendResult(success=True, message_id="m1")


class _Fut:
    """Stands in for the scheduling future: returns, times out or raises on ``result``."""

    def __init__(self, *, result: Any = None, error: BaseException | None = None) -> None:
        self._result, self._error = result, error

    def result(self, timeout: float | None = None) -> Any:
        if self._error is not None:
            raise self._error
        return self._result


def _runner(adapter, schedule):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata=None,
        session_key=SESSION, source=SimpleNamespace(chat_id="C1", platform="whatsapp", session_key=SESSION),
    )
    runner._schedule = schedule
    runner._close_native_stream_boundary = lambda _why: None
    return runner


@pytest.fixture
def pending_entry(monkeypatch):
    """A queued approval entry exactly as ``_await_gateway_decision`` registers it."""
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
    entry = _ApprovalEntry(dict(APPROVAL))
    with _approval._lock:
        _approval._gateway_queues[SESSION] = [entry]
    yield entry
    with _approval._lock:
        _approval._gateway_queues.pop(SESSION, None)


def _scheduling(outcome: _Fut, scheduled: list):
    def schedule(coro, _label):
        scheduled.append(coro)
        coro.close()
        return outcome

    return schedule


def test_sent_text_prompt_arms_the_timeout_notice(pending_entry):
    adapter = _TextAdapter()
    runner = _runner(adapter, lambda coro, _label: _Fut(result=asyncio.run(coro)))  # the real adapter send runs

    runner._render_approval_sync(dict(pending_entry.data))

    (text,) = adapter.sends
    assert "rm -rf /tmp/x" in text and "/approve" in text
    assert pending_entry.settle is not None, "a delivered text prompt must register its timeout notice"


def test_ambiguous_send_keeps_the_waiter_armed_without_resending(pending_entry):
    scheduled: list = []
    adapter = _TextAdapter()
    runner = _runner(adapter, _scheduling(_Fut(error=concurrent.futures.TimeoutError()), scheduled))

    runner._render_approval_sync(dict(pending_entry.data))  # must not raise

    assert len(scheduled) == 1, "an ambiguous send must never be repeated"
    assert pending_entry.settle is None
    with _approval._lock:
        assert _approval._gateway_queues[SESSION] == [pending_entry], "the waiter stays armed"


@pytest.mark.parametrize(
    "outcome",
    [
        _Fut(error=RuntimeError("transport down")),
        _Fut(result=SendResult(success=False, error="blocked by recipient")),
        None,
    ],
    ids=["send-raises", "send-returns-failure", "no-scheduling-future"],
)
def test_definitive_failure_raises_so_the_waiter_can_drop_its_entry(pending_entry, outcome):
    scheduled: list = []
    runner = _runner(_TextAdapter(), _scheduling(outcome, scheduled))

    with pytest.raises(Exception):
        runner._render_approval_sync(dict(pending_entry.data))

    assert len(scheduled) == 1, "a failed text send is reported, not retried"
    assert pending_entry.settle is None


def test_undelivered_text_prompt_makes_the_shared_waiter_return_notify_failed(monkeypatch):
    """End to end through ``_await_gateway_decision``: no prompt reached the user, so nothing stays queued."""
    from tools.approval_gateway_wait import _await_gateway_decision

    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 5)
    runner = _runner(_TextAdapter(), _scheduling(_Fut(error=RuntimeError("transport down")), []))

    decision = _await_gateway_decision(SESSION, runner._approval_notify_sync, dict(APPROVAL), surface="gateway")

    assert decision.get("notify_failed") is True
    with _approval._lock:
        assert not _approval._gateway_queues.get(SESSION)
