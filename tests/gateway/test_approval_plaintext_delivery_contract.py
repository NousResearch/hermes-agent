"""A failed exec-approval delivery must release the guarded operation promptly.

Drive the real TurnRunner callback through ``_await_gateway_decision`` so the
test covers both the delivery classification and the central queue cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from tools import approval as _approval
from tools import approval_gateway_wait as _approval_wait

SESSION = "agent:main:telegram:dm:approval-delivery-contract"
APPROVAL = {
    "command": "rm -rf /private/sensitive",
    "description": "recursive delete",
    "pattern_key": "dangerous",
    "pattern_keys": ["dangerous"],
}


class _ResultFuture:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def done(self) -> bool:
        return True

    def result(self, timeout=None):
        if self._error is not None:
            raise self._error
        return self._result


class _Adapter:
    typed_command_prefix = "/"

    def __init__(self, text_mode: str) -> None:
        self.text_mode = text_mode
        self.text_sends: list[str] = []

    def pause_typing_for_chat(self, _chat_id: str) -> None:
        return None

    async def send_exec_approval(self, **_kwargs) -> SendResult:
        return SendResult(success=False, error="button transport unavailable")

    async def send(self, _chat_id: str, message: str, **_kwargs) -> SendResult:
        self.text_sends.append(message)
        if self.text_mode == "send_result":
            return SendResult(success=False, error="Telegram send failed")
        if self.text_mode == "send_exception":
            raise RuntimeError("Telegram transport failed")
        if self.text_mode == "send_timeout_exception":
            raise TimeoutError("Telegram request timed out")
        return SendResult(success=True, message_id="text-1")


def _runner(adapter: _Adapter, text_mode: str):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter,
        _status_chat_id="chat-1",
        _status_thread_metadata={"thread_id": "thread-1"},
        session_key=SESSION,
        source=SimpleNamespace(
            chat_id="chat-1", platform="telegram", session_key=SESSION
        ),
    )
    calls = 0

    def schedule(coro, _label):
        nonlocal calls
        calls += 1
        if calls == 2 and text_mode == "missing_future":
            coro.close()
            return None
        if calls == 2 and text_mode == "schedule_timeout":
            coro.close()
            raise TimeoutError("scheduler timed out")
        try:
            return _ResultFuture(asyncio.run(coro))
        except Exception as exc:
            return _ResultFuture(error=exc)

    runner._schedule = schedule
    runner._close_native_stream_boundary = lambda _reason: None
    return runner


@pytest.fixture
def clean_approval_state(monkeypatch):
    monkeypatch.setattr(
        _approval_wait._ctx, "_fire_approval_hook", lambda *_args, **_kwargs: None
    )
    with _approval._lock:
        _approval._gateway_queues.pop(SESSION, None)
    yield
    with _approval._lock:
        _approval._gateway_queues.pop(SESSION, None)


@pytest.mark.parametrize(
    "text_mode",
    [
        "send_result",
        "send_exception",
        "send_timeout_exception",
        "missing_future",
        "schedule_timeout",
    ],
)
def test_undeliverable_text_prompt_fails_closed_with_correlated_logs(
    text_mode, clean_approval_state, caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    adapter = _Adapter(text_mode)
    runner = _runner(adapter, text_mode)
    notified = {}

    def notify(data):
        notified.update(data)
        runner._approval_notify_sync(data)

    def waiting_after_failed_notify(*_args, **_kwargs):
        pytest.fail(
            "the approval waiter must not wait after a definitive delivery failure"
        )

    monkeypatch.setattr(_approval_wait, "_poll_event", waiting_after_failed_notify)
    decision = _approval_wait._await_gateway_decision(SESSION, notify, APPROVAL)

    assert decision == {"resolved": False, "choice": None, "notify_failed": True}
    assert notified["request_id"]
    assert len(adapter.text_sends) == (
        0 if text_mode in {"missing_future", "schedule_timeout"} else 1
    )
    assert SESSION not in _approval._gateway_queues

    correlated = [
        record
        for record in caplog.records
        if getattr(record, "approval_request_id", None) == notified["request_id"]
    ]
    assert any(
        record.delivery_lane == "text" and record.delivery_outcome == "failed"
        for record in correlated
    )
    assert any(
        record.delivery_lane == "callback"
        and record.delivery_outcome == "notify_failed"
        for record in correlated
    )
    assert "rm -rf /private/sensitive" not in "\n".join(
        record.getMessage() for record in correlated
    )


def test_successful_text_delivery_logs_sent_with_the_same_request_id(
    clean_approval_state, caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    adapter = _Adapter("success")
    runner = _runner(adapter, "success")
    notified = {}

    def notify(data):
        notified.update(data)
        runner._approval_notify_sync(data)

    monkeypatch.setattr(
        _approval_wait, "_poll_event", lambda *_args, **_kwargs: "timeout"
    )
    decision = _approval_wait._await_gateway_decision(SESSION, notify, APPROVAL)

    assert decision == {"resolved": False, "choice": None, "reason": None}
    assert adapter.text_sends
    assert any(
        getattr(record, "approval_request_id", None) == notified["request_id"]
        and getattr(record, "delivery_lane", None) == "text"
        and getattr(record, "delivery_outcome", None) == "sent"
        for record in caplog.records
    )
