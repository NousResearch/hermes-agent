"""Text-fallback approval sends honor the send-disposition contract.

The button path classifies via ``_approval_send_outcome`` (sent/ambiguous/
failed); the plain-text path discarded the SendResult, so definitive
failures left the agent blocked on a prompt the user never received, slow
acks were mislogged as failures, and an unschedulable send was silent.
"""

import concurrent.futures
import logging
from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner


def _ctx(adapter):
    return SimpleNamespace(
        _status_adapter=adapter,
        _status_chat_id="chat-1",
        session_key="sess-1",
        _status_thread_metadata={},
    )


def _runner(adapter, fut):
    return SimpleNamespace(
        _ctx=_ctx(adapter),
        _schedule=lambda coro, msg: fut,
        _close_native_stream_boundary=lambda *a: None,
    )


def _adapter():
    # adapter.send() is evaluated eagerly as the _schedule argument, so the
    # stub must neither execute (sync raise) nor create a coroutine nobody
    # awaits (RuntimeWarning) — return an inert placeholder instead.
    def _send(chat_id, content, metadata=None):
        return object()

    return SimpleNamespace(
        pause_typing_for_chat=lambda chat_id: None,
        send=_send,
    )


def _notify(fut, caplog):
    TurnRunner._approval_notify_sync(
        _runner(_adapter(), fut),
        {"command": "rm -rf /tmp/x", "description": "dangerous command"},
    )
    return caplog.text


class _Future:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def result(self, timeout=None):
        if self._error is not None:
            raise self._error
        return self._result


def _ok():
    return SimpleNamespace(success=True, message_id="m1")


def _failed():
    return SimpleNamespace(success=False, error="send down")


class TestTextFallbackDisposition:
    def test_sent_is_quiet(self, caplog):
        with caplog.at_level(logging.WARNING):
            text = _notify(_Future(result=_ok()), caplog)
        assert "not delivered" not in text
        assert "possibly-delivered" not in text

    def test_failed_warns_not_delivered(self, caplog):
        with caplog.at_level(logging.WARNING):
            text = _notify(_Future(result=_failed()), caplog)
        assert "not delivered" in text

    def test_timeout_is_ambiguous_not_failure(self, caplog):
        with caplog.at_level(logging.WARNING):
            text = _notify(
                _Future(error=concurrent.futures.TimeoutError()), caplog)
        assert "possibly-delivered" in text
        assert "not delivered" not in text

    def test_unschedulable_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            text = _notify(None, caplog)
        assert "loop unavailable" in text or "not delivered" in text
