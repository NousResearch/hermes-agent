"""Approval prompt-send TIMEOUT must not trigger the re-ask/fallback lane.

Observed in live relay testing: `send_exec_approval`'s scheduling future can
hit its 15s `.result(timeout=...)` while the card HAS already posted to the
platform — the connector's ack simply arrives after the deadline (slow
platform API call, transient backpressure, event-loop stall). run.py treated
the timeout like a definitive send failure and ran the text fallback, so the
user saw the same approval multiple times; tapping an older card resolved a
prompt whose turn had already moved on ("/approve: nothing pending").

Contract under test (boundary rule — every prompt caller crossing the
send-timeout boundary): concurrent.futures.TimeoutError from the approval
send is AMBIGUOUS (possibly delivered). The gateway must NOT fall back /
re-send; the prompt registration stays live so the user's tap on the
(probably rendered) card still resolves. A definitive error (SendResult
success=False, or a non-timeout exception) keeps today's fallback.
"""

import concurrent.futures
from unittest.mock import MagicMock

import pytest

from gateway.run import _approval_send_outcome


class _Result:
    def __init__(self, success, error=None):
        self.success = success
        self.error = error


def test_timeout_is_ambiguous_not_failure():
    fut = MagicMock()
    fut.result.side_effect = concurrent.futures.TimeoutError()
    outcome = _approval_send_outcome(fut, timeout=0.01)
    assert outcome == "ambiguous", (
        "a send timeout re-ran the fallback — this is the duplicate-approval "
        "re-pop (card posted, ack late); ambiguous must suppress the re-ask"
    )


def test_success_is_sent():
    fut = MagicMock()
    fut.result.return_value = _Result(True)
    assert _approval_send_outcome(fut, timeout=1) == "sent"


def test_definitive_error_result_is_failed():
    fut = MagicMock()
    fut.result.return_value = _Result(False, "relay prompt op unavailable")
    assert _approval_send_outcome(fut, timeout=1) == "failed"


def test_non_timeout_exception_is_failed():
    fut = MagicMock()
    fut.result.side_effect = RuntimeError("loop unavailable")
    assert _approval_send_outcome(fut, timeout=1) == "failed"


def test_missing_future_is_failed():
    assert _approval_send_outcome(None, timeout=1) == "failed"


def test_undeliverable_only_when_every_channel_definitively_missed():
    from gateway.run import _approval_prompt_undeliverable

    assert _approval_prompt_undeliverable("sent", None) is False
    assert _approval_prompt_undeliverable("ambiguous", None) is False
    assert _approval_prompt_undeliverable("failed", "sent") is False
    assert _approval_prompt_undeliverable("failed", "ambiguous") is False
    assert _approval_prompt_undeliverable("failed", "failed") is True
    assert _approval_prompt_undeliverable(None, "failed") is True
    assert _approval_prompt_undeliverable(None, None) is True
