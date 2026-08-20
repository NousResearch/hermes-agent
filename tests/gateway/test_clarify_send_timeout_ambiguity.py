"""Clarify prompt-send TIMEOUT must not tear down the registration.

Sibling of test_approval_send_timeout_ambiguity.py, same boundary rule, same
live physics: send_clarify's scheduling future can hit its 15s deadline while
the clarify card HAS already posted (late connector ack). The old caller
treated any exception — including the timeout — as a definitive failure and
ran clear_session(), so the user answered a rendered card whose registration
was already gone.

Contract under test: TimeoutError is AMBIGUOUS (possibly delivered) — the
registration must stay armed (clear_session NOT called) and the caller must
proceed to the bounded wait (disposition None). A definitive error
(SendResult success=False, non-timeout exception, or no future) keeps
today's teardown + sentinel behavior.
"""

import concurrent.futures
from unittest.mock import MagicMock

from gateway.run import _clarify_send_disposition

SENTINEL = "[clarify prompt could not be delivered]"


class _Result:
    def __init__(self, success, error=None):
        self.success = success
        self.error = error


def test_timeout_keeps_registration_armed_and_proceeds_to_wait():
    fut = MagicMock()
    fut.result.side_effect = concurrent.futures.TimeoutError()
    clarify_mod = MagicMock()
    disposition = _clarify_send_disposition(
        fut, session_key="sk", clarify_mod=clarify_mod
    )
    assert disposition is None, (
        "a send timeout aborted the clarify wait — this is the "
        "cleared-session-under-a-rendered-card bug (card posted, ack late); "
        "ambiguous must fall through to wait_for_response"
    )
    clarify_mod.clear_session.assert_not_called()


def test_successful_send_proceeds_to_wait():
    fut = MagicMock()
    fut.result.return_value = _Result(True)
    clarify_mod = MagicMock()
    assert (
        _clarify_send_disposition(fut, session_key="sk", clarify_mod=clarify_mod)
        is None
    )
    clarify_mod.clear_session.assert_not_called()


def test_definitive_error_result_tears_down_and_aborts():
    fut = MagicMock()
    fut.result.return_value = _Result(False, "relay prompt op unavailable")
    clarify_mod = MagicMock()
    assert (
        _clarify_send_disposition(fut, session_key="sk", clarify_mod=clarify_mod)
        == SENTINEL
    )
    clarify_mod.clear_session.assert_called_once_with("sk")


def test_non_timeout_exception_tears_down_and_aborts():
    fut = MagicMock()
    fut.result.side_effect = RuntimeError("loop unavailable")
    clarify_mod = MagicMock()
    assert (
        _clarify_send_disposition(fut, session_key="sk", clarify_mod=clarify_mod)
        == SENTINEL
    )
    clarify_mod.clear_session.assert_called_once_with("sk")


def test_missing_future_tears_down_and_aborts():
    clarify_mod = MagicMock()
    assert (
        _clarify_send_disposition(None, session_key="sk", clarify_mod=clarify_mod)
        == SENTINEL
    )
    clarify_mod.clear_session.assert_called_once_with("sk")
