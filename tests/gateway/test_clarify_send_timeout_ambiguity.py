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

from gateway.run import _clarify_send_disposition, _clarify_send_then_wait

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


# --- Caller-path contract: the disposition feeds the bounded wait ---------


def test_ambiguous_send_reaches_wait_for_response():
    """The full caller contract, not just the classifier: on a send timeout
    the flow must proceed to wait_for_response with the generated clarify_id
    and the configured timeout — the late reply to the (probably rendered)
    card resolves through that wait."""
    fut = MagicMock()
    fut.result.side_effect = concurrent.futures.TimeoutError()
    clarify_mod = MagicMock()
    clarify_mod.get_clarify_timeout.return_value = 600
    clarify_mod.wait_for_response.return_value = "user picked B"

    out = _clarify_send_then_wait(
        fut, clarify_id="cid123", session_key="sk", clarify_mod=clarify_mod
    )

    assert out == "user picked B"
    clarify_mod.clear_session.assert_not_called()
    clarify_mod.wait_for_response.assert_called_once_with("cid123", timeout=600.0)


def test_sent_reaches_wait_for_response():
    fut = MagicMock()
    fut.result.return_value = _Result(True)
    clarify_mod = MagicMock()
    clarify_mod.get_clarify_timeout.return_value = 600
    clarify_mod.wait_for_response.return_value = "answer"

    assert (
        _clarify_send_then_wait(
            fut, clarify_id="cid123", session_key="sk", clarify_mod=clarify_mod
        )
        == "answer"
    )
    clarify_mod.wait_for_response.assert_called_once_with("cid123", timeout=600.0)


def test_definitive_failure_never_waits():
    fut = MagicMock()
    fut.result.return_value = _Result(False, "relay prompt op unavailable")
    clarify_mod = MagicMock()

    assert (
        _clarify_send_then_wait(
            fut, clarify_id="cid123", session_key="sk", clarify_mod=clarify_mod
        )
        == SENTINEL
    )
    clarify_mod.wait_for_response.assert_not_called()
    clarify_mod.clear_session.assert_called_once_with("sk")


def test_no_response_returns_timeout_sentinel():
    fut = MagicMock()
    fut.result.return_value = _Result(True)
    clarify_mod = MagicMock()
    clarify_mod.get_clarify_timeout.return_value = 600
    clarify_mod.wait_for_response.return_value = None

    assert (
        _clarify_send_then_wait(
            fut, clarify_id="cid123", session_key="sk", clarify_mod=clarify_mod
        )
        == "[user did not respond within 10m]"
    )


def test_interrupted_wait_returns_interrupt_sentinel(monkeypatch):
    fut = MagicMock()
    fut.result.return_value = _Result(True)
    clarify_mod = MagicMock()
    clarify_mod.get_clarify_timeout.return_value = 600
    clarify_mod.wait_for_response.return_value = None
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: True)

    assert (
        _clarify_send_then_wait(
            fut, clarify_id="cid123", session_key="sk", clarify_mod=clarify_mod
        )
        == "[interrupted by user]"
    )


# --- Definitive failures keep their diagnostic detail in the log ----------


def test_failed_send_exception_detail_is_logged(caplog):
    fut = MagicMock()
    fut.result.side_effect = RuntimeError("loop unavailable")
    clarify_mod = MagicMock()
    with caplog.at_level("WARNING", logger="gateway.run"):
        _clarify_send_disposition(fut, session_key="sk", clarify_mod=clarify_mod)
    assert "loop unavailable" in caplog.text


def test_failed_send_result_error_detail_is_logged(caplog):
    fut = MagicMock()
    fut.result.return_value = _Result(False, "relay prompt op unavailable")
    clarify_mod = MagicMock()
    with caplog.at_level("WARNING", logger="gateway.run"):
        _clarify_send_disposition(fut, session_key="sk", clarify_mod=clarify_mod)
    assert "relay prompt op unavailable" in caplog.text
