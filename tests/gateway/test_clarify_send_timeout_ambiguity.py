"""Clarify prompt-send TIMEOUT surfaces delivery-uncertainty (#112684).

Sibling of test_approval_send_timeout_ambiguity.py. send_clarify's scheduling
future can hit its 15s deadline while the clarify card HAS already posted (late
connector ack) — or while it was never delivered at all (the Telegram case in
#112684). The old caller fell through to the full bounded wait, which then
reported "[user did not respond within Nm]": a delivery failure misreported as
user inactivity.

Contract under test: TimeoutError is AMBIGUOUS (delivery unconfirmed) — the
caller returns a delivery-uncertainty sentinel immediately, never waits, and
retires the registration (a stale armed entry with no waiter would swallow the
user's next message via get_pending_for_session's oldest-first routing). A
definitive error (SendResult success=False, non-timeout exception, or no
future) keeps the teardown + "could not be delivered" sentinel behavior.
"""

import concurrent.futures
from unittest.mock import MagicMock

from gateway.run import _clarify_send_disposition, _clarify_send_then_wait

SENTINEL = "[clarify prompt could not be delivered]"
UNCERTAIN = "[clarify prompt delivery uncertain: send timed out before confirmation]"


class _Result:
    def __init__(self, success, error=None):
        self.success = success
        self.error = error


def test_timeout_surfaces_delivery_uncertainty_and_retires_registration():
    # #112684: a send timeout must not fall through to the bounded wait — the card
    # may never have been delivered, and waiting out the full timeout misreports
    # that as user inactivity.
    fut = MagicMock()
    fut.result.side_effect = concurrent.futures.TimeoutError()
    clarify_mod = MagicMock()
    disposition = _clarify_send_disposition(
        fut, session_key="sk", clarify_mod=clarify_mod
    )
    assert disposition == UNCERTAIN, (
        "a send timeout fell through to the bounded wait (disposition None) — "
        "a possibly-undelivered prompt must surface delivery-uncertainty"
    )
    clarify_mod.clear_session.assert_called_once_with("sk")


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


def test_ambiguous_send_returns_uncertainty_without_waiting():
    """The full caller contract, not just the classifier: on a send timeout the
    flow must return the delivery-uncertainty sentinel immediately — never the
    bounded wait, never "[user did not respond]"."""
    fut = MagicMock()
    fut.result.side_effect = concurrent.futures.TimeoutError()
    clarify_mod = MagicMock()
    clarify_mod.get_clarify_timeout.return_value = 600
    clarify_mod.wait_for_response.return_value = None

    out = _clarify_send_then_wait(
        fut, clarify_id="cid123", session_key="sk", clarify_mod=clarify_mod
    )

    assert out == (UNCERTAIN, False)
    clarify_mod.wait_for_response.assert_not_called()
    clarify_mod.clear_session.assert_called_once_with("sk")


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
        == ("answer", True)
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
        == (SENTINEL, False)
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
        == ("[user did not respond within 10m]", False)
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
