"""A local listen-port conflict must retry fast, then fall back to the backoff ladder.

Regression: on gateway restart the incoming instance starts binding while the predecessor is
still draining, so platforms that bind their own port (a2a, wecom_callback) fail with a
conflict that clears itself within seconds. With only the 30/60/120s ladder every restart
waited out a whole backoff period after the port was already free (~50s of dead air).
"""
from gateway.run_adapters import (
    _RECONNECT_BIND_CONFLICT_FAST_ATTEMPTS,
    _RECONNECT_BIND_CONFLICT_RETRY_SECS,
    is_local_bind_conflict,
)


def test_bind_failed_code_is_a_local_conflict():
    assert is_local_bind_conflict("bind_failed", "") is True


def test_code_match_ignores_case_and_whitespace():
    assert is_local_bind_conflict("  BIND_FAILED  ", "boom") is True


def test_already_in_use_message_is_a_local_conflict():
    assert is_local_bind_conflict(None, "[Errno 48] Address already in use") is True


def test_unrelated_failures_are_not_local_conflicts():
    assert is_local_bind_conflict("auth_failed", "invalid token") is False
    assert is_local_bind_conflict(None, "connection reset by peer") is False


def test_missing_inputs_are_not_local_conflicts():
    assert is_local_bind_conflict(None, None) is False


def test_fast_window_covers_the_drain_but_still_escalates():
    """The window must cover the 50s drain ceiling; a longer one would spin on a real conflict."""
    window = _RECONNECT_BIND_CONFLICT_FAST_ATTEMPTS * _RECONNECT_BIND_CONFLICT_RETRY_SECS
    assert 50 <= window <= 120
