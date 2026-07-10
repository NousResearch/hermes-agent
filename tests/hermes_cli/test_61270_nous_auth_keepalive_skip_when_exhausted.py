"""
Regression test for issue #61270 - Gateway startup blocks on Nous Portal
OAuth exhaustion with no user-visible error.

The fix: start_nous_auth_keepalive() now checks the rate-limit state
before starting the keepalive thread. If Portal is currently rate-limited,
the keepalive is skipped, a clear warning is logged, and the user is told
to run `hermes auth reset nous && hermes auth add nous` to recover.

Tests:
  1. test_start_skips_when_portal_exhausted - failing-first
  2. test_start_proceeds_when_portal_available
  3. test_start_proceeds_when_remaining_is_zero
  4. test_start_proceeds_when_check_fails
"""

import threading
from unittest import mock


def test_start_skips_when_portal_exhausted():
    from hermes_cli import nous_auth_keepalive
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()

    with mock.patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=120.0), \
         mock.patch("agent.nous_rate_guard.format_remaining", return_value="2m"), \
         mock.patch.object(nous_auth_keepalive, "logger") as mock_logger:
        result = nous_auth_keepalive.start_nous_auth_keepalive()

    assert result is None, (
        f"start_nous_auth_keepalive should return None when rate-limited, got {result!r}"
    )
    assert nous_auth_keepalive._keepalive_thread is None
    mock_logger.warning.assert_called()
    warning_msg = mock_logger.warning.call_args[0][0]
    assert "rate-limit" in warning_msg or "rate limit" in warning_msg
    assert "hermes auth reset nous" in warning_msg


def test_start_proceeds_when_portal_available():
    from hermes_cli import nous_auth_keepalive
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()

    with mock.patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=None):
        result = nous_auth_keepalive.start_nous_auth_keepalive()

    assert result is not None
    assert isinstance(result, threading.Thread)
    nous_auth_keepalive._keepalive_stop.set()
    result.join(timeout=2.0)
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()


def test_start_proceeds_when_remaining_is_zero():
    from hermes_cli import nous_auth_keepalive
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()

    with mock.patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=0):
        result = nous_auth_keepalive.start_nous_auth_keepalive()

    assert result is not None
    assert isinstance(result, threading.Thread)
    nous_auth_keepalive._keepalive_stop.set()
    result.join(timeout=2.0)
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()


def test_start_proceeds_when_check_fails():
    from hermes_cli import nous_auth_keepalive
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()

    def _raise(*_):
        raise RuntimeError("state file not found")
    with mock.patch("agent.nous_rate_guard.nous_rate_limit_remaining", side_effect=_raise):
        result = nous_auth_keepalive.start_nous_auth_keepalive()

    assert result is not None
    assert isinstance(result, threading.Thread)
    nous_auth_keepalive._keepalive_stop.set()
    result.join(timeout=2.0)
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()
