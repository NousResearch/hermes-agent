"""
Regression test for issue #61270 - Gateway startup blocks on Nous Portal
OAuth exhaustion with no user-visible error.

The reporter's symptom: when Nous Portal auth is rate-limited, starting
the gateway appears to hang. The user sees "gateway keeps failing to
connect" with no clear indication that the root cause is a rate-limited
OAuth credential.

The fix: start_nous_auth_keepalive() now checks the rate-limit state
before starting the keepalive thread. If Portal is currently rate-limited,
the keepalive is skipped, a clear warning is logged, and the user is told
to run `hermes auth reset nous && hermes auth add nous` to recover.

This test asserts:
  1. When Portal is rate-limited (remaining > 0), start_nous_auth_keepalive
     returns None (no thread started).
  2. When Portal is NOT rate-limited (remaining is None or 0), the
     keepalive thread starts as before.
  3. A clear warning is logged with the recovery instructions.
"""

import logging
import threading
from unittest.mock import MagicMock, patch


def test_start_skips_when_portal_exhausted(monkeypatch):
    """When nous_rate_limit_remaining() > 0, the keepalive is not started."""
    from hermes_cli import nous_auth_keepalive

    # Reset module-level state for the test
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()

    # Simulate Portal rate-limited
    with patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=120.0), \
         patch("agent.nous_rate_guard.format_remaining", return_value="2m"):
        with patch.object(nous_auth_keepalive, "logger") as mock_logger:
            result = nous_auth_keepalive.start_nous_auth_keepalive()

    assert result is None, (
        f"start_nous_auth_keepalive should return None when rate-limited, "
        f"got {result!r}"
    )

    # The keepalive thread was NOT started
    assert nous_auth_keepalive._keepalive_thread is None

    # A warning was logged with the recovery hint
    mock_logger.warning.assert_called()
    warning_msg = mock_logger.warning.call_args[0][0]
    assert "rate-limited" in warning_msg or "rate limit" in warning_msg
    assert "hermes auth reset nous" in warning_msg, (
        f"warning should include the recovery command, got: {warning_msg!r}"
    )


def test_start_proceeds_when_portal_available(monkeypatch):
    """When nous_rate_limit_remaining() is None, the keepalive thread starts."""
    from hermes_cli import nous_auth_keepalive

    # Reset module-level state
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()

    with patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=None):
        result = nous_auth_keepalive.start_nous_auth_keepalive()

    # The keepalive thread WAS started
    assert result is not None, (
        f"start_nous_auth_keepalive should return a Thread when Portal "
        f"is not rate-limited, got {result!r}"
    )
    assert isinstance(result, threading.Thread)
    assert result.is_alive() or result.daemon, "thread should be a daemon, started"

    # Clean up: stop the keepalive
    nous_auth_keepalive._keepalive_stop.set()
    result.join(timeout=2.0)
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()


def test_start_proceeds_when_remaining_is_zero(monkeypatch):
    """When nous_rate_limit_remaining() returns 0 (window just cleared), proceed."""
    from hermes_cli import nous_auth_keepalive

    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()

    with patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=0):
        result = nous_auth_keepalive.start_nous_auth_keepalive()

    # remaining=0 means "not currently rate-limited" (or window just cleared);
    # the keepalive should proceed.
    assert result is not None
    assert isinstance(result, threading.Thread)

    nous_auth_keepalive._keepalive_stop.set()
    result.join(timeout=2.0)
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()


def test_start_proceeds_when_check_fails(monkeypatch):
    """If nous_rate_limit_remaining() raises, fall through to existing behavior."""
    from hermes_cli import nous_auth_keepalive

    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()

    def _raise(_unused=None):
        raise RuntimeError("rate-limit state file not found")

    # Patch at the import site of start_nous_auth_keepalive
    with patch("agent.nous_rate_guard.nous_rate_limit_remaining", side_effect=_raise):
        result = nous_auth_keepalive.start_nous_auth_keepalive()

    # The keepalive should still start - we don't want to break the
    # gateway just because the rate-limit check is unavailable.
    assert result is not None
    assert isinstance(result, threading.Thread)

    nous_auth_keepalive._keepalive_stop.set()
    result.join(timeout=2.0)
    nous_auth_keepalive._keepalive_thread = None
    nous_auth_keepalive._keepalive_stop.clear()
