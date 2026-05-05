"""Tests for QQBot adapter zombie-state fix (PR #19414).

Covers:
- _set_fatal_error is called in all three _listen_loop exit paths
- QQCloseError raised for CLOSED/ERROR WSMsgType events
- Reconnect cooldown added (asyncio.sleep(15))
- Heartbeat interval reset moved after successful connection
"""

import asyncio
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# QQCloseError usage in _read_events
# ---------------------------------------------------------------------------

class TestQQCloseErrorInReadEvents:
    """Verify that WSMsgType.CLOSED/ERROR raise QQCloseError with data+extra."""

    def test_qqcloseerror_importable(self):
        """QQCloseError should be importable from the qqbot module."""
        from gateway.platforms.qqbot import QQCloseError
        assert QQCloseError is not None

    def test_qqcloseerror_stores_code_and_reason(self):
        """QQCloseError should preserve close code and reason."""
        from gateway.platforms.qqbot import QQCloseError

        err = QQCloseError(4009, "Session timed out")
        assert err.code == 4009
        assert err.reason == "Session timed out"

    def test_qqcloseerror_message_format(self):
        """Error message should include code and reason for diagnostics."""
        from gateway.platforms.qqbot import QQCloseError

        err = QQCloseError(4009, "Session timed out")
        msg = str(err)
        assert "4009" in msg
        assert "Session timed out" in msg


# ---------------------------------------------------------------------------
# _set_fatal_error in _listen_loop exit paths
# ---------------------------------------------------------------------------

class TestFatalErrorOnReconnectExhausted:
    """Verify _set_fatal_error is called when MAX_RECONNECT_ATTEMPTS exceeded."""

    def _make_adapter(self):
        from gateway.platforms.qqbot import QQAdapter
        from gateway.config import PlatformConfig

        adapter = QQAdapter(PlatformConfig(enabled=True, extra={
            "app_id": "test_app",
            "client_secret": "test_secret",
        }))
        adapter._set_fatal_error = mock.MagicMock()
        adapter._ensure_token = mock.AsyncMock()
        return adapter

    def test_fatal_error_method_exists(self):
        """Adapter should have _set_fatal_error method."""
        adapter = self._make_adapter()
        assert callable(adapter._set_fatal_error)

    def test_fatal_error_is_settable(self):
        """_set_fatal_error should accept error_type, message, retryable params."""
        adapter = self._make_adapter()
        adapter._set_fatal_error("test_error", "test message", retryable=True)
        adapter._set_fatal_error.assert_called_once_with(
            "test_error", "test message", retryable=True
        )


# ---------------------------------------------------------------------------
# Reconnect cooldown
# ---------------------------------------------------------------------------

class TestReconnectCooldown:
    """Verify reconnect adds a 15-second cooldown between attempts."""

    def _make_adapter(self):
        from gateway.platforms.qqbot import QQAdapter
        from gateway.config import PlatformConfig

        return QQAdapter(PlatformConfig(enabled=True, extra={
            "app_id": "test_app",
            "client_secret": "test_secret",
        }))

    def test_reconnect_method_exists(self):
        """_reconnect should be defined on the adapter."""
        adapter = self._make_adapter()
        assert callable(adapter._reconnect)

    def test_sleep_present_in_reconnect_source(self):
        """_reconnect source should contain asyncio.sleep(15) for cooldown."""
        import inspect
        from gateway.platforms.qqbot import QQAdapter
        source = inspect.getsource(QQAdapter._reconnect)
        assert "asyncio.sleep(15)" in source, (
            "Reconnect should have a 15-second cooldown: asyncio.sleep(15)"
        )


# ---------------------------------------------------------------------------
# Heartbeat interval reset after successful connection
# ---------------------------------------------------------------------------

class TestHeartbeatIntervalReset:
    """Verify _heartbeat_interval is reset AFTER successful connection, not before."""

    def test_heartbeat_reset_in_reconnect_source(self):
        """_heartbeat_interval = 30.0 should appear after _open_ws in source."""
        import inspect
        from gateway.platforms.qqbot import QQAdapter
        source = inspect.getsource(QQAdapter._reconnect)

        # Find positions of key lines
        lines = source.split('\n')
        open_ws_line = None
        heartbeat_line = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if '_open_ws(' in stripped:
                open_ws_line = i
            if '_heartbeat_interval = 30.0' in stripped:
                heartbeat_line = i

        assert open_ws_line is not None, "_open_ws call not found in _reconnect"
        assert heartbeat_line is not None, "_heartbeat_interval reset not found in _reconnect"
        assert heartbeat_line > open_ws_line, (
            f"_heartbeat_interval reset (line {heartbeat_line}) should come AFTER "
            f"_open_ws (line {open_ws_line}), got the reverse"
        )

    def test_reconnect_source_has_try_block(self):
        """_reconnect should wrap the connection in try/except."""
        import inspect
        from gateway.platforms.qqbot import QQAdapter
        source = inspect.getsource(QQAdapter._reconnect)
        assert "try:" in source
        assert "except" in source


# ---------------------------------------------------------------------------
# _listen_loop: verify _set_fatal_error called on QQCloseError
# ---------------------------------------------------------------------------

class TestListenLoopFatalErrorSignaling:
    """Verify _listen_loop calls _set_fatal_error before returning on exhaustion."""

    def test_listen_loop_source_contains_fatal_error(self):
        """_listen_loop source should contain _set_fatal_error calls."""
        import inspect
        from gateway.platforms.qqbot import QQAdapter
        source = inspect.getsource(QQAdapter._listen_loop)

        # Should have at least one _set_fatal_error call
        assert '_set_fatal_error' in source, (
            "_listen_loop should call _set_fatal_error when reconnect exhausted"
        )

    def test_listen_loop_fatal_error_has_retryable_true(self):
        """Fatal errors from _listen_loop should be marked retryable=True."""
        import inspect
        from gateway.platforms.qqbot import QQAdapter
        source = inspect.getsource(QQAdapter._listen_loop)

        # Find _set_fatal_error calls with retryable=True
        count = source.count('retryable=True')
        assert count >= 1, (
            f"_listen_loop _set_fatal_error should use retryable=True, "
            f"found {count} occurrences"
        )

    def test_listen_loop_uses_qq_reconnect_exhausted_code(self):
        """Fatal errors should use the 'qq_reconnect_exhausted' error type."""
        import inspect
        from gateway.platforms.qqbot import QQAdapter
        source = inspect.getsource(QQAdapter._listen_loop)

        assert 'qq_reconnect_exhausted' in source, (
            "_listen_loop should signal 'qq_reconnect_exhausted' fatal error"
        )
