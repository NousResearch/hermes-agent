"""Tests for PtyBridge.close() zombie reaping and async bridge.close() in pty_ws.

Bug: Dashboard wedges (SIGKILL required) when hosted as a persistent service.
- Bug 1: bridge.close() blocks the ASGI event loop (up to 1.5s synchronous sleep)
- Bug 2: zombie node processes never get reaped after PTY session ends
"""

from __future__ import annotations

import os
import signal
import time
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# PtyBridge.close() zombie reaping
# ---------------------------------------------------------------------------


class TestPtyBridgeZombieReaping:
    """close() must call os.waitpid() to reap the child process."""

    def test_close_calls_waitpid_after_signal_loop(self, tmp_path):
        """After the signal loop, os.waitpid(pid, WNOHANG) is called."""
        from hermes_cli.pty_bridge import PtyBridge

        # Create a real child process to act as a stand-in for the PTY child.
        # We can't easily spawn a real PTY in a unit test, so mock the
        # internals and verify waitpid is called.
        bridge = PtyBridge.__new__(PtyBridge)
        bridge._closed = False

        fake_proc = mock.MagicMock()
        fake_proc.pid = 99999  # non-existent PID
        fake_proc.isalive.return_value = False  # already dead
        bridge._proc = fake_proc

        with mock.patch("os.getpgid", return_value=99999), \
             mock.patch("os.waitpid") as mock_waitpid, \
             mock.patch("os.killpg"), \
             mock.patch.object(bridge._proc, "close"):
            bridge.close()

        mock_waitpid.assert_called_once_with(99999, os.WNOHANG)

    def test_close_handles_already_reaped_child(self):
        """ChildProcessError from waitpid (already reaped) must not propagate."""
        from hermes_cli.pty_bridge import PtyBridge

        bridge = PtyBridge.__new__(PtyBridge)
        bridge._closed = False

        fake_proc = mock.MagicMock()
        fake_proc.pid = 99999
        fake_proc.isalive.return_value = False
        bridge._proc = fake_proc

        with mock.patch("os.getpgid", return_value=99999), \
             mock.patch("os.waitpid", side_effect=ChildProcessError), \
             mock.patch("os.killpg"), \
             mock.patch.object(bridge._proc, "close"):
            # Must not raise
            bridge.close()

        assert bridge._closed is True

    def test_close_handles_waitpid_os_error(self):
        """OSError from waitpid (invalid pid) must not propagate."""
        from hermes_cli.pty_bridge import PtyBridge

        bridge = PtyBridge.__new__(PtyBridge)
        bridge._closed = False

        fake_proc = mock.MagicMock()
        fake_proc.pid = 99999
        fake_proc.isalive.return_value = False
        bridge._proc = fake_proc

        with mock.patch("os.getpgid", return_value=99999), \
             mock.patch("os.waitpid", side_effect=OSError), \
             mock.patch("os.killpg"), \
             mock.patch.object(bridge._proc, "close"):
            bridge.close()

        assert bridge._closed is True

    def test_close_skips_waitpid_when_no_pid(self):
        """If proc.pid is None/0, skip waitpid gracefully."""
        from hermes_cli.pty_bridge import PtyBridge

        bridge = PtyBridge.__new__(PtyBridge)
        bridge._closed = False

        fake_proc = mock.MagicMock()
        fake_proc.pid = None
        fake_proc.isalive.return_value = False
        bridge._proc = fake_proc

        with mock.patch("os.getpgid", side_effect=OSError), \
             mock.patch("os.waitpid") as mock_waitpid, \
             mock.patch.object(bridge._proc, "close"):
            bridge.close()

        mock_waitpid.assert_not_called()

    def test_close_is_idempotent(self):
        """Calling close() twice must not double-reap."""
        from hermes_cli.pty_bridge import PtyBridge

        bridge = PtyBridge.__new__(PtyBridge)
        bridge._closed = False

        fake_proc = mock.MagicMock()
        fake_proc.pid = 99999
        fake_proc.isalive.return_value = False
        bridge._proc = fake_proc

        with mock.patch("os.getpgid", return_value=99999), \
             mock.patch("os.waitpid") as mock_waitpid, \
             mock.patch("os.killpg"), \
             mock.patch.object(bridge._proc, "close"):
            bridge.close()
            bridge.close()  # second call

        # waitpid should only be called once (second close() is a no-op)
        mock_waitpid.assert_called_once()


# ---------------------------------------------------------------------------
# pty_ws handler: bridge.close() must be async
# ---------------------------------------------------------------------------


class TestPtyWsAsyncClose:
    """pty_ws must call bridge.close() via asyncio.to_thread, not synchronously."""

    def test_bridge_close_uses_to_thread_in_source(self):
        """Source-level check: bridge.close() must be wrapped in asyncio.to_thread."""
        import inspect
        from hermes_cli import web_server

        src = inspect.getsource(web_server)
        # The fix replaces `bridge.close()` with `await asyncio.to_thread(bridge.close)`
        assert "await asyncio.to_thread(bridge.close)" in src, (
            "bridge.close() must be called via asyncio.to_thread to avoid "
            "blocking the ASGI event loop"
        )

    def test_no_synchronous_bridge_close_in_finally(self):
        """Source-level guard: bare bridge.close() must not appear in pty_ws."""
        import inspect
        from hermes_cli import web_server

        src = inspect.getsource(web_server)
        # Find the pty_ws function
        pty_ws_start = src.find("async def pty_ws(")
        assert pty_ws_start >= 0, "pty_ws function not found"

        # Extract just the pty_ws function body (rough heuristic)
        pty_ws_body = src[pty_ws_start:]

        # There should be no bare `bridge.close()` — only the wrapped version
        lines = pty_ws_body.split("\n")
        for line in lines:
            stripped = line.strip()
            # Skip the async.to_thread version
            if "asyncio.to_thread" in stripped:
                continue
            # A bare bridge.close() in the finally block is the bug
            if stripped == "bridge.close()":
                pytest.fail(
                    "Found bare bridge.close() in pty_ws — must use "
                    "await asyncio.to_thread(bridge.close)"
                )
