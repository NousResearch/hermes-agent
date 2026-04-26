"""Unit tests for hermes_cli.pty_bridge — PTY spawning + byte forwarding.

These tests drive the bridge with minimal POSIX processes (echo, env, sleep,
printf) to verify it behaves like a PTY you can read/write/resize/close.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

pytest.importorskip("ptyprocess", reason="ptyprocess not installed")

from hermes_cli.pty_bridge import PtyBridge, PtyUnavailableError


skip_on_windows = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="PTY bridge is POSIX-only"
)


def _read_until(bridge: PtyBridge, needle: bytes, timeout: float = 5.0) -> bytes:
    """Accumulate PTY output until we see `needle` or time out."""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        chunk = bridge.read(timeout=0.2)
        if chunk is None:
            break
        buf.extend(chunk)
        if needle in buf:
            return bytes(buf)
    return bytes(buf)


@skip_on_windows
class TestPtyBridgeSpawn:
    def test_is_available_on_posix(self):
        assert PtyBridge.is_available() is True

    def test_spawn_returns_bridge_with_pid(self):
        bridge = PtyBridge.spawn(["true"])
        try:
            assert bridge.pid > 0
        finally:
            bridge.close()

    def test_spawn_raises_on_missing_argv0(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            PtyBridge.spawn([str(tmp_path / "definitely-not-a-real-binary")])


@skip_on_windows
class TestPtyBridgeIO:
    def test_reads_child_stdout(self):
        bridge = PtyBridge.spawn(["/bin/sh", "-c", "printf hermes-ok"])
        try:
            output = _read_until(bridge, b"hermes-ok")
            assert b"hermes-ok" in output
        finally:
            bridge.close()

    def test_write_sends_to_child_stdin(self):
        # `cat` with no args echoes stdin back to stdout.  We write a line,
        # read it back, then signal EOF to let cat exit cleanly.
        bridge = PtyBridge.spawn(["/bin/cat"])
        try:
            bridge.write(b"hello-pty\n")
            output = _read_until(bridge, b"hello-pty")
            assert b"hello-pty" in output
        finally:
            bridge.close()

    def test_read_returns_none_after_child_exits(self):
        bridge = PtyBridge.spawn(["/bin/sh", "-c", "printf done"])
        try:
            _read_until(bridge, b"done")
            # Give the child a beat to exit cleanly, then drain until EOF.
            deadline = time.monotonic() + 3.0
            while bridge.is_alive() and time.monotonic() < deadline:
                bridge.read(timeout=0.1)
            # Next reads after exit should return None (EOF), not raise.
            got_none = False
            for _ in range(10):
                if bridge.read(timeout=0.1) is None:
                    got_none = True
                    break
            assert got_none, "PtyBridge.read did not return None after child EOF"
        finally:
            bridge.close()


@skip_on_windows
class TestPtyBridgeResize:
    def test_resize_updates_child_winsize(self):
        # tput reads COLUMNS/LINES from the TTY ioctl (TIOCGWINSZ).
        # Spawn a shell, resize, then ask tput for the dimensions.
        bridge = PtyBridge.spawn(
            ["/bin/sh", "-c", "sleep 0.1; tput cols; tput lines"],
            cols=80,
            rows=24,
        )
        try:
            bridge.resize(cols=123, rows=45)
            output = _read_until(bridge, b"45", timeout=5.0)
            # tput prints just the numbers, one per line
            assert b"123" in output
            assert b"45" in output
        finally:
            bridge.close()


@skip_on_windows
class TestPtyBridgeClose:
    def test_close_is_idempotent(self):
        bridge = PtyBridge.spawn(["/bin/sh", "-c", "sleep 30"])
        bridge.close()
        bridge.close()  # must not raise
        assert not bridge.is_alive()

    def test_close_terminates_long_running_child(self):
        bridge = PtyBridge.spawn(["/bin/sh", "-c", "sleep 30"])
        pid = bridge.pid
        bridge.close()
        # Give the kernel a moment to reap
        deadline = time.monotonic() + 3.0
        reaped = False
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                reaped = True
                break
        assert reaped, f"pid {pid} still running after close()"


@skip_on_windows
class TestPtyBridgeEnv:
    def test_cwd_is_respected(self, tmp_path):
        bridge = PtyBridge.spawn(
            ["/bin/sh", "-c", "pwd"],
            cwd=str(tmp_path),
        )
        try:
            output = _read_until(bridge, str(tmp_path).encode())
            assert str(tmp_path).encode() in output
        finally:
            bridge.close()

    def test_env_is_forwarded(self):
        bridge = PtyBridge.spawn(
            ["/bin/sh", "-c", "printf %s \"$HERMES_PTY_TEST\""],
            env={**os.environ, "HERMES_PTY_TEST": "pty-env-works"},
        )
        try:
            output = _read_until(bridge, b"pty-env-works")
            assert b"pty-env-works" in output
        finally:
            bridge.close()


class TestPtyBridgeUnavailable:
    """Platform fallback semantics — PtyUnavailableError is importable and
    carries a user-readable message."""

    def test_error_carries_user_message(self):
        err = PtyUnavailableError("platform not supported")
        assert "platform" in str(err)
