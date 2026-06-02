"""Unit tests for hermes_cli.pty_bridge.

These tests drive the bridge with minimal processes to verify it behaves like
a PTY you can read/write/resize/close. POSIX uses ptyprocess; native Windows
uses pywinpty/ConPTY.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import time

import pytest

from hermes_cli.pty_bridge import PtyBridge, PtyUnavailableError


_POSIX_PTY_AVAILABLE = importlib.util.find_spec("ptyprocess") is not None
_WINDOWS_PTY_AVAILABLE = importlib.util.find_spec("winpty") is not None

skip_posix_unavailable = pytest.mark.skipif(
    sys.platform.startswith("win") or not _POSIX_PTY_AVAILABLE,
    reason="POSIX PTY bridge requires ptyprocess",
)

skip_windows_unavailable = pytest.mark.skipif(
    not sys.platform.startswith("win") or not _WINDOWS_PTY_AVAILABLE,
    reason="Windows PTY bridge requires pywinpty/ConPTY",
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


@skip_posix_unavailable
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


@skip_posix_unavailable
class TestPtyBridgeIO:
    def test_reads_child_stdout(self):
        bridge = PtyBridge.spawn(["/bin/sh", "-c", "printf hermes-ok"])
        try:
            output = _read_until(bridge, b"hermes-ok")
            assert b"hermes-ok" in output
        finally:
            bridge.close()

    def test_write_sends_to_child_stdin(self):
        # `cat` with no args echoes stdin back to stdout. We write a line,
        # read it back, then close the PTY to let cat exit cleanly.
        bridge = PtyBridge.spawn([shutil.which("cat") or "cat"])
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
            deadline = time.monotonic() + 3.0
            while bridge.is_alive() and time.monotonic() < deadline:
                bridge.read(timeout=0.1)
            got_none = False
            for _ in range(10):
                if bridge.read(timeout=0.1) is None:
                    got_none = True
                    break
            assert got_none, "PtyBridge.read did not return None after child EOF"
        finally:
            bridge.close()


@skip_posix_unavailable
class TestPtyBridgeResize:
    def test_resize_updates_child_winsize(self):
        winsize_script = (
            "import fcntl, struct, termios, time; "
            "time.sleep(0.1); "
            "rows, cols, *_ = struct.unpack('HHHH', "
            "fcntl.ioctl(0, termios.TIOCGWINSZ, b'\\0' * 8)); "
            "print(cols); print(rows)"
        )
        bridge = PtyBridge.spawn(
            [sys.executable, "-c", winsize_script],
            cols=80,
            rows=24,
        )
        try:
            bridge.resize(cols=123, rows=45)
            output = _read_until(bridge, b"45", timeout=5.0)
            assert b"123" in output
            assert b"45" in output
        finally:
            bridge.close()


@skip_posix_unavailable
class TestPtyBridgeClose:
    def test_close_is_idempotent(self):
        bridge = PtyBridge.spawn(["/bin/sh", "-c", "sleep 30"])
        bridge.close()
        bridge.close()
        assert not bridge.is_alive()

    def test_close_terminates_long_running_child(self):
        bridge = PtyBridge.spawn(["/bin/sh", "-c", "sleep 30"])
        pid = bridge.pid
        bridge.close()
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


@skip_posix_unavailable
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


@skip_windows_unavailable
class TestWindowsPtyBridge:
    def test_is_available_on_windows(self):
        assert PtyBridge.is_available() is True

    def test_reads_child_stdout(self):
        bridge = PtyBridge.spawn(["cmd.exe", "/c", "echo hermes-winpty-ok"])
        try:
            output = _read_until(bridge, b"hermes-winpty-ok")
            assert b"hermes-winpty-ok" in output
        finally:
            bridge.close()

    def test_write_sends_to_child_stdin(self):
        bridge = PtyBridge.spawn(["cmd.exe"])
        try:
            _read_until(bridge, b">")
            bridge.write(b"echo hello-winpty\r")
            output = _read_until(bridge, b"hello-winpty")
            assert b"hello-winpty" in output
        finally:
            bridge.write(b"exit\r")
            bridge.close()

    def test_read_returns_none_after_child_exits(self):
        bridge = PtyBridge.spawn(["cmd.exe", "/c", "echo done-winpty"])
        try:
            _read_until(bridge, b"done-winpty")
            deadline = time.monotonic() + 3.0
            while bridge.is_alive() and time.monotonic() < deadline:
                bridge.read(timeout=0.1)
            got_none = False
            for _ in range(10):
                if bridge.read(timeout=0.1) is None:
                    got_none = True
                    break
            assert got_none, "PtyBridge.read did not return None after child EOF"
        finally:
            bridge.close()

    def test_resize_does_not_raise(self):
        bridge = PtyBridge.spawn(["cmd.exe"])
        try:
            bridge.resize(cols=99, rows=41)
        finally:
            bridge.write(b"exit\r")
            bridge.close()

    def test_cwd_is_respected(self, tmp_path):
        bridge = PtyBridge.spawn(["cmd.exe", "/c", "cd"], cwd=str(tmp_path))
        try:
            output = _read_until(bridge, str(tmp_path).encode())
            assert str(tmp_path).encode() in output
        finally:
            bridge.close()

    def test_env_is_forwarded(self):
        bridge = PtyBridge.spawn(
            ["cmd.exe", "/c", "echo %HERMES_PTY_TEST%"],
            env={**os.environ, "HERMES_PTY_TEST": "winpty-env-works"},
        )
        try:
            output = _read_until(bridge, b"winpty-env-works")
            assert b"winpty-env-works" in output
        finally:
            bridge.close()

    def test_close_is_idempotent(self):
        bridge = PtyBridge.spawn(["cmd.exe"])
        bridge.close()
        bridge.close()
        assert not bridge.is_alive()


class TestPtyBridgeUnavailable:
    """PtyUnavailableError is importable and carries a user-readable message."""

    def test_error_carries_user_message(self):
        err = PtyUnavailableError("platform not supported")
        assert "platform" in str(err)
