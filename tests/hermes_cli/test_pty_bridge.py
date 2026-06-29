"""Unit tests for hermes_cli.pty_bridge — PTY spawning + byte forwarding.

These tests drive the bridge with minimal POSIX processes (echo, env, sleep,
printf) to verify it behaves like a PTY you can read/write/resize/close.
"""

from __future__ import annotations

import os
import shutil
import signal
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
        # Query the TTY ioctl directly instead of using tput, which requires
        # TERM and fails in GitHub Actions' non-interactive environment.
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
            # tput prints just the numbers, one per line
            assert b"123" in output
            assert b"45" in output
        finally:
            bridge.close()

    def test_resize_clamps_wsl_garbage_dimensions(self):
        # WSL2 reports columns=131072, rows=1 from a broken winsize probe.
        # 131072 > 65535 (unsigned short max) used to raise struct.error in
        # resize() — uncaught, since only OSError was handled — and broke the
        # dashboard /chat resize path (blank/disappearing text). The clamp
        # must coerce the width down to the sane max and never raise.
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
            # Must not raise struct.error.
            bridge.resize(cols=131072, rows=1)
            output = _read_until(bridge, b"\n", timeout=5.0)
            # Width clamped to the sane maximum (2000), height floored to 1.
            assert b"2000" in output
        finally:
            bridge.close()


@skip_on_windows
class TestClampDimension:
    def test_clamps_above_max(self):
        from hermes_cli.pty_bridge import _MAX_COLS, _MAX_ROWS, _clamp_dimension

        assert _clamp_dimension(131072, _MAX_COLS) == _MAX_COLS
        assert _clamp_dimension(131072, _MAX_ROWS) == _MAX_ROWS

    def test_floors_at_one(self):
        from hermes_cli.pty_bridge import _MAX_COLS, _clamp_dimension

        assert _clamp_dimension(0, _MAX_COLS) == 1
        assert _clamp_dimension(-5, _MAX_COLS) == 1

    def test_passes_through_sane_values(self):
        from hermes_cli.pty_bridge import _MAX_COLS, _clamp_dimension

        assert _clamp_dimension(80, _MAX_COLS) == 80
        assert _clamp_dimension(2000, _MAX_COLS) == 2000

    def test_non_numeric_falls_back_to_min(self):
        from hermes_cli.pty_bridge import _MAX_COLS, _clamp_dimension

        assert _clamp_dimension(None, _MAX_COLS) == 1  # type: ignore[arg-type]
        assert _clamp_dimension(float("nan"), _MAX_COLS) == 1  # type: ignore[arg-type]
        assert _clamp_dimension(float("inf"), _MAX_COLS) == 1  # type: ignore[arg-type]

    def test_clamped_values_pack_as_unsigned_short(self):
        # The whole point: clamped output must never raise struct.error.
        import struct as _struct

        from hermes_cli.pty_bridge import _MAX_COLS, _MAX_ROWS, _clamp_dimension

        cols = _clamp_dimension(131072, _MAX_COLS)
        rows = _clamp_dimension(1, _MAX_ROWS)
        # Should not raise.
        _struct.pack("HHHH", rows, cols, 0, 0)


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

    def test_close_signals_child_process_group(self, monkeypatch):
        sent: list[tuple[int, signal.Signals]] = []

        class _FakeProc:
            pid = 12345
            fd = -1

            def __init__(self):
                self.alive = True

            def isalive(self):
                return self.alive

            def kill(self, sig):
                raise AssertionError(f"single-process kill used: {sig}")

            def close(self, force=False):
                self.closed = force

        fake = _FakeProc()

        def fake_killpg(pgid, sig):
            sent.append((pgid, sig))
            fake.alive = False

        monkeypatch.setattr(os, "getpgid", lambda pid: 67890)
        monkeypatch.setattr(os, "killpg", fake_killpg)

        bridge = PtyBridge.__new__(PtyBridge)
        bridge._proc = fake
        bridge._fd = -1
        bridge._closed = False

        bridge.close()

        assert sent == [(67890, signal.SIGHUP)]
        assert bridge._closed is True

    def test_close_falls_back_to_single_process_signal_when_group_unknown(self, monkeypatch):
        sent: list[signal.Signals] = []

        class _FakeProc:
            pid = 12345
            fd = -1

            def __init__(self):
                self.alive = True

            def isalive(self):
                return self.alive

            def kill(self, sig):
                sent.append(sig)
                self.alive = False

            def close(self, force=False):
                self.closed = force

        monkeypatch.setattr(os, "getpgid", lambda pid: (_ for _ in ()).throw(OSError()))

        bridge = PtyBridge.__new__(PtyBridge)
        bridge._proc = _FakeProc()
        bridge._fd = -1
        bridge._closed = False

        bridge.close()

        assert sent == [signal.SIGHUP]


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

    def test_dumb_term_gets_pty_default(self):
        bridge = PtyBridge.spawn(
            ["/bin/sh", "-c", "printf %s \"$TERM\""],
            env={**os.environ, "TERM": "dumb"},
        )
        try:
            output = _read_until(bridge, b"xterm-256color")
            assert b"xterm-256color" in output
        finally:
            bridge.close()


class TestPtyBridgeUnavailable:
    """Platform fallback semantics — PtyUnavailableError is importable and
    carries a user-readable message."""

    def test_error_carries_user_message(self):
        err = PtyUnavailableError("platform not supported")
        assert "platform" in str(err)
