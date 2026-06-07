"""Unit tests for hermes_cli.pty_bridge — PTY spawning + byte forwarding.

These tests drive the bridge with minimal POSIX processes (echo, env, sleep,
printf) to verify it behaves like a PTY you can read/write/resize/close.
"""

from __future__ import annotations

import os
import shutil
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

    def test_close_does_not_send_sighup(self, monkeypatch):
        """Regression: ``close()`` used to fire SIGHUP → SIGTERM → SIGKILL
        with 0.5s per stage. The first two signals landed on the child
        within the same second, racing the TUI gateway's own
        ``_log_signal`` shutdown handler (`tui_gateway/entry.py`) which
        installs a 1s daemon timer for atexit drain.

        Reproduction (real-world, captured 2026-05-24):
        Opening a second ``hermes dashboard --tui`` browser tab while
        the first is mid-vision triggers the first tab's
        ``WebSocketDisconnect`` → ``bridge.close()``. The 0.5s-per-stage
        cascade resulted in ``tui_gateway_crash.log`` recording both
        SIGHUP and SIGTERM at the same timestamp.

        Fix: drop SIGHUP entirely. Closing the master fd in
        ``self._proc.close(force=True)`` already delivers EOF on the
        child's stdin, which is the kernel's way of signalling the
        controlling terminal is gone — sending SIGHUP on top duplicates
        the signal and races the child's own shutdown handler.
        """
        # Set a very short grace so the test runs fast; SIGHUP behavior
        # is independent of grace duration.
        monkeypatch.setenv("HERMES_PTY_BRIDGE_TERM_GRACE_S", "0.2")

        # Use a tempfile marker the child writes if it receives SIGHUP.
        import tempfile
        marker = tempfile.NamedTemporaryFile(
            prefix="pty_bridge_sighup_", suffix=".marker", delete=False
        )
        marker.close()
        os.unlink(marker.name)  # we want the SIGHUP handler to create it

        # Child: trap SIGHUP and write the marker. Trap SIGTERM and exit
        # cleanly. If close() sends SIGHUP first (old behavior), the
        # marker will be created. If close() sends only SIGTERM (new
        # behavior), the marker won't exist.
        bridge = PtyBridge.spawn([
            "/bin/sh", "-c",
            f"trap 'echo HUP > {marker.name}; exit 0' HUP; "
            "trap 'exit 0' TERM; "
            "while true; do sleep 0.05; done",
        ])
        time.sleep(0.2)  # let trap install
        bridge.close()
        time.sleep(0.3)  # give the SIGHUP handler time to create marker

        try:
            assert not os.path.exists(marker.name), (
                "SIGHUP marker was created — close() still sends SIGHUP "
                "(should be SIGTERM only, with master-fd-close as the "
                "controlling-terminal-gone signal)"
            )
        finally:
            try:
                os.unlink(marker.name)
            except FileNotFoundError:
                pass

    def test_close_grace_lets_child_finish_atexit(self, monkeypatch):
        """SIGTERM grace must let the child run a small graceful shutdown
        before SIGKILL. Default grace is 2.0s in production; tunable for
        tests via HERMES_PTY_BRIDGE_TERM_GRACE_S.

        Verifies a child trapping SIGTERM and exiting cleanly inside the
        grace window is reaped naturally — not killed by SIGKILL.
        """
        monkeypatch.setenv("HERMES_PTY_BRIDGE_TERM_GRACE_S", "1.0")

        # Child traps SIGTERM, prints a marker, exits 0 after 0.2s —
        # well inside 1.0s grace. If close() incorrectly delivers SIGKILL
        # before the grace expires, the trap output would never reach
        # stdout (SIGKILL is uncatchable).
        bridge = PtyBridge.spawn([
            "/bin/sh", "-c",
            "trap 'echo TERM_TRAPPED; sleep 0.2; exit 0' TERM; "
            "while true; do sleep 0.05; done",
        ])
        time.sleep(0.2)  # let trap install
        pid = bridge.pid

        before_close = time.monotonic()
        bridge.close()
        elapsed = time.monotonic() - before_close

        # The child should have exited via SIGTERM trap in < 1.0s
        # (0.2s sleep + ~0.05s overhead). If we ran the full grace +
        # SIGKILL stage, elapsed would be > 1.0s. Allow generous slack
        # for slow CI: < 1.5s confirms SIGKILL never fired.
        assert elapsed < 1.5, (
            f"close() took {elapsed:.2f}s — SIGKILL likely fired before "
            f"SIGTERM grace expired. Expected child to exit via SIGTERM "
            f"trap in ~0.25s."
        )

        # Confirm reaped.
        deadline = time.monotonic() + 1.0
        reaped = False
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.02)
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
