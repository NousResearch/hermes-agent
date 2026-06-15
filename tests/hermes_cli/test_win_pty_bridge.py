"""Unit tests for hermes_cli.win_pty_bridge — ConPTY spawning + byte forwarding.

Windows-only counterpart to tests/hermes_cli/test_pty_bridge.py.  Drives
``WinPtyBridge`` with minimal Windows processes (``cmd.exe``, ``python -c …``)
to verify it behaves like a PTY you can read/write/resize/close, then a small
set of platform-fallback assertions (``is_available``, ``PtyUnavailableError``)
that run on every OS so the import surface stays exercised in CI.

The bridge is the ConPTY backend behind the dashboard ``/chat`` tab — see
``hermes_cli/web_server.py`` ``/api/pty`` handler — so these tests are the
unit-level half of the integration check that the dashboard chat pane is
actually live on native Windows.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

# WinPtyBridge can be imported on every platform — ``is_available`` just
# returns False when pywinpty isn't usable.  Importing the module itself
# must never raise, otherwise the web_server import branch becomes a trap.
from hermes_cli.win_pty_bridge import PtyUnavailableError, WinPtyBridge

windows_only = pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="ConPTY bridge is Windows-only",
)


def _read_until(bridge: WinPtyBridge, needle: bytes, timeout: float = 10.0) -> bytes:
    """Accumulate PTY output until we see ``needle`` or time out.

    Mirrors the helper in test_pty_bridge.py so failures look familiar.
    """
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


# ---------------------------------------------------------------------------
# Cross-platform fallback semantics
# ---------------------------------------------------------------------------


class TestWinPtyBridgeUnavailable:
    """Module-level surface that must stay importable on every OS so the
    web_server platform branch doesn't blow up at import time when pywinpty
    is missing or the host isn't Windows."""

    def test_error_is_importable_and_carries_message(self):
        err = PtyUnavailableError("conpty missing")
        assert "conpty" in str(err)

    def test_bridge_class_is_importable(self):
        # The platform-branched import in web_server.py relies on this:
        #     from hermes_cli.win_pty_bridge import WinPtyBridge, PtyUnavailableError
        # Both symbols must always exist; ``is_available()`` is the gate.
        assert WinPtyBridge is not None
        assert callable(WinPtyBridge.is_available)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="non-Windows only")
    def test_spawn_raises_unavailable_off_windows(self):
        with pytest.raises(PtyUnavailableError):
            WinPtyBridge.spawn(["true"])


# ---------------------------------------------------------------------------
# Windows-only end-to-end behaviour
# ---------------------------------------------------------------------------


@windows_only
class TestWinPtyBridgeSpawn:
    def test_is_available_on_windows(self):
        assert WinPtyBridge.is_available() is True

    def test_spawn_returns_bridge_with_pid(self):
        bridge = WinPtyBridge.spawn(["cmd.exe", "/c", "exit 0"])
        try:
            assert bridge.pid > 0
        finally:
            bridge.close()

    def test_spawn_raises_on_missing_argv0(self, tmp_path):
        # pywinpty wraps CreateProcessW failures; surface as OSError / RuntimeError.
        bogus = str(tmp_path / "definitely-not-a-real-binary.exe")
        with pytest.raises((FileNotFoundError, OSError, RuntimeError, PtyUnavailableError)):
            WinPtyBridge.spawn([bogus])


@windows_only
class TestWinPtyBridgeIO:
    def test_reads_child_stdout(self):
        bridge = WinPtyBridge.spawn(["cmd.exe", "/c", "echo hermes-ok"])
        try:
            output = _read_until(bridge, b"hermes-ok")
            assert b"hermes-ok" in output
        finally:
            bridge.close()

    def test_write_sends_to_child_stdin(self):
        # python -c reads stdin, echoes a marker, exits.  More reliable than
        # ``cat`` (not on Windows) and doesn't depend on a particular shell.
        script = (
            "import sys; "
            "line = sys.stdin.readline().strip(); "
            "sys.stdout.write('GOT:' + line + '\\n'); "
            "sys.stdout.flush()"
        )
        bridge = WinPtyBridge.spawn([sys.executable, "-c", script])
        try:
            bridge.write(b"hello-pty\r\n")
            output = _read_until(bridge, b"GOT:hello-pty")
            assert b"GOT:hello-pty" in output
        finally:
            bridge.close()

    def test_write_after_close_is_silent(self):
        bridge = WinPtyBridge.spawn(["cmd.exe", "/c", "exit 0"])
        bridge.close()
        # Must not raise — the dashboard WebSocket reader sometimes writes
        # a final keystroke after the user has already closed the tab.
        bridge.write(b"ignored")

    def test_read_returns_none_after_child_exits(self):
        bridge = WinPtyBridge.spawn(["cmd.exe", "/c", "echo done"])
        try:
            _read_until(bridge, b"done")
            # Give the child a beat to exit, then drain until EOF.
            deadline = time.monotonic() + 5.0
            while bridge.is_alive() and time.monotonic() < deadline:
                bridge.read(timeout=0.1)
            got_none = False
            for _ in range(20):
                if bridge.read(timeout=0.1) is None:
                    got_none = True
                    break
            assert got_none, "WinPtyBridge.read did not return None after child EOF"
        finally:
            bridge.close()


@windows_only
class TestWinPtyBridgeResize:
    def test_resize_does_not_raise_on_live_child(self):
        # ConPTY exposes no ioctl-equivalent for reading the child's current
        # winsize from Python land, so we can't verify the new dimensions
        # the way the POSIX test does (which reads TIOCGWINSZ).  What we
        # CAN guarantee is what the dashboard depends on: ``resize`` never
        # raises, the bridge stays alive, and subsequent I/O still works.
        bridge = WinPtyBridge.spawn(
            [sys.executable, "-c", "import time; time.sleep(1.0)"],
            cols=80,
            rows=24,
        )
        try:
            bridge.resize(cols=123, rows=45)
            assert bridge.is_alive()
        finally:
            bridge.close()

    def test_resize_clamps_garbage_dimensions(self):
        # Mirror the POSIX clamp test: a broken winsize probe must never
        # propagate to the ConPTY API.  131072 > unsigned short max — the
        # bridge has to coerce it down without raising.
        bridge = WinPtyBridge.spawn(
            [sys.executable, "-c", "import time; time.sleep(1.0)"],
            cols=80,
            rows=24,
        )
        try:
            bridge.resize(cols=131072, rows=1)  # must not raise
            bridge.resize(cols=0, rows=-5)      # nor this
            assert bridge.is_alive()
        finally:
            bridge.close()

    def test_resize_after_close_is_silent(self):
        bridge = WinPtyBridge.spawn(["cmd.exe", "/c", "exit 0"])
        bridge.close()
        # Must not raise — closed bridges still receive late resize escapes
        # from xterm.js when the browser tab is closed mid-stream.
        bridge.resize(cols=100, rows=40)


@windows_only
class TestClampDimension:
    """The clamp helper is the load-bearing piece — the dashboard sends
    untrusted winsize values straight from xterm.js, and pywinpty's
    setwinsize will happily raise on out-of-range u16 values."""

    def test_clamps_above_max(self):
        from hermes_cli.win_pty_bridge import _MAX_COLS, _MAX_ROWS, _clamp

        assert _clamp(131072, _MAX_COLS) == _MAX_COLS
        assert _clamp(131072, _MAX_ROWS) == _MAX_ROWS

    def test_floors_at_one(self):
        from hermes_cli.win_pty_bridge import _MAX_COLS, _clamp

        assert _clamp(0, _MAX_COLS) == 1
        assert _clamp(-5, _MAX_COLS) == 1

    def test_passes_through_sane_values(self):
        from hermes_cli.win_pty_bridge import _MAX_COLS, _clamp

        assert _clamp(80, _MAX_COLS) == 80
        assert _clamp(2000, _MAX_COLS) == 2000

    def test_non_numeric_falls_back_to_min(self):
        from hermes_cli.win_pty_bridge import _MAX_COLS, _clamp

        assert _clamp(None, _MAX_COLS) == 1  # type: ignore[arg-type]
        assert _clamp("not-a-number", _MAX_COLS) == 1  # type: ignore[arg-type]
        assert _clamp(float("nan"), _MAX_COLS) == 1  # type: ignore[arg-type]
        assert _clamp(float("inf"), _MAX_COLS) == 1  # type: ignore[arg-type]


@windows_only
class TestWinPtyBridgeClose:
    def test_close_is_idempotent(self):
        bridge = WinPtyBridge.spawn(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        bridge.close()
        bridge.close()  # must not raise
        assert not bridge.is_alive()

    def test_close_terminates_long_running_child(self):
        bridge = WinPtyBridge.spawn(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        pid = bridge.pid
        assert bridge.is_alive(), f"child pid {pid} not alive before close"
        bridge.close()
        # The bridge itself reports liveness via pywinpty.isalive(), which is
        # the same probe the dashboard PTY reader uses to decide when to stop
        # forwarding bytes — verifying that flips to False is the contract
        # that matters for /api/pty.
        deadline = time.monotonic() + 5.0
        while bridge.is_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not bridge.is_alive(), (
            f"WinPtyBridge.is_alive() still True after close(); pid {pid}"
        )


@windows_only
class TestWinPtyBridgeEnv:
    def test_cwd_is_respected(self, tmp_path):
        bridge = WinPtyBridge.spawn(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=str(tmp_path),
        )
        try:
            # Path is case-insensitive on Windows; compare lowercased.
            needle_resolved = str(tmp_path.resolve()).lower().encode()
            deadline = time.monotonic() + 5.0
            buf = bytearray()
            while time.monotonic() < deadline:
                chunk = bridge.read(timeout=0.2)
                if chunk is None:
                    break
                buf.extend(chunk)
                if needle_resolved in bytes(buf).lower():
                    break
            assert needle_resolved in bytes(buf).lower(), (
                f"cwd {tmp_path!s} not echoed by child; got {bytes(buf)!r}"
            )
        finally:
            bridge.close()

    def test_env_is_forwarded(self):
        bridge = WinPtyBridge.spawn(
            [
                sys.executable,
                "-c",
                "import os; print('HERMES_PTY_TEST=' + os.environ.get('HERMES_PTY_TEST',''))",
            ],
            env={**os.environ, "HERMES_PTY_TEST": "pty-env-works"},
        )
        try:
            output = _read_until(bridge, b"pty-env-works")
            assert b"pty-env-works" in output
        finally:
            bridge.close()

    def test_spawn_defaults_term_when_not_set(self):
        # The bridge should set TERM=xterm-256color when the caller's env
        # doesn't already carry one — xterm.js expects ANSI/SGR sequences.
        env = {k: v for k, v in os.environ.items() if k.upper() != "TERM"}
        bridge = WinPtyBridge.spawn(
            [
                sys.executable,
                "-c",
                "import os; print('TERM=' + os.environ.get('TERM',''))",
            ],
            env=env,
        )
        try:
            output = _read_until(bridge, b"TERM=")
            assert b"TERM=xterm-256color" in output
        finally:
            bridge.close()
