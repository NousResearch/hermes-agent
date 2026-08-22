"""Windows gateway detached-spawn trampoline (#84311).

On Windows the desktop app tree-kills its backend on close
(``taskkill /PID <backend> /T /F``). The messaging gateway used to die with
it whenever a live management intermediate still linked it into the backend's
process tree — and nothing restarted it on reopen.

``gateway_windows._spawn_gateway_via_trampoline`` spawns the gateway through a
short-lived intermediate that exits ~immediately after spawning, so the
gateway's parent chain back to the desktop backend is broken within ~100ms.
These tests run the real spawn + tree-kill on Windows and assert the gateway
survives exactly like the Desktop close would.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

pytestmark = [
    pytest.mark.windows_only,
    pytest.mark.live_system_guard_bypass,  # real process spawn + taskkill
]

# CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB —
# the same detach flags the repo's gateway/backend spawns use.
_WIN_DETACH_FLAGS = 0x00000200 | 0x08000000 | 0x01000000

_SLEEP_SRC = "import time; time.sleep(120)"


def _pid_exists(pid: int) -> bool:
    import psutil

    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def _ppid_of(pid: int) -> int | None:
    import psutil

    try:
        return psutil.Process(pid).ppid()
    except psutil.NoSuchProcess:
        return None


def _taskkill(pid: int) -> int:
    result = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode


def test_spawned_gateway_survives_backend_tree_kill(tmp_path):
    """The trampoline breaks the parent chain, so `taskkill /T` on the
    spawning backend cannot reach the gateway (the Desktop close scenario)."""
    from hermes_cli.gateway_windows import _spawn_gateway_via_trampoline

    # A fake "desktop backend" (the desktop spawns `hermes serve` this way).
    backend = subprocess.Popen(
        [sys.executable, "-c", _SLEEP_SRC],
        creationflags=_WIN_DETACH_FLAGS,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    gw_pid: int | None = None
    try:
        fake_gateway = [sys.executable, "-c", _SLEEP_SRC]
        stray_log = tmp_path / "gateway-stdio.log"
        gw_pid = _spawn_gateway_via_trampoline(fake_gateway, os.getcwd(), {}, stray_log)

        assert gw_pid != backend.pid
        # The returned PID is the real gateway (trampoline exits right after).
        time.sleep(1.0)
        assert _pid_exists(gw_pid), "gateway did not come up"
        assert _ppid_of(gw_pid) != backend.pid, "gateway is still in the backend tree"

        # The desktop close tree-kills the backend — the gateway must survive.
        assert _taskkill(backend.pid) == 0
        time.sleep(0.5)
        assert _pid_exists(gw_pid), "gateway was killed by the backend tree-kill"
    finally:
        for pid in (gw_pid, backend.pid):
            if pid is not None:
                try:
                    _taskkill(pid)
                except Exception:
                    pass


def test_trampoline_returns_real_gateway_pid(tmp_path):
    """The PID contract: callers get the gateway PID, not the trampoline's."""
    from hermes_cli.gateway_windows import _spawn_gateway_via_trampoline

    fake_gateway = [sys.executable, "-c", _SLEEP_SRC]
    stray_log = tmp_path / "gateway-stdio.log"
    gw_pid = _spawn_gateway_via_trampoline(fake_gateway, os.getcwd(), {}, stray_log)
    try:
        # The gateway's own process must match the reported PID.
        time.sleep(0.5)
        assert _pid_exists(gw_pid)
        # And it must not be the parent of anything weird — its parent is the
        # dead trampoline (dangling PID), not this test process.
        assert _ppid_of(gw_pid) != os.getpid()
    finally:
        try:
            _taskkill(gw_pid)
        except Exception:
            pass
