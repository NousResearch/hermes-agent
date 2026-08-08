"""Contract for the parent-death watchdog on the headless ``hermes serve`` backend.

The desktop Electron app spawns its backend as ``hermes serve --host 127.0.0.1
--port 0`` and tears it down in ``before-quit``.  That teardown is correct but
structurally cannot cover a force-quit / SIGKILL / fatal GPU abort, because the
parent's code never runs.  macOS has no ``PR_SET_PDEATHSIG``, so the kernel
reparents the backend to pid 1 instead of reaping it and it keeps its LISTEN
socket and its ~100 MB forever.  Three such orphans were recovered from one
3-minute restart burst.

No parent-side fix can close this, so the backend reaps itself: it records its
ppid at startup and exits once that ppid changes.  These tests pin the
predicate, the three gates that keep it off every other launch shape, its
placement after the re-exec, and — end to end — that a real serve-shaped child
actually dies when its parent is SIGKILLed.

Mirrors ``tests/test_slash_worker_watchdog.py``, whose watchdog this copies.
"""

from __future__ import annotations

import inspect
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import main as main_mod

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── the orphan predicate ──────────────────────────────────────────────────


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to init/a subreaper.
    assert main_mod._serve_is_orphaned(1234, getppid=lambda: 1) is True


def test_is_orphaned_false_when_direct_parent_is_unchanged():
    original_ppid = 1234
    assert main_mod._serve_is_orphaned(original_ppid, getppid=lambda: original_ppid) is False


# ── the three gates ───────────────────────────────────────────────────────


def test_watchdog_runs_for_a_desktop_spawned_serve_on_posix():
    assert main_mod._should_start_serve_watchdog(
        headless_backend=True, env={"HERMES_DESKTOP": "1"}, os_name="posix"
    ) is True


def test_watchdog_skips_the_interactive_dashboard():
    # cmd_dashboard backs BOTH `dashboard` and `serve`. A human's foreground
    # `hermes dashboard` must never self-reap.
    assert main_mod._should_start_serve_watchdog(
        headless_backend=False, env={"HERMES_DESKTOP": "1"}, os_name="posix"
    ) is False


def test_watchdog_skips_a_standalone_serve():
    # `nohup hermes serve &` legitimately reparents to pid 1 when the launching
    # shell exits — that is a deliberate daemon, not an orphan. Only the
    # desktop's own backend (HERMES_DESKTOP=1) is in scope.
    assert main_mod._should_start_serve_watchdog(
        headless_backend=True, env={}, os_name="posix"
    ) is False


def test_watchdog_skips_windows():
    # The profile re-exec is os.execvpe on POSIX (pid/ppid preserved, so the
    # recorded ppid stays valid) but subprocess.Popen on Windows, where it
    # would not be. Same gate tools/mcp_tool.py uses.
    assert main_mod._should_start_serve_watchdog(
        headless_backend=True, env={"HERMES_DESKTOP": "1"}, os_name="nt"
    ) is False


# ── contract + placement ──────────────────────────────────────────────────


def test_watchdog_contract_has_no_create_time_plumbing():
    assert list(inspect.signature(main_mod._serve_is_orphaned).parameters) == [
        "original_ppid",
        "getppid",
    ]
    assert list(
        inspect.signature(main_mod._start_serve_parent_death_watchdog).parameters
    ) == ["original_ppid"]


def test_cmd_dashboard_starts_the_watchdog_after_the_reexec():
    # The ppid must be recorded on the far side of the named-profile re-exec.
    # Recorded before it, the value would be the pre-exec parent's — and on the
    # Windows branch (subprocess.Popen, not execvpe) a whole different process.
    src = inspect.getsource(main_mod.cmd_dashboard)
    assert "_start_serve_parent_death_watchdog" in src, (
        "cmd_dashboard never starts the watchdog — the helper is dead code"
    )
    assert src.index("os.execvpe") < src.index("_start_serve_parent_death_watchdog"), (
        "watchdog must be started AFTER the re-exec block, not before"
    )


# ── end to end: a real orphan reaps itself ────────────────────────────────


def _child_source(pidfile: Path) -> str:
    """A serve-shaped child: holds a LISTEN socket, runs the real watchdog."""
    return (
        "import os, socket, sys, time\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from hermes_cli.main import _start_serve_parent_death_watchdog\n"
        # The orphans each held a live 127.0.0.1 LISTEN socket; keep one so the
        # test exercises the same shape rather than a bare sleeper.
        "s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(8)\n"
        "_start_serve_parent_death_watchdog(os.getppid())\n"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
        "while True: time.sleep(0.05)\n"
    )


def _parent_source(child_script: Path) -> str:
    # Exit if the child dies, so a child that fails to start surfaces as an
    # immediate "parent died before the child came up" instead of a timeout.
    return (
        "import subprocess, sys\n"
        f"sys.exit(subprocess.Popen([sys.executable, {str(child_script)!r}]).wait())\n"
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _read_pid(pidfile: Path) -> int | None:
    try:
        return int(pidfile.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _reap(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        # Cleanup must never raise: it runs in `finally`, and an exception
        # here would replace the AssertionError that names the real failure.
        pass


# Real SIGKILL delivery is the whole point — a mocked signal cannot orphan a
# process. Both PIDs are spawned by this test; the child is deliberately
# reparented to init, which is exactly what the subtree guard refuses.
@pytest.mark.live_system_guard_bypass
def test_serve_child_self_reaps_when_its_parent_is_sigkilled(tmp_path):
    pidfile = tmp_path / "child.pid"
    child_script = tmp_path / "child.py"
    parent_script = tmp_path / "parent.py"
    child_script.write_text(_child_source(pidfile))
    parent_script.write_text(_parent_source(child_script))

    env = dict(os.environ)
    env["HERMES_DESKTOP"] = "1"
    env["HERMES_SERVE_WATCHDOG_POLL_S"] = "0.1"  # read at import; keeps this fast
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    parent = subprocess.Popen([sys.executable, str(parent_script)], env=env)
    child_pid = None
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            child_pid = _read_pid(pidfile)
            if child_pid is not None:
                break
            if parent.poll() is not None:
                raise AssertionError("parent died before the child came up")
            time.sleep(0.05)
        assert child_pid is not None, "child never started"
        assert _alive(child_pid), "child exited before the parent was killed"

        # Force-quit / fatal abort: the parent's teardown code never runs.
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=10)

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not _alive(child_pid):
                return  # self-reaped
            time.sleep(0.05)
        raise AssertionError(
            f"orphaned serve child {child_pid} survived its parent's SIGKILL — "
            "this is the leak the watchdog exists to prevent"
        )
    finally:
        _reap(parent.pid)
        _reap(child_pid if child_pid is not None else _read_pid(pidfile))
