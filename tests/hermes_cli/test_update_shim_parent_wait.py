"""A child spawned off the Windows ``hermes.exe`` shim waits for its parent before touching the venv.

Regression for #101600: the post-swap / re-exec child reached the shim quarantine while the
shim-run parent was still alive (relaunching gateways, or merely tearing down), the single
``os.rename`` failed with a PermissionError and the whole dependency install was deferred. The
wait is host-independent (env pid + psutil), so it is exercised with real processes here,
through the production entry point (``cmd_update``) rather than the helpers alone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from hermes_cli import main_install_repair, update_handoff, update_lock

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs ``cmd_update`` as the detached child would, with every phase past the hand-off boundary
# stubbed; the first thing after the Windows pause step (``_desktop_app_present``) reports what
# the child saw and stops the run.
_CHILD = r"""
import atexit, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import psutil
from hermes_cli import main as cli_main, update_cmd, update_handoff, update_lock

seen = {"paused": 0, "registered": None}
parent_pid = int(os.environ[update_handoff.SHIM_PARENT_PID_ENV])
cli_main._update_preflight_handled = lambda args: False
cli_main._install_hangup_protection = lambda **kw: None
cli_main._finalize_update_output = lambda state: None
cli_main._finalize_update_receipt = lambda *a, **kw: None
cli_main._run_pre_update_backup = lambda args: None
update_cmd._resolve_update_options = lambda args, gateway_mode: SimpleNamespace(gw_input_fn=None, assume_yes=True)
update_cmd._begin_update_receipt_and_plan = lambda args: None
update_cmd._record_pre_update_backup_outcome = lambda args, snapshot_id: None

def _pause():
    seen["paused"] += 1
    return {"resume_needed": True, "profiles": {"rediscovered": 1}}

cli_main._pause_windows_gateways_for_update = _pause
_register = atexit.register
def _capture(fn, *a, **kw):
    if a and isinstance(a[0], dict):
        seen["registered"] = a[0]
    else:
        _register(fn, *a, **kw)
atexit.register = _capture

class Stop(Exception):
    pass

def _phase(desktop_dir):
    try:
        seen["parent_status"] = psutil.Process(parent_pid).status()
    except psutil.NoSuchProcess:
        seen["parent_status"] = "gone"
    try:
        seen["lock_owner"] = int(update_lock.update_marker_path().read_text().split()[0])
    except OSError:
        seen["lock_owner"] = None
    seen["own_pid"] = os.getpid()
    seen["env_left"] = [k for k in (update_handoff.SHIM_PARENT_PID_ENV, update_handoff.GATEWAY_RESUME_ENV) if k in os.environ]
    raise Stop

update_cmd._desktop_app_present = _phase

Path(os.environ["HERMES_TEST_READY"]).touch()
try:
    cli_main.cmd_update(SimpleNamespace(post_swap=None, yes=True, gateway=False))
except Stop:
    pass
seen["marker_after"] = update_lock.update_marker_path().exists()
print(json.dumps(seen))
"""


def test_update_child_outwaits_shim_parent_then_owns_the_lock_and_the_resume_token(tmp_path):
    """Real topology: the shim-run parent is OLDER than the child and holds the update marker;
    the child (a fresh ``hermes update``) must not pause gateways, scan holders or claim the
    lock until the parent is gone — then it runs under its OWN marker, with the parent's pause
    token instead of a fresh discovery (#101600)."""
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    ready = tmp_path / "ready"
    token = {"resume_needed": True, "profiles": {"default": 4}, "unmapped": []}
    # The parent lives until we close its stdin — the marker is its claim, as in a real run.
    parent = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    marker = hermes_home / update_lock.MARKER_NAME
    marker.write_text(f"{parent.pid}\n{int(time.time())}\n", encoding="utf-8")
    env = {
        **os.environ, "HERMES_HOME": str(hermes_home), "HERMES_TEST_READY": str(ready),
        update_lock.HANDOFF_PID_ENV: str(parent.pid),
        update_handoff.SHIM_PARENT_PID_ENV: str(parent.pid),
        update_handoff.GATEWAY_RESUME_ENV: json.dumps(token),
    }
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD], cwd=str(REPO_ROOT), env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert child.poll() is None, child.communicate()[1]
            assert time.monotonic() < deadline, "child never reached cmd_update"
            time.sleep(0.05)
        # The child is inside cmd_update now; give a child that does NOT wait ample time to
        # reach the phase while the parent is still alive.
        time.sleep(1.0)
        assert child.poll() is None, child.communicate()[1]
        # Parent exits: its lock release removes the marker, exactly as cmd_update's finally does.
        parent.stdin.close()
        parent.wait(timeout=30)
        marker.unlink(missing_ok=True)
        out, err = child.communicate(timeout=60)
    finally:
        parent.kill()
        parent.wait()
    assert child.returncode == 0, err
    seen = json.loads(out.strip().splitlines()[-1])
    assert seen["parent_status"] in ("gone", psutil.STATUS_ZOMBIE), seen
    # Its own claim, not the departed parent's. Compare against the interpreter's own pid: on
    # Windows a uv venv's Scripts\python.exe is a trampoline, so Popen.pid is the launcher and
    # the process that ran cmd_update is its child (wine2e run 35429725262).
    assert seen["lock_owner"] == seen["own_pid"] != parent.pid, seen
    assert seen["paused"] == 0 and seen["registered"] == token, seen
    assert seen["env_left"] == [] and seen["marker_after"] is False, seen


def test_wait_is_bounded_and_skips_absent_garbage_and_recycled_pids(monkeypatch):
    def _sleeper(seconds: float) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({seconds})"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # No shim parent named (every non-shim run, incl. Linux): no wait at all.
    monkeypatch.delenv(update_handoff.SHIM_PARENT_PID_ENV, raising=False)
    assert update_handoff.wait_for_shim_parent_exit(timeout=5) is True
    monkeypatch.setenv(update_handoff.SHIM_PARENT_PID_ENV, "not-a-pid")
    assert update_handoff.wait_for_shim_parent_exit(timeout=5) is True

    # A live process YOUNGER than us cannot be our parent: a recycled pid is never waited on.
    younger = _sleeper(30)
    try:
        assert psutil.Process(younger.pid).create_time() >= psutil.Process().create_time()
        monkeypatch.setenv(update_handoff.SHIM_PARENT_PID_ENV, str(younger.pid))
        started = time.monotonic()
        assert update_handoff.wait_for_shim_parent_exit(timeout=5) is True
        assert time.monotonic() - started < 2.0 and younger.poll() is None
    finally:
        younger.kill()
        younger.wait()

    # A parent that never exits only delays the child by the bound; the strict shim quarantine
    # downstream stays the fail-closed guard. Our own parent is older than us and outlives us.
    monkeypatch.setenv(update_handoff.SHIM_PARENT_PID_ENV, str(os.getppid()))
    assert update_handoff.wait_for_shim_parent_exit(timeout=0.5) is False


def test_shim_holder_is_the_launcher_ancestor_not_the_interpreter_it_ran():
    """``hermes.exe`` spawns python.exe with the shim as argv[0] and exits only after reaping it,
    so the pid the child outwaits must be the ancestor whose EXECUTABLE is the shim."""
    me = psutil.Process()
    launcher = next((p for p in me.parents() if p.exe() != me.exe()), None)
    assert launcher is not None, "test runner has no ancestor with a different executable"

    def match(candidate):
        return Path(candidate) if str(candidate) == launcher.exe() else None

    assert main_install_repair._windows_shim_ancestor(match) == (Path(launcher.exe()), launcher.pid)
    assert main_install_repair._windows_shim_ancestor(lambda candidate: None) is None
