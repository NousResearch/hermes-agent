"""LIVE Windows proof: Hermes' intentional detaches launched from inside a real
``LocalEnvironment._run_bash`` terminal command leave its membership-only job.

``_run_bash`` contains the Git Bash wrapper in a job WITHOUT BREAKAWAY_OK (MSYS
fork/exec requests CREATE_BREAKAWAY_FROM_JOB itself, so allowing breakaway would let
the MSYS-handoff grep escape the job again). A plain ``CREATE_BREAKAWAY_FROM_JOB``
spawn from a member therefore gets ERROR_ACCESS_DENIED. ``popen_detached`` uses an
exiting trampoline under the job's non-member owner (PROC_THREAD_ATTRIBUTE_PARENT_PROCESS),
so the child also leaves the owner's live tree. A sacrificial owner is tree-killed live.

Driven through the REAL ``_run_bash`` (real Git Bash, real job), no mocks of the code
under test. The paths exercised from inside the command:
  * ``gateway_windows._spawn_detached`` (gateway start, update relaunch),
  * ``gateway._spawn_gateway_restart_watcher`` (watcher + its respawn).
Each must succeed, be a non-member of the job (``IsProcessInJob``), and survive
``job.terminate()``. Ownership of every process is proven by (pid, create_time) taken
from the process itself, never by name or argv.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import subprocess
import sys
import textwrap
import time
from ctypes import wintypes
from pathlib import Path

import psutil
import pytest

from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.platforms("windows")  # live native Git Bash proof

_REPO_ROOT = Path(__file__).resolve().parents[2]

_INNER = textwrap.dedent(r'''
    import json, os, subprocess, sys, time
    repo, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, repo)
    os.environ["HERMES_HOME"] = os.path.join(out, "home")
    from hermes_cli import gateway, gateway_windows as gw
    from hermes_cli import _windows_job_escape as esc
    ANCHOR_ENV = esc.ANCHOR_ENV
    real_anchor_spawn = esc._spawn_via_anchor
    anchor_errors = []
    def observe_anchor(*args, **kwargs):
        try:
            return real_anchor_spawn(*args, **kwargs)
        except OSError as exc:
            import traceback
            anchor_errors.append(traceback.format_exc())
            raise
    esc._spawn_via_anchor = observe_anchor
    real_spawn, spawned = esc.popen_detached, []
    def observe(*args, **kwargs):
        proc = real_spawn(*args, **kwargs)
        spawned.append(proc.pid)
        return proc
    esc.popen_detached = observe

    def record(path_literal):
        return ("import os, time, json; "
                f"open({path_literal!r}, 'w').write(json.dumps([os.getpid(), time.time()])); "
                "print('detach-stdio-ok', flush=True); time.sleep(120)")

    child_python = os.path.join(os.path.dirname(sys.executable), "python.exe")
    result = {"anchor_present": ANCHOR_ENV in os.environ}
    gw._build_gateway_argv = lambda home=None: (
        [child_python, "-c", record(os.path.join(out, "gateway.json"))], out, {})
    result["gateway_pid"] = gw._spawn_detached()
    result["breakaway_fallback"] = gw._LAST_SPAWN_BREAKAWAY_FALLBACK["fallback"]
    # A job MEMBER the watcher waits on; the test's job.terminate() ends it.
    old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    result["old_pid"] = old.pid
    respawn = [child_python, "-c", record(os.path.join(out, "respawn.json"))]
    result["watcher_started"] = gateway._spawn_gateway_restart_watcher(old.pid, respawn)
    result["watcher_pid"] = spawned[-1]
    result["anchor_errors"] = anchor_errors
    with open(os.path.join(out, "inner.json"), "w") as fh:
        json.dump(result, fh)
''')


def _in_job(pid: int, job_handle) -> bool:
    from hermes_cli._windows_job_escape import _kernel32
    k32 = _kernel32()
    k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    assert handle, f"cannot open pid {pid}"
    try:
        member = wintypes.BOOL()
        assert k32.IsProcessInJob(handle, job_handle, ctypes.byref(member))
        return bool(member.value)
    finally:
        k32.CloseHandle(handle)


def _wait_json(path: Path, timeout: float = 30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            with contextlib.suppress(ValueError, OSError):
                return json.loads(path.read_text())
        time.sleep(0.1)
    raise AssertionError(f"{path.name} never appeared")


_ANCHOR = textwrap.dedent(r'''
    import json, os, site, sys, time
    from pathlib import Path
    repo, out, runtime = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    site.addsitedir(str(Path(runtime).parents[1] / "Lib" / "site-packages"))
    sys.path.insert(0, repo)
    os.environ["HERMES_HOME"] = str(out / "home")
    from tools.environments.local import LocalEnvironment
    env = LocalEnvironment(cwd=str(out), timeout=60)
    proc = env._run_bash(" ".join('"' + Path(p).as_posix() + '"' for p in (
        runtime, out / "inner.py", repo, out)))
    assert proc.wait(timeout=60) == 0, proc.stdout.read()
    (out / "anchor.json").write_text(json.dumps({
        "pid": os.getpid(), "job_handle": proc._hermes_win32_job._handle}))
    time.sleep(120)
''')


def _borrow_job(anchor, job_handle):
    from hermes_cli._windows_job_escape import _kernel32
    k32 = _kernel32()
    parent = k32.OpenProcess(0x40, False, anchor.pid)
    assert parent
    try:
        handle = wintypes.HANDLE()
        assert k32.DuplicateHandle(parent, job_handle, k32.GetCurrentProcess(),
                                   ctypes.byref(handle), 0, False, 2)
        return k32, handle
    finally:
        k32.CloseHandle(parent)


def _assert_outside(pid, job, anchors):
    proc = psutil.Process(pid)
    assert not _in_job(pid, job), f"detach {pid} joined terminal job"
    for anchor in anchors:
        descendants = {p.pid for p in anchor.children(recursive=True)}
        assert pid not in descendants, f"detach {pid} remains a live descendant of anchor {anchor.pid}"
    return proc


@pytest.mark.parametrize("venv_anchor, windowless", [(False, False), (True, False), (False, True)])
def test_intentional_detaches_leave_the_terminal_job_and_anchor_tree(tmp_path, venv_anchor, windowless):
    (tmp_path / "inner.py").write_text(_INNER)
    (tmp_path / "anchor.py").write_text(_ANCHOR)
    owned: list[psutil.Process] = []
    k32, job = None, None
    # Exercise both a managed bare host and an extra venv-launcher job around that host.
    runtime = sys.executable if venv_anchor else sys._base_executable
    command_python = str(Path(sys.executable).with_name("pythonw.exe")) if windowless else sys.executable
    launcher = subprocess.Popen([runtime, str(tmp_path / "anchor.py"),
                                 str(_REPO_ROOT), str(tmp_path), command_python],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    launcher_identity = psutil.Process(launcher.pid)
    try:
        info = _wait_json(tmp_path / "anchor.json", timeout=60)
        anchor = psutil.Process(info["pid"])
        anchors = [anchor, launcher_identity]
        k32, job = _borrow_job(anchor, info["job_handle"])
        inner = _wait_json(tmp_path / "inner.json")
        assert inner["anchor_present"]
        assert not inner["anchor_errors"], inner["anchor_errors"]
        assert inner["watcher_started"] is True
        assert inner["breakaway_fallback"] is False
        old = psutil.Process(inner["old_pid"])
        owned.append(old)
        assert _in_job(old.pid, job), "ordinary child must be a job member"
        gateway_pid, _ = _wait_json(tmp_path / "gateway.json")
        owned.append(_assert_outside(gateway_pid, job, anchors))
        owned.append(_assert_outside(inner["gateway_pid"], job, anchors))
        watcher = _assert_outside(inner["watcher_pid"], job, anchors)
        owned.append(watcher)
        # Check both ends of a venv launcher's chain.
        watcher_children = watcher.children(recursive=True)
        for child in watcher_children:
            owned.append(_assert_outside(child.pid, job, anchors))

        killed = subprocess.run(["taskkill", "/PID", str(launcher.pid), "/T", "/F"],
                                capture_output=True, text=True, timeout=15)
        assert killed.returncode == 0, killed.stdout + killed.stderr
        launcher.wait(timeout=10)
        anchor.wait(timeout=10)
        assert not anchor.is_running()
        time.sleep(2)  # Allow asynchronous tree/job teardown to finish before claiming survival.
        assert watcher.is_running(), "restart watcher died with the anchor tree"
        assert all(p.is_running() for p in watcher_children)
        assert psutil.Process(gateway_pid).is_running(), "gateway died with the anchor tree"

        # Keep the actual job handle after its owner exits: sweep the ordinary member and
        # release the watcher. The respawn must also be outside that job.
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        assert k32.TerminateJobObject(job, 1)
        old.wait(timeout=10)
        respawn_pid, _ = _wait_json(tmp_path / "respawn.json")
        owned.append(_assert_outside(respawn_pid, job, []))
        assert psutil.Process(gateway_pid).is_running(), "gateway died with the terminal job"
        assert psutil.Process(respawn_pid).is_running(), "respawn died with the terminal job"
        assert k32.TerminateJobObject(job, 1)
        assert psutil.Process(respawn_pid).is_running()
        log = tmp_path / "home" / "logs" / "gateway-stdio.log"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and log.read_text().count("detach-stdio-ok") < 2:
            time.sleep(0.1)
        assert log.read_text().count("detach-stdio-ok") == 2
    finally:
        if job is not None:
            k32.TerminateJobObject(job, 1)
            k32.CloseHandle(job)
        for proc in list(owned):
            with contextlib.suppress(psutil.Error):
                owned.extend(proc.children(recursive=True))
        for proc in reversed(owned):
            with contextlib.suppress(psutil.Error):
                proc.kill()
        if launcher.poll() is None:
            with contextlib.suppress(psutil.Error):
                for child in launcher_identity.children(recursive=True):
                    child.kill()
                launcher_identity.kill()
        launcher.wait(timeout=10)
        launcher.stdout.close()


def test_stale_anchor_is_not_trusted(monkeypatch):
    """A recycled/dead PID never becomes a parent: identity is (pid, creation time)."""
    from hermes_cli import _windows_job_escape as esc

    k32 = esc._kernel32()
    me = psutil.Process()
    handle = esc._open_anchor(k32, esc.current_anchor())
    assert handle is not None
    k32.CloseHandle(handle)
    assert esc._open_anchor(k32, f"{me.pid}:1") is None  # right pid, wrong creation time
    assert esc._open_anchor(k32, "not-an-anchor") is None


def test_environment_block_normalizes_windows_names():
    from hermes_cli._windows_job_escape import _env_block
    block = _env_block({"z": "last", "Path": "old", "a": "first", "PATH": "new"})
    assert block[:] == "a=first\0PATH=new\0z=last\0\0"


def test_terminal_anchor_is_not_saved_in_bash_snapshot(tmp_path):
    import shlex
    from tools.environments.base_session_env import _export_dump_excluding_session_vars
    from hermes_cli._windows_job_escape import ANCHOR_ENV

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    snapshot = tmp_path / "snapshot.sh"
    snippet = _export_dump_excluding_session_vars(shlex.quote(snapshot.as_posix()))
    try:
        proc = env._run_bash("export USER_KEEP=kept; " + snippet)
        assert proc.wait(timeout=30) == 0, proc.stdout.read()
        contents = snapshot.read_text()
        assert ANCHOR_ENV not in contents
        assert "USER_KEEP" in contents
    finally:
        env.cleanup()
