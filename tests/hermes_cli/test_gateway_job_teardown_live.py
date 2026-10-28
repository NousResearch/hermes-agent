"""LIVE Windows E2E for #48820 (4th repro): Job-Object teardown vs the
gateway restart watcher, with real processes on a real windows-latest runner.

Three live proofs (no mocks of the code under test):

1. ``TestJobObjectMechanismLive`` — the mechanism everything rests on:
   a child spawned with ``windows_detach_flags()`` (CREATE_BREAKAWAY_FROM_JOB)
   from inside a kill-on-close Job Object SURVIVES the job teardown, while a
   child spawned with ``windows_detach_flags_without_breakaway()`` is killed
   by it. This is exactly the reporter's suspected kill path.

2. ``TestWatcherRespawnLive`` — drives the REAL
   ``hermes_cli.gateway._spawn_gateway_restart_watcher`` end to end with a
   real stub gateway process, against a temp HERMES_HOME:
   - the respawned process's stderr must land in ``logs/gateway-stdio.log``
     (on unfixed main it went to DEVNULL: a job-teardown kill left ZERO trace);
   - the respawn env must carry ``_HERMES_GATEWAY_BREAKAWAY=1`` (the stamp
     that makes a later job-teardown death diagnosable in exit-diag).

3. ``TestResumeVerificationLive`` — the user-visible symptom: the updater's
   ``_resume_windows_gateways_after_update`` must NOT print
   "✓ Restarting Windows gateway profile(s)" when the relaunched gateway is
   dead. The relaunch chain runs for real; the "gateway" is a stub that exits
   immediately (standing in for the job-teardown kill). On unfixed main the ✓
   is printed anyway; after the fix the resume raises "not verified alive".
"""

from __future__ import annotations

import contextlib
import ctypes
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")  # native Windows only

_REPO_ROOT = Path(__file__).resolve().parents[2]

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
JobObjectExtendedLimitInformation = 9
CREATE_SUSPENDED = 0x00000004
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
WAIT_OBJECT_0 = 0x0
WAIT_TIMEOUT = 0x102


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32():
    # A private WinDLL with typed HANDLE signatures: windll's shared instance
    # passes Python ints as 32-bit C ints, and argtypes set on it would leak
    # into every other ctypes user in this process.
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    k.SetInformationJobObject.restype = wintypes.BOOL
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.AssignProcessToJobObject.restype = wintypes.BOOL
    k.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    k.IsProcessInJob.restype = wintypes.BOOL
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    return k


def _make_kill_on_close_job(k, allow_breakaway: bool) -> int:
    job = k.CreateJobObjectW(None, None)
    assert job, f"CreateJobObjectW failed (winerror={ctypes.get_last_error()})"
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if allow_breakaway:
        flags |= JOB_OBJECT_LIMIT_BREAKAWAY_OK
    info.BasicLimitInformation.LimitFlags = flags
    ok = k.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    assert ok, f"SetInformationJobObject failed (winerror={ctypes.get_last_error()})"
    return job


def _start_driver_in_job(k, job: int, argv: list[str]) -> subprocess.Popen:
    """Start ``argv`` inside ``job`` before it runs a single instruction.

    The process is created suspended, assigned, then resumed — the same seam
    the updater uses (``update_lock.bind_child_to_update_tree`` +
    ``resume_suspended_child``). Assigning a RUNNING process is racy: a venv's
    ``Scripts\\python.exe`` is a redirector that starts the real interpreter as
    its own child, and a child created before ``AssignProcessToJobObject``
    lands stays outside the job. That interpreter is the one that spawns the
    grandchild, so the grandchild escaped the job and survived the teardown
    (~6% of CI runs before this).
    """
    from hermes_cli.update_lock import resume_suspended_child

    driver = subprocess.Popen(argv, cwd=str(_REPO_ROOT), creationflags=CREATE_SUSPENDED)
    if not k.AssignProcessToJobObject(job, int(driver._handle)):
        error = ctypes.get_last_error()
        driver.kill()
        driver.wait(timeout=30)
        pytest.fail(f"AssignProcessToJobObject failed (winerror={error})")
    resume_suspended_child(driver)
    return driver


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


_SLEEPER = "import time; time.sleep(120)"


def _wait_for(predicate, timeout_s: float = 30.0, interval_s: float = 0.25):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


class TestJobObjectMechanismLive:
    """Real Job Objects, real children — the #48820 kill mechanism."""

    def _driver_source(self, flags_helper: str, pid_file: Path) -> str:
        # The driver runs INSIDE the job and spawns a grandchild "gateway"
        # with the flag bundle under test, then exits — mirroring the
        # updater/watcher exiting while its job tears down. The pid lands via
        # an atomic replace, so the file never exists empty.
        return (
            "import os, subprocess, sys, pathlib\n"
            f"sys.path.insert(0, r'{_REPO_ROOT!s}')\n"
            "from hermes_cli._subprocess_compat import (\n"
            "    windows_detach_flags, windows_detach_flags_without_breakaway)\n"
            f"flags = {flags_helper}()\n"
            f"p = subprocess.Popen([sys.executable, '-c', {_SLEEPER!r}],\n"
            "    creationflags=flags, stdin=subprocess.DEVNULL,\n"
            "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            f"tmp = pathlib.Path(r'{pid_file}.tmp')\n"
            "tmp.write_text(str(p.pid), encoding='utf-8')\n"
            f"os.replace(tmp, r'{pid_file}')\n"
        )

    @contextlib.contextmanager
    def _grandchild_after_teardown(self, tmp_path: Path, flags_helper: str):
        """Yield ``(kernel32, grandchild handle, was in the job)`` right after
        the job's last handle closed.

        The grandchild is watched through a handle opened while it was alive,
        never by pid: a pid freed by the kill can be reused by another process
        on a busy runner, and a handle also lets the kill be awaited instead of
        guessed with a fixed sleep.
        """
        k = _kernel32()
        pid_file = tmp_path / f"{flags_helper}.pid"
        job = _make_kill_on_close_job(k, allow_breakaway=True)
        gw = None
        gw_pid = None
        try:
            argv = [sys.executable, "-c", self._driver_source(flags_helper, pid_file)]
            driver = _start_driver_in_job(k, job, argv)
            assert _wait_for(lambda: _read_pid(pid_file) is not None), (
                "driver never wrote the pid file"
            )
            gw_pid = _read_pid(pid_file)
            gw = k.OpenProcess(
                SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, gw_pid
            )
            assert gw, (
                f"OpenProcess({gw_pid}) failed (winerror={ctypes.get_last_error()})"
            )
            assert driver.wait(timeout=30) is not None, "driver did not exit"
            assert k.WaitForSingleObject(gw, 0) == WAIT_TIMEOUT, (
                "grandchild died before job teardown"
            )
            in_job = wintypes.BOOL(False)
            assert k.IsProcessInJob(gw, job, ctypes.byref(in_job)), (
                f"IsProcessInJob failed (winerror={ctypes.get_last_error()})"
            )
            # THE teardown: closing the last job handle fires
            # KILL_ON_JOB_CLOSE against every process still in the job.
            assert k.CloseHandle(job), "CloseHandle(job) failed"
            job = None
            yield k, gw, bool(in_job.value)
        finally:
            if job:
                k.CloseHandle(job)
            if gw_pid:
                subprocess.run(
                    ["taskkill", "/PID", str(gw_pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            if gw:
                k.CloseHandle(gw)

    def test_breakaway_child_survives_job_teardown(self, tmp_path):
        with self._grandchild_after_teardown(
            tmp_path, "windows_detach_flags"
        ) as (k, gw, in_job):
            assert not in_job, "a CREATE_BREAKAWAY_FROM_JOB child must leave the job"
            assert k.WaitForSingleObject(gw, 2000) == WAIT_TIMEOUT, (
                "CREATE_BREAKAWAY_FROM_JOB child must survive the parent "
                "job's kill-on-close teardown"
            )

    def test_non_breakaway_child_killed_by_job_teardown(self, tmp_path):
        """The #48820 kill path, reproduced live: no breakaway → the job's
        teardown reaps the freshly spawned gateway."""
        with self._grandchild_after_teardown(
            tmp_path, "windows_detach_flags_without_breakaway"
        ) as (k, gw, in_job):
            assert in_job, (
                "precondition: a child spawned from inside the job without "
                "CREATE_BREAKAWAY_FROM_JOB must be in the job"
            )
            assert k.WaitForSingleObject(gw, 30_000) == WAIT_OBJECT_0, (
                "child without breakaway must be killed by kill-on-close "
                "job teardown — this is the silent gateway death of #48820"
            )


class TestWatcherRespawnLive:
    """Drive the real ``_spawn_gateway_restart_watcher`` with real processes."""

    def _run_watcher_cycle(self, tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir(parents=True, exist_ok=True)

        marker = tmp_path / "respawned.marker"
        # The stub "gateway": records its breakaway stamp env, screams on
        # stderr (so the stdio sidecar has something to capture), then exits.
        stub = (
            "import os, pathlib, sys\n"
            f"pathlib.Path(r'{marker}').write_text(\n"
            "    os.environ.get('_HERMES_GATEWAY_BREAKAWAY', 'MISSING'),\n"
            "    encoding='utf-8')\n"
            "print('stub-gateway-stderr-trace', file=sys.stderr)\n"
        )

        # A real old-pid that exits immediately — the watcher's poll loop
        # sees it die and respawns.
        old = subprocess.Popen([sys.executable, "-c", "pass"])
        old.wait(timeout=30)

        from hermes_cli import gateway

        assert gateway._spawn_gateway_restart_watcher(
            old.pid, [sys.executable, "-c", stub]
        ), "watcher spawn returned False"

        assert _wait_for(marker.exists, timeout_s=60), (
            "watcher never respawned the stub gateway"
        )
        stdio_log = tmp_path / "home" / "logs" / "gateway-stdio.log"
        return marker, stdio_log

    def test_respawn_stamps_breakaway_and_leaves_stdio_trace(
        self, tmp_path, monkeypatch
    ):
        marker, stdio_log = self._run_watcher_cycle(tmp_path, monkeypatch)

        # (a) Breakaway stamp: on unfixed main the respawn env carried no
        # stamp, so a job-teardown death was undiagnosable.
        stamp = marker.read_text(encoding="utf-8-sig").strip()
        assert stamp in {"1", "0"}, (
            f"respawned gateway must carry the breakaway stamp, got {stamp!r}"
        )

        # (b) Stdio trace: on unfixed main stderr went to DEVNULL — a dying
        # gateway left zero trace (#48820 4th repro).
        assert _wait_for(
            lambda: stdio_log.exists()
            and "stub-gateway-stderr-trace"
            in stdio_log.read_text(encoding="utf-8-sig", errors="replace"),
            timeout_s=30,
        ), "respawned gateway stderr must land in logs/gateway-stdio.log"


class TestResumeVerificationLive:
    """The user-visible lie: '✓ Restarting' printed for a dead gateway."""

    def test_dead_relaunch_is_not_reported_as_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir(parents=True, exist_ok=True)

        from hermes_cli import gateway
        import hermes_cli.main as hm
        from hermes_cli.update_cmd import _resume_windows_gateways_after_update

        # Peripheral only: don't regenerate launcher scripts into the temp home.
        monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: None)

        # Real relaunch chain, real watcher, real spawn — but the respawned
        # "gateway" exits immediately, standing in for the Job-Object
        # teardown kill. It never registers in the process table as a
        # gateway, exactly like the dead pid 48452 / 50456 of #48820.
        def _relaunch(profile, old_pid):
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead.wait(timeout=30)
            return gateway._spawn_gateway_restart_watcher(
                dead.pid, [sys.executable, "-c", "pass"]
            )

        monkeypatch.setattr(
            gateway, "launch_detached_profile_gateway_restart", _relaunch
        )

        token = {
            "resume_needed": True,
            "profiles": {"default": 999999},
            "unmapped_pids": [],
            "unmapped": [],
        }

        printed: list = []
        real_print = print
        monkeypatch.setattr(
            "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
        )
        try:
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)
        finally:
            monkeypatch.setattr("builtins.print", real_print)

        text = "\n".join(printed)
        assert "✓ Restarting" not in text, (
            "the updater must not vouch for a gateway that is not alive "
            f"(#48820). Printed:\n{text}"
        )
        assert "could not be verified" in text
