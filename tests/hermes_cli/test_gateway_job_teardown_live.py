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

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.windows_only,
    pytest.mark.skipif(sys.platform != "win32", reason="native Windows only"),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
JobObjectExtendedLimitInformation = 9
PROCESS_ALL_ACCESS = 0x001FFFFF


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


def _make_kill_on_close_job(allow_breakaway: bool) -> int:
    kernel32 = ctypes.windll.kernel32
    job = kernel32.CreateJobObjectW(None, None)
    assert job, "CreateJobObjectW failed"
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if allow_breakaway:
        flags |= JOB_OBJECT_LIMIT_BREAKAWAY_OK
    info.BasicLimitInformation.LimitFlags = flags
    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    assert ok, "SetInformationJobObject failed"
    return job


def _assign_to_job(job: int, proc: subprocess.Popen) -> None:
    kernel32 = ctypes.windll.kernel32
    ok = kernel32.AssignProcessToJobObject(job, int(proc._handle))
    assert ok, f"AssignProcessToJobObject failed (winerror={ctypes.GetLastError()})"


def _pid_alive(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(h)


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

    def _driver_source(self, flags_helper: str, pid_file: str) -> str:
        # The driver runs INSIDE the job and spawns a grandchild "gateway"
        # with the flag bundle under test, then exits — mirroring the
        # updater/watcher exiting while its job tears down.
        return (
            "import subprocess, sys, pathlib\n"
            "sys.path.insert(0, r'%s')\n"
            "from hermes_cli._subprocess_compat import (\n"
            "    windows_detach_flags, windows_detach_flags_without_breakaway)\n"
            "flags = %s()\n"
            "p = subprocess.Popen([sys.executable, '-c', %r],\n"
            "    creationflags=flags, stdin=subprocess.DEVNULL,\n"
            "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "pathlib.Path(r'%s').write_text(str(p.pid), encoding='utf-8')\n"
        ) % (str(_REPO_ROOT), flags_helper, _SLEEPER, pid_file)

    def _run_in_job(self, tmp_path: Path, flags_helper: str) -> int:
        pid_file = tmp_path / f"{flags_helper}.pid"
        job = _make_kill_on_close_job(allow_breakaway=True)
        kernel32 = ctypes.windll.kernel32
        try:
            driver = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    # Handshake: wait until the test has assigned us to the
                    # job before spawning the grandchild.
                    "import pathlib, sys, time\n"
                    f"go = pathlib.Path(r'{tmp_path / 'go.marker'}')\n"
                    "deadline = time.monotonic() + 30\n"
                    "while not go.exists():\n"
                    "    assert time.monotonic() < deadline, 'no go marker'\n"
                    "    time.sleep(0.1)\n"
                    + self._driver_source(flags_helper, str(pid_file)),
                ],
                cwd=str(_REPO_ROOT),
            )
            _assign_to_job(job, driver)
            (tmp_path / "go.marker").write_text("go", encoding="utf-8")
            assert _wait_for(pid_file.exists), "driver never wrote the pid file"
            gw_pid = int(pid_file.read_text(encoding="utf-8"))
            assert _wait_for(lambda: driver.poll() is not None), (
                "driver did not exit"
            )
            assert _pid_alive(gw_pid), "grandchild died before job teardown"
            # THE teardown: closing the last job handle fires
            # KILL_ON_JOB_CLOSE against every process still in the job.
            kernel32.CloseHandle(job)
            job = None
            time.sleep(2.0)
            return gw_pid
        finally:
            (tmp_path / "go.marker").unlink(missing_ok=True)
            if job:
                kernel32.CloseHandle(job)

    def test_breakaway_child_survives_job_teardown(self, tmp_path):
        pid = self._run_in_job(tmp_path, "windows_detach_flags")
        try:
            assert _pid_alive(pid), (
                "CREATE_BREAKAWAY_FROM_JOB child must survive the parent "
                "job's kill-on-close teardown"
            )
        finally:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True
            )

    def test_non_breakaway_child_killed_by_job_teardown(self, tmp_path):
        """The #48820 kill path, reproduced live: no breakaway → the job's
        teardown reaps the freshly spawned gateway."""
        pid = self._run_in_job(tmp_path, "windows_detach_flags_without_breakaway")
        try:
            assert not _pid_alive(pid), (
                "child without breakaway must be killed by kill-on-close "
                "job teardown — this is the silent gateway death of #48820"
            )
        finally:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True
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

        import hermes_cli.gateway as gateway

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
        stamp = marker.read_text(encoding="utf-8").strip()
        assert stamp in {"1", "0"}, (
            f"respawned gateway must carry the breakaway stamp, got {stamp!r}"
        )

        # (b) Stdio trace: on unfixed main stderr went to DEVNULL — a dying
        # gateway left zero trace (#48820 4th repro).
        assert _wait_for(
            lambda: stdio_log.exists()
            and "stub-gateway-stderr-trace"
            in stdio_log.read_text(encoding="utf-8", errors="replace"),
            timeout_s=30,
        ), "respawned gateway stderr must land in logs/gateway-stdio.log"


class TestResumeVerificationLive:
    """The user-visible lie: '✓ Restarting' printed for a dead gateway."""

    def test_dead_relaunch_is_not_reported_as_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir(parents=True, exist_ok=True)

        import hermes_cli.gateway as gateway
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
