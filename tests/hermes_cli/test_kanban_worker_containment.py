"""Per-attempt containment for Kanban workers — behaviour contract.

A worker's death must not leave its descendants running into the next attempt.
The measured field failure this pins: a killed worker's PowerShell descendant kept
running for 288s and overlapped an entire 255s recovery attempt, because
``start_new_session`` is ``setsid()`` — POSIX-only and a silent no-op on Windows.

Two classes of test live here:

* ``Test*`` — dependency-injected unit tests. No real job objects, no real
  processes; the Windows-only ctypes paths are driven through fakes.
* ``TestNativeWindowsLifecycle`` — real processes and real job objects, skipped
  off Windows. These are the ones that would have caught the one-shot-dispatch
  ownership defect, so they are not optional.

The live end-to-end proof is still the Kanban qualification harness
(see references/kanban-qualification-pilot.md); these tests pin the contract.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import subprocess
import sys
import time
from ctypes import wintypes

import pytest

from hermes_cli import kanban_worker_containment as kc

IS_WINDOWS = sys.platform == "win32"
REPO_ROOT = pathlib.Path(kc.__file__).resolve().parents[2]
native = pytest.mark.skipif(not IS_WINDOWS, reason="native Windows job objects required")


def _alive(pid) -> bool:
    """True while ``pid`` names a live (non-reaped) process."""
    if not pid:
        return False
    import psutil

    try:
        proc = psutil.Process(int(pid))
        return proc.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _wait_gone(pid, timeout=15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.2)
    return not _alive(pid)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated board home, matching the fixture the other kanban tests use."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


class TestCreationFlags:
    def test_windows_requests_suspended_so_containment_precedes_execution(self, monkeypatch):
        """A descendant that started before assignment could never be contained."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        assert kc.windows_job_creationflags() & kc._CREATE_SUSPENDED

    def test_posix_adds_no_flag_because_it_has_process_groups(self, monkeypatch):
        monkeypatch.setattr(kc, "_IS_WINDOWS", False)
        assert kc.windows_job_creationflags() == 0


class TestJobNaming:
    def test_name_is_derived_from_the_worker_pid_so_any_process_can_reopen_it(self):
        """Cleanup runs in a different dispatcher than the one that spawned."""
        assert kc.attempt_job_name(4242) == kc.attempt_job_name(4242)
        assert kc.attempt_job_name(4242) != kc.attempt_job_name(4243)
        assert str(4242) in kc.attempt_job_name(4242)

    def test_name_stays_inside_the_object_name_limit(self):
        assert len(kc.attempt_job_name(2**31)) <= kc._JOB_NAME_MAX


class TestTerminateWorkerJob:
    def test_reports_no_containment_when_no_job_name_is_known(self):
        """Absence must be loud: uncontained is reported, never assumed clean."""
        result = kc.terminate_worker_job(4242, None)
        assert result == {"contained": False, "tree_terminated": False, "survivors": None}

    def test_reports_no_containment_for_a_missing_pid(self):
        assert kc.terminate_worker_job(None, "hermes-kanban-attempt-1")["contained"] is False

    def test_reports_uncontained_when_the_job_no_longer_exists(self, monkeypatch):
        """A job vanishes when its last handle closes — that is not a failure."""
        monkeypatch.setattr(kc, "_k32", lambda: _FakeKernel32(job_open=False))
        result = kc.terminate_worker_job(99, "hermes-kanban-attempt-99")
        assert result["contained"] is False

    def test_ends_the_named_tree_and_reports_zero_survivors(self, monkeypatch):
        """The tree ends through the OS job, re-opened by name — not by PID."""
        fake = _FakeKernel32(job_open=True, active=0)
        monkeypatch.setattr(kc, "_k32", lambda: fake)

        result = kc.terminate_worker_job(99, "hermes-kanban-attempt-99")

        assert result == {"contained": True, "tree_terminated": True, "survivors": 0}
        assert fake.opened == ["hermes-kanban-attempt-99"]
        assert fake.terminated, "the named job must actually be terminated"
        assert fake.closed, "the reopened handle must be closed"

    def test_ignores_the_terminate_result_at_its_peril(self, monkeypatch):
        """Claiming the tree ended when TerminateJobObject failed would be a lie."""
        fake = _FakeKernel32(job_open=True, active=1, terminate_ok=False)
        monkeypatch.setattr(kc, "_k32", lambda: fake)

        result = kc.terminate_worker_job(99, "hermes-kanban-attempt-99")

        assert result["tree_terminated"] is False

    def test_reports_survivors_rather_than_assuming_emptiness(self, monkeypatch):
        """Termination is asynchronous; the count must be measured, not assumed."""
        fake = _FakeKernel32(job_open=True, active=2)
        monkeypatch.setattr(kc, "_k32", lambda: fake)

        result = kc.terminate_worker_job(99, "hermes-kanban-attempt-99")

        assert result["tree_terminated"] is False
        assert result["survivors"] == 2


class TestContainAndResume:
    def test_is_posix_inert(self, monkeypatch):
        monkeypatch.setattr(kc, "_IS_WINDOWS", False)
        assert kc.contain_and_resume(123, "job")["contained"] is False

    def test_a_failed_job_creation_never_returns_success(self, monkeypatch):
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "_k32", lambda: _FakeKernel32(create_ok=False))
        monkeypatch.setattr(kc, "_terminate_spawned_child", lambda proc, pid: None)

        with pytest.raises(kc.WorkerSpawnError):
            kc.contain_and_resume(123, "job")

    def test_an_unopenable_worker_fails_closed(self, monkeypatch):
        """Strict raise: a worker whose tree cannot be owned is never a spawn."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "_k32", lambda: _FakeKernel32(open_ok=False))
        killed = []
        monkeypatch.setattr(kc, "_terminate_spawned_child", lambda proc, pid: killed.append(pid))

        with pytest.raises(kc.WorkerSpawnError):
            kc.contain_and_resume(123, "job")

        assert killed == [123], "the stranded suspended child must be terminated"

    def test_a_refused_assignment_never_returns_success(self, monkeypatch):
        """A worker that cannot be contained must not be reported as spawned."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "_k32", lambda: _FakeKernel32(assign_ok=False))
        killed = []
        monkeypatch.setattr(kc, "_terminate_spawned_child", lambda proc, pid: killed.append(pid))

        with pytest.raises(kc.WorkerSpawnError):
            kc.contain_and_resume(123, "job")

        assert killed == [123], "the uncontainable worker must be terminated, not stranded"

    def test_a_failed_duplicate_never_returns_success(self, monkeypatch):
        """Without the duplicate the worker would not own its containment."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "_k32", lambda: _FakeKernel32(duplicate_ok=False))
        monkeypatch.setattr(kc, "_terminate_spawned_child", lambda proc, pid: None)

        with pytest.raises(kc.WorkerSpawnError):
            kc.contain_and_resume(123, "job")

    def test_a_failed_resume_never_returns_success(self, monkeypatch):
        """Returning a PID for a permanently suspended worker is the defect."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "_k32", lambda: _FakeKernel32())
        monkeypatch.setattr(kc, "resume_worker", lambda pid: False)
        killed = []
        monkeypatch.setattr(kc, "_terminate_spawned_child", lambda proc, pid: killed.append(pid))

        with pytest.raises(kc.WorkerSpawnError):
            kc.contain_and_resume(123, "job")

        assert killed == [123]

    def test_success_reports_the_recorded_identity(self, monkeypatch):
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "_k32", lambda: _FakeKernel32())
        monkeypatch.setattr(kc, "resume_worker", lambda pid: True)
        monkeypatch.setattr(kc, "process_create_time", lambda pid: 1234.5)

        result = kc.contain_and_resume(123, "job")

        assert result["contained"] is True
        assert result["job_name"] == "job"
        assert result["worker_pid"] == 123
        assert result["worker_create_time"] == 1234.5

    def test_the_failure_path_prefers_the_spawned_process_object(self, monkeypatch):
        """Before resume the child cannot have descendants: kill THAT process."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "_k32", lambda: _FakeKernel32(assign_ok=False))
        called = []

        class FakeProc:
            pid = 123

            def kill(self):
                called.append("kill")

            def wait(self, timeout=None):
                called.append("wait")

        ran = []
        monkeypatch.setattr(kc.subprocess, "run", lambda *a, **k: ran.append(a) or None)

        with pytest.raises(kc.WorkerSpawnError):
            kc.contain_and_resume(123, "job", process=FakeProc())

        assert called == ["kill", "wait"], "the exact spawned process must be terminated and reaped"
        assert ran == [], "no PID-addressed taskkill before the child has ever run"


class TestTerminatePidTree:
    def test_uses_taskkill_slash_T_slash_F_on_windows(self, monkeypatch):
        """The portable tree kill already used by agent.deadline.kill_process_tree."""
        calls = []

        class Completed:
            returncode = 0

        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc.subprocess, "run", lambda argv, **k: (calls.append(argv), Completed())[1])

        assert kc.terminate_pid_tree(555) is True
        assert calls[0][:4] == ["taskkill", "/F", "/T", "/PID"]
        assert calls[0][4] == "555"

    def test_reports_false_when_taskkill_fails(self, monkeypatch):
        class Completed:
            returncode = 1

        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc.subprocess, "run", lambda *a, **k: Completed())
        assert kc.terminate_pid_tree(555) is False

    def test_is_posix_inert(self, monkeypatch):
        """POSIX keeps its process-group path; this must not become a second mechanism."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", False)
        assert kc.terminate_pid_tree(555) is False

    def test_refuses_when_identity_is_not_proven(self, monkeypatch):
        """PID reuse: an unprovable identity must refuse, not guess."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        ran = []
        monkeypatch.setattr(kc.subprocess, "run", lambda *a, **k: ran.append(a) or None)

        assert kc.terminate_pid_tree(555, None) is False
        assert ran == [], "an unproven identity must not reach taskkill"

    def test_refuses_when_the_recorded_creation_time_differs(self, monkeypatch):
        """A recycled PID must never be aimed at an unrelated process."""
        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "process_create_time", lambda pid: 9999.0)
        ran = []
        monkeypatch.setattr(kc.subprocess, "run", lambda *a, **k: ran.append(a) or None)

        assert kc.terminate_pid_tree(555, 1234.0) is False
        assert ran == []

    def test_proceeds_when_identity_matches(self, monkeypatch):
        class Completed:
            returncode = 0

        monkeypatch.setattr(kc, "_IS_WINDOWS", True)
        monkeypatch.setattr(kc, "process_create_time", lambda pid: 1234.0)
        monkeypatch.setattr(kc.subprocess, "run", lambda *a, **k: Completed())

        assert kc.terminate_pid_tree(555, 1234.0) is True


class TestRetryIsNotPermittedOverALiveTree:
    """Behaviour: the crash path must end the dead attempt's tree and record it.

    Driven through ``detect_crashed_workers`` with its real DB path and the
    containment module's public API observed — no source inspection.
    """

    @pytest.fixture
    def crashed_board(self, kanban_home, monkeypatch):
        import hermes_cli.kanban_db as _kb
        import hermes_cli.kanban_db_connect as kbc
        import hermes_cli.kanban_db_dispatch as kbd

        # A stable fake create_time so the recorded spawn identity is provable:
        # the fallback sweep refuses an unprovable identity by design. Patch the
        # module as resolved through sys.modules — that is the object
        # ``_set_worker_pid`` imports at call time, so this survives any reload
        # another test performs and does not depend on this file's ``kc`` alias.
        import sys

        live = sys.modules["hermes_cli.kanban_worker_containment"]
        monkeypatch.setattr(live, "process_create_time", lambda pid: 1234.0)

        conn = kbc.connect()
        tid = _kb.create_task(conn, title="containment on crash", assignee="worker")
        _kb.claim_task(conn, tid)
        kbd._set_worker_pid(conn, tid, 98765)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(_kb, "_resolve_crash_grace_seconds", lambda: 0)
        yield conn, tid, kbd
        conn.close()

    def test_crash_path_ends_the_dead_attempts_tree(self, crashed_board, monkeypatch):
        conn, tid, kbd = crashed_board
        ended = []

        def fake_terminate(pid, job_name=None):
            ended.append((pid, job_name))
            return {"contained": True, "tree_terminated": True, "survivors": 0}

        monkeypatch.setattr(kc, "terminate_worker_job", fake_terminate)
        assert kbd.detect_crashed_workers(conn) == [tid]
        assert ended, "the dead worker's containment job must be ended on crash"
        assert ended[0][0] == 98765
        assert ended[0][1] == kc.attempt_job_name(98765), (
            "cleanup must address the job by its derived name, not an in-memory handle"
        )

    def test_cleanup_disposition_reaches_the_run_record(self, crashed_board, monkeypatch):
        """An operator must be able to tell the tree was swept before the retry."""
        conn, tid, kbd = crashed_board
        monkeypatch.setattr(
            kc, "terminate_worker_job",
            lambda pid, job_name=None: {"contained": True, "tree_terminated": True, "survivors": 0},
        )
        assert kbd.detect_crashed_workers(conn) == [tid]

        import hermes_cli.kanban_db as kb

        run = kb.latest_run(conn, tid)
        metadata = run.metadata if isinstance(run.metadata, dict) else {}
        cleanup = metadata.get("cleanup")
        assert cleanup is not None, "crash run metadata must carry a cleanup disposition"
        assert cleanup["tree_terminated"] is True
        assert cleanup["survivors"] == 0

    def test_an_uncontained_worker_is_swept_by_tree_walk_and_reported(self, crashed_board, monkeypatch):
        """No job held (older build / failed assignment) must not mean no cleanup."""
        conn, tid, kbd = crashed_board
        monkeypatch.setattr(
            kc, "terminate_worker_job",
            lambda pid, job_name=None: {"contained": False, "tree_terminated": False, "survivors": None},
        )
        walked = []
        monkeypatch.setattr(kc, "terminate_pid_tree", lambda pid, ident=None: (walked.append(pid), True)[1])
        monkeypatch.setattr(_kb_flags(), "_IS_WINDOWS", True, raising=False)

        assert kbd.detect_crashed_workers(conn) == [tid]
        assert 98765 in walked, "an uncontained worker's tree must still be swept"

    def test_cleanup_failure_never_blocks_the_crash_path(self, crashed_board, monkeypatch):
        """Containment is defence: its failure must not strand a dead worker."""
        conn, tid, kbd = crashed_board

        def boom(pid, job_name=None):
            raise OSError("nope")

        monkeypatch.setattr(kc, "terminate_worker_job", boom)
        assert kbd.detect_crashed_workers(conn) == [tid]

    def test_the_recorded_spawn_identity_is_persisted_with_the_pid(self, crashed_board):
        """The crash path needs ``(pid, create_time)``; the pair must be on record."""
        import hermes_cli.kanban_db_dispatch as kbd

        conn, tid, _ = crashed_board
        assert kbd._worker_spawn_identity(conn, tid, 98765) == 1234.0
        assert kbd._worker_spawn_identity(conn, tid, 4242) is None


class TestNativeWindowsLifecycle:
    """Real processes, real job objects. These are the tests that matter most.

    The one-shot regression these encode: with a dispatcher-owned handle, the
    dispatcher exiting closed the last handle and killed the worker it had just
    spawned. No faked ctypes call can observe that — only a real parent exit can.
    """

    @native
    def test_one_shot_dispatcher_exit_does_not_kill_its_worker(self, tmp_path):
        """The regression test for the ownership defect."""
        marker = tmp_path / "worker.pid"
        script = (
            "import sys,time,os;"
            "sys.path.insert(0, sys.argv[1]);"
            "from hermes_cli import kanban_worker_containment as kc;"
            "import subprocess;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],"
            "creationflags=0x08000000|kc._CREATE_SUSPENDED);"
            "kc.contain_and_resume(p.pid, kc.attempt_job_name(p.pid));"
            "open(sys.argv[2],'w').write(str(p.pid))"
        )
        parent = subprocess.run(
            [sys.executable, "-c", script, str(REPO_ROOT), str(marker)],
            capture_output=True, text=True, timeout=120,
        )
        assert parent.returncode == 0, parent.stderr
        worker_pid = int(marker.read_text().strip())
        try:
            time.sleep(2)
            assert _alive(worker_pid), (
                "the worker died when its one-shot dispatcher exited — "
                "the job handle must live in the worker, not the dispatcher"
            )
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(worker_pid)], capture_output=True)

    @native
    def test_worker_crash_reaps_a_depth_two_descendant(self, tmp_path):
        """Containment must not have been traded away for surviving handoff."""
        beat = tmp_path / "beat.txt"
        desc_script = tmp_path / "desc.py"
        desc_script.write_text(
            "import time\n"
            "for _ in range(400):\n"
            "    open(r'%s','a').write('x')\n"
            "    time.sleep(0.1)\n" % beat
        )
        desc_pid_file = tmp_path / "desc.pid"
        # Depth-1 worker: spawn a depth-2 descendant, record its pid, then idle.
        worker_script = tmp_path / "worker.py"
        worker_script.write_text(
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, r'%s'], creationflags=0x08000000)\n"
            "open(r'%s', 'w').write(str(p.pid))\n"
            "time.sleep(120)\n" % (desc_script, desc_pid_file)
        )
        # The dispatcher spawns that worker suspended, contains it, then idles so
        # the test can kill the worker while its dispatcher is still alive.
        dispatcher_script = tmp_path / "dispatcher.py"
        dispatcher_script.write_text(
            "import subprocess, sys, time\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from hermes_cli import kanban_worker_containment as kc\n"
            "w = subprocess.Popen([sys.executable, r'%s'],\n"
            "                     creationflags=0x08000000 | kc._CREATE_SUSPENDED)\n"
            "kc.contain_and_resume(w.pid, kc.attempt_job_name(w.pid))\n"
            "print(w.pid, flush=True)\n"
            "time.sleep(120)\n" % worker_script
        )

        dispatcher = subprocess.Popen(
            [sys.executable, str(dispatcher_script), str(REPO_ROOT)],
            stdout=subprocess.PIPE, text=True,
        )
        worker_pid = None
        try:
            worker_pid = int(dispatcher.stdout.readline().strip())
            for _ in range(150):
                if desc_pid_file.exists() and desc_pid_file.read_text().strip():
                    break
                time.sleep(0.1)
            assert desc_pid_file.exists() and desc_pid_file.read_text().strip(), (
                "test setup: the depth-2 descendant never started"
            )
            desc_pid = int(desc_pid_file.read_text().strip())
            assert _alive(desc_pid), "test setup: the descendant is not running"

            # Kill the WORKER the way a crash would — no /T, no graceful path.
            subprocess.run(["taskkill", "/F", "/PID", str(worker_pid)], capture_output=True)

            assert _wait_gone(desc_pid), (
                "the depth-2 descendant outlived its worker — containment is gone"
            )
        finally:
            if worker_pid:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(worker_pid)], capture_output=True)
            dispatcher.kill()

    @native
    def test_timeout_path_reopens_the_named_job_and_proves_it_empty(self, tmp_path):
        """Reclaim/timeout runs in a *different* process than the spawner."""
        marker = tmp_path / "w.pid"
        script = (
            "import sys,subprocess,time;"
            "sys.path.insert(0, sys.argv[1]);"
            "from hermes_cli import kanban_worker_containment as kc;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],"
            "creationflags=0x08000000|kc._CREATE_SUSPENDED);"
            "kc.contain_and_resume(p.pid, kc.attempt_job_name(p.pid));"
            "print(p.pid, flush=True);"
            "time.sleep(120)"
        )
        parent = subprocess.Popen([sys.executable, "-c", script, str(REPO_ROOT)],
                                  stdout=subprocess.PIPE, text=True)
        try:
            worker_pid = int(parent.stdout.readline().strip())
            time.sleep(2)
            assert _alive(worker_pid)

            # A different process reaches the attempt's tree having only the PID.
            result = kc.terminate_worker_job(worker_pid, kc.attempt_job_name(worker_pid))

            assert result["contained"] is True, "the named job must be reopenable cross-process"
            assert result["tree_terminated"] is True
            assert result["survivors"] == 0
            assert _wait_gone(worker_pid)
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(worker_pid)], capture_output=True)
            parent.kill()

    @native
    def test_normal_worker_exit_leaves_no_job_handle_behind(self, tmp_path):
        """A completed attempt must not leave a handle (or a job) lying around."""
        marker = tmp_path / "done.pid"
        script = (
            "import sys,subprocess;"
            "sys.path.insert(0, sys.argv[1]);"
            "from hermes_cli import kanban_worker_containment as kc;"
            "p=subprocess.Popen([sys.executable,'-c','pass'],"
            "creationflags=0x08000000|kc._CREATE_SUSPENDED);"
            "kc.contain_and_resume(p.pid, kc.attempt_job_name(p.pid));"
            "p.wait();"
            "print(p.pid, flush=True)"
        )
        parent = subprocess.run([sys.executable, "-c", script, str(REPO_ROOT)],
                                capture_output=True, text=True, timeout=120)
        assert parent.returncode == 0, parent.stderr
        worker_pid = int(parent.stdout.strip())

        # The worker exited and its duplicated handle closed, so the job is gone:
        # nothing is left to reopen AND nothing was leaked by the dispatcher.
        assert _wait_gone(worker_pid)
        result = kc.terminate_worker_job(worker_pid, kc.attempt_job_name(worker_pid))
        assert result["contained"] is False, "a finished attempt must not leave a live job"

    @native
    def test_pid_reuse_defense_refuses_the_wrong_process(self, tmp_path):
        """Same PID, different create_time must refuse rather than kill blindly."""
        victim = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"],
                                  creationflags=0x08000000)
        try:
            actual = kc.process_create_time(victim.pid)
            assert actual is not None

            assert kc.terminate_pid_tree(victim.pid, actual + 500.0) is False
            time.sleep(1)
            assert _alive(victim.pid), "a mismatched identity must not kill an unrelated process"

            assert kc.terminate_pid_tree(victim.pid, actual) is True
            assert _wait_gone(victim.pid), "a matching identity must still reap the tree"
        finally:
            subprocess.run(["taskkill", "/F", "/PID", str(victim.pid)], capture_output=True)

    @native
    def test_assignment_failure_leaves_no_suspended_worker(self):
        """Atomicity, end to end: a real suspended child, a real failed assign."""
        real = kc._k32()

        class _FailingAssign:
            def __getattr__(self, name):
                return getattr(real, name)

            def AssignProcessToJobObject(self, job, handle):
                return 0

        proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"],
                                creationflags=0x08000000 | kc._CREATE_SUSPENDED)
        import unittest.mock as mock

        with mock.patch.object(kc, "_k32", lambda: _FailingAssign()):
            with pytest.raises(kc.WorkerSpawnError):
                kc.contain_and_resume(proc.pid, kc.attempt_job_name(proc.pid))

        assert _wait_gone(proc.pid), "a worker that could not be contained must not survive"

    @native
    def test_handle_values_survive_a_real_round_trip(self):
        """64-bit handles must not be truncated by ctypes' default int restype."""
        k = kc._k32()
        # Every HANDLE-returning entry point must declare an explicit restype.
        for name in ("CreateJobObjectW", "OpenJobObjectW", "OpenProcess", "OpenThread",
                     "CreateToolhelp32Snapshot", "GetCurrentProcess"):
            fn = getattr(k, name)
            assert fn.restype not in (None, ctypes.c_int), (
                "%s left on ctypes' default int restype truncates a 64-bit HANDLE" % name
            )

        job = k.CreateJobObjectW(None, kc.attempt_job_name(os.getpid()))
        assert job, "CreateJobObjectW returned a null handle"
        try:
            assert int(job) > 0
            dup = ctypes.wintypes.HANDLE()
            assert k.DuplicateHandle(k.GetCurrentProcess(), job, k.GetCurrentProcess(),
                                     ctypes.byref(dup), 0, False, kc._DUPLICATE_SAME_ACCESS)
            assert int(dup.value) > 0, "DuplicateHandle truncated the handle"
            k.CloseHandle(dup)
        finally:
            k.CloseHandle(job)


class _FakeKernel32:
    """Minimal kernel32 stand-in; records what the code under test actually asked for."""

    def __init__(self, *, create_ok=True, assign_ok=True, duplicate_ok=True, open_ok=True,
                 job_open=True, active=0, terminate_ok=True):
        self.create_ok = create_ok
        self.assign_ok = assign_ok
        self.duplicate_ok = duplicate_ok
        self.open_ok = open_ok
        self.job_open = job_open
        self.active = active
        self.terminate_ok = terminate_ok
        self.opened = []
        self.terminated = []
        self.closed = []

    def GetLastError(self):
        return 0

    def CreateJobObjectW(self, attrs, name):
        return 1234 if self.create_ok else 0

    def SetInformationJobObject(self, job, cls, info, size):
        return 1

    def OpenJobObjectW(self, access, inherit, name):
        self.opened.append(name)
        return 4321 if self.job_open else 0

    def OpenProcess(self, access, inherit, pid):
        return 555 if self.open_ok else 0

    def AssignProcessToJobObject(self, job, handle):
        return 1 if self.assign_ok else 0

    def DuplicateHandle(self, src, handle, dst, out, access, inherit, options):
        if not self.duplicate_ok:
            return 0
        out._obj.value = 6789  # ctypes byref
        return 1

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1

    def GetCurrentProcess(self):
        return 999

    def TerminateJobObject(self, job, code):
        self.terminated.append(job)
        return 1 if self.terminate_ok else 0

    def QueryInformationJobObject(self, job, cls, info, size, returned):
        info._obj.ActiveProcesses = self.active  # ctypes byref
        return 1

    def ResumeThread(self, thread):
        return 1

    def CreateToolhelp32Snapshot(self, flags, pid):
        return 777

    def Thread32First(self, snap, entry):
        return 0

    def Thread32Next(self, snap, entry):
        return 0

    def OpenThread(self, access, inherit, tid):
        return 888


def _kb_flags():
    import hermes_cli.kanban_db as _kb
    return _kb
