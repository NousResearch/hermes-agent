"""Native-Windows lifecycle tests for the detached cron script runner.

These tests exercise REAL Win32 job-object and handle semantics; they are
skipped everywhere except native Windows (win32).

Isolation note: ``test_breakaway_denied_inside_restrictive_job`` assigns a
throwaway *child* python process (not the pytest process) to a restrictive
job object, so it is safe to run in-process. No test here assigns the
pytest process itself to a job.

Covers PR #43252:

* a detached, file-backed child survives the death of its launcher and
  completes both its side effect and its output writes (the Codex D2
  scenario that kills a pipe-backed child);
* a restrictive job produces exactly one script execution under either
  observed Windows behavior: constructor denial with a one-time fallback, or
  successful creation with silent retention in the forbidding job.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="native Windows lifecycle semantics"
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _wait_for(predicate, timeout=30.0, interval=0.2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestDetachedChildSurvivesParentDeath:
    def test_output_producing_child_survives_launcher_kill(self, tmp_path):
        """Launcher process uses _run_script_windows_detached on a child
        that writes repeatedly (more than a pipe buffer in total) and then
        records a durable side effect. We hard-kill the launcher mid-run.

        With anonymous pipes (rejected package) the child dies on its next
        write. With file-backed capture it must finish.
        """
        marker = tmp_path / "done.txt"
        child_script = tmp_path / "child.py"
        child_script.write_text(textwrap.dedent(f"""\
            import sys, time
            from pathlib import Path
            # ~1.5 MiB in chunks, over ~6s, far beyond any pipe buffer.
            for i in range(60):
                sys.stdout.write("x" * 25600)
                sys.stdout.flush()
                time.sleep(0.1)
            Path({str(marker)!r}).write_text("side effect completed")
            print("child done")
        """))

        launcher = tmp_path / "launcher.py"
        launcher.write_text(textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(Path(__file__).parent.parent.parent)!r})
            from cron.scheduler import _run_script_windows_detached
            import os
            _run_script_windows_detached(
                [sys.executable, {str(child_script)!r}],
                timeout=120,
                cwd={str(tmp_path)!r},
                env=os.environ.copy(),
            )
        """))

        proc = subprocess.Popen(
            [sys.executable, str(launcher)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)  # child is mid-write-loop now
        proc.kill()      # hard parent death: no cleanup, no drain
        proc.wait()

        assert _wait_for(marker.exists, timeout=30), (
            "Detached child did not complete after launcher death — "
            "output sink or job membership is not durable"
        )
        assert marker.read_text() == "side effect completed"

    def test_no_temp_output_files_leak_after_parent_death(self, tmp_path):
        """O_TEMPORARY files must self-delete once the surviving child
        exits; parent death must not strand unredacted output on disk."""
        temp_dir = Path(os.environ.get("TMP", os.environ.get("TEMP")))
        before = {p.name for p in temp_dir.glob("tmp*")}

        run2 = tmp_path / "run2"
        run2.mkdir()
        self.test_output_producing_child_survives_launcher_kill(run2)

        # Give the OS a moment to finalize delete-on-close.
        time.sleep(2.0)
        after = {p.name for p in temp_dir.glob("tmp*")}
        leaked = after - before
        # Filter to files still present and non-openable-exclusively is
        # noisy; assert no *persistent* growth instead.
        assert len(leaked) == 0, f"Leaked temp output files: {leaked}"


class TestBreakawayPolicy:
    def test_restrictive_job_executes_script_once(self, tmp_path):
        """Run a throwaway Python child inside a restrictive job.

        Windows has exhibited two valid behaviors for a breakaway request
        when some job in a nested chain forbids it:

        * CreateProcessW reports WinError 5, so the runner performs its single
          constructor fallback without CREATE_BREAKAWAY_FROM_JOB; or
        * creation succeeds but Windows silently retains the child in the
          forbidding job, so no fallback occurs.

        Both outcomes must execute the script exactly once. This test pins the
        public invariant rather than one OS-build-specific error mechanism.
        """
        probe = tmp_path / "probe_in_job.py"
        probe.write_text(textwrap.dedent(f"""\
            import json, subprocess, sys
            sys.path.insert(0, {str(Path(__file__).parent.parent.parent)!r})
            from cron import scheduler as sched_mod

            calls = []
            real_popen = subprocess.Popen

            def counting_popen(*args, **kwargs):
                calls.append(kwargs.get("creationflags", 0))
                return real_popen(*args, **kwargs)

            sched_mod.subprocess.Popen = counting_popen

            import os
            rc, out, err = sched_mod._run_script_windows_detached(
                [sys.executable, "-c", "print('ran once')"],
                timeout=60,
                cwd=".",
                env=os.environ.copy(),
            )
            print(json.dumps({{
                "rc": rc,
                "out": out.strip(),
                "attempts": len(calls),
                "flags": calls,
            }}))
        """))

        harness = tmp_path / "job_harness.py"
        harness.write_text(textwrap.dedent(f"""\
            import ctypes, subprocess, sys
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_uint64) for n in (
                    "ReadOperationCount", "WriteOperationCount",
                    "OtherOperationCount", "ReadTransferCount",
                    "WriteTransferCount", "OtherTransferCount")]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            JobObjectExtendedLimitInformation = 9
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
            # Deliberately omit JOB_OBJECT_LIMIT_BREAKAWAY_OK (0x0800).
            # Depending on Windows build and job nesting, the request may be
            # rejected with WinError 5 or accepted with silent retention.

            job = kernel32.CreateJobObjectW(None, None)
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            assert kernel32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info))

            CREATE_SUSPENDED = 0x00000004
            proc = subprocess.Popen(
                [sys.executable, {str(probe)!r}],
                creationflags=CREATE_SUSPENDED,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            PROCESS_ALL_ACCESS = 0x1FFFFF
            h = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, proc.pid)
            assert kernel32.AssignProcessToJobObject(job, h)
            # Resume the suspended main thread.
            import ctypes.wintypes as wt
            THREAD_SUSPEND_RESUME = 0x0002
            # Use NtResumeProcess for simplicity:
            ctypes.windll.ntdll.NtResumeProcess(h)
            out, err = proc.communicate(timeout=120)
            sys.stdout.write(out)
            sys.stderr.write(err)
            sys.exit(proc.returncode)
        """))

        result = subprocess.run(
            [sys.executable, str(harness)],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, (
            f"harness failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        import json
        data = json.loads(result.stdout.strip().splitlines()[-1])
        assert data["rc"] == 0
        assert data["out"] == "ran once"
        assert data["attempts"] in (1, 2), (
            "expected one successful full-flags construction or one "
            f"breakaway-denied fallback, got {data}"
        )
        assert len(data["flags"]) == data["attempts"]
        # Every path starts with CREATE_BREAKAWAY_FROM_JOB.
        assert data["flags"][0] & 0x01000000
        if data["attempts"] == 2:
            # A constructor denial may retry once, without breakaway.
            assert not (data["flags"][1] & 0x01000000)
