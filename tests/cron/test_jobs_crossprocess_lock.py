"""Regression test for the jobs.json cross-process lock.

Background: ``hermes cron pause`` runs in its own process (CLI → cronjob tool →
``pause_job`` → ``update_job`` → ``save_jobs``), entirely separate from the
gateway process that also writes ``jobs.json`` (``mark_job_run`` /
``advance_next_run`` / due-fast-forward). The module's ``threading.Lock`` only
serializes writers *inside one process*, so a CLI pause issued while the gateway
was live could be silently lost to a concurrent gateway write — the job kept
firing even though the CLI reported "Paused".

``_jobs_lock()`` closes that gap with a short-held cross-process advisory file
lock. This test proves the lock actually excludes a *separate process*, which an
in-process ``threading.Lock`` cannot do.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from cron import jobs


@pytest.mark.skipif(jobs.fcntl is None, reason="POSIX fcntl/flock required")
def test_jobs_lock_excludes_another_process(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    output_dir = cron_dir / "output"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", output_dir)

    ready = tmp_path / "child_holds_lock"
    release = tmp_path / "child_may_release"
    blocker_started = tmp_path / "blocker_started"
    blocker_acquired = tmp_path / "blocker_acquired"
    holder = tmp_path / "holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            import sys, time, pathlib
            from cron import jobs

            jobs.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
            jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"

            with jobs._jobs_lock():
                pathlib.Path({str(ready)!r}).write_text("1")
                # Hold the lock until the parent signals (bounded so a wedged
                # test can never hang CI).
                for _ in range(1000):
                    if pathlib.Path({str(release)!r}).exists():
                        break
                    time.sleep(0.01)
            """
        )
    )

    blocker = tmp_path / "blocker.py"
    blocker.write_text(
        textwrap.dedent(
            f"""
            import sys, pathlib
            from cron import jobs

            jobs.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
            jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"

            pathlib.Path({str(blocker_started)!r}).write_text("1")
            with jobs._jobs_lock():
                pathlib.Path({str(blocker_acquired)!r}).write_text("1")
            """
        )
    )

    child = subprocess.Popen([sys.executable, str(holder)])
    blocker_child = None
    try:
        # Wait until the child is inside the critical section.
        for _ in range(1000):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists(), "child never acquired _jobs_lock()"

        # While the child holds it, a non-blocking acquire of the SAME lock file
        # from this process must fail. A threading.Lock could never block here.
        lock_file = jobs._jobs_lock_file()
        fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(OSError):
                jobs.fcntl.flock(fd, jobs.fcntl.LOCK_EX | jobs.fcntl.LOCK_NB)
        finally:
            os.close(fd)

        # A second _jobs_lock() caller in another process should block until the
        # holder releases, rather than falling through with only a process-local
        # threading lock.
        blocker_child = subprocess.Popen([sys.executable, str(blocker)])
        for _ in range(1000):
            if blocker_started.exists():
                break
            time.sleep(0.01)
        assert blocker_started.exists(), "blocker process never started"
        time.sleep(0.05)
        assert not blocker_acquired.exists(), "second process entered _jobs_lock() while held"
    finally:
        release.write_text("1")
        child.wait(timeout=15)
        if blocker_child is not None:
            blocker_child.wait(timeout=15)

    assert blocker_acquired.exists(), "second process did not acquire _jobs_lock() after release"

    # Once the child has released, the lock is freely acquirable again.
    with jobs._jobs_lock():
        pass
