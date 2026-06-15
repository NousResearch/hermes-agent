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
from hermes_constants import get_hermes_home

# Repo root (parent of the ``cron`` package) so the child process can import it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(jobs.__file__)))


@pytest.mark.skipif(jobs.fcntl is None, reason="POSIX fcntl/flock required")
def test_jobs_lock_excludes_another_process(tmp_path):
    ready = tmp_path / "child_holds_lock"
    release = tmp_path / "child_may_release"
    holder = tmp_path / "holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            import sys, time, pathlib
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import jobs

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

    child = subprocess.Popen([sys.executable, str(holder)])
    try:
        # Wait until the child is inside the critical section.
        for _ in range(1000):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists(), "child never acquired _jobs_lock()"

        # While the child holds it, a non-blocking acquire of the SAME lock file
        # from this process must fail. A threading.Lock could never block here.
        # Resolve the lock path at runtime (not jobs._JOBS_LOCK_FILE, which is
        # bound at import time) so it matches the child even when the test suite
        # redirects HERMES_HOME to a per-test tempdir.
        lock_file = get_hermes_home() / "cron" / ".jobs.lock"
        fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(OSError):
                jobs.fcntl.flock(fd, jobs.fcntl.LOCK_EX | jobs.fcntl.LOCK_NB)
        finally:
            os.close(fd)
    finally:
        release.write_text("1")
        child.wait(timeout=15)

    # Once the child has released, the lock is freely acquirable again.
    with jobs._jobs_lock():
        pass
