"""Regression test for the suggestions.json cross-process lock.

Background: cron/suggestions.py's docstring claims storage "mirrors
cron/jobs.py" (atomic writes, a lock), but until this fix it only used a
process-local ``threading.Lock`` — unlike cron/jobs.py's ``_jobs_lock()``,
which was hardened with a cross-process advisory flock (#60703) specifically
because the gateway process and a separate ``hermes`` CLI invocation
(``hermes_cli/suggestions_cmd.py``) can both write ``jobs.json``/
``suggestions.json`` at the same time. A CLI dismiss/accept racing a
concurrent gateway ``add_suggestion`` (e.g. from the catalog seeder or a
blueprint install) could silently lose one side's write.

Mirrors ``tests/cron/test_jobs_crossprocess_lock.py``: proves the lock
actually excludes a *separate process*, which an in-process
``threading.Lock``/``RLock`` cannot do.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from cron import suggestions


# Repo root (parent of the ``cron`` package) so the child process can import it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(suggestions.__file__)))


@pytest.mark.skipif(suggestions.fcntl is None, reason="POSIX fcntl/flock required")
def test_suggestions_lock_excludes_another_process(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(suggestions, "CRON_DIR", cron_dir)
    monkeypatch.setattr(suggestions, "SUGGESTIONS_FILE", cron_dir / "suggestions.json")
    monkeypatch.setattr(suggestions, "_SUGGESTIONS_LOCK_FILE", cron_dir / ".suggestions.lock")

    ready = tmp_path / "child_holds_lock"
    release = tmp_path / "child_may_release"
    blocker_started = tmp_path / "blocker_started"
    blocker_acquired = tmp_path / "blocker_acquired"
    holder = tmp_path / "holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            import sys, time, pathlib
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import suggestions

            suggestions.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            suggestions.SUGGESTIONS_FILE = suggestions.CRON_DIR / "suggestions.json"
            suggestions._SUGGESTIONS_LOCK_FILE = suggestions.CRON_DIR / ".suggestions.lock"

            with suggestions._suggestions_lock():
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
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import suggestions

            suggestions.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            suggestions.SUGGESTIONS_FILE = suggestions.CRON_DIR / "suggestions.json"
            suggestions._SUGGESTIONS_LOCK_FILE = suggestions.CRON_DIR / ".suggestions.lock"

            pathlib.Path({str(blocker_started)!r}).write_text("1")
            with suggestions._suggestions_lock():
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
        assert ready.exists(), "child never acquired _suggestions_lock()"

        # While the child holds it, a non-blocking acquire of the SAME lock
        # file from this process must fail. A threading.Lock/RLock could
        # never block here.
        lock_file = suggestions._SUGGESTIONS_LOCK_FILE
        fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(OSError):
                suggestions.fcntl.flock(fd, suggestions.fcntl.LOCK_EX | suggestions.fcntl.LOCK_NB)
        finally:
            os.close(fd)

        # A second _suggestions_lock() caller in another process should block
        # until the holder releases, rather than falling through with only a
        # process-local lock.
        blocker_child = subprocess.Popen([sys.executable, str(blocker)])
        for _ in range(1000):
            if blocker_started.exists():
                break
            time.sleep(0.01)
        assert blocker_started.exists(), "blocker process never started"
        time.sleep(0.05)
        assert not blocker_acquired.exists(), "second process entered _suggestions_lock() while held"
    finally:
        release.write_text("1")
        child.wait(timeout=15)
        if blocker_child is not None:
            blocker_child.wait(timeout=15)

    assert blocker_acquired.exists(), "second process did not acquire _suggestions_lock() after release"

    # Once the child has released, the lock is freely acquirable again.
    with suggestions._suggestions_lock():
        pass
