"""A state.db lock hold by a cron script names the script, not the bootstrap's first 80 characters.

``cron/scheduler_script.py`` starts a job's ``.py`` script on a managed install as
``python -c <bootstrap> <repo> <script>``. ``describe_holder_pid`` cut that command line to 80
characters, which is the bootstrap's own imports
(``python -c import importlib.machinery, os, sys, types repo, script = sys.argv[1],``), so every
``write lock unavailable`` line named a holder nobody could act on.
"""
import subprocess
import sys
import time

import pytest

from cron.scheduler_script import _POSIX_SCRIPT_BOOTSTRAP
from hermes_state_holders import describe_holder_pid

pytestmark = pytest.mark.platforms("linux", "macos")


def _describe_live(argv):
    """``describe_holder_pid`` for a real live process started with *argv*."""
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            line = describe_holder_pid(proc.pid)
            if "command line unavailable" not in line:
                return line
            time.sleep(0.05)
        pytest.fail(f"could not read the command line of pid {proc.pid}")
    finally:
        proc.kill()
        proc.wait()


def test_a_cron_script_hold_is_named_by_its_script(tmp_path):
    """The exact launcher the scheduler uses: the script path, and what the job passed to it."""
    script = tmp_path / "nightly-report.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    line = _describe_live([sys.executable, "-c", _POSIX_SCRIPT_BOOTSTRAP, str(tmp_path), str(script)])
    assert str(script) in line, line
    assert "cron script" in line, line
    assert "importlib.machinery" not in line, f"the bootstrap text is not a holder name: {line}"


def test_arguments_after_the_script_are_kept(tmp_path):
    """A job that runs its script with arguments is still identifiable by them."""
    script = tmp_path / "sync-feeds.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    line = _describe_live(
        [sys.executable, "-c", _POSIX_SCRIPT_BOOTSTRAP, str(tmp_path), str(script), "--dry-run"])
    assert str(script) in line, line
    assert "--dry-run" in line, line


def test_an_ordinary_python_program_is_not_called_a_cron_script():
    """A ``-c`` program that does not read argv[1] and argv[2] is not the cron launcher."""
    line = _describe_live([sys.executable, "-c", "import time; time.sleep(30)"])
    assert "cron script" not in line, line
    assert line.startswith("PID "), line
