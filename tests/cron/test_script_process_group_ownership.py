"""The cron script fallback kill must never signal a group the script does not lead.

Regression for #97296 (teardown half, cron flavor). ``_terminate_process_group`` is the
fallback when ``kill_process_tree`` cannot run, and it ``killpg``'d the script's PGID. A
script spawned without its own session shares the scheduler's group, so the fallback
SIGTERM'd the scheduler itself. Real processes in a throwaway session, as in
tests/tools/test_local_kill_process_group_ownership.py.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HARNESS = textwrap.dedent("""
    import json, os, subprocess, sys
    try:
        os.setsid()
    except PermissionError:
        pass  # already a session leader (the runner honoured start_new_session)
    if os.getpgrp() != os.getpid():
        sys.exit(3)  # never run the kill outside a throwaway group
    from cron.scheduler_script import _terminate_process_group

    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    bystander = subprocess.Popen(sleep)  # an unrelated member of the scheduler's group
    script = subprocess.Popen(sleep)  # a script that did not get its own session
    _terminate_process_group(script)
    report = {"script_stopped": script.wait(timeout=10) is not None,
              "bystander_alive": bystander.poll() is None}
    bystander.kill()
    bystander.wait()
    print(json.dumps(report))
""")


@pytest.mark.live_system_guard_bypass
def test_fallback_never_signals_a_group_the_script_does_not_lead():
    harness = subprocess.run(
        [sys.executable, "-c", _HARNESS], cwd=_REPO_ROOT, capture_output=True, text=True,
        timeout=120, start_new_session=True)
    assert harness.returncode == 0, (
        f"harness exited {harness.returncode}: the fallback signalled the scheduler's own "
        f"process group\n{harness.stderr[-2000:]}")
    assert json.loads(harness.stdout.splitlines()[-1]) == {
        "script_stopped": True, "bystander_alive": True}
