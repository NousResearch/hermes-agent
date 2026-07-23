"""Cron script jobs must reap the whole process GROUP on timeout.

`subprocess.run(timeout=...)` SIGKILLs only the direct child, so a backgrounded
grandchild is orphaned and a child wedged in D-state hangs the reap. `_run_job_script`
now runs the script in its own session/process group and, on TimeoutExpired,
`os.killpg`s the whole group. This is the regression test for that behavior.
"""
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

import cron.scheduler as sched
from cron.scheduler import _run_job_script


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="process-group reaping is POSIX-only; Windows uses proc.kill()",
)
def test_timeout_kills_whole_process_group(tmp_path, monkeypatch):
    # Point the scripts dir at a temp HERMES_HOME and drop a valid .sh in it.
    monkeypatch.setattr(sched, "_get_hermes_home", lambda: tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "reap.sh").write_text("#!/bin/bash\nsleep 300 &\nsleep 300\n")

    fake = MagicMock()
    fake.pid = 4242
    fake.returncode = -9
    # First communicate() (with the job timeout) wedges; the post-kill reap returns.
    fake.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="reap.sh", timeout=1),
        ("", ""),
    ]

    with patch("cron.scheduler.subprocess.Popen", return_value=fake) as popen, \
            patch("cron.scheduler.os.getpgid", return_value=4242) as getpgid, \
            patch("cron.scheduler.os.killpg") as killpg:
        ok, msg = _run_job_script("reap.sh")

    # Child is launched in its own session so the group is killable...
    assert popen.call_args.kwargs.get("start_new_session") is True
    # ...and the WHOLE group is killed on timeout, not just the direct child.
    getpgid.assert_called_once_with(4242)
    killpg.assert_called_once()
    assert killpg.call_args.args[0] == 4242
    # The timeout is still reported back to the caller.
    assert ok is False
    assert "timed out" in msg.lower()
