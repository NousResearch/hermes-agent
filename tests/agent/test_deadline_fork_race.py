"""Real-process deadline regression; the probe owns and reaps its whole subtree."""
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

import pytest


@pytest.mark.linux_only
def test_cron_timeout_closes_the_descendant_snapshot_fork_window(tmp_path):
    probe = Path(__file__).resolve().parents[2] / "evals" / "cron_timeout_fork_race.py"
    result = subprocess.run(
        [sys.executable, str(probe)],
        env={**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path)},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.linux_only
@pytest.mark.parametrize("initially_stopped", [False, True])
def test_refused_hard_kill_preserves_the_targets_original_run_state(monkeypatch, initially_stopped):
    import psutil
    from agent.deadline import kill_process_tree

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    process = psutil.Process(proc.pid)
    real_kill, real_killpg = os.kill, os.killpg

    def refuse_owned_kill(pid, sig):
        assert pid == proc.pid
        if sig == signal.SIGKILL:
            raise PermissionError("owned probe: hard kill refused")
        return real_kill(pid, sig)

    def refuse_owned_group(pgid, sig):
        assert pgid == proc.pid
        if sig == signal.SIGKILL:
            raise PermissionError("owned probe: hard kill refused")
        return real_killpg(pgid, sig)

    def wait_for_state(stopped):
        deadline = time.monotonic() + 5
        while (process.status() == psutil.STATUS_STOPPED) != stopped and time.monotonic() < deadline:
            time.sleep(0.005)
        assert (process.status() == psutil.STATUS_STOPPED) == stopped

    try:
        if initially_stopped:
            process.suspend()
            wait_for_state(True)
        with monkeypatch.context() as patcher:
            patcher.setattr(os, "kill", refuse_owned_kill)
            patcher.setattr(os, "killpg", refuse_owned_group)
            assert kill_process_tree(proc.pid) is False
        wait_for_state(initially_stopped)
    finally:
        proc.kill()
        proc.wait(timeout=5)
