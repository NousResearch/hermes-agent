"""Live direct runs and successor registrations must survive stale cleanup.

Narrow regressions for the in-memory ownership portion of #88688.
"""
from concurrent.futures import Future
import threading
import time
from unittest.mock import patch

import pytest

from cron import jobs, scheduler as sched
from tools import cronjob_tools as tools


def test_live_direct_run_survives_stale_sweep(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    started, release = tmp_path / "started", tmp_path / "release"
    (scripts / "live.py").write_text(
        "from pathlib import Path\nimport time\n"
        f"Path({str(started)!r}).touch()\n"
        "deadline = time.monotonic() + 30\n"
        f"while not Path({str(release)!r}).exists():\n"
        "    if time.monotonic() > deadline: raise RuntimeError('timed out')\n"
        "    time.sleep(0.02)\n"
        "print('direct-run-ok')\n"
    )
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(None, "every 15m", script=str(scripts / "live.py"),
                              no_agent=True, deliver="local")
        result = {}
        def run():
            with jobs.use_cron_store(tmp_path):
                result.update(tools._execute_job_now(job))
        thread = threading.Thread(target=run)
        thread.start()
        try:
            deadline = time.monotonic() + 15
            while not started.exists() and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert started.exists(), result
            key = sched._inflight_key(job["id"])
            with sched._running_lock:
                sched._running_since[key] -= 86400
            assert sched.sweep_stale_inflight([jobs.get_job(job["id"])]) == []
            saved = jobs.get_job(job["id"])
            assert saved is not None
            assert not tools._execute_job_now(saved)["claimed"]
        finally:
            release.touch()
            thread.join(35)
        assert not thread.is_alive()
        assert not sched.is_job_running(job["id"])
        assert result["success"], result
        saved = jobs.get_job(job["id"])
        assert saved is not None
        assert saved["last_status"] == "ok"
        assert not saved.get("fire_claim")
        outputs = (tmp_path / "cron" / "output").rglob("*.md")
        assert any("direct-run-ok" in output.read_text() for output in outputs)
        from cron.executions import latest_executions
        assert latest_executions([job["id"]])[job["id"]]["status"] == "completed"


@pytest.mark.parametrize("path", ["direct", "direct-error", "ticker", "submit-error", "setup-error"])
def test_dispatch_cleanup_preserves_successor(path):
    job = jobs.create_job("offline", "every 15m", deliver="local")
    key = sched._inflight_key(job["id"])
    successor = []

    def replace(*args, **kwargs):
        sched.release_running_job(job["id"])
        assert sched.try_register_running_job(job["id"])
        successor.append(sched._running_futures[key])
        if path.endswith("error"):
            raise RuntimeError("old dispatch failed")
        return True

    class Pool:
        def submit(self, callback):
            if path == "submit-error":
                replace()
            future = Future()
            future.set_result(callback())
            return future

    try:
        if path.startswith("direct"):
            with patch.object(sched, "run_one_job", side_effect=replace):
                tools._run_claimed_job(job)
        elif path == "setup-error":
            with patch.object(sched, "create_execution", side_effect=replace):
                sched._submit_with_guard(job, Pool(), replace)
        else:
            sched._submit_with_guard(job, Pool(), replace)
        assert sched.is_job_running(job["id"])
        assert sched._running_futures[key] is successor[0]
    finally:
        sched.release_running_job(job["id"])
