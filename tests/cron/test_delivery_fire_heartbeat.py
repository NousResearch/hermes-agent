"""Delivery must not make a live fire cancel its own completed work."""

import contextlib
import logging
import signal
import subprocess
import sys
import threading
import time

import pytest


@pytest.mark.parametrize("entry_race", [False, True])
@pytest.mark.parametrize("delivery_error", [None, "fixture transport failed"])
def test_slow_delivery_does_not_reclassify_completed_run(
    tmp_path, monkeypatch, entry_race, delivery_error,
):
    from cron import executions, jobs, scheduler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    heartbeat_at_lock = tmp_path / "heartbeat-at-lock"
    script = "print('completed artifact')\n"
    if entry_race:
        script = (
            "import time\nfrom pathlib import Path\n"
            f"ready = Path({str(heartbeat_at_lock)!r})\n"
            "deadline = time.monotonic() + 5\n"
            "while not ready.exists():\n"
            "    assert time.monotonic() < deadline, 'heartbeat never reached lock'\n"
            "    time.sleep(0.01)\n" + script
        )
    (scripts / "complete.py").write_text(script)
    job = jobs.create_job(
        prompt="", schedule="every 5m", name="slow-delivery",
        script="complete.py", no_agent=True, deliver="telegram:fixture",
    )
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    # Run the real in-process worker, not a systemd service on the test host.
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda _job: False)
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    delivering = threading.Event()
    heartbeat_done = threading.Event()
    false_cancel_logged = threading.Event()
    renewals = []
    delivered = []
    real_heartbeat = jobs.heartbeat_fire_claim
    real_fire_lock = jobs._fire_job_lock
    worker_thread = threading.current_thread()

    @contextlib.contextmanager
    def observe_fire_lock(job_id):
        # Pause the heartbeat after its first protected-owner lookup, before
        # real acquisition. Delivery can then win the actual lock race.
        if threading.current_thread() is not worker_thread:
            heartbeat_at_lock.touch()
            assert delivering.wait(5), "delivery never entered its fence"
        with real_fire_lock(job_id) as acquired:
            yield acquired

    if entry_race:
        monkeypatch.setattr(jobs, "_fire_job_lock", observe_fire_lock)

    def observe_heartbeat(*args, **kwargs):
        renewed = real_heartbeat(*args, **kwargs)
        if delivering.is_set():
            renewals.append(renewed)
            heartbeat_done.set()
        return renewed

    class LossObserver(logging.Handler):
        def emit(self, record):
            if "fire claim ownership lost; interrupting stale run" in record.getMessage():
                false_cancel_logged.set()

    def slow_transport(_job, content, **_kwargs):
        delivering.set()
        assert heartbeat_done.wait(5), "heartbeat never ran during delivery"
        if not renewals[-1]:
            assert false_cancel_logged.wait(5), "claim failure was not handled"
        # Keep the transport in flight beyond the shortened lock window even
        # when the corrected heartbeat no longer waits on that lock.
        threading.Event().wait(0.1)
        delivered.append(content)
        delivering.clear()
        return delivery_error

    handler = LossObserver()
    scheduler.logger.addHandler(handler)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", observe_heartbeat)
    # Only the outbound transport is a fixture; script, fences, heartbeat,
    # artifact and execution/job persistence all execute their production path.
    monkeypatch.setattr(scheduler, "_deliver_result", slow_transport)
    try:
        assert scheduler.run_one_job(claimed) is True
    finally:
        scheduler.logger.removeHandler(handler)
    stored = jobs.get_job(job["id"])
    execution = executions.latest_executions([job["id"]])[job["id"]]
    assert delivered == ["completed artifact"]
    assert isinstance(stored, dict)
    assert stored["last_status"] == ("delivery_failed" if delivery_error else "ok"), stored["last_error"]
    assert execution["status"] == "completed", execution["error"]
    assert stored["last_delivery_error"] == delivery_error
    assert execution["delivery_outcome"] == ("failed" if delivery_error else "delivered")
    # A later stale completion cannot reclassify an already-terminal attempt.
    assert executions.finish_execution(
        execution["id"], success=False, error="late claim loss",
    ) is None
    assert executions.get_execution(execution["id"]) == execution
    assert stored["fire_claim"] is None
    assert renewals and all(renewals)
    outputs = list((tmp_path / "cron" / "output" / job["id"]).glob("*.md"))
    assert len(outputs) == 1 and "completed artifact" in outputs[0].read_text()


@pytest.mark.linux_only
def test_real_sigterm_mid_script_still_records_interrupted(tmp_path, monkeypatch):
    """A real signal enters the worker cancellation path; no live gateway involved."""
    from cron import executions, jobs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    started = tmp_path / "script-started"
    worker_code = """
import os, signal, threading
from pathlib import Path
from cron import jobs, scheduler
home = Path(os.environ['HERMES_HOME'])
scripts = home / 'scripts'
scripts.mkdir()
started = home / 'script-started'
(scripts / 'blocking.py').write_text(
    'import os, time\\nfrom pathlib import Path\\n'
    + f'Path({str(started)!r}).write_text(str(os.getpid()))\\n'
    + 'time.sleep(30)\\nprint("must not succeed")\\n'
)
job = jobs.create_job(prompt='', schedule='every 5m', name='interrupt-fixture',
                      script='blocking.py', no_agent=True, deliver='local')
claimed = jobs.claim_job_for_fire(job['id'], return_job=True)
cancel = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: cancel.set())
scheduler._launch_external_cron_worker = lambda _job: False
assert scheduler.run_one_job(claimed, cancel_event=cancel) is True
"""
    process = subprocess.Popen(
        [sys.executable, "-c", worker_code], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while (not started.exists() or not started.read_text()) and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started.exists(), "script did not reach its running phase"

        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        stored, = jobs.load_jobs()
        execution = executions.latest_executions([stored["id"]])[stored["id"]]
        assert stored["last_status"] == "error"
        assert "Interrupted" in stored["last_error"]
        assert execution["status"] == "failed"
        assert "Interrupted" in execution["error"]
        assert stored["fire_claim"] is None
    finally:
        if process.poll() is None:
            # Let the worker cancel/reap its own script through its live handle.
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
