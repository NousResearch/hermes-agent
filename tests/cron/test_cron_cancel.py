"""Coverage for on-demand cancellation of an in-flight cron run (`hermes cron cancel`)."""

import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_cancels(monkeypatch, tmp_path):
    """Point the marker store at a scratch home so a cancel can never see a real profile."""
    from cron import cancellation

    monkeypatch.setattr(cancellation, "_cancels_dir", lambda: tmp_path / "cancels")
    yield


def _claim(job_id: str) -> dict:
    from cron.executions import create_execution, mark_execution_running

    record = create_execution(job_id, source="test")
    mark_execution_running(record["id"])
    return record


def test_marker_drives_the_runs_cancel_view():
    from cron.cancellation import CancelRequest, cancel_requested, clear_cancel, request_cancel

    request = CancelRequest("exec-a")
    assert not request.is_set()

    request_cancel("exec-a", reason="operator asked")
    assert request.is_set()
    assert cancel_requested("exec-a")
    # The recorded reason is what the run reports — not a generic "cancelled".
    assert request.reason == "operator asked"

    clear_cancel("exec-a")
    assert not request.is_set()


def test_a_marker_only_ever_reaches_its_own_execution():
    from cron.cancellation import CancelRequest

    finished, upcoming = CancelRequest("exec-old"), CancelRequest("exec-new")
    from cron.cancellation import request_cancel
    request_cancel("exec-old")

    assert finished.is_set()
    # Recurring jobs reuse the job id every fire, so a stale marker must not cancel the next run.
    assert not upcoming.is_set()


def test_cancel_run_targets_the_live_execution():
    from cron.cancellation import CancelRequest, cancel_run

    record = _claim("job-live")
    result = cancel_run("job-live", reason="wedged")

    assert result["success"] and result["cancelled"]
    assert result["execution_id"] == record["id"]
    assert CancelRequest(record["id"]).is_set()


def test_cancel_is_a_noop_when_nothing_is_in_flight():
    from cron.cancellation import cancel_run

    _claim("job-done")
    from cron.executions import finish_execution
    from cron.executions import latest_execution
    finish_execution(latest_execution("job-done")["id"], success=True)

    result = cancel_run("job-done")
    assert result["success"] and not result["cancelled"]
    assert result["execution_id"] is None


def test_operator_cancel_is_not_treated_as_claim_loss():
    from cron import scheduler
    from cron.cancellation import CancelRequest, request_cancel

    job = {"id": "job-fence", "fire_claim": {"by": "someone-else"}}
    request_cancel("exec-fenced")
    fence = scheduler._FireOwnership(job, None, None, CancelRequest("exec-fenced"))

    # The run stops...
    assert fence.cancel_event.is_set()
    assert scheduler.cancel_reason(fence.cancel_event) == "cancelled on request (hermes cron cancel)"
    # ...but a cancel is a real outcome the run records itself, not a lost fire claim.
    assert not fence.transport_cancelled()
    assert fence.claim_lost is None


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_cancelling_a_script_job_kills_it_and_names_the_reason(tmp_path, monkeypatch):
    import os
    import sys
    from pathlib import Path

    import psutil
    from cron import scheduler, scheduler_script
    from cron.cancellation import CancelRequest, request_cancel

    # Script paths resolve under the firing profile's home; point it at the scratch dir.
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(scheduler_script, "_get_script_timeout", lambda: 60)
    # Spawn through a scratch dependency environment: the store interpreter resolves under the
    # real Hermes home, which the home-I/O guard refuses. Pinned the same way
    # tests/cron/test_cron_script.py pins this seam — what is under test is the marker and the
    # kill, not interpreter selection.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda repo: Path(sys.executable))
    monkeypatch.setattr("pm.environments.selected_venv", lambda repo: venv)
    # Spawning sanitizes PATH, which resolves the real install's bin dir; the guard refuses that
    # read. The bin dir is irrelevant to the kill, and None is the documented "no install" answer.
    from tools.environments import local as local_env

    monkeypatch.setattr(local_env, "_resolve_hermes_bin_dir", lambda: None)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    ready = tmp_path / "script.pid"
    script = scripts / "blocking.py"
    script.write_text(
        "import os, time\n"
        f"open({str(ready) + '.tmp'!r}, 'w').write(str(os.getpid()))\n"
        f"os.replace({str(ready) + '.tmp'!r}, {str(ready)!r})\n"
        "time.sleep(120)\n",
        encoding="utf-8")

    cancel = CancelRequest("exec-live-script")
    results: list = []
    thread = threading.Thread(
        target=lambda: results.append(
            scheduler_script._run_job_script(str(script), workdir=str(tmp_path), cancel_event=cancel)))
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), f"script must start before it can be cancelled (results={results})"
        pid = int(ready.read_text(encoding="utf-8"))

        request_cancel("exec-live-script", reason="operator asked")
        thread.join(timeout=20)
        assert not thread.is_alive()
        assert results[0][0] is False
        assert "operator asked" in results[0][1]

        deadline = time.monotonic() + 5
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(pid), "the cancelled script must not outlive the cancel"
    finally:
        if ready.exists():
            pid = int(ready.read_text(encoding="utf-8"))
            if _alive(pid):
                os.kill(pid, 9)
        thread.join(timeout=20)


def test_a_second_request_keeps_the_reason_the_run_already_reported():
    from cron.cancellation import CancelRequest, request_cancel

    request_cancel("exec-reason", reason="operator asked")

    request_cancel("exec-reason", reason="second ask")
    request_cancel("exec-reason")

    # The docstring promises the first reason wins; the run may already have reported it.
    assert CancelRequest("exec-reason").reason == "operator asked"


def test_a_second_request_still_prunes_stale_markers(monkeypatch):
    from cron import cancellation
    from cron.cancellation import request_cancel

    pruned: list = []
    monkeypatch.setattr(cancellation, "_prune_stale_markers", lambda now=None: pruned.append(now))

    request_cancel("exec-pruned", reason="first")
    request_cancel("exec-pruned", reason="second")

    # An idempotent write must not skip the sweep that rides along with it.
    assert len(pruned) == 2


def _alive(pid: int) -> bool:
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
