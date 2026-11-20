"""Recovery and the external waiter must distinguish a live wedge from observed death."""

import subprocess
import sys

import pytest

from cron import executions, incidents, jobs, scheduler


@pytest.mark.parametrize("alert_first", [True, False])
@pytest.mark.parametrize("dead_first", [True, False])
def test_live_wedged_waiter_keeps_mixed_recovery_incident_only(
    tmp_path, monkeypatch, alert_first, dead_first,
):
    _exercise_waiter(tmp_path, monkeypatch, alert_first=alert_first, dead_first=dead_first)


def test_worker_exit_after_wedged_recovery_keeps_observed_death_notice(tmp_path, monkeypatch):
    _exercise_waiter(tmp_path, monkeypatch, alert_first=True, dead_first=False, exits=True)


def _exercise_waiter(tmp_path, monkeypatch, *, alert_first, dead_first, exits=False):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: 1.0)
    monkeypatch.setattr(executions, "_stale_age_seconds", lambda *_args: 10.0)
    monkeypatch.setattr(scheduler, "load_config_readonly", lambda: {"cron": {"failure_repeat_alert_hours": 0}})
    notices, background_reaps = [], []
    monkeypatch.setattr(scheduler, "_deliver_result", lambda job, *args, **kw: notices.append(job["id"]))
    monkeypatch.setattr(
        "cron.scheduler_detached_worker.reap_terminal_worker_in_background", background_reaps.append,
    )
    with jobs.use_cron_store(tmp_path / "cron"):
        live_job = jobs.create_job(prompt="wedged", schedule="every 1h", failure_deliver="slack:alerts")
        dead_job = jobs.create_job(prompt="dead", schedule="every 1h", failure_deliver="slack:alerts")
        claimed = jobs.claim_job_for_fire(live_job["id"], manual=True, return_job=True)
        assert claimed
        current_id = executions.create_execution(live_job["id"], source="test")["id"]
        claimed["execution_id"] = current_id
        dead = executions.create_execution(dead_job["id"], source="test")
        with subprocess.Popen(
            [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline(); sys.exit(9)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        ) as worker:
            assert worker.stdout.readline().strip() == "ready"
            dead_pid = subprocess.run(
                [sys.executable, "-c", "import os; print(os.getpid())"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            with executions._transaction() as conn:
                conn.execute(
                    "UPDATE executions SET process_id='external-worker', pid=?, process_started_at=? WHERE id=?",
                    (worker.pid, executions._process_start_time(worker.pid), current_id),
                )
                conn.execute(
                    "UPDATE executions SET process_id='departed-worker', pid=?, process_started_at=NULL WHERE id=?",
                    (int(dead_pid), dead["id"]),
                )
            recovered = executions.recover_interrupted_executions()
            assert recovered == 2
            records = sorted(recovered.records, key=lambda row: (row["id"] == current_id) == dead_first)
            batch = executions.RecoveredExecutions(records)
            assert executions.get_execution(current_id)["error"] == executions._OWNER_WEDGED_REASON
            if alert_first:
                scheduler._alert_reclaimed_executions(batch)
            real_wait = worker.wait

            def launch(job):
                if exits:
                    worker.stdin.write("\n")
                    worker.stdin.flush()
                    assert real_wait(timeout=30) == 9
                else:
                    assert worker.poll() is None

                    def timeout(timeout=None):
                        raise subprocess.TimeoutExpired(worker.args, timeout)

                    monkeypatch.setattr(worker, "wait", timeout)
                return scheduler._wait_for_external_cron_worker(
                    worker, execution_id=current_id,
                )

            monkeypatch.setattr(scheduler, "_launch_external_cron_worker", launch)
            try:
                assert scheduler.run_one_job(claimed)
                if not alert_first:
                    scheduler._alert_reclaimed_executions(batch)
                assert jobs.get_job(live_job["id"])["fire_claim"] is None
                current = executions.get_execution(current_id)
                assert current["status"] == "unknown"
                if exits:
                    assert notices == [dead_job["id"], live_job["id"]]
                    assert "exited with status 9" in jobs.get_job(live_job["id"])["last_error"]
                else:
                    assert worker.poll() is None
                    assert background_reaps == [worker]
                    assert notices == [dead_job["id"]]
                    assert current["delivery_outcome"] == "incident_only_wedged"
                    wedge_incident = next(row for row in incidents.list_incidents() if row["job_id"] == live_job["id"])
                    assert wedge_incident["state"] == "detected"
            finally:
                monkeypatch.setattr(worker, "wait", real_wait)
                if worker.poll() is None:
                    worker.stdin.write("\n")
                    worker.stdin.flush()
                real_wait(timeout=30)
