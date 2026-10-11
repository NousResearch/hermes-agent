"""Recovery alerts compose with progress-based liveness and worker failure bookkeeping."""

from datetime import timedelta

import pytest

from cron import executions, incidents, scheduler
from cron.jobs import create_job, get_job, use_cron_store
from cron.scheduler_provider import InProcessCronScheduler


@pytest.fixture
def recovery_env(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "_last_dead_owner_reap_at", {})
    monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: 7200.0)
    monkeypatch.setattr(executions, "_owner_is_live", lambda pid, _started: pid != 99999)
    deliveries = []

    def deliver(job, content, **kwargs):
        assert kwargs["for_failure"] is True
        targets = scheduler._resolve_delivery_targets(job, for_failure=True)
        assert targets[0]["chat_id"] == "D0ALERTS"
        deliveries.append((job["id"], content, kwargs))

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    with use_cron_store(home):
        yield deliveries


def seed_attempt(*, live, progress_age=None, job=None):
    job = job or create_job(
        prompt="probe", schedule="every 1h", deliver="slack:D0MAIN",
        failure_deliver="slack:D0ALERTS",
    )
    record = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(record["id"])
    now = executions._hermes_now()
    progress = None if progress_age is None else (now - timedelta(seconds=progress_age)).isoformat()
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='external-worker', pid=?, claimed_at=?, "
            "progress_at=? WHERE id=?",
            (88888 if live else 99999, (now - timedelta(hours=3)).isoformat(), progress, record["id"]),
        )
    return record["id"]


@pytest.mark.parametrize("entry", ["provider", "tick", "manual"])
def test_recovery_keeps_active_progress_and_separates_wedged_from_dead(
    recovery_env, monkeypatch, entry,
):
    active = seed_attempt(live=True, progress_age=120)
    wedged = seed_attempt(live=True, progress_age=7300)
    dead = seed_attempt(live=False, progress_age=120)
    adapters, loop = object(), object()

    def recover():
        if entry == "provider":
            count = InProcessCronScheduler().recover_interrupted(adapters=adapters, loop=loop)
            assert type(count) is int
            return count
        if entry == "tick":
            monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [])
            monkeypatch.setattr("tools.mcp_tool_lifecycle._kill_orphaned_mcp_children", lambda: None)
            return scheduler.tick(verbose=False, adapters=adapters, loop=loop)
        from tools.cronjob_tools import _reap_stale_executions

        return _reap_stale_executions("probe")

    recover()
    assert executions.get_execution(active)["status"] == "running"
    assert executions.get_execution(active)["delivery_outcome"] is None
    assert executions.get_execution(wedged)["error"] == executions._OWNER_WEDGED_REASON
    assert executions.get_execution(wedged)["delivery_outcome"] == "incident_only_wedged"
    assert executions.get_execution(dead)["error"] == executions._OWNER_GONE_REASON
    assert executions.get_execution(dead)["delivery_outcome"] == "delivered"
    assert len(recovery_env) == 1
    dead_job = executions.get_execution(dead)["job_id"]
    assert recovery_env[0][0] == dead_job
    if entry != "manual":
        assert recovery_env[0][2]["adapters"] is adapters
        assert recovery_env[0][2]["loop"] is loop
    by_job = {row["job_id"]: row for row in incidents.list_incidents()}
    assert len(by_job) == 2
    assert by_job[dead_job]["state"] == "alerted"
    assert by_job[executions.get_execution(wedged)["job_id"]]["state"] == "detected"

    # A later stale progress stamp now releases the active run, without a death page.
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET progress_at=? WHERE id=?",
            ((executions._hermes_now() - timedelta(seconds=7300)).isoformat(), active),
        )
    monkeypatch.setattr(scheduler, "_last_dead_owner_reap_at", {})
    recover()
    assert executions.get_execution(active)["delivery_outcome"] == "incident_only_wedged"
    assert len(recovery_env) == 1
    assert len(incidents.list_incidents()) == 3


@pytest.mark.parametrize("sweep_first", [False, True])
@pytest.mark.parametrize("same_job", [False, True])
def test_worker_fallback_recovers_other_jobs_and_notifies_current_fire_once(
    recovery_env, monkeypatch, sweep_first, same_job,
):
    from cron.scheduler_worker_failure import record_unknown_worker_outcome

    claimed = InProcessCronScheduler().claim_fire(
        create_job(
            prompt="probe", schedule="every 1h", deliver="slack:D0MAIN",
            failure_deliver="slack:D0ALERTS",
        )["id"], manual=True,
    )
    assert claimed is not None
    current = claimed["execution_id"]
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='dead-worker', pid=99999 WHERE id=?", (current,),
        )
    other = seed_attempt(live=False, job=claimed if same_job else None)
    active = seed_attempt(live=True, progress_age=120)
    wedged = seed_attempt(live=True, progress_age=7300)
    monkeypatch.setattr(scheduler, "terminalize_dead_owner", lambda *a, **kw: False)
    # Attempt dedup must work even when repeat-alert cooldown is disabled.
    monkeypatch.setattr(scheduler, "_failure_repeat_alert_hours", lambda: 0)
    if sweep_first:
        assert scheduler._recover_interrupted_executions_with_alerts() == 3

    class ExitedWorker:
        def wait(self, timeout):
            return 9

    with pytest.raises(RuntimeError, match="exited with status 9") as failure:
        scheduler._wait_for_external_cron_worker_body(ExitedWorker(), execution_id=current)
    assert executions.get_execution(current)["status"] == "unknown"
    assert executions.get_execution(other)["delivery_outcome"] == "delivered"
    assert executions.get_execution(active)["status"] == "running"
    assert executions.get_execution(wedged)["delivery_outcome"] == "incident_only_wedged"
    expected_before_waiter = [executions.get_execution(other)["job_id"]]
    if sweep_first:
        expected_before_waiter.append(claimed["id"])
    assert sorted(job_id for job_id, *_ in recovery_env) == sorted(expected_before_waiter)

    # The established post-handoff lane owns the current fire's notification and claim release.
    assert record_unknown_worker_outcome(claimed, error=str(failure.value))
    assert get_job(claimed["id"])["fire_claim"] is None
    assert sorted(job_id for job_id, *_ in recovery_env) == sorted([
        claimed["id"], executions.get_execution(other)["job_id"],
    ])
    assert len(incidents.list_incidents()) == (2 if same_job and sweep_first else 3)
    assert record_unknown_worker_outcome(claimed, error=str(failure.value))
    assert scheduler._recover_interrupted_executions_with_alerts() == 0
    assert len(recovery_env) == 2
    assert "exited with status 9" in get_job(claimed["id"])["last_error"]


@pytest.mark.parametrize("notify_first", [False, True])
def test_wedged_worker_bookkeeping_does_not_send_a_death_notice(recovery_env, notify_first):
    from cron.scheduler_worker_failure import record_unknown_worker_outcome

    job = create_job(
        prompt="probe", schedule="every 1h", deliver="local", failure_deliver="slack:D0ALERTS",
    )
    claimed = InProcessCronScheduler().claim_fire(job["id"], manual=True)
    assert claimed is not None
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='wedged-worker', pid=88888, claimed_at=? WHERE id=?",
            ((executions._hermes_now() - timedelta(hours=3)).isoformat(), claimed["execution_id"]),
        )
    recovered = executions.recover_interrupted_executions()
    assert recovered == 1
    if notify_first:
        scheduler._alert_reclaimed_executions(recovered)
    assert record_unknown_worker_outcome(claimed)
    scheduler._alert_reclaimed_executions(recovered)
    assert get_job(job["id"])["fire_claim"] is None
    assert "treated as wedged" in get_job(job["id"])["last_error"]
    assert executions.get_execution(claimed["execution_id"])["delivery_outcome"] == "incident_only_wedged"
    assert len(incidents.list_incidents()) == 1
    assert recovery_env == []


def test_worker_notifies_before_recovery_callback_without_suppressing_another_attempt(
    recovery_env, monkeypatch,
):
    from cron.scheduler_worker_failure import record_unknown_worker_outcome

    job = create_job(
        prompt="probe", schedule="every 1h", deliver="local", failure_deliver="slack:D0ALERTS",
    )
    old = seed_attempt(live=False, job=job)
    claimed = InProcessCronScheduler().claim_fire(job["id"], manual=True)
    assert claimed is not None
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='dead-worker', pid=99999 WHERE id=?",
            (claimed["execution_id"],),
        )
    monkeypatch.setattr(scheduler, "_failure_repeat_alert_hours", lambda: 0)
    recovered = executions.recover_interrupted_executions()
    assert recovered == 2
    assert record_unknown_worker_outcome(claimed, error="observed exit 9")
    assert len(recovery_env) == 1
    assert scheduler._alert_reclaimed_executions(recovered) == 2
    assert len(recovery_env) == 2
    assert executions.get_execution(old)["delivery_outcome"] == "delivered"
    assert executions.get_execution(claimed["execution_id"])["delivery_outcome"] == "delivered"


def test_recovery_delivery_uses_owning_profile_scope(recovery_env, monkeypatch):
    from cron.scheduler_provider import _profile_cron_scope

    home = scheduler._get_hermes_home()
    (home / ".env").write_text("SLACK_HOME_CHANNEL=D0ALERTS\n", encoding="utf-8")
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "D0LAUNCH")
    job = create_job(prompt="probe", schedule="every 1h", deliver="local", failure_deliver="slack")
    dead = seed_attempt(live=False, job=job)
    with _profile_cron_scope(home):
        assert InProcessCronScheduler().recover_interrupted() == 1
    assert executions.get_execution(dead)["delivery_outcome"] == "delivered"
    assert len(recovery_env) == 1


def test_recovered_records_preserve_integer_contract(recovery_env):
    dead = seed_attempt(live=False)
    recovered = executions.recover_interrupted_executions()
    assert isinstance(recovered, int)
    assert recovered == 1 and recovered + 2 == 3
    assert [row["id"] for row in recovered] == [dead]
    assert recovered.records[0]["status"] == "unknown"
    assert executions.recover_interrupted_executions() == 0
