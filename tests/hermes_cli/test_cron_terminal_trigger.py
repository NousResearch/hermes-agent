"""Desktop terminal cron refusals must explain recovery without firing (#140493)."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from cron import jobs
from hermes_cli.web_routers import cron as routes


@pytest.mark.parametrize("success", [True, False])
def test_completed_oneshot_trigger_explains_rearm_without_attempt(tmp_path, monkeypatch, success):
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    job = jobs.create_job("report", "in 30m", repeat=1)
    jobs.mark_job_run(job["id"], success=success, error=None if success else "provider failed")
    before = jobs.load_jobs()
    monkeypatch.setattr(routes, "_job_profile", lambda *args: "worker")
    monkeypatch.setattr(
        routes, "_call_cron_for_profile",
        lambda profile, method, *args: getattr(jobs, method)(*args),
    )
    from cron.scheduler_provider import InProcessCronScheduler
    from cron.executions import list_executions

    fire = Mock(side_effect=lambda profile, job_id, **kwargs: bool(
        InProcessCronScheduler().claim_fire(job_id, **kwargs)
    ))
    monkeypatch.setattr(routes, "_fire_cron_job_for_profile", fire)
    attempts_before = list_executions(job_id=job["id"])

    with pytest.raises(HTTPException) as exc:
        routes._trigger_cron_job_sync(job["id"], "worker")

    assert exc.value.status_code == 409
    assert "completed" in exc.value.detail
    assert "--run-now" in exc.value.detail
    assert "-p worker" in exc.value.detail
    fire.assert_not_called()
    assert jobs.load_jobs() == before
    assert list_executions(job_id=job["id"]) == attempts_before


@pytest.mark.parametrize("state,kind,enabled,force", [
    ("scheduled", "once", True, False),
    ("paused", "once", False, True),
    ("error", "cron", True, False),
])
def test_nonterminal_trigger_retains_existing_fire_path(monkeypatch, state, kind, enabled, force):
    job = {"id": "job-1", "state": state, "schedule": {"kind": kind}, "enabled": enabled}
    monkeypatch.setattr(routes, "_job_profile", lambda *args: "worker")
    monkeypatch.setattr(routes, "_call_cron_for_profile", lambda *args: job)
    fire = Mock(return_value=True)
    monkeypatch.setattr(routes, "_fire_cron_job_for_profile", fire)
    assert routes._trigger_cron_job_sync("job-1", "worker") == job
    fire.assert_called_once_with("worker", "job-1", force=force)
    fire.return_value = False
    with pytest.raises(HTTPException, match="already running"):
        routes._trigger_cron_job_sync("job-1", "worker")
    fire.side_effect = RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        routes._trigger_cron_job_sync("job-1", "worker")


def test_completed_recurring_does_not_offer_oneshot_rearm(monkeypatch):
    job = {"id": "job-1", "state": "completed", "schedule": {"kind": "cron"}}
    monkeypatch.setattr(routes, "_job_profile", lambda *args: "worker")
    monkeypatch.setattr(routes, "_call_cron_for_profile", lambda *args: job)
    fire = Mock()
    monkeypatch.setattr(routes, "_fire_cron_job_for_profile", fire)
    with pytest.raises(HTTPException) as exc:
        routes._trigger_cron_job_sync("job-1", "worker")
    assert "--run-now" not in exc.value.detail
    assert "completed" in exc.value.detail
    fire.assert_not_called()


def test_missing_job_remains_not_found(monkeypatch):
    monkeypatch.setattr(routes, "_job_profile", lambda *args: "worker")
    monkeypatch.setattr(routes, "_call_cron_for_profile", lambda *args: None)
    fire = Mock()
    monkeypatch.setattr(routes, "_fire_cron_job_for_profile", fire)
    with pytest.raises(HTTPException) as exc:
        routes._trigger_cron_job_sync("job-1", "worker")
    assert exc.value.status_code == 404
    fire.assert_not_called()
