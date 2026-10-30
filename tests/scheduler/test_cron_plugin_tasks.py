"""Plugin cron tasks: mutually exclusive payload kind, in-process dispatch, visible failure."""

from types import SimpleNamespace

import pytest

from scheduler import cron_api


def test_plugin_task_job_is_a_mutually_exclusive_payload_kind():
    from cron.jobs import create_job

    job = create_job(prompt="", schedule="*/5 * * * *", name="acme.probe", deliver="local",
                     plugin_task={"plugin_id": "acme", "task_name": "acme.probe"})
    assert job["plugin_task"] == {"plugin_id": "acme", "task_name": "acme.probe"}
    assert job["prompt"] == "" and job["no_agent"] is False
    with pytest.raises(ValueError, match="mutually exclusive|combined"):
        create_job(prompt="do work", schedule="*/5 * * * *",
                   plugin_task={"plugin_id": "acme", "task_name": "acme.probe"})


def test_dispatch_runs_the_registered_handler(monkeypatch):
    monkeypatch.setattr("cron.jobs.mark_job_run",
                        lambda *a, **k: (calls.append(("mark", a, k)) or True))
    monkeypatch.setattr("cron.executions.finish_execution",
                        lambda *a, **k: calls.append(("finish", a, k)))
    calls = []
    seen = []
    with cron_api._LOCK:
        cron_api._REGISTRY["job1"] = (lambda ctx: seen.append(ctx.phase), 0.5, "acme", "acme.probe")
    try:
        cron_api.run_plugin_task_job(
            {"id": "job1", "plugin_task": {"plugin_id": "acme", "task_name": "acme.probe"}}, "exec1")
    finally:
        with cron_api._LOCK:
            cron_api._REGISTRY.pop("job1", None)
    assert seen == ["post"]
    assert [entry[0] for entry in calls] == ["mark", "finish"]
    assert calls[0][1][1] is True


def test_missing_handler_fails_the_fire_visibly(monkeypatch):
    marked = []
    monkeypatch.setattr("cron.jobs.mark_job_run",
                        lambda job_id, ok, error=None, **k: marked.append((job_id, ok, error)))
    monkeypatch.setattr("cron.executions.finish_execution", lambda *a, **k: None)
    cron_api.run_plugin_task_job(
        {"id": "gone", "plugin_task": {"plugin_id": "acme", "task_name": "acme.gone"}}, "exec2")
    assert marked and marked[0][1] is False
    assert "unavailable" in marked[0][2]


class _Ctx:
    plugin_id = "acme"

    def __init__(self):
        self.released = lambda: None

    def _track(self, kind, key, release):
        self.released = release


def test_schedule_cron_task_creates_a_job_and_teardown_pauses_it(monkeypatch):
    from cron.jobs import create_job, list_jobs

    paused = []
    created = []

    def fake_create(**kwargs):
        created.append(kwargs)
        return create_job(**kwargs)

    monkeypatch.setattr("cron.scheduler.create_job_with_scheduler_registration", fake_create)
    monkeypatch.setattr("cron.jobs.pause_job", lambda job_id, reason=None: paused.append((job_id, reason)))
    ctx = _Ctx()
    handle = cron_api.schedule_cron_task(
        ctx, name="acme.probe", cron="*/5 * * * *", handler=lambda task_ctx: None)
    assert created[0]["plugin_task"] == {"plugin_id": "acme", "task_name": "acme.probe"}
    assert created[0]["prompt"] == "" and created[0]["deliver"] == "local"
    with cron_api._LOCK:
        assert handle.job_id in cron_api._REGISTRY
    ctx.released()
    assert paused == [(handle.job_id, "plugin unloaded")]
    with cron_api._LOCK:
        assert handle.job_id not in cron_api._REGISTRY

    # A reboot/reload must reuse the job, not mint a second live schedule.
    again = cron_api.schedule_cron_task(
        _Ctx(), name="acme.probe", cron="*/5 * * * *", handler=lambda task_ctx: None)
    assert again.job_id == handle.job_id
    assert len(created) == 1
    assert len([job for job in list_jobs(include_disabled=True)
                if (job.get("plugin_task") or {}).get("task_name") == "acme.probe"]) == 1
    assert again.job_id in cron_api._REGISTRY


def test_schedule_cron_task_rejects_non_cron_expressions():
    ctx = SimpleNamespace(plugin_id="acme", _track=lambda *a: None)
    with pytest.raises(ValueError, match="cron expression"):
        cron_api.schedule_cron_task(ctx, name="acme.probe", cron="every 5m",
                                    handler=lambda c: None)
    with pytest.raises(ValueError, match="prefixed"):
        cron_api.schedule_cron_task(ctx, name="probe", cron="*/5 * * * *",
                                    handler=lambda c: None)
