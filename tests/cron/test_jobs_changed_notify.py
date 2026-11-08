"""Tests for on_jobs_changed wiring (Phase 4F.1).

After a store mutation via the consumer surfaces (model tool / CLI / REST), the
active scheduler provider's on_jobs_changed() must be invoked so an external
provider (Chronos) re-provisions/cancels. The built-in's no-op default means
the default path is unchanged.
"""

import pytest


def _fail_registration(job):
    raise RuntimeError("private callback URL and token")


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def test_notify_helper_calls_provider_on_jobs_changed(monkeypatch):
    """cron.scheduler._notify_provider_jobs_changed resolves the provider and
    calls on_jobs_changed exactly once."""
    import cron.scheduler_provider as sp
    import cron.scheduler as sched

    calls = []

    class Spy(sp.CronScheduler):
        @property
        def name(self):
            return "spy"

        def start(self, stop_event, **kw):
            pass

        def on_jobs_changed(self):
            calls.append(1)

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: Spy())
    sched._notify_provider_jobs_changed()
    assert calls == [1]




def test_create_registers_first_trigger_with_active_provider(
    temp_home, monkeypatch, make_cron_provider
):
    """A successful create is not reported until the provider sees the job."""
    import cron.scheduler_provider as sp
    import cron.scheduler as sched

    registered = []
    provider = make_cron_provider(register_job=registered.append)
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: provider)

    job = sched.create_job_with_scheduler_registration(
        prompt="echo hi",
        schedule="every 5m",
        name="w",
    )

    assert registered == [job]


def test_create_failure_preserves_job_and_hides_provider_details(
    temp_home, monkeypatch, make_cron_provider
):
    """Registration failure is explicit without losing the durable local job."""
    from cron import jobs
    import cron.scheduler_provider as sp
    import cron.scheduler as sched

    provider = make_cron_provider(register_job=_fail_registration, name="failing")
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: provider)

    with pytest.raises(sched.CronSchedulerRegistrationError) as exc_info:
        sched.create_job_with_scheduler_registration(
            prompt="echo hi",
            schedule="every 5m",
            name="w",
        )

    error = exc_info.value
    assert jobs.get_job(error.job["id"]) == error.job
    assert "private callback URL and token" not in str(error)
    # Human-facing variant hides the exception class name and names the job.
    assert "RuntimeError" not in error.user_message()
    assert "'w'" in error.user_message()


def test_tool_create_registers_provider_before_reporting_success(
    temp_home, monkeypatch, make_cron_provider
):
    """The model-tool success response includes a provider-registered job."""
    import cron.scheduler_provider as sp

    registered = []
    provider = make_cron_provider(register_job=registered.append, name="recording")
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: provider)

    from tools.cronjob_tools import cronjob
    import json

    out = json.loads(
        cronjob(action="create", prompt="echo hi", schedule="every 5m", name="w")
    )

    assert out["success"] is True
    assert [job["id"] for job in registered] == [out["job_id"]]


def test_tool_create_reports_partial_registration_failure(
    temp_home, monkeypatch, make_cron_provider
):
    """The model tool must not claim a remotely unregistered job succeeded."""
    import cron.scheduler_provider as sp

    provider = make_cron_provider(register_job=_fail_registration, name="failing")
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: provider)

    from tools.cronjob_tools import cronjob
    import json

    out = json.loads(cronjob(action="create", prompt="echo hi", schedule="every 5m", name="w"))
    assert out["success"] is False
    assert out["job_saved"] is True
    assert out["scheduler_registered"] is False
    assert out["retry_create"] is False
    assert out["job_id"]
    assert "private callback URL and token" not in out["error"]
    # The tool surface gets the human-facing variant: no exception class name either
    # (#150973 — the model must not be told to re-create a job that exists).
    assert "RuntimeError" not in out["error"]
    assert "Saved cron job 'w'" in out["error"]
    assert "don't re-create it" in out["error"]


def test_cli_create_reports_saved_job_on_scheduler_registration_failure(
    temp_home, monkeypatch, make_cron_provider, capsys
):
    """A stored job must not appear absent when its external trigger fails (#150973).

    The CLI headline used to be a red "Failed to create job", which contradicts the body
    ("the job is kept — don't re-create it") and makes a wrapper script treat a successful
    creation as a failure.
    """
    from types import SimpleNamespace

    import cron.scheduler_provider as sp
    from cron.jobs import list_jobs
    from hermes_cli.cron import cron_create

    provider = make_cron_provider(register_job=_fail_registration, name="chronos")
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: provider)

    rc = cron_create(SimpleNamespace(
        schedule="every 5m", prompt="echo hi", name="w", deliver="local",
    ))
    output = capsys.readouterr().out
    jobs = list_jobs()

    assert len(jobs) == 1
    assert jobs[0]["name"] == "w"
    assert rc == 1  # The external scheduler is still not registered.
    assert "Saved cron job 'w'" in output
    assert "don't re-create it" in output
    assert f"Job ID: {jobs[0]['id']}" in output
    assert "Failed to create job" not in output
    assert "RuntimeError" not in output
    assert "private callback URL and token" not in output
