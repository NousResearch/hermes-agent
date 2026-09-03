"""Spend-neutral cron backoff for provider-declared reset windows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _rate_limit_error(reset: str) -> str:
    return (
        "RuntimeError: HTTP 429: Weekly/Monthly Limit Exhausted. "
        f"Your limit will reset at {reset}"
    )


def test_declared_reset_is_deferred_until_reset_plus_deterministic_jitter():
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 11, 24, tzinfo=timezone(timedelta(hours=7)))
    error = _rate_limit_error("2026-09-04 10:03:01")

    first = plan_provider_backoff({"id": "job-a"}, error, now=now)
    second = plan_provider_backoff({"id": "job-a"}, error, now=now)

    assert first == second
    assert first is not None
    reset_at = datetime.fromisoformat(first["reset_at"])
    until = datetime.fromisoformat(first["until"])
    assert reset_at == datetime(
        2026, 9, 4, 10, 3, 1, tzinfo=timezone(timedelta(hours=7))
    )
    assert timedelta(seconds=1) <= until - reset_at <= timedelta(minutes=5)
    assert first["source"] == "provider_reset"


def test_repeated_rate_limits_without_reset_hint_back_off_exponentially():
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    first = plan_provider_backoff({"id": "job-a"}, "HTTP 429", now=now)
    second = plan_provider_backoff(
        {"id": "job-a", "provider_backoff": first}, "rate limit reached", now=now
    )

    assert first is not None and second is not None
    assert first["attempt"] == 1
    assert second["attempt"] == 2
    assert datetime.fromisoformat(second["until"]) > datetime.fromisoformat(first["until"])
    assert second["source"] == "fallback"


def test_invalid_reset_hint_uses_bounded_fallback():
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    planned = plan_provider_backoff(
        {"id": "job-a"}, _rate_limit_error("not-a-timestamp"), now=now
    )

    assert planned is not None
    assert planned["source"] == "fallback"
    assert now + timedelta(minutes=15) < datetime.fromisoformat(planned["until"])
    assert datetime.fromisoformat(planned["until"]) <= now + timedelta(minutes=16)


def test_retry_after_http_date_is_respected():
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    reset = now + timedelta(hours=2)
    error = f"HTTP 429; Retry-After: {format_datetime(reset, usegmt=True)}"

    planned = plan_provider_backoff({"id": "job-a"}, error, now=now)

    assert planned is not None
    assert planned["source"] == "provider_reset"
    assert datetime.fromisoformat(planned["reset_at"]) == reset


def test_locale_reset_timestamp_is_respected():
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    error = "HTTP 429; your limit will reset at September 4, 2026 10:03 AM"

    planned = plan_provider_backoff({"id": "job-a"}, error, now=now)

    assert planned is not None
    assert planned["source"] == "provider_reset"
    assert datetime.fromisoformat(planned["reset_at"]) == datetime(
        2026, 9, 4, 10, 3, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("spelling", ["reset_at=", "reset-at: "])
def test_machine_reset_at_spellings_are_respected(spelling):
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    planned = plan_provider_backoff(
        {"id": "job-a"}, f"HTTP 429; {spelling}2026-09-04T10:03:01Z", now=now
    )

    assert planned is not None
    assert planned["source"] == "provider_reset"
    assert planned["reset_at"] == "2026-09-04T10:03:01+00:00"


def test_far_future_persisted_marker_fails_open():
    from cron.rate_limit_backoff import provider_backoff_active

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    job = {"provider_backoff": {"until": (now + timedelta(days=365)).isoformat()}}

    assert provider_backoff_active(job, now=now) is False


def test_non_rate_limit_failure_does_not_create_backoff():
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    assert plan_provider_backoff({"id": "job-a"}, "HTTP 500 upstream error", now=now) is None
    assert plan_provider_backoff(
        {"id": "job-a", "no_agent": True}, "HTTP 429 from script output", now=now
    ) is None


def test_backoff_persists_across_reload_and_suppresses_due_run(temp_home, monkeypatch):
    from cron.jobs import create_job, get_due_jobs, get_job, mark_job_run
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job("work", "every 10m", name="eng-completion")
    error = _rate_limit_error("2026-09-04 10:03:01+00:00")
    backoff = plan_provider_backoff(job, error, now=now)

    assert mark_job_run(job["id"], False, error, provider_backoff=backoff)
    persisted = get_job(job["id"])
    assert persisted["last_status"] == "error"
    assert persisted["last_error"] == error
    assert persisted["provider_backoff"] == backoff

    due_time = datetime.fromisoformat(persisted["next_run_at"]) + timedelta(seconds=1)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: due_time)
    assert get_due_jobs() == []
    assert get_job(job["id"])["provider_backoff"] == backoff


def test_expired_backoff_resumes_automatically_and_success_clears_it(
    temp_home, monkeypatch
):
    from cron.jobs import create_job, get_due_jobs, get_job, mark_job_run
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job("work", "every 10m", name="eng-completion")
    error = "HTTP 429"
    backoff = plan_provider_backoff(job, error, now=now)
    assert mark_job_run(job["id"], False, error, provider_backoff=backoff)

    after = datetime.fromisoformat(backoff["until"]) + timedelta(seconds=1)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: after)
    due = get_due_jobs()
    assert [item["id"] for item in due] == [job["id"]]
    assert due[0]["provider_backoff"]["attempt"] == 1
    persisted = get_job(job["id"])
    assert persisted is not None
    assert persisted["provider_backoff"] == backoff

    repeated = plan_provider_backoff(due[0], error, now=after)
    assert repeated is not None
    assert repeated["attempt"] == 2

    assert mark_job_run(job["id"], True)
    recovered = get_job(job["id"])
    assert recovered["enabled"] is True
    assert recovered["state"] == "scheduled"
    assert recovered["last_status"] == "ok"
    assert recovered["failure_streak"] == 0


def test_scheduler_records_backoff_without_hiding_failed_execution(monkeypatch):
    import cron.scheduler as scheduler

    now = datetime(2026, 9, 3, 11, 24, tzinfo=timezone(timedelta(hours=7)))
    error = _rate_limit_error("2026-09-04 10:03:01")
    job = {
        "id": "job-a",
        "name": "eng-completion",
        "prompt": "work",
        "deliver": "local",
        "execution_id": "execution-a",
    }
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: now)

    with patch("cron.scheduler.claim_dispatch", return_value=True), patch(
        "cron.scheduler.mark_execution_running"
    ), patch("agent.secret_scope.set_secret_scope", return_value=None), patch(
        "agent.secret_scope.build_profile_secret_scope", return_value=None
    ), patch("agent.secret_scope.reset_secret_scope"), patch(
        "cron.scheduler.run_job", return_value=(False, "full output", "", error)
    ), patch(
        "cron.scheduler.save_job_output", return_value="/tmp/out.md"
    ), patch(
        "cron.scheduler._deliver_result", return_value=None
    ), patch(
        "cron.scheduler._upsert_incident_for_failure", return_value=(False, "incident-a")
    ), patch(
        "cron.scheduler.mark_job_run", return_value=True
    ) as mark_run, patch(
        "cron.scheduler.finish_execution"
    ) as finish:
        assert scheduler.run_one_job(job) is True

    assert mark_run.call_args.args[:3] == (job["id"], False, error)
    backoff = mark_run.call_args.kwargs["provider_backoff"]
    assert backoff["source"] == "provider_reset"
    finish.assert_called_once_with(
        "execution-a",
        success=False,
        error=error,
        delivery_outcome="suppressed",
    )


def test_cron_tool_manual_run_bypasses_active_backoff(temp_home, monkeypatch):
    from cron.jobs import create_job, get_job, mark_job_run
    from cron.rate_limit_backoff import plan_provider_backoff
    from tools import cronjob_tools

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job("work", "every 10m")
    backoff = plan_provider_backoff(job, "HTTP 429", now=now)
    assert mark_job_run(job["id"], False, "HTTP 429", provider_backoff=backoff)

    persisted = get_job(job["id"])
    assert persisted is not None
    with patch.object(
        cronjob_tools,
        "_run_claimed_job",
        return_value={"claimed": True, "success": True, "error": None},
    ):
        result = cronjob_tools._execute_job_now(persisted)

    assert result["claimed"] is True
    assert result["success"] is True


def test_cron_tool_stale_snapshot_cannot_resume_concurrently_paused_job(
    temp_home, monkeypatch
):
    from cron.jobs import create_job, get_job, mark_job_run, pause_job
    from cron.rate_limit_backoff import plan_provider_backoff
    from tools import cronjob_tools

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job("work", "every 10m")
    backoff = plan_provider_backoff(job, "HTTP 429", now=now)
    assert mark_job_run(job["id"], False, "HTTP 429", provider_backoff=backoff)
    stale = get_job(job["id"])
    assert stale is not None
    assert pause_job(job["id"]) is not None

    result = cronjob_tools._execute_job_now(stale)

    assert result["claimed"] is False
    persisted = get_job(job["id"])
    assert persisted is not None
    assert persisted["state"] == "paused"
    assert persisted["enabled"] is False


def test_triggered_run_bypasses_backoff_on_scheduler_claim(temp_home, monkeypatch):
    from cron.jobs import (
        advance_next_runs,
        claim_job_for_fire,
        create_job,
        get_due_jobs,
        mark_job_run,
        trigger_job,
    )
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job("work", "every 10m")
    backoff = plan_provider_backoff(job, "HTTP 429", now=now)
    assert mark_job_run(job["id"], False, "HTTP 429", provider_backoff=backoff)

    triggered = trigger_job(job["id"])

    assert triggered is not None
    assert triggered["manual_run_at"] == triggered["next_run_at"]
    due = get_due_jobs()
    assert [item["id"] for item in due] == [job["id"]]
    advance_next_runs([job["id"]])
    assert claim_job_for_fire(
        job["id"], allow_provider_backoff=True
    ) is True


def test_external_fire_claim_is_suppressed_until_backoff_expires(
    temp_home, monkeypatch
):
    from cron.jobs import claim_job_for_fire, create_job, mark_job_run
    from cron.rate_limit_backoff import plan_provider_backoff

    now = datetime(2026, 9, 3, 4, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    job = create_job("work", "every 10m")
    backoff = plan_provider_backoff(job, "HTTP 429", now=now)
    assert mark_job_run(job["id"], False, "HTTP 429", provider_backoff=backoff)
    assert claim_job_for_fire(job["id"]) is False
    assert claim_job_for_fire(job["id"], force=True) is True
    assert mark_job_run(job["id"], True)

    # A normal unattended claim is allowed again after the reset window.
    monkeypatch.setattr(
        "cron.jobs._hermes_now",
        lambda: datetime.fromisoformat(backoff["until"]) + timedelta(seconds=1),
    )
    assert claim_job_for_fire(job["id"]) is True
