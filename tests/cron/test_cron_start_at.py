"""Recurring cron jobs honor an optional future ``start_at`` (#106908).

``start_at`` is an independent ISO-8601-with-tz parameter on create. It must
not change interval/cron parsing, one-shot ``run_at`` semantics, or
post-first-fire advancement.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron.jobs import create_job, get_due_jobs, get_job, mark_job_run, parse_schedule, resume_job


T0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@pytest.fixture()
def freeze_now(monkeypatch):
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: T0)
    return T0


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_interval_job_honors_future_start_at(tmp_cron_dir, freeze_now):
    start_at = _iso(T0 + timedelta(hours=1))

    job = create_job(prompt="x", schedule="every 10m", start_at=start_at)

    assert job["schedule"]["kind"] == "interval"
    assert job["start_at"] == start_at
    assert job["next_run_at"] == start_at
    stored = get_job(job["id"])
    assert stored["start_at"] == start_at
    assert stored["next_run_at"] == start_at
    assert job["id"] not in {j["id"] for j in get_due_jobs()}


def test_omitted_start_at_keeps_legacy_interval_anchor(tmp_cron_dir, freeze_now):
    job = create_job(prompt="x", schedule="every 10m")

    expected = (T0 + timedelta(minutes=10)).isoformat()
    assert job["next_run_at"] == expected
    assert "start_at" not in job


def test_empty_start_at_keeps_legacy_interval_anchor(tmp_cron_dir, freeze_now):
    job = create_job(prompt="x", schedule="every 10m", start_at="  ")

    assert job["next_run_at"] == (T0 + timedelta(minutes=10)).isoformat()
    assert "start_at" not in job


def test_cron_job_honors_future_start_at(tmp_cron_dir, freeze_now):
    pytest.importorskip("croniter")
    start_at = _iso(T0 + timedelta(hours=3))

    job = create_job(prompt="x", schedule="0 * * * *", start_at=start_at)

    assert job["schedule"]["kind"] == "cron"
    assert job["next_run_at"] == start_at
    assert job["start_at"] == start_at
    # Legacy croniter(now).get_next would be T0+1h, not start_at.
    legacy = parse_schedule("0 * * * *")
    from cron.jobs import compute_next_run
    assert compute_next_run(legacy) != start_at


def test_first_fire_then_interval_advances_from_last_run(tmp_cron_dir, freeze_now, monkeypatch):
    start_at_dt = T0 + timedelta(hours=1)
    job = create_job(prompt="x", schedule="every 10m", start_at=_iso(start_at_dt))
    assert job["next_run_at"] == _iso(start_at_dt)

    fire_at = start_at_dt
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: fire_at)
    assert mark_job_run(job["id"], success=True)

    refreshed = get_job(job["id"])
    assert refreshed["last_run_at"] == fire_at.isoformat()
    assert refreshed["next_run_at"] == (fire_at + timedelta(minutes=10)).isoformat()


def test_past_start_at_fail_open_to_legacy_interval(tmp_cron_dir, freeze_now):
    past = _iso(T0 - timedelta(hours=2))
    job = create_job(prompt="x", schedule="every 10m", start_at=past)

    assert job["next_run_at"] == (T0 + timedelta(minutes=10)).isoformat()
    assert "start_at" not in job
    assert job["id"] not in {j["id"] for j in get_due_jobs()}


def test_once_schedule_ignores_start_at(tmp_cron_dir, freeze_now):
    start_at = _iso(T0 + timedelta(hours=2))
    job = create_job(prompt="x", schedule="in 30m", start_at=start_at)

    assert job["schedule"]["kind"] == "once"
    assert job["next_run_at"] == (T0 + timedelta(minutes=30)).isoformat()
    assert "start_at" not in job


def test_naive_start_at_raises(tmp_cron_dir, freeze_now):
    with pytest.raises(ValueError, match="timezone"):
        create_job(prompt="x", schedule="every 10m", start_at="2026-09-10T13:00:00")


def test_unparseable_start_at_raises(tmp_cron_dir, freeze_now):
    with pytest.raises(ValueError, match="start_at"):
        create_job(prompt="x", schedule="every 10m", start_at="next tuesday")


def test_resume_never_run_job_still_honors_future_start_at(tmp_cron_dir, freeze_now):
    start_at = _iso(T0 + timedelta(hours=1))
    job = create_job(prompt="x", schedule="every 10m", start_at=start_at, paused=True)
    assert job["next_run_at"] is None
    assert job["start_at"] == start_at

    resumed = resume_job(job["id"])
    assert resumed["next_run_at"] == start_at


def test_cronjob_tool_create_forwards_start_at(tmp_cron_dir, freeze_now, monkeypatch):
    import json

    from tools.cronjob_tools import cronjob

    monkeypatch.setattr(
        "cron.scheduler.create_job_with_scheduler_registration",
        lambda **kw: create_job(**kw),
    )
    start_at = _iso(T0 + timedelta(hours=1))
    payload = json.loads(cronjob(
        action="create", prompt="x", schedule="every 10m", start_at=start_at,
    ))
    assert payload["success"] is True
    stored = get_job(payload["job_id"])
    assert stored["next_run_at"] == start_at
    assert stored["start_at"] == start_at
