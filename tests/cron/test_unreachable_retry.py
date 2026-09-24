"""Cowork-inspired bounded automatic re-runs for cron fires that never reached the model.

Contract (cron/unreachable_retry.py): a recurring job whose run fails with a transient
network/DNS error before ANY model call gets its ``next_run_at`` pulled earlier along a
bounded ladder (5/15/30 min); a run that reaches the model resets the ladder, and the
ladder never fires past its last rung.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron import unreachable_retry as ur
from cron.jobs import (
    advance_next_runs, create_job, get_due_jobs, get_job, load_jobs, mark_job_run, save_jobs,
)


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_unreachable_failure_pulls_next_run_earlier_then_ladder_exhausts(tmp_cron_home):
    """Failed-unreachable runs re-fire on the 5/15/30-minute ladder instead of waiting a
    full period, and the ladder stops after its last rung (falls back to the schedule)."""
    # Interval, not a cron expression: the natural next fire is always a full day out. A
    # fixed clock time ("0 3 * * *") makes the 30-minute rung land past the natural fire
    # in the half hour before it, and plan_retry rightly yields to the schedule (CI red).
    job = create_job("nightly report", "every 24h")
    job_id = job["id"]

    now = datetime.now(timezone.utc)
    for i, delay in enumerate(ur.RETRY_DELAYS_SECONDS):
        assert mark_job_run(job_id, False, "ConnectError: dns", model_unreachable=True)
        j = get_job(job_id)
        nxt = datetime.fromisoformat(j["next_run_at"])
        # Pulled to roughly now + ladder delay, far before the daily occurrence.
        assert timedelta(0) < nxt - now <= timedelta(seconds=delay + 120), (
            f"attempt {i}: expected retry ~{delay}s out, got {nxt - now}")
        assert j[ur.STATE_KEY]["attempt"] == i + 1

    # Ladder exhausted: the next unreachable failure keeps the natural schedule.
    assert mark_job_run(job_id, False, "ConnectError: dns", model_unreachable=True)
    j = get_job(job_id)
    assert j.get(ur.STATE_KEY) is None
    assert datetime.fromisoformat(j["next_run_at"]) - now > timedelta(hours=1)


def test_reaching_the_model_resets_ladder_and_oneshots_never_retry(tmp_cron_home):
    """Any run that reached the model clears retry state; one-shots (pre-claimed
    dispatch, at-most-times #38758) never enter the ladder."""
    job = create_job("hourly sync", "every 12h")
    job_id = job["id"]
    assert mark_job_run(job_id, False, "ConnectError: dns", model_unreachable=True)
    assert get_job(job_id)[ur.STATE_KEY]["attempt"] == 1

    # A normal failed run (model reached) resets the ladder and stays on schedule.
    assert mark_job_run(job_id, False, "agent error")
    j = get_job(job_id)
    assert j.get(ur.STATE_KEY) is None
    now = datetime.now(timezone.utc)
    assert datetime.fromisoformat(j["next_run_at"]) - now > timedelta(hours=11)

    # One-shot: flag is ignored, no retry state, no resurrection.
    once = create_job("one shot", _iso(datetime.now(timezone.utc) + timedelta(minutes=1)))
    assert mark_job_run(once["id"], False, "ConnectError: dns", model_unreachable=True)
    remaining = get_job(once["id"])
    assert remaining is None or remaining.get(ur.STATE_KEY) is None


def _fail_daily_cron_fire(monkeypatch):
    """A "0 9 * * *" job whose 09:00 fire never reached the model, driven through the due
    scan and completion the tick uses. Returns (job id, clock cell, stored retry instant)."""
    import cron.jobs as jobs

    monkeypatch.setenv("HERMES_TIMEZONE", "UTC")
    clock = [datetime(2026, 9, 21, 8, 58, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(ur, "_hermes_now", lambda: clock[0])
    job_id = create_job("daily report", "0 9 * * *")["id"]

    clock[0] = datetime(2026, 9, 21, 9, 0, 30, tzinfo=timezone.utc)
    assert [j["id"] for j in get_due_jobs()] == [job_id]
    advance_next_runs([job_id])
    assert mark_job_run(job_id, False, "ConnectError: dns", model_unreachable=True)
    retry_at = get_job(job_id)["next_run_at"]
    assert datetime.fromisoformat(retry_at) == clock[0] + timedelta(
        seconds=ur.RETRY_DELAYS_SECONDS[0])
    return job_id, clock, retry_at


def test_cron_expression_retry_is_due_at_its_off_expression_instant(tmp_cron_home, monkeypatch):
    """09:05:30 is not an occurrence of "0 9 * * *", but it is the re-run the ladder promised
    (and held the failure notice back for). The due scan must fire it, not take it for a
    hand-edited expression and re-anchor to tomorrow unfired."""
    job_id, clock, retry_at = _fail_daily_cron_fire(monkeypatch)

    clock[0] = datetime.fromisoformat(retry_at) + timedelta(seconds=30)
    assert [j["id"] for j in get_due_jobs()] == [job_id]
    assert get_job(job_id)["next_run_at"] == retry_at


def test_off_expression_instant_other_than_the_retry_still_reanchors(tmp_cron_home, monkeypatch):
    """The exemption is the exact instant plan_retry wrote. Any other off-expression
    next_run_at (a hand edit while a retry is pending) keeps the stale-expression guard."""
    job_id, clock, retry_at = _fail_daily_cron_fire(monkeypatch)
    edited = (datetime.fromisoformat(retry_at) + timedelta(minutes=2)).isoformat()
    stored = load_jobs()
    for j in stored:
        if j["id"] == job_id:
            j["next_run_at"] = edited
    save_jobs(stored)

    clock[0] = datetime.fromisoformat(edited) + timedelta(seconds=30)
    assert get_due_jobs() == []
    assert get_job(job_id)["next_run_at"] == "2026-09-22T09:00:00+00:00"
