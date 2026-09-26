"""Cowork-inspired bounded automatic re-runs for cron fires that never reached the model.

Contract (cron/unreachable_retry.py): a recurring job whose run fails with a transient
network/DNS error before ANY model call gets its ``next_run_at`` pulled earlier along a
bounded ladder (5/15/30 min); a run that reaches the model resets the ladder, and the
ladder never fires past its last rung.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron import unreachable_retry as ur
from cron.jobs import create_job, get_due_jobs, get_job, load_jobs, mark_job_run, save_jobs


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_unreachable_failure_pulls_next_run_earlier_then_ladder_exhausts(
    tmp_cron_home, monkeypatch,
):
    """Failed-unreachable runs re-fire on the 5/15/30-minute ladder instead of waiting a
    full period, and the ladder stops after its last rung (falls back to the schedule). A
    cron job's ladder instant is off its lattice yet must be due, not re-anchored as a stale
    expression edit."""
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

    pinned = datetime(2026, 9, 18, 12, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: pinned)
    monkeypatch.setattr(ur, "_hermes_now", lambda: pinned)
    weekly = create_job("weekly digest", "0 12 * * 5")
    assert mark_job_run(weekly["id"], False, "ConnectError: dns", model_unreachable=True)
    retry_at = datetime.fromisoformat(get_job(weekly["id"])["next_run_at"])
    assert retry_at == pinned + timedelta(seconds=ur.RETRY_DELAYS_SECONDS[0])
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: retry_at + timedelta(seconds=1))
    assert weekly["id"] in {due["id"] for due in get_due_jobs()}

    # A direct jobs.json expression edit while a retry is parked re-anchors without firing.
    assert mark_job_run(weekly["id"], False, "ConnectError: dns", model_unreachable=True)
    retry_at = datetime.fromisoformat(get_job(weekly["id"])["next_run_at"])
    jobs = load_jobs()
    next(j for j in jobs if j["id"] == weekly["id"])["schedule"]["expr"] = "0 9 * * 1"
    save_jobs(jobs)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: retry_at + timedelta(seconds=1))
    assert weekly["id"] not in {due["id"] for due in get_due_jobs()}


def test_will_retry_mirrors_plan_retry_yield(tmp_cron_home):
    """The notice-suppression predictor must mirror what ``plan_retry`` will actually do
    for THIS failure:

    - a cadence at or under the next ladder rung makes the schedule's own fire the retry
      (``plan_retry`` yields without consuming an attempt), so no re-run is scheduled and
      the failure notice must go out — otherwise a fast job holds every failure notice
      for as long as the outage lasts (the "ladder exhausted" escape is unreachable when
      every rung yields);
    - a slow job still gets its silent bounded re-runs (regression guard for the feature).
    """
    fast = create_job("fast poll", "every 2m")
    assert mark_job_run(fast["id"], False, "ConnectError: dns", model_unreachable=True)
    j = get_job(fast["id"])
    assert j is not None
    assert j.get(ur.STATE_KEY) is None, "2m cadence beats the 5m rung: plan_retry yields"
    assert ur.will_retry(j) is False, "yielded: no re-run is scheduled, notice must go out"

    slow = create_job("nightly report", "every 24h")
    assert mark_job_run(slow["id"], False, "ConnectError: dns", model_unreachable=True)
    js = get_job(slow["id"])
    assert js is not None
    assert js[ur.STATE_KEY]["attempt"] == 1
    assert ur.will_retry(js) is True, "5m rung beats the 24h cadence: re-run is scheduled"

    mid = create_job("ten minute sync", "every 10m")
    assert mark_job_run(mid["id"], False, "ConnectError: dns", model_unreachable=True)
    jm = get_job(mid["id"])
    assert jm is not None
    # 10m cadence beats the 15m and 30m rungs: the ladder can never climb past attempt 1,
    # so the exhaustion escape is unreachable and the notice must go out immediately.
    assert jm[ur.STATE_KEY]["attempt"] == 1
    assert ur.will_retry(jm) is False, "10m cadence beats the 15m rung: yielded, notice goes out"

    last = create_job("final run", "every 24h", repeat=1)
    assert ur.will_retry(get_job(last["id"])) is False, "final finite repeat completes the job"


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
