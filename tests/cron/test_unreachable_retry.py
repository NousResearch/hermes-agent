"""Cowork-inspired bounded automatic re-runs for cron fires that never reached the model.

Contract (cron/unreachable_retry.py): a recurring job whose run fails with a transient
network/DNS error before ANY model call gets its ``next_run_at`` pulled earlier along a
bounded ladder (5/15/30 min); a run that reaches the model resets the ladder, and the
ladder never fires past its last rung.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron import unreachable_retry as ur
from cron.jobs import create_job, get_job, mark_job_run


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
    job = create_job("nightly report", "0 3 * * *")  # daily — natural gap is hours
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
