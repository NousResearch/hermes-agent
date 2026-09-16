"""Interval jobs arm their next slot at FIRE time, not at run completion.

WorkHub WH-CREATED-9FACA739BCC9 (option A of the automation-throughput review). An interval
schedule is fixed-rate: ``scheduler.tick`` arms the next slot from the FIRE instant
(``advance_next_runs``) before dispatching, and ``claim_job_for_fire`` does the same for
external fires. ``mark_job_run`` then recomputed ``next_run_at`` from the FINISH instant,
which added the run's wall time to every period — measured on the 15-minute executor slot:
median inter-fire gap 928s against a 900s interval (~70 real ticks/day instead of 96).

Contract pinned here: a slot already armed ahead of us is left alone; only a slot that has
already elapsed by completion (a run that overran its interval, or a record with no armed
slot) is recomputed forward from now, so the due-scan's catch-up window still decides how
many missed slots actually re-run.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron.jobs import (
    _ensure_aware,
    create_job,
    get_job,
    mark_job_run,
    parse_schedule,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory (same seam as tests/cron/test_jobs.py)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class _Clock:
    """A settable ``_hermes_now`` so a run's start and finish are exact, not slept through."""

    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture()
def clock(monkeypatch):
    c = _Clock(datetime(2026, 9, 17, 0, 0, 0, tzinfo=timezone.utc))
    monkeypatch.setattr("cron.jobs._hermes_now", c)
    return c


def _next_run(job_id: str) -> datetime:
    return _ensure_aware(datetime.fromisoformat(get_job(job_id)["next_run_at"]))


def test_interval_slot_survives_a_short_run_anchored_at_run_start(tmp_cron_dir, clock):
    """A 93s run on a 15m interval slot must not push the slot out by 93s.

    RED on the pre-fix behaviour: ``mark_job_run`` recomputed from the finish instant, so
    ``next_run_at`` came back as finish + 15m instead of the start + 15m the tick armed.
    """
    job = create_job(prompt="Executor tick", schedule="every 15m")
    fire_at = clock.now
    armed_at_fire = _next_run(job["id"])
    assert abs((armed_at_fire - (fire_at + timedelta(minutes=15))).total_seconds()) < 2

    clock.now = fire_at + timedelta(seconds=93)
    assert mark_job_run(job["id"], success=True) is True

    after = _next_run(job["id"])
    assert abs((after - (fire_at + timedelta(minutes=15))).total_seconds()) < 2, (
        f"interval slot must stay anchored at run start + interval; got {after.isoformat()} "
        f"(finish-anchored would be {(clock.now + timedelta(minutes=15)).isoformat()})")


def test_interval_slot_elapsed_by_an_overrunning_run_is_re_anchored_forward(tmp_cron_dir, clock):
    """A run that outlives its interval gets one fresh period, never an immediately-due slot.

    While such a run is in flight the next due slot is skipped by the scheduler's
    single-flight guard (``_submit_with_guard``); the record must not also leave behind a
    past-due slot that would queue the skipped occurrence as a burst.
    """
    job = create_job(prompt="Long tick", schedule="every 5m")
    fire_at = clock.now

    clock.now = fire_at + timedelta(minutes=11)  # outran the 5m interval
    assert mark_job_run(job["id"], success=True) is True

    after = _next_run(job["id"])
    assert after > clock.now, "a completed run must not leave the job already due"
    assert abs((after - (clock.now + timedelta(minutes=5))).total_seconds()) < 2


def test_cron_and_oneshot_schedules_keep_their_own_next_run_semantics(tmp_cron_dir, clock):
    """The interval anchor rule must not leak into cron-expression or one-shot schedules."""
    cron_job = create_job(prompt="Hourly", schedule="0 * * * *")
    clock.now = clock.now + timedelta(minutes=1)
    assert mark_job_run(cron_job["id"], success=True) is True
    cron_next = _next_run(cron_job["id"])
    assert cron_next > clock.now
    assert cron_next.minute == 0

    oneshot = create_job(prompt="Run once", schedule="in 30m")
    oneshot_next = _next_run(oneshot["id"])
    clock.now = clock.now + timedelta(minutes=1)
    assert mark_job_run(oneshot["id"], success=True) is True
    assert get_job(oneshot["id"])["next_run_at"] is None

    assert parse_schedule("every 15m") == {
        "kind": "interval", "minutes": 15, "display": "every 15m"}
