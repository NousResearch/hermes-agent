"""Regression: with no Hermes timezone configured (the shipped default), cron resolves every wall
clock in the host's zone with the DST rules of that wall clock's own date.

The server-local clock is ``datetime.now().astimezone()``, a FIXED UTC offset, and cron attached
it to wall clocks on other dates. On a New York host the first ``0 9 * * *`` run after
spring-forward was stored at 09:00-05:00 and fired at 10:00 EDT (logging a false
``timezone_migration.catch_up``), and a naive one-shot typed in February for 1 April fired at
10:00 EDT. The host zone is set the way a real host sets it (``TZ`` + ``time.tzset()``), and the
scheduler clock is pinned to that host clock at each call, as ``mark_job_run`` computes the next
run right after the run it records.
"""

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

import hermes_time
from cron import jobs

pytestmark = pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset() is POSIX-only")

NEW_YORK = ZoneInfo("America/New_York")
MORNING = {"kind": "cron", "expr": "0 9 * * *"}


@pytest.fixture
def new_york_host(monkeypatch):
    """A host in America/New_York with no Hermes timezone configured; yields a monkeypatch whose
    changes (clock pins included) are undone before the process zone is restored."""
    with monkeypatch.context() as mp:
        mp.setenv("TZ", "America/New_York")
        time.tzset()
        hermes_time.reset_cache()
        assert hermes_time.get_timezone() is None
        yield mp
    time.tzset()
    hermes_time.reset_cache()


def _pin_clock(mp, instant: datetime) -> datetime:
    """Pin the scheduler clock to *instant* (naive = host-local) as ``hermes_time.now()`` reads
    it: in the configured zone, else at the host clock's fixed offset."""
    now = instant.astimezone(hermes_time.get_timezone())
    mp.setattr(jobs, "_hermes_now", lambda: now)
    return now


def _next_after_run(mp, schedule, ran_at: datetime) -> datetime:
    last = _pin_clock(mp, ran_at)
    return datetime.fromisoformat(jobs.compute_next_run(schedule, last_run_at=last.isoformat()))


def _configured_next_after_run(mp, schedule, ran_at: datetime) -> datetime:
    """The same computation with the host's zone configured explicitly (the reference path)."""
    mp.setenv("HERMES_TIMEZONE", "America/New_York")
    hermes_time.reset_cache()
    try:
        return _next_after_run(mp, schedule, ran_at)
    finally:
        mp.delenv("HERMES_TIMEZONE")
        hermes_time.reset_cache()


class TestCronOccurrencesKeepTheHostWallClock:
    def test_first_run_after_each_dst_change_is_9am(self, new_york_host):
        """Spring-forward (Mar 8 2026) and fall-back (Nov 1 2026): 10:00 EDT / 08:00 EST before."""
        for ran_at in (datetime(2026, 3, 7, 9, 0, 5), datetime(2026, 10, 31, 9, 0, 5)):
            wall = _next_after_run(new_york_host, MORNING, ran_at).astimezone(NEW_YORK)
            assert (wall.date(), wall.hour, wall.minute) == (ran_at.date() + timedelta(days=1), 9, 0)

    def test_full_year_walk_never_drifts(self, new_york_host):
        ran_at = datetime(2026, 1, 1, 9, 0)
        for _ in range(365):
            nxt = _next_after_run(new_york_host, MORNING, ran_at)
            wall = nxt.astimezone(NEW_YORK)
            assert (wall.hour, wall.minute) == (9, 0), f"drift after {ran_at}: {wall.isoformat()}"
            ran_at = nxt

    @pytest.mark.parametrize("expr", ["30 2 * * *", "30 1 * * *", "*/20 * * * *"])
    def test_transition_hours_resolve_like_the_configured_zone(self, new_york_host, expr):
        """The skipped spring hour and the repeated autumn hour: every occurrence stepped across
        both transition nights lands on the instant the configured-zone path picks."""
        schedule = {"kind": "cron", "expr": expr}
        for start in (datetime(2026, 3, 7, 0, 0), datetime(2026, 10, 31, 0, 0)):
            ran_at, end = start.astimezone(), (start + timedelta(days=2)).astimezone()
            while ran_at < end:
                host = _next_after_run(new_york_host, schedule, ran_at)
                configured = _configured_next_after_run(new_york_host, schedule, ran_at)
                assert host == configured, f"{expr} after {ran_at}: {host} != {configured}"
                ran_at = host

    def test_daily_job_fires_at_9am_on_spring_forward_day(self, new_york_host):
        """Through the real store: the Saturday run must leave Sunday's occurrence due at 09:00
        EDT, with no timezone-migration catch-up recorded for it."""
        _pin_clock(new_york_host, datetime(2026, 3, 7, 8, 0))
        job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="morning")
        _pin_clock(new_york_host, datetime(2026, 3, 7, 9, 0, 30))
        assert job["id"] in [j["id"] for j in jobs.get_due_jobs()]
        assert jobs.advance_next_run(job["id"])
        assert jobs.mark_job_run(job["id"], success=True)
        catchups = jobs.get_timezone_migration_catchup_stats()["timezone_migration_catchups"]

        _pin_clock(new_york_host, datetime(2026, 3, 8, 9, 0, 30))

        assert job["id"] in [j["id"] for j in jobs.get_due_jobs()]
        assert jobs.get_timezone_migration_catchup_stats()["timezone_migration_catchups"] == catchups


class TestNaiveWallClocksReadOnTheirOwnDate:
    @pytest.mark.parametrize("created, typed", [
        (datetime(2026, 2, 10, 12, 0), "2026-04-01T09:00"),  # made in EST, due in EDT: was 10:00
        (datetime(2026, 8, 10, 12, 0), "2026-12-01T09:00"),  # made in EDT, due in EST: was 08:00
    ])
    def test_one_shot_fires_at_the_typed_wall_clock(self, new_york_host, created, typed):
        _pin_clock(new_york_host, created)
        run_at = datetime.fromisoformat(jobs.parse_schedule(typed)["run_at"])
        assert run_at == datetime.fromisoformat(typed).replace(tzinfo=NEW_YORK)

    @pytest.mark.parametrize("typed", ["0001-01-01T00:00", "9999-12-31T23:59"])
    def test_one_shot_at_the_datetime_bounds_is_still_a_plain_rejection(self, new_york_host, typed):
        """Reading a wall clock a day either side cannot overflow into an unexpected error type."""
        with pytest.raises(ValueError):
            jobs.create_job(prompt="x", schedule=typed)

    def test_legacy_naive_timestamp_keeps_its_wall_clock(self, new_york_host):
        """Naive values from older builds are host-local wall time. One of these two sits across a
        DST change from whatever day the suite runs, so reading both with today's offset moves one
        of them by an hour."""
        for naive in (datetime(2026, 1, 15, 9, 0), datetime(2026, 7, 15, 9, 0)):
            assert jobs._ensure_aware(naive) == naive.replace(tzinfo=NEW_YORK)
