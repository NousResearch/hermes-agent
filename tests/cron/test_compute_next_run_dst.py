"""Regression: ``compute_next_run`` must hold a cron job's wall-clock hour in the configured
IANA timezone across DST.

croniter works in the UTC *offset* of its start time, never the zone, so a ``0 9 * * *`` job
in America/Toronto fired at 08:00 local on spring-forward day and 10:00 on fall-back day, and
stayed an hour off for the rest of each season. The zone is configured the way production
configures it (``HERMES_TIMEZONE``) instead of by patching ``get_timezone``, so
``get_timezone()`` and the stored-timestamp normalization agree and these tests fail on the
unfixed code with the real symptom. Filed from the 09:00 America/Toronto morning routine.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

pytest.importorskip("croniter")

import hermes_time
from cron.jobs import compute_next_run

TORONTO = ZoneInfo("America/Toronto")
MORNING = {"kind": "cron", "expr": "0 9 * * *"}


@pytest.fixture
def toronto(monkeypatch):
    """Configure the active profile's zone through the real resolution path."""
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Toronto")
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


@pytest.fixture
def singapore(monkeypatch):
    """Configure the active profile's zone to Asia/Singapore (UTC+8)."""
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Singapore")
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


def _next_local(last_run_at: str) -> datetime:
    return datetime.fromisoformat(
        compute_next_run(MORNING, last_run_at=last_run_at)).astimezone(TORONTO)


class TestCronNextRunHoldsConfiguredWallClock:
    def test_dst_transition_days_fire_at_9am_local(self, toronto):
        """Spring-forward (Mar 8 2026) and fall-back (Nov 1 2026) days: 08:00 / 10:00 before."""
        spring = _next_local("2026-03-07T09:00:00-05:00")
        fall = _next_local("2026-10-31T09:00:00-04:00")
        assert (spring.hour, spring.minute) == (9, 0)
        assert (fall.hour, fall.minute) == (9, 0)

    def test_full_year_walk_never_drifts(self, toronto):
        """Walking 2026 crosses both transitions; the hour must stay 09:00 throughout."""
        last = datetime(2026, 1, 1, 9, 0, tzinfo=TORONTO)
        for _ in range(365):
            nxt = datetime.fromisoformat(
                compute_next_run(MORNING, last_run_at=last.isoformat()))
            wall = nxt.astimezone(TORONTO)
            assert (wall.hour, wall.minute) == (9, 0), \
                f"drift at {last.isoformat()}: {wall.isoformat()}"
            last = nxt


class TestFallBackStrictlyAfterContract:
    """qwen-code#11723 class: a naive wall clock re-attached to the zone resolves the
    repeated autumn hour to its earlier occurrence, so a base inside the second
    occurrence used to get a next_run in the PAST (immediate re-fire loop)."""

    def test_base_in_second_occurrence_gets_future_instant(self, toronto):
        base = datetime(2026, 11, 1, 1, 15, tzinfo=TORONTO, fold=1)  # 01:15 EST
        nxt = datetime.fromisoformat(
            compute_next_run({"kind": "cron", "expr": "30 1 * * *"},
                             last_run_at=base.isoformat()))
        assert nxt.timestamp() > base.timestamp()
        wall = nxt.astimezone(TORONTO)
        assert (wall.hour, wall.minute) == (1, 30)

    def test_every_minute_never_returns_past(self, toronto):
        base = datetime(2026, 11, 1, 1, 59, tzinfo=TORONTO, fold=0)  # last EDT minute
        nxt = datetime.fromisoformat(
            compute_next_run({"kind": "cron", "expr": "* * * * *"},
                             last_run_at=base.isoformat()))
        assert nxt.timestamp() > base.timestamp()


class TestScheduleSpecificTimezone:
    def test_schedule_tz_overrides_profile_timezone(self, singapore):
        """When profile is Asia/Singapore (UTC+8), a job with tz='America/New_York'
        evaluates against New York wall clock (09:00 EDT = 21:00 SGT), not 09:00 SGT."""
        ny = ZoneInfo("America/New_York")
        sg = ZoneInfo("Asia/Singapore")
        schedule = {"kind": "cron", "expr": "0 9 * * 1-5", "tz": "America/New_York"}
        # Base is in Singapore time: Monday morning 08:00 SGT (which is Sunday night in NY)
        base = datetime(2026, 10, 12, 8, 0, tzinfo=sg)
        nxt_str = compute_next_run(schedule, last_run_at=base.isoformat())
        assert nxt_str is not None
        nxt = datetime.fromisoformat(nxt_str)
        # Next run should be Monday 09:00 NY time (21:00 SGT)
        wall_ny = nxt.astimezone(ny)
        assert (wall_ny.year, wall_ny.month, wall_ny.day) == (2026, 10, 12)
        assert (wall_ny.hour, wall_ny.minute) == (9, 0)
        wall_sg = nxt.astimezone(sg)
        assert (wall_sg.hour, wall_sg.minute) == (21, 0)

    def test_invalid_schedule_tz_falls_back_with_warning(self, singapore, caplog):
        """Invalid timezone falls back to profile timezone and logs a warning."""
        schedule = {"kind": "cron", "expr": "0 9 * * 1-5", "tz": "Not/AZone"}
        base = datetime(2026, 10, 12, 8, 0, tzinfo=ZoneInfo("Asia/Singapore"))
        nxt_str = compute_next_run(schedule, last_run_at=base.isoformat())
        assert nxt_str is not None
        nxt = datetime.fromisoformat(nxt_str)
        # Falls back to profile timezone (Singapore): next run is Monday 09:00 SGT
        wall_sg = nxt.astimezone(ZoneInfo("Asia/Singapore"))
        assert (wall_sg.hour, wall_sg.minute) == (9, 0)
        assert any("Invalid timezone 'Not/AZone'" in r.message for r in caplog.records)

    def test_create_and_update_job_with_tz(self, tmp_path, monkeypatch):
        """create_job and update_job persist and validate schedule tz."""
        import cron.jobs as jobs_module
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")

        from cron.jobs import create_job, update_job

        # Rejects invalid timezone
        with pytest.raises(ValueError, match="Invalid timezone 'Bad/TZ'"):
            create_job(prompt="ping", schedule="0 9 * * *", tz="Bad/TZ")

        # Rejects timezone on non-cron schedules (interval, once)
        with pytest.raises(ValueError, match="only supported for cron schedules"):
            create_job(prompt="ping", schedule="every 1h", tz="America/New_York")

        # Stores valid timezone
        job = create_job(prompt="ping", schedule="0 9 * * *", tz="America/New_York")
        assert job["schedule"].get("tz") == "America/New_York"
        assert job["schedule_display"] == "0 9 * * * (America/New_York)"
        assert job["schedule"]["display"] == "0 9 * * * (America/New_York)"

        # update_job repeatedly updates timezone without accumulating suffixes
        up1 = update_job(job["id"], {"tz": "Europe/London"})
        assert up1 is not None
        assert up1["schedule"].get("tz") == "Europe/London"
        assert up1["schedule_display"] == "0 9 * * * (Europe/London)"
        assert up1["schedule"]["display"] == "0 9 * * * (Europe/London)"

        up2 = update_job(job["id"], {"tz": "Asia/Tokyo"})
        assert up2 is not None
        assert up2["schedule"].get("tz") == "Asia/Tokyo"
        assert up2["schedule_display"] == "0 9 * * * (Asia/Tokyo)"

        up3 = update_job(job["id"], {"tz": "America/New_York"})
        assert up3 is not None
        assert up3["schedule"].get("tz") == "America/New_York"
        assert up3["schedule_display"] == "0 9 * * * (America/New_York)"

        # Updating schedule expression preserves existing tz when tz not in updates
        up4 = update_job(job["id"], {"schedule": "0 10 * * 1-5"})
        assert up4 is not None
        assert up4["schedule"].get("tz") == "America/New_York"
        assert up4["schedule_display"] == "0 10 * * 1-5 (America/New_York)"

        # update_job clears timezone with empty string and cleanly removes suffix from display
        cleared = update_job(job["id"], {"tz": ""})
        assert cleared is not None
        assert "tz" not in cleared["schedule"]
        assert cleared["schedule_display"] == "0 10 * * 1-5"
        assert cleared["schedule"]["display"] == "0 10 * * 1-5"

        # Combined schedule and tz update keeps schedule_display consistent
        comb = update_job(job["id"], {"schedule": "0 8 * * *", "tz": "Europe/Paris"})
        assert comb is not None
        assert comb["schedule"].get("tz") == "Europe/Paris"
        assert comb["schedule_display"] == "0 8 * * * (Europe/Paris)"
        assert comb["schedule"]["display"] == "0 8 * * * (Europe/Paris)"

        # Converting a tz-carrying cron schedule to non-cron drops tz cleanly
        non_cron = update_job(job["id"], {"schedule": "every 1h"})
        assert non_cron is not None
        assert "tz" not in non_cron["schedule"]
        assert "(Europe/Paris)" not in non_cron["schedule_display"]
        assert non_cron["schedule"]["kind"] == "interval"

        # Explicitly updating non-cron schedule with tz fails
        with pytest.raises(ValueError, match="only supported for cron schedules"):
            update_job(job["id"], {"tz": "America/New_York"})

        # Dict schedule with tz and no 'display' key updates safely without KeyError
        dict_sched = update_job(
            job["id"],
            {"schedule": {"kind": "cron", "expr": "0 8 * * *", "tz": "Asia/Tokyo"}},
        )
        assert dict_sched is not None
        assert dict_sched["schedule"].get("tz") == "Asia/Tokyo"
        assert dict_sched["schedule_display"] == "0 8 * * * (Asia/Tokyo)"
        assert dict_sched["schedule"]["display"] == "0 8 * * * (Asia/Tokyo)"
        # Verify no stray top-level tz key leaks onto the record
        assert "tz" not in dict_sched

