"""Cron schedules must carry the timezone they are evaluated in (#88220).

Background
----------
A cron expression describes *local wall-clock* intent ("run at 14:30"), but
``next_run_at`` is persisted as an absolute instant. Which zone that wall clock
belongs to used to be implicit: every process re-resolved it from
``HERMES_TIMEZONE`` / ``config.yaml`` at the moment it read or wrote the job.

That made a persisted ``next_run_at`` ambiguous whenever two readers disagreed:
a long-running gateway pins its zone at boot, while a freshly spawned CLI /
web-worker picks up an edited ``timezone:`` immediately. The same string then
meant two different instants, and the due check silently switched between
"absolute instant" and "naive wall clock" semantics depending on an offset
comparison — shifting a job by the whole UTC offset (8h for ``Asia/Shanghai``)
on the next gateway restart, firing it early and swallowing the real run.

The fix stamps the evaluating zone onto the schedule (``schedule["tz"]``) so
every process computes the same instant, and replaces the offset-delta
heuristic with an explicit zone-identity rebase for stamped jobs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

from cron.jobs import (  # noqa: E402
    compute_next_run,
    create_job,
    get_due_jobs,
    get_job,
    parse_schedule,
    save_jobs,
)


@pytest.fixture(autouse=True)
def _isolate_timezone_cache():
    """Keep this module's zone pinning out of every other test.

    ``hermes_time`` caches the resolved zone in module globals, which monkeypatch
    cannot restore — without this, a zone set here would leak into unrelated
    tests running later in the same process.
    """
    import hermes_time

    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def use_zone(monkeypatch, tz_name: str, wall_clock: datetime) -> datetime:
    """Pin the process' configured zone AND its ``now()`` for the test.

    ``wall_clock`` is a naive local wall-clock reading in ``tz_name``; the
    returned aware datetime is what ``_hermes_now()`` will yield.
    """
    import hermes_time

    monkeypatch.setenv("HERMES_TIMEZONE", tz_name)
    hermes_time.reset_cache()
    now = wall_clock.replace(tzinfo=ZoneInfo(tz_name))
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    return now


def use_server_local_utc(monkeypatch, wall_clock_utc: datetime) -> datetime:
    """Model the process that has NO configured Hermes zone.

    This is the shape of the gateway in #88220: it booted before ``timezone:``
    was added to config.yaml, so it runs on the container's UTC clock with no
    configured zone at all. Such a process must honour what the job says
    instead of reinterpreting it.
    """
    import hermes_time

    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    monkeypatch.setattr(hermes_time, "get_timezone_name", lambda: "")
    now = wall_clock_utc.replace(tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    return now


def a_job(**overrides) -> dict:
    job = {
        "id": "job-1",
        "name": "daily report",
        "prompt": "...",
        "schedule": {"kind": "cron", "expr": "30 14 * * *", "display": "30 14 * * *"},
        "schedule_display": "30 14 * * *",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": "2026-08-14T14:30:00+08:00",
        "next_run_at": "2026-08-17T14:30:00+08:00",
        "last_run_at": "2026-08-16T14:30:00+08:00",
        "last_status": "ok",
        "last_error": None,
        "deliver": "local",
        "origin": None,
    }
    job.update(overrides)
    return job


# =============================================================================
# 1. The schedule records the zone it is evaluated in
# =============================================================================

class TestScheduleCarriesItsZone:
    def test_parse_schedule_stamps_configured_zone_on_cron(self, monkeypatch):
        use_zone(monkeypatch, "Asia/Shanghai", datetime(2026, 8, 17, 22, 0))
        assert parse_schedule("30 14 * * *")["tz"] == "Asia/Shanghai"

    def test_parse_schedule_omits_zone_when_none_configured(self, monkeypatch):
        import hermes_time

        monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
        monkeypatch.setattr(hermes_time, "get_timezone_name", lambda: "")
        # No configured zone → server-local semantics stay exactly as before.
        assert "tz" not in parse_schedule("30 14 * * *")

    def test_created_job_persists_its_zone(self, tmp_cron_dir, monkeypatch):
        use_zone(monkeypatch, "Asia/Shanghai", datetime(2026, 8, 17, 22, 0))
        job = create_job(prompt="report", schedule="30 14 * * *", name="daily")
        assert get_job(job["id"])["schedule"]["tz"] == "Asia/Shanghai"

    def test_interval_and_oneshot_schedules_are_untouched(self, monkeypatch):
        use_zone(monkeypatch, "Asia/Shanghai", datetime(2026, 8, 17, 22, 0))
        # Only cron expressions carry wall-clock intent; interval/once are pure
        # instants and must not grow a field other readers don't expect.
        assert "tz" not in parse_schedule("every 30m")
        assert "tz" not in parse_schedule("2h")


# =============================================================================
# 2. compute_next_run is deterministic across processes
# =============================================================================

class TestComputeNextRunIsProcessIndependent:
    def test_same_instant_regardless_of_reader_zone(self, monkeypatch):
        """The bug's core: two processes, two zones, one schedule.

        A gateway pinned to UTC and a CLI resolving Asia/Shanghai must agree on
        the next fire time down to the instant.
        """
        schedule = {"kind": "cron", "expr": "30 14 * * *", "tz": "Asia/Shanghai"}
        last_run = "2026-08-16T14:30:00+08:00"

        use_zone(monkeypatch, "Asia/Shanghai", datetime(2026, 8, 16, 14, 31))
        from_cli = compute_next_run(schedule, last_run_at=last_run)

        use_server_local_utc(monkeypatch, datetime(2026, 8, 16, 6, 31))
        from_gateway = compute_next_run(schedule, last_run_at=last_run)

        assert datetime.fromisoformat(from_cli) == datetime.fromisoformat(from_gateway)
        assert datetime.fromisoformat(from_cli) == datetime(
            2026, 8, 17, 14, 30, tzinfo=ZoneInfo("Asia/Shanghai")
        )

    def test_offset_label_matches_the_instant_it_encodes(self, monkeypatch):
        """A ``+08:00`` label must never sit on a non-Shanghai wall clock."""
        use_server_local_utc(monkeypatch, datetime(2026, 8, 16, 6, 31))
        schedule = {"kind": "cron", "expr": "30 14 * * *", "tz": "Asia/Shanghai"}
        result = datetime.fromisoformat(
            compute_next_run(schedule, last_run_at="2026-08-16T14:30:00+08:00")
        )
        assert result.utcoffset() == timedelta(hours=8)
        assert result.astimezone(timezone.utc).hour == 6  # 14:30+08:00 == 06:30Z

    def test_unstamped_schedule_keeps_process_zone_behaviour(self, monkeypatch):
        """Legacy jobs (no ``tz``) must behave exactly as before the fix."""
        use_zone(monkeypatch, "Asia/Shanghai", datetime(2026, 8, 16, 14, 31))
        schedule = {"kind": "cron", "expr": "30 14 * * *"}
        result = datetime.fromisoformat(
            compute_next_run(schedule, last_run_at="2026-08-16T14:30:00+08:00")
        )
        assert result == datetime(2026, 8, 17, 14, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


# =============================================================================
# 3. The #88220 reproducer, end to end
# =============================================================================

class TestGatewayRestartDoesNotShiftTheJob:
    def test_stamped_job_is_not_reinterpreted_by_a_utc_gateway(
        self, tmp_cron_dir, monkeypatch
    ):
        """Gateway pinned to UTC reads a Shanghai-stamped job.

        Stored ``2026-08-17T14:30:00+08:00`` is 06:30Z. At 06:00Z the job is
        NOT due, and the due scan must not rewrite it into the UTC wall clock
        (which is what moved the job by 8 hours).
        """
        job = a_job()
        job["schedule"]["tz"] = "Asia/Shanghai"
        save_jobs([job])

        use_server_local_utc(monkeypatch, datetime(2026, 8, 17, 6, 0))
        assert get_due_jobs() == []

        stored = get_job("job-1")
        assert datetime.fromisoformat(stored["next_run_at"]) == datetime(
            2026, 8, 17, 14, 30, tzinfo=ZoneInfo("Asia/Shanghai")
        )
        assert stored["schedule"]["tz"] == "Asia/Shanghai"

    def test_stamped_job_fires_at_its_own_zone_instant(self, tmp_cron_dir, monkeypatch):
        job = a_job()
        job["schedule"]["tz"] = "Asia/Shanghai"
        save_jobs([job])

        # 06:30:10Z == 14:30:10 Shanghai → genuinely due.
        use_server_local_utc(monkeypatch, datetime(2026, 8, 17, 6, 30, 10))
        assert [j["id"] for j in get_due_jobs()] == ["job-1"]

    def test_restart_does_not_replay_the_job_early(self, tmp_cron_dir, monkeypatch):
        """Full reproducer: fire, persist, 'restart' into another zone, re-scan.

        The job must not become due a second time inside the same period.
        """
        use_zone(monkeypatch, "Asia/Shanghai", datetime(2026, 8, 17, 14, 30, 4))
        job = a_job()
        job["schedule"]["tz"] = "Asia/Shanghai"
        job["next_run_at"] = compute_next_run(
            job["schedule"], last_run_at="2026-08-17T14:30:04+08:00"
        )
        job["last_run_at"] = "2026-08-17T14:30:04+08:00"
        save_jobs([job])

        # Gateway restarts and resolves UTC instead (stale/absent config).
        for wall in (
            datetime(2026, 8, 17, 6, 35),   # just after the run, UTC clock
            datetime(2026, 8, 17, 14, 30),  # the old, wrong 14:30 *UTC* slot
            datetime(2026, 8, 18, 6, 25),   # five minutes before the real slot
        ):
            use_server_local_utc(monkeypatch, wall)
            assert get_due_jobs() == [], f"job fired early at {wall}Z"

        use_server_local_utc(monkeypatch, datetime(2026, 8, 18, 6, 30, 5))
        assert [j["id"] for j in get_due_jobs()] == ["job-1"]


# =============================================================================
# 4. A real timezone change still re-anchors wall-clock intent (#28934)
# =============================================================================

class TestConfiguredZoneChange:
    def test_zone_change_rebases_and_restamps_once(self, tmp_cron_dir, monkeypatch):
        job = a_job()
        job["schedule"]["tz"] = "Asia/Shanghai"
        save_jobs([job])

        # Operator switches config.yaml to Europe/Berlin (+02:00 in August).
        use_zone(monkeypatch, "Europe/Berlin", datetime(2026, 8, 17, 8, 0))
        assert get_due_jobs() == []

        stored = get_job("job-1")
        assert stored["schedule"]["tz"] == "Europe/Berlin"
        rebased = datetime.fromisoformat(stored["next_run_at"])
        assert rebased == datetime(
            2026, 8, 17, 14, 30, tzinfo=ZoneInfo("Europe/Berlin")
        ), "cron wall-clock intent (14:30) must be preserved in the new zone"

    def test_rebase_is_idempotent(self, tmp_cron_dir, monkeypatch):
        job = a_job()
        job["schedule"]["tz"] = "Asia/Shanghai"
        save_jobs([job])

        use_zone(monkeypatch, "Europe/Berlin", datetime(2026, 8, 17, 8, 0))
        get_due_jobs()
        first = get_job("job-1")["next_run_at"]

        use_zone(monkeypatch, "Europe/Berlin", datetime(2026, 8, 17, 8, 1))
        get_due_jobs()
        assert get_job("job-1")["next_run_at"] == first

    def test_zone_change_never_drops_a_due_run_in_the_same_zone(
        self, tmp_cron_dir, monkeypatch
    ):
        """No rebase without an actual zone change: a due job still fires."""
        job = a_job()
        job["schedule"]["tz"] = "Asia/Shanghai"
        job["next_run_at"] = "2026-08-17T14:30:00+08:00"
        save_jobs([job])

        use_zone(monkeypatch, "Asia/Shanghai", datetime(2026, 8, 17, 14, 30, 20))
        assert [j["id"] for j in get_due_jobs()] == ["job-1"]


# =============================================================================
# 5. DST must not look like a migration
# =============================================================================

class TestDaylightSaving:
    def test_dst_offset_change_does_not_rebase_a_stamped_job(
        self, tmp_cron_dir, monkeypatch
    ):
        """Berlin runs +01:00 in winter and +02:00 in summer.

        The pre-fix heuristic keyed on that offset delta, so a DST boundary
        looked exactly like a timezone migration and silently skipped the
        pending occurrence. Zone identity does not change across DST, so the
        stamped job must be left alone and simply fire.
        """
        berlin = ZoneInfo("Europe/Berlin")
        job = a_job(
            schedule={"kind": "cron", "expr": "30 2 * * *", "tz": "Europe/Berlin"},
            # Persisted in winter time (+01:00), due right after the spring
            # forward when the live offset is +02:00.
            next_run_at=datetime(2026, 3, 29, 3, 30, tzinfo=berlin).isoformat(),
            last_run_at=datetime(2026, 3, 28, 2, 30, tzinfo=berlin).isoformat(),
        )
        save_jobs([job])

        use_zone(monkeypatch, "Europe/Berlin", datetime(2026, 3, 29, 3, 30, 15))
        assert [j["id"] for j in get_due_jobs()] == ["job-1"]

    @pytest.mark.parametrize(
        "tz_name,start_utc",
        [
            ("Europe/Berlin", datetime(2026, 3, 27, tzinfo=timezone.utc)),
            ("Europe/Berlin", datetime(2026, 10, 23, tzinfo=timezone.utc)),
            ("Pacific/Auckland", datetime(2026, 9, 25, tzinfo=timezone.utc)),
            ("Pacific/Auckland", datetime(2026, 4, 3, tzinfo=timezone.utc)),
            ("America/Los_Angeles", datetime(2026, 3, 6, tzinfo=timezone.utc)),
        ],
    )
    def test_daily_job_fires_once_a_day_across_a_dst_boundary(
        self, tz_name, start_utc, tmp_cron_dir, monkeypatch
    ):
        """Tick a virtual clock across the transition and count the fires.

        Handing croniter a zone-aware base makes it walk a *fixed* UTC offset,
        so a daily job gained an extra fire on a spring-forward day and lost one
        on a fall-back day — with no timezone change involved at all.
        """
        zone = ZoneInfo(tz_name)
        clock = {"now": start_utc}
        monkeypatch.setenv("HERMES_TIMEZONE", tz_name)
        monkeypatch.setattr(
            "cron.jobs._hermes_now", lambda: clock["now"].astimezone(zone)
        )

        save_jobs([
            a_job(
                schedule={"kind": "cron", "expr": "0 9 * * *", "tz": tz_name},
                next_run_at=None,
                last_run_at=start_utc.astimezone(zone).isoformat(),
            )
        ])

        fires = []
        end = start_utc + timedelta(days=5)
        while clock["now"] <= end:
            for job in get_due_jobs():
                fires.append(clock["now"])
                from cron.jobs import mark_job_run

                mark_job_run(job["id"], True)
            clock["now"] += timedelta(minutes=1)

        assert len(fires) == 5, f"{tz_name}: expected 5 daily runs, got {len(fires)}"
        for fired_at in fires:
            local = fired_at.astimezone(zone)
            assert (local.hour, local.minute) == (9, 0), (
                f"{tz_name}: fired at {local.isoformat()}, not 09:00 local"
            )
        assert len({f.astimezone(zone).date() for f in fires}) == 5

    def test_next_run_after_dst_keeps_wall_clock(self, monkeypatch):
        """09:00 local stays 09:00 local across the spring-forward boundary."""
        use_zone(monkeypatch, "Europe/Berlin", datetime(2026, 3, 28, 9, 1))
        schedule = {"kind": "cron", "expr": "0 9 * * *", "tz": "Europe/Berlin"}
        nxt = datetime.fromisoformat(
            compute_next_run(schedule, last_run_at="2026-03-28T09:00:00+01:00")
        )
        assert (nxt.hour, nxt.minute) == (9, 0)
        assert nxt.utcoffset() == timedelta(hours=2)  # summer time


# =============================================================================
# 6. i18n / offset matrix — every configured zone must round-trip
# =============================================================================

ZONE_MATRIX = [
    "UTC",
    "Asia/Shanghai",        # +08:00, no DST
    "Asia/Kolkata",         # +05:30, half-hour offset
    "Asia/Kathmandu",       # +05:45, quarter-hour offset
    "Australia/Eucla",      # +08:45
    "Pacific/Kiritimati",   # +14:00, max positive
    "Pacific/Niue",         # -11:00
    "America/St_Johns",     # -03:30 / -02:30 with DST
    "Europe/Berlin",        # northern DST
    "America/Sao_Paulo",    # southern hemisphere
    "Pacific/Auckland",     # southern DST, +12/+13
    "Africa/Casablanca",    # Ramadan-shifted DST
    "America/Los_Angeles",
]


class TestZoneMatrix:
    @pytest.mark.parametrize("tz_name", ZONE_MATRIX)
    def test_wall_clock_intent_survives_a_utc_reader(self, tz_name, monkeypatch):
        """``30 14 * * *`` means 14:30 in the job's zone for every zone.

        Computed from a UTC-pinned process, which is precisely the reader that
        used to reinterpret the stored value.
        """
        zone = ZoneInfo(tz_name)
        last_run = datetime(2026, 8, 16, 14, 30, tzinfo=zone)

        use_server_local_utc(monkeypatch, datetime(2026, 8, 16, 12, 0))
        nxt = datetime.fromisoformat(
            compute_next_run(
                {"kind": "cron", "expr": "30 14 * * *", "tz": tz_name},
                last_run_at=last_run.isoformat(),
            )
        )
        assert nxt.astimezone(zone).hour == 14
        assert nxt.astimezone(zone).minute == 30
        assert nxt > last_run
        assert (nxt - last_run) <= timedelta(hours=25)

    @pytest.mark.parametrize("tz_name", ZONE_MATRIX)
    def test_due_scan_never_fires_before_the_stored_instant(
        self, tz_name, tmp_cron_dir, monkeypatch
    ):
        """Across every zone, a UTC gateway must respect the stored instant."""
        zone = ZoneInfo(tz_name)
        fire_at = datetime(2026, 8, 17, 14, 30, tzinfo=zone)

        job = a_job(
            schedule={"kind": "cron", "expr": "30 14 * * *", "tz": tz_name},
            next_run_at=fire_at.isoformat(),
            last_run_at=(fire_at - timedelta(days=1)).isoformat(),
        )
        save_jobs([job])

        one_minute_early = (fire_at - timedelta(minutes=1)).astimezone(
            ZoneInfo("UTC")
        )
        use_server_local_utc(monkeypatch, one_minute_early.replace(tzinfo=None))
        assert get_due_jobs() == [], f"{tz_name}: fired before its instant"

        just_due = (fire_at + timedelta(seconds=10)).astimezone(ZoneInfo("UTC"))
        use_server_local_utc(monkeypatch, just_due.replace(tzinfo=None))
        assert [j["id"] for j in get_due_jobs()] == ["job-1"], f"{tz_name}: never fired"


# =============================================================================
# 7. Legacy jobs keep their pre-fix behaviour and get adopted
# =============================================================================

class TestLegacyJobs:
    def test_unstamped_job_is_adopted_by_the_configured_zone(
        self, tmp_cron_dir, monkeypatch
    ):
        job = a_job()
        job["schedule"].pop("tz", None)
        job["next_run_at"] = "2026-08-18T14:30:00+08:00"
        save_jobs([job])

        use_zone(monkeypatch, "Asia/Shanghai", datetime(2026, 8, 17, 15, 0))
        get_due_jobs()
        assert get_job("job-1")["schedule"]["tz"] == "Asia/Shanghai"

    def test_no_configured_zone_leaves_schedules_unstamped(
        self, tmp_cron_dir, monkeypatch
    ):
        """Server-local installs keep the legacy offset-heuristic path."""
        import hermes_time

        monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
        monkeypatch.setattr(hermes_time, "get_timezone_name", lambda: "")
        now = datetime(2026, 8, 17, 15, 0, tzinfo=timezone(timedelta(hours=8)))
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        job = a_job()
        job["schedule"].pop("tz", None)
        job["next_run_at"] = "2026-08-18T14:30:00+08:00"
        save_jobs([job])

        get_due_jobs()
        assert "tz" not in get_job("job-1")["schedule"]
