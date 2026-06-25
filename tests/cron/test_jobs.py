"""Tests for cron/jobs.py — schedule parsing, job CRUD, and due-job detection."""

import threading
import pytest
from datetime import datetime, timedelta, timezone

from cron.jobs import (
    parse_duration,
    parse_schedule,
    compute_next_run,
    create_job,
    load_jobs,
    save_jobs,
    get_job,
    list_jobs,
    update_job,
    pause_job,
    resume_job,
    remove_job,
    mark_job_run,
    advance_next_run,
    claim_dispatch,
    claim_job_for_fire,
    heartbeat_run_claim,
    get_due_jobs,
    save_job_output,
    _hermes_now,
)


# =========================================================================
# parse_duration
# =========================================================================

class TestParseDuration:
    def test_minutes(self):
        assert parse_duration("30m") == 30
        assert parse_duration("1min") == 1
        assert parse_duration("5mins") == 5
        assert parse_duration("10minute") == 10
        assert parse_duration("120minutes") == 120


    def test_bare_units_default_to_one(self):
        assert parse_duration("hour") == 60
        assert parse_duration("hr") == 60
        assert parse_duration("h") == 60
        assert parse_duration("minute") == 1
        assert parse_duration("m") == 1
        assert parse_duration("day") == 1440
        assert parse_duration("d") == 1440

    def test_every_bare_unit_schedule(self):
        result = parse_schedule("every hour")
        assert result["kind"] == "interval"
        assert result["minutes"] == 60
        result = parse_schedule("every day")
        assert result["kind"] == "interval"
        assert result["minutes"] == 1440

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_duration("abc")
        with pytest.raises(ValueError):
            parse_duration("hourx")
        with pytest.raises(ValueError):
            parse_duration("30x")
        with pytest.raises(ValueError):
            parse_duration("")
        with pytest.raises(ValueError):
            parse_duration("m30")


# =========================================================================
# parse_schedule
# =========================================================================

class TestParseSchedule:
    def test_duration_becomes_once(self):
        result = parse_schedule("30m")
        assert result["kind"] == "once"
        assert "run_at" in result
        # run_at should be a valid ISO timestamp string ~30 minutes from now
        run_at_str = result["run_at"]
        assert isinstance(run_at_str, str)
        run_at = datetime.fromisoformat(run_at_str)
        now = datetime.now().astimezone()
        assert run_at > now
        assert run_at < now + timedelta(minutes=31)

    def test_every_becomes_interval(self):
        result = parse_schedule("every 2h")
        assert result["kind"] == "interval"
        assert result["minutes"] == 120


    # ---- Natural-language weekday/daily phrases → cron (issue: documented
    # "every monday 9am" format was rejected because the "every" branch only
    # accepted durations). ----

    def test_every_weekday_time_becomes_cron(self):
        pytest.importorskip("croniter")
        result = parse_schedule("every monday 9am")
        assert result["kind"] == "cron"
        assert result["expr"] == "0 9 * * 1"
        # Display preserves the user's natural phrasing.
        assert result["display"] == "every monday 9am"

    def test_every_sunday_maps_to_zero(self):
        pytest.importorskip("croniter")
        # Cron weekday numbering puts Sunday at 0.
        assert parse_schedule("every sunday 9am")["expr"] == "0 9 * * 0"

    def test_every_weekday_abbreviations(self):
        pytest.importorskip("croniter")
        assert parse_schedule("every mon 9am")["expr"] == "0 9 * * 1"
        assert parse_schedule("every fri 5pm")["expr"] == "0 17 * * 5"

    def test_every_day_keyword_is_daily(self):
        pytest.importorskip("croniter")
        assert parse_schedule("every day at 9am")["expr"] == "0 9 * * *"
        assert parse_schedule("every day 7am")["expr"] == "0 7 * * *"

    def test_every_weekday_keyword_is_business_days(self):
        pytest.importorskip("croniter")
        assert parse_schedule("every weekday at 9am")["expr"] == "0 9 * * 1-5"

    def test_every_weekend_keyword(self):
        pytest.importorskip("croniter")
        assert parse_schedule("every weekend at 10am")["expr"] == "0 10 * * 0,6"

    def test_every_time_formats(self):
        pytest.importorskip("croniter")
        # 24-hour, explicit minutes, noon/midnight, bare hour.
        assert parse_schedule("every monday 14:30")["expr"] == "30 14 * * 1"
        assert parse_schedule("every monday 9:05am")["expr"] == "5 9 * * 1"
        assert parse_schedule("every monday noon")["expr"] == "0 12 * * 1"
        assert parse_schedule("every monday midnight")["expr"] == "0 0 * * 1"
        assert parse_schedule("every monday at 7")["expr"] == "0 7 * * 1"

    def test_every_12_hour_boundaries(self):
        pytest.importorskip("croniter")
        # 12am is midnight (00:00), 12pm is noon (12:00).
        assert parse_schedule("every monday 12am")["expr"] == "0 0 * * 1"
        assert parse_schedule("every monday 12pm")["expr"] == "0 12 * * 1"

    def test_every_weekday_time_is_case_insensitive(self):
        pytest.importorskip("croniter")
        assert parse_schedule("Every Monday 9AM")["expr"] == "0 9 * * 1"

    def test_every_weekday_schedule_computes_next_run(self):
        pytest.importorskip("croniter")
        # End-to-end: the produced cron schedule is usable by compute_next_run.
        schedule = parse_schedule("every monday 9am")
        next_run = compute_next_run(schedule)
        assert next_run is not None
        dt = datetime.fromisoformat(next_run)
        assert dt.weekday() == 0  # Python: Monday == 0
        assert (dt.hour, dt.minute) == (9, 0)

    def test_every_duration_still_interval(self):
        # The interval path must keep working unchanged.
        assert parse_schedule("every 30m")["kind"] == "interval"
        assert parse_schedule("every 1d")["minutes"] == 1440

    def test_every_weekday_without_time_raises(self):
        # A weekday with no time is ambiguous — reject rather than guess.
        with pytest.raises(ValueError):
            parse_schedule("every monday")

    def test_every_invalid_time_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("every monday 25am")
        with pytest.raises(ValueError):
            parse_schedule("every monday 9pm pizza")

    def test_cron_expression(self):
        pytest.importorskip("croniter")
        result = parse_schedule("0 9 * * *")
        assert result["kind"] == "cron"
        assert result["expr"] == "0 9 * * *"

    def test_cron_named_weekdays_and_months(self):
        # Named months/weekdays (and ranges/lists) are valid cron and must
        # route to croniter, not be rejected as "Invalid schedule".
        pytest.importorskip("croniter")
        for expr in (
            "0 9 * * MON",
            "*/15 9-17 * * MON-FRI",
            "0 9 1 JAN *",
            "0 9 * * MON,WED,FRI",
        ):
            result = parse_schedule(expr)
            assert result["kind"] == "cron", expr
            assert result["expr"] == expr

    def test_invalid_named_cron_still_rejected(self):
        pytest.importorskip("croniter")
        with pytest.raises(ValueError):
            parse_schedule("0 9 * * FUNDAY")

    def test_iso_timestamp(self):
        result = parse_schedule("2030-01-15T14:00:00")
        assert result["kind"] == "once"
        assert "2030-01-15" in result["run_at"]


    def test_naive_iso_anchors_to_configured_tz_not_server_local(self, monkeypatch):
        """A naive ISO timestamp must be interpreted in the CONFIGURED Hermes
        timezone, NOT the server's local timezone (#51021).

        Regression: when the configured zone differs from the server's local
        zone (common on cloud hosts running UTC), parse_schedule used
        ``dt.astimezone()`` (server-local), baking in the wrong offset. The
        due-check compares against ``_hermes_now()`` (configured zone), so the
        stored instant landed hours off the user's wall-clock intent — far
        enough that one-shots never became due. This asserts the parsed offset
        matches the configured-now offset, the invariant that keeps the stored
        instant on the same clock the scheduler checks against.
        """
        configured_now = datetime(2026, 6, 22, 20, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: configured_now)

        result = parse_schedule("2026-06-22T20:07:00")  # naive, user wall-clock

        assert result["kind"] == "once"
        parsed = datetime.fromisoformat(result["run_at"])
        assert parsed.utcoffset() == configured_now.utcoffset()
        # Same wall-clock the user typed, on the configured clock.
        assert parsed.replace(tzinfo=None) == datetime(2026, 6, 22, 20, 7, 0)


# =========================================================================
# Timezone-divergence regression (#51021)
# =========================================================================

class TestNaiveScheduleTimezoneDivergence:
    """End-to-end: a one-shot created with a naive recent-past timestamp must
    become due even when the configured Hermes timezone differs from the
    server's local timezone. Before #51021 the naive value was anchored to
    server-local, so the job never fired."""

    def test_recent_past_oneshot_is_due_under_diverging_tz(self, tmp_cron_dir, monkeypatch):
        # Configured zone: a fixed +05:30 offset. The server's actual local
        # zone is irrelevant to the parse now — that is the whole point.
        configured = timezone(timedelta(hours=5, minutes=30))
        now = datetime(2026, 6, 22, 20, 7, 30, tzinfo=configured)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        # 30s ago in the configured wall clock, supplied as a NAIVE string.
        naive_str = (now - timedelta(seconds=30)).replace(tzinfo=None).isoformat()
        job = create_job(prompt="test message", schedule=naive_str, deliver="local")

        due = get_due_jobs()
        assert any(d["id"] == job["id"] for d in due), (
            f"one-shot should be due; next_run_at={job['next_run_at']}"
        )


# =========================================================================
# compute_next_run
# =========================================================================

class TestComputeNextRun:
    def test_once_future_returns_time(self):
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        schedule = {"kind": "once", "run_at": future}
        assert compute_next_run(schedule) == future

    def test_once_recent_past_within_grace_returns_time(self, monkeypatch):
        now = datetime(2026, 3, 18, 4, 22, 3, tzinfo=timezone.utc)
        run_at = "2026-03-18T04:22:00+00:00"
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "once", "run_at": run_at}

        assert compute_next_run(schedule) == run_at


    def test_once_with_last_run_returns_none_even_within_grace(self, monkeypatch):
        now = datetime(2026, 3, 18, 4, 22, 3, tzinfo=timezone.utc)
        run_at = "2026-03-18T04:22:00+00:00"
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "once", "run_at": run_at}

        assert compute_next_run(schedule, last_run_at=now.isoformat()) is None

    def test_interval_first_run(self):
        schedule = {"kind": "interval", "minutes": 60}
        result = compute_next_run(schedule)
        next_dt = datetime.fromisoformat(result)
        # Should be ~60 minutes from now
        assert next_dt > datetime.now().astimezone() + timedelta(minutes=59)


# =========================================================================
# Job CRUD (with tmp file storage)
# =========================================================================

@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestJobCRUD:
    def test_cjk_and_emoji_round_trip_readable_in_jobs_json(self, tmp_cron_dir):
        """CJK/emoji job text must round-trip AND stay human-readable on disk.

        With json.dump's default ensure_ascii=True, every non-ASCII char in
        jobs.json is written as \\uXXXX escapes, which users reported as
        unreadable garbage when inspecting their job store (#52302, #29754).
        ensure_ascii=False + the existing encoding="utf-8" writer keeps the
        text literal; the utf-8-sig reader must parse it back identically.
        """
        name = "日次レポート 🎉 café"
        job = create_job(prompt=f"Summarize {name}", schedule="30m", name=name)

        # Round-trip through save/load is lossless.
        fetched = get_job(job["id"])
        assert fetched["name"] == name
        assert name in fetched["prompt"]

        # On-disk representation is literal UTF-8, not \uXXXX escapes.
        from cron.jobs import JOBS_FILE
        raw = JOBS_FILE.read_text(encoding="utf-8")
        assert "日次レポート" in raw
        assert "🎉" in raw
        assert "\\u65e5" not in raw

    def test_create_and_get(self, tmp_cron_dir):
        job = create_job(prompt="Check server status", schedule="30m")
        assert job["id"]
        assert job["prompt"] == "Check server status"
        assert job["enabled"] is True
        assert job["schedule"]["kind"] == "once"

        fetched = get_job(job["id"])
        assert fetched is not None
        assert fetched["prompt"] == "Check server status"

    def test_list_jobs(self, tmp_cron_dir):
        create_job(prompt="Job 1", schedule="every 1h")
        create_job(prompt="Job 2", schedule="every 2h")
        jobs = list_jobs()
        assert len(jobs) == 2


    def test_remove_job(self, tmp_cron_dir):
        job = create_job(prompt="Temp job", schedule="30m")
        assert remove_job(job["id"]) is True
        assert get_job(job["id"]) is None


    def test_auto_repeat_for_once(self, tmp_cron_dir):
        job = create_job(prompt="One-shot", schedule="1h")
        assert job["repeat"]["times"] == 1

    def test_rejects_stale_past_one_shot_at_creation(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 3, 18, 4, 30, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        stale = (now - timedelta(minutes=5)).isoformat()

        with pytest.raises(ValueError, match="past and cannot be scheduled"):
            create_job(prompt="Too late", schedule=stale)

        assert load_jobs() == []


    def test_default_delivery_origin(self, tmp_cron_dir):
        job = create_job(
            prompt="Test", schedule="30m",
            origin={"platform": "telegram", "chat_id": "123"},
        )
        assert job["deliver"] == "origin"


class TestUpdateJob:
    def test_update_name(self, tmp_cron_dir):
        job = create_job(prompt="Check server status", schedule="every 1h", name="Old Name")
        assert job["name"] == "Old Name"
        updated = update_job(job["id"], {"name": "New Name"})
        assert updated is not None
        assert isinstance(updated, dict)
        assert updated["name"] == "New Name"
        # Verify other fields are preserved
        assert updated["prompt"] == "Check server status"
        assert updated["id"] == job["id"]
        assert updated["schedule"] == job["schedule"]
        # Verify persisted to disk
        fetched = get_job(job["id"])
        assert fetched["name"] == "New Name"


class TestPauseResumeJob:
    def test_pause_sets_state(self, tmp_cron_dir):
        job = create_job(prompt="Pause me", schedule="every 1h")
        paused = pause_job(job["id"], reason="user paused")
        assert paused is not None
        assert paused["enabled"] is False
        assert paused["state"] == "paused"
        assert paused["paused_reason"] == "user paused"
        assert paused.get("paused_at")

    def test_pause_is_authoritative_due_jobs_do_not_fire(self, tmp_cron_dir):
        """Behavioural invariant: after pause, a past-due job must not be due.

        Checks that last_run_at cannot advance via the scheduler path — not
        merely that pause() returned success. Regression for the 07-30 outage
        where state=paused coexisted with enabled=true and jobs kept firing.
        """
        job = create_job(prompt="Must not fire while paused", schedule="every 1h")
        past = (_hermes_now() - timedelta(hours=2)).isoformat()
        # Force the job overdue, then pause.
        updated = update_job(job["id"], {"next_run_at": past})
        assert updated["enabled"] is True
        assert job["id"] in {j["id"] for j in get_due_jobs()}

        paused = pause_job(job["id"], reason="outage freeze")
        assert paused["enabled"] is False
        assert paused["state"] == "paused"
        assert paused.get("paused_at")
        # Scheduler-honoured flag and pause markers must never contradict.
        assert not (paused.get("enabled") and paused.get("paused_at"))

        due_ids = {j["id"] for j in get_due_jobs()}
        assert job["id"] not in due_ids

        before = get_job(job["id"])
        assert before["last_run_at"] is None or before["last_run_at"] == job.get("last_run_at")
        # claim path also closed
        assert claim_job_for_fire(job["id"]) is False
        after = get_job(job["id"])
        assert after["last_run_at"] == before.get("last_run_at")
        assert after["enabled"] is False

    def test_contradictory_half_pause_self_disables_and_does_not_fire(self, tmp_cron_dir):
        """enabled=true + paused_at must not fire; scan heals enabled=false."""
        now = _hermes_now()
        job = {
            "id": "half-paused-1",
            "name": "half-paused",
            "prompt": "should never run",
            "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
            "schedule_display": "every 5m",
            "repeat": {"times": None, "completed": 0},
            # The contradiction from the 07-30 outage:
            "enabled": True,
            "state": "paused",
            "paused_at": (now - timedelta(hours=20)).isoformat(),
            "paused_reason": "operator thought this was frozen",
            "next_run_at": (now - timedelta(hours=1)).isoformat(),
            "last_run_at": None,
            "last_status": None,
            "last_error": None,
            "last_delivery_error": None,
            "created_at": (now - timedelta(days=1)).isoformat(),
            "deliver": "local",
        }
        save_jobs([job])

        # Display must NOT say paused while enabled (honest list).
        from cron.jobs import effective_job_state, list_jobs

        assert effective_job_state(job) == "scheduled"
        listed = {j["id"]: j for j in list_jobs(include_disabled=True)}
        # Honest list: enabled=true half-pause must not render as paused.
        assert listed["half-paused-1"]["enabled"] is True
        assert listed["half-paused-1"]["state"] != "paused"

        assert claim_job_for_fire("half-paused-1") is False
        due = get_due_jobs()
        assert "half-paused-1" not in {j["id"] for j in due}

        healed = get_job("half-paused-1")
        assert healed is not None
        assert healed["enabled"] is False
        assert healed["state"] == "paused"
        assert healed.get("paused_at")
        # Still not due after heal
        assert "half-paused-1" not in {j["id"] for j in get_due_jobs()}

    def test_resume_rejects_past_oneshot(self, tmp_cron_dir, monkeypatch):
        """Resuming a paused one-shot whose time is now in the past must raise
        ValueError — the revived job would silently never fire."""
        now = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        # Create directly — bypass create_job's past-oneshot guard so we can
        # test the resume path independently.
        job = {
            "id": "test-resume-past",
            "name": "test-resume-past",
            "prompt": "Past one-shot",
            "schedule": {"kind": "once", "run_at": (now - timedelta(minutes=5)).isoformat(), "display": "once"},
            "repeat": {"times": 1, "completed": 0},
            "enabled": False,
            "state": "paused",
            "paused_at": now.isoformat(),
            "paused_reason": "test",
            "next_run_at": None,
            "last_run_at": None,
            "last_status": None,
            "last_error": None,
            "last_delivery_error": None,
            "created_at": (now - timedelta(hours=1)).isoformat(),
            "deliver": "local",
        }
        save_jobs([job])
        with pytest.raises(ValueError, match="in the past"):
            resume_job("test-resume-past")


class TestResolveJobRef:
    """Name-based job lookup for CLI/tool callers (PR #2627, @buntingszn)."""

    def test_resolve_by_exact_id(self, tmp_cron_dir):
        from cron.jobs import resolve_job_ref

        job = create_job(prompt="A", schedule="1h", name="alpha")
        assert resolve_job_ref(job["id"])["id"] == job["id"]


    def test_mutations_refuse_ambiguous_name(self, tmp_cron_dir):
        """pause/resume/trigger/remove must refuse to act on an ambiguous name."""
        from cron.jobs import AmbiguousJobReference, trigger_job

        create_job(prompt="A", schedule="1h", name="dup")
        create_job(prompt="B", schedule="1h", name="dup")
        for fn in (pause_job, resume_job, trigger_job):
            with pytest.raises(AmbiguousJobReference):
                fn("dup")
        with pytest.raises(AmbiguousJobReference):
            remove_job("dup")


class TestMarkJobRun:
    def test_increments_completed(self, tmp_cron_dir):
        job = create_job(prompt="Test", schedule="every 1h")
        mark_job_run(job["id"], success=True)
        updated = get_job(job["id"])
        assert updated["repeat"]["completed"] == 1
        assert updated["last_status"] == "ok"

    def test_repeat_limit_retains_completed_record(self, tmp_cron_dir):
        """A finished one-shot must stay inspectable, not vanish from the store."""
        job = create_job(prompt="Once", schedule="30m", repeat=1)
        mark_job_run(job["id"], success=True)
        updated = get_job(job["id"])
        assert updated is not None, "completed one-shot was deleted from jobs.json"
        assert updated["state"] == "completed"
        assert updated["enabled"] is False
        assert updated["next_run_at"] is None
        assert updated["last_status"] == "ok"

    def test_repeat_limit_retains_delivery_error(self, tmp_cron_dir):
        """A one-shot whose delivery failed must keep the error on its record."""
        job = create_job(prompt="Once", schedule="30m", repeat=1)
        mark_job_run(
            job["id"], success=True,
            delivery_error="platform 'telegram' not configured",
        )
        updated = get_job(job["id"])
        assert updated is not None
        assert updated["state"] == "completed"
        assert updated["last_delivery_error"] == "platform 'telegram' not configured"

    def test_completed_oneshot_visible_in_list(self, tmp_cron_dir):
        """list_jobs(include_disabled=True) surfaces the completed record."""
        job = create_job(prompt="Once", schedule="30m", repeat=1)
        mark_job_run(job["id"], success=True, delivery_error="send failed: 502")
        listed = {j["id"]: j for j in list_jobs(include_disabled=True)}
        assert job["id"] in listed
        assert listed[job["id"]]["state"] == "completed"
        assert listed[job["id"]]["last_delivery_error"] == "send failed: 502"
        # Default (enabled-only) listing hides it, matching paused/disabled jobs.
        assert job["id"] not in {j["id"] for j in list_jobs()}

    def test_completed_oneshot_not_due(self, tmp_cron_dir):
        """A retained completed one-shot must never be dispatched again."""
        job = create_job(prompt="Once", schedule="30m", repeat=1)
        mark_job_run(job["id"], success=True)
        assert job["id"] not in {j["id"] for j in get_due_jobs()}


    def test_error_status(self, tmp_cron_dir):
        job = create_job(prompt="Fail", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="timeout")
        updated = get_job(job["id"])
        assert updated["last_status"] == "error"
        assert updated["last_error"] == "timeout"

    def test_delivery_error_tracked_separately(self, tmp_cron_dir):
        """Agent succeeds but delivery fails — both tracked independently."""
        job = create_job(prompt="Report", schedule="every 1h")
        mark_job_run(job["id"], success=True, delivery_error="platform 'telegram' not configured")
        updated = get_job(job["id"])
        assert updated["last_status"] == "ok"
        assert updated["last_error"] is None
        assert updated["last_delivery_error"] == "platform 'telegram' not configured"

    def test_failure_streak_increments_and_resets(self, tmp_cron_dir):
        """failure_streak counts consecutive agent failures; success resets."""
        job = create_job(prompt="Flaky", schedule="every 1h")
        assert get_job(job["id"])["failure_streak"] == 0
        mark_job_run(job["id"], success=False, error="timeout")
        mark_job_run(job["id"], success=False, error="timeout")
        assert get_job(job["id"])["failure_streak"] == 2
        mark_job_run(job["id"], success=True)
        assert get_job(job["id"])["failure_streak"] == 0

    def test_failure_streak_ignores_delivery_errors(self, tmp_cron_dir):
        """A successful run with a delivery error must not count as a failure."""
        job = create_job(prompt="Report", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="timeout")
        mark_job_run(job["id"], success=True, delivery_error="send failed: 502")
        assert get_job(job["id"])["failure_streak"] == 0

    def test_failure_streak_backcompat_missing_field(self, tmp_cron_dir):
        """Jobs persisted before the field existed increment from 0."""
        job = create_job(prompt="Old", schedule="every 1h")
        # Simulate a pre-field record on disk.
        jobs = load_jobs()
        for j in jobs:
            j.pop("failure_streak", None)
        save_jobs(jobs)
        mark_job_run(job["id"], success=False, error="boom")
        assert get_job(job["id"])["failure_streak"] == 1


    def test_recurring_cron_not_disabled_when_croniter_missing(self, tmp_cron_dir, monkeypatch):
        """Regression test for issue #16265.

        If the gateway runs in an env where `croniter` went missing after a
        recurring cron job was persisted, `compute_next_run()` returns None.
        `mark_job_run()` must NOT treat that as terminal completion — the job
        has to stay enabled with state=error so the user notices, rather than
        silently flipping to enabled=false, state=completed.
        """
        pytest.importorskip("croniter")  # need it to create the job
        job = create_job(prompt="Recurring", schedule="0 7,15,23 * * *")
        assert job["schedule"]["kind"] == "cron"

        # Simulate the runtime env having lost croniter between job creation
        # and this run.
        monkeypatch.setattr("cron.jobs.HAS_CRONITER", False)

        mark_job_run(job["id"], success=True)

        updated = get_job(job["id"])
        assert updated is not None, "recurring cron job was deleted"
        assert updated["enabled"] is True, (
            "recurring cron job was disabled despite croniter-missing being "
            "a runtime dep issue, not a terminal completion"
        )
        assert updated["state"] == "error"
        assert updated["state"] != "completed"
        assert updated["next_run_at"] is None
        assert updated["last_error"]
        assert "croniter" in updated["last_error"].lower()


class TestAdvanceNextRun:
    """Tests for advance_next_run() — crash-safety for recurring jobs."""

    def test_advances_interval_job(self, tmp_cron_dir):
        """Interval jobs should have next_run_at bumped to the next future occurrence."""
        job = create_job(prompt="Recurring check", schedule="every 1h")
        # Force next_run_at to 5 minutes ago (i.e. the job is due)
        jobs = load_jobs()
        old_next = (datetime.now() - timedelta(minutes=5)).isoformat()
        jobs[0]["next_run_at"] = old_next
        save_jobs(jobs)

        result = advance_next_run(job["id"])
        assert result is True

        updated = get_job(job["id"])
        from cron.jobs import _ensure_aware, _hermes_now
        new_next_dt = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        assert new_next_dt > _hermes_now(), "next_run_at should be in the future after advance"


    def test_skips_oneshot_job(self, tmp_cron_dir):
        """One-shot jobs should NOT be advanced — they need to retry on restart."""
        job = create_job(prompt="Run once", schedule="30m")
        original_next = get_job(job["id"])["next_run_at"]

        result = advance_next_run(job["id"])
        assert result is False

        updated = get_job(job["id"])
        assert updated["next_run_at"] == original_next, "one-shot next_run_at should be unchanged"


    def test_crash_safety_scenario(self, tmp_cron_dir):
        """Simulate the crash-loop scenario: after advance, the job should NOT be due."""
        job = create_job(prompt="Crash test", schedule="every 1h")
        # Force next_run_at to 5 minutes ago (job is due)
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
        save_jobs(jobs)

        # Job should be due before advance
        due_before = get_due_jobs()
        assert len(due_before) == 1

        # Advance (simulating what tick() does before run_job)
        advance_next_run(job["id"])

        # Now the job should NOT be due (simulates restart after crash)
        due_after = get_due_jobs()
        assert len(due_after) == 0, "Job should not be due after advance_next_run"


class TestGetDueJobs:
    def test_past_due_within_window_returned(self, tmp_cron_dir):
        """Jobs within the dynamic grace window are still considered due (not stale).

        For an hourly job, grace = 30 min (half the period, clamped to [120s, 2h]).
        """
        job = create_job(prompt="Due now", schedule="every 1h")
        # Force next_run_at to 10 minutes ago (within the 30-min grace for hourly)
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=10)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 1
        assert due[0]["id"] == job["id"]

    def test_stale_past_due_runs_once_and_fast_forwards(self, tmp_cron_dir):
        """Recurring jobs past their grace window run once now and fast-forward next_run_at.

        For an hourly job, grace = 30 min. Setting 35 min late exceeds the window.
        The job should be returned as due (execute once) with next_run_at in the future.
        """
        job = create_job(prompt="Stale", schedule="every 1h")
        # Force next_run_at to 35 minutes ago (beyond the 30-min grace for hourly)
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=35)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        # Job is returned as due — execute once now instead of skipping
        assert len(due) == 1
        assert due[0]["id"] == job["id"]
        # next_run_at should be fast-forwarded to the future (accumulated slots skipped)
        updated = get_job(job["id"])
        from cron.jobs import _ensure_aware, _hermes_now
        next_dt = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        assert next_dt > _hermes_now()


    def test_idless_job_does_not_crash_or_block_sibling_jobs(self, tmp_cron_dir):
        """A job missing its 'id' key must not crash the tick or freeze siblings.

        Regression: jobs authored by a direct jobs.json edit (bypassing
        create_job) sometimes used the key 'job_id' instead of 'id'. The logging
        helpers evaluated ``job.get("name", job["id"])`` -- Python evaluates the
        default argument ``job["id"]`` eagerly, so an id-less job raised
        ``KeyError: 'id'`` mid-tick. That exception aborted
        ``_get_due_jobs_locked()`` BEFORE ``save_jobs()`` ran, so every healthy
        job's fast-forwarded next_run_at was computed in memory then discarded --
        the whole profile's scheduler froze in a per-minute loop.
        """
        healthy = create_job(prompt="Healthy", schedule="every 1h")

        jobs = load_jobs()
        # Push the healthy job beyond its grace window so the fast-forward path
        # (one of the id-less-crash sites) runs.
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=35)).isoformat()
        # A malformed record: no 'id' key, mirroring the real corruption.
        jobs.append({
            "name": "idless-job",
            "schedule": {"kind": "cron", "expr": "0 4 * * *"},
            "enabled": True,
            "no_agent": True,
            "next_run_at": None,
        })
        save_jobs(jobs)

        # Must not raise KeyError.
        due = get_due_jobs()

        # The healthy sibling is still discovered despite the malformed neighbor.
        assert any(d.get("id") == healthy["id"] for d in due)


    def test_long_execution_does_not_perpetually_defer(self, tmp_cron_dir, monkeypatch):
        """#33315: a recurring job whose runtime exceeds interval+grace must still
        run once when the tick comes back, not skip forever.

        Reproduces the production loop: a 5-min interval job whose previous run
        overran the interval, leaving next_run_at ~11 min in the past — beyond
        the 150s grace for a 5m interval. The job must be returned as due (run
        once) AND have next_run_at fast-forwarded (so accumulated missed slots
        don't all fire)."""
        from cron.jobs import _ensure_aware, _hermes_now
        job = create_job(prompt="Long job", schedule="every 5m")
        jobs = load_jobs()
        # 11 minutes ago: > grace (150s for a 5m interval) — the "still running" miss.
        stale = (_hermes_now() - timedelta(minutes=11)).isoformat()
        jobs[0]["next_run_at"] = stale
        jobs[0]["last_run_at"] = (_hermes_now() - timedelta(minutes=1)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert [j["id"] for j in due] == [job["id"]], "long-execution job was skipped (perpetual-defer bug)"
        # next_run_at fast-forwarded into the future (no burst of missed slots).
        nxt = _ensure_aware(datetime.fromisoformat(get_job(job["id"])["next_run_at"]))
        assert nxt > _hermes_now()


    def test_broken_recent_one_shot_without_next_run_is_recovered(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 3, 18, 4, 22, 30, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        run_at = "2026-03-18T04:22:00+00:00"
        save_jobs(
            [{
                "id": "oneshot-recover",
                "name": "Recover me",
                "prompt": "Word of the day",
                "schedule": {"kind": "once", "run_at": run_at, "display": "once at 2026-03-18 04:22"},
                "schedule_display": "once at 2026-03-18 04:22",
                "repeat": {"times": 1, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
                "created_at": "2026-03-18T04:21:00+00:00",
                "next_run_at": None,
                "last_run_at": None,
                "last_status": None,
                "last_error": None,
                "deliver": "local",
                "origin": None,
            }]
        )

        due = get_due_jobs()

        assert [job["id"] for job in due] == ["oneshot-recover"]
        # Recovery restores next_run_at to the original run time; the
        # cross-process double-exec guard (#59229) is a separate run_claim
        # stamped under the lock, not a next_run_at mutation.
        recovered = get_job("oneshot-recover")
        assert recovered["next_run_at"] == run_at
        assert recovered.get("run_claim") is not None
        assert recovered["run_claim"]["at"] == now.isoformat()


    def test_one_shot_not_redispatched_while_running(self, tmp_cron_dir, monkeypatch):
        """#59229: two concurrent schedulers must not double-execute a one-shot.

        Reproduces the reported failure with a job whose run OUTLIVES the tick
        interval (a ~2.5-min research prompt). Process A's tick returns it as
        due and stamps a run_claim; while A is still running, every later tick
        (process B, or A's own next tick) must see the fresh claim and skip —
        not just for one tick window but for the whole run.
        """
        from cron.jobs import _hermes_now
        t0 = _hermes_now()
        run_at = (t0 - timedelta(seconds=5)).isoformat()
        save_jobs([{
            "id": "long-oneshot", "name": "R", "prompt": "2.5min research",
            "schedule": {"kind": "once", "run_at": run_at},
            "next_run_at": run_at, "enabled": True, "state": "scheduled",
        }])

        # Process A tick: picks it up + claims it.
        dueA = get_due_jobs()
        assert [j["id"] for j in dueA] == ["long-oneshot"]
        assert get_job("long-oneshot").get("run_claim") is not None

        # Process B (and A's own subsequent ticks) while A is still running:
        # 28s later (the exact gap in the report) AND 61s later (past any
        # fixed +60s window) — both must skip.
        for gap in (28, 61, 130):
            monkeypatch.setattr("cron.jobs._hermes_now",
                                lambda t0=t0, g=gap: t0 + timedelta(seconds=g))
            assert get_due_jobs() == [], f"double-dispatched at +{gap}s"


    def test_run_claim_heartbeat_keeps_long_run_claimed_past_ttl(
        self, tmp_cron_dir, monkeypatch
    ):
        """#62002 cross-process leg: a heartbeat-refreshed claim never expires
        while the run is alive, so no other tick re-dispatches or stale-removes
        the job even when the run outlives the original TTL horizon."""
        monkeypatch.delenv("HERMES_CRON_TIMEOUT", raising=False)
        from cron.jobs import _hermes_now, _oneshot_run_claim_ttl_seconds
        ttl = _oneshot_run_claim_ttl_seconds()
        t0 = _hermes_now()
        run_at = (t0 - timedelta(seconds=5)).isoformat()
        save_jobs([{
            "id": "slowrun", "name": "R", "prompt": "x",
            "schedule": {"kind": "once", "run_at": run_at},
            "next_run_at": run_at, "enabled": True, "state": "scheduled",
            "repeat": {"times": 1, "completed": 0},
        }])

        # Tick claims + dispatches the job.
        assert [j["id"] for j in get_due_jobs()] == ["slowrun"]
        assert claim_dispatch("slowrun") is True

        # Mid-run heartbeat before the TTL horizon refreshes the claim.
        monkeypatch.setattr("cron.jobs._hermes_now",
                            lambda: t0 + timedelta(seconds=ttl - 60))
        owner = get_job("slowrun")["run_claim"]["by"]
        assert heartbeat_run_claim("slowrun", expected_owner=owner) is True

        # Past the ORIGINAL claim's TTL horizon: without the heartbeat this
        # tick would stale-remove the maxed one-shot; with it the claim is
        # fresh, so the job is skipped and the record survives.
        monkeypatch.setattr("cron.jobs._hermes_now",
                            lambda: t0 + timedelta(seconds=ttl + 10))
        assert get_due_jobs() == []
        assert get_job("slowrun") is not None

        # Run completes → outcome lands on a record that still exists
        # (times=1 reached, so mark_job_run retires the job as a terminal
        # completed record instead of deleting it).
        mark_job_run("slowrun", True)
        retired = get_job("slowrun")
        assert retired is not None
        assert retired["state"] == "completed"
        assert retired["enabled"] is False
        assert retired["last_status"] == "ok"


    def test_heartbeat_run_claim_rejects_replaced_owner(self, tmp_cron_dir):
        """A resumed stale runner must not keep a newer owner's claim alive."""
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        original_at = datetime.now(timezone.utc).isoformat()
        save_jobs([{
            "id": "reclaimed", "name": "R", "prompt": "x",
            "schedule": {"kind": "once", "run_at": future},
            "next_run_at": future, "enabled": True, "state": "scheduled",
            "run_claim": {"at": original_at, "by": "new-owner"},
        }])

        assert heartbeat_run_claim("reclaimed", expected_owner="old-owner") is False
        assert get_job("reclaimed")["run_claim"] == {
            "at": original_at,
            "by": "new-owner",
        }


class TestEnabledToolsets:
    def test_enabled_toolsets_stored(self, tmp_cron_dir):
        job = create_job(prompt="monitor", schedule="every 1h", enabled_toolsets=["web", "terminal"])
        assert job["enabled_toolsets"] == ["web", "terminal"]


class TestMarkJobRunConcurrency:
    """Regression tests for concurrent parallel job state writes.

    tick() dispatches multiple jobs to separate threads simultaneously.
    Without _jobs_file_lock protecting the load→modify→save cycle in
    mark_job_run(), concurrent writes can clobber each other's updates
    (last-writer-wins), leaving some jobs with stale last_status / last_run_at.
    """

    def test_three_concurrent_mark_job_run_no_overwrites(self, tmp_cron_dir):
        """Run mark_job_run() for 3 jobs in parallel threads; all must land correctly."""
        # Create 3 distinct recurring jobs
        job_a = create_job(prompt="Job A", schedule="every 1h")
        job_b = create_job(prompt="Job B", schedule="every 1h")
        job_c = create_job(prompt="Job C", schedule="every 1h")

        errors: list = []

        def run_mark(job_id: str, success: bool, error_msg=None):
            try:
                mark_job_run(job_id, success=success, error=error_msg)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        # Fire all three concurrently
        threads = [
            threading.Thread(target=run_mark, args=(job_a["id"], True)),
            threading.Thread(target=run_mark, args=(job_b["id"], False, "timeout")),
            threading.Thread(target=run_mark, args=(job_c["id"], True)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected exceptions in worker threads: {errors}"

        # Verify each job has the correct state — no overwrites
        a = get_job(job_a["id"])
        b = get_job(job_b["id"])
        c = get_job(job_c["id"])

        assert a is not None, "Job A was unexpectedly deleted"
        assert b is not None, "Job B was unexpectedly deleted"
        assert c is not None, "Job C was unexpectedly deleted"

        assert a["last_status"] == "ok", f"Job A last_status wrong: {a['last_status']}"
        assert a["last_run_at"] is not None, "Job A last_run_at not set"
        assert a["repeat"]["completed"] == 1, f"Job A completed count wrong: {a['repeat']['completed']}"

        assert b["last_status"] == "error", f"Job B last_status wrong: {b['last_status']}"
        assert b["last_error"] == "timeout", f"Job B last_error wrong: {b['last_error']}"
        assert b["last_run_at"] is not None, "Job B last_run_at not set"
        assert b["repeat"]["completed"] == 1, f"Job B completed count wrong: {b['repeat']['completed']}"

        assert c["last_status"] == "ok", f"Job C last_status wrong: {c['last_status']}"
        assert c["last_run_at"] is not None, "Job C last_run_at not set"
        assert c["repeat"]["completed"] == 1, f"Job C completed count wrong: {c['repeat']['completed']}"


class TestBadNextRunAtRecovery:
    """Regression: malformed next_run_at must not crash the due scan or starve siblings.

    Mirrors the id-less and non-dict-schedule patterns: a single bad persisted
    record in jobs.json must not abort _get_due_jobs_locked before save.
    """

    def test_bad_next_run_at_does_not_crash_or_block_sibling_jobs(self, tmp_cron_dir):
        """One job with unparseable next_run_at + one healthy due sibling.

        get_due_jobs must succeed and return the healthy job; the bad record
        must be repaired (next_run_at cleared so recovery can set a sane value).
        """
        from datetime import timezone, timedelta as td
        now = datetime.now(timezone.utc)
        past = (now - td(seconds=30)).isoformat()
        future = (now + td(days=1)).isoformat()

        # Bad record: next_run_at is not a valid ISO string (e.g. from hand-edit or corruption)
        # Healthy sibling is past due with good schedule.
        bad_job = {
            "id": "bad-next",
            "schedule": {"kind": "interval", "minutes": 60},
            "next_run_at": "not-a-valid-iso-timestamp!!!",
            "enabled": True,
            "created_at": past,
        }
        good_job = {
            "id": "good-sibling",
            "schedule": {"kind": "interval", "minutes": 5},
            "next_run_at": past,
            "enabled": True,
            "created_at": past,
        }
        save_jobs([bad_job, good_job])

        # Must not raise
        due = get_due_jobs()

        # The healthy job must still be returned
        ids = [j["id"] for j in due]
        assert "good-sibling" in ids, f"healthy sibling missing from due jobs: {ids}"
        assert "bad-next" not in ids  # bad one may be repaired and/or not yet due after repair

        # Bad job should have been auto-repaired (next_run_at stripped or fixed)
        repaired = get_job("bad-next")
        assert repaired is not None
        nr = repaired.get("next_run_at")
        if nr is not None:
            # If still present it must now be parseable
            datetime.fromisoformat(nr)

        # Calling again must remain stable (no crash on re-scan)
        due2 = get_due_jobs()
        assert any(j["id"] == "good-sibling" for j in due2)


class TestPerJobScanContainment:
    """Structural guard: ANY per-job exception in the due scan must degrade to
    skipping that one job for the tick — never abort the scan and starve
    healthy siblings (the freeze class behind bad id / schedule / next_run_at).
    """

    def test_unforeseen_per_job_exception_does_not_starve_siblings(self, tmp_cron_dir):
        """Simulate a FUTURE malformed-field variant none of the shape
        normalizers repair, by making grace computation raise for one job
        only. The per-job guard must skip it and still return the sibling."""
        from datetime import timezone, timedelta as td
        from unittest.mock import patch as mock_patch

        now = datetime.now(timezone.utc)
        past = (now - td(seconds=30)).isoformat()

        poison = {
            "id": "poison",
            # minutes=7 tags this schedule so the patched helper can target it
            "schedule": {"kind": "interval", "minutes": 7},
            "next_run_at": past,
            "enabled": True,
            "created_at": past,
        }
        good = {
            "id": "good-sibling",
            "schedule": {"kind": "interval", "minutes": 5},
            "next_run_at": past,
            "enabled": True,
            "created_at": past,
        }
        save_jobs([poison, good])

        import cron.jobs as jobs_mod
        real_grace = jobs_mod._compute_grace_seconds

        def selective_grace(schedule):
            if schedule.get("minutes") == 7:
                raise RuntimeError("simulated unforeseen malformed field")
            return real_grace(schedule)

        with mock_patch.object(jobs_mod, "_compute_grace_seconds", selective_grace):
            due = get_due_jobs()  # must not raise

        ids = [j["id"] for j in due]
        assert "good-sibling" in ids, f"healthy sibling starved: {ids}"
        assert "poison" not in ids

        # Scheduler stays alive on subsequent ticks too.
        with mock_patch.object(jobs_mod, "_compute_grace_seconds", selective_grace):
            due2 = get_due_jobs()
        assert any(j["id"] == "good-sibling" for j in due2)


class TestSaveJobOutput:
    def test_creates_output_file(self, tmp_cron_dir):
        output_file = save_job_output("test123", "# Results\nEverything ok.")
        assert output_file.exists()
        assert output_file.read_text() == "# Results\nEverything ok."
        assert "test123" in str(output_file)


class TestCronOutputRetention:
    """Per-run cron output must self-prune so long deploys don't fill the disk (#52383)."""

    @staticmethod
    def _seed(d, count):
        d.mkdir(parents=True, exist_ok=True)
        names = [f"2026-06-25_10-00-{i:02d}.md" for i in range(count)]
        for n in names:
            (d / n).write_text("x")
        return names

    def test_prune_keeps_newest_n(self, tmp_path):
        from cron.jobs import _prune_job_output
        d = tmp_path / "job"
        names = self._seed(d, 10)
        assert _prune_job_output(d, keep=3) == 7
        assert sorted(p.name for p in d.glob("*.md")) == names[-3:]


# =========================================================================
# claim_dispatch — pre-run one-shot crash safety (issue #38758)
# =========================================================================

class TestClaimDispatch:
    """One-shot jobs must commit their dispatch BEFORE the side effect runs, so
    a tick that dies mid-execution (gateway kill, OOM, hard-timeout) can re-fire
    the job at most ``repeat.times`` times instead of infinitely."""

    def _oneshot(self, times=1, completed=0):
        return {
            "id": "os1",
            "name": "one-shot",
            "enabled": True,
            "schedule": {"kind": "once", "run_at": "2026-01-01T00:00:00+00:00"},
            "repeat": {"times": times, "completed": completed},
        }

    def test_claim_increments_and_persists(self, tmp_cron_dir):
        save_jobs([self._oneshot(times=1, completed=0)])
        assert claim_dispatch("os1") is True
        # Persisted BEFORE any side effect — survives a crash.
        assert load_jobs()[0]["repeat"]["completed"] == 1

    def test_already_dispatched_oneshot_is_removed(self, tmp_cron_dir):
        # A prior tick claimed (completed==times) then died before mark_job_run
        # could remove the job.  The next claim must refuse AND clean up.
        save_jobs([self._oneshot(times=1, completed=1)])
        assert claim_dispatch("os1") is False
        assert load_jobs() == []  # removed, will not re-fire


    def test_mark_job_run_does_not_double_count_preclaimed_oneshot(self, tmp_cron_dir):
        # Full lifecycle: claim bumps completed to times, then mark_job_run must
        # NOT increment again — it recognizes the pre-claim and retires the job
        # as a terminal completed record (retained for inspection, not re-fired).
        save_jobs([self._oneshot(times=1, completed=0)])
        assert claim_dispatch("os1") is True
        assert load_jobs()[0]["repeat"]["completed"] == 1
        mark_job_run("os1", success=True)
        retired = load_jobs()
        assert len(retired) == 1  # completed once, retired — not fired twice
        assert retired[0]["repeat"]["completed"] == 1  # no double count
        assert retired[0]["state"] == "completed"
        assert retired[0]["enabled"] is False


    def test_get_due_jobs_removes_stale_maxed_oneshot(self, tmp_cron_dir):
        # A claimed one-shot whose tick died leaves completed>=times with
        # last_run_at still unset, so the recovery helper re-arms it as due.
        # get_due_jobs must drop it instead of returning it for another fire.
        past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        save_jobs([{
            "id": "os1",
            "name": "one-shot",
            "enabled": True,
            "schedule": {"kind": "once", "run_at": past},
            "repeat": {"times": 1, "completed": 1},
            "next_run_at": None,
        }])
        due = get_due_jobs()
        assert due == []
        assert load_jobs() == []  # cleaned up


class TestLateEnvRepointScopesStore:
    """A HERMES_HOME set AFTER cron.jobs import must scope the store even
    without use_cron_store(): fixtures that patch the environment too late
    previously read/wrote the import-time jobs.json — the user's real file."""

    def test_late_env_repoint_scopes_store(self, tmp_path, monkeypatch):
        import cron.jobs as jobs

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = jobs._current_cron_store()
        expected = tmp_path.resolve() / "cron"
        assert store.cron_dir == expected
        assert store.jobs_file == expected / "jobs.json"
        assert store.output_dir == expected / "output"
        # the import-time compatibility constants are untouched
        assert jobs.JOBS_FILE != store.jobs_file


    def test_use_cron_store_override_still_wins(self, tmp_path, monkeypatch):
        import cron.jobs as jobs

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "env-home"))
        with jobs.use_cron_store(tmp_path / "override-home"):
            store = jobs._current_cron_store()
            assert store.jobs_file == (tmp_path / "override-home").resolve() / "cron" / "jobs.json"

    def test_heartbeat_does_not_recreate_deleted_named_profile(self, tmp_path):
        import cron.jobs as jobs

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        deleted_home = profiles_dir / "deleted"

        with jobs.use_cron_store(deleted_home):
            jobs.record_ticker_heartbeat()

        assert not deleted_home.exists()

    def test_heartbeat_initializes_existing_named_profile(self, tmp_path):
        import cron.jobs as jobs

        profile_home = tmp_path / "profiles" / "active"
        profile_home.mkdir(parents=True)

        with jobs.use_cron_store(profile_home):
            jobs.record_ticker_heartbeat()

        assert (profile_home / "cron" / "ticker_heartbeat").is_file()


    def test_public_io_after_late_env_repoint_leaves_old_file_untouched(
        self, tmp_path, monkeypatch
    ):
        """The public API, not the store internals: save_jobs()/load_jobs()
        called after a post-import HERMES_HOME repoint must operate on the NEW
        home's jobs.json and leave the import-time file byte-identical.

        The "import-time home" is SIMULATED at a tmp location by patching the
        module constants and the import-time snapshot together (so they still
        compare equal and the deliberate-repoint branch does not fire). The
        test must never touch the real import-time jobs.json: if this module
        was first imported before the suite's env isolation applied, that
        path IS the developer's live file — writing a sentinel there is
        exactly the incident this PR exists to prevent."""
        import cron.jobs as jobs

        sim_old_home = tmp_path / "import-time-home"
        sim_cron = sim_old_home / "cron"
        monkeypatch.setattr(jobs, "HERMES_DIR", sim_old_home)
        monkeypatch.setattr(jobs, "CRON_DIR", sim_cron)
        monkeypatch.setattr(jobs, "JOBS_FILE", sim_cron / "jobs.json")
        monkeypatch.setattr(jobs, "OUTPUT_DIR", sim_cron / "output")
        monkeypatch.setattr(
            jobs, "_IMPORT_STORE",
            jobs._CronStorePaths(jobs.CRON_DIR, jobs.JOBS_FILE, jobs.OUTPUT_DIR),
        )

        # Plant a sentinel at the (simulated) import-time location — the file
        # a late-patching fixture used to clobber.
        old_file = jobs.JOBS_FILE
        old_file.parent.mkdir(parents=True, exist_ok=True)
        sentinel = '[{"id": "sentinel-do-not-touch"}]'
        old_file.write_text(sentinel, encoding="utf-8")

        new_home = tmp_path / "late-home"
        monkeypatch.setenv("HERMES_HOME", str(new_home))

        job = {
            "id": "lateenvjob01",
            "name": "late-env",
            "prompt": None,
            "schedule_display": None,
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "enabled": True,
        }
        save_jobs([job])

        # public read round-trips from the NEW home...
        loaded = load_jobs()
        assert [j["id"] for j in loaded] == ["lateenvjob01"]
        new_file = new_home.resolve() / "cron" / "jobs.json"
        assert new_file.is_file()
        # ...and the import-time file is byte-identical to the sentinel.
        assert old_file.read_text(encoding="utf-8") == sentinel


# =========================================================================
# UTF-8 BOM on jobs.json (Windows Notepad / PowerShell 5.1)
# =========================================================================

class TestJobsJsonShapes:
    def test_load_jobs_normalizes_id_keyed_jobs_mapping(self, tmp_cron_dir):
        import json
        from cron.jobs import JOBS_FILE

        job_a = {
            "id": "cron1234abcd",
            "name": "daily briefing",
            "enabled": True,
            "prompt": "Summarize overnight incidents",
            "schedule": {"kind": "interval", "minutes": 1440, "display": "every 24h"},
        }
        job_b = {
            "id": "cron5678efgh",
            "name": "disabled cleanup",
            "enabled": False,
            "prompt": "Clean stale scratch files",
            "schedule": {"kind": "once", "run_at": "2030-01-15T14:00:00+00:00"},
        }
        payload = {
            "jobs": {
                job_a["id"]: job_a,
                job_b["id"]: job_b,
            },
            "updated_at": "2026-08-23T00:00:00+00:00",
        }
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_jobs()
        assert isinstance(loaded, list)
        assert {job["id"] for job in loaded} == {job_a["id"], job_b["id"]}

        listed = {job["id"]: job for job in list_jobs(include_disabled=True)}
        assert set(listed) == {job_a["id"], job_b["id"]}
        for expected in (job_a, job_b):
            actual = listed[expected["id"]]
            assert actual["id"] == expected["id"]
            assert actual["name"] == expected["name"]
            assert actual["prompt"] == expected["prompt"]
            assert actual["schedule"] == expected["schedule"]
            assert actual["enabled"] is expected["enabled"]


class TestJobsJsonUtf8Bom:
    """jobs.json readers must accept a leading UTF-8 BOM.

    Matching the env-class dialect (utf-8-sig): a BOM from Windows editors
    must not raise JSONDecodeError / RuntimeError on load_jobs().
    """

    def test_load_jobs_accepts_utf8_bom(self, tmp_cron_dir):
        """BOM'd jobs.json loads — the pre-fix crash repro."""
        import json
        from pathlib import Path
        from cron.jobs import JOBS_FILE, load_jobs

        payload = {
            "jobs": [
                {
                    "id": "bomjob01",
                    "name": "bom-test",
                    "enabled": True,
                    "prompt": "hello",
                    "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                }
            ]
        }
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_bytes(
            b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8")
        )

        loaded = load_jobs()
        assert [j["id"] for j in loaded] == ["bomjob01"]
        assert loaded[0]["name"] == "bom-test"

    def test_load_jobs_bomless_regression(self, tmp_cron_dir):
        """BOM-less UTF-8 jobs.json must keep loading after utf-8-sig."""
        import json
        from cron.jobs import JOBS_FILE, load_jobs

        payload = {
            "jobs": [
                {
                    "id": "plainjob01",
                    "name": "plain",
                    "enabled": True,
                    "prompt": "hi",
                    "schedule": {"kind": "interval", "minutes": 30, "display": "every 30m"},
                }
            ]
        }
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_jobs()
        assert [j["id"] for j in loaded] == ["plainjob01"]




# =========================================================================
# ID-keyed jobs map on jobs.json (external tools / hand edits) — #92935
# =========================================================================

class TestJobsJsonIdKeyedMap:
    """load_jobs() must flatten an ID-keyed ``jobs`` map to the list contract.

    A store written as ``{"jobs": {"<job_id>": {...}, ...}}`` (external tool
    or hand edit — Hermes' own save_jobs() only ever writes a list) made
    load_jobs() return a dict. Every consumer iterates it as a list, so
    ``list_jobs()`` → ``_normalize_job_record`` → ``dict(<id-string>)`` raised
    ``ValueError: dictionary update sequence element #0 has length 1; 2 is
    required`` and took down ``hermes cron list``, the ``cronjob(action=
    "list")`` tool, and the Dashboard cron view. The values already carry
    their own ``id`` matching the map key, so flattening is lossless.
    """

    _ID_KEYED = {
        "jobs": {
            "cron1234abcd": {
                "id": "cron1234abcd",
                "name": "Example job",
                "enabled": True,
                "prompt": "do a thing",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            },
            "cron5678efgh": {
                "id": "cron5678efgh",
                "name": "Second job",
                "enabled": True,
                "prompt": "do another",
                "schedule": {"kind": "interval", "minutes": 30, "display": "every 30m"},
            },
        },
        "updated_at": "2026-08-23T10:10:12+08:00",
    }

    def test_load_jobs_flattens_id_keyed_map(self, tmp_cron_dir):
        """The pre-fix repro: load_jobs() returns a list, not the raw dict."""
        import json
        from cron.jobs import JOBS_FILE, load_jobs

        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(self._ID_KEYED), encoding="utf-8")

        loaded = load_jobs()
        assert isinstance(loaded, list)
        assert {j["id"] for j in loaded} == {"cron1234abcd", "cron5678efgh"}
        assert all(isinstance(j, dict) for j in loaded)

    def test_list_jobs_survives_id_keyed_map(self, tmp_cron_dir):
        """The reported traceback path (hermes cron list / cronjob list tool)."""
        import json
        from cron.jobs import JOBS_FILE, list_jobs

        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(self._ID_KEYED), encoding="utf-8")

        # Pre-fix this raised ValueError from _normalize_job_record(dict(<str>)).
        jobs = list_jobs(include_disabled=True)
        assert {j["id"] for j in jobs} == {"cron1234abcd", "cron5678efgh"}

    def test_id_keyed_map_repaired_to_list_on_disk(self, tmp_cron_dir):
        """Loading rewrites the store into the canonical {"jobs": [...]} form."""
        import json
        from cron.jobs import JOBS_FILE, load_jobs

        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(self._ID_KEYED), encoding="utf-8")

        load_jobs()

        on_disk = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        assert isinstance(on_disk["jobs"], list)
        assert {j["id"] for j in on_disk["jobs"]} == {"cron1234abcd", "cron5678efgh"}

        # A second load reads the repaired list unchanged (idempotent).
        reloaded = load_jobs()
        assert {j["id"] for j in reloaded} == {"cron1234abcd", "cron5678efgh"}

    def test_empty_id_keyed_map_returns_empty_list(self, tmp_cron_dir):
        """An empty ``jobs`` map must not crash and yields no jobs."""
        import json
        from cron.jobs import JOBS_FILE, load_jobs

        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps({"jobs": {}}), encoding="utf-8")

        assert load_jobs() == []

    def test_map_value_without_inline_id_adopts_key(self, tmp_cron_dir):
        """A value lacking an inline "id" gets the map key as its id."""
        import json
        from cron.jobs import JOBS_FILE, load_jobs

        payload = {
            "jobs": {
                "cronkeyonly1": {
                    "name": "keyed only",
                    "enabled": True,
                    "prompt": "no inline id here",
                    "schedule": {"kind": "interval", "minutes": 15, "display": "every 15m"},
                },
                "cron-ignored-key": {
                    "id": "croninline99",
                    "name": "inline id wins",
                    "enabled": True,
                    "prompt": "inline id present",
                    "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
                },
            }
        }
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(payload), encoding="utf-8")

        loaded = {j["id"]: j for j in load_jobs()}
        # Key adopted when the value has no inline id.
        assert "cronkeyonly1" in loaded
        assert loaded["cronkeyonly1"]["name"] == "keyed only"
        # Inline id wins over a differing map key.
        assert "croninline99" in loaded
        assert "cron-ignored-key" not in loaded

        # Self-heal persisted the id-merged records.
        on_disk = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        assert isinstance(on_disk["jobs"], list)
        assert {j["id"] for j in on_disk["jobs"]} == {"cronkeyonly1", "croninline99"}

    def test_non_dict_map_values_skipped_with_warning(self, tmp_cron_dir, caplog):
        """Junk (non-dict) values in the map are skipped, never crash."""
        import json
        import logging
        from cron.jobs import JOBS_FILE, list_jobs, load_jobs

        payload = {
            "jobs": {
                "goodjob1": {
                    "name": "survivor",
                    "enabled": True,
                    "prompt": "keep me",
                    "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                },
                "junk-string": "i am not a job",
                "junk-number": 42,
                "junk-null": None,
            }
        }
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(payload), encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="cron.jobs"):
            loaded = load_jobs()
        assert [j["id"] for j in loaded] == ["goodjob1"]
        assert any("non-dict" in rec.getMessage() for rec in caplog.records)

        # The reported traceback path also survives the junk.
        jobs = list_jobs(include_disabled=True)
        assert {j["id"] for j in jobs} == {"goodjob1"}

        # Self-heal wrote only the valid record, canonical list shape.
        on_disk = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        assert isinstance(on_disk["jobs"], list)
        assert [j["id"] for j in on_disk["jobs"]] == ["goodjob1"]

    def test_all_junk_map_values_yield_empty_list(self, tmp_cron_dir):
        """A map of only junk values flattens to [] without crashing."""
        import json
        from cron.jobs import JOBS_FILE, load_jobs

        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(
            json.dumps({"jobs": {"a": "junk", "b": 1}}), encoding="utf-8"
        )

        assert load_jobs() == []




class TestAdvanceNextRuns:
    """Tests for advance_next_runs() — the batched due-set advance.

    The scheduler's pre-dispatch loop advanced each due job individually:
    N due jobs = N full load_jobs() + N full save_jobs() of the jobs file
    (~110 ms at N=50, measured). The batch form does one load + at most
    one save (~2 ms). Imported inside test bodies so the pre-fix tree
    fails with a real test failure (ImportError), not a collection error.
    """

    def _make_due(self, tmp_cron_dir, n_recurring=3, n_oneshot=1):
        rec = [create_job(prompt=f"rec {i}", schedule="every 1h")
               for i in range(n_recurring)]
        one = [create_job(prompt=f"one {i}", schedule="30m")
               for i in range(n_oneshot)]
        jobs = load_jobs()
        old = (datetime.now() - timedelta(minutes=5)).isoformat()
        for j in jobs:
            j["next_run_at"] = old
        save_jobs(jobs)
        return [j["id"] for j in rec], [j["id"] for j in one]

    def test_batch_advances_recurring_skips_oneshots(self, tmp_cron_dir):
        from cron.jobs import advance_next_runs
        rec_ids, one_ids = self._make_due(tmp_cron_dir)
        advanced = advance_next_runs(rec_ids + one_ids)
        assert advanced == len(rec_ids)
        from cron.jobs import _ensure_aware, _hermes_now
        for jid in rec_ids:
            nxt = _ensure_aware(datetime.fromisoformat(get_job(jid)["next_run_at"]))
            assert nxt > _hermes_now()
        for jid in one_ids:
            # one-shots keep their (past) next_run_at for restart retry
            assert datetime.fromisoformat(get_job(jid)["next_run_at"]) < datetime.now()

    def test_batch_single_load_and_save(self, tmp_cron_dir, monkeypatch):
        """I/O pin: the whole due set costs one load + one save, not N+N.
        Fails pre-fix (function absent) and would fail on any regression
        back to per-job I/O."""
        from cron.jobs import advance_next_runs
        rec_ids, _ = self._make_due(tmp_cron_dir, n_recurring=10, n_oneshot=0)
        import cron.jobs as cj
        counts = {"load": 0, "save": 0}
        real_load, real_save = cj.load_jobs, cj.save_jobs
        monkeypatch.setattr(cj, "load_jobs", lambda *a, **k: (
            counts.__setitem__("load", counts["load"] + 1), real_load(*a, **k))[1])
        monkeypatch.setattr(cj, "save_jobs", lambda *a, **k: (
            counts.__setitem__("save", counts["save"] + 1), real_save(*a, **k))[1])
        advance_next_runs(rec_ids)
        assert counts == {"load": 1, "save": 1}

    def test_batch_no_save_when_nothing_advances(self, tmp_cron_dir, monkeypatch):
        from cron.jobs import advance_next_runs
        rec_ids, one_ids = self._make_due(tmp_cron_dir, n_recurring=0, n_oneshot=2)
        import cron.jobs as cj
        saves = [0]
        real_save = cj.save_jobs
        monkeypatch.setattr(cj, "save_jobs", lambda *a, **k: (
            saves.__setitem__(0, saves[0] + 1), real_save(*a, **k))[1])
        assert advance_next_runs(one_ids + ["missing-id"]) == 0
        assert saves[0] == 0

    def test_wrapper_semantics_unchanged(self, tmp_cron_dir):
        """advance_next_run keeps its per-job contract over the batch."""
        rec_ids, one_ids = self._make_due(tmp_cron_dir)
        assert advance_next_run(rec_ids[0]) is True
        assert advance_next_run(one_ids[0]) is False
        assert advance_next_run("missing-id") is False


# =========================================================================
# Completed one-shot retention sweep
# =========================================================================

class TestCompletedOneshotRetentionSweep:
    """Completed one-shots are retained for inspection, then pruned by age."""

    def _completed_oneshot(self, age_days: float):
        """Create a one-shot, complete it, and backdate its last_run_at."""
        job = create_job(prompt="Once", schedule="30m", repeat=1)
        mark_job_run(job["id"], success=True, delivery_error="boom")
        stamp = (
            datetime.now(timezone.utc) - timedelta(days=age_days)
        ).isoformat()
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                j["last_run_at"] = stamp
        save_jobs(jobs)
        return job["id"]

    def test_sweep_prunes_old_completed_oneshot(self, tmp_cron_dir):
        old_id = self._completed_oneshot(age_days=30)
        get_due_jobs()  # sweep runs as part of the due scan
        assert get_job(old_id) is None

    def test_sweep_keeps_recent_completed_oneshot(self, tmp_cron_dir):
        recent_id = self._completed_oneshot(age_days=1)
        get_due_jobs()
        kept = get_job(recent_id)
        assert kept is not None
        assert kept["state"] == "completed"
        assert kept["last_delivery_error"] == "boom"

    def test_sweep_ignores_recurring_jobs(self, tmp_cron_dir):
        """Old recurring jobs are never candidates, whatever their history."""
        job = create_job(prompt="Recurring", schedule="every 1h")
        stamp = (
            datetime.now(timezone.utc) - timedelta(days=365)
        ).isoformat()
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                j["last_run_at"] = stamp
        save_jobs(jobs)
        get_due_jobs()
        assert get_job(job["id"]) is not None

    def test_sweep_disabled_by_nonpositive_retention(self, tmp_cron_dir, monkeypatch):
        monkeypatch.setattr(
            "cron.jobs._completed_oneshot_retention_days", lambda: 0.0
        )
        old_id = self._completed_oneshot(age_days=30)
        get_due_jobs()
        assert get_job(old_id) is not None

    def test_recurring_jobs_unaffected_by_retention_change(self, tmp_cron_dir):
        """A recurring job still cycles normally alongside retained one-shots."""
        recurring = create_job(prompt="Recurring", schedule="every 1h")
        self._completed_oneshot(age_days=1)
        mark_job_run(recurring["id"], success=True)
        updated = get_job(recurring["id"])
        assert updated["enabled"] is True
        assert updated["state"] == "scheduled"
        assert updated["next_run_at"] is not None


class TestEnsureCronDirWidened:
    """Tests for the widened _ensure_cron_dir covering all cron mkdir sites."""

    def test_ensure_cron_dir_named_profile_subdir_fails_closed(self, tmp_path):
        """A subdir under a deleted named profile's cron/ must not recreate it."""
        import cron.jobs as jobs

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        deleted_home = profiles_dir / "deleted"
        # cron_dir doesn't exist because the profile was deleted
        output_dir = deleted_home / "cron" / "output" / "job_123"

        import pytest
        with pytest.raises(FileNotFoundError):
            jobs._ensure_cron_dir(output_dir)
        assert not deleted_home.exists()

    def test_ensure_cron_dir_default_home_creates_subdir(self, tmp_path):
        """A subdir under a default home's cron/ should be created normally."""
        import cron.jobs as jobs

        default_home = tmp_path / "default_home"
        default_home.mkdir()
        output_dir = default_home / "cron" / "output" / "job_123"

        jobs._ensure_cron_dir(output_dir)
        assert output_dir.is_dir()

    def test_ensure_cron_dir_named_profile_cron_dir_fails_closed(self, tmp_path):
        """The cron dir of a deleted named profile must not be recreated."""
        import cron.jobs as jobs

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        deleted_home = profiles_dir / "deleted"
        cron_dir = deleted_home / "cron"

        import pytest
        with pytest.raises(FileNotFoundError):
            jobs._ensure_cron_dir(cron_dir)
        assert not deleted_home.exists()

    def test_ensure_cron_dir_existing_named_profile_cron_dir_works(self, tmp_path):
        """An existing named profile's cron dir should be created normally."""
        import cron.jobs as jobs

        profiles_dir = tmp_path / "profiles"
        active_home = profiles_dir / "active"
        active_home.mkdir(parents=True)
        cron_dir = active_home / "cron"

        jobs._ensure_cron_dir(cron_dir)
        assert cron_dir.is_dir()

    def test_ensure_cron_dir_scripts_dir_under_named_profile_fails_closed(self, tmp_path):
        """A scripts dir under a deleted named profile must not be recreated."""
        import cron.jobs as jobs

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        deleted_home = profiles_dir / "deleted"
        scripts_dir = deleted_home / "scripts"

        import pytest
        with pytest.raises(FileNotFoundError):
            jobs._ensure_cron_dir(scripts_dir)
        assert not deleted_home.exists()
