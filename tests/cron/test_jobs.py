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
    get_due_jobs,
    save_job_output,
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

    def test_hours(self):
        assert parse_duration("2h") == 120
        assert parse_duration("1hr") == 60
        assert parse_duration("3hrs") == 180
        assert parse_duration("1hour") == 60
        assert parse_duration("24hours") == 1440

    def test_days(self):
        assert parse_duration("1d") == 1440
        assert parse_duration("7day") == 7 * 1440
        assert parse_duration("2days") == 2 * 1440

    def test_whitespace_tolerance(self):
        assert parse_duration("  30m  ") == 30
        assert parse_duration("2 h") == 120

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_duration("abc")
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

    def test_every_case_insensitive(self):
        result = parse_schedule("Every 30m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 30

    def test_cron_expression(self):
        pytest.importorskip("croniter")
        result = parse_schedule("0 9 * * *")
        assert result["kind"] == "cron"
        assert result["expr"] == "0 9 * * *"

    def test_iso_timestamp(self):
        result = parse_schedule("2030-01-15T14:00:00")
        assert result["kind"] == "once"
        assert "2030-01-15" in result["run_at"]

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("not_a_schedule")

    def test_invalid_cron_raises(self):
        pytest.importorskip("croniter")
        with pytest.raises(ValueError):
            parse_schedule("99 99 99 99 99")

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

    def test_once_past_returns_none(self):
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        schedule = {"kind": "once", "run_at": past}
        assert compute_next_run(schedule) is None

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

    def test_interval_subsequent_run(self):
        schedule = {"kind": "interval", "minutes": 30}
        last = datetime.now().astimezone().isoformat()
        result = compute_next_run(schedule, last_run_at=last)
        next_dt = datetime.fromisoformat(result)
        # Should be ~30 minutes from last run
        assert next_dt > datetime.now().astimezone() + timedelta(minutes=29)

    def test_cron_returns_future(self):
        pytest.importorskip("croniter")
        schedule = {"kind": "cron", "expr": "* * * * *"}  # every minute
        result = compute_next_run(schedule)
        assert isinstance(result, str), f"Expected ISO timestamp string, got {type(result)}"
        assert len(result) > 0
        next_dt = datetime.fromisoformat(result)
        assert isinstance(next_dt, datetime)
        assert next_dt > datetime.now().astimezone()

    def test_unknown_kind_returns_none(self):
        assert compute_next_run({"kind": "unknown"}) is None


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

    def test_list_jobs_normalizes_partial_legacy_records(self, tmp_cron_dir):
        save_jobs([
            {
                "id": "abc123deadbe",
                "name": None,
                "prompt": None,
                "schedule_display": None,
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "enabled": True,
            }
        ])

        jobs = list_jobs()

        assert jobs[0]["id"] == "abc123deadbe"
        assert jobs[0]["name"] == "abc123deadbe"
        assert jobs[0]["prompt"] == ""
        assert jobs[0]["schedule_display"] == "every 60m"
        assert jobs[0]["state"] == "scheduled"

    def test_remove_job(self, tmp_cron_dir):
        job = create_job(prompt="Temp job", schedule="30m")
        assert remove_job(job["id"]) is True
        assert get_job(job["id"]) is None

    def test_remove_job_rejects_unsafe_legacy_id_before_output_cleanup(self, tmp_cron_dir):
        """Legacy unsafe IDs left over from before the create-time guard
        must fail closed without half-applying the removal."""
        job = create_job(prompt="Legacy unsafe", schedule="every 1h")
        job["id"] = "../escape"
        save_jobs([job])
        outside = tmp_cron_dir / "escape"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep", encoding="utf-8")

        with pytest.raises(ValueError, match="output path"):
            remove_job("../escape")

        # Job should still be in the store and the escape dir untouched.
        assert load_jobs()[0]["id"] == "../escape"
        assert (outside / "keep.txt").exists()

    def test_remove_nonexistent_returns_false(self, tmp_cron_dir):
        assert remove_job("nonexistent") is False

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

    def test_recent_past_one_shot_within_grace_still_creates(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 3, 18, 4, 30, 30, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        recent = (now - timedelta(seconds=30)).isoformat()

        job = create_job(prompt="Still valid", schedule=recent)

        assert job["next_run_at"] == recent
        assert load_jobs()[0]["id"] == job["id"]

    def test_interval_no_auto_repeat(self, tmp_cron_dir):
        job = create_job(prompt="Recurring", schedule="every 1h")
        assert job["repeat"]["times"] is None

    def test_default_delivery_origin(self, tmp_cron_dir):
        job = create_job(
            prompt="Test", schedule="30m",
            origin={"platform": "telegram", "chat_id": "123"},
        )
        assert job["deliver"] == "origin"

    def test_default_delivery_local_no_origin(self, tmp_cron_dir):
        job = create_job(prompt="Test", schedule="30m")
        assert job["deliver"] == "local"


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

    def test_update_schedule(self, tmp_cron_dir):
        job = create_job(prompt="Daily report", schedule="every 1h")
        assert job["schedule"]["kind"] == "interval"
        assert job["schedule"]["minutes"] == 60
        old_next_run = job["next_run_at"]
        new_schedule = parse_schedule("every 2h")
        updated = update_job(job["id"], {"schedule": new_schedule, "schedule_display": new_schedule["display"]})
        assert updated is not None
        assert updated["schedule"]["kind"] == "interval"
        assert updated["schedule"]["minutes"] == 120
        assert updated["schedule_display"] == "every 120m"
        assert updated["next_run_at"] != old_next_run
        # Verify persisted to disk
        fetched = get_job(job["id"])
        assert fetched["schedule"]["minutes"] == 120
        assert fetched["schedule_display"] == "every 120m"

    def test_update_to_past_oneshot_rejected(self, tmp_cron_dir, monkeypatch):
        """Updating a job's schedule to a one-shot >ONESHOT_GRACE_SECONDS in the
        past must raise ValueError — otherwise the ghost-job bug (#59395) re-enters
        through the update door (next_run_at=None stored with state='scheduled').
        The original job must be left unchanged on disk."""
        now = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        job = create_job(prompt="Recurring", schedule="every 1h", deliver="local")
        past = parse_schedule((now - timedelta(minutes=10)).isoformat())
        with pytest.raises(ValueError, match="past and cannot be scheduled"):
            update_job(job["id"], {"schedule": past})
        # Original job unchanged — still the recurring interval, still scheduled.
        fetched = get_job(job["id"])
        assert fetched["schedule"]["kind"] == "interval"
        assert fetched["next_run_at"] is not None

    def test_update_to_future_oneshot_accepted(self, tmp_cron_dir, monkeypatch):
        """Updating to a FUTURE one-shot still works — only past ones are rejected."""
        now = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        job = create_job(prompt="Recurring", schedule="every 1h", deliver="local")
        future = parse_schedule((now + timedelta(hours=2)).isoformat())
        updated = update_job(job["id"], {"schedule": future})
        assert updated is not None
        assert updated["schedule"]["kind"] == "once"
        assert updated["next_run_at"] is not None

    def test_update_enable_disable(self, tmp_cron_dir):
        job = create_job(prompt="Toggle me", schedule="every 1h")
        assert job["enabled"] is True
        updated = update_job(job["id"], {"enabled": False})
        assert updated["enabled"] is False
        fetched = get_job(job["id"])
        assert fetched["enabled"] is False

    def test_update_nonexistent_returns_none(self, tmp_cron_dir):
        result = update_job("nonexistent_id", {"name": "X"})
        assert result is None

    def test_update_rejects_id_change(self, tmp_cron_dir):
        """Job IDs are filesystem path components — must be immutable."""
        job = create_job(prompt="Original", schedule="every 1h")

        with pytest.raises(ValueError, match="id"):
            update_job(job["id"], {"id": "../escape"})

        # Original job still resolvable, no rename happened.
        assert get_job(job["id"]) is not None
        assert get_job("../escape") is None


class TestPauseResumeJob:
    def test_pause_sets_state(self, tmp_cron_dir):
        job = create_job(prompt="Pause me", schedule="every 1h")
        paused = pause_job(job["id"], reason="user paused")
        assert paused is not None
        assert paused["enabled"] is False
        assert paused["state"] == "paused"
        assert paused["paused_reason"] == "user paused"

    def test_resume_reenables_job(self, tmp_cron_dir):
        job = create_job(prompt="Resume me", schedule="every 1h")
        pause_job(job["id"], reason="user paused")
        resumed = resume_job(job["id"])
        assert resumed is not None
        assert resumed["enabled"] is True
        assert resumed["state"] == "scheduled"
        assert resumed["paused_at"] is None
        assert resumed["paused_reason"] is None

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

    def test_resolve_by_name(self, tmp_cron_dir):
        from cron.jobs import resolve_job_ref

        job = create_job(prompt="A", schedule="1h", name="alpha")
        assert resolve_job_ref("alpha")["id"] == job["id"]

    def test_resolve_by_name_case_insensitive(self, tmp_cron_dir):
        from cron.jobs import resolve_job_ref

        job = create_job(prompt="A", schedule="1h", name="MyJob")
        assert resolve_job_ref("myjob")["id"] == job["id"]
        assert resolve_job_ref("MYJOB")["id"] == job["id"]

    def test_resolve_returns_none_when_not_found(self, tmp_cron_dir):
        from cron.jobs import resolve_job_ref

        create_job(prompt="A", schedule="1h", name="alpha")
        assert resolve_job_ref("does-not-exist") is None
        assert resolve_job_ref("") is None

    def test_resolve_id_wins_over_name(self, tmp_cron_dir):
        """If a job's name happens to equal another job's ID, ID match wins."""
        from cron.jobs import resolve_job_ref

        j1 = create_job(prompt="A", schedule="1h")
        # Create a second job whose name is j1's ID
        j2 = create_job(prompt="B", schedule="1h", name=j1["id"])
        # Looking up j1["id"] must return j1, not the colliding-name job j2
        assert resolve_job_ref(j1["id"])["id"] == j1["id"]
        assert resolve_job_ref(j1["id"])["id"] != j2["id"]

    def test_resolve_ambiguous_name_raises(self, tmp_cron_dir):
        """Two jobs sharing a name → refuse to pick, surface both IDs."""
        from cron.jobs import AmbiguousJobReference, resolve_job_ref

        j1 = create_job(prompt="A", schedule="1h", name="dup")
        j2 = create_job(prompt="B", schedule="1h", name="dup")
        with pytest.raises(AmbiguousJobReference) as exc_info:
            resolve_job_ref("dup")
        ids = {m["id"] for m in exc_info.value.matches}
        assert ids == {j1["id"], j2["id"]}
        # Error message mentions both IDs so the user can pick one
        assert j1["id"] in str(exc_info.value)
        assert j2["id"] in str(exc_info.value)

    def test_trigger_by_name(self, tmp_cron_dir):
        from cron.jobs import trigger_job

        job = create_job(prompt="A", schedule="1h", name="alpha")
        result = trigger_job("alpha")
        assert result is not None
        assert result["id"] == job["id"]

    def test_pause_by_name(self, tmp_cron_dir):
        job = create_job(prompt="A", schedule="1h", name="alpha")
        result = pause_job("alpha", reason="manual")
        assert result is not None
        assert result["id"] == job["id"]
        assert result["state"] == "paused"

    def test_remove_by_name(self, tmp_cron_dir):
        job = create_job(prompt="A", schedule="1h", name="alpha")
        assert remove_job("alpha") is True
        assert get_job(job["id"]) is None

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

    def test_repeat_limit_removes_job(self, tmp_cron_dir):
        job = create_job(prompt="Once", schedule="30m", repeat=1)
        mark_job_run(job["id"], success=True)
        # Job should be removed after hitting repeat limit
        assert get_job(job["id"]) is None

    def test_repeat_negative_one_is_infinite(self, tmp_cron_dir):
        # LLMs often pass repeat=-1 to mean "infinite/forever".
        # The job must NOT be deleted after runs when repeat <= 0.
        job = create_job(prompt="Forever", schedule="every 1h", repeat=-1)
        # -1 should be normalised to None (infinite) at create time
        assert job["repeat"]["times"] is None
        # Running it multiple times should never delete it
        for _ in range(3):
            mark_job_run(job["id"], success=True)
            assert get_job(job["id"]) is not None, "job was deleted after run despite infinite repeat"

    def test_repeat_zero_is_infinite(self, tmp_cron_dir):
        # repeat=0 should also be treated as None (infinite), not "run zero times".
        job = create_job(prompt="ZeroRepeat", schedule="every 1h", repeat=0)
        assert job["repeat"]["times"] is None
        mark_job_run(job["id"], success=True)
        assert get_job(job["id"]) is not None

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

    def test_delivery_error_cleared_on_success(self, tmp_cron_dir):
        """Successful delivery clears the previous delivery error."""
        job = create_job(prompt="Report", schedule="every 1h")
        mark_job_run(job["id"], success=True, delivery_error="network timeout")
        updated = get_job(job["id"])
        assert updated["last_delivery_error"] == "network timeout"
        # Next run delivers successfully
        mark_job_run(job["id"], success=True, delivery_error=None)
        updated = get_job(job["id"])
        assert updated["last_delivery_error"] is None

    def test_both_agent_and_delivery_error(self, tmp_cron_dir):
        """Agent fails AND delivery fails — both errors recorded."""
        job = create_job(prompt="Report", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="model timeout",
                     delivery_error="platform 'discord' not enabled")
        updated = get_job(job["id"])
        assert updated["last_status"] == "error"
        assert updated["last_error"] == "model timeout"
        assert updated["last_delivery_error"] == "platform 'discord' not enabled"

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

    def test_recurring_interval_not_disabled_when_next_run_is_none(self, tmp_cron_dir, monkeypatch):
        """Defensive sibling of the cron test — any recurring schedule that
        somehow yields next_run_at=None must stay enabled with state=error.
        """
        job = create_job(prompt="Recurring", schedule="every 1h")
        assert job["schedule"]["kind"] == "interval"

        # Force compute_next_run to return None for this call — simulates
        # any future regression where a recurring schedule loses its
        # next-run computation (missing dep, corrupt schedule, etc.).
        monkeypatch.setattr("cron.jobs.compute_next_run", lambda *a, **kw: None)

        mark_job_run(job["id"], success=True)

        updated = get_job(job["id"])
        assert updated is not None
        assert updated["enabled"] is True
        assert updated["state"] == "error"
        assert updated["state"] != "completed"

    def test_oneshot_still_completes_when_next_run_is_none(self, tmp_cron_dir):
        """One-shot jobs must still flip to enabled=false, state=completed
        when next_run_at cannot be computed — the #16265 fix must not
        regress this path. We bypass create_job and craft a minimal
        one-shot record directly so that the repeat-limit branch doesn't
        pop the job before we observe the terminal-completion branch.
        """
        jobs = [{
            "id": "oneshot-test",
            "prompt": "Once",
            "schedule": {"kind": "once", "run_at": "2020-01-01T00:00:00+00:00", "display": "once"},
            "repeat": {"times": None, "completed": 0},
            "enabled": True,
            "state": "scheduled",
            "next_run_at": "2020-01-01T00:00:00+00:00",
            "last_run_at": None,
            "last_status": None,
            "last_error": None,
            "last_delivery_error": None,
            "created_at": "2020-01-01T00:00:00+00:00",
        }]
        save_jobs(jobs)

        mark_job_run("oneshot-test", success=True)

        updated = get_job("oneshot-test")
        assert updated is not None
        assert updated["next_run_at"] is None
        assert updated["enabled"] is False
        assert updated["state"] == "completed"


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

    def test_advances_cron_job(self, tmp_cron_dir):
        """Cron-expression jobs should have next_run_at bumped to the next occurrence."""
        pytest.importorskip("croniter")
        job = create_job(prompt="Daily wakeup", schedule="15 6 * * *")
        # Force next_run_at to 30 minutes ago
        jobs = load_jobs()
        old_next = (datetime.now() - timedelta(minutes=30)).isoformat()
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

    def test_nonexistent_job_returns_false(self, tmp_cron_dir):
        result = advance_next_run("nonexistent-id")
        assert result is False

    def test_already_future_stays_future(self, tmp_cron_dir):
        """If next_run_at is already in the future, advance keeps it in the future (no harm)."""
        job = create_job(prompt="Future job", schedule="every 1h")
        # next_run_at is already set to ~1h from now by create_job
        advance_next_run(job["id"])
        # Regardless of return value, the job should still be in the future
        updated = get_job(job["id"])
        from cron.jobs import _ensure_aware, _hermes_now
        new_next_dt = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        assert new_next_dt > _hermes_now(), "next_run_at should remain in the future"

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


    def test_stale_repeat_limited_job_consumes_one_run_on_catchup(self, tmp_cron_dir, monkeypatch):
        """#33315 behavior note: a stale recurring job with a repeat.times limit
        fires ONCE on catch-up and consumes one of its runs (it is no longer
        silently skipped). Pins the documented repeat-count interaction so it
        isn't changed accidentally."""
        from cron.jobs import _hermes_now
        job = create_job(prompt="Limited", schedule="every 5m", repeat=3)
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (_hermes_now() - timedelta(minutes=11)).isoformat()
        jobs[0]["last_run_at"] = (_hermes_now() - timedelta(minutes=11)).isoformat()
        save_jobs(jobs)

        # The stale job is returned to fire once (not skipped).
        due = get_due_jobs()
        assert [j["id"] for j in due] == [job["id"]]
        # Simulate the run completing: mark_job_run increments completed.
        mark_job_run(job["id"], True)
        survived = get_job(job["id"])
        assert survived is not None, "job should survive (3 > 1 completed)"
        assert survived["repeat"]["completed"] == 1

    def test_future_not_returned(self, tmp_cron_dir):
        create_job(prompt="Not yet", schedule="every 1h")
        due = get_due_jobs()
        assert len(due) == 0

    def test_disabled_not_returned(self, tmp_cron_dir):
        job = create_job(prompt="Disabled", schedule="every 1h")
        jobs = load_jobs()
        jobs[0]["enabled"] = False
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 0

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

    def test_broken_stale_one_shot_without_next_run_is_not_recovered(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 3, 18, 4, 30, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        save_jobs(
            [{
                "id": "oneshot-stale",
                "name": "Too old",
                "prompt": "Word of the day",
                "schedule": {"kind": "once", "run_at": "2026-03-18T04:22:00+00:00", "display": "once at 2026-03-18 04:22"},
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

        assert get_due_jobs() == []
        assert get_job("oneshot-stale")["next_run_at"] is None

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

    def test_one_shot_run_claim_expires_after_ttl(self, tmp_cron_dir, monkeypatch):
        """A claiming tick that DIED mid-run must not wedge the one-shot forever:
        once the run_claim is older than the TTL it is re-dispatched (recovered)."""
        # Pin the inactivity timeout unset so the derived TTL is deterministic.
        monkeypatch.delenv("HERMES_CRON_TIMEOUT", raising=False)
        from cron.jobs import _hermes_now, _oneshot_run_claim_ttl_seconds
        ttl = _oneshot_run_claim_ttl_seconds()
        t0 = _hermes_now()
        run_at = (t0 - timedelta(seconds=5)).isoformat()
        save_jobs([{
            "id": "wedged", "name": "R", "prompt": "x",
            "schedule": {"kind": "once", "run_at": run_at},
            "next_run_at": run_at, "enabled": True, "state": "scheduled",
        }])
        assert [j["id"] for j in get_due_jobs()] == ["wedged"]  # A claims, then dies

        # Just inside the TTL: still claimed → skipped.
        monkeypatch.setattr("cron.jobs._hermes_now",
                            lambda: t0 + timedelta(seconds=ttl - 10))
        assert get_due_jobs() == []

        # Just past the TTL: stale claim → re-dispatched (recovered), re-claimed.
        monkeypatch.setattr("cron.jobs._hermes_now",
                            lambda: t0 + timedelta(seconds=ttl + 10))
        recovered = get_due_jobs()
        assert [j["id"] for j in recovered] == ["wedged"]

    def test_run_claim_ttl_derived_from_cron_timeout(self, tmp_cron_dir, monkeypatch):
        """The stale-recovery TTL tracks HERMES_CRON_TIMEOUT (3x headroom), with
        the fixed constant as a floor, and falls back to the constant when runs
        are unbounded (timeout=0)."""
        from cron.jobs import (
            _oneshot_run_claim_ttl_seconds as ttl,
            ONESHOT_RUN_CLAIM_TTL_SECONDS as FLOOR,
        )
        # Unset → default 600s inactivity → 1800s (== the historical constant).
        monkeypatch.delenv("HERMES_CRON_TIMEOUT", raising=False)
        assert ttl() == 1800.0

        # A large custom timeout scales the TTL up (3x headroom).
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "1200")
        assert ttl() == 3600.0

        # A tiny timeout is floored so a claim can never expire mid-run.
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "30")
        assert ttl() == float(FLOOR)

        # Unlimited runs (0) → no finite bound → fall back to the floor.
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        assert ttl() == float(FLOOR)

        # Invalid value → treated as the default 600s → 1800s.
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "not-a-number")
        assert ttl() == 1800.0


    def test_mark_job_run_clears_one_shot_run_claim(self, tmp_cron_dir, monkeypatch):
        """mark_job_run() clears the run_claim on completion so a re-dispatched
        one-shot (e.g. a stale-recovered retry) is claimable again."""
        from cron.jobs import _hermes_now
        t0 = _hermes_now()
        run_at = (t0 - timedelta(seconds=5)).isoformat()
        # Give it repeat headroom so mark_job_run keeps the job around.
        save_jobs([{
            "id": "claimclear", "name": "R", "prompt": "x",
            "schedule": {"kind": "once", "run_at": run_at},
            "next_run_at": run_at, "enabled": True, "state": "scheduled",
            "repeat": {"times": 2, "completed": 0},
        }])
        assert [j["id"] for j in get_due_jobs()] == ["claimclear"]
        assert get_job("claimclear").get("run_claim") is not None
        mark_job_run("claimclear", True)
        assert get_job("claimclear")["run_claim"] is None


    def test_broken_cron_without_next_run_is_recovered(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 3, 18, 10, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        save_jobs(
            [{
                "id": "cron-recover",
                "name": "AI Daily Digest",
                "prompt": "...",
                "schedule": {"kind": "cron", "expr": "0 12 * * *", "display": "0 12 * * *"},
                "schedule_display": "0 12 * * *",
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
                "created_at": "2026-03-18T09:00:00+00:00",
                "next_run_at": None,
                "last_run_at": None,
                "last_status": None,
                "last_error": None,
                "deliver": "local",
                "origin": None,
            }]
        )

        assert get_due_jobs() == []
        recovered = get_job("cron-recover")["next_run_at"]
        assert recovered is not None
        recovered_dt = datetime.fromisoformat(recovered)
        if recovered_dt.tzinfo is None:
            recovered_dt = recovered_dt.replace(tzinfo=timezone.utc)
        assert recovered_dt > now

    def test_broken_interval_without_next_run_is_recovered(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 3, 18, 10, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        save_jobs(
            [{
                "id": "interval-recover",
                "name": "Hourly heartbeat",
                "prompt": "...",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "schedule_display": "every 1h",
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
                "created_at": "2026-03-18T09:00:00+00:00",
                "next_run_at": None,
                "last_run_at": None,
                "last_status": None,
                "last_error": None,
                "deliver": "local",
                "origin": None,
            }]
        )

        assert get_due_jobs() == []
        recovered = get_job("interval-recover")["next_run_at"]
        assert recovered is not None
        recovered_dt = datetime.fromisoformat(recovered)
        if recovered_dt.tzinfo is None:
            recovered_dt = recovered_dt.replace(tzinfo=timezone.utc)
        assert recovered_dt > now


    def test_cron_next_run_offset_migration_is_rescheduled_not_fired(self, tmp_cron_dir, monkeypatch):
        current_tz = timezone(timedelta(hours=2))
        now = datetime(2026, 5, 19, 13, 2, 0, tzinfo=current_tz)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        # A 21:00 cron was stored while Hermes/system local time was UTC+10.
        # After the host moves to UTC+02, that absolute timestamp converts to
        # 13:00+02.  At 13:02+02 the old code considered it due and fired, even
        # though the user's local wall-clock cron intent is still 21:00.
        save_jobs(
            [{
                "id": "cron-tz-migrate",
                "name": "Migrated local cron",
                "prompt": "...",
                "schedule": {"kind": "cron", "expr": "0 21 * * 2", "display": "0 21 * * 2"},
                "schedule_display": "0 21 * * 2",
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
                "created_at": "2026-05-12T21:00:00+10:00",
                "next_run_at": "2026-05-19T21:00:00+10:00",
                "last_run_at": "2026-05-12T21:00:00+10:00",
                "last_status": "ok",
                "last_error": None,
                "deliver": "local",
                "origin": None,
            }]
        )

        assert get_due_jobs() == []
        repaired = datetime.fromisoformat(get_job("cron-tz-migrate")["next_run_at"])
        assert repaired == datetime(2026, 5, 19, 21, 0, 0, tzinfo=current_tz)

    def test_cron_offset_migration_does_not_repair_already_passed_wall_time(self, tmp_cron_dir, monkeypatch):
        current_tz = timezone(timedelta(hours=2))
        now = datetime(2026, 5, 19, 13, 2, 0, tzinfo=current_tz)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        save_jobs(
            [{
                "id": "cron-tz-missed",
                "name": "Migrated missed cron",
                "prompt": "...",
                "schedule": {"kind": "cron", "expr": "0 9 * * 2", "display": "0 9 * * 2"},
                "schedule_display": "0 9 * * 2",
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
                "created_at": "2026-05-12T09:00:00+10:00",
                "next_run_at": "2026-05-19T09:00:00+10:00",
                "last_run_at": "2026-05-12T09:00:00+10:00",
                "last_status": "ok",
                "last_error": None,
                "deliver": "local",
                "origin": None,
            }]
        )

        # The wall-clock time has already passed, so this does NOT take the
        # timezone-migration repair path (which is for still-future wall-clock
        # runs). It falls through to the stale-grace path, which — since #33315
        # — runs the job once now and fast-forwards next_run_at (rather than
        # skipping). The key assertion for THIS test is that the repaired
        # next_run_at is the normal next cron occurrence, not the migration
        # path's same-day rebase.
        due = get_due_jobs()
        assert [j["id"] for j in due] == ["cron-tz-missed"]  # runs once now (#33315)
        repaired = datetime.fromisoformat(get_job("cron-tz-missed")["next_run_at"])
        assert repaired == datetime(2026, 5, 26, 9, 0, 0, tzinfo=current_tz)

    def test_same_tz_due_cron_still_fires(self, tmp_cron_dir, monkeypatch):
        """Guard must NOT over-fire: a due cron in the SAME offset fires normally."""
        current_tz = timezone(timedelta(hours=2))
        now = datetime(2026, 5, 19, 21, 0, 30, tzinfo=current_tz)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        save_jobs([{
            "id": "cron-same-tz", "name": "same tz", "prompt": "...",
            "schedule": {"kind": "cron", "expr": "0 21 * * 2", "display": "0 21 * * 2"},
            "schedule_display": "0 21 * * 2",
            "repeat": {"times": None, "completed": 0},
            "enabled": True, "state": "scheduled", "paused_at": None, "paused_reason": None,
            "created_at": "2026-05-12T21:00:00+02:00",
            "next_run_at": "2026-05-19T21:00:00+02:00",  # same offset as now
            "last_run_at": "2026-05-12T21:00:00+02:00",
            "last_status": "ok", "last_error": None, "deliver": "local", "origin": None,
        }])
        # offset matches -> guard skips -> the genuinely-due job is returned to fire.
        due = get_due_jobs()
        assert [j["id"] for j in due] == ["cron-same-tz"]

    def test_interval_job_with_stale_offset_is_unaffected(self, tmp_cron_dir, monkeypatch):
        """The offset-repair guard is cron-only; interval jobs never take it.

        A stale-offset interval job whose converted instant is well past the
        grace window is handled by the pre-existing stale fast-forward path
        (not the cron repair path). Verify it fast-forwards via interval math
        (next = now + interval), proving the cron-only guard didn't touch it.
        """
        current_tz = timezone(timedelta(hours=2))
        now = datetime(2026, 5, 19, 13, 2, 0, tzinfo=current_tz)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        save_jobs([{
            "id": "interval-stale-tz", "name": "interval", "prompt": "...",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 1h"},
            "schedule_display": "every 1h",
            "repeat": {"times": None, "completed": 0},
            "enabled": True, "state": "scheduled", "paused_at": None, "paused_reason": None,
            "created_at": "2026-05-19T10:00:00+10:00",
            "next_run_at": "2026-05-19T12:00:00+10:00",  # stale offset, instant 04:00+02 (well past)
            "last_run_at": "2026-05-19T11:00:00+10:00",
            "last_status": "ok", "last_error": None, "deliver": "local", "origin": None,
        }])
        get_due_jobs()
        # The cron-only repair path would have produced a cron occurrence; instead
        # the interval stale fast-forward recomputes next = now + 60m (interval
        # math), confirming the guard did not intercept this interval job.
        nr = datetime.fromisoformat(get_job("interval-stale-tz")["next_run_at"])
        assert nr == now + timedelta(minutes=60)

    def test_offset_migration_at_wall_clock_equal_now_falls_through(self, tmp_cron_dir, monkeypatch):
        """Boundary: stored wall-clock == now wall-clock (strict >) does NOT take
        the repair path — it falls through to the existing due/fast-forward logic."""
        current_tz = timezone(timedelta(hours=2))
        now = datetime(2026, 5, 19, 13, 0, 0, tzinfo=current_tz)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        save_jobs([{
            "id": "cron-wall-equal", "name": "wall equal", "prompt": "...",
            "schedule": {"kind": "cron", "expr": "0 13 * * 2", "display": "0 13 * * 2"},
            "schedule_display": "0 13 * * 2",
            "repeat": {"times": None, "completed": 0},
            "enabled": True, "state": "scheduled", "paused_at": None, "paused_reason": None,
            "created_at": "2026-05-12T13:00:00+10:00",
            # stored naive wall-clock 13:00 == now naive wall-clock 13:00 -> strict > is False
            "next_run_at": "2026-05-19T13:00:00+10:00",
            "last_run_at": "2026-05-12T13:00:00+10:00",
            "last_status": "ok", "last_error": None, "deliver": "local", "origin": None,
        }])
        # _stored_wall_clock_is_future is strict (>), so 13:00 == 13:00 is False
        # -> repair guard skipped -> existing logic handles it (does not raise).
        get_due_jobs()  # must not raise / must not take the repair branch
        # next_run_at must NOT have been rewritten to a future cron occurrence by
        # the repair path (it either fires or fast-forwards via the normal path).
        nr = get_job("cron-wall-equal")["next_run_at"]
        assert nr is None or datetime.fromisoformat(nr).utcoffset() == now.utcoffset() or "+10:00" in nr


class TestEnabledToolsets:
    def test_enabled_toolsets_stored(self, tmp_cron_dir):
        job = create_job(prompt="monitor", schedule="every 1h", enabled_toolsets=["web", "terminal"])
        assert job["enabled_toolsets"] == ["web", "terminal"]

    def test_enabled_toolsets_persisted(self, tmp_cron_dir):
        job = create_job(prompt="monitor", schedule="every 1h", enabled_toolsets=["web", "file"])
        fetched = get_job(job["id"])
        assert fetched["enabled_toolsets"] == ["web", "file"]

    def test_enabled_toolsets_none_when_omitted(self, tmp_cron_dir):
        job = create_job(prompt="monitor", schedule="every 1h")
        assert job["enabled_toolsets"] is None

    def test_enabled_toolsets_empty_list_normalizes_to_none(self, tmp_cron_dir):
        job = create_job(prompt="monitor", schedule="every 1h", enabled_toolsets=[])
        assert job["enabled_toolsets"] is None

    def test_enabled_toolsets_whitespace_entries_stripped(self, tmp_cron_dir):
        job = create_job(prompt="monitor", schedule="every 1h", enabled_toolsets=["web", " ", "file"])
        assert job["enabled_toolsets"] == ["web", "file"]

    def test_enabled_toolsets_updated_via_update_job(self, tmp_cron_dir):
        job = create_job(prompt="monitor", schedule="every 1h")
        update_job(job["id"], {"enabled_toolsets": ["web", "delegation"]})
        fetched = get_job(job["id"])
        assert fetched["enabled_toolsets"] == ["web", "delegation"]


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

    def test_repeated_concurrent_runs_accumulate_completed_count(self, tmp_cron_dir):
        """Stress test: 10 threads each call mark_job_run on a different job once.

        The completed count for every job must be exactly 1 after all threads finish,
        confirming no thread's write was silently dropped.
        """
        n = 10
        jobs = [create_job(prompt=f"Stress job {i}", schedule="every 1h") for i in range(n)]
        errors: list = []

        def run_mark(job_id: str):
            try:
                mark_job_run(job_id, success=True)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=run_mark, args=(j["id"],)) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected exceptions: {errors}"

        for job in jobs:
            updated = get_job(job["id"])
            assert updated is not None, f"Job {job['id']} was deleted"
            assert updated["last_status"] == "ok", (
                f"Job {job['id']} has wrong last_status: {updated['last_status']}"
            )
            assert updated["repeat"]["completed"] == 1, (
                f"Job {job['id']} completed count is {updated['repeat']['completed']}, expected 1"
            )


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

    @pytest.mark.parametrize("bad_job_id", ["../escape", "nested/escape", ".", "..", ""])
    def test_rejects_unsafe_job_id(self, tmp_cron_dir, bad_job_id):
        """Path-escape attempts must fail closed and never create dirs."""
        with pytest.raises(ValueError, match="output path"):
            save_job_output(bad_job_id, "# Results")
        assert not (tmp_cron_dir / "escape").exists()

    def test_rejects_absolute_job_id(self, tmp_cron_dir):
        """Absolute paths as job IDs must fail closed."""
        with pytest.raises(ValueError, match="output path"):
            save_job_output(str(tmp_cron_dir / "outside"), "# Results")
        assert not (tmp_cron_dir / "outside").exists()


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

    def test_prune_noop_when_under_cap(self, tmp_path):
        from cron.jobs import _prune_job_output
        d = tmp_path / "job"
        self._seed(d, 3)
        assert _prune_job_output(d, keep=5) == 0
        assert len(list(d.glob("*.md"))) == 3

    def test_prune_disabled_when_keep_non_positive(self, tmp_path):
        from cron.jobs import _prune_job_output
        d = tmp_path / "job"
        self._seed(d, 5)
        assert _prune_job_output(d, keep=0) == 0
        assert _prune_job_output(d, keep=-1) == 0
        assert len(list(d.glob("*.md"))) == 5

    def test_prune_ignores_non_md_and_temp_files(self, tmp_path):
        from cron.jobs import _prune_job_output
        d = tmp_path / "job"
        self._seed(d, 4)
        (d / ".output_abc.tmp").write_text("partial")
        (d / "manifest.json").write_text("{}")
        _prune_job_output(d, keep=2)
        assert (d / ".output_abc.tmp").exists()
        assert (d / "manifest.json").exists()
        assert len(list(d.glob("*.md"))) == 2

    def test_save_job_output_prunes_old_runs(self, tmp_cron_dir, monkeypatch):
        from cron.jobs import save_job_output, _job_output_dir
        monkeypatch.setattr("cron.jobs._cron_output_keep", lambda: 3)
        seq = iter(
            datetime(2026, 6, 25, 10, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=i)
            for i in range(8)
        )
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: next(seq))
        for _ in range(8):
            save_job_output("job1", "report")
        files = sorted(_job_output_dir("job1").glob("*.md"))
        assert len(files) == 3  # only the 3 most-recent runs survive

    def test_cron_output_keep_reads_config(self, monkeypatch):
        import cron.jobs as jobs
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {"cron": {"output_retention": 7}}
        )
        assert jobs._cron_output_keep() == 7

    def test_cron_output_keep_defaults_on_bad_config(self, monkeypatch):
        import cron.jobs as jobs
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {"cron": {"output_retention": "oops"}}
        )
        assert jobs._cron_output_keep() == jobs._CRON_OUTPUT_DEFAULT_KEEP


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

    def test_recurring_job_is_not_claimed(self, tmp_cron_dir):
        job = {
            "id": "rec",
            "schedule": {"kind": "interval", "minutes": 5},
            "repeat": {"times": 3, "completed": 0},
        }
        save_jobs([job])
        assert claim_dispatch("rec") is True
        # Recurring jobs use advance_next_run(); claim must NOT touch completed.
        assert load_jobs()[0]["repeat"]["completed"] == 0

    def test_infinite_oneshot_not_claimed(self, tmp_cron_dir):
        job = self._oneshot(times=0, completed=0)  # times<=0 means infinite
        save_jobs([job])
        assert claim_dispatch("os1") is True
        assert load_jobs()[0]["repeat"]["completed"] == 0

    def test_no_repeat_block_not_claimed(self, tmp_cron_dir):
        job = {"id": "os1", "schedule": {"kind": "once", "run_at": "2026-01-01T00:00:00+00:00"}}
        save_jobs([job])
        assert claim_dispatch("os1") is True
        assert "repeat" not in load_jobs()[0]

    def test_missing_job_proceeds(self, tmp_cron_dir):
        # A handed-in job dict not persisted in the store (external provider /
        # direct caller) can't be claimed — proceed rather than suppress it.
        save_jobs([])
        assert claim_dispatch("ghost") is True

    def test_mark_job_run_does_not_double_count_preclaimed_oneshot(self, tmp_cron_dir):
        # Full lifecycle: claim bumps completed to times, then mark_job_run must
        # NOT increment again — it recognizes the pre-claim and removes the job.
        save_jobs([self._oneshot(times=1, completed=0)])
        assert claim_dispatch("os1") is True
        assert load_jobs()[0]["repeat"]["completed"] == 1
        mark_job_run("os1", success=True)
        assert load_jobs() == []  # completed once, removed — not fired twice

    def test_mark_job_run_still_increments_recurring(self, tmp_cron_dir):
        # The double-count guard is one-shot-specific; recurring jobs keep the
        # legacy post-run increment.
        job = {
            "id": "rec",
            "schedule": {"kind": "interval", "minutes": 5},
            "repeat": {"times": 3, "completed": 1},
        }
        save_jobs([job])
        mark_job_run("rec", success=True)
        assert load_jobs()[0]["repeat"]["completed"] == 2

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

    def test_bad_schedule_does_not_crash_or_block_sibling_jobs(self, tmp_cron_dir):
        """Regression for a job with non-dict 'schedule' (null / string / etc.

        from direct jobs.json edit or old writer).

        Such a record must not raise in _get_due_jobs_locked and must not
        prevent healthy sibling jobs from being returned or having their
        next_run_at advanced+persisted. Mirrors the id-less job P1 pattern.
        """
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

        bad = {
            "id": "bad-sched",
            "name": "bad",
            "enabled": True,
            "schedule": None,  # poison: not a dict
            "next_run_at": future,  # not due
        }
        good = {
            "id": "good",
            "name": "good",
            "enabled": True,
            "schedule": {"kind": "interval", "minutes": 5},
            "next_run_at": past,
        }
        save_jobs([bad, good])

        due = get_due_jobs()
        due_ids = [j["id"] for j in due]
        assert "good" in due_ids
        assert "bad-sched" not in due_ids  # bad one ignored, no crash

        # At minimum, the good job's record is still intact (no corruption from the bad neighbor)
        loaded = {j["id"]: j for j in load_jobs()}
        assert "good" in loaded


class TestScheduleNoneSiblingCrashes:
    """A non-dict 'schedule' (direct jobs.json edit, old writer, corruption)

    is repaired by ``_get_due_jobs_locked`` during the periodic due-scan tick,
    but ``resume_job``/``mark_job_run``/``claim_dispatch``/``advance_next_run``/
    ``update_job`` can all run on a record the scan hasn't repaired yet (a
    paused job, or a call made before the scheduler's first tick) — each read
    ``schedule.get(...)`` directly, and a non-dict value raised
    ``AttributeError`` before this fix. Sibling of the id-less/malformed
    next_run_at/non-dict schedule due-scan fixes, but for the direct-call
    paths rather than the scan.
    """

    def _write_bad_job(self, job_id="bad-sched", **overrides):
        job = {
            "id": job_id,
            "name": "bad",
            "prompt": "test",
            "enabled": True,
            "schedule": None,  # poison: not a dict
            "state": "scheduled",
            "next_run_at": None,
        }
        job.update(overrides)
        save_jobs([job])
        return job

    def test_resume_job_does_not_crash(self, tmp_cron_dir):
        self._write_bad_job(enabled=False, state="paused")
        result = resume_job("bad-sched")
        assert result is not None
        assert result["schedule"] == {}

    def test_mark_job_run_does_not_crash(self, tmp_cron_dir):
        self._write_bad_job(enabled=False, state="paused")
        mark_job_run("bad-sched", success=True)  # must not raise
        assert load_jobs()[0]["schedule"] == {}

    def test_claim_dispatch_does_not_crash(self, tmp_cron_dir):
        self._write_bad_job(enabled=False, state="paused")
        assert claim_dispatch("bad-sched") is True
        # The in-place schedule repair must be persisted, not just avoid the
        # crash in memory — otherwise jobs.json stays malformed on disk and
        # every future call pays the same repair-without-persist cost.
        assert load_jobs()[0]["schedule"] == {}

    def test_advance_next_run_does_not_crash(self, tmp_cron_dir):
        self._write_bad_job(enabled=False, state="paused")
        assert advance_next_run("bad-sched") is False
        assert load_jobs()[0]["schedule"] == {}

    def test_update_job_does_not_crash_on_unrelated_field(self, tmp_cron_dir):
        """Updating a field other than 'schedule' must not crash just
        because the stored schedule is malformed."""
        self._write_bad_job(enabled=True, state="scheduled")
        result = update_job("bad-sched", {"assignee": "someone"})
        assert result is not None
        assert result["schedule"] == {}
        assert result["assignee"] == "someone"

    def test_update_job_still_parses_raw_string_schedule(self, tmp_cron_dir):
        """The malformed-schedule guard must not swallow a legitimate raw
        string schedule update (e.g. "every 10m") before it's parsed."""
        job = create_job(prompt="hi", schedule="every 1h")
        updated = update_job(job["id"], {"schedule": "every 2h"})
        assert updated["schedule"]["kind"] == "interval"
        assert updated["schedule"]["minutes"] == 120
