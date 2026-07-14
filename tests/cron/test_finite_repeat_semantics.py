"""HARNESS-P1: cron finite-repeat / schedule semantics.

Bare relative durations ('20m') are one-shot delays. Combining them with
repeat>1 previously created a job that fired once, recorded 1/N, then
completed with no next_run_at — silently dropping the remaining N-1 runs.

These tests pin the rejection, correct recurring finite-repeat behavior,
tool response semantics, diagnostics, and related edge cases.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cron.jobs import (
    claim_dispatch,
    compute_next_run,
    create_job,
    describe_schedule_semantics,
    find_unsatisfiable_repeat_jobs,
    get_job,
    load_jobs,
    mark_job_run,
    parse_schedule,
    pause_job,
    preview_next_runs,
    resume_job,
    save_jobs,
    update_job,
    validate_repeat_for_schedule,
)


@pytest.fixture
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# ---------------------------------------------------------------------------
# Parser + validation
# ---------------------------------------------------------------------------


class TestRelativeVsEverySemantics:
    def test_bare_duration_is_once_not_interval(self):
        result = parse_schedule("20m")
        assert result["kind"] == "once"
        assert "run_at" in result

    def test_every_duration_is_interval(self):
        result = parse_schedule("every 20m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 20

    def test_iso_timestamp_is_once(self, monkeypatch):
        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        future = (now + timedelta(hours=1)).isoformat()
        result = parse_schedule(future)
        assert result["kind"] == "once"


class TestValidateRepeatForSchedule:
    def test_rejects_once_with_repeat_gt_1(self):
        schedule = parse_schedule("20m")
        with pytest.raises(ValueError, match="cannot be satisfied by one-shot"):
            validate_repeat_for_schedule(schedule, 8)

    def test_allows_once_with_repeat_1(self):
        validate_repeat_for_schedule(parse_schedule("20m"), 1)

    def test_allows_once_with_none(self):
        # create_job will auto-set times=1; validator itself permits None.
        validate_repeat_for_schedule(parse_schedule("20m"), None)

    def test_allows_interval_with_finite_repeat(self):
        validate_repeat_for_schedule(parse_schedule("every 20m"), 8)

    def test_allows_cron_with_finite_repeat(self):
        validate_repeat_for_schedule(
            {"kind": "cron", "expr": "*/5 * * * *", "display": "*/5 * * * *"},
            3,
        )


# ---------------------------------------------------------------------------
# Create / update rejection (incident reproduction)
# ---------------------------------------------------------------------------


class TestCreateRejectsUnsatisfiableRepeat:
    def test_incident_shape_20m_repeat_8_rejected(self, tmp_cron_dir):
        """Reproduce Blazar's edd042027e2b shape: schedule='20m', repeat=8."""
        with pytest.raises(ValueError, match="every 20m"):
            create_job(prompt="fleet update", schedule="20m", repeat=8)
        assert load_jobs() == []

    def test_iso_oneshot_repeat_gt_1_rejected(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        future = (now + timedelta(minutes=10)).isoformat()
        with pytest.raises(ValueError, match="one-shot"):
            create_job(prompt="iso", schedule=future, repeat=3)
        assert load_jobs() == []

    def test_once_repeat_1_still_ok(self, tmp_cron_dir):
        job = create_job(prompt="once", schedule="30m", repeat=1)
        assert job["schedule"]["kind"] == "once"
        assert job["repeat"]["times"] == 1

    def test_every_20m_repeat_8_accepted(self, tmp_cron_dir):
        job = create_job(prompt="fleet", schedule="every 20m", repeat=8)
        assert job["schedule"]["kind"] == "interval"
        assert job["schedule"]["minutes"] == 20
        assert job["repeat"]["times"] == 8
        assert job["next_run_at"] is not None


class TestUpdateRejectsUnsatisfiableRepeat:
    def test_update_repeat_on_oneshot_rejected(self, tmp_cron_dir):
        job = create_job(prompt="once", schedule="30m")
        with pytest.raises(ValueError, match="one-shot"):
            update_job(job["id"], {"repeat": {"times": 5, "completed": 0}})
        assert get_job(job["id"])["repeat"]["times"] == 1

    def test_update_schedule_to_oneshot_with_finite_repeat_rejected(self, tmp_cron_dir):
        job = create_job(prompt="recurring", schedule="every 1h", repeat=5)
        with pytest.raises(ValueError, match="one-shot"):
            update_job(job["id"], {"schedule": parse_schedule("45m")})
        # Unchanged
        fetched = get_job(job["id"])
        assert fetched["schedule"]["kind"] == "interval"
        assert fetched["repeat"]["times"] == 5

    def test_update_interval_finite_repeat_ok(self, tmp_cron_dir):
        job = create_job(prompt="recurring", schedule="every 1h")
        updated = update_job(job["id"], {"repeat": {"times": 4, "completed": 0}})
        assert updated["repeat"]["times"] == 4


# ---------------------------------------------------------------------------
# Finite recurring repeat tracking
# ---------------------------------------------------------------------------


class TestIntervalFiniteRepeat:
    def test_every_duration_finite_repeat_counts_and_removes(self, tmp_cron_dir):
        job = create_job(prompt="finite", schedule="every 20m", repeat=3)
        jid = job["id"]

        mark_job_run(jid, success=True)
        j1 = get_job(jid)
        assert j1 is not None
        assert j1["repeat"]["completed"] == 1
        assert j1["next_run_at"] is not None
        assert j1["state"] == "scheduled"

        mark_job_run(jid, success=True)
        j2 = get_job(jid)
        assert j2["repeat"]["completed"] == 2
        assert j2["next_run_at"] is not None

        mark_job_run(jid, success=True)
        assert get_job(jid) is None  # auto-removed at limit

    def test_delivery_failure_still_counts_toward_repeat(self, tmp_cron_dir):
        job = create_job(prompt="deliv", schedule="every 5m", repeat=2)
        mark_job_run(job["id"], success=True, delivery_error="platform down")
        mid = get_job(job["id"])
        assert mid["repeat"]["completed"] == 1
        assert mid["last_delivery_error"] == "platform down"
        assert mid["next_run_at"] is not None
        mark_job_run(job["id"], success=True)
        assert get_job(job["id"]) is None


class TestCronFiniteRepeat:
    def test_cron_expr_finite_repeat(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        job = create_job(prompt="cron finite", schedule="*/5 * * * *", repeat=2)
        assert job["schedule"]["kind"] == "cron"
        assert job["repeat"]["times"] == 2
        assert job["next_run_at"] is not None

        mark_job_run(job["id"], success=True)
        mid = get_job(job["id"])
        assert mid["repeat"]["completed"] == 1
        assert mid["next_run_at"] is not None

        mark_job_run(job["id"], success=True)
        assert get_job(job["id"]) is None


class TestIsoOneshot:
    def test_iso_oneshot_fires_once_and_completes(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        future = (now + timedelta(minutes=5)).isoformat()
        job = create_job(prompt="iso once", schedule=future)
        assert job["repeat"]["times"] == 1
        # claim + mark path for finite oneshot
        assert claim_dispatch(job["id"]) is True
        mark_job_run(job["id"], success=True)
        # times=1 → removed after claim+mark
        assert get_job(job["id"]) is None


# ---------------------------------------------------------------------------
# Pause / resume / restart / catch-up
# ---------------------------------------------------------------------------


class TestPauseResumeFinite:
    def test_pause_resume_preserves_finite_count(self, tmp_cron_dir):
        job = create_job(prompt="p", schedule="every 10m", repeat=5)
        mark_job_run(job["id"], success=True)
        paused = pause_job(job["id"], reason="ops")
        assert paused["state"] == "paused"
        assert paused["repeat"]["completed"] == 1
        resumed = resume_job(job["id"])
        assert resumed["state"] == "scheduled"
        assert resumed["repeat"]["completed"] == 1
        assert resumed["repeat"]["times"] == 5
        assert resumed["next_run_at"] is not None


class TestProcessRestartPreservesState:
    def test_reload_jobs_preserves_completed_count(self, tmp_cron_dir):
        job = create_job(prompt="r", schedule="every 15m", repeat=4)
        mark_job_run(job["id"], success=True)
        # Simulate process restart: re-load from disk via get_job/load_jobs
        reloaded = load_jobs()
        assert len(reloaded) == 1
        assert reloaded[0]["repeat"]["completed"] == 1
        assert reloaded[0]["repeat"]["times"] == 4
        assert reloaded[0]["next_run_at"] is not None


class TestMissedTickCatchup:
    def test_stale_interval_still_advances_and_counts(self, tmp_cron_dir, monkeypatch):
        """A stale next_run_at still consumes one finite-repeat slot on fire."""
        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        job = create_job(prompt="stale", schedule="every 5m", repeat=3)
        # Backdate next_run_at far into the past (missed ticks)
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (now - timedelta(hours=2)).isoformat()
        save_jobs(jobs)

        mark_job_run(job["id"], success=True)
        after = get_job(job["id"])
        assert after["repeat"]["completed"] == 1
        # next_run_at re-anchored from last_run (now), not forever-stuck
        assert after["next_run_at"] is not None
        next_dt = datetime.fromisoformat(after["next_run_at"])
        assert next_dt > now - timedelta(seconds=1)


# ---------------------------------------------------------------------------
# Preview / semantics / diagnostics
# ---------------------------------------------------------------------------


class TestSemanticsAndPreview:
    def test_describe_once(self):
        s = parse_schedule("20m")
        text = describe_schedule_semantics(s, 1)
        assert "one-shot" in text
        assert "every" in text.lower() or "not an interval" in text

    def test_describe_interval_finite(self):
        s = parse_schedule("every 20m")
        text = describe_schedule_semantics(s, 8)
        assert "finite repeat 8" in text

    def test_preview_once_has_single_tick(self, monkeypatch):
        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        s = parse_schedule("20m")
        ticks = preview_next_runs(s, n=2)
        assert len(ticks) == 1

    def test_preview_interval_has_two_ticks(self, monkeypatch):
        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        s = parse_schedule("every 20m")
        ticks = preview_next_runs(s, n=2)
        assert len(ticks) == 2
        t0 = datetime.fromisoformat(ticks[0])
        t1 = datetime.fromisoformat(ticks[1])
        assert t1 - t0 == timedelta(minutes=20)


class TestDiagnosticUnsatisfiable:
    def test_finds_legacy_oneshot_repeat_n(self, tmp_cron_dir, monkeypatch):
        """Legacy jobs already on disk are NOT mutated — only warned about."""
        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        # Bypass create_job validation by writing a legacy record directly.
        save_jobs(
            [
                {
                    "id": "legacydeadbe",
                    "name": "stuck-fleet",
                    "prompt": "update",
                    "schedule": {
                        "kind": "once",
                        "run_at": (now + timedelta(minutes=20)).isoformat(),
                        "display": "once in 20m",
                    },
                    "schedule_display": "once in 20m",
                    "repeat": {"times": 8, "completed": 1},
                    "enabled": False,
                    "state": "completed",
                    "next_run_at": None,
                    "last_run_at": now.isoformat(),
                }
            ]
        )
        issues = find_unsatisfiable_repeat_jobs()
        assert len(issues) == 1
        assert issues[0]["job_id"] == "legacydeadbe"
        assert "1/8" in issues[0]["warning"] or "1/8" in f"{issues[0]['completed']}/{issues[0]['times']}"
        # Store preserved
        assert load_jobs()[0]["repeat"]["times"] == 8


# ---------------------------------------------------------------------------
# Tool response surface
# ---------------------------------------------------------------------------


class TestToolResponseSemantics:
    def test_create_response_includes_semantics_and_ticks(self, tmp_cron_dir, monkeypatch):
        from tools.cronjob_tools import cronjob
        import json

        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        raw = cronjob(
            action="create",
            prompt="fleet",
            schedule="every 20m",
            repeat=8,
            deliver="local",
            name="fleet-updates",
        )
        data = json.loads(raw)
        assert data["success"] is True
        assert data["schedule_kind"] == "interval"
        assert "finite repeat 8" in data["semantics"]
        assert len(data["expected_next_ticks"]) == 2
        assert data["repeat"] == "8 times"

    def test_create_tool_rejects_20m_repeat_8(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob
        import json

        raw = cronjob(
            action="create",
            prompt="fleet",
            schedule="20m",
            repeat=8,
            deliver="local",
        )
        data = json.loads(raw)
        assert data["success"] is False
        assert "one-shot" in data["error"].lower() or "cannot be satisfied" in data["error"]

    def test_list_warns_on_legacy_stuck(self, tmp_cron_dir, monkeypatch):
        from tools.cronjob_tools import cronjob
        import json

        now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
        save_jobs(
            [
                {
                    "id": "legacywarn01",
                    "name": "stuck",
                    "prompt": "x",
                    "schedule": {
                        "kind": "once",
                        "run_at": now.isoformat(),
                        "display": "once in 20m",
                    },
                    "schedule_display": "once in 20m",
                    "repeat": {"times": 8, "completed": 1},
                    "enabled": False,
                    "state": "completed",
                    "next_run_at": None,
                }
            ]
        )
        data = json.loads(cronjob(action="list", include_disabled=True))
        assert data["success"] is True
        assert data.get("warnings")
        assert any(w["job_id"] == "legacywarn01" for w in data["warnings"])


# ---------------------------------------------------------------------------
# Final state / count invariants
# ---------------------------------------------------------------------------


class TestFinalStateAndCount:
    def test_oneshot_default_final_removed(self, tmp_cron_dir):
        job = create_job(prompt="o", schedule="30m")
        claim_dispatch(job["id"])
        mark_job_run(job["id"], success=True)
        assert get_job(job["id"]) is None

    def test_interval_forever_never_removed(self, tmp_cron_dir):
        job = create_job(prompt="forever", schedule="every 1h")
        for _ in range(5):
            mark_job_run(job["id"], success=True)
        assert get_job(job["id"]) is not None
        assert get_job(job["id"])["repeat"]["completed"] == 5
        assert get_job(job["id"])["repeat"]["times"] is None

    def test_compute_next_run_once_after_last_is_none(self):
        s = {"kind": "once", "run_at": "2099-01-01T00:00:00+00:00"}
        assert compute_next_run(s, last_run_at="2099-01-01T00:00:01+00:00") is None
