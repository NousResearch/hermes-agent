"""Tests for internal cron.jobs functions that lack direct coverage.

These functions are exercised through the public API but benefit from
direct unit tests to guard edge cases and ensure behavior stability.
"""

import pytest
from datetime import datetime, timedelta, timezone

from cron.jobs import (
    _job_output_dir,
    _schedule_display_for_job,
    _normalize_job_record,
    _compute_grace_seconds,
    parse_schedule,
    compute_next_run,
)


class TestJobOutputDir:
    """Tests for _job_output_dir — path safety for job output directories."""

    def test_valid_job_id_returns_output_path(self):
        result = _job_output_dir("abc123deadbe")
        assert result.name == "abc123deadbe"

    def test_valid_job_id_with_hyphens(self):
        result = _job_output_dir("job-123-abc")
        assert result.name == "job-123-abc"

    def test_valid_job_id_with_underscores(self):
        result = _job_output_dir("job_123_abc")
        assert result.name == "job_123_abc"

    def test_empty_job_id_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir("")

    def test_dot_job_id_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir(".")

    def test_dotdot_job_id_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir("..")

    def test_slash_in_job_id_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir("abc/def")

    def test_backslash_in_job_id_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir("abc\\def")

    def test_absolute_path_job_id_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir("/etc/passwd")

    def test_drive_letter_job_id_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir("C:\\Windows")

    def test_job_id_with_path_traversal_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir("../escape")

    def test_job_id_with_nested_path_raises(self):
        with pytest.raises(ValueError, match="Invalid cron job id"):
            _job_output_dir("subdir/job")


class TestScheduleDisplayForJob:
    """Tests for _schedule_display_for_job — human-readable schedule strings."""

    def test_explicit_schedule_display_used(self):
        job = {"schedule_display": "every 30m"}
        assert _schedule_display_for_job(job) == "every 30m"

    def test_schedule_display_preferred_over_schedule_dict_display(self):
        job = {
            "schedule_display": "custom display",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        }
        assert _schedule_display_for_job(job) == "custom display"

    def test_schedule_dict_with_display_key(self):
        job = {"schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"}}
        assert _schedule_display_for_job(job) == "every 60m"

    def test_schedule_dict_with_value_key(self):
        job = {"schedule": {"kind": "interval", "minutes": 60, "value": "every 60m"}}
        assert _schedule_display_for_job(job) == "every 60m"

    def test_schedule_dict_with_expr_key(self):
        job = {"schedule": {"kind": "cron", "expr": "0 9 * * *", "expr": "0 9 * * *"}}
        assert _schedule_display_for_job(job) == "0 9 * * *"

    def test_schedule_dict_with_run_at_key(self):
        job = {"schedule": {"kind": "once", "run_at": "2026-01-01T12:00:00"}}
        assert _schedule_display_for_job(job) == "2026-01-01T12:00:00"

    def test_raw_schedule_string(self):
        job = {"schedule": "every 2h"}
        assert _schedule_display_for_job(job) == "every 2h"

    def test_missing_schedule_returns_question_mark(self):
        job = {}
        assert _schedule_display_for_job(job) == "?"

    def test_none_schedule_returns_question_mark(self):
        job = {"schedule": None}
        assert _schedule_display_for_job(job) == "?"

    def test_empty_schedule_display_falls_back_to_schedule(self):
        job = {
            "schedule_display": "",
            "schedule": {"kind": "interval", "minutes": 30, "display": "every 30m"},
        }
        assert _schedule_display_for_job(job) == "every 30m"

    def test_whitespace_only_schedule_display_falls_back(self):
        job = {
            "schedule_display": "   ",
            "schedule": {"kind": "interval", "minutes": 30, "display": "every 30m"},
        }
        assert _schedule_display_for_job(job) == "every 30m"


class TestNormalizeJobRecord:
    """Tests for _normalize_job_record — read-safe job shape for consumers."""

    def test_basic_job_normalized(self):
        job = {
            "id": "abc123",
            "prompt": "Check server",
            "name": "Server Check",
            "enabled": True,
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["id"] == "abc123"
        assert result["prompt"] == "Check server"
        assert result["name"] == "Server Check"
        assert result["state"] == "scheduled"
        assert result["schedule_display"] == "every 60m"
        assert "skills" in result
        assert "skill" in result

    def test_none_prompt_becomes_empty_string(self):
        job = {
            "id": "abc123",
            "prompt": None,
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["prompt"] == ""

    def test_missing_name_derived_from_prompt(self):
        job = {
            "id": "abc123",
            "prompt": "This is a long prompt that should be truncated at fifty chars",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        # Name is truncated to 50 chars then stripped (trailing space removed)
        assert result["name"] == "This is a long prompt that should be truncated at"
        assert len(result["name"]) == 49

    def test_missing_name_and_prompt_uses_skill(self):
        job = {
            "id": "abc123",
            "prompt": "",
            "skills": ["blogwatcher"],
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["name"] == "blogwatcher"

    def test_missing_name_prompt_skill_uses_script(self):
        job = {
            "id": "abc123",
            "prompt": "",
            "script": "myscript.sh",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["name"] == "myscript.sh"

    def test_missing_everything_uses_id(self):
        job = {
            "id": "abc123deadbeef",
            "prompt": "",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["name"] == "abc123deadbeef"

    def test_legacy_skill_field_converted_to_skills(self):
        job = {
            "id": "abc123",
            "skill": "legacy-skill",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["skills"] == ["legacy-skill"]
        assert result["skill"] == "legacy-skill"

    def test_skills_field_preserved(self):
        job = {
            "id": "abc123",
            "skills": ["skill-a", "skill-b"],
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["skills"] == ["skill-a", "skill-b"]
        assert result["skill"] == "skill-a"

    def test_both_skill_and_skills_merged_unique(self):
        job = {
            "id": "abc123",
            "skill": "skill-a",
            "skills": ["skill-a", "skill-b"],
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["skills"] == ["skill-a", "skill-b"]
        assert result["skill"] == "skill-a"

    def test_missing_state_defaults_to_scheduled_when_enabled(self):
        job = {
            "id": "abc123",
            "enabled": True,
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["state"] == "scheduled"

    def test_missing_state_defaults_to_paused_when_disabled(self):
        job = {
            "id": "abc123",
            "enabled": False,
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["state"] == "paused"

    def test_explicit_state_preserved(self):
        job = {
            "id": "abc123",
            "enabled": True,
            "state": "error",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "repeat": {"times": None, "completed": 0},
        }
        result = _normalize_job_record(job)
        assert result["state"] == "error"

    def test_repeat_none_times_preserved(self):
        job = {
            "id": "abc123",
            "repeat": {"times": None, "completed": 3},
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        }
        result = _normalize_job_record(job)
        assert result["repeat"]["times"] is None
        assert result["repeat"]["completed"] == 3


class TestComputeGraceSeconds:
    """Tests for _compute_grace_seconds — grace periods for recurring jobs."""

    def test_interval_job_grace_half_period(self):
        schedule = {"kind": "interval", "minutes": 60}  # 1 hour
        grace = _compute_grace_seconds(schedule)
        # Half of 3600s = 1800s, clamped between 120 and 7200
        assert grace == 1800

    def test_interval_job_grace_minimum_clamp(self):
        schedule = {"kind": "interval", "minutes": 1}  # 1 minute = 60s
        grace = _compute_grace_seconds(schedule)
        # Half of 60s = 30s, clamped to minimum 120s
        assert grace == 120

    def test_interval_job_grace_maximum_clamp(self):
        schedule = {"kind": "interval", "minutes": 300}  # 5 hours = 18000s
        grace = _compute_grace_seconds(schedule)
        # Half of 18000s = 9000s, clamped to maximum 7200s
        assert grace == 7200

    def test_interval_job_zero_minutes_treated_as_one(self):
        schedule = {"kind": "interval", "minutes": 0}
        grace = _compute_grace_seconds(schedule)
        # Half of 60s (0 treated as 1) = 30s, clamped to 120s
        assert grace == 120

    def test_cron_job_grace_computed(self):
        schedule = {"kind": "cron", "expr": "0 9 * * *"}  # Daily at 9am
        grace = _compute_grace_seconds(schedule)
        # Should be roughly half a day in seconds, clamped
        assert 120 <= grace <= 7200

    def test_cron_job_grace_falls_back_to_minimum_on_error(self):
        schedule = {"kind": "cron", "expr": "invalid cron"}
        grace = _compute_grace_seconds(schedule)
        assert grace == 120

    def test_unknown_kind_returns_minimum(self):
        schedule = {"kind": "unknown"}
        grace = _compute_grace_seconds(schedule)
        assert grace == 120

    def test_cron_job_without_croniter_returns_minimum(self, monkeypatch):
        import cron.jobs as jobs_mod
        monkeypatch.setattr(jobs_mod, "HAS_CRONITER", False)
        schedule = {"kind": "cron", "expr": "0 9 * * *"}
        grace = _compute_grace_seconds(schedule)
        assert grace == 120


class TestParseScheduleIntegration:
    """Integration tests for parse_schedule — validate parsing behavior."""

    def test_parse_interval_every_minutes(self):
        result = parse_schedule("every 30m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 30
        assert result["display"] == "every 30m"

    def test_parse_interval_every_hours(self):
        result = parse_schedule("every 2h")
        assert result["kind"] == "interval"
        assert result["minutes"] == 120

    def test_parse_interval_every_days(self):
        result = parse_schedule("every 1d")
        assert result["kind"] == "interval"
        assert result["minutes"] == 1440

    def test_parse_cron_expression(self):
        result = parse_schedule("0 9 * * *")
        assert result["kind"] == "cron"
        assert result["expr"] == "0 9 * * *"
        assert result["display"] == "0 9 * * *"

    def test_parse_iso_timestamp(self):
        result = parse_schedule("2026-01-15T14:30:00")
        assert result["kind"] == "once"
        assert "run_at" in result
        assert result["display"].startswith("once at 2026-01-15")

    def test_parse_iso_timestamp_with_z(self):
        result = parse_schedule("2026-01-15T14:30:00Z")
        assert result["kind"] == "once"
        assert result["run_at"].endswith("+00:00") or "Z" in result["run_at"]

    def test_parse_date_only(self):
        result = parse_schedule("2026-01-15")
        assert result["kind"] == "once"
        assert "run_at" in result

    def test_parse_duration_one_shot(self):
        result = parse_schedule("30m")
        assert result["kind"] == "once"
        assert "run_at" in result
        assert result["display"] == "once in 30m"

    def test_parse_duration_hours_one_shot(self):
        result = parse_schedule("2h")
        assert result["kind"] == "once"
        assert "run_at" in result

    def test_parse_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="Invalid schedule"):
            parse_schedule("not a valid schedule")


class TestComputeNextRunIntegration:
    """Integration tests for compute_next_run — next run computation."""

    def test_once_schedule_returns_run_at(self):
        schedule = {"kind": "once", "run_at": "2099-01-01T12:00:00+00:00"}
        result = compute_next_run(schedule)
        assert result == "2099-01-01T12:00:00+00:00"

    def test_once_schedule_past_grace_returns_none(self):
        """One-shot jobs past their grace window return None (completed)."""
        schedule = {"kind": "once", "run_at": "2020-01-01T12:00:00+00:00"}
        result = compute_next_run(schedule)
        # Past the grace window (120s), one-shot jobs are considered done
        assert result is None

    def test_once_schedule_within_grace_returns_run_at(self):
        """One-shot jobs within grace window still return their run_at."""
        from cron.jobs import _hermes_now
        now = _hermes_now()
        # Schedule within the grace window (ONESHOT_GRACE_SECONDS = 120s)
        run_at = now + timedelta(seconds=30)
        schedule = {"kind": "once", "run_at": run_at.isoformat()}
        result = compute_next_run(schedule)
        assert result == run_at.isoformat()

    def test_interval_schedule_returns_future_time(self):
        schedule = {"kind": "interval", "minutes": 60}
        result = compute_next_run(schedule)
        assert result is not None
        # Should be roughly now + 60 minutes
        run_dt = datetime.fromisoformat(result)
        now = datetime.now(timezone.utc)
        diff = (run_dt - now).total_seconds()
        assert 3500 < diff < 3700  # ~60 minutes

    def test_interval_schedule_with_last_run(self):
        schedule = {"kind": "interval", "minutes": 60}
        last_run = datetime.now(timezone.utc).isoformat()
        result = compute_next_run(schedule, last_run_at=last_run)
        assert result is not None
        run_dt = datetime.fromisoformat(result)
        last_dt = datetime.fromisoformat(last_run)
        diff = (run_dt - last_dt).total_seconds()
        assert 3500 < diff < 3700

    def test_cron_schedule_returns_future_time(self):
        schedule = {"kind": "cron", "expr": "0 * * * *"}  # Hourly
        result = compute_next_run(schedule)
        assert result is not None
        run_dt = datetime.fromisoformat(result)
        now = datetime.now(timezone.utc)
        diff = (run_dt - now).total_seconds()
        assert 0 <= diff <= 3600  # Within the next hour
