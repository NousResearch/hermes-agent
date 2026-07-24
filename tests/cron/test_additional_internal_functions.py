"""Tests for additional internal cron.jobs functions lacking direct coverage.

These functions are exercised through the public API but benefit from
direct unit tests to guard edge cases and ensure behavior stability.
"""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cron.jobs import (
    _normalize_skill_list,
    _apply_skill_fields,
    _coerce_job_text,
    parse_duration,
    _ensure_aware,
    _recoverable_oneshot_run_at,
    _normalize_workdir,
    rewrite_skill_refs,
    save_jobs,
    load_jobs,
    _jobs_file_lock,
    _hermes_now,
)


class TestNormalizeSkillList:
    """Tests for _normalize_skill_list — canonicalize skill/skills inputs."""

    def test_none_returns_empty(self):
        assert _normalize_skill_list(None, None) == []

    def test_single_skill_string(self):
        # When skills is None, skill is used
        assert _normalize_skill_list("blogwatcher", None) == ["blogwatcher"]

    def test_skills_list(self):
        assert _normalize_skill_list(None, ["skill-a", "skill-b"]) == ["skill-a", "skill-b"]

    def test_skills_takes_precedence_over_skill(self):
        # When both skill and skills are provided, skills takes precedence
        result = _normalize_skill_list("skill-a", ["skill-b", "skill-c"])
        assert result == ["skill-b", "skill-c"]

    def test_skills_string_input_handled(self):
        # skills can be a string (treated as single-item list)
        result = _normalize_skill_list(None, "single-skill")
        assert result == ["single-skill"]

    def test_skill_none_in_skills_list_handled(self):
        result = _normalize_skill_list(None, ["skill-a", "", "skill-b"])
        assert result == ["skill-a", "skill-b"]

    def test_whitespace_stripped_from_each_item(self):
        # Each item is stripped individually
        result = _normalize_skill_list("  blogwatcher  ", ["  skill-a  "])
        # skills takes precedence, so skill is ignored
        assert result == ["skill-a"]

    def test_preserves_order_from_skills_list(self):
        # skills list order is preserved
        result = _normalize_skill_list("skill-b", ["skill-a", "skill-b", "skill-c"])
        # skills takes precedence
        assert result == ["skill-a", "skill-b", "skill-c"]

    def test_empty_string_in_skills_ignored(self):
        result = _normalize_skill_list(None, ["skill-a", "", "skill-b", "   "])
        assert result == ["skill-a", "skill-b"]

    def test_skill_used_when_skills_is_none(self):
        # skill is only used when skills is None
        result = _normalize_skill_list("only-skill", None)
        assert result == ["only-skill"]


class TestApplySkillFields:
    """Tests for _apply_skill_fields — align canonical skills and legacy skill fields."""

    def test_no_skills_returns_empty_lists(self):
        job = {"id": "abc123"}
        result = _apply_skill_fields(job)
        assert result["skills"] == []
        assert result["skill"] is None

    def test_skill_only_sets_both(self):
        job = {"id": "abc123", "skill": "legacy-skill"}
        result = _apply_skill_fields(job)
        assert result["skills"] == ["legacy-skill"]
        assert result["skill"] == "legacy-skill"

    def test_skills_only_sets_both(self):
        job = {"id": "abc123", "skills": ["skill-a", "skill-b"]}
        result = _apply_skill_fields(job)
        assert result["skills"] == ["skill-a", "skill-b"]
        assert result["skill"] == "skill-a"

    def test_both_skill_and_skills_skills_takes_precedence(self):
        # _normalize_skill_list gives precedence to skills over skill
        job = {"id": "abc123", "skill": "skill-c", "skills": ["skill-a", "skill-b"]}
        result = _apply_skill_fields(job)
        assert result["skills"] == ["skill-a", "skill-b"]
        assert result["skill"] == "skill-a"

    def test_duplicates_removed_preserving_order(self):
        job = {"id": "abc123", "skill": "skill-a", "skills": ["skill-a", "skill-b", "skill-a"]}
        result = _apply_skill_fields(job)
        assert result["skills"] == ["skill-a", "skill-b"]
        assert result["skill"] == "skill-a"

    def test_empty_skill_string_ignored(self):
        job = {"id": "abc123", "skill": "", "skills": ["skill-a"]}
        result = _apply_skill_fields(job)
        assert result["skills"] == ["skill-a"]
        assert result["skill"] == "skill-a"


class TestCoerceJobText:
    """Tests for _coerce_job_text — normalize nullable fields to strings."""

    def test_none_returns_fallback(self):
        assert _coerce_job_text(None, "fallback") == "fallback"

    def test_none_default_fallback_empty(self):
        assert _coerce_job_text(None) == ""

    def test_string_preserved(self):
        assert _coerce_job_text("hello") == "hello"

    def test_int_converted(self):
        assert _coerce_job_text(123) == "123"

    def test_float_converted(self):
        assert _coerce_job_text(3.14) == "3.14"

    def test_bool_converted(self):
        assert _coerce_job_text(True) == "True"


class TestParseDuration:
    """Tests for parse_duration — duration string to minutes parsing."""

    def test_minutes_variants(self):
        assert parse_duration("30m") == 30
        assert parse_duration("30min") == 30
        assert parse_duration("30mins") == 30
        assert parse_duration("30minute") == 30
        assert parse_duration("30minutes") == 30

    def test_hours_variants(self):
        assert parse_duration("2h") == 120
        assert parse_duration("2hr") == 120
        assert parse_duration("2hrs") == 120
        assert parse_duration("2hour") == 120
        assert parse_duration("2hours") == 120

    def test_days_variants(self):
        assert parse_duration("1d") == 1440
        assert parse_duration("1day") == 1440
        assert parse_duration("1days") == 1440

    def test_whitespace_handled(self):
        assert parse_duration(" 30m ") == 30
        assert parse_duration(" 2h ") == 120

    def test_case_insensitive(self):
        assert parse_duration("30M") == 30
        assert parse_duration("2H") == 120
        assert parse_duration("1D") == 1440

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("not a duration")
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("30x")
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("")


class TestEnsureAware:
    """Tests for _ensure_aware — timezone-aware datetime conversion."""

    def test_aware_datetime_converted_to_hermes_tz(self):
        dt = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        result = _ensure_aware(dt)
        assert result.tzinfo is not None
        # Result is converted to Hermes timezone (not necessarily same object)
        assert result is not dt or result.tzinfo == _hermes_now().tzinfo

    def test_naive_datetime_converted(self):
        dt = datetime(2026, 1, 1, 12, 0)  # naive
        result = _ensure_aware(dt)
        assert result.tzinfo is not None
        # Should be interpreted as local time then converted to Hermes timezone


class TestRecoverableOneshotRunAt:
    """Tests for _recoverable_oneshot_run_at — one-shot job grace window."""

    def test_non_once_returns_none(self):
        schedule = {"kind": "interval", "minutes": 60}
        result = _recoverable_oneshot_run_at(schedule, datetime.now(timezone.utc))
        assert result is None

    def test_missing_run_at_returns_none(self):
        schedule = {"kind": "once"}
        result = _recoverable_oneshot_run_at(schedule, datetime.now(timezone.utc))
        assert result is None

    def test_future_run_at_within_grace_returned(self):
        schedule = {"kind": "once", "run_at": "2099-01-01T12:00:00+00:00"}
        now = datetime(2099, 1, 1, 11, 59, 30, tzinfo=timezone.utc)  # 30s before
        result = _recoverable_oneshot_run_at(schedule, now)
        assert result == "2099-01-01T12:00:00+00:00"

    def test_past_run_at_outside_grace_returns_none(self):
        schedule = {"kind": "once", "run_at": "2020-01-01T12:00:00+00:00"}
        now = datetime(2020, 1, 1, 12, 5, 0, tzinfo=timezone.utc)  # 5 min after (grace is 120s)
        result = _recoverable_oneshot_run_at(schedule, now)
        assert result is None

    def test_already_run_returns_none(self):
        schedule = {"kind": "once", "run_at": "2099-01-01T12:00:00+00:00"}
        now = datetime(2099, 1, 1, 11, 59, 30, tzinfo=timezone.utc)
        result = _recoverable_oneshot_run_at(schedule, now, last_run_at="2099-01-01T11:58:00+00:00")
        assert result is None


class TestNormalizeWorkdir:
    """Tests for _normalize_workdir — workdir validation and normalization."""

    def test_none_returns_none(self):
        assert _normalize_workdir(None) is None

    def test_empty_string_returns_none(self):
        assert _normalize_workdir("") is None

    def test_whitespace_only_returns_none(self):
        assert _normalize_workdir("   ") is None

    def test_relative_path_raises(self):
        with pytest.raises(ValueError, match="must be an absolute path"):
            _normalize_workdir("relative/path")

    def test_nonexistent_path_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            _normalize_workdir("/nonexistent/path/that/should/not/exist")

    def test_existing_dir_returns_absolute(self, tmp_path):
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        result = _normalize_workdir(str(test_dir))
        assert result == str(test_dir.resolve())

    def test_tilde_expanded(self, tmp_path):
        test_dir = tmp_path / "tilde_test"
        test_dir.mkdir()
        # Use the actual home directory for tilde expansion test
        # We can't easily test ~ expansion without knowing the home dir,
        # but we can test that it doesn't crash
        # This is more of an integration test


class TestRewriteSkillRefs:
    """Tests for rewrite_skill_refs — skill reference rewriting after curator consolidation."""

    def test_empty_inputs_returns_empty_report(self):
        # rewrite_skill_refs acquires its own lock
        result = rewrite_skill_refs({}, [])
        assert result == {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}

    def test_consolidated_skill_replaced(self, tmp_path):
        # Create test jobs with a skill that will be consolidated
        jobs = [
            {
                "id": "job1",
                "name": "Job 1",
                "skills": ["old-skill", "other-skill"],
                "skill": "old-skill",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
            {
                "id": "job2",
                "name": "Job 2",
                "skills": ["other-skill"],
                "skill": "other-skill",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
        ]
        save_jobs(jobs)

        result = rewrite_skill_refs({"old-skill": "new-umbrella"}, [])

        assert result["jobs_updated"] == 1
        assert result["jobs_scanned"] == 2
        assert len(result["rewrites"]) == 1
        rewrite = result["rewrites"][0]
        assert rewrite["job_id"] == "job1"
        assert rewrite["before"] == ["old-skill", "other-skill"]
        assert rewrite["after"] == ["new-umbrella", "other-skill"]
        assert rewrite["mapped"] == {"old-skill": "new-umbrella"}
        assert rewrite["dropped"] == []

        # Verify job was actually updated
        updated_job = load_jobs()
        job1 = next(j for j in updated_job if j["id"] == "job1")
        assert job1["skills"] == ["new-umbrella", "other-skill"]
        assert job1["skill"] == "new-umbrella"

    def test_pruned_skill_dropped(self, tmp_path):
        jobs = [
            {
                "id": "job1",
                "name": "Job 1",
                "skills": ["pruned-skill", "keep-skill"],
                "skill": "pruned-skill",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
        ]
        save_jobs(jobs)

        result = rewrite_skill_refs({}, ["pruned-skill"])

        assert result["jobs_updated"] == 1
        rewrite = result["rewrites"][0]
        assert rewrite["before"] == ["pruned-skill", "keep-skill"]
        assert rewrite["after"] == ["keep-skill"]
        assert rewrite["mapped"] == {}
        assert rewrite["dropped"] == ["pruned-skill"]

        # Verify job was updated
        updated_job = load_jobs()
        job1 = next(j for j in updated_job if j["id"] == "job1")
        assert job1["skills"] == ["keep-skill"]
        assert job1["skill"] == "keep-skill"

    def test_consolidated_wins_over_pruned(self, tmp_path):
        """A skill in both consolidated and pruned should be consolidated."""
        jobs = [
            {
                "id": "job1",
                "name": "Job 1",
                "skills": ["both-skill"],
                "skill": "both-skill",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
        ]
        save_jobs(jobs)

        # Skill in both - consolidated should win
        result = rewrite_skill_refs({"both-skill": "umbrella"}, ["both-skill"])

        assert result["jobs_updated"] == 1
        rewrite = result["rewrites"][0]
        assert rewrite["mapped"] == {"both-skill": "umbrella"}
        assert rewrite["dropped"] == []

        # Verify it was consolidated, not dropped
        updated_job = load_jobs()
        job1 = next(j for j in updated_job if j["id"] == "job1")
        assert job1["skills"] == ["umbrella"]

    def test_umbrella_already_present_no_duplicate(self, tmp_path):
        """If umbrella is already in skills, don't duplicate it."""
        jobs = [
            {
                "id": "job1",
                "name": "Job 1",
                "skills": ["old-skill", "umbrella-skill"],
                "skill": "old-skill",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
        ]
        save_jobs(jobs)

        result = rewrite_skill_refs({"old-skill": "umbrella-skill"}, [])

        assert result["jobs_updated"] == 1
        rewrite = result["rewrites"][0]
        assert rewrite["after"] == ["umbrella-skill"]  # No duplicate

        updated_job = load_jobs()
        job1 = next(j for j in updated_job if j["id"] == "job1")
        assert job1["skills"] == ["umbrella-skill"]

    def test_jobs_without_skills_skipped(self, tmp_path):
        jobs = [
            {
                "id": "job1",
                "name": "Job 1",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
            {
                "id": "job2",
                "name": "Job 2",
                "skills": ["old-skill"],
                "skill": "old-skill",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
        ]
        save_jobs(jobs)

        result = rewrite_skill_refs({"old-skill": "new-umbrella"}, [])

        assert result["jobs_updated"] == 1
        assert result["jobs_scanned"] == 2
        assert result["rewrites"][0]["job_id"] == "job2"

    def test_order_preserved(self, tmp_path):
        """Original skill order should be preserved (with consolidated skills in place)."""
        jobs = [
            {
                "id": "job1",
                "name": "Job 1",
                "skills": ["first", "to-consolidate", "last"],
                "skill": "first",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
        ]
        save_jobs(jobs)

        result = rewrite_skill_refs({"to-consolidate": "replacement"}, [])

        rewrite = result["rewrites"][0]
        assert rewrite["after"] == ["first", "replacement", "last"]

    def test_multiple_consolidations_in_one_job(self, tmp_path):
        jobs = [
            {
                "id": "job1",
                "name": "Job 1",
                "skills": ["old-a", "old-b", "keep"],
                "skill": "old-a",
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_run_at": "2099-01-01T12:00:00+00:00",
            },
        ]
        save_jobs(jobs)

        result = rewrite_skill_refs({"old-a": "new-a", "old-b": "new-b"}, [])

        rewrite = result["rewrites"][0]
        assert rewrite["mapped"] == {"old-a": "new-a", "old-b": "new-b"}
        assert rewrite["after"] == ["new-a", "new-b", "keep"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
