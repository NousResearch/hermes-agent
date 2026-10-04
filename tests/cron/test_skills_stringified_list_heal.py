"""Regression: skills passed as a stringified list (LLM tool callers) must heal.

Observed live: a job created via the cronjob tool stored ``skills: ["['x']"]`` — the
model passed ``"['x']"`` (a python repr) instead of ``['x']``. The run then treated
the whole repr as one bogus skill name, the skill never loaded, and the job quietly
did nothing. Every skills-normalization seam must unwrap the shape; doctor must also
flag the stored corruption (the read path heals, hiding it from naive readers).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.jobs import _normalize_skill_list, _apply_skill_fields
from cron.scheduler_prompt import _job_skill_names


class TestNormalizeSkillListHealsStringifiedShapes:
    def test_repr_string_in_skills(self):
        assert _normalize_skill_list(None, "['x']") == ["x"]

    def test_repr_string_inside_list(self):
        assert _normalize_skill_list(None, ['["x", "y"]']) == ["x", "y"]

    def test_legacy_skill_field_repr(self):
        assert _normalize_skill_list("['x', 'y']", None) == ["x", "y"]

    def test_plain_string_not_a_literal_stays(self):
        # A skill name that merely looks like a fragment of a literal must never mangle.
        assert _normalize_skill_list(None, "not-a-list") == ["not-a-list"]

    def test_non_string_literal_entries_kept_verbatim(self):
        assert _normalize_skill_list(None, "[1, 2]") == ["[1, 2]"]

    def test_normal_list_dedupe_and_blanks(self):
        assert _normalize_skill_list(None, ["a", " b", "a", ""]) == ["a", "b"]

    def test_nested_list_flattened(self):
        assert _normalize_skill_list(None, [["x", "y"], "z"]) == ["x", "y", "z"]

    def test_dict_literal_values_unwrapped(self):
        assert _normalize_skill_list(None, "{'skills': ['x']}") == ["{'skills': ['x']}"]

    def test_apply_skill_fields_heals_stored_record(self):
        healed = _apply_skill_fields({"id": "j1", "skills": ["['x']"], "skill": None})
        assert healed["skills"] == ["x"]
        assert healed["skill"] == "x"


class TestSchedulerJobSkillNamesHeals:
    def test_repr_string_in_skills(self):
        assert _job_skill_names({"skills": ["['x']"]}) == ["x"]

    def test_repr_string_legacy_skill(self):
        assert _job_skill_names({"skill": "['x', 'y']"}) == ["x", "y"]

    def test_normal_job_unchanged(self):
        assert _job_skill_names({"skills": ["a", "b"]}) == ["a", "b"]
        assert _job_skill_names({}) == []


class TestStoredRecordHealsOnRead:
    """End-to-end: a malformed record in jobs.json reads back healed (create_job →
    hand-corrupt → list_jobs), without rewriting storage on read."""

    def test_list_jobs_heals_corrupted_skills(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
        from cron.jobs import create_job, list_jobs, load_jobs, save_jobs

        create_job(prompt="Watch the news", schedule="every 1h")
        jobs = load_jobs()
        jobs[0]["skills"] = "['newsy']"
        jobs[0]["skill"] = "['newsy']"
        save_jobs(jobs)

        listed = list_jobs()
        assert listed[0]["skills"] == ["newsy"]
        assert listed[0]["skill"] == "newsy"
        # Storage is NOT rewritten by the read path (healing is read-time only).
        assert load_jobs()[0]["skills"] == "['newsy']"