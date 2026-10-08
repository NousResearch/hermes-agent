"""Tests for the cronjob skills repr fix (issue #132867).

A model caller can pass ``skills="['x', 'y']"`` — a *string repr* of a list instead of a
list. Before the fix every normalization step str()-ed that string into ONE bogus skill
literally named ``['x', 'y']``: the job stored it, the fire-time loader silently skipped
it (jobs ran without their skills), and ``cron doctor`` reported the record clean.

Three layers are covered:
- tools/cronjob_job_args.py::_canonical_skills — the create/update write path;
- cron/jobs.py::_normalize_skill_list — the storage read path (heals pre-fix records);
- hermes_cli/cron.py::_cron_doctor_issues_for_job — doctor now flags repr-shaped fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# --- Layer 1: tool-side create/update normalization --------------------------

def test_canonical_skills_unwraps_repr_string():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skills="['x', 'y']") == ["x", "y"]
    assert _canonical_skills(skills="['x']") == ["x"]


def test_canonical_skills_unwraps_repr_single_skill_param():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skill="['a','b']") == ["a", "b"]


def test_canonical_skills_unwraps_tuple_repr():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skills="('t1', 't2')") == ["t1", "t2"]


def test_canonical_skills_leaves_plain_names_alone():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skills=["x", "y"]) == ["x", "y"]
    assert _canonical_skills(skills="plain-skill") == ["plain-skill"]
    assert _canonical_skills(skill="my skill (v2)") == ["my skill (v2)"]
    assert _canonical_skills(skills=None, skill="one") == ["one"]


def test_canonical_skills_leaves_non_repr_brackets_alone():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skills="not [ a list") == ["not [ a list"]
    assert _canonical_skills(skills="[unclosed") == ["[unclosed"]
    assert _canonical_skills(skills="[]") == []


# --- Layer 2: storage read path heals pre-fix records -------------------------

def test_normalize_skill_list_heals_repr_stored():
    from cron.jobs import _normalize_skill_list
    assert _normalize_skill_list(skills="['x', 'y']") == ["x", "y"]
    assert _normalize_skill_list(skill="['a','b']") == ["a", "b"]
    assert _normalize_skill_list(skills=("t1", "t2")) == ["t1", "t2"]


def test_normalize_skill_list_plain_paths_unchanged():
    from cron.jobs import _normalize_skill_list
    assert _normalize_skill_list(skills=["x", "x", "y"]) == ["x", "y"]
    assert _normalize_skill_list(skill="one") == ["one"]
    assert _normalize_skill_list() == []
    assert _normalize_skill_list(skills="plain-skill") == ["plain-skill"]
    assert _normalize_skill_list(skills="[unclosed") == ["[unclosed"]


def test_apply_skill_fields_heals_corrupted_record():
    from cron.jobs import _apply_skill_fields
    healed = _apply_skill_fields({"skills": "['news-digest', 'summarize']", "id": "j1"})
    assert healed["skills"] == ["news-digest", "summarize"]
    assert healed["skill"] == "news-digest"


# --- Layer 3: cron doctor detection ------------------------------------------

def _job(**over):
    base = {"id": "job-1", "enabled": True, "state": "active",
            "next_run_at": "2099-01-01T00:00:00", "skills": ["ok-skill"]}
    base.update(over)
    return base


def test_doctor_flags_repr_shaped_skills():
    from hermes_cli.cron import _cron_doctor_issues_for_job
    issues = _cron_doctor_issues_for_job(_job(skills="['news-digest', 'summarize']"))
    assert any("string repr" in i for i in issues)


def test_doctor_flags_repr_shaped_legacy_skill_field():
    from hermes_cli.cron import _cron_doctor_issues_for_job
    issues = _cron_doctor_issues_for_job(_job(skills=[], skill="['a', 'b']"))
    assert any("string repr" in i for i in issues)


def test_doctor_quiet_on_clean_job():
    from hermes_cli.cron import _cron_doctor_issues_for_job
    assert not any("string repr" in i
                   for i in _cron_doctor_issues_for_job(_job()))


def test_doctor_quiet_on_bracket_named_skill():
    """A real skill whose name merely contains brackets must not trip the check."""
    from hermes_cli.cron import _cron_doctor_issues_for_job
    issues = _cron_doctor_issues_for_job(_job(skills=["[weird-name]"]))
    assert not any("string repr" in i for i in issues)
