"""Cron skill references are fail-closed at create, update, preflight and load."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import cron.jobs as cron_jobs
from cron.scheduler import _build_job_prompt, _preflight_check_skills
from cron.skill_refs import CronSkillReferenceError


def _payload(*, success=True, content="# skill\nbody", error=None):
    return json.dumps(
        {
            "success": success,
            "content": content,
            "error": error,
            "readiness_status": "available",
            "setup_needed": False,
        }
    )


def test_create_rejects_unresolved_skill_before_persist(tmp_path):
    with cron_jobs.use_cron_store(tmp_path), patch(
        "tools.skills_tool.skill_view",
        return_value=_payload(success=False, error="Skill 'missing' not found"),
    ):
        with pytest.raises(CronSkillReferenceError, match="missing"):
            cron_jobs.create_job(
                prompt="run",
                schedule="every 1h",
                skills=["missing"],
            )
        assert cron_jobs.load_jobs() == []


def test_create_accepts_resolved_skill(tmp_path):
    with cron_jobs.use_cron_store(tmp_path), patch(
        "tools.skills_tool.skill_view",
        return_value=_payload(),
    ):
        job = cron_jobs.create_job(
            prompt="run",
            schedule="every 1h",
            skills=["ready"],
        )
    assert job["skills"] == ["ready"]


def test_update_rejects_new_unresolved_skill_and_preserves_job(tmp_path):
    with cron_jobs.use_cron_store(tmp_path):
        job = cron_jobs.create_job(prompt="run", schedule="every 1h")
        with patch(
            "tools.skills_tool.skill_view",
            return_value=_payload(success=False, error="not installed"),
        ):
            with pytest.raises(CronSkillReferenceError, match="missing"):
                cron_jobs.update_job(job["id"], {"skills": ["missing"]})
        stored = cron_jobs.get_job(job["id"])
    assert stored is not None
    assert stored.get("skills") == []


def test_fire_preflight_blocks_legacy_missing_skill():
    job = {"id": "legacy", "skills": ["missing"]}
    with patch(
        "tools.skills_tool.skill_view",
        return_value=_payload(success=False, error="not installed"),
    ):
        reason = _preflight_check_skills(job)
    assert reason is not None
    assert "unresolved or unadmitted" in reason
    assert "missing" in reason


def test_fire_preflight_fails_closed_when_resolution_raises():
    job = {"id": "legacy", "skills": ["broken"]}
    with patch(
        "cron.skill_refs.resolve_skill_references",
        side_effect=RuntimeError("boom"),
    ), patch(
        "cron.scheduler._preflight_check_provider_key",
        return_value=None,
    ), patch(
        "cron.scheduler._preflight_check_delivery",
        return_value=None,
    ):
        from cron.scheduler import _preflight_job_config

        reason = _preflight_job_config(job, {})
    assert reason is not None
    assert "failing closed" in reason


def test_prompt_builder_refuses_missing_skill_even_if_preflight_is_bypassed():
    job = {
        "id": "legacy",
        "name": "legacy",
        "prompt": "run",
        "skills": ["missing"],
    }
    with patch(
        "tools.skills_tool.skill_view",
        return_value=_payload(success=False, error="not installed"),
    ):
        with pytest.raises(CronSkillReferenceError, match="missing"):
            _build_job_prompt(job)


def test_prompt_builder_loads_resolved_skill_content():
    job = {
        "id": "ready",
        "name": "ready",
        "prompt": "run",
        "skills": ["ready"],
    }
    with patch("tools.skills_tool.skill_view", return_value=_payload()):
        prompt = _build_job_prompt(job)
    assert "# skill" in prompt
    assert "run" in prompt
