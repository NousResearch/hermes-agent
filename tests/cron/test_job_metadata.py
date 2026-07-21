"""Tests for cron job organizational metadata (#68482).

Covers the model layer (create_job/update_job persistence and clearing) and
the cronjob tool layer (create/update passthrough, list visibility, and
category/tags filtering). Hermetic: cron storage is redirected to tmp_path.
"""

import json

import pytest

from cron.jobs import (
    create_job,
    get_job,
    load_jobs,
    mark_job_run,
    pause_job,
    resume_job,
    update_job,
)
from tools.cronjob_tools import cronjob


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCreateJobMetadata:
    def test_metadata_stored_when_set(self, tmp_cron_dir):
        job = create_job(
            prompt="daily report",
            schedule="every 1d",
            tags=["daily", "report"],
            category="reporting",
            description="Generates the daily report",
            workflow="collect data -> analyze -> generate report",
        )
        stored = get_job(job["id"])
        assert stored["tags"] == ["daily", "report"]
        assert stored["category"] == "reporting"
        assert stored["description"] == "Generates the daily report"
        assert stored["workflow"] == "collect data -> analyze -> generate report"

    def test_unset_metadata_keys_absent_from_storage(self, tmp_cron_dir):
        create_job(prompt="plain job", schedule="every 1h")
        raw = load_jobs()[0]
        for field in ("tags", "category", "description", "workflow"):
            assert field not in raw

    def test_tags_normalized_and_deduped_case_insensitively(self, tmp_cron_dir):
        job = create_job(
            prompt="j",
            schedule="every 1h",
            tags=["  Daily ", "daily", "", "report", "DAILY"],
        )
        assert get_job(job["id"])["tags"] == ["Daily", "report"]

    def test_blank_metadata_treated_as_unset(self, tmp_cron_dir):
        job = create_job(
            prompt="j",
            schedule="every 1h",
            tags=["", "  "],
            category="   ",
            description="",
        )
        raw = next(j for j in load_jobs() if j["id"] == job["id"])
        for field in ("tags", "category", "description", "workflow"):
            assert field not in raw


class TestUpdateJobMetadata:
    def test_update_sets_and_normalizes_metadata(self, tmp_cron_dir):
        job = create_job(prompt="j", schedule="every 1h")
        updated = update_job(
            job["id"],
            {"tags": [" daily "], "category": " system ", "description": "x"},
        )
        assert updated["tags"] == ["daily"]
        assert updated["category"] == "system"
        assert updated["description"] == "x"

    def test_empty_update_values_clear_metadata_from_storage(self, tmp_cron_dir):
        job = create_job(
            prompt="j",
            schedule="every 1h",
            tags=["daily"],
            category="reporting",
            workflow="a -> b",
        )
        update_job(job["id"], {"tags": [], "category": "", "workflow": None})
        raw = next(j for j in load_jobs() if j["id"] == job["id"])
        for field in ("tags", "category", "workflow"):
            assert field not in raw

    def test_metadata_survives_scheduler_operations(self, tmp_cron_dir):
        """mark_job_run / pause / resume must not clobber metadata — the
        issue's reported failure mode for hand-edited custom fields."""
        job = create_job(
            prompt="j",
            schedule="every 1h",
            tags=["daily"],
            category="reporting",
            description="d",
        )
        mark_job_run(job["id"], True)
        pause_job(job["id"], reason="test")
        resume_job(job["id"])
        stored = get_job(job["id"])
        assert stored["tags"] == ["daily"]
        assert stored["category"] == "reporting"
        assert stored["description"] == "d"


class TestCronjobToolMetadata:
    def test_create_and_list_show_metadata(self, tmp_cron_dir):
        created = json.loads(
            cronjob(
                action="create",
                prompt="daily report",
                schedule="every 1d",
                tags=["daily", "report"],
                category="reporting",
                description="Generates the daily report",
                workflow="collect -> analyze -> report",
            )
        )
        assert created["success"] is True
        assert created["job"]["tags"] == ["daily", "report"]
        assert created["job"]["category"] == "reporting"

        listed = json.loads(cronjob(action="list"))
        assert listed["jobs"][0]["description"] == "Generates the daily report"
        assert listed["jobs"][0]["workflow"] == "collect -> analyze -> report"
        assert "filters" not in listed

    def test_jobs_without_metadata_keep_old_output_shape(self, tmp_cron_dir):
        cronjob(action="create", prompt="plain", schedule="every 1h")
        listed = json.loads(cronjob(action="list"))
        for field in ("tags", "category", "description", "workflow"):
            assert field not in listed["jobs"][0]

    def test_list_filters_by_category_case_insensitively(self, tmp_cron_dir):
        cronjob(action="create", prompt="a", schedule="every 1h", category="Reporting")
        cronjob(action="create", prompt="b", schedule="every 1h", category="system")
        cronjob(action="create", prompt="c", schedule="every 1h")

        listed = json.loads(cronjob(action="list", category="reporting"))
        assert listed["count"] == 1
        assert listed["jobs"][0]["prompt_preview"] == "a"
        assert listed["filters"] == {"category": "reporting"}

    def test_list_filters_by_tags_subset_match(self, tmp_cron_dir):
        cronjob(
            action="create", prompt="a", schedule="every 1h", tags=["daily", "report"]
        )
        cronjob(action="create", prompt="b", schedule="every 1h", tags=["daily"])
        cronjob(action="create", prompt="c", schedule="every 1h")

        one_tag = json.loads(cronjob(action="list", tags=["Daily"]))
        assert one_tag["count"] == 2

        both_tags = json.loads(cronjob(action="list", tags=["daily", "report"]))
        assert both_tags["count"] == 1
        assert both_tags["jobs"][0]["prompt_preview"] == "a"

    def test_update_via_tool_sets_and_clears_metadata(self, tmp_cron_dir):
        created = json.loads(
            cronjob(action="create", prompt="j", schedule="every 1h", tags=["old"])
        )
        job_id = created["job_id"]

        updated = json.loads(
            cronjob(
                action="update",
                job_id=job_id,
                tags=["new"],
                category="system",
            )
        )
        assert updated["job"]["tags"] == ["new"]
        assert updated["job"]["category"] == "system"

        cleared = json.loads(
            cronjob(action="update", job_id=job_id, tags=[], category="")
        )
        assert "tags" not in cleared["job"]
        assert "category" not in cleared["job"]

    def test_metadata_only_update_is_accepted(self, tmp_cron_dir):
        created = json.loads(
            cronjob(action="create", prompt="j", schedule="every 1h")
        )
        result = json.loads(
            cronjob(
                action="update",
                job_id=created["job_id"],
                description="added later",
            )
        )
        assert result["success"] is True
        assert result["job"]["description"] == "added later"
