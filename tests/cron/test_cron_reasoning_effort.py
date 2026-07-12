"""Cron reasoning-effort overrides across tool-facing surfaces."""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCronjobToolReasoningEffort:
    def test_create_update_clear_and_list_reasoning_effort(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
                reasoning_effort="ultra",
            )
        )

        assert created["success"] is True
        assert created["job"]["reasoning_effort"] == "ultra"
        assert created["job"]["reasoning_effort_status"] == "override"

        job_id = created["job_id"]
        updated = json.loads(
            cronjob(action="update", job_id=job_id, reasoning_effort="none")
        )
        assert updated["success"] is True
        assert updated["job"]["reasoning_effort"] == "none"

        listed = json.loads(cronjob(action="list"))
        assert listed["jobs"][0]["reasoning_effort"] == "none"
        assert listed["jobs"][0]["reasoning_effort_status"] == "override"

        cleared = json.loads(
            cronjob(action="update", job_id=job_id, reasoning_effort="")
        )
        assert cleared["success"] is True
        assert cleared["job"]["reasoning_effort"] is None
        assert cleared["job"]["reasoning_effort_status"] == "inherit"

    def test_create_preserves_no_agent_reasoning_effort_as_inactive(
        self, tmp_cron_dir
    ):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(
                action="create",
                schedule="every 1h",
                script="watch.py",
                no_agent=True,
                reasoning_effort="max",
            )
        )

        assert created["success"] is True
        assert created["job"]["reasoning_effort"] == "max"
        assert created["job"]["reasoning_effort_status"] == "not_applicable"

    def test_invalid_reasoning_effort_returns_tool_error(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
                reasoning_effort="turbo",
            )
        )

        assert result["success"] is False
        assert "Invalid cron reasoning_effort" in result["error"]

    def test_registry_handler_can_clear_reasoning_effort_with_null(
        self, tmp_cron_dir
    ):
        from tools.registry import registry
        import tools.cronjob_tools  # noqa: F401  ensure registration

        entry = registry.get_entry("cronjob")
        assert entry is not None

        created = json.loads(
            entry.handler(
                {
                    "action": "create",
                    "prompt": "hi",
                    "schedule": "every 1h",
                    "reasoning_effort": "high",
                }
            )
        )
        job_id = created["job_id"]

        cleared = json.loads(
            entry.handler(
                {
                    "action": "update",
                    "job_id": job_id,
                    "reasoning_effort": None,
                }
            )
        )

        assert cleared["success"] is True
        assert cleared["job"]["reasoning_effort"] is None
