"""Scheduler registration and the cronjob tool surface for paused creation.

create_job_with_scheduler_registration must NOT register a job born paused
(there is no first trigger to arm), and the model-facing cronjob() tool must
forward paused/paused_reason with the same contract — including rejecting
invalid combinations before anything is persisted.
"""

import json

import pytest

from cron.jobs import create_job, load_jobs, resume_job


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate the cron store (same pattern as tests/cron/test_jobs.py)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path / "cron"


@pytest.fixture()
def registering_provider(make_cron_provider, monkeypatch):
    """Resolve to a provider whose register_job records every call."""
    calls = []

    def _register(job):
        calls.append(job["id"])

    provider = make_cron_provider(register_job=_register)
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler", lambda: provider
    )
    return calls


class TestPausedCreationRegistration:
    def test_paused_creation_is_not_registered(
        self, tmp_cron_dir, registering_provider
    ):
        from cron.scheduler import create_job_with_scheduler_registration

        job = create_job_with_scheduler_registration(
            prompt="canary", schedule="every 1h", paused=True
        )
        assert job["enabled"] is False
        assert registering_provider == []

    def test_ordinary_creation_still_registers(
        self, tmp_cron_dir, registering_provider
    ):
        from cron.scheduler import create_job_with_scheduler_registration

        job = create_job_with_scheduler_registration(prompt="hi", schedule="every 1h")
        assert registering_provider == [job["id"]]

    def test_failed_registration_of_paused_job_cannot_lose_the_job(
        self, tmp_cron_dir, make_cron_provider, monkeypatch
    ):
        """A disabled job must not surface as a partial-failure contract either."""

        def _boom(job):
            raise RuntimeError("provider down")

        monkeypatch.setattr(
            "cron.scheduler_provider.resolve_cron_scheduler",
            lambda: make_cron_provider(register_job=_boom, name="boom"),
        )
        from cron.scheduler import create_job_with_scheduler_registration

        job = create_job_with_scheduler_registration(
            prompt="canary", schedule="every 1h", paused=True
        )
        assert job["enabled"] is False


class TestPausedCreationTool:
    def test_tool_creates_paused_job_atomically(
        self, tmp_cron_dir, registering_provider
    ):
        from tools.cronjob_tools import cronjob

        raw = cronjob(
            action="create",
            schedule="every 1h",
            prompt="canary",
            paused=True,
            paused_reason="canary — awaiting operator approval",
        )
        result = json.loads(raw)
        assert result["success"] is True
        job = result["job"]
        assert job["enabled"] is False
        assert job["state"] == "paused"
        assert job["paused_at"] is not None
        assert job["paused_reason"] == "canary — awaiting operator approval"
        assert job["next_run_at"] is None
        assert registering_provider == []
        assert "PAUSED" in result["message"]

    def test_tool_ordinary_create_unchanged(self, tmp_cron_dir, registering_provider):
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(action="create", schedule="every 1h", prompt="hi"))
        assert result["success"] is True
        assert result["job"]["enabled"] is True
        assert result["job"]["state"] == "scheduled"
        assert result["job"]["next_run_at"] is not None
        assert registering_provider == [result["job_id"]]

    def test_tool_rejects_reason_without_paused(
        self, tmp_cron_dir, registering_provider
    ):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create", schedule="every 1h", prompt="x", paused_reason="orphan"
            )
        )
        assert result["success"] is False
        assert "paused_reason requires paused=true" in result["error"]
        assert load_jobs() == []
        assert registering_provider == []

    def test_tool_rejects_non_boolean_paused(self, tmp_cron_dir, registering_provider):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(action="create", schedule="every 1h", prompt="x", paused="yes")
        )
        assert result["success"] is False
        assert "paused must be a boolean" in result["error"]
        assert load_jobs() == []
        assert registering_provider == []

    def test_tool_resume_after_paused_creation(
        self, tmp_cron_dir, registering_provider
    ):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(action="create", schedule="every 1h", prompt="canary", paused=True)
        )
        job_id = created["job_id"]
        resumed = json.loads(cronjob(action="resume", job_id=job_id))
        assert resumed["success"] is True
        assert resumed["job"]["enabled"] is True
        assert resumed["job"]["state"] == "scheduled"
        assert resumed["job"]["next_run_at"] is not None
