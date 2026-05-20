"""Tests for hermes_cli.cron command handling."""

from argparse import Namespace

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli.cron import cron_command


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCronCommandLifecycle:
    def test_pause_resume_run(self, tmp_cron_dir, capsys):
        job = create_job(prompt="Check server status", schedule="every 1h")

        cron_command(Namespace(cron_command="pause", job_id=job["id"]))
        paused = get_job(job["id"])
        assert paused["state"] == "paused"

        cron_command(Namespace(cron_command="resume", job_id=job["id"]))
        resumed = get_job(job["id"])
        assert resumed["state"] == "scheduled"

        cron_command(Namespace(cron_command="run", job_id=job["id"]))
        triggered = get_job(job["id"])
        assert triggered["state"] == "scheduled"

        out = capsys.readouterr().out
        assert "Paused job" in out
        assert "Resumed job" in out
        assert "Triggered job" in out

    def test_edit_can_replace_and_clear_skills(self, tmp_cron_dir, capsys):
        job = create_job(
            prompt="Combine skill outputs",
            schedule="every 1h",
            skill="blogwatcher",
        )

        cron_command(
            Namespace(
                cron_command="edit",
                job_id=job["id"],
                schedule="every 2h",
                prompt="Revised prompt",
                name="Edited Job",
                deliver=None,
                repeat=None,
                skill=None,
                skills=["maps", "blogwatcher"],
                profile="default",
                clear_skills=False,
            )
        )
        updated = get_job(job["id"])
        assert updated["skills"] == ["maps", "blogwatcher"]
        assert updated["name"] == "Edited Job"
        assert updated["prompt"] == "Revised prompt"
        assert updated["schedule_display"] == "every 120m"
        assert updated["profile"] == "default"

        cron_command(
            Namespace(
                cron_command="edit",
                job_id=job["id"],
                schedule=None,
                prompt=None,
                name=None,
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                profile="",
                clear_skills=True,
            )
        )
        cleared = get_job(job["id"])
        assert cleared["skills"] == []
        assert cleared["skill"] is None
        assert cleared["profile"] is None

        out = capsys.readouterr().out
        assert "Updated job" in out

    def test_create_with_multiple_skills(self, tmp_cron_dir, capsys):
        cron_command(
            Namespace(
                cron_command="create",
                schedule="every 1h",
                prompt="Use both skills",
                name="Skill combo",
                deliver=None,
                repeat=None,
                skill=None,
                skills=["blogwatcher", "maps"],
                profile="default",
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out

        jobs = list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["skills"] == ["blogwatcher", "maps"]
        assert jobs[0]["name"] == "Skill combo"
        assert jobs[0]["profile"] == "default"


class TestCronListMCPCompatibility:
    """Regression tests for issue #28662: cron list crash on MCP-created jobs."""

    def test_list_handles_string_schedule(self, monkeypatch, capsys):
        # MCP cronjob tool stores schedule as a plain string.
        jobs = [{
            "id": "abc123",
            "name": "mcp-string-schedule",
            "schedule": "0 7 * * 1",
            "repeat": {"completed": 0, "times": None},
            "enabled": True,
        }]
        monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: jobs)
        from hermes_cli.cron import cron_list
        cron_list()
        out = capsys.readouterr().out
        assert "mcp-string-schedule" in out
        assert "0 7 * * 1" in out

    def test_list_handles_none_repeat(self, monkeypatch, capsys):
        # MCP cronjob tool may store repeat as bare None.
        jobs = [{
            "id": "abc124",
            "name": "mcp-none-repeat",
            "schedule": "0 8 * * *",
            "repeat": None,
            "enabled": True,
        }]
        monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: jobs)
        from hermes_cli.cron import cron_list
        cron_list()
        out = capsys.readouterr().out
        assert "mcp-none-repeat" in out
        assert "∞" in out

    def test_list_dict_schedule_prefers_display(self, monkeypatch, capsys):
        jobs = [{
            "id": "abc125",
            "name": "cli-wizard-job",
            "schedule": {"display": "every hour", "expr": "0 * * * *"},
            "repeat": {"completed": 2, "times": 10},
            "enabled": True,
        }]
        monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: jobs)
        from hermes_cli.cron import cron_list
        cron_list()
        out = capsys.readouterr().out
        assert "every hour" in out
        assert "2/10" in out

    def test_list_does_not_crash_on_mixed_jobs(self, monkeypatch, capsys):
        # Regression: prior code crashed mid-loop, hiding later jobs.
        jobs = [
            {"id": "j1", "name": "first", "schedule": "* * * * *", "repeat": None, "enabled": True},
            {"id": "j2", "name": "second", "schedule": {"display": "ok"}, "repeat": {"times": None}, "enabled": True},
        ]
        monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: jobs)
        from hermes_cli.cron import cron_list
        cron_list()
        out = capsys.readouterr().out
        assert "first" in out
        assert "second" in out
