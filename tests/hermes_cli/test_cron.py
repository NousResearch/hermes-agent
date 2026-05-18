"""Tests for hermes_cli.cron command handling."""

import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli.cron import cron_command


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
                clear_skills=False,
            )
        )
        updated = get_job(job["id"])
        assert updated["skills"] == ["maps", "blogwatcher"]
        assert updated["name"] == "Edited Job"
        assert updated["prompt"] == "Revised prompt"
        assert updated["schedule_display"] == "every 120m"

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
                clear_skills=True,
            )
        )
        cleared = get_job(job["id"])
        assert cleared["skills"] == []
        assert cleared["skill"] is None

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
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out

        jobs = list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["skills"] == ["blogwatcher", "maps"]
        assert jobs[0]["name"] == "Skill combo"


class TestCronPreview:
    @staticmethod
    def _run_cron_cli(*argv):
        return subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "cron", *argv],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=PROJECT_ROOT,
        )

    def test_schedule_preview_prints_future_occurrences(self, tmp_cron_dir, monkeypatch, capsys):
        now = datetime(2026, 4, 1, 9, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        result = cron_command(Namespace(cron_command="preview", schedule="every 1h", job_id=None, next=5))

        assert result == 0
        out = capsys.readouterr().out
        assert "Preview for schedule: every 1h" in out
        for index in range(1, 6):
            assert f"{index}. 2026-04-01T" in out

    def test_schedule_preview_reports_when_no_future_runs_exist(self, tmp_cron_dir, monkeypatch, capsys):
        now = datetime(2026, 4, 1, 9, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        result = cron_command(
            Namespace(
                cron_command="preview",
                schedule="2026-04-01T08:00:00+00:00",
                job_id=None,
                next=5,
            )
        )

        assert result == 0
        assert capsys.readouterr().out.strip() == (
            "Preview for schedule: 2026-04-01T08:00:00+00:00\n"
            "No future runs found for this schedule."
        )

    def test_job_preview_reports_paused_state(self, tmp_cron_dir, capsys):
        job = create_job(prompt="Check server status", schedule="every 1h")
        cron_command(Namespace(cron_command="pause", job_id=job["id"]))
        capsys.readouterr()

        result = cron_command(Namespace(cron_command="preview", schedule=None, job_id=job["id"], next=5))

        assert result == 0
        assert capsys.readouterr().out.strip() == (
            f"Preview for job: {job['id']}\nJob is paused. Resume it to preview future runs."
        )

    @pytest.mark.parametrize(
        "schedule,job_id",
        [(None, None), ("every 1h", "job-123")],
    )
    def test_preview_requires_exactly_one_selector(self, tmp_cron_dir, capsys, schedule, job_id):
        result = cron_command(Namespace(cron_command="preview", schedule=schedule, job_id=job_id, next=5))

        assert result == 1
        assert capsys.readouterr().out.strip() == "Choose exactly one of --schedule or --job-id."

    @pytest.mark.parametrize("count", [0, 21])
    def test_preview_rejects_out_of_range_next(self, tmp_cron_dir, capsys, count):
        result = cron_command(Namespace(cron_command="preview", schedule="every 1h", job_id=None, next=count))

        assert result == 1
        assert capsys.readouterr().out.strip() == "--next must be between 1 and 20."

    def test_preview_cli_exits_nonzero_for_out_of_range_next(self):
        result = self._run_cron_cli("preview", "--schedule", "every 1h", "--next", "0")

        assert result.returncode != 0
        assert "--next must be between 1 and 20." in result.stdout

    def test_preview_cli_exits_nonzero_for_conflicting_selectors(self):
        result = self._run_cron_cli(
            "preview",
            "--schedule",
            "every 1h",
            "--job-id",
            "job-123",
        )

        assert result.returncode != 0
        assert "Choose exactly one of --schedule or --job-id." in result.stdout

    def test_preview_cli_succeeds_for_valid_schedule(self):
        result = self._run_cron_cli("preview", "--schedule", "every 1h", "--next", "1")

        assert result.returncode == 0, result.stderr
        assert "Preview for schedule: every 1h" in result.stdout
        assert "1. " in result.stdout

    def test_cron_help_includes_preview(self):
        result = self._run_cron_cli("--help")

        assert result.returncode == 0, result.stderr
        assert "preview" in result.stdout
