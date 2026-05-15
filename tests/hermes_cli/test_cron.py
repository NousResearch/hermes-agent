"""Tests for hermes_cli.cron command handling."""

from argparse import Namespace

import pytest

import cron.jobs as jobs_mod
import cron.scheduler as scheduler_mod
import hermes_cli.cron as cron_mod
from cron.jobs import create_job, get_job, list_jobs, pause_job
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


def _block_unsafe_paths(monkeypatch):
    """Patch every helper the test-run command must NEVER call."""

    def boom(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"test-run must not call {name}; got args={args} kwargs={kwargs}")
        return _raise

    monkeypatch.setattr(scheduler_mod, "tick", boom("scheduler.tick"))
    monkeypatch.setattr(scheduler_mod, "_deliver_result", boom("scheduler._deliver_result"))
    monkeypatch.setattr(jobs_mod, "mark_job_run", boom("jobs.mark_job_run"))
    monkeypatch.setattr(jobs_mod, "advance_next_run", boom("jobs.advance_next_run"))
    monkeypatch.setattr(jobs_mod, "update_job", boom("jobs.update_job"))
    monkeypatch.setattr(jobs_mod, "get_due_jobs", boom("jobs.get_due_jobs"))
    monkeypatch.setattr(cron_mod, "_cron_api", boom("hermes_cli.cron._cron_api"))


class TestCronTestRun:
    """Phase 1 failing tests: `hermes cron test-run <job_id>` safe one-job verification."""

    def test_routes_to_cron_test_run(self, tmp_cron_dir, monkeypatch):
        """cron_command must dispatch 'test-run' to a new cron_test_run function."""
        calls = []

        def fake(job_id):
            calls.append(job_id)
            return 0

        monkeypatch.setattr(cron_mod, "cron_test_run", fake)
        cron_command(Namespace(cron_command="test-run", job_id="abc123"))
        assert calls == ["abc123"]

    def test_calls_get_job_once(self, tmp_cron_dir, monkeypatch):
        """The new command must look up exactly one job via get_job."""
        job = create_job(prompt="check", schedule="every 1h")
        seen = []

        def recorder(job_id):
            seen.append(job_id)
            return job

        monkeypatch.setattr(cron_mod, "get_job", recorder)
        monkeypatch.setattr(cron_mod, "run_job", lambda j: (True, "out", "final", None))
        monkeypatch.setattr(cron_mod, "save_job_output", lambda jid, out: "/tmp/x.md")
        cron_mod.cron_test_run(job["id"])
        assert seen == [job["id"]]

    def test_calls_run_job_once_with_fetched_job(self, tmp_cron_dir, monkeypatch):
        """run_job must be called exactly once with the job dict from get_job."""
        job = create_job(prompt="check", schedule="every 1h")
        runs = []

        def recorder(j):
            runs.append(j)
            return (True, "out", "final", None)

        monkeypatch.setattr(cron_mod, "run_job", recorder)
        monkeypatch.setattr(cron_mod, "save_job_output", lambda jid, out: "/tmp/x.md")
        cron_mod.cron_test_run(job["id"])
        assert len(runs) == 1
        assert runs[0]["id"] == job["id"]

    def test_saves_output_once(self, tmp_cron_dir, monkeypatch):
        """save_job_output must be called exactly once with (job_id, output)."""
        job = create_job(prompt="check", schedule="every 1h")
        saves = []

        def recorder(jid, out):
            saves.append((jid, out))
            return "/tmp/saved.md"

        monkeypatch.setattr(cron_mod, "run_job", lambda j: (True, "OUTPUT_BODY", "final", None))
        monkeypatch.setattr(cron_mod, "save_job_output", recorder)
        cron_mod.cron_test_run(job["id"])
        assert saves == [(job["id"], "OUTPUT_BODY")]

    def test_never_calls_unsafe_paths(self, tmp_cron_dir, monkeypatch):
        """test-run must not touch tick, delivery, run marking, schedule advancement, due lookup, or the queued cron API."""
        job = create_job(prompt="check", schedule="every 1h")
        _block_unsafe_paths(monkeypatch)
        monkeypatch.setattr(cron_mod, "run_job", lambda j: (True, "out", "final", None))
        monkeypatch.setattr(cron_mod, "save_job_output", lambda jid, out: "/tmp/x.md")
        cron_mod.cron_test_run(job["id"])

    def test_missing_job_returns_nonzero_without_running_or_saving(self, tmp_cron_dir, monkeypatch, capsys):
        def should_not_run(*args, **kwargs):
            raise AssertionError("missing job must not run or save output")

        monkeypatch.setattr(cron_mod, "run_job", should_not_run)
        monkeypatch.setattr(cron_mod, "save_job_output", should_not_run)

        rc = cron_mod.cron_test_run("missing-job-id")
        out = capsys.readouterr().out

        assert rc != 0
        assert "Job not found" in out
        assert "missing-job-id" in out

    def test_success_stdout_includes_evidence(self, tmp_cron_dir, monkeypatch, capsys):
        """Success path stdout must contain job id, success status, output path, and output length."""
        job = create_job(prompt="check", schedule="every 1h")
        output_body = "Hello world, this is the output."
        monkeypatch.setattr(cron_mod, "run_job", lambda j: (True, output_body, "final", None))
        monkeypatch.setattr(cron_mod, "save_job_output", lambda jid, out: "/tmp/cron/output/abc.md")
        rc = cron_mod.cron_test_run(job["id"])
        out = capsys.readouterr().out
        assert rc == 0
        assert job["id"] in out
        assert "success" in out.lower()
        assert "/tmp/cron/output/abc.md" in out
        assert str(len(output_body)) in out

    def test_failure_returns_nonzero_and_prints_error(self, tmp_cron_dir, monkeypatch, capsys):
        """Failure path must return non-zero, still save output, and print path + length + error text."""
        job = create_job(prompt="check", schedule="every 1h")
        partial = "Partial output before failure"
        monkeypatch.setattr(cron_mod, "run_job", lambda j: (False, partial, "", "boom: model errored"))
        monkeypatch.setattr(cron_mod, "save_job_output", lambda jid, out: "/tmp/cron/output/fail.md")
        rc = cron_mod.cron_test_run(job["id"])
        out = capsys.readouterr().out
        assert rc != 0
        assert job["id"] in out
        assert "/tmp/cron/output/fail.md" in out
        assert str(len(partial)) in out
        assert "boom: model errored" in out

    def test_preserves_paused_state_and_last_run(self, tmp_cron_dir, monkeypatch):
        """A paused job must remain paused and its last_run timestamp must not change."""
        job = create_job(prompt="check", schedule="every 1h")
        pause_job(job["id"])
        before = get_job(job["id"])
        assert before["state"] == "paused"
        last_run_before = before.get("last_run")

        monkeypatch.setattr(cron_mod, "run_job", lambda j: (True, "out", "final", None))
        monkeypatch.setattr(cron_mod, "save_job_output", lambda jid, out: "/tmp/x.md")
        cron_mod.cron_test_run(job["id"])

        after = get_job(job["id"])
        assert after["state"] == "paused"
        assert after.get("last_run") == last_run_before
