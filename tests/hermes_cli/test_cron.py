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

    def test_list_does_not_crash_when_repeat_is_null(self, tmp_cron_dir, capsys):
        """A one-shot job can be persisted with ``"repeat": null``. `cron
        list` must render it as ∞ rather than crashing on .get(...)\\.get."""
        from cron.jobs import load_jobs, save_jobs

        create_job(prompt="One shot", schedule="every 1h")
        # Force the present-but-null shape that .get("repeat", {}) mishandles.
        jobs = load_jobs()
        jobs[0]["repeat"] = None
        save_jobs(jobs)

        cron_command(Namespace(cron_command="list", all=True))

        out = capsys.readouterr().out
        assert "Repeat:    ∞" in out

    def test_list_does_not_crash_when_deliver_is_null(self, tmp_cron_dir, capsys):
        """A job can be persisted with ``"deliver": null`` (present-but-null).
        `cron list` must fall back to the default channel rather than crashing
        on ``", ".join(None)`` — same dict-default pitfall as ``repeat`` (#32896).
        """
        from cron.jobs import load_jobs, save_jobs

        create_job(prompt="No deliver", schedule="every 1h")
        jobs = load_jobs()
        jobs[0]["deliver"] = None
        save_jobs(jobs)

        cron_command(Namespace(cron_command="list", all=True))

        out = capsys.readouterr().out
        assert "Deliver:   local" in out


class TestGatewayNotRunningWarning:
    """`cron create` / `cron list` must warn when the gateway (and thus the
    cron ticker) isn't running, since jobs only fire inside the gateway.
    Regression guard for #51038 — the most common cron 'jobs never fired'
    report was simply a gateway that was never started.
    """

    def test_create_warns_when_gateway_absent(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(
            Namespace(
                cron_command="create",
                schedule="0 11 * * *",
                prompt="Daily report",
                name="Daily 1130",
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                script=None,
                workdir=None,
                no_agent=False,
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out
        assert "Gateway is not running" in out

    def test_create_silent_when_gateway_running(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4242])
        cron_command(
            Namespace(
                cron_command="create",
                schedule="0 11 * * *",
                prompt="Daily report",
                name="Daily 1130",
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                script=None,
                workdir=None,
                no_agent=False,
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out
        assert "Gateway is not running" not in out

    def test_list_warns_when_gateway_absent(self, tmp_cron_dir, capsys, monkeypatch):
        create_job(prompt="Daily report", schedule="0 11 * * *")
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(Namespace(cron_command="list", all=True))
        out = capsys.readouterr().out
        assert "Gateway is not running" in out


class TestExternalCronProviderStatus:
    """With an external cron provider (e.g. Chronos), jobs fire via a
    NAS-mediated webhook, NOT the in-process ticker. The ticker-heartbeat /
    gateway-process heuristics are meaningless there, so neither
    `cron status` nor the create/list warning must claim the gateway being
    absent means jobs won't fire — that was a false-negative on every healthy
    Chronos instance (the heartbeat is intentionally never written).
    """

    def test_status_reports_provider_not_ticker_for_chronos(
        self, tmp_cron_dir, capsys, monkeypatch
    ):
        create_job(prompt="Ping", schedule="every 2m")
        monkeypatch.setattr(
            "hermes_cli.cron._active_cron_provider_name", lambda: "chronos"
        )
        # Even with NO gateway process and NO ticker heartbeat, Chronos status
        # must NOT report a stall / "not firing".
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(Namespace(cron_command="status"))
        out = capsys.readouterr().out
        assert "chronos" in out
        assert "managed scheduler" in out
        assert "not firing" not in out.lower()
        assert "STALLED" not in out
        assert "Gateway is not running" not in out
        # Still surfaces the active-job summary.
        assert "active job(s)" in out

    def test_status_unchanged_for_builtin(self, tmp_cron_dir, capsys, monkeypatch):
        create_job(prompt="Ping", schedule="every 2m")
        monkeypatch.setattr(
            "hermes_cli.cron._active_cron_provider_name", lambda: "builtin"
        )
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(Namespace(cron_command="status"))
        out = capsys.readouterr().out
        # Built-in path is the historical ticker-based report.
        assert "Gateway is not running" in out
        assert "managed scheduler" not in out

    def test_create_silent_for_chronos_even_without_gateway(
        self, tmp_cron_dir, capsys, monkeypatch
    ):
        # The create-time "gateway not running" nag is a ticker-only concern;
        # an external provider doesn't depend on a live in-process ticker.
        monkeypatch.setattr(
            "hermes_cli.cron._active_cron_provider_name", lambda: "chronos"
        )
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
        cron_command(
            Namespace(
                cron_command="create",
                schedule="every 2m",
                prompt="Ping",
                name="Ping",
                deliver=None,
                repeat=None,
                skill=None,
                skills=None,
                script=None,
                workdir=None,
                no_agent=False,
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out
        assert "Gateway is not running" not in out
