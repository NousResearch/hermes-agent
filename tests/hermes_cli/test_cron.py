"""Tests for hermes_cli.cron command handling."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli import cron as cron_cli
from hermes_cli.cron import cron_command


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCronCommandLifecycle:

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
                add_skills=None,
                remove_skills=None,
                script=None,
                workdir=None,
                no_agent=None,
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
                add_skills=None,
                remove_skills=None,
                script=None,
                workdir=None,
                no_agent=None,
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
                script=None,
                workdir=None,
                no_agent=False,
            )
        )
        out = capsys.readouterr().out
        assert "Created job" in out

        jobs = list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["skills"] == ["blogwatcher", "maps"]
        assert jobs[0]["name"] == "Skill combo"



class TestGatewayNotRunningWarning:
    """`cron create` / `cron list` must warn when the gateway (and thus the
    cron ticker) isn't running, since jobs only fire inside the gateway.
    Regression guard for #51038 — the most common cron 'jobs never fired'
    report was simply a gateway that was never started.
    """


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


def test_cron_list_warns_when_gateway_not_running(monkeypatch, capsys):
    monkeypatch.setattr(
        "cron.jobs.list_jobs",
        lambda include_disabled=False: [
            {
                "id": "job-1",
                "name": "Nightly docs",
                "schedule_display": "every day",
                "state": "scheduled",
                "enabled": True,
                "next_run_at": "2026-06-01T00:00:00Z",
                "deliver": ["local"],
            }
        ],
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "builtin")

    cron_cli.cron_list()

    out = capsys.readouterr().out
    assert "Gateway is not running" in out
    assert "Nightly docs" in out


def test_cron_tick_invokes_scheduler_tick_with_verbose(monkeypatch):
    calls = []
    monkeypatch.setattr("cron.scheduler.tick", lambda verbose=False: calls.append(verbose))

    cron_cli.cron_tick()

    assert calls == [True]


def test_cron_create_failure_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(cron_cli, "_cron_api", lambda **kwargs: {"success": False, "error": "boom"})

    args = SimpleNamespace(
        schedule="every day",
        prompt="refresh docs",
        name=None,
        deliver=None,
        repeat=None,
        skill=None,
        skills=None,
        script=None,
        workdir=None,
        no_agent=False,
    )

    rc = cron_cli.cron_create(args)

    out = capsys.readouterr().out
    assert rc == 1
    assert "Failed to create job: boom" in out


class TestCronInterpreterCli:
    """``hermes cron create/edit`` plumb the ``--interpreter`` flag through."""

    def test_create_passes_interpreter_to_api(self, monkeypatch):
        captured = {}

        def fake_api(**kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "job_id": "job-1",
                "name": "Report",
                "schedule": "0 8 * * *",
                "skills": [],
                "next_run_at": "2026-06-01T00:00:00Z",
                "job": {"interpreter": "~/venvs/reporting/bin/python3"},
            }

        monkeypatch.setattr(cron_cli, "_cron_api", fake_api)
        monkeypatch.setattr(cron_cli, "_warn_if_gateway_not_running", lambda: None)

        args = SimpleNamespace(
            schedule="0 8 * * *",
            prompt="Daily report",
            name="Report",
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script=None,
            workdir=None,
            interpreter="~/venvs/reporting/bin/python3",
            no_agent=False,
        )
        rc = cron_cli.cron_create(args)
        assert rc == 0
        # The flag must be forwarded to the underlying tool/api.
        assert captured.get("interpreter") == "~/venvs/reporting/bin/python3"

    def test_create_shows_interpreter_in_confirmation(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cron_cli,
            "_cron_api",
            lambda **kwargs: {
                "success": True,
                "job_id": "job-1",
                "name": "Report",
                "schedule": "0 8 * * *",
                "skills": [],
                "next_run_at": "2026-06-01T00:00:00Z",
                "job": {"interpreter": "~/venvs/reporting/bin/python3"},
            },
        )
        monkeypatch.setattr(cron_cli, "_warn_if_gateway_not_running", lambda: None)

        args = SimpleNamespace(
            schedule="0 8 * * *",
            prompt="Daily report",
            name="Report",
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script=None,
            workdir=None,
            interpreter="~/venvs/reporting/bin/python3",
            no_agent=False,
        )
        cron_cli.cron_create(args)
        out = capsys.readouterr().out
        assert "Python: ~/venvs/reporting/bin/python3" in out

    def test_edit_passes_interpreter_to_api(self, monkeypatch):
        captured = {}

        def fake_api(**kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "job": {
                    "job_id": "job-1",
                    "name": "Report",
                    "schedule": "0 8 * * *",
                    "skills": [],
                    "interpreter": "~/venvs/reporting/bin/python3",
                },
            }

        monkeypatch.setattr(cron_cli, "_cron_api", fake_api)

        # resolve_job_ref is called at the top of cron_edit; point the job it
        # returns at a real record so the skills-diff logic has something to
        # read. The captured kwargs are what we assert on.
        job = create_job(prompt="Report", schedule="0 8 * * *")
        monkeypatch.setattr(
            "cron.jobs.resolve_job_ref", lambda ref: get_job(job["id"])
        )

        args = SimpleNamespace(
            job_id=job["id"],
            schedule=None,
            prompt=None,
            name=None,
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            add_skills=None,
            remove_skills=None,
            clear_skills=False,
            script=None,
            workdir=None,
            interpreter="~/venvs/reporting/bin/python3",
            no_agent=None,
        )
        rc = cron_cli.cron_edit(args)
        assert rc == 0
        assert captured.get("interpreter") == "~/venvs/reporting/bin/python3"

    def test_list_shows_interpreter_when_set(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr(cron_cli, "_warn_if_gateway_not_running", lambda: None)
        create_job(
            prompt="Report",
            schedule="0 8 * * *",
            script="daily.py",
            interpreter="~/venvs/reporting/bin/python3",
        )
        cron_command(Namespace(cron_command="list", all=True))
        out = capsys.readouterr().out
        assert "Python:" in out
        assert "~/venvs/reporting/bin/python3" in out
