"""Tests for hermes_cli.cron command handling."""

import argparse
from argparse import Namespace
from types import SimpleNamespace

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli import cron as cron_cli
from hermes_cli.cron import cron_command
from hermes_cli.subcommands.cron import build_cron_parser


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCronCommandLifecycle:

    def test_edit_persists_user_owned_inference_pins(self, tmp_cron_dir, capsys):
        job = create_job(prompt="Daily report", schedule="every 1h")
        parser = argparse.ArgumentParser(prog="hermes")
        subparsers = parser.add_subparsers(dest="command")
        build_cron_parser(subparsers, cmd_cron=cron_command)

        args = parser.parse_args(
            [
                "cron",
                "edit",
                job["id"],
                "--model",
                "new-model",
                "--provider",
                "nous",
            ]
        )
        cron_command(args)

        updated = get_job(job["id"])
        assert updated["model"] == "new-model"
        assert updated["provider"] == "nous"
        assert "Updated job" in capsys.readouterr().out

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


class TestCronRunBackgroundDispatch:
    """`hermes cron run` must not report 'failed' when the run was dispatched
    to the background delegation worker.

    The CLI process inherits the gateway/desktop session env, so a manual run
    can be dispatched to the daemon instead of executing inline. Such
    responses carry execution_mode='background' / delegation_id and the job
    keeps running after the CLI exits — a terminal success/failure verdict
    would be a lie (#83340). The CLI must report the background dispatch
    instead, and leave synchronous runs unchanged.
    """

    def _run_cmd(self, capsys):
        rc = cron_command(Namespace(cron_command="run", job_id="job-1"))
        return rc, capsys.readouterr().out

    def test_background_dispatch_with_delegation_id_does_not_report_failed(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            cron_cli,
            "_cron_api",
            lambda **kwargs: {
                "success": True,
                "job": {
                    "id": "job-1",
                    "name": "Watchdog",
                    "execution_mode": "background",
                    "delegation_id": "del-abc123",
                    # No execution_success — the inline verdict must not apply.
                    "executed": True,
                },
            },
        )

        rc, out = self._run_cmd(capsys)

        assert rc == 0
        assert "Running in background (delegation del-abc123)." in out
        assert "failed" not in out.lower()
        assert "Ran now" not in out

    def test_background_dispatch_without_delegation_id(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cron_cli,
            "_cron_api",
            lambda **kwargs: {
                "success": True,
                "job": {
                    "id": "job-1",
                    "name": "Watchdog",
                    "execution_mode": "background",
                },
            },
        )

        rc, out = self._run_cmd(capsys)

        assert rc == 0
        assert "Running in background." in out
        assert "failed" not in out.lower()

    def test_sync_run_success_unchanged(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cron_cli,
            "_cron_api",
            lambda **kwargs: {
                "success": True,
                "job": {
                    "id": "job-1",
                    "name": "Watchdog",
                    "executed": True,
                    "execution_success": True,
                },
            },
        )

        rc, out = self._run_cmd(capsys)

        assert rc == 0
        assert "Ran now: succeeded." in out

    def test_sync_run_failure_still_reported(self, monkeypatch, capsys):
        # A genuine synchronous failure must keep reporting 'failed' — only
        # background-dispatched runs are exempt from the terminal verdict.
        monkeypatch.setattr(
            cron_cli,
            "_cron_api",
            lambda **kwargs: {
                "success": True,
                "job": {
                    "id": "job-1",
                    "name": "Watchdog",
                    "executed": True,
                    "execution_success": False,
                },
            },
        )

        rc, out = self._run_cmd(capsys)

        assert rc == 0
        assert "Ran now: failed." in out

    def test_delegation_id_alone_counts_as_background(self, monkeypatch, capsys):
        # Some dispatchers may not set execution_mode but always return the
        # delegation_id — either marker alone must suppress the verdict.
        monkeypatch.setattr(
            cron_cli,
            "_cron_api",
            lambda **kwargs: {
                "success": True,
                "job": {"id": "job-1", "name": "Watchdog", "delegation_id": "del-xyz"},
            },
        )

        rc, out = self._run_cmd(capsys)

        assert rc == 0
        assert "Running in background (delegation del-xyz)." in out
        assert "failed" not in out.lower()
