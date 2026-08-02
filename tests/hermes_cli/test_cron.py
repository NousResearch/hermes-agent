"""Tests for hermes_cli.cron command handling."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli.reliability_doctor import diagnose_cron
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
    monkeypatch.setattr(
        "cron.scheduler.tick", lambda verbose=False: calls.append(verbose)
    )

    cron_cli.cron_tick()

    assert calls == [True]


def test_cron_create_failure_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(
        cron_cli, "_cron_api", lambda **kwargs: {"success": False, "error": "boom"}
    )

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


def _base_create_args(**overrides):
    data = {
        "cron_command": "create",
        "schedule": "every 1h",
        "prompt": "Run",
        "name": "Smoke",
        "deliver": None,
        "repeat": None,
        "skill": None,
        "skills": None,
        "script": None,
        "workdir": None,
        "model": None,
        "model_provider": None,
        "no_agent": False,
        "smoke_file": None,
        "skip_preflight": False,
        "strict_preflight": False,
    }
    data.update(overrides)
    return Namespace(**data)


def _base_edit_args(job_id, **overrides):
    data = {
        "cron_command": "edit",
        "job_id": job_id,
        "schedule": None,
        "prompt": None,
        "name": None,
        "deliver": None,
        "repeat": None,
        "skill": None,
        "skills": None,
        "clear_skills": False,
        "add_skills": None,
        "remove_skills": None,
        "script": None,
        "workdir": None,
        "model": None,
        "model_provider": None,
        "no_agent": None,
        "smoke_file": None,
        "clear_smoke": False,
        "skip_preflight": False,
        "strict_preflight": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_cron_create_validates_smoke_file_before_saving(tmp_cron_dir, tmp_path, capsys):
    smoke_file = tmp_path / "smoke.yaml"
    smoke_file.write_text(
        "version: 1\nprobes:\n  - type: shell\n    command: echo nope\n"
    )

    rc = cron_cli.cron_create(_base_create_args(smoke_file=str(smoke_file)))

    assert rc == 1
    assert list_jobs(include_disabled=True) == []
    assert "Failed to load smoke file" in capsys.readouterr().out


def test_cron_create_rejects_command_probe_smoke_file_before_saving(
    tmp_cron_dir, tmp_path, capsys
):
    smoke_file = tmp_path / "smoke.yaml"
    smoke_file.write_text(
        "\n".join([
            "version: 1",
            "probes:",
            "  - type: command",
            "    argv: ['true']",
            "    expected_exit_codes: [0]",
        ]),
        encoding="utf-8",
    )

    rc = cron_cli.cron_create(_base_create_args(smoke_file=str(smoke_file)))

    assert rc == 1
    assert list_jobs(include_disabled=True) == []
    assert "Failed to load smoke file" in capsys.readouterr().out


def test_cron_create_saves_then_runs_static_preflight_by_default(
    tmp_cron_dir, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        cron_cli,
        "_run_saved_job_preflight",
        lambda job_id, strict: calls.append((job_id, strict)) or 0,
    )

    rc = cron_cli.cron_create(_base_create_args())

    [job] = list_jobs(include_disabled=True)
    assert rc == 0
    assert calls == [(job["id"], False)]


def test_cron_create_skip_preflight_does_not_diagnose(tmp_cron_dir, monkeypatch):
    monkeypatch.setattr(
        cron_cli,
        "_run_saved_job_preflight",
        lambda *args, **kwargs: pytest.fail("preflight should be skipped"),
        raising=False,
    )

    assert cron_cli.cron_create(_base_create_args(skip_preflight=True)) == 0


def test_cron_create_warn_mode_returns_zero_after_failed_preflight(
    tmp_cron_dir, monkeypatch, capsys
):
    monkeypatch.setattr(
        cron_cli,
        "diagnose_cron",
        lambda job_id: [
            cron_cli.DiagnosticResult(
                "cron", job_id, "smoke-schema", "smoke", "fail", "invalid_smoke"
            )
        ],
    )

    rc = cron_cli.cron_create(_base_create_args())

    out = capsys.readouterr().out
    assert rc == 0
    assert "Created job:" in out
    assert "Job saved. Preflight failed." in out


def test_cron_create_strict_mode_says_saved_and_returns_nonzero(
    tmp_cron_dir, monkeypatch, capsys
):
    monkeypatch.setattr(
        cron_cli,
        "diagnose_cron",
        lambda job_id: [
            cron_cli.DiagnosticResult(
                "cron", job_id, "smoke-schema", "smoke", "fail", "invalid_smoke"
            )
        ],
    )

    rc = cron_cli.cron_create(_base_create_args(strict_preflight=True))

    out = capsys.readouterr().out
    assert rc == 1
    assert "Created job:" in out
    assert "Job saved. Preflight failed." in out
    assert "rollback" not in out.lower()


def test_cron_create_no_agent_script_passes_strict_static_preflight(
    tmp_cron_dir, monkeypatch
):
    script = tmp_cron_dir / "scripts" / "watchdog.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.reliability_doctor.get_hermes_home", lambda: tmp_cron_dir
    )

    rc = cron_cli.cron_create(
        _base_create_args(
            prompt=None,
            no_agent=True,
            script="watchdog.py",
            strict_preflight=True,
        )
    )

    assert rc == 0


def test_cron_edit_clear_smoke_removes_metadata(tmp_cron_dir):
    job = create_job(
        prompt="Run",
        schedule="every 1h",
        smoke={"version": 1, "probes": [{"type": "env-present", "name": "GH_TOKEN"}]},
    )

    rc = cron_cli.cron_edit(
        _base_edit_args(job["id"], clear_smoke=True, skip_preflight=True)
    )

    assert rc == 0
    assert "smoke" not in get_job(job["id"])


def test_cron_edit_strict_failure_does_not_claim_rollback(
    tmp_cron_dir, monkeypatch, capsys
):
    job = create_job(prompt="Run", schedule="every 1h")
    monkeypatch.setattr(
        cron_cli,
        "diagnose_cron",
        lambda job_id: [
            cron_cli.DiagnosticResult("cron", job_id, "cron", job_id, "fail", "broken")
        ],
    )

    rc = cron_cli.cron_edit(
        _base_edit_args(job["id"], name="Updated", strict_preflight=True)
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert get_job(job["id"])["name"] == "Updated"
    assert "Updated job:" in out
    assert "Job saved. Preflight failed." in out
    assert "rollback" not in out.lower()


def test_cron_edit_invalid_yaml_does_not_mutate_job(tmp_cron_dir, tmp_path):
    job = create_job(prompt="Run", schedule="every 1h", name="Original")
    smoke_file = tmp_path / "smoke.yaml"
    smoke_file.write_text("version: [", encoding="utf-8")

    rc = cron_cli.cron_edit(
        _base_edit_args(job["id"], name="Mutated", smoke_file=str(smoke_file))
    )

    assert rc == 1
    assert get_job(job["id"])["name"] == "Original"


def test_integration_cron_static_smoke_create_diagnose_and_clear(
    tmp_cron_dir, tmp_path, monkeypatch
):
    workdir = tmp_path / "project"
    workdir.mkdir()
    (workdir / "marker.txt").write_text("ok", encoding="utf-8")
    smoke_file = tmp_path / "smoke.yaml"
    smoke_file.write_text(
        "\n".join([
            "version: 1",
            "probes:",
            "  - type: file-exists",
            "    root: workdir",
            "    path: marker.txt",
            "  - type: env-present",
            "    name: PATH",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])

    create_rc = cron_cli.cron_create(
        _base_create_args(
            smoke_file=str(smoke_file),
            workdir=str(workdir),
        )
    )

    [job] = list_jobs(include_disabled=True)
    assert create_rc == 0
    assert job["smoke"]["probes"][0]["type"] == "file-exists"

    static_results = diagnose_cron(job["id"])
    assert any(
        result.probe_type == "file-exists" and result.status == "pass"
        for result in static_results
    )
    assert any(
        result.probe_type == "env-present"
        and result.status == "pass"
        and result.reason == "env_present"
        for result in static_results
    )

    edit_rc = cron_cli.cron_edit(
        _base_edit_args(job["id"], clear_smoke=True, skip_preflight=True)
    )

    assert edit_rc == 0
    assert "smoke" not in get_job(job["id"])
