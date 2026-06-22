import json
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

from hermes_cli import health as health_mod
from hermes_cli.subcommands.health import build_health_parser


def _write_profile(home, *, model=True, state_db=True):
    home.mkdir(parents=True, exist_ok=True)
    if model:
        (home / "config.yaml").write_text(
            "model:\n  provider: test-provider\n  default: test-model\n",
            encoding="utf-8",
        )
    else:
        (home / "config.yaml").write_text("display:\n  interface: cli\n", encoding="utf-8")
    if state_db:
        conn = sqlite3.connect(home / "state.db")
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id text)")
        conn.commit()
        conn.close()
    cron_dir = home / "cron"
    cron_dir.mkdir(exist_ok=True)
    (cron_dir / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "completed-job",
                        "enabled": True,
                        "last_run_at": "2026-05-28T12:00:01+00:00",
                        "last_status": "ok",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _run_cli(home, *args):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = os.getcwd()
    env.setdefault("HERMES_DISABLE_UPDATE_CHECK", "1")
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def _run_health_cli(home, *args):
    return _run_cli(home, "health", *args)


def test_collect_health_healthy_exit_zero(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = health_mod.collect_health()

    assert result["status"] == "healthy"
    assert result["exit_code"] == 0
    assert result["schema_version"] == 1
    assert "hermes_version" in result
    assert {row["id"] for row in result["checks"]} >= {
        "profile_config",
        "state_db",
        "cron_storage",
        "provider_routing",
        "disk",
        "runtime_modules",
    }
    provider_row = next(row for row in result["checks"] if row["id"] == "provider_routing")
    assert "no provider/network probe run" in provider_row["detail"]


def test_collect_health_warning_exit_one_for_missing_state_db(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile(home, state_db=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = health_mod.collect_health()

    assert result["status"] == "warning"
    assert result["exit_code"] == 1
    state_row = next(row for row in result["checks"] if row["id"] == "state_db")
    assert state_row["status"] == "warning"


def test_collect_health_critical_exit_two_for_bad_config(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("model: [unterminated\n", encoding="utf-8")
    conn = sqlite3.connect(home / "state.db")
    conn.execute("CREATE TABLE t (id integer)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = health_mod.collect_health()

    assert result["status"] == "critical"
    assert result["exit_code"] == 2
    assert any(row["id"] == "profile_config" and row["status"] == "critical" for row in result["checks"])


def test_collect_health_critical_exit_two_for_non_mapping_config(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    sqlite3.connect(home / "state.db").close()
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = health_mod.collect_health()

    assert result["status"] == "critical"
    assert result["exit_code"] == 2
    config_row = next(row for row in result["checks"] if row["id"] == "profile_config")
    assert config_row["status"] == "critical"
    assert "must be a mapping/object" in config_row["detail"]


def test_collect_health_reads_legacy_cron_without_mutating(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile(home)
    jobs_path = home / "cron" / "jobs.json"
    original = '[{"id":"legacy-job","enabled":true}]\n'
    jobs_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = health_mod.collect_health()

    cron_row = next(row for row in result["checks"] if row["id"] == "cron_storage")
    assert cron_row["status"] == "warning"
    assert "legacy jobs.json list format" in cron_row["detail"]
    assert jobs_path.read_text(encoding="utf-8") == original


def test_collect_health_reports_latest_persisted_cron_run(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile(home)
    jobs_path = home / "cron" / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {"id": "older", "last_run_at": "2026-05-28T11:00:00+00:00"},
                    {"id": "never-run"},
                    {"id": "latest", "last_run_at": "2026-05-28T13:00:00+00:00"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = health_mod.collect_health()

    cron_row = next(row for row in result["checks"] if row["id"] == "cron_storage")
    assert "last run 2026-05-28T13:00:00+00:00" in cron_row["detail"]


def test_collect_health_orders_cron_runs_by_instant_across_offsets(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile(home)
    jobs_path = home / "cron" / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {"id": "lexically-later", "last_run_at": "2026-05-28T13:00:00+02:00"},
                    {"id": "chronologically-later", "last_run_at": "2026-05-28T12:00:00+00:00"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = health_mod.collect_health()

    cron_row = next(row for row in result["checks"] if row["id"] == "cron_storage")
    assert "last run 2026-05-28T12:00:00+00:00" in cron_row["detail"]


def test_collect_health_does_not_mutate_profile_home(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile(home)
    (home / "config.yaml").write_text("model: [unterminated\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = {
        path.relative_to(home): path.read_bytes()
        for path in home.rglob("*")
        if path.is_file()
    }

    health_mod.collect_health()

    after = {
        path.relative_to(home): path.read_bytes()
        for path in home.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_run_health_json_outputs_machine_readable_contract(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profile"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    code = health_mod.run_health(SimpleNamespace(json=True, quiet=False))

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert code == 0
    assert payload["schema_version"] == 1
    assert payload["status"] == "healthy"
    assert payload["exit_code"] == 0
    assert "hermes_version" in payload
    assert isinstance(payload["checks"], list)
    assert all("id" in row for row in payload["checks"])


def test_quiet_healthy_output_is_empty(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profile"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    code = health_mod.run_health(SimpleNamespace(json=False, quiet=True))

    assert code == 0
    assert capsys.readouterr().out == ""


def test_quiet_warning_still_prints_human_output(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profile"
    _write_profile(home, state_db=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    code = health_mod.run_health(SimpleNamespace(json=False, quiet=True))

    out = capsys.readouterr().out
    assert code == 1
    assert "Hermes Health" in out
    assert "state DB availability" in out


def test_status_aggregation_prefers_highest_severity():
    rows = [
        health_mod.HealthRow("a", "a", "healthy", "ok"),
        health_mod.HealthRow("b", "b", "warning", "warn"),
        health_mod.HealthRow("c", "c", "critical", "bad"),
    ]
    assert health_mod._aggregate_status(rows) == "critical"


def test_early_cli_subcommand_distinguishes_command_from_argument():
    from hermes_cli.main import _early_cli_subcommand

    assert _early_cli_subcommand(["chat", "health"]) == "chat"
    assert _early_cli_subcommand(["--profile", "dev", "health"]) == "health"
    assert _early_cli_subcommand(["--profile=dev", "health"]) == "health"
    assert _early_cli_subcommand(["--provider", "auto", "health"]) == "health"
    assert _early_cli_subcommand(["--model", "test-model", "health"]) == "health"
    assert _early_cli_subcommand(["--toolsets", "all", "health"]) == "health"
    assert _early_cli_subcommand(["--model", "health", "chat"]) == "chat"


def test_health_subparser_registered():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    def sentinel(args):
        return args

    build_health_parser(subparsers, cmd_health=sentinel)

    args = parser.parse_args(["health", "--json", "--quiet"])

    assert args.command == "health"
    assert args.json is True
    assert args.quiet is True
    assert args.func is sentinel


def test_health_cli_e2e_json_exit_zero_with_temp_home(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home)

    proc = _run_health_cli(home, "--json")

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "healthy"
    assert payload["exit_code"] == 0
    assert payload["hermes_home"] == str(home)


def test_health_cli_e2e_does_not_mutate_profile_home(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home)
    (home / ".env").write_bytes(b"OPENROUTER_API_KEY=abc\x00def\n")
    before = {
        path.relative_to(home): path.read_bytes()
        for path in home.rglob("*")
        if path.is_file()
    }

    proc = _run_health_cli(home, "--json")

    assert proc.returncode == 0, proc.stderr
    after = {
        path.relative_to(home): path.read_bytes()
        for path in home.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_health_cli_e2e_global_value_options_do_not_enable_file_logging(tmp_path):
    invocations = (
        ("--provider", "auto", "health", "--json"),
        ("--model", "test-model", "health", "--json"),
        ("--toolsets", "all", "health", "--json"),
    )
    for index, invocation in enumerate(invocations):
        home = tmp_path / f"profile-{index}"
        _write_profile(home)
        before = {
            path.relative_to(home): path.read_bytes()
            for path in home.rglob("*")
            if path.is_file()
        }

        proc = _run_cli(home, *invocation)

        assert proc.returncode == 0, proc.stderr
        after = {
            path.relative_to(home): path.read_bytes()
            for path in home.rglob("*")
            if path.is_file()
        }
        assert after == before


def test_health_cli_e2e_does_not_invoke_external_secret_source(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile(home)
    with (home / "config.yaml").open("a", encoding="utf-8") as config_file:
        config_file.write(
            "secrets:\n"
            "  bitwarden:\n"
            "    enabled: true\n"
            "    project_id: proj-1\n"
            "    access_token_env: BWS_ACCESS_TOKEN\n"
            "    auto_install: false\n"
        )
    marker = tmp_path / "bws-invoked"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_bws = bin_dir / "bws"
    fake_bws.write_text(
        f"#!/bin/sh\n: > '{marker}'\nprintf '[]'\n",
        encoding="utf-8",
    )
    fake_bws.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test.token")

    proc = _run_health_cli(home, "--json")

    assert proc.returncode == 0, proc.stderr
    assert not marker.exists()


def test_health_cli_e2e_missing_state_db_exits_one(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home, state_db=False)

    proc = _run_health_cli(home, "--json")

    assert proc.returncode == 1, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "warning"
    assert payload["exit_code"] == 1


def test_health_cli_e2e_bad_config_exits_two(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("model: [unterminated\n", encoding="utf-8")
    sqlite3.connect(home / "state.db").close()

    proc = _run_health_cli(home, "--json")

    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "critical"
    assert payload["exit_code"] == 2


def test_health_cli_e2e_quiet_healthy_is_silent(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home)

    proc = _run_health_cli(home, "--quiet")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
