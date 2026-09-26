import json
import base64
import hashlib
import os
import sqlite3
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_state_db_rejects_corrupt_bytes_without_uri_path_truncation(tmp_path):
    home = tmp_path / "profile?reserved#percent%"
    home.mkdir()
    db_path = home / "state.db"
    original = b"not sqlite at all"
    db_path.write_bytes(original)

    row = health_mod._check_state_db(home)

    assert row.status == "critical"
    assert db_path.read_bytes() == original
    assert not (tmp_path / "profile").exists()


def test_state_db_wal_snapshot_does_not_create_profile_shm(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    source_db = source / "state.db"
    conn = sqlite3.connect(source_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE sessions (id text)")
    conn.execute("INSERT INTO sessions VALUES ('from-wal')")
    conn.commit()
    home = tmp_path / "profile"
    home.mkdir()
    db_path = home / "state.db"
    db_path.write_bytes(source_db.read_bytes())
    Path(f"{db_path}-wal").write_bytes(Path(f"{source_db}-wal").read_bytes())
    conn.close()
    before = {path.name: path.read_bytes() for path in home.iterdir()}

    row = health_mod._check_state_db(home)

    assert row.status == "healthy"
    assert {path.name: path.read_bytes() for path in home.iterdir()} == before
    assert not Path(f"{db_path}-shm").exists()


def test_state_db_snapshot_retries_across_wal_checkpoint(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    db_path = home / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE sessions (id text)")
    conn.execute("INSERT INTO sessions VALUES ('from-wal')")
    conn.commit()
    wal_path = Path(f"{db_path}-wal")
    assert wal_path.exists()

    original_copy = health_mod._copy_with_sha256
    checkpointed = False

    def copy_and_checkpoint(source, destination):
        nonlocal checkpointed
        digest = original_copy(source, destination)
        if source == wal_path and not checkpointed:
            checkpointed = True
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return digest

    monkeypatch.setattr(health_mod, "_copy_with_sha256", copy_and_checkpoint)
    try:
        row = health_mod._check_state_db(home)
    finally:
        conn.close()

    assert checkpointed is True
    assert row.status == "healthy"


def test_state_db_snapshot_copies_rollback_journal(tmp_path):
    db_path = tmp_path / "state.db"
    snapshot_path = tmp_path / "snapshot" / "state.db"
    snapshot_path.parent.mkdir()
    db_path.write_bytes(b"database generation")
    journal_path = Path(f"{db_path}-journal")
    journal_path.write_bytes(b"rollback journal generation")

    health_mod._copy_coherent_state_snapshot(db_path, snapshot_path)

    assert snapshot_path.read_bytes() == db_path.read_bytes()
    assert Path(f"{snapshot_path}-journal").read_bytes() == journal_path.read_bytes()


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
    config_row = next(row for row in result["checks"] if row["id"] == "profile_config")
    assert config_row["status"] == "critical"
    assert "line 2, column 1" in config_row["detail"]


def test_bad_config_diagnostic_does_not_disclose_source_line(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profile"
    home.mkdir()
    sentinel = "SUPER_SECRET_CREDENTIAL_123"
    (home / "config.yaml").write_text(f"api_key: [{sentinel}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    for json_output in (True, False):
        code = health_mod.run_health(SimpleNamespace(json=json_output, quiet=False))
        output = capsys.readouterr().out
        assert code == 2
        assert sentinel not in output
        assert "line 2, column 1" in output


def test_bad_config_diagnostic_does_not_disclose_tag_or_alias(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    sqlite3.connect(home / "state.db").close()
    sentinel = "SUPER_SECRET_CREDENTIAL_123"

    for malformed in (
        f"key: !{sentinel} value\n",
        f"key: *{sentinel}\n",
    ):
        (home / "config.yaml").write_text(malformed, encoding="utf-8")
        for output_args in (("--json",), ()):
            proc = _run_health_cli(home, *output_args)
            assert proc.returncode == 2, proc.stderr
            assert sentinel not in proc.stdout
            assert "config.yaml invalid" in proc.stdout


def test_unexpected_health_failure_does_not_disclose_exception_text(monkeypatch, capsys):
    sentinel = "SUPER_SECRET_CREDENTIAL_123"

    def fail_collection():
        raise RuntimeError(sentinel)

    monkeypatch.setattr(health_mod, "collect_health", fail_collection)
    for json_output in (True, False):
        code = health_mod.run_health(SimpleNamespace(json=json_output, quiet=False))
        output = capsys.readouterr().out
        assert code == 2
        assert sentinel not in output
        assert "health collection failed (RuntimeError)" in output


def test_row_failures_do_not_disclose_exception_text(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profile"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    sentinel = "SUPER_SECRET_CREDENTIAL_123"
    original_read_text = Path.read_text

    def raise_sentinel(*_args, **_kwargs):
        raise RuntimeError(sentinel)

    def install_cron_failure(patcher):
        def read_text(path, *args, **kwargs):
            if path.name == "jobs.json":
                raise RuntimeError(sentinel)
            return original_read_text(path, *args, **kwargs)

        patcher.setattr(Path, "read_text", read_text)

    failure_installers = (
        lambda patcher: patcher.setattr(
            health_mod, "_copy_coherent_state_snapshot", raise_sentinel
        ),
        install_cron_failure,
        lambda patcher: patcher.setattr(health_mod.shutil, "disk_usage", raise_sentinel),
    )

    for install_failure in failure_installers:
        with monkeypatch.context() as patcher:
            install_failure(patcher)
            for json_output in (True, False):
                code = health_mod.run_health(SimpleNamespace(json=json_output, quiet=False))
                output = capsys.readouterr().out
                assert code == 2
                assert sentinel not in output
                if json_output:
                    assert "RuntimeError" in output


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


@pytest.mark.parametrize("jobs", [
    {"outer": {"id": "inline", "enabled": True}, "keyed": {"enabled": False}},
    {"keyed": {"enabled": True}, "junk": "ignored by runtime"},
    {},
])
def test_health_reads_id_keyed_cron_without_repair(tmp_path, monkeypatch, jobs):
    home = tmp_path / "profile"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    jobs_path = home / "cron" / "jobs.json"
    original = json.dumps({"jobs": jobs}).encode("utf-8")
    jobs_path.write_bytes(original)
    before = _file_bytes_under(home)
    records, storage, error = health_mod._read_cron_jobs_read_only(home)
    assert records == [{**value, "id": value.get("id") or key}
                       for key, value in jobs.items() if isinstance(value, dict)]
    assert storage == "legacy_map"
    assert error is None
    result = health_mod.collect_health()
    cron_row = next(row for row in result["checks"] if row["id"] == "cron_storage")
    assert cron_row["status"] == "warning"
    assert "ID-keyed" in cron_row["detail"]
    assert result["exit_code"] == 1
    assert _file_bytes_under(home) == before


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


def test_collect_health_ignores_timestamp_that_overflows_utc_normalization(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile(home)
    (home / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"last_run_at": "0001-01-01T00:00:00+14:00"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = health_mod.collect_health()

    cron_row = next(row for row in result["checks"] if row["id"] == "cron_storage")
    assert "no run history yet" in cron_row["detail"]


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


def test_run_health_converts_unexpected_collection_failure_to_critical_json(monkeypatch, capsys):
    def fail_collection():
        raise OSError("broken storage")

    monkeypatch.setattr(health_mod, "collect_health", fail_collection)
    code = health_mod.run_health(SimpleNamespace(json=True, quiet=False))

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "critical"
    assert payload["checks"][0]["id"] == "health_collection"


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
    from hermes_cli._early_recovery import early_cli_subcommand as _early_cli_subcommand

    assert _early_cli_subcommand(["chat", "health"]) == "chat"
    assert _early_cli_subcommand(["--profile", "dev", "health"]) == "health"
    assert _early_cli_subcommand(["--profile=dev", "health"]) == "health"
    assert _early_cli_subcommand(["--provider", "auto", "health"]) == "health"
    assert _early_cli_subcommand(["--model", "test-model", "health"]) == "health"
    assert _early_cli_subcommand(["--toolsets", "all", "health"]) == "health"
    assert _early_cli_subcommand(["--reasoning", "high", "health"]) == "health"
    assert _early_cli_subcommand(["--in", "/tmp/workspace", "health"]) == "health"
    assert _early_cli_subcommand(["--model", "health", "chat"]) == "chat"


def test_health_process_reports_broken_dependency_without_early_repair(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='probe'\n", encoding="utf-8")
    recovery_marker = root / ".update-incomplete"
    recovery_marker.write_text("interrupted\n", encoding="utf-8")
    original_marker = recovery_marker.read_bytes()
    repair_marker = tmp_path / "repair-invoked"
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "ruamel").mkdir()
    (shadow / "ruamel" / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "ruamel" / "yaml.py").write_text(
        "raise ImportError('deliberately broken ruamel')\n", encoding="utf-8"
    )
    script = f"""
import pathlib
import sys
import hermes_cli._early_recovery as recovery
recovery._project_root = lambda: pathlib.Path({str(root)!r})
def forbidden_lock(project_root):
    pathlib.Path({str(repair_marker)!r}).write_text("called")
    raise AssertionError("health attempted dependency repair")
recovery._claim_recovery_lock = forbidden_lock
sys.argv = ["hermes", "health", "--json"]
recovery.recover_if_needed(pathlib.Path({str(root)!r}), argv=sys.argv[1:])
recovery._emit_health_dependency_failure(sys.argv[1:], recovery._probe_broken_packages())
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": f"{shadow}{os.pathsep}{os.getcwd()}"},
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "critical"
    assert payload["exit_code"] == 2
    assert payload["checks"][0]["id"] == "runtime_dependencies"
    assert "ruamel.yaml" in payload["checks"][0]["detail"]
    assert recovery_marker.read_bytes() == original_marker
    assert not repair_marker.exists()


def test_broken_dependency_health_resolves_explicit_and_sticky_profiles(tmp_path):
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "ruamel").mkdir()
    (shadow / "ruamel" / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "ruamel" / "yaml.py").write_text(
        "raise ImportError('deliberately broken ruamel')\n", encoding="utf-8"
    )
    user_home = tmp_path / "user"
    hermes_root = user_home / ".hermes"
    (hermes_root / "profiles" / "coder").mkdir(parents=True)
    (hermes_root / "active_profile").write_text("coder\n", encoding="utf-8")
    script = """
import sys
sys.argv = ["hermes", *sys.argv[1:]]
import hermes_cli.main
"""
    base_env = {
        **os.environ,
        "HOME": str(user_home),
        "PYTHONPATH": f"{shadow}{os.pathsep}{os.getcwd()}",
    }
    base_env.pop("HERMES_HOME", None)
    base_env.pop("HERMES_PROFILE", None)

    for args in (
        ["--profile", "coder", "health", "--json"],
        ["-p", "coder", "health", "--json"],
        ["--profile=coder", "health", "--json"],
        ["health", "--json"],
    ):
        proc = subprocess.run(
            [sys.executable, "-c", script, *args],
            cwd=os.getcwd(),
            env=base_env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 2, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["profile"] == "coder"
        assert payload["hermes_home"] == str(hermes_root / "profiles" / "coder")

    custom_root = tmp_path / "custom-hermes-root"
    (custom_root / "profiles" / "coder").mkdir(parents=True)
    custom_env = {**base_env, "HERMES_HOME": str(custom_root)}
    proc = subprocess.run(
        [sys.executable, "-c", script, "--profile", "coder", "health", "--json"],
        cwd=os.getcwd(),
        env=custom_env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["profile"] == "coder"
    assert payload["hermes_home"] == str(custom_root / "profiles" / "coder")


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


def test_health_main_skips_mutating_startup_maintenance(monkeypatch):
    import hermes_cli.main as main_mod

    def forbidden():
        raise AssertionError("health invoked mutating startup maintenance")

    monkeypatch.setattr(main_mod, "_EARLY_CLI_COMMAND", "health")
    monkeypatch.setattr(main_mod, "_cleanup_quarantined_exes", forbidden)
    monkeypatch.setattr(main_mod, "_sweep_stale_bytecode_if_checkout_changed", forbidden)
    monkeypatch.setattr(main_mod, "_try_termux_fast_tui_launch", lambda: True)

    main_mod.main()


def test_health_json_skips_early_tui_mouse_output(monkeypatch):
    import hermes_cli.main as main_mod

    writes = []
    monkeypatch.setattr(sys, "argv", ["hermes", "--tui", "health", "--json"])
    monkeypatch.setenv("HERMES_TUI", "1")
    monkeypatch.setattr(main_mod.os, "isatty", lambda _fd: True)
    monkeypatch.setattr(main_mod.os, "write", lambda fd, data: writes.append((fd, data)))

    main_mod._suppress_mouse_residue_early()

    assert writes == []


def test_cron_reader_accepts_utf8_bom_without_mutation(tmp_path):
    home = tmp_path / "profile"
    jobs_path = home / "cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True)
    original = b'\xef\xbb\xbf{"jobs": [{"id": "bom-job", "enabled": true}]}\n'
    jobs_path.write_bytes(original)

    jobs, status, error = health_mod._read_cron_jobs_read_only(home)

    assert status == "current"
    assert error is None
    assert jobs == [{"id": "bom-job", "enabled": True}]
    assert jobs_path.read_bytes() == original


def test_health_cli_e2e_json_exit_zero_with_temp_home(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home)

    proc = _run_health_cli(home, "--json")

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "healthy"
    assert payload["exit_code"] == 0
    assert payload["hermes_home"] == str(home)


@pytest.mark.parametrize("route_key", ["default", "model", "name"])
@pytest.mark.parametrize("nested_key", ["model", "default"])
@pytest.mark.parametrize("outer_provider", [None, "auto", "explicit-provider"])
def test_health_cli_nested_route_displays_only_identity(
    tmp_path, route_key, nested_key, outer_provider
):
    sentinel = "SECRET_HEALTH_NESTED_ROUTE"
    nested = {
        nested_key: "nested-model",
        "provider": "nested-provider",
        "api_key": sentinel,
        "base_url": f"https://user:{sentinel}@example.invalid/v1",
        "headers": {"Authorization": sentinel},
    }
    model = {route_key: nested}
    if outer_provider is not None:
        model["provider"] = outer_provider
    _assert_health_route_process(
        tmp_path, {"model": model}, sentinel,
        outer_provider if outer_provider == "explicit-provider" else "nested-provider",
        "nested-model",
    )


@pytest.mark.parametrize("malformed", [{"api_key": "SECRET_HEALTH_BAD_TYPE"},
                                       ["SECRET_HEALTH_BAD_TYPE"]])
@pytest.mark.parametrize("location", ["model", "default", "name", "provider",
                                      "nested_model", "nested_provider", "root_provider"])
@pytest.mark.parametrize("managed_route", [False, True])
def test_health_cli_route_never_stringifies_containers(tmp_path, malformed, location, managed_route):
    config: dict = {"model": {"default": "safe-model"}}
    expected_model, expected_provider = "safe-model", "auto"
    if location == "model":
        config["model"] = malformed
        expected_model = "(not set)"
    elif location in ("default", "name"):
        config["model"] = {location: malformed}
        expected_model = "(not set)"
    elif location == "provider":
        config["model"]["provider"] = malformed
    elif location == "root_provider":
        config["provider"] = malformed
    else:
        config["model"]["default"] = {
            "model": malformed if location == "nested_model" else "safe-model",
            "provider": malformed if location == "nested_provider" else "safe-provider",
        }
        if location == "nested_model":
            expected_model, expected_provider = "(not set)", "safe-provider"
    _assert_health_route_process(
        tmp_path, {} if managed_route else config, "SECRET_HEALTH_BAD_TYPE",
        expected_provider, expected_model, managed=config if managed_route else None,
    )


@pytest.mark.parametrize("config,provider,model", [
    ({"model": "legacy-model", "provider": "root-provider"}, "root-provider", "legacy-model"),
    ({"model": {"model": "alias-model", "name": "other"}}, "auto", "alias-model"),
    ({"model": {"default": "first", "model": "second", "name": "third"}}, "auto", "first"),
    ({"model": {"default": {"model": "first", "default": "second", "provider": "nested"}},
      "provider": "root"}, "nested", "first"),
])
def test_health_cli_raw_route_precedence(tmp_path, config, provider, model):
    _assert_health_route_process(tmp_path, config, "SECRET_HEALTH_UNUSED", provider, model)


@pytest.mark.parametrize("managed,provider,model", [
    ({"model": {"provider": "managed-provider", "default": "managed-model"}},
     "managed-provider", "managed-model"),
    ({"model": {"provider": "managed-provider", "name": "managed-alias"}},
     "managed-provider", "managed-alias"),
    ({"model": {"provider": "managed-provider", "model": "managed-legacy-alias"}},
     "managed-provider", "managed-legacy-alias"),
    ({"model": {"default": {"provider": "nested-provider", "model": "nested-model"}}},
     "nested-provider", "nested-model"),
    ({"provider": "managed-root", "model": "managed-scalar"},
     "managed-root", "managed-scalar"),
])
def test_health_cli_reports_effective_managed_route_read_only(tmp_path, managed, provider, model):
    _assert_health_route_process(
        tmp_path, {"model": {"provider": "user-provider", "default": "user-model"}},
        "SECRET_HEALTH_UNUSED", provider, model, managed=managed,
    )


@pytest.mark.parametrize("managed_route", [False, True])
@pytest.mark.parametrize("reference", ["${HEALTH4_ROUTE_MODEL}", "${env:HEALTH4_ROUTE_MODEL}"])
def test_health_cli_preserves_env_references_without_disclosing_values(tmp_path, managed_route, reference):
    user = {"model": {"provider": "user-provider", "default": reference}}
    managed = {"model": {"provider": "managed-provider", "name": reference}} if managed_route else None
    _assert_health_route_process(
        tmp_path, user, "SYNTHETIC_PRIVATE_ENV_VALUE",
        "managed-provider" if managed_route else "user-provider", reference,
        managed=managed,
        route_env={"HEALTH4_ROUTE_MODEL": "SYNTHETIC_PRIVATE_ENV_VALUE"},
    )


@pytest.mark.parametrize("override", [[], 0, False, "", None])
@pytest.mark.parametrize("root_key,root_value,provider", [
    ("provider", "managed-provider", "managed-provider"),
    ("base_url", "https://example.invalid", "user-provider"),
    ("api_base", "https://example.invalid", "user-provider"),
    ("context_length", 8192, "user-provider"),
])
def test_health_managed_root_aliases_use_canonical_normalization(
    tmp_path, override, root_key, root_value, provider,
):
    _assert_health_route_process(
        tmp_path, {"model": {"provider": "user-provider", "default": "user-model"}},
        "SECRET_HEALTH_UNUSED", provider, "user-model",
        managed={"model": override, root_key: root_value},
    )


@pytest.mark.parametrize("override", [False, 0, [], ["not-a-model-scalar"]])
def test_health_cli_managed_nonmapping_override_clears_user_route(tmp_path, override):
    _assert_health_route_process(
        tmp_path, {"model": {"provider": "user-provider", "default": "user-model"}},
        "SECRET_HEALTH_UNUSED", "auto", "(not set)", managed={"model": override},
    )


def test_health_cli_managed_only_route_with_unrelated_user_config(tmp_path):
    _assert_health_route_process(
        tmp_path, {"display": {"interface": "cli"}}, "SECRET_HEALTH_UNUSED",
        "managed-provider", "managed-model",
        managed={"model": {"provider": "managed-provider", "default": "managed-model"}},
    )


def test_health_cli_managed_null_default_clears_user_route(tmp_path):
    _assert_health_route_process(
        tmp_path, {"model": {"provider": "user-provider", "default": "user-model"}},
        "SECRET_HEALTH_UNUSED", "user-provider", "(not set)",
        managed={"model": {"default": None}},
    )


def test_health_cli_managed_provider_replaces_legacy_user_scalar_model(tmp_path):
    _assert_health_route_process(
        tmp_path, {"model": "legacy-user-model"},
        "SECRET_HEALTH_UNUSED", "managed-provider", "(not set)",
        managed={"model": {"provider": "managed-provider"}},
    )


def test_health_cli_managed_null_does_not_erase_nested_user_model(tmp_path):
    _assert_health_route_process(
        tmp_path, {"model": {"default": {"provider": "nested-user", "model": "user-model"}}},
        "SECRET_HEALTH_UNUSED", "nested-user", "user-model",
        managed={"model": {"default": None}},
    )


def test_health_cli_managed_empty_scalar_clears_user_default(tmp_path):
    _assert_health_route_process(
        tmp_path, {"model": {"provider": "user-provider", "default": "user-model"}},
        "SECRET_HEALTH_UNUSED", "user-provider", "(not set)",
        managed={"model": ""},
    )


def test_health_cli_managed_only_route_when_user_config_missing(tmp_path):
    user_home = tmp_path / "user"
    home = user_home / ".hermes"
    _write_profile(home)
    (home / "config.yaml").unlink()
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(
        'model:\n  provider: managed-provider\n  name: managed-model\n', encoding="utf-8",
    )
    before = {str(p): p.read_bytes() for root in (user_home, managed_dir)
              for p in root.rglob("*") if p.is_file()}
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(user_home),
           "HERMES_HOME": str(home), "HERMES_MANAGED_DIR": str(managed_dir),
           "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "hermes_cli.main", "health", "--json"],
        cwd=os.getcwd(), env=env, capture_output=True, text=True, timeout=30,
    )
    rows = {row["id"]: row for row in json.loads(proc.stdout)["checks"]}
    assert proc.returncode == 1, proc.stderr  # Missing user file still warrants a warning.
    assert rows["profile_config"]["status"] == "warning"
    assert rows["provider_routing"]["status"] == "healthy"
    assert "configured route managed-provider/managed-model;" in rows["provider_routing"]["detail"]
    assert before == {str(p): p.read_bytes() for root in (user_home, managed_dir)
                      for p in root.rglob("*") if p.is_file()}
    assert not (home / "backups").exists()


def _assert_health_route_process(tmp_path, config, sentinel, provider, model,
                                 *, managed=None, route_env=None):
    user_home = tmp_path / "user"
    home = user_home / ".hermes"
    _write_profile(home)
    # JSON is valid YAML and preserves deliberately malformed container types.
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    managed_dir = tmp_path / "managed"
    if managed is not None:
        managed_dir.mkdir()
        (managed_dir / "config.yaml").write_text(json.dumps(managed), encoding="utf-8")

    def snapshot():
        return {
            path.relative_to(user_home): (
                path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns
            )
            for path in user_home.rglob("*")
        }

    before = snapshot()
    script = """
import runpy, sys
sys.argv = ['hermes', 'health', *sys.argv[1:]]
try:
    runpy.run_module('hermes_cli.main', run_name='__main__')
finally:
    assert 'hermes_cli.config' not in sys.modules
    assert 'providers' not in sys.modules
"""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(user_home),
           "HERMES_HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1"}
    if managed is not None:
        env["HERMES_MANAGED_DIR"] = str(managed_dir)
    env.update(route_env or {})
    for args in (["--json"], []):
        proc = subprocess.run(
            [sys.executable, "-B", "-c", script, *args], cwd=os.getcwd(), env=env,
            capture_output=True, text=True, timeout=30,
        )
        assert sentinel not in proc.stdout + proc.stderr
        assert proc.returncode == (1 if model == "(not set)" else 0), proc.stderr
        assert not proc.stderr
        assert snapshot() == before
        expected = f"provider={provider} model={model}"
        if args:
            rows = {row["id"]: row for row in json.loads(proc.stdout)["checks"]}
            assert expected in rows["profile_config"]["detail"]
            if model != "(not set)":
                assert f"configured route {provider}/{model};" in rows["provider_routing"]["detail"]
        else:
            detail = f"profile=default {expected}"
            assert (detail[:55] + "..." if len(detail) > 58 else detail) in proc.stdout


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


def test_health_bootstrap_does_not_mutate_startup_state(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home)
    hermes_tmp = Path(os.environ["TMPDIR"]).resolve()
    marker = tmp_path / "startup-mutation"
    script = f"""
import sys
from pathlib import Path
sys.argv = ["hermes", "health", "--json"]
from hermes_cli import _early_recovery as recovery
import hermes_cli.venv_sync as venv_sync
import hermes_constants
marker = Path({str(marker)!r})
def forbidden(*_args, **_kwargs):
    marker.write_text("called", encoding="utf-8")
    raise AssertionError("health attempted startup mutation")
recovery._claim_recovery_lock = forbidden
recovery.restore_interrupted_pull = forbidden
venv_sync.prepare_launch = forbidden
hermes_constants.export_scratch_tmp_env = forbidden
import hermes_cli.main
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["TMPDIR"] = str(hermes_tmp)
    env["HERMES_SCRATCH_DIR"] = str(hermes_tmp)
    env["PYTHONPATH"] = os.getcwd()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "healthy"
    assert not marker.exists()
    assert not (home / "cache" / "scratch").exists()


def test_health_cli_e2e_global_value_options_do_not_enable_file_logging(tmp_path):
    invocations = (
        ("--provider", "auto", "health", "--json"),
        ("--model", "test-model", "health", "--json"),
        ("--toolsets", "all", "health", "--json"),
        ("--reasoning", "high", "health", "--json"),
        ("--in", str(tmp_path), "health", "--json"),
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


def _committed_health_generation(*, broken=False):
    """A real venv selected by PM facts, with the test interpreter's installed packages."""
    from pm.environments import install_state_dir, runtime_facts_path, site_packages

    root = Path(__file__).resolve().parents[2]
    state = install_state_dir(root)
    selected = state / "environments" / "current" / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(selected)],
                   check=True, capture_output=True, timeout=60)
    site = site_packages(selected)
    # The selected generation carries a site path; no installs or edits to the live venv.
    (site / "test-dependencies.pth").write_text(sysconfig.get_paths()["purelib"] + "\n", encoding="utf-8")
    if not broken:
        (site / "yaml.py").write_text("raise ImportError('PyYAML is not a core dependency')\n", encoding="utf-8")
    if broken:
        package = site / "ruamel"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "yaml.py").write_text("raise ImportError('damaged selected parser')\n", encoding="utf-8")
    facts = runtime_facts_path(root)
    facts.write_text(json.dumps({"schema": 1, "packages": {"venv": {"environment": str(selected)}}}),
                     encoding="utf-8")
    return state, facts


def _health_from_pm_interpreter(home, *, preload_parser=False, poison_parser=False):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "HERMES_HOME": str(home), "HOME": str(home.parent),
           "HERMES_RUNTIME_DIR": str(Path(sys.base_prefix).resolve().parent),
           "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("__HERMES_ACTIVATED", None)
    preload = (
        f"import sys; sys.path.insert(0, {sysconfig.get_paths()['purelib']!r}); import ruamel.yaml; "
        if preload_parser else ""
    )
    poison = (
        "import sys, types; "
        "sys.modules['ruamel'] = types.ModuleType('ruamel'); "
        "sys.modules['ruamel'].__path__ = []; "
        "sys.modules['ruamel.yaml'] = types.ModuleType('ruamel.yaml'); "
        if poison_parser else ""
    )
    script = preload + poison + "import sys; sys.argv = ['hermes', 'health', '--json']; import hermes_cli.main"
    return subprocess.run([getattr(sys, "_base_executable", sys.executable), "-c", script],
                          cwd=root, env=env, capture_output=True, text=True, timeout=40)


def _file_bytes_under(*roots):
    return {str(path): path.read_bytes() for root in roots for path in root.rglob("*") if path.is_file()}


def test_health_does_not_execute_selected_generation_pth_hooks(tmp_path, monkeypatch):
    from pm.environments import committed_venv, site_packages

    home = tmp_path / "user" / ".hermes"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _committed_health_generation()
    selected = committed_venv(Path(__file__).resolve().parents[2])
    assert selected is not None
    marker = home / "pth-hook-ran"
    (site_packages(selected) / "side-effect.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    before = _file_bytes_under(home)
    result = _health_from_pm_interpreter(home)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "healthy"
    assert not marker.exists()
    assert _file_bytes_under(home) == before


def test_health_refuses_cross_version_generation_before_loading_site(tmp_path, monkeypatch):
    from pm.environments import committed_venv, site_packages

    home = tmp_path / "user" / ".hermes"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _committed_health_generation()
    root = Path(__file__).resolve().parents[2]
    selected = committed_venv(root)
    assert selected is not None
    original_site = site_packages(selected)
    # A committed generation declares its own interpreter version, not the
    # launcher's. It must be rejected before executing any of its site hooks.
    version = f"{sys.version_info.major}.{sys.version_info.minor + 1}.0"
    cfg = selected / "pyvenv.cfg"
    lines = cfg.read_text(encoding="utf-8").splitlines()
    cfg.write_text("\n".join(
        f"version = {version}" if line.partition("=")[0].strip() == "version" else line
        for line in lines
    ) + "\n", encoding="utf-8")
    selected_site = site_packages(selected)
    if selected_site != original_site:
        selected_site.parent.mkdir(parents=True, exist_ok=True)
        original_site.rename(selected_site)
    marker = home / "site-hook-ran"
    (selected_site / "abi-check.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('unsafe bind')\n",
        encoding="utf-8",
    )
    before = _file_bytes_under(home)
    result = _health_from_pm_interpreter(home)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["checks"][0]["id"] == "runtime_dependencies"
    assert _file_bytes_under(home) == before
    assert not marker.exists()


def test_health_selected_pm_generation_with_pending_publication_is_read_only(tmp_path, monkeypatch):
    home = tmp_path / "user" / ".hermes"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    state, facts = _committed_health_generation()
    config = home / "config.yaml"
    prior = config.read_bytes()
    after = b"model:\n  default: incomplete-new\n"
    config.write_bytes(after)
    journal = state / "publication.json"
    journal.write_text(json.dumps({
        "kind": "config", "config": str(config), "previous": base64.b64encode(prior).decode(),
        "config_after": hashlib.sha256(after).hexdigest(),
        "facts_before": hashlib.sha256(facts.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    before = _file_bytes_under(home)
    result = _health_from_pm_interpreter(home)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["checks"][0]["status"] == "healthy"
    assert _file_bytes_under(home) == before
    assert journal.is_file()
    assert not (state / ".install.lock").exists()
    assert not (state / ".recovery.lock").exists()


def test_health_probes_broken_selected_generation_not_launcher_cache(tmp_path, monkeypatch):
    home = tmp_path / "user" / ".hermes"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    state, _ = _committed_health_generation(broken=True)
    before = _file_bytes_under(home)
    result = _health_from_pm_interpreter(home, preload_parser=True)

    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["checks"][0]["id"] == "runtime_dependencies"
    assert "ruamel.yaml" in payload["checks"][0]["detail"]
    assert _file_bytes_under(home) == before
    assert not (state / ".install.lock").exists()


def test_health_good_selection_ignores_poisoned_launcher_import(tmp_path, monkeypatch):
    home = tmp_path / "user" / ".hermes"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _committed_health_generation()
    before = _file_bytes_under(home)
    result = _health_from_pm_interpreter(home, poison_parser=True)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "healthy"
    assert _file_bytes_under(home) == before


def test_health_rejects_invalid_recorded_pm_selection_without_recovery(tmp_path, monkeypatch):
    from pm.environments import runtime_facts_path

    home = tmp_path / "user" / ".hermes"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    facts = runtime_facts_path(Path(__file__).resolve().parents[2])
    facts.parent.mkdir(parents=True)
    facts.write_text(json.dumps({"packages": {"venv": {"environment": str(tmp_path / "missing")}}}),
                     encoding="utf-8")
    before = _file_bytes_under(home)
    result = _health_from_pm_interpreter(home)

    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["checks"][0]["id"] == "runtime_dependencies"
    assert _file_bytes_under(home) == before
    assert not (facts.parent / ".install.lock").exists()


def test_health_canonical_yaml_rejects_duplicate_config_keys(tmp_path):
    home = tmp_path / "profile"
    _write_profile(home)
    (home / "config.yaml").write_text("model: {default: first}\nmodel: {default: second}\n", encoding="utf-8")
    result = _run_health_cli(home, "--json")

    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert next(row for row in payload["checks"] if row["id"] == "profile_config")["status"] == "critical"


@pytest.mark.parametrize("profile_flag", [["--profile", "Coder"], ["--profile=Coder"]])
def test_broken_dependency_fallback_normalizes_explicit_profile(tmp_path, monkeypatch, capsys, profile_flag):
    from hermes_cli._early_recovery import _emit_health_dependency_failure

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    with pytest.raises(SystemExit) as error:
        _emit_health_dependency_failure([*profile_flag, "health", "--json"], ["ruamel.yaml"])
    assert error.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "coder"
    assert payload["hermes_home"] == str(home / "profiles" / "coder")
    assert not home.exists()


def test_health_broken_dependency_custom_home_uses_custom_profile(tmp_path):
    home = tmp_path / "user" / "unregistered-hermes"
    _write_profile(home)
    shadow = tmp_path / "shadow"
    (shadow / "ruamel").mkdir(parents=True)
    (shadow / "ruamel" / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "ruamel" / "yaml.py").write_text("raise ImportError('damaged')\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "HOME": str(home.parent), "HERMES_HOME": str(home),
           "PYTHONPATH": f"{root}{os.pathsep}{shadow}"}
    env.pop("__HERMES_ACTIVATED", None)
    script = f"import sys; sys.path.insert(0, {str(shadow)!r}); sys.argv = ['hermes', 'health', '--json']; import hermes_cli.main"
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["profile"] == "custom"
    assert payload["hermes_home"] == str(home)
