"""ZIP completion uses the Git maintenance/fleet path after a real local swap."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from urllib.request import urlretrieve
import zipfile

import pytest

from hermes_cli import main, update_cmd, update_cmd_fleet as fleet, update_cmd_maint as maint
from hermes_cli import update_cmd_zip, update_receipt
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
import hermes_yaml


@pytest.fixture
def zip_update(tmp_path, monkeypatch):
    home = tmp_path / "home"
    active = home / ".hermes"
    sibling = active / "profiles/other"
    sibling.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(active))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    for profile in (active, sibling):
        (profile / "config.yaml").write_text(
            f"_config_version: {DEFAULT_CONFIG['_config_version'] - 1}\n"
            "model:\n  default: retained-model\n", encoding="utf-8")
    (active / ".env").write_text("EXAMPLE_TOKEN=retained\n", encoding="utf-8")
    jobs = active / "cron/jobs.json"
    jobs.parent.mkdir()
    original_jobs = {"jobs": [{"id": "keep-me", "prompt": "retained schedule"}]}
    jobs.write_text(json.dumps(original_jobs), encoding="utf-8")

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nversion="1.0"\n', encoding="utf-8")
    (root / "payload.txt").write_text("old", encoding="utf-8")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("hermes-agent-main/pyproject.toml", '[project]\nversion="2.0"\n')
        out.writestr("hermes-agent-main/payload.txt", "new")
    # Only redirect transport: extraction, staging, dirty recheck and swap run.
    monkeypatch.setattr("urllib.request.urlretrieve", lambda url, dst: urlretrieve(archive.as_uri(), dst))
    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    events = []
    token = {"resume_needed": True, "profiles": {}, "unmapped": []}
    plan = UpdatePlan(install_method="git", expected_version="1.0", profiles=["default", "other"],
                      runtimes=[RuntimeRecord(kind="serve", profile="default", pid=99999999,
                                              supervisor="manual-serve")])
    monkeypatch.setattr("hermes_cli.update_inventory.collect_runtime_inventory", lambda: plan)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: token)
    monkeypatch.setattr("atexit.register", lambda *args: None)
    monkeypatch.setattr(main, "_desktop_packaged_executable", lambda root: None)
    monkeypatch.setattr(main, "_desktop_dist_exists", lambda root: False)
    monkeypatch.setattr(update_cmd, "_source_update_channel", lambda args: "main")
    monkeypatch.setattr(maint, "_sweep_bytecode_after_update", lambda branch: None)

    def prepare(selected, *, desktop):
        assert selected == root and desktop is False
        assert (root / "payload.txt").read_text() == "new"
        events.append("prepare")
        # The pre-update snapshot must reach the real cron-loss safety net.
        jobs.write_text('{"jobs": []}', encoding="utf-8")
    monkeypatch.setattr(maint, "_prepare_updated_checkout", prepare)
    # PM/builds and machine-level repair are independently covered. Keep real
    # config migration, profile env backfill, snapshot recovery and receipts.
    monkeypatch.setattr("hermes_cli.macos_tcc_anchor.ensure_tcc_anchor", lambda: None)
    monkeypatch.setattr(maint, "_print_post_update_notices_and_self_heals", lambda: None)
    monkeypatch.setattr(maint, "_print_bundled_skills_sync_report", lambda: None)
    monkeypatch.setattr("hermes_cli.profiles.seed_profile_skills", lambda *a, **kw: {})
    monkeypatch.setattr("plugins.memory.honcho.cli.sync_honcho_profiles_quiet", lambda: [])
    monkeypatch.setattr(update_cmd, "_reload_config_modules", lambda: None)
    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (True, None))
    monkeypatch.setattr(fleet, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(maint, "_refresh_dashboard_after_update", lambda **kwargs: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(fleet, "_collect_fleet_snapshot", lambda *args: [])
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)

    def resume(received):
        assert received is token
        assert token["resume_needed"], "completion must resume only once"
        token["resume_needed"] = False
        events.append("resume")
    monkeypatch.setattr(main, "_resume_windows_gateways_after_update", resume)
    monkeypatch.setattr(update_cmd, "_write_gateway_update_exit_code", lambda ok: events.append(("marker", ok)))

    def restart(received, gateway_mode):
        assert received is plan
        events.append("restart")
        return fleet._GatewayRestartOutcome(
            incomplete=False, phase_errors=[], pre_restart_gateway_pids=[],
            restarted_services=[], failed_or_stale_units=[], relaunched_profiles=[],
            externally_supervised_profiles=[], killed_pids=set())
    monkeypatch.setattr(update_cmd, "_restart_gateway_fleet_after_update", restart)
    real_finalize = update_receipt.finalize_update_receipt

    def finalize(*args, **kwargs):
        events.append("finalize")
        return real_finalize(*args, **kwargs)
    monkeypatch.setattr(update_receipt, "finalize_update_receipt", finalize)
    yield SimpleNamespace(root=root, active=active, sibling=sibling, jobs=jobs,
                          original_jobs=original_jobs, events=events, token=token, plan=plan)
    update_receipt._current.set(None)


@pytest.mark.parametrize("route", ["direct", "git-failure"])
@pytest.mark.parametrize("gateway_mode", [False, True])
def test_zip_command_migrates_profiles_recovers_snapshot_and_verifies_fleet(
    zip_update, monkeypatch, route, gateway_mode, capsys,
):
    state = zip_update
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (route == "direct", ["git"], False))
    monkeypatch.setattr(main, "_warn_orphaned_update_autostashes", lambda *args: None)

    def fail_fetch(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["git", "fetch"])
    monkeypatch.setattr(update_cmd, "_git_run", fail_fetch)
    # Choose the fallback branch without pretending this host is Windows.
    monkeypatch.setattr(update_cmd, "_should_zip_fallback_on_update_error", lambda exc: True)
    update_cmd._cmd_update_impl(SimpleNamespace(branch="main", yes=True), gateway_mode)

    for profile in (state.active, state.sibling):
        config = hermes_yaml.safe_load((profile / "config.yaml").read_text())
        assert config["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert config["model"]["default"] == "retained-model"
    assert (state.sibling / ".env").read_bytes() == (state.active / ".env").read_bytes()
    assert json.loads(state.jobs.read_text()) == state.original_jobs
    assert (state.root / "payload.txt").read_text() == "new"
    assert "v1.0 → v2.0" in capsys.readouterr().out
    assert state.events == ["prepare", *([("marker", True)] if gateway_mode else []),
                            "restart", "resume", "finalize"]
    receipt = json.loads((state.active / "logs/update_receipts/latest.json").read_text())
    assert receipt["outcome"] == "success"
    assert receipt["runtime_outcomes"][0]["outcome"] == "restarted"
    assert update_receipt._current.get() is None
    assert state.token["resume_needed"] is False
    assert not list(state.root.glob("*.hermes-update-*"))


@pytest.mark.parametrize("verdict", ["healthy", "unsafe-sqlite", "stale-fleet"])
def test_zip_helper_preserves_bool_contract_after_real_verification(zip_update, monkeypatch, verdict):
    state = zip_update
    args = SimpleNamespace(branch="main", yes=True, gateway=True)
    plan = update_cmd._begin_update_receipt_and_plan(args)
    snapshot = main._run_pre_update_backup(args)
    if verdict == "unsafe-sqlite":
        monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (
            False, SimpleNamespace(sqlite_version_string="unsafe test runtime")))
    elif verdict == "stale-fleet":
        monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [
            {"pid": plan.runtimes[0].pid, "profile": "default"}])
    result = update_cmd_zip._update_via_zip(
        args, pre_update_snapshot_id=snapshot, _pre_update_plan=plan,
        _windows_gateway_resume=state.token)
    assert result is (verdict == "healthy")
    assert state.events == ["prepare", ("marker", verdict != "unsafe-sqlite"),
                            "restart", "resume", "finalize"]
    receipt = json.loads((state.active / "logs/update_receipts/latest.json").read_text())
    assert receipt["outcome"] == ("success" if verdict == "healthy" else "partial")
    assert receipt["runtime_outcomes"][0]["outcome"] == (
        "unaccounted" if verdict == "stale-fleet" else "restarted")
    assert json.loads(state.jobs.read_text()) == state.original_jobs


@pytest.mark.parametrize("route", ["direct", "git-failure"])
@pytest.mark.parametrize("failure", ["swap", "preparation"])
def test_zip_failure_recovers_pause_without_completion_mutations(zip_update, monkeypatch, route, failure):
    import os
    import pm

    state = zip_update
    before = {profile: (profile / "config.yaml").read_bytes()
              for profile in (state.active, state.sibling)}
    old_project = (state.root / "pyproject.toml").read_bytes()
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (route == "direct", ["git"], False))
    monkeypatch.setattr(main, "_warn_orphaned_update_autostashes", lambda *args: None)
    monkeypatch.setattr(update_cmd, "_should_zip_fallback_on_update_error", lambda exc: True)

    def fail_fetch(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["git", "fetch"])
    monkeypatch.setattr(update_cmd, "_git_run", fail_fetch)
    installed = []
    if failure == "swap":
        rename = os.rename

        def fail_second_swap(src, dst):
            if str(src).endswith(".hermes-update-staging"):
                installed.append(dst)
                if len(installed) == 2:
                    raise OSError("locked replacement")
            return rename(src, dst)
        monkeypatch.setattr(os, "rename", fail_second_swap)
        expected = SystemExit
    else:
        def fail_preparation(*args, **kwargs):
            assert (state.root / "payload.txt").read_text() == "new"
            raise pm.InstallError("venv", "preparation stopped")
        monkeypatch.setattr(maint, "_prepare_updated_checkout", fail_preparation)
        expected = pm.InstallError
    with pytest.raises(expected) as raised:
        update_cmd._cmd_update_impl(SimpleNamespace(branch="main", yes=True), gateway_mode=True)
    if failure == "swap":
        assert raised.value.code == 1
        assert len(installed) == 2
        assert (state.root / "pyproject.toml").read_bytes() == old_project
        assert (state.root / "payload.txt").read_text() == "old"
    assert state.events == ["resume"]
    assert state.token["resume_needed"] is False
    assert {profile: (profile / "config.yaml").read_bytes() for profile in before} == before
    assert not (state.sibling / ".env").exists()
    assert json.loads(state.jobs.read_text()) == state.original_jobs
    assert not (state.active / "logs/update_receipts/latest.json").exists()
    assert not list(state.root.glob("*.hermes-update-*"))
