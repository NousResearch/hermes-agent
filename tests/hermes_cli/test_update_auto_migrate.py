"""Legacy migration through the real dispatcher and systemd transaction backend."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto as auto, update_auto_migrate as migration
from hermes_cli import update_auto_schedule as scheduler, update_auto_schedule_common as common
from hermes_cli import update_auto_state as state
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_channel import install_id, set_install_channel
from tests.hermes_cli.test_update_auto_schedule_systemd import FakeSystemd

pytestmark = pytest.mark.platforms("linux")


def _args(*parts):
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=lambda args: pytest.fail("manual updater ran"))
    return parser.parse_args(["update", "auto", *parts])


@pytest.fixture
def environment(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    for directory in (home, work):
        (directory / "config.yaml").write_text("# retain configuration\nmodel: fixture\n")
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    context = state.AutoUpdateContext(root, home, home / "logs" / "update_receipts")
    monkeypatch.setattr(state.AutoUpdateContext, "current", classmethod(lambda cls: context))
    monkeypatch.setattr(auto, "require_source_install", lambda c: None)
    monkeypatch.setattr(auto, "run_update", lambda *a: pytest.fail("updater must not run during migration"))
    info = scheduler.paths(auto._spec(context, "04:00", ["21:00"]))
    manager = FakeSystemd(info)
    monkeypatch.setattr(common, "locate_command", lambda name: SimpleNamespace(command=("/fake/" + name,)))
    monkeypatch.setattr(common.subprocess, "run", manager.run)
    return context, work, manager


def _legacy(environment, home, time="04:00", plans=("21:00",)):
    context, _, manager = environment
    identity = "v1-" + hashlib.sha256(f"{context.install}\0{home}".encode()).hexdigest()[:24]
    spec = scheduler.SchedulerSpec(identity, ["/fake/hermes", "update", "auto", "run-scheduled"], home, time, plans)
    info = scheduler.paths(spec)
    manager.paths.update({info[key].name: info[key] for key in ("path", "service_path")})
    scheduler.enable(spec)
    status = {"schema": 2, "enabled": True, "mode": "scheduled", "schedule": time,
              "planSchedule": list(plans), "schedulerIdentity": identity, "installationRoot": str(context.install),
              "profileHome": str(home), "status": "not_configured", "schedulerType": info["backend"],
              "schedulerPath": str(info["path"]), "logPath": str(home / "logs" / "update.log")}
    path = home / "state" / "update-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status))
    return path, spec


def _run(*args):
    options = _args(*args)
    options.func(options)


def _snapshots(environment):
    context, work, manager = environment
    paths = [*manager.paths.values(), *(home / "state" / "update-status.json" for home in (context.home, work))]
    return {path: common.snapshot_file(path) for path in paths}


def test_two_profiles_migrate_into_one_installation_timer_and_shared_status(environment, monkeypatch):
    context, work, manager = environment
    first, _ = _legacy(environment, context.home)
    second, _ = _legacy(environment, work)
    _run("migrate")
    saved = state.read_status(context)
    assert saved["enabled"] and saved["status"] == "migrated"
    assert saved["schedule"] == "04:00" and saved["planSchedule"] == ["21:00"]
    assert manager.active == manager.enabled == {Path(saved["schedulerPath"]).name}
    for path in (first, second):
        legacy = json.loads(path.read_text())
        assert legacy["enabled"] is False
        assert legacy["migratedTo"] == str(context.status_path)
    named = state.AutoUpdateContext(context.install, work, context.receipt_directory)
    assert state.read_status(named) == saved
    assert migration.legacy_schedules(named) == []
    assert all(row["contentBase64"] for row in saved["migration"]["previousArtifacts"])
    monkeypatch.setenv("HERMES_HOME", str(work))
    _run("enable", "--time", "05:00")
    assert state.read_status(named)["schedule"] == "05:00"
    assert manager.active == manager.enabled == {Path(saved["schedulerPath"]).name}


def test_conflicting_times_need_an_explicit_schedule_and_keep_every_old_artifact(environment):
    context, work, manager = environment
    _legacy(environment, context.home)
    _legacy(environment, work, "06:00")
    before = _snapshots(environment)
    calls = len(manager.calls)
    with pytest.raises(SystemExit):
        _run("migrate")
    assert _snapshots(environment) == before
    assert len(manager.calls) == calls
    assert not context.status_path.exists()
    _run("enable", "--time", "03:00")
    assert state.read_status(context)["schedule"] == "03:00"
    assert len(manager.active) == 1


@pytest.mark.parametrize("named_timer_enabled", [False, True])
def test_default_channel_of_an_existing_timer_is_not_implicitly_changed(environment, named_timer_enabled):
    context, work, _ = environment
    _legacy(environment, context.home)
    if named_timer_enabled:
        _legacy(environment, work)
    (work / "config.yaml").write_text(json.dumps({"update": {"installs": {
        install_id(context.install): {"path": str(context.install), "channel": "stable"}}}}))
    before = _snapshots(environment)
    with pytest.raises(SystemExit):
        _run("migrate")
    assert _snapshots(environment) == before
    set_install_channel("stable", context.install)
    _run("migrate")
    assert state.read_status(context)["enabled"]


def test_new_scheduler_failure_restores_old_files_modes_and_manager_state(environment):
    context, work, manager = environment
    _legacy(environment, context.home)
    _legacy(environment, work)
    before = _snapshots(environment)
    enabled, active = set(manager.enabled), set(manager.active)
    manager.fail_once = "enable"
    with pytest.raises(SystemExit):
        _run("migrate")
    assert _snapshots(environment) == before
    assert (manager.enabled, manager.active) == (enabled, active)
    assert not context.status_path.exists()
    assert "# retain configuration" in (context.home / "config.yaml").read_text()


def test_late_status_failure_restores_migrated_profiles_and_all_schedulers(environment, monkeypatch):
    context, work, manager = environment
    _legacy(environment, context.home)
    _legacy(environment, work)
    before = _snapshots(environment)
    write = migration.write_status

    def fail_final(ctx, data):
        if data.get("status") == "migrated":
            raise OSError("injected terminal state failure")
        write(ctx, data)

    monkeypatch.setattr(migration, "write_status", fail_final)
    with pytest.raises(SystemExit):
        _run("migrate")
    assert _snapshots(environment) == before
    assert len(manager.active) == 2
    assert not context.status_path.exists()


def test_running_legacy_service_is_not_stopped_or_killed(environment):
    context, _, manager = environment
    _, spec = _legacy(environment, context.home)
    info = scheduler.paths(spec)
    manager.active.add(info["service_path"].name)
    before = _snapshots(environment)
    with pytest.raises(SystemExit):
        _run("migrate")
    assert info["service_path"].name in manager.active
    assert _snapshots(environment) == before
    assert not any(call[0] == "stop" for call in manager.calls)


def test_old_timer_firing_reports_migration_instead_of_updating_or_unloading_itself(environment, capsys):
    context, _, manager = environment
    _legacy(environment, context.home)
    calls = len(manager.calls)
    with pytest.raises(SystemExit):
        _run("run-scheduled")
    assert "hermes update auto migrate" in capsys.readouterr().err
    assert len(manager.calls) == calls
    assert not context.status_path.exists()


def test_disable_can_remove_conflicting_legacy_schedules_without_choosing_one(environment):
    context, work, manager = environment
    _legacy(environment, context.home)
    _legacy(environment, work, "06:00")
    _run("disable")
    assert not manager.active and not manager.enabled
    assert state.read_status(context)["enabled"] is False
    assert migration.legacy_schedules(context) == []


def test_status_lists_legacy_schedules_without_any_mutation(environment, capsys):
    context, _, manager = environment
    _legacy(environment, context.home)
    before = _snapshots(environment)
    calls = len(manager.calls)
    _run("status")
    saved = json.loads(capsys.readouterr().out)
    assert saved["migrationRequired"]
    assert saved["legacySchedules"][0]["home"] == str(context.home)
    assert _snapshots(environment) == before
    assert len(manager.calls) == calls
    assert not context.status_path.exists()


def test_interrupted_migration_leaves_recovery_evidence_and_blocks_new_mutations(environment, monkeypatch):
    context, _, _ = environment
    _legacy(environment, context.home)
    monkeypatch.setattr(scheduler, "enable", lambda spec: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _run("migrate")
    saved = state.read_status(context)
    assert saved["status"] == "migration_running"
    assert saved["migration"]["previousArtifacts"]
    with pytest.raises(SystemExit):
        _run("enable", "--time", "05:00")
    assert state.read_status(context) == saved


def test_missing_installation_state_cannot_take_over_an_existing_timer(environment):
    context, _, manager = environment
    spec = auto._spec(context, "04:00", [])
    scheduler.enable(spec)
    before = _snapshots(environment)
    with pytest.raises(SystemExit):
        _run("enable", "--time", "06:00")
    assert _snapshots(environment) == before
    assert len(manager.active) == 1


def test_legacy_data_root_alias_keeps_its_original_timer_identity(environment):
    context, _, manager = environment
    alias = context.home.parent / "home-alias"
    alias.symlink_to(context.home, target_is_directory=True)
    path, _ = _legacy(environment, alias)
    _run("migrate")
    assert state.read_status(context)["enabled"]
    assert json.loads(path.read_text())["migratedTo"] == str(context.status_path)
    assert len(manager.active) == 1


@pytest.mark.parametrize("orphan", ["disabled-status", "missing-status"])
def test_unclaimed_legacy_artifact_cannot_be_ignored_when_enabling(environment, orphan):
    context, _, manager = environment
    path, _ = _legacy(environment, context.home)
    if orphan == "missing-status":
        path.unlink()
    else:
        status = json.loads(path.read_text())
        status["enabled"] = False
        path.write_text(json.dumps(status))
    before = _snapshots(environment)
    with pytest.raises(SystemExit):
        _run("enable", "--time", "06:00")
    assert _snapshots(environment) == before
    assert len(manager.active) == 1


def test_legacy_record_cannot_claim_an_unrelated_data_root(environment):
    context, _, _ = environment
    path, _ = _legacy(environment, context.home)
    payload = json.loads(path.read_text())
    payload["profileHome"] = str(context.home.parent / "another-home")
    path.write_text(json.dumps(payload))
    before = _snapshots(environment)
    with pytest.raises(SystemExit):
        _run("migrate")
    assert _snapshots(environment) == before


def test_old_enable_lock_in_an_inactive_profile_prevents_a_second_timer(environment):
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.update_lock import marker_mutex

    context, work, manager = environment
    path = work / "state" / "update-auto-operation"
    path.parent.mkdir()
    with marker_mutex(path, wait=0), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run, "enable", "--time", "04:00")
        with pytest.raises(SystemExit):
            future.result(timeout=5)
    assert not manager.active and not manager.enabled
    assert not context.status_path.exists()


def test_two_profile_aliases_to_one_home_retire_one_legacy_timer(environment):
    context, _, manager = environment
    external = context.home.parent / "external-profile"
    external.mkdir()
    (external / "config.yaml").write_text("{}")
    first = context.home / "profiles" / "alias-a"
    second = context.home / "profiles" / "alias-z"
    first.symlink_to(external, target_is_directory=True)
    second.symlink_to(external, target_is_directory=True)
    _legacy(environment, second)
    assert len(migration.legacy_schedules(context)) == 1
    _run("migrate")
    assert state.read_status(context)["enabled"]
    assert len(manager.active) == 1


def test_stale_old_timer_cannot_run_after_its_status_was_migrated(environment, monkeypatch):
    context, _, _ = environment
    _legacy(environment, context.home)
    _run("migrate")
    monkeypatch.setattr(auto, "check_update", lambda *a: pytest.fail("old timer must not check or apply an update"))
    with pytest.raises(SystemExit):
        _run("run-scheduled")
    assert state.read_status(context)["enabled"]
