"""Contract C3: after the commit point nothing fails `hermes update`.

Post-commit step failures are ⚠ + a receipt follow-up + an armed obligation; the receipt
is durable while the run is open; receipts resolve to the root home.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from hermes_cli import update_receipt


def _latest(home):
    return json.loads((home / "logs/update_receipts/latest.json").read_text())


def test_open_receipt_is_durably_running_and_finalizes_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    update_receipt.begin_update_receipt()
    running = _latest(tmp_path)
    assert running["outcome"] == "running" and running["pid"] == os.getpid()
    run_files = list((tmp_path / "logs/update_receipts").glob(f"update_*_{running['update_id']}.json"))
    assert len(run_files) == 1
    update_receipt.record_stage("apply", "success")
    assert [s["name"] for s in _latest(tmp_path)["stages"]] == ["apply"]
    update_receipt.record_followup("build", "web UI build: npm exited 1")
    update_receipt.finalize_update_receipt("success")
    final = _latest(tmp_path)
    assert final["outcome"] == "success"
    assert [(f["step"], f["reason"]) for f in final["followups"]] == [("build", "web UI build: npm exited 1")]
    # The terminal record replaced the running one in the SAME archive file (no duplicate per run).
    assert list((tmp_path / "logs/update_receipts").glob(f"update_*_{final['update_id']}.json")) == run_files


@pytest.mark.parametrize("finish", [lambda: update_receipt.finalize_update_receipt("success"),
                                    lambda: update_receipt.finalize_pending_update_receipt(1, "local changes parked")])
def test_parked_local_changes_never_finalize_as_success(tmp_path, monkeypatch, finish):
    # A follow-up is retried by the next launch; a stash whose restore conflicted is not, so the
    # committed run must stay partial (#122557) on both the verify path and the boundary net.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    update_receipt.begin_update_receipt()
    update_receipt.record_user_action("local_changes", "⚠ hermes update stashed 1 local modification(s)\n  Stash ref: abc")
    finish()
    final = _latest(tmp_path)
    assert final["outcome"] == "partial"
    assert final["user_action"] == {"step": "local_changes",
                                    "reason": "⚠ hermes update stashed 1 local modification(s) Stash ref: abc"}


def test_dead_running_record_is_reported_interrupted_by_the_next_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    dead = subprocess.Popen(["true"])
    dead.wait()
    update_receipt.begin_update_receipt()
    update_receipt.record_stage("deps", "success")
    killed = update_receipt._current.get().data
    killed_id = killed["update_id"]
    # Simulate the kill: the record on disk names a process that is gone.
    for path in (tmp_path / "logs/update_receipts").glob("*.json"):
        data = json.loads(path.read_text())
        data.update(pid=dead.pid, writer_pid=dead.pid, pid_create_time=None)
        path.write_text(json.dumps(data))
    update_receipt._current.set(None)

    update_receipt.begin_update_receipt()
    out = capsys.readouterr().out
    assert "interrupted" in out and "last stage: deps" in out
    (archived,) = (tmp_path / "logs/update_receipts").glob(f"update_*_{killed_id}.json")
    assert json.loads(archived.read_text())["outcome"] == "interrupted"
    assert _latest(tmp_path)["update_id"] != killed_id
    update_receipt.finalize_update_receipt("success")


def test_live_running_record_is_not_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    update_receipt.begin_update_receipt()
    own_id = update_receipt._current.get().data["update_id"]
    for path in (tmp_path / "logs/update_receipts").glob("*.json"):
        data = json.loads(path.read_text())
        data.update(pid=os.getppid(), writer_pid=os.getppid(), pid_create_time=None)  # alive
        path.write_text(json.dumps(data))
    assert update_receipt.reconcile_interrupted_runs() == []
    (archived,) = (tmp_path / "logs/update_receipts").glob(f"update_*_{own_id}.json")
    assert json.loads(archived.read_text())["outcome"] == "running"
    update_receipt.finalize_update_receipt("success")


def test_profile_process_writes_receipts_to_the_root_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles/work"
    profile.mkdir(parents=True)
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: root)
    monkeypatch.setattr("hermes_constants._default_hermes_root_memo", None)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert update_receipt._receipt_dir() == root / "logs/update_receipts"


def test_failed_web_build_does_not_skip_the_tui(tmp_path, monkeypatch, capsys):
    from hermes_cli import source_build

    built = []
    monkeypatch.setattr("hermes_cli.main_install_repair._install_configured_features_missing_deps", lambda root: None)
    monkeypatch.setattr("hermes_cli.update_stage.publish_stage", lambda text: None)
    monkeypatch.setattr("hermes_cli.memory_provider_migration.migrate_all_homes", lambda: None)
    monkeypatch.setattr(source_build, "source_frontends", lambda root: ("web", "ui-tui"))
    monkeypatch.setattr(source_build, "source_build_env", lambda **kw: {"PATH": ""})
    monkeypatch.setattr(source_build, "prepare_source_dependencies", lambda *a, **kw: None)
    monkeypatch.setattr(source_build, "source_product_current", lambda *a: False)
    monkeypatch.setattr(source_build, "build_source_tui", lambda root, env: built.append("tui"))

    def broken_web(root, env):
        raise subprocess.CalledProcessError(1, ["npm", "run", "build"])

    monkeypatch.setattr(source_build, "build_source_web", broken_web)
    with pytest.raises(source_build.ProductBuildError) as failure:
        source_build.build_update_products(tmp_path, desktop=False)
    assert built == ["tui"]
    assert [name for name, _ in failure.value.failures] == ["web UI build"]
    assert "npm run build exited 1" in str(failure.value)


def test_failed_config_migration_is_owed_and_later_maintenance_runs(tmp_path, monkeypatch, capsys):
    from hermes_cli import update_cmd, update_cmd_maint as maint

    calls = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("_verify_and_restore_state_dbs_post_update", "_invalidate_live_plugin_catalog_caches",
                 "_print_bundled_skills_sync_report", "_sync_profiles_after_update"):
        monkeypatch.setattr(maint, name, lambda name=name: calls.append(name))
    monkeypatch.setattr("hermes_cli.gitlock.fetch_full_commit_graph", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.model_catalog.seed_cache_from_checkout", lambda root: False)

    def broken_migration(**kwargs):
        raise SystemExit(1)  # a migration helper that exits must not fail the committed update

    monkeypatch.setattr(update_cmd, "_check_and_apply_config_migration", broken_migration)
    monkeypatch.setattr(maint, "_print_verified_update_completion", lambda message: calls.append("verdict") or True)
    monkeypatch.setattr(maint, "_print_post_update_notices_and_self_heals", lambda: calls.append("notices"))
    update_receipt.begin_update_receipt()
    owed: list = []
    assert maint._run_post_update_maintenance(
        assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
        had_desktop_app_before_update=False, pre_update_version=None, followups=owed) is True
    assert [step for step, _ in owed] == ["config_migration"]
    assert calls[-2:] == ["verdict", "notices"]
    assert "⚠ Update follow-up 'config_migration'" in capsys.readouterr().out
    update_receipt.finalize_update_receipt("success")
    assert [f["step"] for f in _latest(tmp_path)["followups"]] == ["config_migration"]


def test_owed_restart_names_its_gateways_so_a_dead_fleet_cannot_discharge_it(tmp_path, monkeypatch):
    # A gateway that died at boot leaves no live row. An inventory-less obligation would be settled
    # by the gateway-less discharge, silencing the warning that replaces exit 1 under C3.
    from hermes_cli import update_cmd_fleet as fleet
    from hermes_cli import update_cmd_fleet_verify as fleet_verify
    from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    fleet._write_fleet_restart_pending_marker(expected_sha="a" * 40)
    plan = UpdatePlan(runtimes=[RuntimeRecord(kind="gateway", profile="default", pid=4242),
                                RuntimeRecord(kind="dashboard", profile="default", pid=4343)])
    fleet_verify._record_owed_gateway_inventory(plan)

    inventory = json.loads(fleet._obligation_fields()["inventory"])
    assert [(r["kind"], r["profile"]) for r in inventory["runtimes"]] == [("gateway", "default")]
    # HEAD still holds the pulled code, so only the fleet evidence can settle it.
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "a" * 40)
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", lambda: [])
    assert fleet._marker_only_restart_obsolete() is False
    assert fleet._fleet_restart_obligation_armed()


def test_owed_restart_rearms_a_settled_obligation_and_a_later_run_keeps_owing_it(tmp_path, monkeypatch):
    # A pre-restart probe can settle an inventory-less record; the owed restart must re-arm it, and
    # a later verify that finds nothing to restart must keep owing the named gateway.
    from hermes_cli import update_cmd_fleet as fleet
    from hermes_cli import update_cmd_fleet_verify as fleet_verify
    from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "b" * 40)
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", lambda: [])
    assert not fleet._fleet_restart_obligation_armed()
    fleet_verify._record_owed_gateway_inventory(UpdatePlan(runtimes=[RuntimeRecord(kind="gateway", profile="default")]))
    assert fleet._obligation_fields()["expected_sha"] == "b" * 40
    # A later verify with nothing live to restart still owes the named gateway (and keeps it armed).
    assert fleet_verify._named_gateways_still_owed()
    assert fleet._fleet_restart_obligation_armed()
