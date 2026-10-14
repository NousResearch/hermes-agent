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
        from hermes_cli.process_identity import _process_create_time

        parent_time = _process_create_time(os.getppid())
        data.update(pid=os.getppid(), writer_pid=os.getppid(),
                    pid_create_time=parent_time, writer_create_time=parent_time)
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
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", list)
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
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", list)
    assert not fleet._fleet_restart_obligation_armed()
    fleet_verify._record_owed_gateway_inventory(UpdatePlan(runtimes=[RuntimeRecord(kind="gateway", profile="default")]))
    assert fleet._obligation_fields()["expected_sha"] == "b" * 40
    # A later verify with nothing live to restart still owes the named gateway (and keeps it armed).
    assert fleet_verify._named_gateways_still_owed()
    assert fleet._fleet_restart_obligation_armed()


def test_followup_line_is_the_one_both_desktop_handoffs_parse(tmp_path, monkeypatch, capsys):
    """The hand-offs read owed follow-ups from this printed line (review regression 1): pin the
    producer against the exact parsers in posix.sh (sed) and windows.ps1 (.NET regex)."""
    import re
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    update_receipt.record_followup("gateway_restart", "x" * 900)
    update_receipt.record_followup("windows_resume", "Windows gateway recovery failed: 'quoted' reason")
    out = capsys.readouterr().out
    scripts = Path(__file__).resolve().parents[2] / "scripts/desktop-update"
    posix = (scripts / "posix.sh").read_text(encoding="utf-8")
    function = re.search(r"^owed_followup_steps\(\) \{.*?^\}", posix, re.DOTALL | re.MULTILINE).group(0)
    steps = subprocess.run(["bash", "-c", f"{function}\nowed_followup_steps"], env={**os.environ, "OUT": out},
                           capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()
    assert steps == "gateway_restart windows_resume"
    pattern = re.search(r"\[regex\]::Matches\(\(\$res\.Output -join \"`n\"\), \"([^\"]+)\"\)",
                        (scripts / "windows.ps1").read_text(encoding="utf-8")).group(1)
    assert [m.group(1) for m in re.finditer(pattern, out)] == ["gateway_restart", "windows_resume"]


def test_interrupt_after_the_run_closed_as_success_never_tells_the_gateway_1(tmp_path, monkeypatch):
    """Review regression 3, the ``cmd_update`` boundary: once this run's receipt says ``success``,
    an interrupt escaping afterwards must not write 1 to the gateway /update status."""
    from types import SimpleNamespace
    from hermes_cli import main, update_cmd, update_owning_install

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)
    monkeypatch.setattr(update_owning_install, "retarget_to_owning_install", lambda root: None)  # not the target

    def committed_then_interrupted(args, gateway_mode):
        update_receipt.begin_update_receipt()
        update_receipt.finalize_update_receipt("success")
        raise KeyboardInterrupt()

    monkeypatch.setattr(update_cmd, "_cmd_update_impl", committed_then_interrupted)
    with pytest.raises(KeyboardInterrupt):
        main.cmd_update(SimpleNamespace(gateway=True))
    assert _latest(tmp_path)["outcome"] == "success"
    assert (tmp_path / ".update_exit_code").read_text(encoding="utf-8").strip() == "0"


def test_a_failure_before_the_run_closed_still_tells_the_gateway_1(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import main, update_cmd, update_owning_install

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)
    monkeypatch.setattr(update_owning_install, "retarget_to_owning_install", lambda root: None)  # not the target

    def interrupted_while_open(args, gateway_mode):
        update_receipt.begin_update_receipt()
        raise KeyboardInterrupt()

    monkeypatch.setattr(update_cmd, "_cmd_update_impl", interrupted_while_open)
    with pytest.raises(KeyboardInterrupt):
        main.cmd_update(SimpleNamespace(gateway=True))
    assert (tmp_path / ".update_exit_code").read_text(encoding="utf-8").strip() == "1"


def test_completion_child_interrupted_after_verification_answers_its_success(tmp_path, monkeypatch):
    """Review regression 3, the selected completion child: verification finalized ``success`` and
    then an interrupt landed. Exit, result and gateway status follow the receipt (0), not 130/1."""
    from hermes_cli import update_completion

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    update_receipt.begin_update_receipt()
    data = dict(update_receipt._current.get().data)
    update_receipt._current.set(None)

    def verified_then_interrupted(request):
        update_receipt.finalize_update_receipt("success")
        raise KeyboardInterrupt()

    monkeypatch.setattr(update_completion, "_complete_selected", verified_then_interrupted)
    request = {"receipt": data, "pm_receipt": None, "gateway_mode": True, "windows_resume": None}
    result_path = tmp_path / "result.json"
    assert update_completion._finish(request, result_path) == 0
    answer = json.loads(result_path.read_text(encoding="utf-8"))
    assert answer["exit_code"] == 0 and answer["receipt"]["outcome"] == "success"
    assert not (tmp_path / ".update_exit_code").exists() or \
        (tmp_path / ".update_exit_code").read_text(encoding="utf-8").strip() == "0"


def test_both_completion_children_run_in_utf8_mode(tmp_path, monkeypatch):
    """-I drops PYTHONIOENCODING, and both completion children print the ✓/⚠ follow-up protocol
    into a pipe: each argv carries -X utf8 (win-utf8 review; the bootstrap child got it first)."""
    import pm
    from pm import client, environments, receipt
    from hermes_cli import gitlock, update_completion, venv_sync

    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        raise OSError("stop after capturing the argv")

    monkeypatch.setattr(subprocess, "call", lambda command, **kw: calls.append(command) or 0)
    monkeypatch.setattr(subprocess, "Popen", popen)
    # The treeless-checkout conversion's git probe would be captured as a third spawn.
    for module, name in ((venv_sync, "refuse_foreign_owned_venv"), (venv_sync, "arm_completion"),
                         (venv_sync, "collect_superseded_generations"), (client, "ensure_tools_for_sync"),
                         (pm, "sync_venv"), (gitlock, "convert_treeless_checkout_first")):
        monkeypatch.setattr(module, name, lambda *a, **k: None)
    monkeypatch.setattr(receipt, "last_for_update", lambda *a, **k: None)
    monkeypatch.setattr(environments, "project_python", lambda root: "python")
    monkeypatch.setattr(environments, "activation_environment", lambda root: {})
    monkeypatch.setattr(update_completion, "_settle_after_commit", lambda *a: 0)
    request = {"source": str(tmp_path), "receipt": {"update_id": "u1"}, "bytecode_cache": str(tmp_path / "bc")}
    update_completion._prepare(request, tmp_path / "request.json", tmp_path / "result.json")
    monkeypatch.setattr(update_completion.sys, "platform", "win32")  # skip the Linux libatomic pre-install
    with pytest.raises(OSError):
        update_completion.run_completion({**request, "home": str(tmp_path)})
    assert len(calls) == 2
    for command in calls:
        assert command[command.index("-I"):].count("utf8") == 1 and "-X" in command, command
