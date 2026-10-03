"""Only a successful terminal receipt may announce update/action completion."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from hermes_cli import update_completion, update_cmd, update_cmd_maint as maint


@pytest.fixture
def completion(tmp_path, monkeypatch):
    from hermes_cli import source_build, venv_sync, gitlock, model_catalog, macos_tcc_anchor
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_ACTION_ID", "d" * 32)
    for name in ("_verify_and_restore_state_dbs_post_update", "_invalidate_live_plugin_catalog_caches",
                 "_print_bundled_skills_sync_report", "_sync_profiles_after_update",
                 "_print_post_update_notices_and_self_heals"):
        monkeypatch.setattr(maint, name, lambda *a, **kw: None)
    monkeypatch.setattr(update_cmd, "_check_and_apply_config_migration", lambda **kw: None)
    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (True, SimpleNamespace()))
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda: "")
    monkeypatch.setattr(maint, "_checkout_version", lambda: None)
    monkeypatch.setattr(gitlock, "fetch_full_commit_graph", lambda *a, **kw: False)
    monkeypatch.setattr(model_catalog, "seed_cache_from_checkout", lambda *a: False)
    monkeypatch.setattr(macos_tcc_anchor, "ensure_tcc_anchor", lambda: None)
    monkeypatch.setattr(source_build, "build_update_products", lambda *a, **kw: None)
    monkeypatch.setattr(venv_sync, "publish_launchers", lambda *a: None)
    monkeypatch.setattr(venv_sync, "clear_completion", lambda *a: None)
    monkeypatch.setattr(update_cmd, "_sweep_bytecode_after_update", lambda *a: None)
    monkeypatch.setattr(update_cmd, "_fleet_restart_skip_reason", lambda plan: None)
    monkeypatch.setattr(update_cmd, "_restart_gateway_fleet_after_update", lambda *a: SimpleNamespace(incomplete=False))
    monkeypatch.setattr(update_cmd, "_resume_windows_gateways_and_merge_outcome", lambda *a: None)
    monkeypatch.setattr(update_cmd, "_resume_windows_gateways_after_update", lambda *a: None)
    request = {"source": str(tmp_path), "home": str(tmp_path), "branch": "main",
               "receipt": {"schema": 1, "update_id": "d" * 32, "started_at": "before",
                           "steps": [], "skips": [], "gateway_restart": {}, "fleet": []},
               "desktop": False, "assume_yes": True, "gateway_mode": False,
               "plan": None, "sibling_snapshots": {}, "windows_resume": None,
               "snapshot_id": None, "pre_update_version": None}
    return request, tmp_path / "result.json"


def test_failed_fleet_never_announces_completion(completion, monkeypatch, capsys):
    request, result = completion
    seen = []
    def verify(*a, **kw):
        seen.append(capsys.readouterr().out)
        raise SystemExit(1)
    monkeypatch.setattr(update_cmd, "_verify_fleet_after_update", verify)
    assert update_completion._finish(request, result) == 1
    output = "".join(seen) + capsys.readouterr().out
    assert "Update complete" not in output
    assert "=== hermes-update completed" not in output
    terminal = json.loads(result.read_text())
    assert terminal["exit_code"] == 1
    assert terminal["receipt"]["outcome"] == "failed"


def test_success_announced_only_after_correlated_terminal_result(completion, monkeypatch, capsys):
    request, result = completion
    before_result = []
    monkeypatch.setattr(update_cmd, "_verify_fleet_after_update", lambda *a, **kw: None)
    write = update_completion._write_json
    def publish(path, data):
        before_result.append(capsys.readouterr().out)
        assert data["receipt"]["update_id"] == request["receipt"]["update_id"]
        assert data["receipt"]["finished_at"] and data["receipt"]["outcome"] == "success"
        write(path, data)
    monkeypatch.setattr(update_completion, "_write_json", publish)
    assert update_completion._finish(request, result) == 0
    assert "Update complete" not in "".join(before_result)
    assert "=== hermes-update completed" not in "".join(before_result)
    output = capsys.readouterr().out
    assert "✓ Update complete!" in output
    assert output.count("=== hermes-update completed " + "d" * 32 + " ===") == 1


def test_gateway_status_not_success_before_fleet_verification(completion, monkeypatch):
    request, result = completion
    request["gateway_mode"] = True
    statuses = []
    before_verify = []
    monkeypatch.setattr(update_cmd, "_write_gateway_update_exit_code", statuses.append)
    def verify(*a, **kw):
        before_verify.extend(statuses)
        raise SystemExit(1)
    monkeypatch.setattr(update_cmd, "_verify_fleet_after_update", verify)
    assert update_completion._finish(request, result) == 1
    assert before_verify == []
    assert statuses == [False]


def test_explicit_restart_deferral_reports_code_only_completion(completion, monkeypatch, capsys):
    request, result = completion
    request["no_gateway_restart"] = True
    monkeypatch.setattr(update_cmd, "_restart_gateway_fleet_after_update",
                        lambda *a: pytest.fail("explicitly deferred restart attempted"))
    assert update_completion._finish(request, result) == 0
    output = capsys.readouterr().out
    assert "✓ Update complete!" not in output
    assert "✓ Code update complete; gateway restart deferred" in output
    receipt = json.loads(result.read_text())["receipt"]
    assert receipt["outcome"] == "success"
    assert any(skip["name"] == "gateway_restart" and "deferred" in skip["reason"]
               for skip in receipt["skips"])


def test_dashboard_receipt_identity_is_action_identity(monkeypatch):
    from hermes_cli import update_inventory, update_receipt
    monkeypatch.setenv("HERMES_ACTION_ID", "e" * 32)
    monkeypatch.setattr(update_inventory, "collect_runtime_inventory", lambda: SimpleNamespace(
        runtimes=[], to_dict=lambda: {}))
    with update_receipt.update_receipt_scope():
        update_cmd._begin_update_receipt_and_plan(SimpleNamespace())
        assert update_receipt.current_correlation_id() == "e" * 32
        assert update_receipt._current.get().data["update_id"] == "e" * 32
