"""Manual serve restart obligations survive gateway completion and receipt rotation (#107224)."""

import json
from dataclasses import asdict

import pytest

from hermes_cli import process_identity
from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_receipt
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
from hermes_cli.update_serve_obligations import defer_manual_serve, retain_receipt_manual_serves, warn_pending_manual_serves
from hermes_constants import get_hermes_home


@pytest.mark.parametrize("kind", ["serve", "dashboard"])
@pytest.mark.parametrize("condition", ["alive", "unknown", "write-error", "missing-identity", "failed-unit"])
def test_manual_deferral_survives_receipt_rotation(monkeypatch, capsys, kind, condition):
    runtime = RuntimeRecord(kind=kind, profile="work", pid=900, supervisor="manual-serve", restart_via="respawn-argv", detail={"create_time": 1000.0})
    plan = UpdatePlan(runtimes=[runtime])
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    monkeypatch.setattr(process_identity, "ledger_entries", lambda: [{"pid": 900, "purpose": kind, "create_time": 1000.0}])
    monkeypatch.setattr(fleet, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd._finish_dashboard_update_cleanup", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **k: [])
    restart = fleet._GatewayRestartOutcome(False, [], [], [], [], [], [], set())
    update_receipt.begin_update_receipt()
    fleet._write_fleet_restart_pending_marker(expected_sha="new")
    if condition == "unknown":
        monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: None)
    elif condition == "write-error":
        (get_hermes_home() / "serve_restart_pending").write_text("not a directory")
    elif condition == "missing-identity":
        runtime.detail.clear()
    elif condition == "failed-unit":
        restart.failed_or_stale_units.append("hermes-serve-work.service")
        restart.incomplete = True
    if condition != "alive":
        with pytest.raises(SystemExit) as exc:
            fleet._verify_fleet_after_update(restart, _pre_update_plan=plan, _windows_gateway_resume=None, node_failures=[], update_complete=True)
        assert exc.value.code == 1
        assert fleet._fleet_restart_obligation_armed()
        assert update_receipt.read_latest_receipt()["outcome"] == "partial"
        return
    fleet._verify_fleet_after_update(restart, _pre_update_plan=plan, _windows_gateway_resume=None, node_failures=[], update_complete=True)
    receipt = update_receipt.read_latest_receipt()
    assert receipt["runtime_outcomes"][0]["outcome"] == "deferred"
    assert not fleet._fleet_restart_obligation_armed()
    assert "hermes-serve.service" not in capsys.readouterr().out
    update_receipt.begin_update_receipt()
    update_receipt.finalize_update_receipt("success", fleet=[])
    fleet._warn_pending_fleet_restart_on_startup()
    warning = capsys.readouterr().err
    assert f"{kind} [work] pid 900" in warning
    assert "relaunch" in warning
    assert "hermes gateway restart" not in warning
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: None)
    fleet._warn_pending_fleet_restart_on_startup()
    assert "900" in capsys.readouterr().err
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: False)
    fleet._warn_pending_fleet_restart_on_startup()
    assert "900" not in capsys.readouterr().err


@pytest.mark.parametrize("alive", [True, None, False])
@pytest.mark.parametrize("historical_fleet", ["stale", "current", "empty"])
@pytest.mark.parametrize("marker", [True, False])
@pytest.mark.parametrize("gateway_present", [True, False])
def test_historical_manual_obligation_does_not_block_healthy_gateway(monkeypatch, capsys, alive, historical_fleet, marker, gateway_present):
    runtime = RuntimeRecord(kind="serve", profile="work", pid=900, supervisor="manual-serve", restart_via="respawn-argv", detail={"create_time": 1000.0})
    receipt = {"outcome": "partial", "plan": {"runtimes": [asdict(runtime), {"kind": "gateway", "profile": "default"}]}, "fleet": [{"profile": "default", "state": "stale", "code_sha": "old"}]}
    if historical_fleet == "current":
        receipt["fleet"] = [{"profile": "default", "state": "current", "code_sha": "new"}]
    elif historical_fleet == "empty":
        receipt["fleet"] = []
    if not gateway_present:
        receipt["plan"]["runtimes"] = [asdict(runtime)]
        receipt["fleet"] = []
    root = get_hermes_home() / "logs" / "update_receipts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr("hermes_cli.update_cmd._current_checkout_sha", lambda: "new")
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: alive)
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **k: [{"profile": "default", "state": "current", "code_sha": "new"}] if gateway_present else [])
    if marker:
        fleet._write_fleet_restart_pending_marker(expected_sha="new")
    # An inventory-less marker never inherits inventory from a historical receipt, but it
    # discharges when the live fleet provably serves its expected SHA (#115638).
    pending = marker and not gateway_present
    assert fleet._pending_fleet_restart_needed() is pending
    fleet._warn_pending_fleet_restart_on_startup()
    warning = capsys.readouterr().err
    assert ("serve [work] pid 900" in warning) is (alive is not False)
    assert ("hermes gateway restart" in warning) is pending
    assert json.loads((root / "latest.json").read_text()) == receipt


@pytest.mark.parametrize("marker", [True, False])
def test_stamped_manual_only_history_has_no_gateway_obligation(monkeypatch, capsys, marker):
    runtime = asdict(RuntimeRecord(kind="serve", profile="work", pid=900, code_sha="old", supervisor="manual-serve", restart_via="respawn-argv", detail={"create_time": 1000.0}))
    receipt = {"outcome": "partial", "plan": {"runtimes": [runtime]}, "fleet": []}
    root = get_hermes_home() / "logs" / "update_receipts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    monkeypatch.setattr("hermes_cli.update_cmd._current_checkout_sha", lambda: "new")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **k: [])
    if marker:
        fleet._write_fleet_restart_pending_marker(expected_sha="new")
    assert fleet._pending_fleet_restart_needed() is marker
    fleet._warn_pending_fleet_restart_on_startup()
    warning = capsys.readouterr().err
    assert "serve [work] pid 900" in warning
    assert ("hermes gateway restart" in warning) is marker


@pytest.mark.parametrize("manual_first", [True, False])
@pytest.mark.parametrize("unsupported", [{"kind": "serve"}, {"kind": "gateway", "profile": "unknown"}, None])
def test_historical_retention_is_independent_of_plan_order(monkeypatch, capsys, manual_first, unsupported):
    manual = asdict(RuntimeRecord(kind="serve", profile="work", pid=900, supervisor="manual-serve", restart_via="respawn-argv", detail={"create_time": 1000.0}))
    rows = [manual, unsupported] if manual_first else [unsupported, manual]
    receipt = {"outcome": "partial", "plan": {"runtimes": [{"kind": "gateway", "profile": "default"}, *rows]}, "fleet": [{"profile": "default", "state": "current", "code_sha": "new"}]}
    root = get_hermes_home() / "logs" / "update_receipts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    monkeypatch.setattr("hermes_cli.update_cmd._current_checkout_sha", lambda: "new")
    from hermes_cli.update_serve_obligations import retain_receipt_manual_serves
    pending_manual = retain_receipt_manual_serves(receipt)
    assert fleet._receipt_owed_gateways(receipt, pending_manual) is None
    assert list((get_hermes_home() / "serve_restart_pending").glob("*.json"))
    update_receipt.begin_update_receipt()
    update_receipt.finalize_update_receipt("success", fleet=[])
    fleet._warn_pending_fleet_restart_on_startup()
    assert "serve [work] pid 900" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["mkdir", "write", "replace"])
@pytest.mark.parametrize("gateway_state", ["current", "stale"])
def test_historical_retention_failure_warns_and_survives_rotation(monkeypatch, capsys, failure, gateway_state):
    from pathlib import Path
    from hermes_cli import update_serve_obligations as obligations

    manual = asdict(RuntimeRecord(kind="serve", profile="work", pid=900, supervisor="manual-serve", restart_via="respawn-argv", detail={"create_time": 1000.0}))
    receipt = {"outcome": "partial", "plan": {"runtimes": [manual]}, "fleet": [{"profile": "default", "state": gateway_state, "code_sha": "new"}]}
    root = get_hermes_home() / "logs" / "update_receipts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    monkeypatch.setattr("hermes_cli.update_cmd._current_checkout_sha", lambda: "new")
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **k: [{"profile": "default", "state": "current", "code_sha": "new"}])
    directory = get_hermes_home() / "serve_restart_pending"

    def fail(*args, **kwargs):
        raise OSError("injected persistence failure")

    with monkeypatch.context() as broken:
        if failure == "mkdir":
            original = Path.mkdir
            def mkdir(path, *args, **kwargs):
                if path == directory:
                    fail()
                return original(path, *args, **kwargs)
            broken.setattr(Path, "mkdir", mkdir)
        elif failure == "write":
            broken.setattr(obligations.json, "dump", fail)
        else:
            broken.setattr(obligations.os, "replace", fail)
        fleet._warn_pending_fleet_restart_on_startup()
        warning = capsys.readouterr().err
        assert "serve [work] pid 900" in warning
        assert "could not be saved" in warning
        assert json.loads((root / "latest.json").read_text()) == receipt
        for _ in range(2):
            update_receipt.begin_update_receipt()
            assert update_receipt.finalize_update_receipt("success", fleet=[]) is not None
            fleet._warn_pending_fleet_restart_on_startup()
            assert "serve [work] pid 900" in capsys.readouterr().err
    fleet._warn_pending_fleet_restart_on_startup()
    warning = capsys.readouterr().err
    assert "serve [work] pid 900" in warning
    assert "could not be saved" not in warning
    assert len(list(directory.iterdir())) == 1
    update_receipt.begin_update_receipt()
    update_receipt.finalize_update_receipt("success", fleet=[])
    fleet._warn_pending_fleet_restart_on_startup()
    assert "serve [work] pid 900" in capsys.readouterr().err
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: False)
    fleet._warn_pending_fleet_restart_on_startup()
    assert "900" not in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["serve", "dashboard"])
@pytest.mark.parametrize("alive", [True, None, False])
def test_unreadable_create_time_discharges_only_a_proven_dead_pid(monkeypatch, kind, alive):
    """#116507: a row without a readable create_time is discharged once its pid is provably dead,
    stays pending while it is live or unknowable, and never counts as a live handoff."""
    runtime = asdict(RuntimeRecord(kind=kind, profile="work", pid=900, supervisor="manual-serve", restart_via="respawn-argv", detail={"create_time": None}))
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: alive)
    pending = retain_receipt_manual_serves({"plan": {"runtimes": [runtime]}})
    assert pending == ([] if alive is False else [runtime])
    assert defer_manual_serve(runtime, require_alive=True) is False


def test_unreadable_create_time_warning_names_identity_not_storage(monkeypatch, capsys):
    runtime = asdict(RuntimeRecord(kind="serve", profile="work", pid=900, supervisor="manual-serve", restart_via="respawn-argv", detail={"create_time": None}))
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    warn_pending_manual_serves(pending_manual=[runtime])
    out = capsys.readouterr().out
    assert "could not read the process creation time" in out
    assert "storage permissions" not in out
    assert "relaunch" in out


def test_launchd_serve_row_never_pessimize_gateway_coverage():
    """#116503: a launchd-owned serve/dashboard row is the post-update dashboard cleanup pass's
    to kickstart, so it must not make the receipt's gateway coverage unverified (owed=None keeps
    ``fleet_restart_pending`` armed with nothing gateway-side left to restart)."""
    launchd = asdict(RuntimeRecord(
        kind="serve", profile="work", pid=900, supervisor="launchd", restart_via="launchd", detail={}))
    receipt = {"plan": {"runtimes": [{"kind": "gateway", "profile": "default"}, launchd]}, "fleet": []}
    assert fleet._receipt_owed_gateways(receipt, []) == {("gateway", "default")}
