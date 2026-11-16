"""Protected gateway ownership remains an owed restart across reconciliation."""

from types import SimpleNamespace
import json

from hermes_cli import update_cmd_fleet_verify as verify


def test_settling_another_unit_does_not_clear_protected_gateway_debt(monkeypatch):
    restart = SimpleNamespace(
        pre_restart_gateway_pids=[11], failed_or_stale_units=["hermes-gateway"],
        unrestarted_units=["hermes-gateway"], restarted_services=[], phase_errors=[],
        protected_gateway_pids={11}, incomplete=True,
    )
    monkeypatch.setattr(verify, "_gateway_unit_runs_one_of", lambda unit, pids: True)

    assert verify._settle_units_proven_current(restart, [{"state": "current", "pid": 22}])
    assert restart.incomplete
    assert not restart.failed_or_stale_units


def test_protected_gateway_debt_is_not_classified_as_serve_only():
    restart = SimpleNamespace(
        incomplete=True, failed_or_stale_units=["hermes-dashboard.service"], phase_errors=[],
        protected_gateway_pids={11},
    )

    assert verify._restart_phase_owes_gateway(restart)
    reason = verify._owed_restart_reason(restart, [], gateway_owed=True, stale_serves=set(), down_serves=set())
    assert "PID(s) 11" in reason
    assert "supervisor or owner" in reason


def test_marker_does_not_credit_unrelated_current_gateways_for_unmapped_protected_pids(monkeypatch, tmp_path):
    from hermes_cli import update_cmd_fleet as fleet, update_host_obligation as host, update_receipt
    from hermes_cli.update_inventory import UpdatePlan

    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "head")
    fleet._write_fleet_restart_pending_marker(expected_sha="head", runtimes=[])
    verify._record_owed_gateway_inventory(UpdatePlan(runtimes=[]), protected_gateway_pids={11, 12})
    rows = json.loads(fleet._obligation_fields()["inventory"])["runtimes"]
    assert {row["pid"] for row in rows if row.get("protected_unmapped")} == {11, 12}

    current = [{"profile": "default", "state": "current", "code_sha": "head", "pid": 21}]
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kwargs: list(current))
    assert not fleet._marker_only_restart_obsolete()
    current.append({"profile": "work", "state": "current", "code_sha": "head", "pid": 22})
    assert not fleet._marker_only_restart_obsolete()


def test_marker_clears_only_for_verified_successor_of_protected_home(monkeypatch, tmp_path):
    from gateway import status
    from hermes_cli import update_cmd_fleet as fleet, update_cmd_gateway_ownership as ownership
    from hermes_cli import update_fleet_scope, update_host_obligation as host, update_receipt
    from hermes_cli.update_inventory import UpdatePlan

    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "head")
    monkeypatch.setattr(update_fleet_scope, "gateway_pid_home", lambda pid: str(tmp_path / "gateway"))
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 123)
    monkeypatch.setattr(ownership, "_systemd_service_cgroup_for_pid", lambda pid: "/system.slice/custom.service")
    fleet._write_fleet_restart_pending_marker(expected_sha="head", runtimes=[])
    verify._record_owed_gateway_inventory(UpdatePlan(runtimes=[]), protected_gateway_pids={11})

    current = [{"profile": "default", "state": "current", "code_sha": "head", "pid": 21}]
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kwargs: list(current))
    monkeypatch.setattr(status, "live_gateway_pid_for_home", lambda home: 22)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
    assert not fleet._marker_only_restart_obsolete()  # unrelated live fleet row
    current[0]["pid"] = 22
    monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
    assert not fleet._marker_only_restart_obsolete()  # original incarnation is still alive
    monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
    assert fleet._marker_only_restart_obsolete()


def test_mapped_plan_can_supply_home_for_protected_pid(monkeypatch, tmp_path):
    from hermes_cli import update_cmd_fleet as fleet, update_host_obligation as host, update_receipt
    from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "head")
    monkeypatch.setattr(update_receipt, "_profile_homes", lambda: [("default", tmp_path / "home")])
    fleet._write_fleet_restart_pending_marker(expected_sha="head", runtimes=[])
    identity = {"kind": "gateway", "profile": None, "pid": 11, "protected_unmapped": True,
                "home": None, "birth": 123}
    verify._record_owed_gateway_inventory(UpdatePlan(runtimes=[RuntimeRecord(kind="gateway", profile="default", pid=11)]),
                                           protected_gateway_pids={11}, protected_identities={11: identity})
    rows = json.loads(fleet._obligation_fields()["inventory"])["runtimes"]
    assert next(row for row in rows if row.get("protected_unmapped"))["home"] == str(tmp_path / "home")
