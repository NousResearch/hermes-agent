"""Explicit acknowledgments remove only the protected debt the operator named."""

from concurrent.futures import ThreadPoolExecutor

from hermes_cli import update_host_obligation as host
import pytest


@pytest.fixture(autouse=True)
def no_real_receipt(monkeypatch):
    from hermes_cli import update_receipt
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: None)


def _protected(pid):
    return {"kind": "gateway", "profile": None, "pid": pid, "protected_unmapped": True,
            "incident_id": f"incident-{pid}", "home": None, "birth": None}


def test_acknowledgment_preserves_other_debts_and_rejects_stale_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    rows = [_protected(11), _protected(12), {"kind": "gateway", "profile": "default", "pid": 20}]
    assert host.write_host_obligation(expected_sha="abc", runtimes=rows)
    tokens = dict(host.protected_acknowledgments())
    assert host.acknowledge_protected_gateway(tokens[11])
    assert host.read_host_obligation()["inventory"]["runtimes"] == rows[1:]
    assert not host.acknowledge_protected_gateway(tokens[11])
    host.amend_host_obligation(expected_sha="new-sha")
    assert not host.acknowledge_protected_gateway(tokens[12])
    assert host.read_host_obligation()["inventory"]["runtimes"] == rows[1:]


def test_simultaneous_acknowledgments_do_not_lose_other_debts(monkeypatch, tmp_path):
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    rows = [_protected(11), _protected(12), {"kind": "gateway", "profile": "default", "pid": 20}]
    assert host.write_host_obligation(expected_sha="abc", runtimes=rows)
    tokens = dict(host.protected_acknowledgments())
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(host.acknowledge_protected_gateway, [tokens[11], tokens[12]]))
    assert host.read_host_obligation()["inventory"]["runtimes"] == [rows[2]]


def test_ambiguous_duplicate_and_missing_records_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    assert not host.acknowledge_protected_gateway("unknown")
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11), _protected(11), _protected(12)])
    row, count, token = host.protected_acknowledgment_details()[0]
    assert row == _protected(11) and count == 2
    assert host.acknowledge_protected_gateway(token)
    assert not host.acknowledge_protected_gateway(token)
    assert host.read_host_obligation()["inventory"]["runtimes"] == [_protected(12)]


def test_old_token_does_not_acknowledge_new_incident_with_same_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    token = host.protected_acknowledgments()[0][1]
    assert host.acknowledge_protected_gateway(token)
    replacement = {**_protected(11), "incident_id": "new-incident"}
    host.amend_host_obligation(inventory={"version": 1, "runtimes": [replacement]})
    assert not host.acknowledge_protected_gateway(token)
    assert host.read_host_obligation()["inventory"]["runtimes"] == [replacement]


def test_acknowledgment_is_scoped_to_witnessed_receipt_rows_across_two_startups(monkeypatch, tmp_path):
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "abc")
    row = {"pid": 11, "profile": "default", "state": "stale", "code_sha": "old",
           "protected_incident_id": "incident-11"}
    receipt = {"update_id": "receipt-1", "post_update": {"sha": "abc"}, "outcome": "success",
               "fleet": [row], "plan": {"runtimes": []}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    other = {**receipt, "update_id": "receipt-2"}
    assert fleet._update_owes_fleet_restart(receipt=other, pending_manual=[])


def test_receipt_filter_keeps_unwitnessed_and_non_gateway_rows(monkeypatch, tmp_path):
    from hermes_cli import update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    acknowledged = {"pid": 11, "profile": "default", "state": "stale", "protected_incident_id": "incident-11"}
    reused = {"pid": 11, "profile": "work", "state": "stale"}
    serve = {"pid": 11, "kind": "serve", "profile": "default", "protected_incident_id": "incident-11"}
    receipt = {"update_id": "receipt-1", "post_update": {"sha": "abc"}, "fleet": [acknowledged, reused],
               "plan": {"runtimes": [serve]}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    filtered = host.receipt_without_acknowledged_gateways(receipt)
    assert filtered["fleet"] == [reused]
    assert filtered["plan"]["runtimes"] == [serve]


def test_receipt_incident_tag_requires_matching_birth(monkeypatch):
    from gateway import status
    from hermes_cli import update_cmd_fleet_verify as verify
    identities = {11: {"birth": 100, "incident_id": "incident-11"}}
    row = {"pid": 11, "profile": "default"}
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 500)
    verify._tag_protected_fleet_rows([row], identities)
    assert "protected_incident_id" not in row
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 100)
    verify._tag_protected_fleet_rows([row], identities)
    assert row["protected_incident_id"] == "incident-11"


def test_unwitnessed_stale_plan_remains_owed_after_ack(monkeypatch, tmp_path):
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "abc")
    receipt = {"update_id": "receipt-1", "post_update": {"sha": "abc"}, "outcome": "partial",
               "fleet": [{"pid": 11, "profile": "default", "state": "stale", "code_sha": "old",
                          "protected_incident_id": "incident-11"}],
               "plan": {"runtimes": [{"kind": "gateway", "pid": 11, "profile": "default", "code_sha": "old"}]}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    token = next(token for source, row, token in host.receipt_gateway_acknowledgments(receipt) if source == "plan")
    assert host.acknowledge_receipt_gateway(token)
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])


def test_rediscovery_of_same_incarnation_keeps_original_incident(monkeypatch, tmp_path):
    from gateway import status
    from hermes_cli import update_cmd_gateway_ownership as ownership, update_fleet_scope
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    old = {**_protected(11), "home": "/home/gateway", "birth": 100}
    assert host.write_host_obligation(expected_sha="abc", runtimes=[old])
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 100)
    monkeypatch.setattr(update_fleet_scope, "gateway_pid_home", lambda pid: "/home/gateway")
    assert ownership.protected_gateway_identity(11)["incident_id"] == "incident-11"
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 200)
    assert ownership.protected_gateway_identity(11)["incident_id"] != "incident-11"


def test_receipt_row_ack_preserves_host_and_other_receipt_debts(monkeypatch, tmp_path):
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    receipt = {"update_id": "receipt-1", "post_update": {"sha": "abc"}, "outcome": "partial",
               "fleet": [{"pid": 11, "profile": "default", "state": "stale", "code_sha": "old"},
                         {"pid": 12, "profile": "work", "state": "stale", "code_sha": "old"}],
               "gateway_restart": {"incomplete": True, "phase_error": "unit failed"},
               "plan": {"runtimes": [{"kind": "serve", "pid": 11, "profile": "default", "code_sha": "old"}]}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11), _protected(30)])
    assert host.acknowledge_protected_gateway(dict(host.protected_acknowledgments())[11])
    tokens = host.receipt_gateway_acknowledgments(receipt)
    first = next(token for source, row, token in tokens if source == "fleet" and row["pid"] == 11)
    assert host.acknowledge_receipt_gateway(first)
    assert host.read_host_obligation()["inventory"]["runtimes"] == [_protected(30)]
    filtered = host.receipt_without_acknowledged_gateways(receipt)
    assert [row["pid"] for row in filtered["fleet"]] == [12]
    assert filtered["plan"]["runtimes"] == receipt["plan"]["runtimes"]
    assert filtered["gateway_restart"] == receipt["gateway_restart"]
    host.clear_host_obligation()
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])


def test_generic_restart_debt_requires_its_own_attestation(monkeypatch, tmp_path):
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    receipt = {"update_id": "receipt-1", "post_update": {"sha": "abc"}, "outcome": "partial",
               "fleet": [{"pid": 11, "profile": "default", "state": "stale", "code_sha": "old"}],
               "gateway_restart": {"incomplete": True, "phase_error": "unit failed"}, "plan": {"runtimes": []}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    tokens = host.receipt_gateway_acknowledgments(receipt)
    assert host.acknowledge_receipt_gateway(next(token for source, row, token in tokens if source == "fleet"))
    assert not any(source == "restart" for source, row, token in host.receipt_gateway_acknowledgments(receipt))
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert fleet._pending_fleet_restart_needed(receipt=receipt, pending_manual=[])


def test_receipt_token_rejects_changed_snapshot_and_identical_ambiguous_rows(monkeypatch, tmp_path):
    from hermes_cli import update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    row = {"pid": 11, "profile": "default", "state": "stale"}
    receipt = {"update_id": "receipt-1", "post_update": {"sha": "abc"}, "fleet": [row], "plan": {"runtimes": []}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    token = host.receipt_gateway_acknowledgments(receipt)[0][2]
    receipt["fleet"].append({"pid": 12, "profile": "work", "state": "stale"})
    assert not host.acknowledge_receipt_gateway(token)
    receipt["fleet"] = [row, dict(row)]
    # A changed snapshot cannot reuse the previous context or its token.
    assert host.receipt_gateway_acknowledgments(receipt) == []


def test_judged_host_discharge_preserves_new_concurrent_obligation(monkeypatch, tmp_path):
    from hermes_cli import update_cmd_fleet as fleet
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    assert host.write_host_obligation(expected_sha="old", runtimes=[])
    path = host.host_obligation_path()
    old = host.read_host_obligation()
    assert host.write_host_obligation(expected_sha="new", runtimes=[_protected(11)])
    assert not fleet._clear_judged_fleet_marker((path, old))
    assert host.read_host_obligation()["expected_sha"] == "new"


def test_fallback_marker_acknowledges_protected_incident_without_host_storage(monkeypatch, tmp_path):
    import json
    from hermes_cli import update_cmd_fleet as fleet, update_receipt
    marker = tmp_path / "fleet_restart_pending"
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "unavailable" / "host.json")
    monkeypatch.setattr(fleet, "_fleet_restart_pending_marker_path", lambda: marker)
    receipt = {"update_id": "r1", "post_update": {"sha": "abc"}, "fleet": [], "plan": {"runtimes": []}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    inventory = {"version": 1, "runtimes": [_protected(11), _protected(12)]}
    marker.write_text("started=123\npid=1\nexpected_sha=abc\ninventory=" + json.dumps(inventory) + "\n")
    tokens = dict(host.protected_acknowledgments())
    assert host.acknowledge_protected_gateway(tokens[11])
    fields = fleet._obligation_fields()
    assert json.loads(fields["inventory"])["runtimes"] == [_protected(12)]
    assert not host.acknowledge_protected_gateway(tokens[11])


def test_explicit_recovery_clears_only_proven_protected_receipt_twice(monkeypatch, tmp_path):
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "abc")
    receipt = {"update_id": "receipt-1", "post_update": {"sha": "abc"}, "outcome": "partial",
               "fleet": [{"pid": 11, "profile": "default", "state": "stale", "code_sha": "old"}],
               "gateway_restart": {"incomplete": True, "failed_units": [], "phase_error": ""},
               "followups": [{"step": "gateway_restart", "reason": "protected PID 11"}],
               "plan": {"runtimes": [{"kind": "gateway", "pid": 11, "profile": "default", "code_sha": "old"}]}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    for source in ("fleet", "plan", "restart", "followup"):
        tokens = host.receipt_gateway_acknowledgments(receipt)
        assert host.acknowledge_receipt_gateway(next(token for kind, _, token in tokens if kind == source))
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])


def test_inventory_merge_preserves_concurrent_gateway_debts(monkeypatch, tmp_path):
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(10)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(lambda pid: host.merge_owed_gateway_inventory("abc", [_protected(pid)]), [11, 12]))
    assert {row["pid"] for row in host.read_host_obligation()["inventory"]["runtimes"]} == {10, 11, 12}
    assert not host.merge_owed_gateway_inventory("different", [_protected(13)])
    assert {row["pid"] for row in host.read_host_obligation()["inventory"]["runtimes"]} == {10, 11, 12}


def test_fallback_acknowledgments_survive_two_startups(monkeypatch, tmp_path):
    import json
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
    marker = tmp_path / "fleet_restart_pending"
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "unavailable" / "host.json")
    monkeypatch.setattr(fleet, "_fleet_restart_pending_marker_path", lambda: marker)
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    receipt = {"update_id": "r1", "post_update": {"sha": "abc"}, "outcome": "partial",
               "fleet": [{"pid": 11, "profile": "unknown", "state": "stale", "code_sha": "old"}],
               "plan": {"runtimes": []}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    marker.write_text("started=123\npid=1\nexpected_sha=abc\ninventory="
                      + json.dumps({"version": 1, "runtimes": [_protected(11)]}) + "\n")
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    tokens = host.receipt_gateway_acknowledgments(receipt)
    assert host.acknowledge_receipt_gateway(next(token for source, _, token in tokens if source == "fleet"))
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])


def test_fallback_inventory_can_be_added_to_initial_inventoryless_marker(monkeypatch, tmp_path):
    import json
    from hermes_cli import update_cmd_fleet as fleet, update_gateway_ack_legacy as legacy
    marker = tmp_path / "fleet_restart_pending"
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "unavailable" / "host.json")
    monkeypatch.setattr(fleet, "_fleet_restart_pending_marker_path", lambda: marker)
    marker.write_text("started=123\npid=1\nexpected_sha=abc\n")
    assert legacy.merge_fallback_inventory("abc", [_protected(11)])
    assert json.loads(fleet._obligation_fields()["inventory"])["runtimes"] == [_protected(11)]
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])


def test_newer_receipt_requires_explicit_entry_acknowledgment(monkeypatch, tmp_path):
    from hermes_cli import update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    row = {"pid": 11, "profile": "unknown", "state": "stale", "protected_incident_id": "incident-11"}
    receipt = {"update_id": "r1", "post_update": {"sha": "newer"}, "fleet": [row], "plan": {"runtimes": []}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert host.write_host_obligation(expected_sha="abc", runtimes=[_protected(11)])
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    assert host.receipt_without_acknowledged_gateways(receipt)["fleet"] == [row]
    token = next(token for source, _, token in host.receipt_gateway_acknowledgments(receipt) if source == "fleet")
    assert host.acknowledge_receipt_gateway(token)
    assert host.receipt_without_acknowledged_gateways(receipt)["fleet"] == []


def test_failed_inventory_merge_surfaces_and_records_evidence(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from hermes_cli import update_cmd_fleet as fleet, update_cmd_fleet_verify as verify, update_receipt
    from hermes_cli import update_gateway_ack_legacy as legacy
    monkeypatch.setattr(host, "merge_owed_gateway_inventory", lambda *args: False)
    monkeypatch.setattr(legacy, "merge_fallback_inventory", lambda *args: False)
    monkeypatch.setattr(fleet, "_fleet_restart_obligation_armed", lambda: False)
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "abc")
    active = SimpleNamespace(data={})
    monkeypatch.setattr(update_receipt, "_current", SimpleNamespace(get=lambda: active))
    identity = _protected(11)
    verify._record_owed_gateway_inventory(None, protected_gateway_pids={11}, protected_identities={11: identity})
    assert active.data["unrecorded_gateway_obligations"] == [identity]
    assert "Could not record every gateway restart obligation" in capsys.readouterr().out


def test_unrecorded_receipt_incident_stays_owed_until_explicit_attestation(monkeypatch, tmp_path):
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host.json")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "abc")
    receipt = {"update_id": "r1", "post_update": {"sha": "abc"}, "outcome": "success", "fleet": [],
               "plan": {"runtimes": []}, "unrecorded_gateway_obligations": [_protected(11)]}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert fleet._pending_fleet_restart_needed(receipt=receipt, pending_manual=[])
    token = next(token for source, _, token in host.receipt_gateway_acknowledgments(receipt) if source == "unrecorded")
    assert host.acknowledge_receipt_gateway(token)
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert not fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])


def test_unrecorded_receipt_ack_does_not_clear_corrupt_host_obligation(monkeypatch, tmp_path):
    from hermes_cli import update_cmd_fleet as fleet, update_receipt
    path = tmp_path / "host.json"
    path.write_text("invalid")
    monkeypatch.setattr(host, "host_obligation_path", lambda: path)
    receipt = {"update_id": "r1", "post_update": {"sha": "abc"}, "unrecorded_gateway_obligations": [_protected(11)]}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    token = host.receipt_gateway_acknowledgments(receipt)[0][2]
    assert host.acknowledge_receipt_gateway(token)
    assert path.read_text() == "invalid"
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])


def test_unrecorded_incident_is_not_silenced_by_unrelated_serve_plan(monkeypatch, tmp_path):
    from hermes_cli import update_cmd, update_cmd_fleet as fleet
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host.json")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    receipt = {"update_id": "r1", "outcome": "success", "unrecorded_gateway_obligations": [_protected(11)],
               "plan": {"runtimes": [{"kind": "serve", "profile": "default", "supervisor": "systemd"}]}}
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert fleet._pending_fleet_restart_needed(receipt=receipt, pending_manual=[])


def test_identical_unrecorded_entries_are_one_explicit_snapshot_bound_group(monkeypatch, tmp_path):
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_receipt
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host.json")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "abc")
    row = _protected(11)
    other = _protected(12)
    receipt = {"update_id": "r1", "outcome": "success", "plan": {"runtimes": []},
               "unrecorded_gateway_obligations": [row, dict(row), other]}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    items = host.receipt_gateway_acknowledgments(receipt)
    group, group_token = next((item, token) for source, item, token in items if source == "duplicate_group")
    assert group == {"source": "unrecorded", "entry": row, "count": 2}
    assert host.acknowledge_receipt_gateway(group_token)
    assert not host.acknowledge_receipt_gateway(group_token)
    assert host.receipt_without_acknowledged_gateways(receipt)["unrecorded_gateway_obligations"] == [other]
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    assert fleet._update_owes_fleet_restart(receipt=receipt, pending_manual=[])
    # A receipt change invalidates the old group token, even if its entries recur.
    receipt["extra"] = "changed"
    assert not host.acknowledge_receipt_gateway(group_token)


def test_unavailable_host_state_does_not_hide_fallback_receipt_tokens(monkeypatch, tmp_path):
    from hermes_cli import update_cmd_fleet as fleet, update_receipt
    host_root = tmp_path / "not-a-directory"
    host_root.write_text("not a directory")
    marker = tmp_path / "fleet_restart_pending"
    monkeypatch.setattr(host, "host_obligation_path", lambda: host_root / "host.json")
    monkeypatch.setattr(fleet, "_fleet_restart_pending_marker_path", lambda: marker)
    receipt = {"update_id": "r1", "post_update": {"sha": "abc"}, "fleet": [{"pid": 11, "profile": "unknown", "state": "stale"}],
               "plan": {"runtimes": []}}
    monkeypatch.setattr(update_receipt, "read_latest_receipt", lambda: receipt)
    import json
    marker.write_text("started=123\npid=1\nexpected_sha=abc\ninventory="
                      + json.dumps({"version": 1, "runtimes": [_protected(11)]}) + "\n")
    assert host.acknowledge_protected_gateway(host.protected_acknowledgments()[0][1])
    tokens = host.receipt_gateway_acknowledgments(receipt)
    assert host.acknowledge_receipt_gateway(next(token for source, _, token in tokens if source == "fleet"))
    assert host.receipt_without_acknowledged_gateways(receipt)["fleet"] == []
