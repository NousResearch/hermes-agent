"""Regression for #93406 — post-update fleet version check must fail closed.

``collect_fleet_versions()`` swallows every probe failure via
``logger.debug()`` and ``print_fleet_version_matrix([])`` early-returns
``False``, so an empty fleet snapshot used to read as "healthy fleet" and
``hermes update`` exited 0 with zero rows — even when a gateway was
verifiably live before the update.

The first guard (PR #93410) keyed on ``(restarted_services or killed_pids)``,
which never fires on Windows: ``_pause_windows_gateways_for_update`` /
``_resume_windows_gateways_after_update`` populate neither list.  The fix
hoists the "should the probe have produced rows?" decision into
``_fleet_probe_expected_runtimes`` and keys it on the ROW-CAPABLE pre-update
liveness signals: restart-phase bookkeeping, the pre-restart PID snapshot,
and the gateway-kind records in the pre-update plan inventory (the plan's
serve/dashboard records are row-incapable for this probe, #97332).  The
Windows pause/resume token is deliberately NOT a signal — it is bookkeeping,
not a runtime inventory, and its entries have no corresponding
``collect_fleet_versions()`` rows (see
``test_update_fleet_probe_resume_token.py``).  The same condition gates the
2.0s settle sleep.
"""

from __future__ import annotations

import json
import types

import pytest

from hermes_cli.main import _fleet_probe_expected_runtimes
from hermes_cli.update_inventory import RuntimeRecord


def _plan(runtimes):
    return types.SimpleNamespace(runtimes=runtimes)


class TestCallSiteWiring:
    @pytest.mark.parametrize("had_gateway", [False, True], ids=["idle", "plan-saw-gateway"])
    def test_empty_probe_settles_and_fails_only_when_rows_expected(self, monkeypatch, tmp_path, capsys, had_gateway):
        import json
        from hermes_cli import main, update_cmd, update_cmd_fleet as fleet, update_cmd_fleet_verify as fleet_verify, update_receipt

        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
        monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update", lambda **kw: None)
        monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
        # Reconciliation is a separate guard; it must not supply this test's failure.
        monkeypatch.setattr("hermes_cli.update_inventory.report_unaccounted_runtimes", lambda rows: False)
        monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
        events = []
        now = [0.0]

        def sleep(seconds):
            events.append("settle")
            now[0] += seconds

        def collect(**kwargs):
            events.append("probe")
            return []

        monkeypatch.setattr(fleet_verify, "_time", types.SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
        monkeypatch.setattr(update_receipt, "collect_fleet_versions", collect)
        restart = fleet._GatewayRestartOutcome(
            incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
            failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
        )
        plan = _plan([RuntimeRecord(kind="gateway", profile="default")] if had_gateway else [])
        update_receipt.begin_update_receipt()

        def verify():
            fleet_verify._verify_fleet_after_update(restart, _pre_update_plan=plan,
                                            _windows_gateway_resume=None, update_complete=True)

        cleared = []
        monkeypatch.setattr(fleet, "_clear_fleet_restart_pending_marker", lambda: cleared.append(True))
        # Contract C3: the code is committed, so verification never fails the update (was SystemExit(1)).
        verify()
        if had_gateway:
            assert events[0] == "settle"
            assert events.count("probe") > 1
            assert "returned no rows" in capsys.readouterr().out
        else:
            assert events == ["probe"]
        # Fail-closed survives as an OWED restart: the fleet obligation is kept armed ...
        assert restart.incomplete is had_gateway
        assert cleared == ([] if had_gateway else [True])
        receipt = json.loads((update_cmd.get_hermes_home() / "logs/update_receipts/latest.json").read_text())
        # ... and the receipt is a success that names the owed step (was outcome "partial").
        assert receipt["outcome"] == "success"
        assert [f["step"] for f in receipt.get("followups", [])] == (["gateway_restart"] if had_gateway else [])



def test_unmapped_stops_are_not_expected_rows():
    # A gateway stopped WITHOUT a successor is listed under "Restart manually" and never
    # publishes a row; counting it made the probe demand rows that cannot exist and the
    # update exited 1 after correctly stopping every unmapped gateway.
    from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome

    out = _GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[101, 102], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids={101, 102}, stopped_unmapped_pids={101, 102},
    )
    pre, killed = out.fleet_probe_signals()
    assert not _fleet_probe_expected_runtimes(_plan([]), pre, None, out.restarted_services, killed)
    # A relaunched profile gateway (not unmapped) still predicts a row.
    out.stopped_unmapped_pids.discard(102)
    pre, killed = out.fleet_probe_signals()
    assert _fleet_probe_expected_runtimes(_plan([]), pre, None, out.restarted_services, killed)


def test_unmapped_stop_keeps_the_restart_owed(monkeypatch, tmp_path):
    # Codemap §6 V7/V12: an unmapped gateway stopped with no successor used to be "accounted for"
    # (update exit 0, obligation cleared) and stayed down silently. It still predicts no row (the
    # test above), but the restart is now OWED: a receipt follow-up and an armed fleet obligation.
    import json
    from hermes_cli import main, update_cmd, update_cmd_fleet as fleet, update_cmd_fleet_verify as fleet_verify, update_receipt

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update", lambda **kw: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: [])
    cleared = []
    monkeypatch.setattr(fleet, "_clear_fleet_restart_pending_marker", lambda: cleared.append(True))
    out = fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[101], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids={101}, stopped_unmapped_pids={101},
    )
    update_receipt.begin_update_receipt()
    fleet_verify._verify_fleet_after_update(out, _pre_update_plan=_plan([]), _windows_gateway_resume=None,
                                     update_complete=True)
    receipt = json.loads((update_cmd.get_hermes_home() / "logs/update_receipts/latest.json").read_text())
    assert receipt["outcome"] == "success"
    assert [f["step"] for f in receipt["followups"]] == ["gateway_restart"]
    assert "101" in receipt["followups"][0]["reason"]
    assert cleared == []


def test_unmapped_stop_debt_survives_startup_until_a_current_gateway_runs(monkeypatch, tmp_path):
    # An unmapped gateway leaves no gateway_state.json, so on the next start the empty host looked
    # gateway-less and the inventory-less obligation was discharged: the stopped gateway's restart
    # debt vanished. Only a live gateway serving the checkout may settle it.
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_cmd_fleet_verify as fleet_verify
    from hermes_cli import update_cmd_fleet_gatewayless as gatewayless, update_host_obligation as host, update_receipt

    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "head")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "head")
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update", lambda **kw: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(gatewayless, "host_owes_no_gateway_restart", lambda: True)
    fleet_rows: list = []
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: list(fleet_rows))
    fleet._write_fleet_restart_pending_marker(expected_sha="head")  # armed by the pull, inventory-less
    out = fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[101], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids={101}, stopped_unmapped_pids={101},
    )
    update_receipt.begin_update_receipt()
    fleet_verify._verify_fleet_after_update(out, _pre_update_plan=_plan([]), _windows_gateway_resume=None,
                                            update_complete=True)

    assert fleet._update_owes_fleet_restart(receipt={}, pending_manual=[]) is True
    assert fleet._pending_fleet_restart_needed(receipt={}, pending_manual=[]) is True  # already-current update
    assert fleet._fleet_restart_obligation_armed()

    fleet_rows.append({"profile": "default", "state": "current", "code_sha": "head"})  # `hermes gateway run`
    assert fleet._update_owes_fleet_restart(receipt={}, pending_manual=[]) is False
    assert not fleet._fleet_restart_obligation_armed()


def test_unmapped_stop_debt_is_not_settled_by_the_mapped_gateways_restart(monkeypatch, tmp_path):
    # The inventory owes `default` AND an unmapped gateway: `default` coming back current is the
    # successor of `default`, not of the unmapped one, so the debt stays until a further local
    # gateway (the unmapped one's successor, whatever profile it names) runs the checkout.
    from hermes_cli import update_cmd_fleet as fleet, update_host_obligation as host, update_receipt

    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "head")
    fleet._write_fleet_restart_pending_marker(expected_sha="head", runtimes=[
        {"kind": "gateway", "profile": "default", "pid": 7},
        {"kind": "gateway", "profile": None, "pid": 101, "stopped_unmapped": True},
    ])
    fleet_rows = [{"profile": "default", "state": "current", "code_sha": "head", "pid": 8}]
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: list(fleet_rows))

    assert fleet._update_owes_fleet_restart(receipt={}, pending_manual=[]) is True
    assert fleet._fleet_restart_obligation_armed()

    fleet_rows.append({"profile": "work", "state": "current", "code_sha": "head", "pid": 9})
    assert fleet._update_owes_fleet_restart(receipt={}, pending_manual=[]) is False
    assert not fleet._fleet_restart_obligation_armed()


def test_same_sha_retry_adds_newly_stopped_unmapped_debt_to_an_existing_inventory(monkeypatch, tmp_path):
    # A same-SHA catch-up keeps the standing inventory (gateway `default`, still down). If that retry
    # also stops an unmapped gateway, its debt must join the inventory: otherwise `default` coming
    # back settles the obligation while the unmapped gateway still has no successor.
    from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_cmd_fleet_verify as fleet_verify
    from hermes_cli import update_host_obligation as host, update_receipt
    from hermes_cli.update_inventory import UpdatePlan

    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host-update-restart.json")
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "head")
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "head")
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update", lambda **kw: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(fleet_verify, "_FLEET_PROBE_SETTLE_TIMEOUT_SECONDS", 0)  # A stays down: one poll
    # Signal delivery is the external boundary: the down row has no process to drain.
    monkeypatch.setattr("hermes_cli.update_cmd_stale_survivors.signal_stale_fleet_survivors", lambda *a: None)
    fleet_rows = [{"profile": "default", "state": "down", "code_sha": "old", "pid": 7}]
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: list(fleet_rows))
    fleet._write_fleet_restart_pending_marker(expected_sha="head", runtimes=[
        {"kind": "gateway", "profile": "default", "pid": 7}])
    out = fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[7999], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids={7999}, stopped_unmapped_pids={7999},
    )
    update_receipt.begin_update_receipt()
    fleet_verify._verify_fleet_after_update(
        out, _pre_update_plan=UpdatePlan(runtimes=[RuntimeRecord(kind="gateway", profile="default", pid=7)]),
        _windows_gateway_resume=None, update_complete=True)

    rows = json.loads(fleet._obligation_fields()["inventory"])["runtimes"]
    assert {"kind": "gateway", "profile": "default", "pid": 7} in rows  # earlier debt kept
    assert {"kind": "gateway", "profile": None, "pid": 7999, "stopped_unmapped": True} in rows
    fleet_rows[:] = [{"profile": "default", "state": "current", "code_sha": "head", "pid": 8}]
    assert fleet._marker_only_restart_obsolete() is False  # `default`'s successor is not 7999's
    assert fleet._fleet_restart_obligation_armed()
    fleet_rows.append({"profile": "work", "state": "current", "code_sha": "head", "pid": 9})
    assert fleet._marker_only_restart_obsolete() is True
