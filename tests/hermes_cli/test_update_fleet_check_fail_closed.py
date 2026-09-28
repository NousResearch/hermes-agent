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
        from hermes_cli import main, update_cmd, update_cmd_fleet as fleet, update_receipt

        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(fleet, "_print_legacy_units_warning", lambda: None)
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

        monkeypatch.setattr(fleet, "_time", types.SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
        monkeypatch.setattr(update_receipt, "collect_fleet_versions", collect)
        restart = fleet._GatewayRestartOutcome(
            incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
            failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
        )
        plan = _plan([RuntimeRecord(kind="gateway", profile="default")] if had_gateway else [])
        update_receipt.begin_update_receipt()

        def verify():
            fleet._verify_fleet_after_update(restart, _pre_update_plan=plan,
                                            _windows_gateway_resume=None, update_complete=True)

        if had_gateway:
            with pytest.raises(SystemExit) as failure:
                verify()
            assert failure.value.code == 1
            assert events[0] == "settle"
            assert events.count("probe") > 1
            assert "returned no rows" in capsys.readouterr().out
        else:
            verify()
            assert events == ["probe"]
        assert restart.incomplete is had_gateway
        receipt = json.loads((update_cmd.get_hermes_home() / "logs/update_receipts/latest.json").read_text())
        assert receipt["outcome"] == ("partial" if had_gateway else "success")



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
