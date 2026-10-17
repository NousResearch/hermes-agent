"""Regression: the owed follow-up must be named for the component that actually failed.

``hermes update`` collapses every post-commit verification failure into ONE
follow-up step name, ``gateway_restart`` (hermes_cli/update_cmd_fleet_verify.py).
The Desktop hand-off scripts parse that name and append a remedy:
``scripts/desktop-update/posix.sh`` adds "Run hermes gateway restart ..." for
``gateway_restart`` and nothing for any other step.

A dashboard/serve backend that the update stopped and could not bring back
(``_refresh_dashboard_after_update`` → ``unrecovered``) is NOT a gateway
failure: ``hermes gateway restart`` never touches it. Reporting it as
``gateway_restart`` sends the user after the wrong component and hides the
real remedy (``hermes dashboard --port <port>``).

The macOS boundary is mocked: these tests run on Linux and never invoke
launchctl, launchd or systemctl.
"""

from __future__ import annotations

import json
import types

import pytest

from hermes_cli.update_inventory import RuntimeRecord


def _plan(runtimes=None):
    return types.SimpleNamespace(runtimes=runtimes or [])


def _verify(monkeypatch, tmp_path, *, dashboards_down, fleet_rows, plan=None, restart=None, named_owed=False, real_reconciliation=False, survivors=()):
    """Run the real post-restart verification with every external probe mocked."""
    from hermes_cli import main, update_cmd, update_cmd_fleet as fleet, update_cmd_fleet_verify as fleet_verify
    from hermes_cli import update_receipt

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update",
                        lambda **kw: set(dashboards_down))
    def _survivors(plan):
        if survivors is None:  # the survivor probe failed: reconciliation stays fail-closed
            raise OSError("ps unavailable")
        return list(survivors)

    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", _survivors)
    if not real_reconciliation:
        monkeypatch.setattr("hermes_cli.update_inventory.report_unaccounted_runtimes", lambda rows: False)
    monkeypatch.setattr("hermes_cli.update_serve_obligations.defer_manual_serve", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.update_cmd_stale_survivors.signal_stale_fleet_survivors", lambda *a: None)
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: list(fleet_rows))
    monkeypatch.setattr(fleet_verify, "_named_gateways_still_owed", lambda: named_owed)
    cleared = []
    monkeypatch.setattr(fleet, "_clear_fleet_restart_pending_marker", lambda: cleared.append(True))
    outcome = restart if restart is not None else fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(),
    )
    update_receipt.begin_update_receipt()
    fleet_verify._verify_fleet_after_update(outcome, _pre_update_plan=plan or _plan(),
                                            _windows_gateway_resume=None, update_complete=True)
    receipt = json.loads((update_cmd.get_hermes_home() / "logs/update_receipts/latest.json").read_text())
    return receipt, outcome, cleared


_CURRENT_FLEET = [{"profile": "default", "state": "current", "code_sha": "head", "pid": 4242}]


def test_dashboard_only_failure_is_not_reported_as_a_gateway_restart(monkeypatch, tmp_path, capsys):
    """A stopped dashboard/serve backend is owed work, but NOT a gateway restart.

    The gateway fleet is proven current (every row ``state == "current"``), no unit
    failed, nothing was stopped without a successor: the only failure is the
    dashboard/serve cleanup. The follow-up must name the dashboard, so the Desktop
    hands the user a remedy that can actually repair it.
    """
    receipt, outcome, cleared = _verify(monkeypatch, tmp_path, dashboards_down=[9119],
                                         fleet_rows=_CURRENT_FLEET)
    assert receipt["outcome"] == "success"  # contract C3: the code committed
    assert outcome.incomplete is True  # a promised restart did not happen
    assert [f["step"] for f in receipt["followups"]] == ["dashboard_restart"]
    printed = capsys.readouterr().out
    assert "Update follow-up 'dashboard_restart' did not finish" in printed
    assert "hermes dashboard" in receipt["followups"][0]["reason"]
    assert cleared == [True]  # dashboard debt must not leave a gateway startup warning


def test_a_gateway_failure_stays_a_gateway_restart(monkeypatch, tmp_path):
    """Guard against over-fixing: a genuine gateway miss keeps its name and remedy."""
    from hermes_cli import update_cmd_fleet as fleet

    stale = fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[101], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(),
    )
    receipt, _outcome, _cleared = _verify(
        monkeypatch, tmp_path, dashboards_down=[],
        fleet_rows=[{"profile": "default", "state": "stale", "code_sha": "old", "pid": 101}],
        restart=stale,
    )
    assert [f["step"] for f in receipt["followups"]] == ["gateway_restart"]


def test_simultaneous_failures_keep_both_followups(monkeypatch, tmp_path):
    """When both fail neither component's outstanding restart may disappear."""
    from hermes_cli import update_cmd_fleet as fleet

    both = fleet._GatewayRestartOutcome(
        incomplete=True, phase_errors=["boom"], pre_restart_gateway_pids=[101], restarted_services=[],
        failed_or_stale_units=["ai.hermes.gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(),
    )
    receipt, _outcome, _cleared = _verify(monkeypatch, tmp_path, dashboards_down=[9119],
                                          fleet_rows=_CURRENT_FLEET, restart=both)
    assert [f["step"] for f in receipt["followups"]] == ["dashboard_restart", "gateway_restart"]


def test_a_clean_run_records_no_followup(monkeypatch, tmp_path):
    """Nothing failed: no follow-up, and the fleet obligation is cleared."""
    receipt, outcome, cleared = _verify(monkeypatch, tmp_path, dashboards_down=[],
                                        fleet_rows=_CURRENT_FLEET,
                                        plan=_plan([RuntimeRecord(kind="gateway", profile="default")]))
    assert receipt.get("followups", []) == []
    assert outcome.incomplete is False
    assert cleared == [True]


def test_dashboard_failure_does_not_discharge_a_standing_gateway_obligation(monkeypatch, tmp_path):
    receipt, _outcome, cleared = _verify(monkeypatch, tmp_path, dashboards_down=[9119],
                                       fleet_rows=_CURRENT_FLEET, named_owed=True)
    assert [followup["step"] for followup in receipt["followups"]] == ["dashboard_restart", "gateway_restart"]
    assert cleared == []


@pytest.mark.parametrize("failed_respawn", [True, False])
def test_real_dashboard_reconciliation_never_invents_gateway_debt(monkeypatch, tmp_path, failed_respawn):
    plan = _plan([RuntimeRecord(kind="dashboard", profile="default", pid=9119, supervisor="manual-serve",
                                restart_via="respawn-argv")])
    receipt, _outcome, cleared = _verify(
        monkeypatch, tmp_path, dashboards_down=[9119] if failed_respawn else [],
        fleet_rows=_CURRENT_FLEET, plan=plan, real_reconciliation=True,
        survivors=[] if failed_respawn else [{"pid": 9119}],
    )
    assert [row["step"] for row in receipt["followups"]] == ["dashboard_restart"]
    assert receipt["runtime_outcomes"][0]["outcome"] == ("failed" if failed_respawn else "unaccounted")
    assert cleared == [True]


def test_unaccounted_serve_without_a_pid_is_still_owed(monkeypatch, tmp_path):
    """A serve reconciliation cannot account for is owed even when it has no PID to name: it
    must not fall between the gateway and dashboard follow-ups and leave the receipt silent."""
    plan = _plan([RuntimeRecord(kind="serve", profile="default", pid=None, supervisor="manual-serve",
                                restart_via="respawn-argv")])
    receipt, outcome, _cleared = _verify(
        monkeypatch, tmp_path, dashboards_down=[], fleet_rows=_CURRENT_FLEET, plan=plan,
        real_reconciliation=True, survivors=None,
    )
    assert receipt["runtime_outcomes"][0]["outcome"] == "unaccounted"
    assert outcome.incomplete is True
    assert [row["step"] for row in receipt["followups"]] == ["dashboard_restart"]
