"""A unit the restart phase recorded failed, but whose own gateway verification proves current.

A ``systemctl`` restart call can time out while systemd still finishes the job (a unit whose stop
takes ~3 minutes). The restart phase then records the unit failed, and the run used to print
"Update incomplete — some units were not restarted" and owe a ``gateway_restart`` follow-up even
after the fleet probe showed the unit's fresh gateway serving the checkout. Only the unit's OWN
process (its MainPID, not running when the restart began) may settle it; a current gateway of
another profile, or the pre-restart process, must not.
"""

from __future__ import annotations

import json
import types

import pytest

from hermes_cli.update_inventory import RuntimeRecord

pytestmark = pytest.mark.platforms("linux")

OLD_PID, NEW_PID, OTHER_PID = 500, 626640, 777


@pytest.mark.parametrize(
    ("main_pid", "fleet_rows", "settled"),
    [
        (NEW_PID, [{"profile": "default", "pid": NEW_PID}], True),
        # The unit is dead; an unrelated profile's current gateway is no evidence for it.
        (0, [{"profile": "coder", "pid": OTHER_PID}], False),
        # The unit still runs the process that predates the restart phase.
        (OLD_PID, [{"profile": "default", "pid": OLD_PID}], False),
    ],
    ids=["own-fresh-mainpid", "other-profile-row", "pre-restart-pid"],
)
def test_failed_unit_settles_only_on_its_own_fresh_current_gateway(
        monkeypatch, capsys, main_pid, fleet_rows, settled):
    out, receipt, cleared = _update_with_failed_unit(monkeypatch, capsys, main_pid, fleet_rows)
    followups = [f["step"] for f in receipt.get("followups", [])]
    if settled:
        assert "Update incomplete" not in out
        assert followups == []
        assert receipt["gateway_restart"]["incomplete"] is False
        assert receipt["gateway_restart"]["failed_units"] == []
        assert cleared == [True]
    else:
        assert "Update incomplete — some units were not restarted" in out
        assert "    - hermes-gateway" in out
        assert followups == ["gateway_restart"]
        assert receipt["gateway_restart"]["incomplete"] is True
        assert cleared == []


def test_settled_unit_keeps_the_receipt_incomplete_while_a_dashboard_is_down(monkeypatch, capsys):
    """Crediting the last failed unit must not save a complete restart while verification still owes
    a ``gateway_restart`` follow-up (a dashboard it stopped did not come back)."""
    out, receipt, cleared = _update_with_failed_unit(
        monkeypatch, capsys, NEW_PID, [{"profile": "default", "pid": NEW_PID}], dashboards_down={4242})
    assert "Update incomplete — some units were not restarted" not in out
    assert [f["step"] for f in receipt.get("followups", [])] == ["gateway_restart"]
    assert receipt["gateway_restart"]["failed_units"] == []
    assert receipt["gateway_restart"]["incomplete"] is True
    assert cleared == []


def _update_with_failed_unit(monkeypatch, capsys, main_pid, fleet_rows, dashboards_down=None):
    from hermes_cli import gateway, update_cmd, update_cmd_fleet as fleet, update_cmd_fleet_verify as verify
    from hermes_cli import update_receipt

    # Restart phase: the unit's restart call timed out, so it lands in the failed list.
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **kw: [OLD_PID])
    monkeypatch.setattr(fleet, "_scoped_manual_gateway_pids", lambda pids, **kw: list(pids))
    monkeypatch.setattr(fleet, "_gateway_drain_budget", lambda: 1.0)
    monkeypatch.setattr(fleet, "_restart_systemd_gateway_units",
                        lambda restarted, failed, *args: failed.append("hermes-gateway"))
    monkeypatch.setattr(fleet, "_restart_manual_gateways", lambda out, budget: None)
    monkeypatch.setattr(fleet, "_force_kill_stuck_gateways", lambda killed: None)
    monkeypatch.setattr(fleet, "_restart_identity_sha", lambda: "new")

    # systemd: the unit exists in user scope only, with *main_pid* as its MainPID.
    def systemctl(cmd, timeout=None):
        user = "--user" in cmd
        stdout = f"LoadState=loaded\nMainPID={main_pid}\n" if user else "LoadState=not-found\nMainPID=0\n"
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(fleet, "_systemctl", systemctl)

    # Verification: every row the probe returns is stamped with the checkout's code.
    rows = [{**row, "code_sha": "new", "code_version": "1", "state": "current"} for row in fleet_rows]
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **kw: [dict(r) for r in rows])
    monkeypatch.setattr(verify, "_time", types.SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda s: None))
    monkeypatch.setattr(verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update", lambda **kw: dashboards_down)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
    cleared = []
    monkeypatch.setattr(fleet, "_clear_fleet_restart_pending_marker", lambda: cleared.append(True))

    plan = types.SimpleNamespace(runtimes=[
        RuntimeRecord(kind="gateway", profile="default", pid=OLD_PID, restart_via="systemd")])
    update_receipt.begin_update_receipt()
    restart = fleet._restart_gateway_fleet_after_update(plan, gateway_mode=False)
    verify._verify_fleet_after_update(restart, _pre_update_plan=plan, _windows_gateway_resume=None,
                                      update_complete=True)

    receipt = json.loads((update_cmd.get_hermes_home() / "logs/update_receipts/latest.json").read_text())
    return capsys.readouterr().out, receipt, cleared
