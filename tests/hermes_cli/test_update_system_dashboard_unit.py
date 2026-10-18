"""``hermes update`` on a system-scope ``hermes-dashboard`` unit without root (#141021).

The fleet loop correctly declines to restart the unit (no non-interactive privilege path) and names
the manual command. Three things then went wrong after it:

* the post-update dashboard pass retried the same unit with a plain ``systemctl restart`` and
  ``sudo -n``, dumping a polkit/tty error under identical advice;
* the inventory labelled the unit's MainPID ``manual-serve``, so the user was told to "ask the
  owner to relaunch `hermes dashboard`" for a process systemd owns;
* the owed follow-up always said "gateways may still run pre-update code … `hermes gateway
  restart`", even with every gateway verified current.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import dashboard_procs, main_dashboard, update_cmd, update_cmd_fleet
from hermes_cli import update_cmd_fleet_verify as fleet_verify
from hermes_cli import update_inventory, update_receipt
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

DASH_PID = 961386
UNIT = "hermes-dashboard.service"
CGROUP = f"/system.slice/{UNIT}"


@pytest.fixture(autouse=True)
def _no_real_launchd_jobs():
    with patch.object(main_dashboard, "_loaded_launchd_backend_jobs", return_value=[]):
        yield


def _systemd_owned_dashboard(monkeypatch, systemctl_calls: list):
    """Host shape from the report: PID 961386 is MainPID of the SYSTEM hermes-dashboard unit; any
    manage verb would hit polkit (no TTY) and ``sudo -n`` would need a password."""
    monkeypatch.setattr(main_dashboard, "_get_pid_cgroup_path", lambda pid: CGROUP if pid == DASH_PID else None)
    monkeypatch.setattr(main_dashboard, "_get_systemd_service_for_pid", lambda pid: UNIT if pid == DASH_PID else None)

    def run_probe(cmd, *, timeout):
        systemctl_calls.append(list(cmd))
        if "list-unit-files" in cmd:
            listed = "--user" not in cmd
            return SimpleNamespace(returncode=0 if listed else 1, stdout=f"{UNIT} enabled enabled\n" if listed else "", stderr="")
        if "is-active" in cmd:
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        if "is-enabled" in cmd:
            return SimpleNamespace(returncode=0, stdout="enabled\n", stderr="")
        if cmd[0] == "sudo":
            return SimpleNamespace(returncode=1, stdout="", stderr="sudo: a password is required")
        return SimpleNamespace(returncode=1, stdout="", stderr="Interactive authentication required.")

    monkeypatch.setattr(main_dashboard, "_run_probe", run_probe)
    monkeypatch.setattr(main_dashboard, "_find_stale_dashboard_pids", lambda **k: [DASH_PID])


def test_dashboard_pass_skips_a_unit_the_fleet_loop_already_declined(monkeypatch, capsys):
    calls: list = []
    _systemd_owned_dashboard(monkeypatch, calls)
    kills: list = []
    monkeypatch.setattr(dashboard_procs, "_kill_pids_posix", lambda pids, killed, failed: kills.extend(pids))

    result = dashboard_procs._kill_stale_dashboard_processes(
        restart_managed=True, already_restarted_units={"hermes-gateway", "hermes-dashboard"})

    assert not [c for c in calls if "restart" in c], "the declined unit was retried"
    assert not result.get("unrecovered")
    out = capsys.readouterr().out
    assert "Interactive authentication required" not in out and "a password is required" not in out
    assert "failed to restart" not in out and "Restarting managed dashboard" not in out


def test_dashboard_pass_still_restarts_the_unit_when_the_fleet_loop_never_saw_it(monkeypatch, capsys):
    """Negative control: without the fleet loop's verdict the managed restart still runs."""
    calls: list = []
    _systemd_owned_dashboard(monkeypatch, calls)
    dashboard_procs._kill_stale_dashboard_processes(restart_managed=True, already_restarted_units={"hermes-gateway"})
    assert ["systemctl", "restart", UNIT] in calls


def test_systemd_owned_ledger_dashboard_is_classified_systemd(monkeypatch):
    calls: list = []
    _systemd_owned_dashboard(monkeypatch, calls)
    entry = {"pid": DASH_PID, "create_time": 1000.0, "purpose": "dashboard", "profile": "",
             "argv": "hermes dashboard --host 0.0.0.0 --port 9119 --no-open", "host": "0.0.0.0", "port": 9119}
    fake_pi = SimpleNamespace(ledger_entries=lambda **k: [entry], spawner_is_dead=lambda e: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", fake_pi)
    plan = UpdatePlan()
    update_inventory._collect_ledger_runtimes(plan, set())
    (row,) = plan.runtimes
    assert (row.supervisor, row.restart_via) == ("systemd", "systemd")
    assert (row.detail["systemd_unit"], row.detail["systemd_scope"]) == (UNIT, "system")


def test_hand_started_dashboard_in_a_foreign_cgroup_stays_manual(monkeypatch):
    """Started from a shell inside cron.service: the cgroup names a unit, but its MainPID is another
    process, so the real ownership rule keeps the row manual-serve (its respawn reminder applies)."""
    monkeypatch.setattr(main_dashboard, "_pid_unified_cgroup_entries", lambda pid: iter(["/system.slice/cron.service"]))
    monkeypatch.setattr(main_dashboard, "_run_probe",
                        lambda cmd, *, timeout: SimpleNamespace(returncode=0, stdout="991\n", stderr=""))
    entry = {"pid": 4321, "create_time": 1.0, "purpose": "dashboard", "argv": "hermes dashboard"}
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity",
                        SimpleNamespace(ledger_entries=lambda **k: [entry], spawner_is_dead=lambda e: None))
    plan = UpdatePlan()
    update_inventory._collect_ledger_runtimes(plan, set())
    assert plan.runtimes[0].supervisor == "manual-serve"


def test_stale_unit_dashboard_hint_names_its_unit_and_scope(monkeypatch, capsys):
    plan = UpdatePlan(runtimes=[RuntimeRecord(
        kind="dashboard", profile="default", pid=DASH_PID, supervisor="systemd", restart_via="systemd",
        detail={"create_time": 1000.0, "systemd_unit": UNIT, "systemd_scope": "system"})])
    monkeypatch.setitem(sys.modules, "hermes_cli.process_identity", SimpleNamespace(
        ledger_entries=lambda: [{"pid": DASH_PID, "purpose": "dashboard", "create_time": 1000.0}]))
    rows = update_cmd._surviving_pre_update_serve_runtimes(plan)
    update_cmd._warn_stale_serve_runtimes(rows)
    out = capsys.readouterr().out
    assert f"`sudo systemctl restart {UNIT}`" in out
    assert "Ask their owner" not in out and "hermes-serve.service" not in out


def _verify(monkeypatch, restart, *, fleet_state="current", plan=None, dashboards_down=frozenset(), survivors=()):
    refreshed: list = []
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: list(survivors))
    monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update",
                        lambda **k: refreshed.append(k) or set(dashboards_down))
    monkeypatch.setattr(update_receipt, "_code_identity", lambda **k: {})
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd_stale_survivors.signal_stale_fleet_survivors", lambda *a: None)
    monkeypatch.setattr(fleet_verify, "_collect_fleet_snapshot",
                        lambda outcome, expected: [{"profile": "default", "pid": 1563620, "state": fleet_state}])
    update_cmd_fleet._write_fleet_restart_pending_marker()
    with update_receipt.update_receipt_scope():
        update_receipt.begin_update_receipt()
        fleet_verify._verify_fleet_after_update(
            restart, _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True)
    return refreshed, update_receipt.read_latest_receipt()


def _restart(**over):
    fields = dict(incomplete=True, phase_errors=[], pre_restart_gateway_pids=[1234],
                  restarted_services=["hermes-gateway"], failed_or_stale_units=["hermes-dashboard"],
                  relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set())
    fields.update(over)
    return update_cmd_fleet._GatewayRestartOutcome(**fields)


def test_field_case_forwards_the_declined_unit_and_blames_only_the_dashboard(monkeypatch, capsys):
    """The reported run end to end through verification: gateway current, dashboard declined."""
    refreshed, receipt = _verify(monkeypatch, _restart())
    assert refreshed == [{"already_restarted_units": {"hermes-gateway", "hermes-dashboard"}}]
    (followup,) = receipt["followups"]
    assert followup["step"] == "gateway_restart"  # Desktop hand-offs parse the step id
    assert "hermes-dashboard were not restarted onto the new code" in followup["reason"]
    assert "hermes gateway restart" not in followup["reason"]
    assert "gateways may still run" not in capsys.readouterr().out
    # The obligation stays armed: a stale dashboard unit is still owed (contract unchanged).
    assert update_cmd_fleet._fleet_restart_obligation_armed() is True


@pytest.mark.parametrize("case", ["stale-gateway", "failed-gateway-unit", "phase-error"])
def test_gateway_debt_still_names_gateway_restart(monkeypatch, case):
    restart = _restart()
    state = "current"
    if case == "stale-gateway":
        state = "stale"
    elif case == "failed-gateway-unit":
        restart.failed_or_stale_units.append("hermes-gateway-work")
    else:
        restart.phase_errors.append("boom")
    _refreshed, receipt = _verify(monkeypatch, restart, fleet_state=state)
    reason = receipt["followups"][0]["reason"]
    assert "hermes gateway restart" in reason and "hermes-dashboard" in reason


def test_is_serve_unit_matches_the_fleet_loop_gate():
    assert fleet_verify._is_serve_unit("hermes-dashboard")
    assert fleet_verify._is_serve_unit("hermes-serve-work.service")
    assert fleet_verify._is_serve_unit("hermes-dashboard-Work.Two")  # the loop admits any suffix
    for name in ("hermes-gateway", "hermes-gateway-work", "hermes-dashboardd", "ai.hermes.gateway",
                 "systemd-user (listing failed)"):
        assert not fleet_verify._is_serve_unit(name)


def test_dashboard_stopped_and_not_brought_back_is_not_called_still_running(monkeypatch):
    _refreshed, receipt = _verify(monkeypatch, _restart(incomplete=False, failed_or_stale_units=[]),
                                  dashboards_down={4242})
    reason = receipt["followups"][0]["reason"]
    assert "4242 were stopped during the update and not brought back" in reason
    assert "still run pre-update code" not in reason and "hermes gateway restart" not in reason


def test_incomplete_run_never_settles_the_host_obligation_mid_verification(monkeypatch):
    """The owed-gateway check can clear the host obligation; on a run a failed dashboard unit already
    made incomplete it must not run (round-2 review): the obligation stays exactly as armed."""
    calls = []
    monkeypatch.setattr(fleet_verify, "_named_gateways_still_owed", lambda: calls.append(1) or False)
    _verify(monkeypatch, _restart())
    assert calls == []
    assert update_cmd_fleet._fleet_restart_obligation_armed() is True


def _unit_dashboard_plan():
    return UpdatePlan(runtimes=[RuntimeRecord(
        kind="dashboard", profile="default", pid=DASH_PID, supervisor="systemd", restart_via="systemd",
        detail={"create_time": 1000.0, "systemd_unit": UNIT, "systemd_scope": "system"})])


def test_unrestarted_unit_dashboard_keeps_a_startup_reminder_until_it_exits(monkeypatch, capsys):
    """The fleet obligation names only gateways, so once they are current the stale dashboard unit
    would never be mentioned again; it keeps its own reminder with the unit's command instead."""
    from hermes_cli import process_identity
    from hermes_cli.update_serve_obligations import warn_pending_manual_serves

    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: True)
    survivor = {"pid": DASH_PID, "kind": "dashboard", "profile": "default", "supervisor": "systemd",
                "systemd_unit": UNIT, "systemd_scope": "system"}
    _verify(monkeypatch, _restart(), plan=_unit_dashboard_plan(), survivors=[survivor])
    capsys.readouterr()
    warn_pending_manual_serves(startup=True, pending_manual=[])
    err = capsys.readouterr().err
    assert f"dashboard [default] pid {DASH_PID}: unit restart still pending" in err
    assert f"`sudo systemctl restart {UNIT}`" in err and "Ask its owner" not in err
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: False)
    warn_pending_manual_serves(startup=True, pending_manual=[])
    assert str(DASH_PID) not in capsys.readouterr().err


def test_unaccounted_unit_dashboard_hint_names_its_unit(capsys):
    plan = UpdatePlan(runtimes=[RuntimeRecord(
        kind="dashboard", profile="work", pid=DASH_PID, supervisor="systemd", restart_via="systemd",
        detail={"create_time": 1000.0, "systemd_unit": UNIT, "systemd_scope": "system"})])
    outcomes = update_inventory.match_runtime_outcomes(
        plan, restarted_services=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(), failed_units=[], stale_serve_pids={DASH_PID})
    assert update_inventory.report_unaccounted_runtimes(outcomes) is True
    out = capsys.readouterr().out
    assert f"sudo systemctl restart {UNIT}" in out and "hermes-serve.service" not in out
