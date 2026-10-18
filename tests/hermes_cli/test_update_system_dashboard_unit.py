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
    assert kills == [] and not result.get("unrecovered")
    out = capsys.readouterr().out
    assert "polkit" not in out and "failed to restart" not in out and "Restarting managed dashboard" not in out


def test_dashboard_pass_still_restarts_the_unit_when_the_fleet_loop_never_saw_it(monkeypatch, capsys):
    """Negative control: without the fleet loop's verdict the managed restart still runs."""
    calls: list = []
    _systemd_owned_dashboard(monkeypatch, calls)
    dashboard_procs._kill_stale_dashboard_processes(restart_managed=True, already_restarted_units={"hermes-gateway"})
    assert ["systemctl", "restart", UNIT] in calls
