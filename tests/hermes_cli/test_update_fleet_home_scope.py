"""#93349 — ``hermes update`` restarts only the gateways of the home it is updating.

``hermes-gateway*`` units and ``gateway run`` processes are host-wide namespaces shared by every
Hermes install under the account. A scratch home's update used to drain and restart the account's
real ``hermes-gateway.service`` and SIGTERM sibling installs' gateways because they were listed,
not because they ran the updated code.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import dashboard_procs

FOREIGN_HOME = "/srv/other-account-home/.hermes"


@pytest.fixture
def own_home(monkeypatch, tmp_path):
    home = tmp_path / "homeA" / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The plan inventory is this install only; the fixture home has no profiles dir.
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", home)])
    return home


def _pid_homes(monkeypatch, mapping: dict):
    monkeypatch.setattr(dashboard_procs, "_hermes_home_for_pid", lambda pid: mapping.get(pid))


def test_systemd_unit_of_another_home_is_left_alone(monkeypatch, own_home):
    """user/hermes-gateway runs on another HERMES_HOME → no drain, no restart, not a failure;
    the same unit running on the updating home is still restarted (control)."""
    homes = {4242: FOREIGN_HOME, 4343: str(own_home)}
    _pid_homes(monkeypatch, homes)
    main_pid = {"value": 4242}
    write_verbs: list[list[str]] = []

    def fake_systemctl(cmd, *, timeout):
        if "is-active" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="active\n", stderr="")
        if "show" in cmd and "--property=MainPID" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{main_pid['value']}\n", stderr="")
        if "show" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        write_verbs.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(fleet, "_systemctl", fake_systemctl)
    monkeypatch.setattr(fleet, "_repair_unit_without_fatal_exit_park", lambda *a, **k: None)
    drained: list[int] = []
    monkeypatch.setattr(fleet, "_drain_or_signal_gateway_for_update", lambda pid, *a, **k: drained.append(pid) or True)
    monkeypatch.setattr(fleet, "_wait_for_service_active", lambda *a, **k: True)

    restarted: list[str] = []
    failed: list[str] = []
    fleet._restart_one_systemd_gateway_unit(
        "hermes-gateway", scope="user", scope_cmd=["systemctl", "--user"], drain_budget=5.0,
        _manage_cmd_cache={}, restarted_services=restarted, failed_or_stale_units=failed,
    )
    assert drained == [] and write_verbs == [] and restarted == [] and failed == []

    main_pid["value"] = 4343  # control: same unit name, this update's home
    fleet._restart_one_systemd_gateway_unit(
        "hermes-gateway", scope="user", scope_cmd=["systemctl", "--user"], drain_budget=5.0,
        _manage_cmd_cache={}, restarted_services=restarted, failed_or_stale_units=failed,
    )
    assert drained == [4343] and restarted == ["hermes-gateway"] and failed == []


def test_manual_gateway_of_another_home_is_not_stopped(monkeypatch, own_home):
    """Of two ``gateway run`` processes on the host, only the one on the updating home is SIGTERMed;
    a process whose home cannot be read is spared as well."""
    _pid_homes(monkeypatch, {111: str(own_home), 222: FOREIGN_HOME, 333: None})
    monkeypatch.setattr("hermes_cli.gateway._get_service_pids", lambda **k: set())
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **k: [111, 222, 333])
    monkeypatch.setattr("hermes_cli.gateway.find_profile_gateway_processes", lambda **k: [])
    monkeypatch.setattr("hermes_cli.gateway._wait_for_gateway_exit", lambda **k: None)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

    out = fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )
    fleet._restart_manual_gateways(out, 5.0)
    assert killed == [(111, signal.SIGTERM)]
    assert out.killed_pids == {111}
