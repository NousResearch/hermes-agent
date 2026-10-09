"""Desktop update custody when its serve/dashboard backend is a systemd MainPID."""

from __future__ import annotations

import pytest

from hermes_cli import main_dashboard, update_cmd_posix_pause as pause
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan


_UNIT = "hermes-desktop.service"
_CGROUP = f"/system.slice/{_UNIT}"


def _systemd_runtime(pid: int = 4321) -> RuntimeRecord:
    return RuntimeRecord(
        kind="serve", profile="default", pid=pid, supervisor="systemd", restart_via="systemd",
        detail={"systemd_scope": "system", "systemd_unit": _UNIT, "systemd_cgroup": _CGROUP},
    )


def _verified_mainpid(monkeypatch):
    monkeypatch.setattr(main_dashboard, "_get_pid_cgroup_path", lambda _pid: _CGROUP)
    monkeypatch.setattr(main_dashboard, "_get_systemd_service_for_pid", lambda _pid: _UNIT)
    monkeypatch.setattr(main_dashboard, "_unit_main_pid_is", lambda *_args: True)


@pytest.mark.platforms("linux")
def test_desktop_update_handoff_moves_updater_before_backend_restart(monkeypatch):
    from hermes_cli import update_cmd_fleet

    plan = UpdatePlan(runtimes=[_systemd_runtime()])
    monkeypatch.setattr(pause, "_pid_cgroup", lambda: _CGROUP)
    monkeypatch.setattr("hermes_cli.gateway._is_pid_ancestor_of_current_process", lambda pid: pid == 4321)
    _verified_mainpid(monkeypatch)
    monkeypatch.setattr(update_cmd_fleet, "_resolve_manage_cmd", lambda *args: ["systemctl"])
    moved = []
    monkeypatch.setattr(pause, "_escape_cgroup", lambda unit: moved.append(unit) or True)

    assert pause.prepare_systemd_serve_update_handoff(plan) is None
    assert moved == [{"scope": "system", "unit": _UNIT, "pid": 4321, "cgroup": _CGROUP}]


@pytest.mark.platforms("linux")
def test_desktop_update_is_refused_when_handoff_or_restart_authority_is_missing(monkeypatch):
    from hermes_cli import update_cmd_fleet

    plan = UpdatePlan(runtimes=[_systemd_runtime()])
    monkeypatch.setattr(pause, "_pid_cgroup", lambda: _CGROUP)
    monkeypatch.setattr("hermes_cli.gateway._is_pid_ancestor_of_current_process", lambda _pid: True)
    _verified_mainpid(monkeypatch)
    moved = []
    monkeypatch.setattr(pause, "_escape_cgroup", lambda unit: moved.append(unit) or False)
    monkeypatch.setattr(update_cmd_fleet, "_resolve_manage_cmd", lambda *args: None)
    denied = pause.prepare_systemd_serve_update_handoff(plan)
    assert "restart permission" in denied
    assert moved == []

    monkeypatch.setattr(update_cmd_fleet, "_resolve_manage_cmd", lambda *args: ["systemctl"])
    refused = pause.prepare_systemd_serve_update_handoff(plan)
    assert "could not move out" in refused
    assert moved


@pytest.mark.platforms("linux")
def test_desktop_action_refuses_when_ancestor_is_not_verified_unit_mainpid(monkeypatch):
    runtime = RuntimeRecord(kind="serve", profile="default", pid=4321, supervisor="desktop")
    plan = UpdatePlan(runtimes=[runtime])
    monkeypatch.setenv("HERMES_ACTION_ID", "a" * 32)
    monkeypatch.setattr(pause, "_pid_cgroup", lambda: _CGROUP)
    monkeypatch.setattr("hermes_cli.gateway._is_pid_ancestor_of_current_process", lambda _pid: True)
    monkeypatch.setattr(main_dashboard, "_get_pid_cgroup_path", lambda _pid: _CGROUP)
    monkeypatch.setattr(main_dashboard, "_get_systemd_service_for_pid", lambda _pid: None)
    moved = []
    monkeypatch.setattr(pause, "_escape_cgroup", lambda unit: moved.append(unit) or True)

    refusal = pause.prepare_systemd_serve_update_handoff(plan)

    assert "MainPID" in refusal
    assert moved == []


@pytest.mark.platforms("linux")
def test_inherited_service_cgroup_without_mainpid_ownership_is_not_moved(monkeypatch):
    runtime = RuntimeRecord(kind="serve", profile="default", pid=4321, supervisor="manual-serve")
    plan = UpdatePlan(runtimes=[runtime])
    monkeypatch.setattr(pause, "_pid_cgroup", lambda: _CGROUP)
    monkeypatch.setattr("hermes_cli.gateway._is_pid_ancestor_of_current_process", lambda _pid: True)
    monkeypatch.setattr(main_dashboard, "_get_pid_cgroup_path", lambda _pid: _CGROUP)
    monkeypatch.setattr(main_dashboard, "_get_systemd_service_for_pid", lambda _pid: None)
    moved = []
    monkeypatch.setattr(pause, "_escape_cgroup", lambda unit: moved.append(unit) or True)

    assert pause.prepare_systemd_serve_update_handoff(plan) is None
    assert moved == []
