"""Service replacements must never enter the updater's later manual-stop sweep."""

import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_cli.gateway as gateway
from hermes_cli import main as hermes_main
from hermes_cli import update_cmd_fleet as fleet


@pytest.mark.parametrize(
    ("snapshot", "mapped"),
    [("running", False), ("running", True), ("empty", False), ("unavailable", False)],
    ids=["unmapped-manual", "profile-manual", "initially-empty", "inventory-failed"],
)
def test_service_replacement_is_not_stopped_as_manual(
    monkeypatch, tmp_path, snapshot, mapped
):
    old_service, old_manual, replacement = 1101, 1102, 2201
    initial = [old_service, old_manual] if snapshot == "running" else []
    current = [replacement] + ([old_manual] if snapshot == "running" else [])
    services_restarted = False

    def discover(**kwargs):
        if not services_restarted:
            if snapshot == "unavailable":
                raise OSError("pre-restart inventory unavailable")
            return initial
        return [pid for pid in current if pid not in kwargs.get("exclude_pids", ())]

    def restart_services(restarted, *_args):
        nonlocal services_restarted
        if not services_restarted:
            restarted.append("test-gateway-service")
            services_restarted = True

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr(gateway, "_get_restart_exit_wait_budget", lambda: 45)
    monkeypatch.setattr(gateway, "find_gateway_pids", discover)
    # Simulate a transient ownership lookup miss after the manager spawned its replacement.
    monkeypatch.setattr(gateway, "_get_service_pids", lambda **kwargs: set())
    monkeypatch.setattr(
        gateway,
        "find_profile_gateway_processes",
        lambda **kwargs: [
            SimpleNamespace(pid=pid, profile=f"profile-{pid}") for pid in current
        ] if mapped else [],
    )
    arm_restart = Mock(return_value="detached")
    drain = Mock(return_value=False)
    signal_process = Mock()
    survivor_sweep = Mock()
    receipt = Mock()
    monkeypatch.setattr(gateway, "_prepare_profile_gateway_update_restart", arm_restart)
    monkeypatch.setattr(gateway, "_wait_for_gateway_exit", lambda **kwargs: None)
    monkeypatch.setattr(fleet, "_drain_or_signal_gateway_for_update", drain)
    monkeypatch.setattr(fleet.os, "kill", signal_process)
    # Stub both native service boundaries without changing the host OS or calling a manager.
    monkeypatch.setattr(fleet, "_restart_systemd_gateway_units", restart_services)
    monkeypatch.setattr(fleet, "_restart_macos_launchd_gateways", restart_services)
    monkeypatch.setattr(fleet, "_force_kill_stuck_gateways", survivor_sweep)
    monkeypatch.setattr(fleet._GatewayRestartOutcome, "record_receipt", receipt)
    monkeypatch.setattr(
        fleet,
        "_recover_after_restart_phase_abort",
        lambda *args, **kwargs: pytest.fail("unexpected restart-phase recovery"),
    )

    outcome = fleet._restart_gateway_fleet_after_update(None, gateway_mode=False)

    expected_stops = {old_manual} if snapshot == "running" else set()
    assert outcome.pre_restart_gateway_pids == (
        None if snapshot == "unavailable" else initial
    )
    assert outcome.restarted_services == ["test-gateway-service"]
    assert outcome.killed_pids == expected_stops
    assert outcome.stopped_unmapped_pids == (set() if mapped else expected_stops)
    assert outcome.incomplete is (snapshot == "unavailable")
    assert outcome.phase_errors == []
    if expected_stops:
        signal_process.assert_called_once_with(old_manual, signal.SIGTERM)
    else:
        signal_process.assert_not_called()
    if mapped:
        arm_restart.assert_called_once_with(f"profile-{old_manual}", old_manual)
        drain.assert_called_once_with(old_manual, 45.0, f"profile-{old_manual}")
        assert outcome.relaunched_profiles == [f"profile-{old_manual}"]
    else:
        arm_restart.assert_not_called()
        drain.assert_not_called()
        assert outcome.relaunched_profiles == []
    survivor_sweep.assert_called_once_with(expected_stops)
    receipt.assert_called_once_with()
