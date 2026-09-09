"""A macOS launchd restart failure (e.g. hermes_cli.gateway no longer importing under a checkout
replaced mid-restart, per _surviving_gateway_pids_after_failed_restart's own docstring) must be
recorded as an incomplete fleet restart — not silently swallowed into a clean success receipt.

_restart_gateway_fleet_after_update used to wrap the call to _restart_macos_launchd_gateways in a
bare suppress(FileNotFoundError, ImportError): on those two exception types, out.failed_or_stale_units
never got a "launchd" entry, so out.incomplete stayed False and the update reported success even
though the whole launchd fleet was left on pre-update code. Its own catch-up sibling,
_run_pending_fleet_restart, already recorded exactly this failure correctly.
"""

from __future__ import annotations

import hermes_cli.update_cmd_fleet as update_cmd_fleet
from hermes_cli.update_inventory import UpdatePlan


def _stub_gateway_helpers(monkeypatch, *, is_macos: bool):
    import hermes_cli.gateway as hermes_gateway

    monkeypatch.setattr(hermes_gateway, "is_macos", lambda: is_macos)
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda **k: [])
    monkeypatch.setattr(hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: [])
    monkeypatch.setattr(hermes_gateway, "_prepare_profile_gateway_update_restart", lambda *a, **k: None)
    monkeypatch.setattr(hermes_gateway, "_get_service_pids", lambda *a, **k: [])
    monkeypatch.setattr(hermes_gateway, "_wait_for_gateway_exit", lambda *a, **k: None)
    monkeypatch.setattr(hermes_gateway, "_get_restart_exit_wait_budget", lambda: 45.0)


def _stub_restart_phases(monkeypatch):
    import hermes_cli.main as hermes_main

    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr(update_cmd_fleet, "_restart_systemd_gateway_units", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd_fleet, "_restart_manual_gateways", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd_fleet, "_force_kill_stuck_gateways", lambda *a, **k: None)


def test_launchd_import_error_marks_fleet_restart_incomplete(monkeypatch):
    _stub_gateway_helpers(monkeypatch, is_macos=True)
    _stub_restart_phases(monkeypatch)

    def _raise_import_error(*a, **k):
        raise ImportError("hermes_cli.gateway")

    monkeypatch.setattr(update_cmd_fleet, "_restart_macos_launchd_gateways", _raise_import_error)

    out = update_cmd_fleet._restart_gateway_fleet_after_update(UpdatePlan(), gateway_mode=False)

    assert out.failed_or_stale_units == ["launchd"]
    assert out.incomplete is True


def test_launchd_file_not_found_marks_fleet_restart_incomplete(monkeypatch):
    """launchctl missing from PATH mid-restart is the other exception type the old bare
    suppress() dropped silently."""
    _stub_gateway_helpers(monkeypatch, is_macos=True)
    _stub_restart_phases(monkeypatch)

    def _raise_file_not_found(*a, **k):
        raise FileNotFoundError("launchctl")

    monkeypatch.setattr(update_cmd_fleet, "_restart_macos_launchd_gateways", _raise_file_not_found)

    out = update_cmd_fleet._restart_gateway_fleet_after_update(UpdatePlan(), gateway_mode=False)

    assert out.failed_or_stale_units == ["launchd"]
    assert out.incomplete is True


def test_launchd_success_leaves_fleet_restart_complete(monkeypatch):
    """Control: a launchd phase that actually restarts everything must NOT be marked incomplete —
    guards against a fix that always appends "launchd" regardless of outcome."""
    _stub_gateway_helpers(monkeypatch, is_macos=True)
    _stub_restart_phases(monkeypatch)

    def _succeed(restarted_services, failed_or_stale_units, drain_budget):
        restarted_services.append("ai.hermes.gateway")

    monkeypatch.setattr(update_cmd_fleet, "_restart_macos_launchd_gateways", _succeed)

    out = update_cmd_fleet._restart_gateway_fleet_after_update(UpdatePlan(), gateway_mode=False)

    assert out.failed_or_stale_units == []
    assert out.incomplete is False
