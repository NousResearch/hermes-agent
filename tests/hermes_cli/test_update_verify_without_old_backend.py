"""The post-update fleet check for a user with no old ``hermes serve`` / ``hermes dashboard`` backend.

Pins main's behavior: the check prints the same text, starts no process and opens no connection.
Holds on main and with the old-backend notice (which only looks further when the update's own
inventory saw such a backend).
"""
import sys

import pytest

from hermes_cli import update_cmd, update_cmd_fleet, update_cmd_fleet_verify as fleet_verify, update_cmd_maint
from hermes_cli import update_receipt
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

_WATCHED = ("subprocess.Popen", "os.posix_spawn", "os.fork", "os.exec", "os.system", "socket.connect")
_events: list[tuple] = []
_watching = [False]


def _hook(event, args):
    if _watching[0] and event in _WATCHED:
        _events.append((event, args[:2]))


sys.addaudithook(_hook)  # audit hooks cannot be removed; it records only while _watching is set


def _gateway(pid):
    return RuntimeRecord(kind="gateway", profile="default", pid=pid, supervisor="systemd")


def _desktop_serve(pid):
    return RuntimeRecord(kind="serve", profile="default", pid=pid, supervisor="desktop")


def _sibling_serve(pid):
    return RuntimeRecord(kind="serve", profile="work", pid=pid, supervisor="manual-serve")


@pytest.mark.parametrize("plan", [
    None,
    UpdatePlan(runtimes=[]),
    UpdatePlan(runtimes=[_gateway(1234)]),
    # Desktop recycles its own backend; it is never an old backend left beside the gateway.
    UpdatePlan(runtimes=[_gateway(1234), _desktop_serve(4321)]),
    # Another profile's backend is that profile's notice, not this one's.
    UpdatePlan(profiles=["default", "work"], runtimes=[_gateway(1234), _sibling_serve(4321)]),
], ids=["no-plan", "nothing-running", "gateway-only", "gateway-and-desktop-backend", "sibling-profile-backend"])
def test_the_fleet_check_prints_main_s_text_and_starts_nothing(plan, monkeypatch, capsys):
    restart = update_cmd_fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[1234],
        restarted_services=["hermes-gateway"], failed_or_stale_units=[],
        relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )
    monkeypatch.setattr(fleet_verify, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(update_cmd_maint, "_refresh_dashboard_after_update", lambda **kwargs: set())
    monkeypatch.setattr(update_receipt, "_code_identity", lambda **kwargs: {})
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
    snapshot = [{"profile": "default", "pid": 1234, "state": "current"}]
    monkeypatch.setattr(fleet_verify, "_collect_fleet_snapshot", lambda outcome, rows_expected: snapshot)

    del _events[:]
    _watching[0] = True
    try:
        with update_receipt.update_receipt_scope():
            update_receipt.begin_update_receipt()
            fleet_verify._verify_fleet_after_update(
                restart, _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True)
    finally:
        _watching[0] = False

    assert _events == []
    out = capsys.readouterr().out
    assert "old Hermes backend" not in out
    assert out == _MAIN_OUTPUT


# What main prints here, byte for byte (the model tip comes from the empty test home).
_MAIN_OUTPUT = (
    "\n"
    "Tip: You can now select a provider and model:\n"
    "  hermes model              # Select provider and model\n"
    "\n"
    "Fleet version check:\n"
    "  ✓ default (pid 1234) @ ? — up to date\n"
)
