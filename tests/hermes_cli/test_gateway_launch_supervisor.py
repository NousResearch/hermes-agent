"""Local readers of a gateway's ``identify`` answer see what launched it, not the protocol's "none".

The session runtime protocol reports a manual or Desktop-spawned gateway as ``supervisor: "none"``.
``hermes gateway restart`` read that as an external supervisor: it SIGUSR1'd an auto-started
gateway, waited for a relaunch that never came, and exited 1 with no gateway running (#134889).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import control_socket
from gateway import status as gateway_status
from hermes_cli import gateway as gw
from hermes_cli import gateway_supervised_restart as supervised

PLAIN_ARGV = ["/usr/bin/python", "-m", "hermes_cli.main", "gateway", "run"]
SUPERVISED_ARGV = [*PLAIN_ARGV, "--external-supervisor"]
MANUAL = {"pid": 4321, "supervisor": "none", "launch_supervisor": "manual"}


@pytest.mark.parametrize("identity, expected", [
    ({"supervisor": "none", "launch_supervisor": "manual"}, "manual"),
    ({"supervisor": "none", "launch_supervisor": "desktop"}, "desktop"),
    ({"supervisor": "systemd", "launch_supervisor": "systemd"}, "systemd"),
    ({"supervisor": "manual"}, "manual"),       # bare control socket, no runtime mapping
    ({"supervisor": "launchd"}, "launchd"),     # a gateway from before launch_supervisor
    ({"supervisor": "none"}, None),             # ...that mapped manual/desktop: unknown
    ({}, None), ({"supervisor": ""}, None), ({"launch_supervisor": 7, "supervisor": "none"}, None),
])
def test_launch_supervisor(identity, expected):
    assert control_socket.launch_supervisor(identity) == expected


def test_identify_payload_carries_the_launch_supervisor(monkeypatch):
    monkeypatch.setattr(control_socket, "_detect_supervisor", lambda: "desktop")
    payload = control_socket.build_identify_payload()
    assert payload["supervisor"] == payload["launch_supervisor"] == "desktop"


@pytest.mark.parametrize("identity, argv, external", [
    (MANUAL, PLAIN_ARGV, False),
    ({**MANUAL, "launch_supervisor": "desktop"}, PLAIN_ARGV, True),
    ({**MANUAL, "launch_supervisor": "external"}, PLAIN_ARGV, True),
    ({"pid": 4321, "supervisor": "none"}, PLAIN_ARGV, False),
    ({"pid": 4321, "supervisor": "none"}, SUPERVISED_ARGV, True),
    ({**MANUAL, "pid": 999}, PLAIN_ARGV, False),  # another process's answer proves nothing
])
def test_declares_external_supervisor(monkeypatch, tmp_path, identity, argv, external):
    monkeypatch.setattr(control_socket, "identify_gateway", lambda home, **k: identity)
    monkeypatch.setattr(gw, "_capture_gateway_argv", lambda pid: argv)
    assert supervised.gateway_declares_external_supervisor(4321, home=tmp_path) is external


def test_restart_of_a_manual_gateway_stops_it_and_runs_a_new_one(monkeypatch):
    calls = {"sigusr1": False, "stopped": False, "started": False}
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda *a, **k: 4321)
    monkeypatch.setattr(gw, "_capture_gateway_argv", lambda pid: PLAIN_ARGV)
    monkeypatch.setattr(control_socket, "identify_gateway", lambda home, **k: MANUAL)
    monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", lambda *a, **k: calls.__setitem__("sigusr1", True))
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: calls.__setitem__("stopped", True) or True)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    monkeypatch.setattr(gw, "_wait_for_api_server_port_free", lambda *a, **k: None)
    monkeypatch.setattr(gw, "run_gateway", lambda **k: calls.__setitem__("started", True))
    monkeypatch.setattr("hermes_cli.main_dashboard._get_systemd_service_for_pid", lambda pid: None)

    gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

    assert calls == {"sigusr1": False, "stopped": True, "started": True}


class _PastOwnerCheck(Exception):
    pass


@pytest.mark.parametrize("identity, refusal", [
    (MANUAL, None),
    ({**MANUAL, "launch_supervisor": "desktop"}, "desktop gateway still owns"),
    ({"pid": 4321, "supervisor": "none"}, "unknown gateway still owns"),
])
def test_profile_data_cleanup_drains_only_a_manual_gateway(monkeypatch, tmp_path, identity, refusal):
    from hermes_cli.data_cleanup_holders import _drain_manual_gateway

    home = Path(tmp_path).resolve()
    answer = {**identity, "hermes_home": str(home), "start_time": 111}
    monkeypatch.setattr(control_socket, "identify_gateway", lambda h, **k: answer)

    def pause(h):
        raise _PastOwnerCheck
    monkeypatch.setattr(control_socket, "pause_gateway_for_update", pause)
    monkeypatch.setattr(gateway_status, "get_process_start_time", lambda pid: 111)
    if refusal is None:
        with pytest.raises(_PastOwnerCheck):
            _drain_manual_gateway(home)
    else:
        with pytest.raises(RuntimeError, match=refusal):
            _drain_manual_gateway(home)


@pytest.mark.parametrize("record, expected", [
    ({**MANUAL, "launch_supervisor": "desktop"}, "desktop"),
    ({"pid": 4321, "supervisor": "none"}, "manual"),  # unknown: the outside-in classifier decides
])
def test_update_inventory_records_the_launch_supervisor(monkeypatch, tmp_path, record, expected):
    from hermes_cli import update_inventory, update_receipt

    monkeypatch.setattr(update_inventory, "_supervisor_classifier", lambda: lambda pid: "manual")
    monkeypatch.setattr(update_receipt, "_socket_identity", lambda home: (4321, record))
    monkeypatch.setattr(gw, "find_profile_gateway_processes", lambda *a, **k: [])
    plan = update_inventory.UpdatePlan()
    update_inventory._collect_gateway_runtimes(plan, [("default", tmp_path)], set())
    assert [runtime.supervisor for runtime in plan.runtimes] == [expected]
