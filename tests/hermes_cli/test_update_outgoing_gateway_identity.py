"""The outgoing gateway can be an updater ancestor absent from the cleanup scan."""

from types import SimpleNamespace

from hermes_cli import gateway, update_cmd_fleet as fleet, update_inventory
from hermes_cli import update_receipt as receipt


def test_socket_verified_outgoing_gateway_survives_cleanup_scan_exclusion(monkeypatch, tmp_path):
    outgoing_pid = 17178
    parents = {gateway.os.getpid(): outgoing_pid, outgoing_pid: 1}
    monkeypatch.setattr(gateway, "_get_parent_pid", parents.get)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "_get_service_pids", lambda **kwargs: [])
    monkeypatch.setattr(gateway.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda **kwargs: [])
    monkeypatch.setattr(
        receipt, "_socket_identity",
        lambda home: (outgoing_pid, {"code_sha": "b" * 40, "supervisor": "manual"}),
    )

    def process_listing(argv, **kwargs):
        assert argv == ["ps", "-Aww", "-o", "pid=,command="]
        return SimpleNamespace(returncode=0, stdout=f"{outgoing_pid} python -m hermes_cli.main gateway run\n")

    monkeypatch.setattr(gateway.subprocess, "run", process_listing)
    assert gateway.find_gateway_pids(all_profiles=True) == []
    plan = update_inventory.UpdatePlan(profiles=["default"])
    update_inventory._collect_gateway_runtimes(plan, [("default", tmp_path)], set())
    assert [(runtime.profile, runtime.pid) for runtime in plan.runtimes] == [("default", outgoing_pid)]

    monkeypatch.setattr(fleet, "_restart_systemd_gateway_units", lambda *args, **kwargs: None)
    monkeypatch.setattr(fleet, "_restart_macos_launchd_gateways", lambda *args, **kwargs: None)
    monkeypatch.setattr(fleet, "_warn_incomplete_gateway_fleet_restart", lambda *args: None)
    monkeypatch.setattr(fleet, "_force_kill_stuck_gateways", lambda *args: None)
    from hermes_cli import update_host_obligation

    monkeypatch.setattr(update_host_obligation, "mark_host_restart_completed", lambda sha: None)
    monkeypatch.setattr(fleet, "_restart_identity_sha", lambda: "b" * 40)
    restart = fleet._restart_gateway_fleet_after_update(plan, gateway_mode=False)
    assert restart.phase_errors == []
    assert restart.pre_restart_gateway_pids is not None
    assert outgoing_pid in restart.pre_restart_gateway_pids
    assert restart.killed_pids == set()
    assert restart.stopped_unmapped_pids == set()
