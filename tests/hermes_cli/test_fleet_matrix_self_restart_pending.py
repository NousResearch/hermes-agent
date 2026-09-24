"""A gateway this updater runs INSIDE (cron update in the gateway tree, #100179) accepted a
self-restart request and can only restart after the updater exits — so at fleet-matrix time it is
still on the pre-update code by construction. It used to render STALE and exit 1 on every nightly
cron update (#119597). Now the ancestor pid set recorded by the restart phase turns that row into
``restart_pending``; every OTHER gateway on the old sha keeps its ``stale`` verdict.
"""

import contextlib
import io
import json
import os

import pytest

import hermes_cli.update_cmd_fleet as fleet_mod
import hermes_cli.update_receipt as ur

OLD = "a" * 40
HEAD = "b" * 40


def _fleet_homes(monkeypatch, tmp_path, records: dict[str, dict]) -> None:
    """Default home + named profiles, each with a state file the identity resolver verifies as-is."""
    root = tmp_path / ".hermes"
    (root / "profiles").mkdir(parents=True)
    homes = {"default": root}
    for profile, record in records.items():
        home = root if profile == "default" else root / "profiles" / profile
        home.mkdir(exist_ok=True)
        homes[profile] = home
        (home / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")
    by_home = {str(home): rec["pid"] for profile, home in homes.items() for rec in [records[profile]]}
    monkeypatch.setattr("hermes_cli.build_info.get_code_identity", lambda refresh=False: {"sha": HEAD, "version": "1.0"})
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: root)
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: root / "profiles")
    monkeypatch.setattr("gateway.control_socket.identify_gateway", lambda h, **k: None)
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda h: by_home.get(str(h)))
    monkeypatch.setattr(ur, "_gateway_code_root", lambda pid, home: None)


def test_ancestor_with_accepted_self_restart_is_pending_while_other_stale_rows_still_fail(monkeypatch, tmp_path):
    ancestor, sibling = os.getpid(), os.getppid()
    _fleet_homes(monkeypatch, tmp_path, {
        "default": {"pid": ancestor, "gateway_state": "running", "code_sha": OLD},
        "ops": {"pid": sibling, "gateway_state": "running", "code_sha": OLD},
    })

    fleet = ur.collect_fleet_versions(pre_restart_pids=[ancestor, sibling], self_restart_pending={ancestor})
    states = {row["profile"]: row["state"] for row in fleet}
    assert states == {"default": ur.RESTART_PENDING_STATE, "ops": "stale"}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert ur.print_fleet_version_matrix(fleet) is True  # the sibling still fails the update
    assert "restart pending" in out.getvalue() and "1 gateway(s) still running the old code" in out.getvalue()

    # Without the sibling, the pending row alone is not a failure — and the same pid NOT recorded
    # as pending (the restart phase never reached the ancestor branch) is a plain stale verdict.
    only_ancestor = [row for row in fleet if row["profile"] == "default"]
    with contextlib.redirect_stdout(io.StringIO()):
        assert ur.print_fleet_version_matrix(only_ancestor) is False
    plain = ur.collect_fleet_versions(pre_restart_pids=[ancestor, sibling])
    assert {row["state"] for row in plain} == {"stale"}


def test_restart_phase_records_accepted_self_restart_and_verify_exits_clean(monkeypatch, tmp_path):
    """The ancestor branch of the drain triage feeds the pid set the matrix reads: end to end the
    verify phase exits 0 (not 1) and clears the pending marker for an update whose only old-code
    gateway is the one it runs inside."""
    import hermes_cli.gateway as gateway
    import hermes_cli.update_cmd as update_cmd

    ancestor = os.getpid()
    monkeypatch.setattr(gateway, "_is_pid_ancestor_of_current_process", lambda pid: pid == ancestor)
    monkeypatch.setattr(gateway, "_request_gateway_self_restart", lambda pid: True)
    pending: set = set()
    with contextlib.redirect_stdout(io.StringIO()):
        assert fleet_mod._drain_or_signal_gateway_for_update(ancestor, 5.0, "default", self_restart_pending=pending)
    assert pending == {ancestor}

    _fleet_homes(monkeypatch, tmp_path, {"default": {"pid": ancestor, "gateway_state": "running", "code_sha": OLD}})
    monkeypatch.setattr(fleet_mod, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(fleet_mod._time, "sleep", lambda s: None)
    cleared = []
    monkeypatch.setattr(fleet_mod, "_clear_fleet_restart_pending_marker", lambda: cleared.append(True))
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
    restart = fleet_mod._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[ancestor], restarted_services=["hermes-gateway"],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
        self_restart_pending_pids=pending,
    )
    with contextlib.redirect_stdout(io.StringIO()) as out:
        fleet_mod._verify_fleet_after_update(
            restart, _pre_update_plan=None, _windows_gateway_resume=None, node_failures=[], update_complete=True,
        )  # a SystemExit(1) here is the #119597 symptom
    assert "restart pending" in out.getvalue()
    assert "Update not complete" not in out.getvalue()
    assert cleared == [True]
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as exc:
        restart.self_restart_pending_pids = set()  # same fleet, identity not threaded → STALE, exit 1
        fleet_mod._verify_fleet_after_update(
            restart, _pre_update_plan=None, _windows_gateway_resume=None, node_failures=[], update_complete=True,
        )
    assert exc.value.code == 1
