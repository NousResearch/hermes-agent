"""The update→restart obligation is HOST-scoped, and one update restarts the host gateway once.

Multiplex-only (Teknium ruling): exactly one ``hermes gateway run`` per host serves every
profile. The obligation used to live in ONE profile's ``HERMES_HOME``, so ``hermes -p coder
update`` armed and cleared coder's copy while restarting the SHARED process; no other profile
could see that obligation, and every profile that ran the catch-up killed the same gateway
again. These tests pin the host-scoped contract:

- an obligation armed from one profile is owed (and dischargeable) from every other profile;
- the host gateway is restarted AT MOST ONCE per obligation, however many profiles run it;
- enumerated systemd units that resolve to one live main PID restart that process once;
- the fresh-process recovery restarts one host process for every profile it serves.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import hermes_cli.update_cmd_fleet as fleet
import hermes_cli.update_host_obligation as host_obligation
import hermes_cli.update_restart_recovery as recovery
from hermes_cli import update_cmd

SHA = "a" * 40


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Two profile HERMES_HOMEs behind ONE host state dir — the real multiplex topology."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "gateway-locks"))
    homes = {}
    for name in ("coder", "writer"):
        home = tmp_path / "profiles" / name
        home.mkdir(parents=True)
        homes[name] = home
    return homes


def _enter(monkeypatch, home) -> None:
    monkeypatch.setenv("HERMES_HOME", str(home))


def _arm(profile_runtime: str) -> None:
    fleet._write_fleet_restart_pending_marker(
        expected_sha=SHA, runtimes=[{"kind": "gateway", "profile": profile_runtime}])


@pytest.fixture
def no_live_fleet(monkeypatch):
    """No fleet matrix rows: the obligation can never be discharged by evidence in these tests."""
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: SHA)
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", lambda: [])


def test_obligation_armed_by_one_profile_is_owed_by_every_other(two_profiles, no_live_fleet, monkeypatch):
    """One host, one obligation: the profile that did not pull still owes — and can discharge — it."""
    _enter(monkeypatch, two_profiles["coder"])
    _arm("coder")

    _enter(monkeypatch, two_profiles["writer"])
    assert fleet._pending_fleet_restart_needed() is True

    fleet._clear_fleet_restart_pending_marker()
    _enter(monkeypatch, two_profiles["coder"])
    assert fleet._pending_fleet_restart_needed() is False


def test_host_gateway_restarts_once_when_two_profiles_run_the_catch_up(
    two_profiles, no_live_fleet, monkeypatch, capsys
):
    """``hermes -p coder update`` then ``hermes -p writer update`` stops the host gateway ONCE."""
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **k: [4242])
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_macos", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_windows", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway._wait_for_gateway_exit", lambda **k: True)
    monkeypatch.setattr(fleet, "_restart_macos_launchd_gateways", lambda *a, **k: None)
    kills: list = []
    monkeypatch.setattr("hermes_cli.gateway.kill_gateway_processes", lambda **k: kills.append(k))

    _enter(monkeypatch, two_profiles["coder"])
    _arm("coder")
    assert update_cmd._run_pending_fleet_restart() is True

    _enter(monkeypatch, two_profiles["writer"])
    _arm("writer")
    assert update_cmd._run_pending_fleet_restart() is True

    assert len(kills) == 1, "the one host gateway must be stopped once per update, not once per profile"
    assert "already restarted for this update" in capsys.readouterr().out


def test_legacy_per_home_marker_is_still_read_and_cleared(two_profiles, no_live_fleet, monkeypatch):
    """An obligation armed by the pre-host-scope code must still be discharged after the upgrade."""
    _enter(monkeypatch, two_profiles["coder"])
    legacy = fleet._fleet_restart_pending_marker_path()
    legacy.write_text(
        f"started=1.0\npid=1\nexpected_sha={SHA}\n"
        + "inventory=" + json.dumps({"version": 1, "runtimes": [{"kind": "gateway", "profile": "coder"}]}) + "\n",
        encoding="utf-8",
    )

    assert fleet._pending_fleet_restart_needed() is True
    fleet._clear_fleet_restart_pending_marker()
    assert not legacy.exists()
    assert fleet._pending_fleet_restart_needed() is False


def _listing(units: list[str]):
    result = SimpleNamespace(returncode=0, stdout="\n".join(f"{u} loaded active running x" for u in units))
    return [("user", ["systemctl", "--user"], result)]


def test_leftover_per_profile_units_restart_their_one_host_process_once(monkeypatch, capsys):
    """Three units, one live main PID = one host gateway: restart it once and name the legacy units."""
    # raising=False keeps this usable as the red-on-base A/B (the helper is the fix).
    monkeypatch.setattr(fleet, "_unit_main_pid", lambda scope_cmd, svc: 4242, raising=False)
    restarted: list[str] = []
    monkeypatch.setattr(
        fleet, "_systemctl_reset_and_restart",
        lambda manage_cmd, svc, scope_cmd=None: restarted.append(svc) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(fleet, "_wait_for_service_active", lambda scope_cmd, svc: True)
    monkeypatch.setattr(fleet, "_SYSTEMD_SCOPES", (("user", ["systemctl", "--user"]),))

    failed: list = []
    fleet._restart_systemd_gateway_units_best_effort(
        failed, _listing(["hermes-gateway.service", "hermes-gateway-coder.service", "hermes-gateway-writer.service"]))

    assert restarted == ["hermes-gateway"]
    assert failed == []
    out = capsys.readouterr().out
    assert "hermes-gateway-coder" in out and "legacy per-profile unit" in out


def test_units_with_distinct_live_pids_are_each_restarted(monkeypatch):
    """Control: genuinely separate processes are still separate restart targets."""
    pids = {"hermes-gateway": 1, "hermes-gateway-coder": 2}
    monkeypatch.setattr(fleet, "_unit_main_pid", lambda scope_cmd, svc: pids[svc], raising=False)
    restarted: list[str] = []
    monkeypatch.setattr(
        fleet, "_systemctl_reset_and_restart",
        lambda manage_cmd, svc, scope_cmd=None: restarted.append(svc) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(fleet, "_wait_for_service_active", lambda scope_cmd, svc: True)
    monkeypatch.setattr(fleet, "_SYSTEMD_SCOPES", (("user", ["systemctl", "--user"]),))

    fleet._restart_systemd_gateway_units_best_effort(
        [], _listing(["hermes-gateway.service", "hermes-gateway-coder.service"]))

    assert sorted(restarted) == ["hermes-gateway", "hermes-gateway-coder"]


def _host_record(tmp_path, monkeypatch, profiles: list[str]) -> None:
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    (lock_dir / "host-gateway.json").write_text(
        json.dumps({"role": "gateway", "pid": os.getpid(), "profiles": profiles}), encoding="utf-8")


def test_recovery_restarts_one_host_process_for_all_the_profiles_it_serves(tmp_path, monkeypatch):
    """N payload profiles served by ONE host gateway = one relaunch, not N that kill each other."""
    _host_record(tmp_path, monkeypatch, ["coder", "writer", "default"])
    argvs: list[list[str]] = []

    def fake_run(argv, **kwargs):
        argvs.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="")

    result = recovery.restart_profiles(["coder", "writer", "default"], run=fake_run)

    relaunches = [argv for argv in argvs if argv[-2:] == ["gateway", "restart"]]
    assert len(relaunches) == 1, "one host process must be relaunched once for every profile it serves"
    reported = [*result["verified"], *result["relaunch_attempted"], *result["failed"]]
    assert sorted(reported) == ["coder", "default", "writer"], "every requested profile keeps an outcome"
    assert result["covered"] == {"coder": ["default", "writer"]}


def test_recovery_keeps_separate_processes_separate(tmp_path, monkeypatch):
    """Control: with no host record, each profile is its own restart target."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "empty-locks"))
    argvs: list[list[str]] = []

    def fake_run(argv, **kwargs):
        argvs.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="")

    recovery.restart_profiles(["coder", "writer"], run=fake_run)

    assert len([argv for argv in argvs if argv[-2:] == ["gateway", "restart"]]) == 2


def test_host_obligation_lives_beside_the_host_rendezvous_record(two_profiles, monkeypatch, tmp_path):
    """The record is written ONCE PER HOST, in the cross-profile rendezvous dir."""
    _enter(monkeypatch, two_profiles["coder"])
    _arm("coder")

    path = host_obligation.host_obligation_path()
    assert path == tmp_path / "gateway-locks" / "host-update-restart.json"
    assert path.is_file()
    assert not (two_profiles["coder"] / "fleet_restart_pending").exists()


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores directory permissions")
def test_unwritable_host_state_dir_still_arms_the_obligation(two_profiles, no_live_fleet, monkeypatch, tmp_path):
    """An unwritable host state dir must never silently disarm the update→restart obligation.

    The host record moved out of ``$HERMES_HOME`` (writable by construction) into the host state
    dir, which a read-only mount or a container UID mismatch can make unwritable. Losing the
    obligation there is the #117275 outage shape: an interrupted update leaves stale code running
    with no warning and no catch-up restart.
    """
    _enter(monkeypatch, two_profiles["coder"])
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.chmod(0o500)
    try:
        _arm("coder")
        assert not host_obligation.host_obligation_present(), "precondition: the record could not be written"
        assert fleet._fleet_restart_obligation_armed() is True
        assert fleet._pending_fleet_restart_needed() is True
    finally:
        lock_dir.chmod(0o700)


def test_unreadable_host_record_is_never_discharged_by_the_legacy_marker(two_profiles, no_live_fleet, monkeypatch, tmp_path):
    """A record whose terms are UNKNOWN cannot be settled by another record's terms.

    A foreign version (a NEWER CLI wrote it) or a corrupt record is fail-closed by contract; the
    legacy per-home marker describes a different obligation and must not discharge it.
    """
    _enter(monkeypatch, two_profiles["coder"])
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / host_obligation.HOST_OBLIGATION_NAME).write_text(
        json.dumps({"version": 99, "expected_sha": SHA}), encoding="utf-8")
    fleet._fleet_restart_pending_marker_path().write_text(
        f"started=1.0\npid=1\nexpected_sha={SHA}\n"
        + "inventory=" + json.dumps({"version": 1, "runtimes": []}) + "\n",
        encoding="utf-8",
    )

    assert fleet._obligation_fields() is None
    assert fleet._pending_fleet_restart_needed() is True


def test_restart_runs_once_per_host_on_a_non_git_install(two_profiles, monkeypatch, capsys):
    """zip/pip/Docker installs resolve no checkout SHA; the restart-once guard must still hold.

    ``mark_host_restart_completed("")`` can never match, so every profile's ``hermes update``
    re-killed the one shared multiplexer on exactly the installs this record exists for.
    """
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: None)
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", lambda: [])
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **k: [4242])
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_macos", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_windows", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway._wait_for_gateway_exit", lambda **k: True)
    kills: list = []
    monkeypatch.setattr("hermes_cli.gateway.kill_gateway_processes", lambda **k: kills.append(k))

    _enter(monkeypatch, two_profiles["coder"])
    _arm("coder")
    assert update_cmd._run_pending_fleet_restart() is True

    _enter(monkeypatch, two_profiles["writer"])
    assert update_cmd._run_pending_fleet_restart() is True

    assert len(kills) == 1, "the host gateway must be stopped once per update, not once per profile"
    assert "already restarted for this update" in capsys.readouterr().out


def test_a_failing_main_pid_probe_keeps_its_own_restart():
    """Any probe error is unproven identity (its own restart), never an aborted restart pass."""
    def boom(unit):
        raise RuntimeError("systemctl exploded")

    restart, covered = host_obligation.collapse_units_to_host_processes(["a.service", "b.service"], boom)

    assert restart == ["a.service", "b.service"]
    assert covered == {}


@pytest.mark.parametrize("env", [
    {"HERMES_GATEWAY_LOCK_DIR": "/srv/override/locks"},
    {"XDG_STATE_HOME": "/srv/xdg-state"},
    {"XDG_STATE_HOME": "relative/state"},
    {},
])
def test_recovery_host_state_dir_matches_the_gateway_resolver(monkeypatch, env):
    """``update_restart_recovery`` re-implements the lock-dir rule (it may import no Hermes code
    at runtime); the duplicate must not drift from ``gateway.status._get_lock_dir``."""
    from gateway.status import _get_lock_dir

    for name in ("HERMES_GATEWAY_LOCK_DIR", "XDG_STATE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert recovery._host_state_dir() == str(_get_lock_dir())
