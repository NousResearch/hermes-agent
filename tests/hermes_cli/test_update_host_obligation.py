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


@pytest.fixture(autouse=True)
def _units_belong_to_this_update(monkeypatch):
    """The fake units here run on invented PIDs (4242, per-unit tables) with no readable home;
    ownership (#93349, ``test_update_fleet_home_scope.py``) is pinned so these tests keep proving
    the once-per-host-process collapse, not home scoping."""
    monkeypatch.setattr(fleet, "_systemd_unit_owned_by_update", lambda scope_cmd, svc_name: True)


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
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", list)


def test_obligation_armed_by_one_profile_is_owed_by_every_other(two_profiles, no_live_fleet, monkeypatch):
    """One host, one obligation: the profile that did not pull still owes — and can discharge — it."""
    _enter(monkeypatch, two_profiles["coder"])
    _arm("coder")

    _enter(monkeypatch, two_profiles["writer"])
    assert fleet._pending_fleet_restart_needed() is True

    fleet._clear_fleet_restart_pending_marker()
    _enter(monkeypatch, two_profiles["coder"])
    assert fleet._pending_fleet_restart_needed() is False


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


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores directory permissions")
def test_unwritable_host_state_dir_never_hides_the_debt_from_another_profile(two_profiles, no_live_fleet, monkeypatch, tmp_path):
    """With a second profile on the install, the arming profile's per-home marker is a debt the
    other profile cannot see: it would read "no restart owed" for a tree about to move. Every
    profile observes the debt, or the arm refuses and the commit point does not move (review S2).
    """
    for home in two_profiles.values():
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")  # real, listed profiles
    _enter(monkeypatch, two_profiles["coder"])
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.chmod(0o500)
    try:
        armed = fleet._write_fleet_restart_pending_marker(
            expected_sha=SHA, runtimes=[{"kind": "gateway", "profile": "coder"}])
        _enter(monkeypatch, two_profiles["writer"])
        assert armed is False or fleet._fleet_restart_obligation_armed(), "writer cannot see the armed debt"
    finally:
        lock_dir.chmod(0o700)


def test_one_installs_completed_restart_never_erases_another_installs_debt(tmp_path):
    """Two installations of one OS user share the host state dir. Install B arming its own pull and
    then completing its restart must leave install A's debt standing; each install arms, reads and
    clears from its own process, the way two checkouts' ``hermes update`` runs do (review S3)."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    source = Path(__file__).resolve().parents[2]
    files = ("hermes_constants.py", "hermes_cli/__init__.py", "hermes_cli/update_host_obligation.py",
             "hermes_cli/update_restart_recovery.py", "hermes_cli/update_lock.py")
    roots = {}
    for name in ("install-a", "install-b"):
        roots[name] = tmp_path / name
        for rel in files:
            (roots[name] / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, roots[name] / rel)
    env = {**os.environ, "HERMES_GATEWAY_LOCK_DIR": str(tmp_path / "gateway-locks")}

    def run(install: str, body: str) -> str:
        code = (f"import sys\nsys.path.insert(0, {str(roots[install])!r})\n"
                "from hermes_cli.update_host_obligation import *\n" + body)
        child = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], env=env, capture_output=True,
                               text=True, encoding="utf-8", stdin=subprocess.DEVNULL, timeout=60, check=False)
        assert child.returncode == 0, child.stdout + child.stderr
        return child.stdout.strip()

    run("install-a", f"assert write_host_obligation(expected_sha={'a' * 40!r}, owner='run-a')")
    run("install-b", f"assert write_host_obligation(expected_sha={'b' * 40!r}, owner='run-b')\n"
                     f"mark_host_restart_completed({'b' * 40!r})\nclear_host_obligation()")

    assert run("install-a", "print((read_host_obligation() or {}).get('expected_sha'))") == "a" * 40
    assert run("install-b", "print(host_obligation_present())") == "False"


@pytest.mark.parametrize("a_sha", ["a" * 40, "b" * 40])
def test_a_completion_never_erases_the_unkeyed_debt_an_older_install_armed(tmp_path, monkeypatch, a_sha):
    """Install A still runs a release that writes the unkeyed record; install B (this release) armed
    its own keyed record. B's completed restart judged only B's record, so A's debt must survive it,
    whether the SHAs differ or match. With no record of its own, B honours the unkeyed one and its
    discharge clears it: an upgrade's own interrupted debt never sticks (review S3 residual)."""
    lock_dir = tmp_path / "gateway-locks"
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    legacy = lock_dir / host_obligation.HOST_OBLIGATION_NAME

    def a_arms() -> None:  # the older release's writer: unkeyed name, no owners, no mutex
        lock_dir.mkdir(parents=True, exist_ok=True)
        legacy.write_text(json.dumps({"version": 1, "pid": 1, "expected_sha": a_sha}), encoding="utf-8")

    assert host_obligation.write_host_obligation(expected_sha="b" * 40, owner="run-b")
    a_arms()
    host_obligation.mark_host_restart_completed("b" * 40)
    host_obligation.clear_host_obligation()

    assert json.loads(legacy.read_text(encoding="utf-8"))["expected_sha"] == a_sha
    assert (host_obligation.read_host_obligation() or {}).get("expected_sha") == a_sha

    host_obligation.clear_host_obligation()  # B now owns no record: the unkeyed one is what it judged
    assert not host_obligation.host_obligation_present()


@pytest.mark.platforms("posix")  # Windows opens no directory handle; NTFS journals the rename
def test_every_host_record_write_makes_its_rename_durable(tmp_path, monkeypatch):
    """Arm and restore share one writer, and both fsync the state dir after the rename: a power loss
    after the arm returned must not roll the directory back to "nothing owed"."""
    import stat

    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "gateway-locks"))
    events, real_fsync, real_replace = [], os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: events.append(
        "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file") or real_fsync(fd))
    monkeypatch.setattr(os, "replace", lambda src, dst: events.append("rename") or real_replace(src, dst))

    assert host_obligation.write_host_obligation(expected_sha=SHA)
    host_obligation.replace_bytes(host_obligation.host_obligation_path(), b"{}")

    assert events == ["file", "rename", "dir"] * 2


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores directory permissions")
@pytest.mark.parametrize("unreadable", ["roster", "profile"])
def test_an_unreadable_profile_roster_never_admits_a_private_restart_marker(
        two_profiles, no_live_fleet, monkeypatch, tmp_path, unreadable):
    """Host record unwritable AND the profile roster (or one profile's identity) unreadable: that is
    an UNKNOWN inventory, not a single-profile install. The default profile's marker would hide the
    debt from the profile nobody could read, so the arm refuses (review S2 residual)."""
    (two_profiles["writer"] / "config.yaml").write_text("{}\n", encoding="utf-8")  # the one named profile
    _enter(monkeypatch, tmp_path)  # the default profile arms
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    blind = tmp_path / "profiles" if unreadable == "roster" else two_profiles["writer"]
    lock_dir.chmod(0o500)
    blind.chmod(0o300 if unreadable == "roster" else 0o000)  # roster: traversable, not listable
    try:
        armed = fleet._write_fleet_restart_pending_marker(expected_sha=SHA)
    finally:
        blind.chmod(0o700)
        lock_dir.chmod(0o700)

    assert armed is False
    assert fleet._fleet_restart_pending_marker_path().is_file(), "the arming profile still keeps its marker"


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
    {"HERMES_REAL_HOME": "/srv/real", "HOME": "/srv/other"},
    {},
])
def test_recovery_host_state_dir_matches_the_gateway_resolver(monkeypatch, env):
    """``update_restart_recovery`` re-implements the lock-dir rule (it may import no Hermes code
    at runtime); the duplicate must not drift from ``gateway.status._get_lock_dir``."""
    from gateway.status import _get_lock_dir

    for name in ("HERMES_GATEWAY_LOCK_DIR", "XDG_STATE_HOME", "HERMES_REAL_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert recovery._host_state_dir() == str(_get_lock_dir())
