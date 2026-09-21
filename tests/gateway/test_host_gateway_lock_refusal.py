"""The host gateway lock REFUSES a second gateway instead of observing it (multiplex-only).

The real flock is taken on the real lock path; nothing is monkeypatched about the lock itself, so
the test fails on any build where losing the host lock still starts a second gateway.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def host_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "gateway-locks"))
    from gateway import host_rendezvous as hr
    hr._lock_handles.clear()
    yield tmp_path
    hr._lock_handles.clear()


def _hold_host_lock_from_another_description(hr):
    """Hold the host gateway lock on a SEPARATE open file description.

    flock is per-description, so this contends with ``claim_host_lock`` exactly the way a second
    process would — and it bypasses the per-process memo that would otherwise answer ACQUIRED.
    """
    import fcntl

    hr.ensure_host_state_dir()
    handle = open(hr.lock_path(hr.ROLE_GATEWAY), "a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


@pytest.mark.skipif(sys.platform == "win32", reason="flock-based contention setup")
def test_second_host_gateway_is_refused_with_75_naming_the_owner_and_the_migrate_command(
    host_lock_dir, capsys,
):
    from gateway import host_rendezvous as hr
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE, GATEWAY_SERVICE_RESTART_EXIT_CODE
    from gateway.run import _claim_host_gateway_role
    from hermes_cli.gateway_migrate import MIGRATE_COMMAND

    hr.publish_record(hr.ROLE_GATEWAY, profiles=("default", "coder"), home=str(host_lock_dir))
    owner = hr.read_record(hr.ROLE_GATEWAY, include_stale=True)
    assert owner is not None
    handle = _hold_host_lock_from_another_description(hr)
    try:
        with pytest.raises(SystemExit) as exc:
            _claim_host_gateway_role()
    finally:
        handle.close()

    # 75 (EX_TEMPFAIL) so systemd/s6/launchd RETRY; 78 would park the unit on a runtime condition.
    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
    assert exc.value.code != GATEWAY_FATAL_CONFIG_EXIT_CODE
    out = capsys.readouterr().out
    assert f"PID {owner.pid}" in out and MIGRATE_COMMAND in out
    assert "--force" in out and "--replace" in out


@pytest.mark.skipif(sys.platform == "win32", reason="flock-based contention setup")
def test_force_still_starts_a_second_gateway_and_an_unusable_lock_dir_is_not_a_refusal(
    host_lock_dir, monkeypatch,
):
    """The two non-refusals: the operator's explicit escape hatch, and a lock dir we cannot open
    (EROFS/EACCES is not evidence of a second gateway, and refusing there takes a healthy
    single-gateway host down)."""
    from gateway import host_rendezvous as hr
    from gateway.run import _claim_host_gateway_role

    hr.publish_record(hr.ROLE_GATEWAY, profiles=(), home=str(host_lock_dir))
    handle = _hold_host_lock_from_another_description(hr)
    try:
        _claim_host_gateway_role(force=True)  # no SystemExit
    finally:
        handle.close()

    hr._lock_handles.clear()
    monkeypatch.setattr(
        hr, "claim_host_lock",
        lambda role: (hr.HostLockOutcome.COULD_NOT_OPEN, OSError("read-only file system")))
    _claim_host_gateway_role()  # no SystemExit


@pytest.mark.skipif(sys.platform == "win32", reason="flock-based contention setup")
def test_an_unmigrated_standalone_fleet_starts_beside_the_owner_instead_of_spinning(
    host_lock_dir, monkeypatch, caplog,
):
    """COMPOSITION with #118236 ('a standalone host owner means START, not a parked unit').

    That change routes a profile whose host owner is another profile's STANDALONE gateway to
    START, because no multiplexer serves it. The host-lock refusal then exits 75, the supervisor
    retries in 5s, and the next claim loses the same race: the lock is per OS USER and every
    gateway takes it, so a second profile can NEVER win it. Composed, the two correct decisions
    are an infinite 5s retry loop for every unmigrated fleet with >=2 profiles — including one
    installed with --force. The refusal must not fire for a START that exists precisely because
    nothing serves this profile.
    """
    import logging

    from gateway import host_rendezvous as hr
    from gateway.run import _claim_host_gateway_role

    hr.publish_record(hr.ROLE_GATEWAY, profiles=("default",), home=str(host_lock_dir))
    owner = hr.read_record(hr.ROLE_GATEWAY, include_stale=True)
    assert owner is not None
    # The owner answers the rescan the way a STANDALONE gateway does: "I do not multiplex."
    # Stubbed at the wire answer every tree has, so a tree without the carve-out fails on the
    # OUTCOME (SystemExit 75) rather than on a missing symbol.
    from gateway.host_attach import HostGateway
    standalone_owner = HostGateway(pid=owner.pid + 1, home=host_lock_dir, profiles=("default",),
                                   served_known=True, standalone=True)  # another process
    monkeypatch.setattr("gateway.host_attach.host_gateway",
                        lambda **kw: standalone_owner)
    monkeypatch.setattr("gateway.host_attach.request_serve_profile",
                        lambda profile, owner=None: standalone_owner)

    handle = _hold_host_lock_from_another_description(hr)
    try:
        with caplog.at_level(logging.WARNING):
            _claim_host_gateway_role()  # must NOT SystemExit: 75 here is an unwinnable retry
    finally:
        handle.close()

    from hermes_cli.gateway_migrate import MIGRATE_COMMAND
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "standalone gateway owns this host" in logged
    assert MIGRATE_COMMAND in logged, "the bounded outcome must name the command that converges"


@pytest.mark.skipif(sys.platform == "win32", reason="flock-based contention setup")
def test_a_multiplexing_owner_is_still_refused(host_lock_dir, monkeypatch):
    """The carve-out is scoped to an unmigrated fleet: losing the race to a MULTIPLEXER is still
    the second-gateway shape, and an owner we cannot interrogate is treated as one."""
    from gateway import host_rendezvous as hr
    from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
    from gateway.run import _claim_host_gateway_role

    hr.publish_record(hr.ROLE_GATEWAY, profiles=("default", "coder"), home=str(host_lock_dir))
    monkeypatch.setattr("gateway.host_attach.request_serve_profile",
                        lambda profile, owner=None: None)  # owner never answers
    handle = _hold_host_lock_from_another_description(hr)
    try:
        with pytest.raises(SystemExit) as exc:
            _claim_host_gateway_role()
    finally:
        handle.close()
    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE


def _publish_unprovable_owner_record(hr, *, owner_pid, owner_home, profiles=("default",)):
    """Publish a rendezvous record for ANOTHER live process whose incarnation cannot be PROVEN.

    The recorded ``createTime`` is an ABSOLUTE epoch value derived from the host's boot time, so a
    boot-time correction (WSL, a resumed VM, an NTP step) moves every record written before it well
    past the 2 s incarnation tolerance -- for a process that never died. That is not a hypothetical:
    it is what put a live standalone owner's record beyond proof on the reporting host. The record
    is written on disk exactly as the owner would have left it; nothing about the probe is patched.
    """
    from gateway.host_attach import invalidate_host_gateway_cache

    hr.publish_record(hr.ROLE_GATEWAY, profiles=profiles, home=str(owner_home))
    path = hr.record_path(hr.ROLE_GATEWAY)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pid"] = owner_pid
    payload["createTime"] = (hr.process_create_time(owner_pid) or 0.0) - 304.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    invalidate_host_gateway_cache()
    assert hr.read_record(hr.ROLE_GATEWAY) is None, "premise: the owner cannot be proven live"
    owner = hr.read_record(hr.ROLE_GATEWAY, include_stale=True)
    assert owner is not None and owner.pid == owner_pid
    return owner


def _write_owner_state(home, *, pid, reason, gateway_state="running", hermes_home=None,
                       age_s=0, served_profiles=()):
    """Write the owner's own ``gateway_state.json`` -- the local, port-less, owner-authored fact."""
    from gateway.status import get_process_start_time

    home.mkdir(parents=True, exist_ok=True)
    (home / "gateway_state.json").write_text(json.dumps({
        "kind": "gateway", "pid": pid, "start_time": get_process_start_time(pid),
        "hermes_home": str(hermes_home or home), "gateway_state": gateway_state,
        "served_profiles": list(served_profiles), "multiplex_standalone_reason": reason,
        "updated_at": (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat(),
    }), encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="flock-based contention setup")
def test_a_standalone_owner_that_cannot_be_interrogated_is_still_started_beside(
    host_lock_dir, caplog,
):
    """An owner whose liveness the rendezvous probe cannot PROVE must not be read as a multiplexer.

    Losing the host lock is itself atomic proof that an owner is alive; the only open question is
    whether it multiplexes. When the probe that would ask cannot even resolve the owner, the owner's
    own ``gateway_state.json`` still answers it -- written by the owner, readable with no port and no
    socket, and fingerprinted against a boot-RELATIVE start time that survives the clock correction
    that put the rendezvous record beyond proof.

    Getting this wrong is unrecoverable rather than merely wrong: the host lock is per OS USER and
    every gateway takes it, so the 75 the refusal exits with buys a retry that loses the identical
    race. The reporting host crash-looped one profile's unit 34 times on exactly this path.
    """
    import logging

    from gateway import host_rendezvous as hr
    from gateway.run import _claim_host_gateway_role
    from hermes_cli.gateway_migrate import MIGRATE_COMMAND

    owner_home = host_lock_dir / "owner-home"
    owner_pid = os.getppid()  # a real live process that is not us
    _publish_unprovable_owner_record(hr, owner_pid=owner_pid, owner_home=owner_home)
    _write_owner_state(owner_home, pid=owner_pid,
                       reason="profile(s) 'coder' still run their own gateway")

    handle = _hold_host_lock_from_another_description(hr)
    try:
        with caplog.at_level(logging.WARNING):
            _claim_host_gateway_role()  # must NOT SystemExit: 75 here is an unwinnable retry
    finally:
        handle.close()

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "standalone gateway owns this host" in logged
    assert MIGRATE_COMMAND in logged, "the bounded outcome must name the command that converges"


@pytest.mark.skipif(sys.platform == "win32", reason="flock-based contention setup")
@pytest.mark.parametrize("ambiguity, state", [
    ("no state file at all", None),
    ("a multiplexer: it declares no standalone reason",
     {"reason": None, "served_profiles": ("default", "coder")}),
    ("the snapshot names a different PID", {"reason": "standalone", "pid_offset": 1}),
    ("the heartbeat is older than the snapshot TTL", {"reason": "standalone", "age_s": 600}),
    ("the snapshot says the gateway stopped", {"reason": "standalone", "gateway_state": "stopped"}),
    ("the snapshot was written for a different home", {"reason": "standalone",
                                                      "hermes_home": "/somewhere/else"}),
])
def test_only_a_positive_standalone_declaration_lifts_the_second_gateway_refusal(
    host_lock_dir, ambiguity, state,
):
    """The fallback is a narrow carve-out, never a way to fail open.

    Same uninterrogable owner as above, so the ONLY difference is what its state file says. Anything
    short of that owner positively declaring itself standalone -- including a real multiplexer, and
    including every flavour of "this snapshot may not be about this owner" -- must still refuse with
    EX_TEMPFAIL. A second gateway beside a multiplexer double-binds the profile's platforms.
    """
    from gateway import host_rendezvous as hr
    from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
    from gateway.run import _claim_host_gateway_role

    owner_home = host_lock_dir / "owner-home"
    owner_pid = os.getppid()
    _publish_unprovable_owner_record(hr, owner_pid=owner_pid, owner_home=owner_home)
    if state is not None:
        kwargs = dict(state)
        pid = owner_pid + kwargs.pop("pid_offset", 0)
        _write_owner_state(owner_home, pid=pid, **kwargs)

    handle = _hold_host_lock_from_another_description(hr)
    try:
        with pytest.raises(SystemExit) as exc:
            _claim_host_gateway_role()
    finally:
        handle.close()
    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE, ambiguity
