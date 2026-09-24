"""The host gateway lock REFUSES a second gateway instead of observing it (multiplex-only).

The real flock is taken on the real lock path; nothing is monkeypatched about the lock itself, so
the test fails on any build where losing the host lock still starts a second gateway.
"""

from __future__ import annotations

import sys

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
    owner = hr.read_record(hr.ROLE_GATEWAY, include_stale=True)
    assert owner is not None
    # Another process holds the record (our own pid would short-circuit _owner_is_standalone before
    # the wire is asked), and it never answers the rescan: an owner we cannot interrogate is a
    # multiplexer.
    from gateway.host_attach import HostGateway
    silent_owner = HostGateway(pid=owner.pid + 1, home=host_lock_dir, profiles=(),
                               served_known=False)
    monkeypatch.setattr("gateway.host_attach.host_gateway", lambda **kw: silent_owner)
    monkeypatch.setattr("gateway.host_attach.request_serve_profile",
                        lambda profile, owner=None: None)  # owner never answers
    handle = _hold_host_lock_from_another_description(hr)
    try:
        with pytest.raises(SystemExit) as exc:
            _claim_host_gateway_role()
    finally:
        handle.close()
    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE


@pytest.mark.skipif(sys.platform == "win32", reason="flock-based contention setup")
@pytest.mark.asyncio
async def test_a_replace_unit_that_replaced_nothing_is_still_refused_when_it_loses_the_lock(
    host_lock_dir, tmp_path, monkeypatch,
):
    """Every unit Hermes generates (launchd, systemd, s6) runs ``gateway run --replace``. When the
    attach check saw no owner yet (the record lands a moment after the owner's claim) nothing was
    replaced, and the lock is the only arbiter of the race. Reading ``--replace`` as ``--force``
    there started a second gateway beside the multiplexer, and the two fought for the same bot
    tokens: each --replace token handoff SIGTERMs the current holder."""
    from unittest.mock import AsyncMock

    from gateway import host_rendezvous as hr
    from gateway import status
    from gateway.config import GatewayConfig
    from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
    from gateway.run import start_gateway

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    class _RunnerMustNotStart:
        def __init__(self, config):
            self.config = config
            self.adapters = {}

        async def start(self):
            raise AssertionError("a second gateway started beside the host owner")

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.run._host_attach_or_none", AsyncMock(return_value=None))
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _RunnerMustNotStart)
    # The lock holder is a multiplexer: the standalone start-beside carve-out must not rescue a
    # --replace unit. Patched where _claim_host_gateway_role reads it; the request_serve_profile
    # patch this used to carry was unreachable (_owner_is_standalone short-circuits on our own pid).
    monkeypatch.setattr("gateway.run._owner_is_standalone", lambda: False)

    hr.publish_record(hr.ROLE_GATEWAY, profiles=("default",), home=str(host_lock_dir))
    handle = _hold_host_lock_from_another_description(hr)
    try:
        with pytest.raises(SystemExit) as exc:
            await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)
    finally:
        handle.close()
        status.remove_pid_file()
        status.release_gateway_runtime_lock()
    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
