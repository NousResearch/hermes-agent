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
