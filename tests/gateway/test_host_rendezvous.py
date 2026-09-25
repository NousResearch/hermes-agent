"""Host-wide singleton invariants (multiplex-only): one lock per host, staleness is proved."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway import host_rendezvous as hr

_CHILD = """
import json, os, sys, time
sys.path.insert(0, {tree!r})
from gateway.status import acquire_gateway_runtime_lock
from gateway import host_rendezvous as hr
print(json.dumps({{
    "per_home": acquire_gateway_runtime_lock(),
    "host": hr.claim_host_lock(hr.ROLE_GATEWAY)[0] is hr.HostLockOutcome.ACQUIRED,
}}), flush=True)
time.sleep(60)
"""

# Publishes a serve record + its 0600 token, arms the exit cleanup, then waits to be killed.
_SIGTERM_CHILD = """
import sys, time
sys.path.insert(0, {tree!r})
from gateway import host_rendezvous as hr
assert hr.claim_host_lock(hr.ROLE_SERVE)[0] is hr.HostLockOutcome.ACQUIRED
hr.publish_record(hr.ROLE_SERVE, host="127.0.0.1", port=9119, token="live-session-token")
hr.cleanup_on_exit(hr.ROLE_SERVE)
print("published", flush=True)
time.sleep(120)
"""


@pytest.fixture
def host_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    yield tmp_path
    hr.release_host_lock(hr.ROLE_GATEWAY)
    hr.release_host_lock(hr.ROLE_SERVE)


def _tree() -> str:
    return str(Path(__file__).resolve().parents[2])


def test_two_homes_take_their_own_per_home_lock_but_only_one_host_lock(host_dir, monkeypatch):
    """The per-home lock is per HERMES_HOME (N profiles = N locks); the host lock is not.

    This is the whole point of the host layer: before it, a second profile's gateway took its
    own ``gateway.lock`` and nothing on the machine noticed.
    """
    tree = _tree()
    home_a, home_b = host_dir / "home_a", host_dir / "home_b"
    for home in (home_a, home_b):
        home.mkdir()

    env = {**os.environ, "HERMES_HOME": str(home_a), "HERMES_GATEWAY_LOCK_DIR": str(host_dir / "locks")}
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(tree=tree)], env=env, cwd=tree,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert child.stdout is not None
    try:
        deadline = time.time() + 60
        line = ""
        while not line.strip() and time.time() < deadline:
            line = child.stdout.readline()
        first = json.loads(line)
        assert first == {"per_home": True, "host": True}

        monkeypatch.setenv("HERMES_HOME", str(home_b))
        from gateway import status

        assert status.acquire_gateway_runtime_lock() is True, "second home must get its OWN lock"
        assert hr.claim_host_lock(hr.ROLE_GATEWAY)[0] is hr.HostLockOutcome.HELD_BY_OTHER, \
            "host lock is not per-home"
    finally:
        child.kill()
        child.wait(timeout=10)
        from gateway import status

        status.release_gateway_runtime_lock()


@pytest.mark.parametrize(
    "pid,create_time",
    [(2**22 - 1, 1.0), (os.getpid(), 1.0)],
    ids=["dead-pid", "creation-time-mismatch"],
)
def test_stale_record_is_never_attachable(host_dir, pid, create_time):
    """A dead PID and a live PID from another incarnation are both stale — attaching to either
    dials whatever now owns that port."""
    record = hr.HostRecord(
        role=hr.ROLE_SERVE, pid=pid, create_time=create_time, host="127.0.0.1", port=9119,
        protocol_version=hr.HOST_PROTOCOL_VERSION, token_fingerprint="", profiles=("default",),
        updated_at="2026-01-01T00:00:00+00:00")
    hr.record_path(hr.ROLE_SERVE).parent.mkdir(parents=True, exist_ok=True)
    hr.record_path(hr.ROLE_SERVE).write_text(json.dumps(record.to_json()), encoding="utf-8")

    assert hr.record_is_stale(record) is True
    assert hr.read_record(hr.ROLE_SERVE) is None
    assert hr.read_record(hr.ROLE_SERVE, include_stale=True) is not None


@pytest.mark.platforms("posix")  # POSIX signal disposition
def test_sigterm_removes_the_record_and_its_live_session_token(host_dir):
    """SIGTERM is the NORMAL stop (systemd stop, docker stop, the update relaunch) and it does not
    run ``atexit``: the record outlived its process and the 0600 token kept a LIVE session token
    on disk indefinitely, so the next launch attached to something that was gone."""
    env = {**os.environ, "HERMES_GATEWAY_LOCK_DIR": str(host_dir / "locks")}
    child = subprocess.Popen(
        [sys.executable, "-c", _SIGTERM_CHILD.format(tree=_tree())], env=env, cwd=_tree(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert child.stdout is not None
    try:
        assert child.stdout.readline().strip() == "published"
        assert hr.record_path(hr.ROLE_SERVE).exists()
        assert hr.token_path(hr.ROLE_SERVE).exists()

        child.send_signal(signal.SIGTERM)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    assert not hr.record_path(hr.ROLE_SERVE).exists(), "record outlived its process"
    assert not hr.token_path(hr.ROLE_SERVE).exists(), "live session token outlived its process"
    assert hr.claim_host_lock(hr.ROLE_SERVE)[0] is hr.HostLockOutcome.ACQUIRED, "host lock not released"


def test_unopenable_lock_dir_is_not_reported_as_another_owner(host_dir, monkeypatch):
    """EROFS/EACCES/ENOSPC is a THIRD outcome: collapsing it into contention told operators
    another gateway owned the host and sent them hunting a process that never existed."""
    blocked = host_dir / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(blocked / "locks"))

    outcome, error = hr.claim_host_lock(hr.ROLE_SERVE)

    assert outcome is hr.HostLockOutcome.COULD_NOT_OPEN
    assert isinstance(error, OSError)


def test_lock_cache_follows_the_lock_directory(host_dir, monkeypatch, tmp_path):
    """The handle cache is keyed by (role, resolved path): keyed by role alone it reported
    "already held" for a directory it had never created, so ``owns_host_lock()`` lied."""
    assert hr.claim_host_lock(hr.ROLE_SERVE)[0] is hr.HostLockOutcome.ACQUIRED
    moved = tmp_path / "other-locks"
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(moved))

    assert hr.owns_host_lock(hr.ROLE_SERVE) is False
    assert hr.claim_host_lock(hr.ROLE_SERVE)[0] is hr.HostLockOutcome.ACQUIRED
    assert (moved / "host-serve.lock").exists()
    hr.release_host_lock(hr.ROLE_SERVE)


def test_probe_owner_refuses_a_closed_port_and_a_foreign_listener(host_dir):
    """The two shapes the record cannot see: the owner's socket is already closed (the
    graceful-shutdown window) and an unrelated process holds the port."""
    import dataclasses

    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    dead_port = closed.getsockname()[1]
    closed.close()
    record = hr.HostRecord(
        role=hr.ROLE_SERVE, pid=os.getpid(), create_time=hr.process_create_time(),
        host="127.0.0.1", port=dead_port, protocol_version=hr.HOST_PROTOCOL_VERSION,
        token_fingerprint="", profiles=(), updated_at="2026-01-01T00:00:00+00:00")

    assert hr.probe_owner(record, timeout=1.0) is None

    foreign = socket.socket()
    foreign.bind(("127.0.0.1", 0))
    foreign.listen(1)
    try:
        assert hr.probe_owner(
            dataclasses.replace(record, port=foreign.getsockname()[1]), timeout=1.0) is None
    finally:
        foreign.close()


def test_relative_xdg_state_home_is_ignored(monkeypatch, tmp_path):
    """XDG spec: a relative ``$XDG_STATE_HOME`` is invalid and must be ignored. Honouring one made
    the host lock dir CWD-relative, so two serves started from different directories would each
    take their own "host" lock."""
    from gateway import status

    monkeypatch.delenv("HERMES_GATEWAY_LOCK_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert status._get_lock_dir().is_absolute()
    assert status._get_lock_dir() == tmp_path / ".local" / "state" / "hermes" / status._LOCKS_DIRNAME
