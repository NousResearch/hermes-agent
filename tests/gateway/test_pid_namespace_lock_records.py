"""A gateway's lock records survive a reader in another PID namespace (#123081).

A gateway under systemd ``PrivatePIDs=`` or in a container records its own namespace's PID (often 1).
Outside, that number is init or nothing, so the owner of the machine-global Telegram token lock read
as dead: a second gateway on another profile reclaimed the lock and both polled ``getUpdates``.
Records now carry ``pidns`` and a foreign one is never probed. The writer here runs in a REAL PID
namespace (``unshare --user --pid``), not a faked stamp.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

from hermes_agent.gateway import lifecycle_ledger, status

pytestmark = pytest.mark.platforms("linux")

_HOLD = textwrap.dedent("""
    import json, os, sys
    sys.path.insert(0, sys.argv[1])
    from gateway import lifecycle_ledger, status
    status._looks_like_gateway_process = lambda pid: True
    assert status.acquire_gateway_runtime_lock()  # a real gateway holds its home's flock while it lives
    status.write_pid_file()
    ok, _ = status.acquire_scoped_lock("telegram-bot-token", "123:fixture")
    lifecycle_ledger.record_startup(defer_integrity_check=True)
    print(json.dumps({"ok": ok, "pid": os.getpid()}), flush=True)
    sys.stdin.read()  # hold the lock until the test closes our stdin
""")


def _unshare_works():
    if shutil.which("unshare") is None:
        return False
    probe = subprocess.run(["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount-proc", "true"],
                           capture_output=True, timeout=10, check=False)
    return probe.returncode == 0


@pytest.fixture
def namespaced_holder(tmp_path, monkeypatch):
    """A process in its own PID namespace holding the token lock and the lifecycle sentinel."""
    if not _unshare_works():
        pytest.skip("unprivileged user+PID namespaces are not available on this host")
    home, locks = tmp_path / "home", tmp_path / "locks"
    home.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "HERMES_GATEWAY_LOCK_DIR": str(locks)}
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen(
        ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount-proc",
         sys.executable, "-c", _HOLD, str(root)],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert line, proc.stderr.read()
        held = json.loads(line)
        assert held["ok"]
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(locks))
        yield held, home, locks, lambda: _end(proc)
    finally:
        _end(proc)


def _end(proc):
    """Close the holder's stdin: it exits without any exit path (no lock release, sentinel left
    ``running``), like a SIGKILL; the kernel drops its flock and the namespace goes with its init."""
    if proc.stdin and not proc.stdin.closed:
        proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def test_a_live_owner_in_another_pid_namespace_keeps_the_token_lock(namespaced_holder, monkeypatch):
    held, _home, locks, _end_holder = namespaced_holder
    record = json.loads(next(locks.glob("*.lock")).read_text())
    assert record["pid"] == held["pid"] and held["pid"] != os.getpid()
    assert record["pidns"] and record["pidns"] != os.readlink("/proc/self/ns/pid")[5:-1]
    monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)
    acquired, existing = status.acquire_scoped_lock("telegram-bot-token", "123:fixture")
    assert not acquired, "a second gateway took the token from a live owner in another namespace"
    assert existing["pid"] == held["pid"]


def test_its_lifecycle_sentinel_is_not_reported_as_an_unclean_death(namespaced_holder):
    _held, home, _locks, _end_holder = namespaced_holder
    assert lifecycle_ledger.detect_unclean_exit(home) is None


def test_its_pid_file_survives_a_status_poll_from_outside(namespaced_holder):
    _held, home, _locks, _end_holder = namespaced_holder
    status.get_running_pid(home / "gateway.pid")
    assert (home / "gateway.pid").exists(), "a status poll unlinked a live namespaced gateway's pid file"


def test_a_dead_owner_in_another_pid_namespace_is_reclaimed_and_reported(namespaced_holder, monkeypatch):
    """After a SIGKILL under ``PrivatePIDs=`` the record still names another namespace, so only the
    home's free ``gateway.lock`` flock can say it is gone. A start must take the token, not exit 78."""
    held, home, _locks, end_holder = namespaced_holder
    end_holder()
    monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)
    acquired, _existing = status.acquire_scoped_lock("telegram-bot-token", "123:fixture")
    assert acquired, "a dead owner in another PID namespace kept the token lock forever"
    evidence = lifecycle_ledger.detect_unclean_exit(home)
    assert evidence is not None and evidence["prior_pid"] == held["pid"]


def test_a_dead_owner_in_this_namespace_is_still_reclaimed(tmp_path, monkeypatch):
    """Control: the namespace check must not wedge the ordinary stale-lock recovery."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    (tmp_path / "home").mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock_path = status._get_scope_lock_path("telegram-bot-token", "123:fixture")
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({**status._build_pid_record(), "pid": dead.pid}))
    acquired, _existing = status.acquire_scoped_lock("telegram-bot-token", "123:fixture")
    assert acquired


def test_a_successor_on_the_same_home_does_not_keep_a_dead_owners_token(namespaced_holder, monkeypatch, tmp_path):
    """The namespaced owner dies; another gateway starts on that home (its token changed) and holds the
    home's flock. That flock is the successor's, not the old token's owner: the old token is free."""
    _held, home, locks, end_holder = namespaced_holder
    end_holder()
    root = Path(__file__).resolve().parents[2]
    # The successor holds the home's flock under its own token (no pid file: the dead owner left one).
    hold = _HOLD.replace('"123:fixture"', '"456:successor"').replace("status.write_pid_file()", "")
    successor = subprocess.Popen(
        [sys.executable, "-c", hold, str(root)],
        env={**os.environ, "HERMES_HOME": str(home), "HERMES_GATEWAY_LOCK_DIR": str(locks)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = successor.stdout.readline()
        assert line and json.loads(line)["ok"], successor.stderr.read()
        other = tmp_path / "other-home"
        other.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(other))
        monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)
        acquired, _existing = status.acquire_scoped_lock("telegram-bot-token", "123:fixture")
        assert acquired, "a successor on the dead owner's home kept its old token reserved"
    finally:
        _end(successor)
