"""PID-namespace qualification for state.db lock/lease holder probes.

A ``pid=<n>`` holder is namespace-relative: a sibling sharing one ``state.db``
from another PID namespace (gateway + webui containers on one volume) reads as
dead to a namespace-blind probe, so its live lease/lock must not be reclaimed.
The local namespace is pinned so the invariants hold on every host.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil
import pytest

import hermes_state_common
import hermes_state_pidns
from agent.conversation_compression import _compression_lock_holder
from hermes_state import SessionDB
from hermes_state_pidns import LocalPidNamespace

_OURS, _SIBLING = "111", "222"


@pytest.fixture(autouse=True)
def _pinned_namespace(monkeypatch):
    monkeypatch.setattr(hermes_state_pidns, "_LOCAL_PID_NS", LocalPidNamespace(_OURS, True))


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    assert not psutil.pid_exists(proc.pid), "reaped pid unexpectedly reused"
    return proc.pid


def test_turn_lease_of_sibling_namespace_is_not_stolen(tmp_path) -> None:
    """The sibling's turn is still in flight; a dead local reading must not
    end it early. The lease releases at TTL."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")

    sibling = f"pid={_dead_pid()}:pidns={_SIBLING}:turn=webui-turn:platform=webui"
    assert db.try_acquire_session_turn_lease("shared", sibling, ttl_seconds=0.3) is True

    contender = f"pid={os.getpid()}:pidns={_OURS}:turn=gw-turn:platform=cli"
    assert db.try_acquire_session_turn_lease("shared", contender, ttl_seconds=300) is False

    time.sleep(0.4)
    assert db.try_acquire_session_turn_lease("shared", contender, ttl_seconds=300) is True

    # Unstamped (pre-upgrade) TTL holder: STRICT — never probed, defers to TTL.
    db.create_session("legacy", source="test")
    legacy = f"pid={_dead_pid()}:turn=old:platform=x"
    assert db.try_acquire_session_turn_lease("legacy", legacy, ttl_seconds=300) is True
    assert db.try_acquire_session_turn_lease("legacy", contender, ttl_seconds=300) is False


def test_flock_holder_record_qualifies_pid_namespaces(monkeypatch, tmp_path) -> None:
    dead = _dead_pid()
    provably_dead = hermes_state_common._lock_holder_provably_dead

    # Writers stamp our namespace: the flock record and the compression holder.
    with open(tmp_path / "lock", "w+b") as handle:
        hermes_state_common._write_lock_holder_record(handle)
        assert hermes_state_common._read_lock_holder_record(handle)["pidns"] == _OURS
    assert f":pidns={_OURS}:" in _compression_lock_holder(object())

    # Same namespace + provably gone: break the orphaned lock (unchanged).
    assert provably_dead({"pid": dead, "pidns": _OURS, "start_ticks": 1, "acquired_at": 0.0}) is True
    # Foreign namespace: a local absence is not proof — defer.
    assert provably_dead({"pid": dead, "pidns": _SIBLING, "start_ticks": 1, "acquired_at": 0.0}) is False
    # Unstamped (pre-upgrade) record: these never expire, so keep probing them.
    assert provably_dead({"pid": dead, "start_ticks": 1, "acquired_at": 0.0}) is True
    assert provably_dead(None) is False

    # Our own namespace lookup failed: a stamped record is unverifiable, but an
    # unstamped one still breaks (#100108 orphan-lock cleanup must not regress).
    monkeypatch.setattr(hermes_state_pidns, "_LOCAL_PID_NS", None)
    monkeypatch.setattr(hermes_state_pidns, "_resolve_local_pid_namespace", lambda: LocalPidNamespace(None, True))
    assert provably_dead({"pid": dead, "pidns": _OURS, "start_ticks": 1, "acquired_at": 0.0}) is False
    assert provably_dead({"pid": dead, "start_ticks": 1, "acquired_at": 0.0}) is True

    # The failed lookup was not cached: once it resolves, a stamped record breaks.
    monkeypatch.setattr(hermes_state_pidns, "_resolve_local_pid_namespace", lambda: LocalPidNamespace(_OURS, True))
    assert provably_dead({"pid": dead, "pidns": _OURS, "start_ticks": 1, "acquired_at": 0.0}) is True
