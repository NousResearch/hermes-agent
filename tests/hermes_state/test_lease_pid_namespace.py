"""PID-namespace qualification for state.db lock/lease holder probes.

A ``pid=<n>`` holder is namespace-relative: siblings sharing one ``state.db``
from different PID namespaces (the standard gateway + webui container layout
on one volume, systemd ``PrivatePIDs=``) see disjoint PID sets, so a live
sibling's holder reads as absent to a namespace-blind ``psutil.pid_exists()``
probe — and the unexpired row is reclaimed under the sibling's feet.  For a
session turn lease that ends the sibling's in-flight turn ("another Hermes
process took over this session"); for a compression lock it splits the
compression lineage.  For the flock holder records (``hermes_state_common``) it
breaks a live holder's lock.

``hermes_state_pidns`` stamps every structured holder with the writer's PID
namespace (``pidns=``, the ``/proc/self/ns/pid`` inode) and qualifies each
probe on it.  Two policies, one rule:

* :func:`holder_pid_checkable` — STRICT, the policy for TTL-bounded rows
  (compression locks, session turn leases).  A same-namespace reading is still
  kernel proof and crash cleanup is unchanged; a foreign-namespace OR unstamped
  holder is never probed — it defers to the row's own expiry (at most the
  remaining TTL; a false defer self-heals, a false reclaim loses a reply).
  Strictness is also what stops a steal during a mixed-version rollout as soon
  as the reader restarts, before every writer has.

* :func:`persistent_record_pidns_checkable` — LEGACY ROLLOUT, the policy for
  records with no expiry (the flock holder records).  A foreign-namespace
  record is never probed; an unstamped one keeps main's behavior (refusing to
  probe it would permanently disable orphaned-lock cleanup).

These tests pin the qualification matrix and the behavioral halves, plus the
sibling-shape regression that motivated the stamp.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil
import pytest

import hermes_state
import hermes_state_common
import hermes_state_pidns
from hermes_state import SessionDB
from hermes_state_pidns import (
    LocalPidNamespace,
    _parse_pid_namespace_link,
    holder_namespace_token,
    holder_pid_checkable,
    persistent_record_pidns_checkable,
    pid_namespace_id,
    recorded_namespace,
)


def _dead_pid() -> int:
    """A PID that is provably gone: spawn, reap, verify absence."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    pid = proc.pid
    assert not psutil.pid_exists(pid), "reaped pid unexpectedly reused"
    return pid


def _sibling_pidns() -> str:
    """A namespace id that is not ours (linux-gated callers only)."""
    return str(int(pid_namespace_id()) + 12345)


def _ttl_holder(pid, pidns=None, *, suffix="turn=t:platform=test") -> str:
    """A structured TTL-row holder string in the production shape."""
    stamp = f":pidns={pidns}" if pidns is not None else ""
    return f"pid={pid}{stamp}:{suffix}"


# ---------------------------------------------------------------------------
# Identity resolution (pure parts)
# ---------------------------------------------------------------------------


def test_parse_pid_namespace_link() -> None:
    assert _parse_pid_namespace_link("pid:[4026532534]") == "4026532534"
    assert _parse_pid_namespace_link("pid:[123]") == "123"
    assert _parse_pid_namespace_link("") is None
    assert _parse_pid_namespace_link("pid:no-brackets") is None


def test_recorded_namespace_extracts_only_a_segment_bounded_token() -> None:
    assert recorded_namespace("pid=1:pidns=42:turn=x") == "42"
    assert recorded_namespace("pidns=42") == "42"
    assert recorded_namespace("pid=1:tid=2:agent=ab:nonce=cd") is None
    assert recorded_namespace("") is None
    assert recorded_namespace(None) is None
    # The token must start at a segment boundary — not some other key's suffix.
    assert recorded_namespace("apidns=7") is None
    assert recorded_namespace("x-pidns=7") is None


@pytest.mark.parametrize(
    ("local", "recorded", "strict", "legacy"),
    [
        # Platform has no namespace concept (macOS/Windows): a pid is still
        # evidence and both predicates keep main's semantics.
        (LocalPidNamespace(None, False), None, True, True),
        (LocalPidNamespace(None, False), "123", True, True),
        # The diverging row: unstamped on Linux — strict defers (TTL-bounded
        # rows must be protected during mixed-version rollouts), legacy probes
        # (no-expiry records must not become permanently unverifiable).
        (LocalPidNamespace("111", True), None, False, True),
        # Linux: our own namespace verifies.
        (LocalPidNamespace("111", True), "111", True, True),
        # Linux: a sibling's namespace is never probed, under either policy.
        (LocalPidNamespace("111", True), "222", False, False),
        # Linux with a failed lookup: no authority — nothing is verifiable.
        (LocalPidNamespace(None, True), None, False, False),
        (LocalPidNamespace(None, True), "111", False, False),
    ],
)
def test_namespace_qualification_matrix(monkeypatch, local, recorded, strict, legacy) -> None:
    monkeypatch.setattr(hermes_state_pidns, "_LOCAL_PID_NS", local)
    assert holder_pid_checkable(_ttl_holder(123, recorded)) is strict
    assert persistent_record_pidns_checkable(recorded) is legacy


@pytest.mark.platforms("linux")
def test_holder_namespace_token_round_trips_on_this_host() -> None:
    token = holder_namespace_token()
    assert token.startswith(":pidns=") and token[len(":pidns="):].isdigit()
    holder = f"pid=123{token}:tid=1:agent=abc:nonce=deadbeef"
    assert recorded_namespace(holder) == pid_namespace_id()
    assert holder_pid_checkable(holder) is True


@pytest.mark.platforms("linux")
def test_failed_namespace_lookup_is_not_cached_and_defers(monkeypatch) -> None:
    """A transient /proc failure must not be cached, and while it lasts every
    structured holder is unverifiable (defer, never probe)."""
    calls: list[str] = []
    real_readlink = os.readlink

    def readlink(path):
        if path == "/proc/self/ns/pid":
            calls.append(path)
            raise OSError("simulated readlink failure")
        return real_readlink(path)

    monkeypatch.setattr(hermes_state_pidns, "_LOCAL_PID_NS", None)
    monkeypatch.setattr(hermes_state_pidns.os, "readlink", readlink)

    assert hermes_state_pidns.pid_namespace_id() is None
    assert hermes_state_pidns.pid_namespace_id() is None
    assert len(calls) == 2, "a failed lookup must be retried, not cached"
    assert holder_pid_checkable(_ttl_holder(1, "111")) is False
    assert holder_pid_checkable(_ttl_holder(1)) is False
    assert persistent_record_pidns_checkable("111") is False
    assert persistent_record_pidns_checkable(None) is False


# ---------------------------------------------------------------------------
# Probe gate (the choke-point both lease kinds share)
# ---------------------------------------------------------------------------


@pytest.mark.platforms("linux")
def test_probe_gate_matrix() -> None:
    dead = _dead_pid()
    probe = hermes_state._compression_lock_holder_process_is_dead

    # Same namespace + provably gone: still reclaimable on kernel proof.
    assert probe(_ttl_holder(dead, pid_namespace_id())) is True
    # Foreign namespace: our absence reading says nothing — defer (the bug).
    assert probe(_ttl_holder(dead, _sibling_pidns())) is False
    # Unstamped (pre-upgrade writer): TTL rows defer — see the module docstring.
    assert probe(_ttl_holder(dead)) is False
    # Unstructured and own-pid holders: never probed (existing contract).
    assert probe("legacy_holder") is False
    assert probe(_ttl_holder(os.getpid(), _sibling_pidns())) is False


# ---------------------------------------------------------------------------
# Behavioral: compression locks
# ---------------------------------------------------------------------------


@pytest.mark.platforms("linux")
def test_sibling_namespace_holder_survives_dead_local_pid(tmp_path) -> None:
    """A sibling container's holder is invisible to our PID probe — the lock
    must be kept (TTL still bounds it) instead of stolen."""
    db = SessionDB(tmp_path / "state.db")
    dead = _dead_pid()
    sibling = f"pid={dead}:pidns={_sibling_pidns()}:tid=1:agent=sibling:nonce=aaaaaaaa"
    assert db.try_acquire_compression_lock("sess1", sibling, ttl_seconds=300) is True

    contender = f"pid={os.getpid()}:pidns={pid_namespace_id()}:tid=2:agent=us:nonce=bbbbbbbb"
    assert db.try_acquire_compression_lock("sess1", contender, ttl_seconds=300) is False
    assert db.get_compression_lock_holder("sess1") == sibling


@pytest.mark.platforms("linux")
def test_unstamped_legacy_holder_defers_to_ttl(tmp_path) -> None:
    """TTL rows written before the stamp cannot be told apart from a live
    sibling's — defer, never steal; the row's own expiry is the release path."""
    db = SessionDB(tmp_path / "state.db")
    dead = _dead_pid()
    legacy = f"pid={dead}:tid=1:agent=old:nonce=cccccccc"
    assert db.try_acquire_compression_lock("sess1", legacy, ttl_seconds=0.2) is True

    contender = f"pid={os.getpid()}:pidns={pid_namespace_id()}:tid=2:agent=us:nonce=dddddddd"
    assert db.try_acquire_compression_lock("sess1", contender, ttl_seconds=300) is False
    assert db.get_compression_lock_holder("sess1") == legacy

    time.sleep(0.35)
    assert db.try_acquire_compression_lock("sess1", contender, ttl_seconds=300) is True


@pytest.mark.platforms("linux")
def test_same_namespace_dead_holder_is_still_reclaimed(tmp_path) -> None:
    """The conservative rule must not regress crash cleanup: a holder that
    died in OUR namespace is still reclaimed ahead of TTL."""
    db = SessionDB(tmp_path / "state.db")
    dead = _dead_pid()
    holder = f"pid={dead}:pidns={pid_namespace_id()}:tid=1:agent=crash:nonce=eeeeeeee"
    assert db.try_acquire_compression_lock("sess1", holder, ttl_seconds=300) is True

    contender = f"pid={os.getpid()}:pidns={pid_namespace_id()}:tid=2:agent=us:nonce=ffffffff"
    assert db.try_acquire_compression_lock("sess1", contender, ttl_seconds=300) is True


# ---------------------------------------------------------------------------
# Behavioral: session turn leases (the reported incident shape)
# ---------------------------------------------------------------------------


@pytest.mark.platforms("linux")
def test_turn_lease_of_sibling_namespace_is_not_stolen(tmp_path) -> None:
    """The incident: the sibling's turn is still in flight; a dead local
    reading must not end it early. The lease releases at TTL."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    dead = _dead_pid()

    sibling = f"pid={dead}:pidns={_sibling_pidns()}:turn=webui-turn:platform=webui"
    assert db.try_acquire_session_turn_lease("shared", sibling, ttl_seconds=0.3) is True

    contender = f"pid={os.getpid()}:pidns={pid_namespace_id()}:turn=gw-turn:platform=cli"
    assert db.try_acquire_session_turn_lease("shared", contender, ttl_seconds=300) is False

    time.sleep(0.4)
    assert db.try_acquire_session_turn_lease("shared", contender, ttl_seconds=300) is True


@pytest.mark.platforms("linux")
def test_turn_lease_same_namespace_dead_holder_is_reclaimed(tmp_path) -> None:
    """A turn owner that crashed in OUR namespace is reclaimed ahead of TTL —
    the e2e crash-resume path (test_compaction_kill9) depends on this."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    dead = _dead_pid()

    crashed = f"pid={dead}:pidns={pid_namespace_id()}:turn=crashed:platform=cli"
    assert db.try_acquire_session_turn_lease("shared", crashed, ttl_seconds=300) is True

    contender = f"pid={os.getpid()}:pidns={pid_namespace_id()}:turn=resumer:platform=cli"
    assert db.try_acquire_session_turn_lease("shared", contender, ttl_seconds=300) is True


# ---------------------------------------------------------------------------
# Behavioral: flock holder records (hermes_state_common)
# ---------------------------------------------------------------------------


@pytest.mark.platforms("linux")
def test_flock_holder_record_qualifies_pid_namespaces() -> None:
    dead = _dead_pid()
    provably_dead = hermes_state_common._lock_holder_provably_dead

    # Same namespace + provably gone: break the orphaned lock (unchanged).
    assert provably_dead(
        {"pid": dead, "pidns": pid_namespace_id(), "start_ticks": 1, "acquired_at": 0.0}
    ) is True
    # Foreign namespace: a local absence is not proof — defer.
    assert provably_dead(
        {"pid": dead, "pidns": _sibling_pidns(), "start_ticks": 1, "acquired_at": 0.0}
    ) is False
    # Unstamped (pre-upgrade record): LEGACY rollout — these records never
    # expire, so refusing to probe would permanently disable cleanup.
    assert provably_dead({"pid": dead, "start_ticks": 1, "acquired_at": 0.0}) is True
    # No/malformed record: defer (existing contract).
    assert provably_dead(None) is False

    # A live same-namespace holder stays unbroken; a recycled reading (its
    # start ticks disagree) still breaks — both proofs survive the stamp.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        ticks = hermes_state_common._proc_start_ticks(child.pid)
        assert ticks is not None
        assert provably_dead(
            {"pid": child.pid, "pidns": pid_namespace_id(), "start_ticks": ticks, "acquired_at": 0.0}
        ) is False
        assert provably_dead(
            {"pid": child.pid, "pidns": pid_namespace_id(), "start_ticks": ticks + 1, "acquired_at": 0.0}
        ) is True
    finally:
        child.kill()
        child.wait(timeout=30)
