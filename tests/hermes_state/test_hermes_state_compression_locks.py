"""Tests for ``SessionDB`` compression-lock primitives.

These cover the atomic per-session lock that prevents two compression
paths from racing on the same ``session_id`` and producing orphan child
sessions (Damien's "parent → two orphan children" repro shape, see
``tests/agent/test_compression_concurrent_fork.py`` for the
behavioural regression test).

Focus here: the lock primitives themselves (acquire, release, TTL,
diagnostic accessor) — not the wiring into compression.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    return SessionDB(tmp_path / "state.db")


# ----------------------------------------------------------------------
# Single-holder semantics
# ----------------------------------------------------------------------


def test_acquire_succeeds_when_unlocked(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    assert db.get_compression_lock_holder("sess1") == "holder1"


def test_acquire_blocks_second_holder(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    assert db.try_acquire_compression_lock("sess1", "holder2") is False
    # First holder still owns it
    assert db.get_compression_lock_holder("sess1") == "holder1"








# ----------------------------------------------------------------------
# Per-session isolation
# ----------------------------------------------------------------------


def test_locks_are_per_session(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    # Different session: independent lock
    assert db.try_acquire_compression_lock("sess2", "holder2") is True
    assert db.get_compression_lock_holder("sess1") == "holder1"
    assert db.get_compression_lock_holder("sess2") == "holder2"


# ----------------------------------------------------------------------
# TTL / expiry recovery
# ----------------------------------------------------------------------


def test_expired_lock_is_reclaimable(db: SessionDB) -> None:
    """A crashed compressor must not permanently block the session."""
    # Acquire with a very short TTL
    db.try_acquire_compression_lock("sess1", "crashed_holder", ttl_seconds=0.05)
    time.sleep(0.1)
    # Holder check honours expiry
    assert db.get_compression_lock_holder("sess1") is None
    # New holder can claim it
    assert db.try_acquire_compression_lock("sess1", "fresh_holder") is True
    assert db.get_compression_lock_holder("sess1") == "fresh_holder"


def test_non_expired_lock_is_held(db: SessionDB) -> None:
    db.try_acquire_compression_lock("sess1", "holder1", ttl_seconds=60)
    # Immediately after, still held
    assert db.try_acquire_compression_lock("sess1", "holder2") is False


def test_non_expired_lock_from_dead_pid_is_reclaimed(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No ``os.name`` pin: the probe below injects a fake ``psutil``, and
    # ``_process_is_gone`` consults psutil *before* its POSIX/nt split — the
    # nt early-return is unreachable here on any host, so faking the platform
    # bought nothing.
    dead_holder = "pid=424242:tid=1:agent=abc:nonce=deadbeef"
    assert db.try_acquire_compression_lock(
        "sess1", dead_holder, ttl_seconds=300
    ) is True

    probed: list[int] = []

    def process_is_gone(pid: int) -> bool:
        probed.append(pid)
        return False

    monkeypatch.setattr(
        hermes_state, "psutil", SimpleNamespace(pid_exists=process_is_gone)
    )

    assert db.try_acquire_compression_lock(
        "sess1", "pid=525252:tid=2:agent=def:nonce=fresh", ttl_seconds=300
    ) is True
    assert probed == [424242]




def test_probe_doubt_keeps_lease_until_ttl(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that errors out is doubt, not proof of death → TTL protects."""
    holder = "pid=424242:tid=1:agent=abc:nonce=doubt"
    assert db.try_acquire_compression_lock(
        "sess1", holder, ttl_seconds=300
    ) is True

    def probe_blows_up(pid: int) -> bool:
        raise RuntimeError("transient probe failure")

    monkeypatch.setattr(
        hermes_state, "psutil", SimpleNamespace(pid_exists=probe_blows_up)
    )

    assert db.try_acquire_compression_lock(
        "sess1", "pid=525252:tid=2:agent=def:nonce=other", ttl_seconds=300
    ) is False
    assert db.get_compression_lock_holder("sess1") == holder


def test_non_expired_lock_from_live_pid_is_not_reclaimed(db: SessionDB) -> None:
    live_holder = f"pid={os.getpid()}:tid=1:agent=abc:nonce=live"
    assert db.try_acquire_compression_lock(
        "sess1", live_holder, ttl_seconds=300
    ) is True
    assert db.try_acquire_compression_lock(
        "sess1", "pid=525252:tid=2:agent=def:nonce=other", ttl_seconds=300
    ) is False




def test_unstructured_holder_waits_for_ttl(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert db.try_acquire_compression_lock(
        "sess1", "legacy_holder", ttl_seconds=300
    ) is True
    monkeypatch.setattr(
        hermes_state,
        "psutil",
        SimpleNamespace(
            pid_exists=lambda _pid: pytest.fail(
                "unstructured holder must not probe a PID"
            )
        ),
    )
    monkeypatch.setattr(
        hermes_state.os,
        "kill",
        lambda *_args: pytest.fail("unstructured holder must not probe a PID"),
    )
    assert db.try_acquire_compression_lock(
        "sess1", "pid=525252:tid=2:agent=def:nonce=other", ttl_seconds=300
    ) is False






# ----------------------------------------------------------------------
# Empty / invalid input
# ----------------------------------------------------------------------








# ----------------------------------------------------------------------
# Concurrency: real threads racing on the same session_id
# ----------------------------------------------------------------------


def test_concurrent_acquire_only_one_winner(db: SessionDB) -> None:
    """Damien's race shape: N threads call acquire on the same session_id;
    exactly one must win, the rest must be cleanly rejected."""
    results: list[bool] = []
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def try_acquire(idx: int) -> None:
        holder = f"thread_{idx}"
        barrier.wait()  # synchronize start
        got = db.try_acquire_compression_lock("contended_session", holder)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=try_acquire, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread acquired
    assert sum(1 for r in results if r is True) == 1
    assert sum(1 for r in results if r is False) == 7
    # The single winner still owns it
    assert db.get_compression_lock_holder("contended_session") is not None
