"""Compression-lock reclaim boundaries: a contender reclaims only strictly after expiry, a live holder's re-acquire keeps its expiry, a provably dead holder of this PID namespace is reclaimed before expiry, and the compaction fence still refuses at the exact expiry."""

import os
from datetime import UTC, datetime

import pytest

from persistence import CompressionInProgress, SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from tests.contracts.authority.test_coordination import dead_holder

LOCK = SessionId("resource")


@pytest.mark.parametrize("ttl", [0.0, 0.375])
@pytest.mark.parametrize("offset", [-0.001, 0.0, 0.001])
def test_contender_reclaims_only_strictly_after_expiry(backend, scope_a, monkeypatch, ttl, offset):
    now = [2000.125]
    monkeypatch.setattr("time.time", lambda: now[0])
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder="owner", ttl_seconds=ttl) is True
    now[0] += ttl + offset
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder="contender", ttl_seconds=60) is (offset > 0)
    assert coord.refresh_compression_lock(session_id=LOCK, holder="owner", ttl_seconds=60) is (offset <= 0)
    assert coord.get_compression_lock_holder(session_id=LOCK) == ("contender" if offset > 0 else "owner")


@pytest.mark.parametrize("ttl", [0.0, 0.375])
@pytest.mark.parametrize("offset", [-0.001, 0.0])
def test_incumbent_reacquire_does_not_extend_original_expiry(backend, scope_a, monkeypatch, ttl, offset):
    now = [2000.125]
    monkeypatch.setattr("time.time", lambda: now[0])
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder="owner", ttl_seconds=ttl) is True
    now[0] += ttl + offset
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder="owner", ttl_seconds=60) is True
    now[0] = 2000.125 + ttl + 0.001
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder="contender", ttl_seconds=60) is True
    assert not coord.refresh_compression_lock(session_id=LOCK, holder="owner", ttl_seconds=60)


def test_lock_reclaims_a_provably_dead_holder_before_expiry(backend, scope_a):
    dead = dead_holder("owner")
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder=dead, ttl_seconds=60) is True
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder="new", ttl_seconds=60) is True
    assert not coord.refresh_compression_lock(session_id=LOCK, holder=dead, ttl_seconds=60)
    coord.release_compression_lock(session_id=LOCK, holder=dead)
    assert coord.get_compression_lock_holder(session_id=LOCK) == "new"


@pytest.mark.parametrize("holder", ["unstructured", f"pid={os.getpid()}:owner", "pid=99999999999999999999999999999:owner"])
def test_lock_keeps_holders_not_provably_dead(backend, scope_a, holder):
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder=holder, ttl_seconds=60) is True
    assert coord.try_acquire_compression_lock(session_id=LOCK, holder="contender", ttl_seconds=60) is False
    assert coord.get_compression_lock_holder(session_id=LOCK) == holder


def test_compaction_write_still_refuses_at_exact_lease_expiry(backend, scope_a, monkeypatch):
    store_a = backend.open(scope_a)
    now = [2000.125]
    monkeypatch.setattr("time.time", lambda: now[0])
    session_id = SessionId("compact")
    store_a.sessions.create(NewSession(session_id, source="cli"))
    store_a.messages.append(session_id, NewMessage(role="user", content="keep", timestamp=datetime(2026, 1, 1, tzinfo=UTC)))
    assert store_a.coordination.try_acquire_compression_lock(session_id=session_id, holder="owner", ttl_seconds=0.375) is True
    now[0] += 0.375
    assert store_a.coordination.try_acquire_compression_lock(session_id=session_id, holder="owner", ttl_seconds=60) is True
    with pytest.raises(CompressionInProgress):
        store_a.messages.compact(session_id, [], lock_holder="owner")
    assert [row.content for row in store_a.messages.read_transcript(session_id)] == ["keep"]
