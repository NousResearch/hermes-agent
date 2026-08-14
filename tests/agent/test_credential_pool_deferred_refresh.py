"""Thread-safety of the deferred single-use-token refresh path (#71775).

The deferred path deliberately runs OAuth network I/O outside the pool
lock. These tests pin the two invariants that make that safe:

1. `select()` does NOT hold the pool lock while the deferred refresh's
   network call runs (the whole point of the PR).
2. The pool mutations that follow the network call (`_replace_entry`,
   `_persist`) DO re-serialize under the pool lock, so a concurrent
   `select()`/rotation cannot tear `self._entries` or double-write
   auth.json.
"""

import threading
from dataclasses import replace

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)


def _codex_entry(entry_id: str = "codex-1") -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=entry_id,
        label="test codex",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token="at-stale",
        refresh_token="rt-stale",
        expires_at_ms=1,  # long expired -> needs refresh
    )


def test_select_does_not_hold_pool_lock_during_deferred_refresh(monkeypatch):
    pool = CredentialPool("openai-codex", [_codex_entry()])
    lock_free_during_refresh = {}

    def _fake_refresh(entry, *, force):
        # If select() still held the pool lock here, this non-blocking
        # acquire would fail — the regression this PR exists to fix.
        acquired = pool._lock.acquire(blocking=False)
        lock_free_during_refresh["value"] = acquired
        if acquired:
            pool._lock.release()
        refreshed = replace(entry, access_token="at-fresh", expires_at_ms=2**53)
        pool._replace_entry(entry, refreshed)
        return refreshed

    monkeypatch.setattr(
        pool, "_entry_needs_refresh", lambda e: e.access_token == "at-stale"
    )
    monkeypatch.setattr(pool, "_refresh_entry", _fake_refresh)
    monkeypatch.setattr(pool, "_persist", lambda **kw: None)

    selected = pool.select()

    assert lock_free_during_refresh.get("value") is True, (
        "select() held the pool lock during the deferred refresh network window"
    )
    assert selected is not None
    assert selected.access_token == "at-fresh"


def test_deferred_mutations_serialize_against_concurrent_rotation(monkeypatch):
    """_replace_entry/_persist from the deferred path must contend on the
    pool lock: with the lock held by another thread, the deferred mutation
    must block rather than mutate concurrently."""
    pool = CredentialPool("openai-codex", [_codex_entry()])
    monkeypatch.setattr(pool, "_persist", lambda **kw: None)

    entry = pool._entries[0]
    refreshed = replace(entry, access_token="at-fresh")

    mutated = threading.Event()

    def _deferred_mutation():
        pool._replace_entry(entry, refreshed)  # self-locking
        mutated.set()

    with pool._lock:
        t = threading.Thread(target=_deferred_mutation)
        t.start()
        # While we hold the lock, the deferred mutation must NOT complete.
        assert not mutated.wait(timeout=0.3), (
            "_replace_entry mutated the pool while another thread held the lock"
        )
    t.join(timeout=5)
    assert mutated.is_set()
    assert pool._entries[0].access_token == "at-fresh"
