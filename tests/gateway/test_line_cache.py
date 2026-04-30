"""Tests for the LINE request cache state machine (PENDING → READY → DELIVERED / ERROR)."""
import time

import pytest

from gateway.platforms.line import (
    RequestCache,
    State,
)


def test_register_pending_returns_uuid():
    cache = RequestCache(ttl_seconds=3600)
    rid = cache.register_pending()
    assert isinstance(rid, str)
    entry = cache.get(rid)
    assert entry is not None
    assert entry.state is State.PENDING


def test_set_ready_transitions_state():
    cache = RequestCache(ttl_seconds=3600)
    rid = cache.register_pending()
    cache.set_ready(rid, "the answer")
    entry = cache.get(rid)
    assert entry.state is State.READY
    assert entry.payload == "the answer"


@pytest.mark.parametrize("method", ["set_ready", "set_error"], ids=["set_ready", "set_error"])
def test_cache_unknown_id_noop(method):
    """Both transition methods must silently ignore unknown request_ids
    (otherwise a postback for an evicted entry would crash the handler)."""
    cache = RequestCache(ttl_seconds=3600)
    getattr(cache, method)("not-a-real-id", "ignored")  # must not raise


def test_mark_delivered_transitions_state():
    cache = RequestCache(ttl_seconds=3600)
    rid = cache.register_pending()
    cache.set_ready(rid, "x")
    cache.mark_delivered(rid)
    entry = cache.get(rid)
    assert entry.state is State.DELIVERED


def test_prune_removes_only_old_ready_and_delivered(monkeypatch):
    cache = RequestCache(ttl_seconds=10)  # default pending_ttl=86400
    rid_pending = cache.register_pending()
    rid_old_ready = cache.register_pending()
    cache.set_ready(rid_old_ready, "stale")
    rid_old_delivered = cache.register_pending()
    cache.set_ready(rid_old_delivered, "x")
    cache.mark_delivered(rid_old_delivered)

    # Backdate the two terminal entries
    now = time.time()
    cache._entries[rid_old_ready].updated_at = now - 11
    cache._entries[rid_old_delivered].updated_at = now - 11
    # Backdate PENDING's created_at past the regular TTL (10s) but well
    # under the ceiling TTL (default 86400s) — proves the regular TTL
    # does NOT apply to PENDING.
    cache._entries[rid_pending].created_at = now - 11

    cache.prune()

    assert cache.get(rid_pending) is not None  # PENDING not pruned via terminal TTL
    assert cache.get(rid_old_ready) is None
    assert cache.get(rid_old_delivered) is None


def test_prune_keeps_recent_entries():
    cache = RequestCache(ttl_seconds=3600)
    rid = cache.register_pending()
    cache.set_ready(rid, "x")
    cache.prune()
    assert cache.get(rid) is not None


def test_set_error_transitions_state():
    cache = RequestCache(ttl_seconds=3600)
    rid = cache.register_pending()
    cache.set_error(rid, "boom")
    entry = cache.get(rid)
    assert entry.state is State.ERROR
    assert entry.payload == "boom"


def test_prune_removes_old_error_entries():
    cache = RequestCache(ttl_seconds=10)
    rid = cache.register_pending()
    cache.set_error(rid, "boom")
    cache._entries[rid].updated_at = time.time() - 11
    cache.prune()
    assert cache.get(rid) is None


def test_prune_removes_old_pending_via_ceiling_ttl():
    cache = RequestCache(ttl_seconds=10, pending_ttl_seconds=20)
    rid = cache.register_pending()
    # Backdate created_at past the ceiling TTL
    cache._entries[rid].created_at = time.time() - 25

    cache.prune()

    assert cache.get(rid) is None


# ---- State-machine guard tests ----

def test_set_ready_after_delivered_is_noop():
    """set_ready on a DELIVERED entry must not roll the state back."""
    cache = RequestCache()
    rid = cache.register_pending()
    cache.set_ready(rid, "first")
    cache.mark_delivered(rid)
    cache.set_ready(rid, "second")
    entry = cache.get(rid)
    assert entry.state is State.DELIVERED
    assert entry.payload == "first"


def test_set_error_after_delivered_is_noop():
    cache = RequestCache()
    rid = cache.register_pending()
    cache.set_ready(rid, "ok")
    cache.mark_delivered(rid)
    cache.set_error(rid, "late error")
    assert cache.get(rid).state is State.DELIVERED


def test_mark_delivered_on_pending_is_noop():
    """Marking a still-PENDING entry delivered must be a no-op."""
    cache = RequestCache()
    rid = cache.register_pending()
    cache.mark_delivered(rid)
    assert cache.get(rid).state is State.PENDING
