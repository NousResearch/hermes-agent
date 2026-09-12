"""Tests for the TTL cache on SessionDB.message_count.

Covers the risk class — invalidation correctness — rather than just "it caches":
- A write (append) invalidates so the next read sees the new count.
- A delete invalidates so the next read sees the new count.
- The count is never stale past a write even when read repeatedly before the write.
- Cache scoping: a write to one state.db does NOT invalidate another DB's
  cached count (the invalidate_prefix fix vs. the original stash's global clear()).
- TTL expiry: a stale entry is re-fetched (defensive; the write hook is the
  real invalidation, but TTL guards against a missed write path).
- Correctness parity with the uncached path (same values).
"""
import time

import pytest

from hermes_state import SessionDB, _metadata_cache


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("s1", "cli")
    return d


@pytest.fixture(autouse=True)
def _clear_cache():
    """Start every test with an empty module-level cache (it is shared)."""
    _metadata_cache.clear()
    yield
    _metadata_cache.clear()


class TestMessageCountCache:
    def test_count_is_correct(self, db):
        assert db.message_count("s1") == 0
        db.append_message("s1", "user", "u1")
        db.append_message("s1", "assistant", "a1")
        assert db.message_count("s1") == 2
        assert db.message_count() == 2  # global count

    def test_append_invalidates_per_session_and_global(self, db):
        assert db.message_count("s1") == 0
        assert db.message_count() == 0
        # both now cached; a write must invalidate both so neither serves stale.
        db.append_message("s1", "user", "u1")
        assert db.message_count("s1") == 1
        assert db.message_count() == 1

    def test_repeated_reads_between_writes_are_cached_not_recounted(self, db, monkeypatch):
        # Probe: patch _read_one to count calls. Two reads with no write between
        # them must hit the cache (one COUNT(*), not two).
        calls = {"n": 0}
        orig = db._read_one

        def _counting(sql, params=()):
            if "COUNT(*)" in sql:
                calls["n"] += 1
            return orig(sql, params)

        db._read_one = _counting
        try:
            db.message_count("s1")
            db.message_count("s1")
            db.message_count("s1")
        finally:
            db._read_one = orig
        assert calls["n"] == 1, "repeated reads between writes should hit the cache"

    def test_delete_invalidates(self, db):
        db.append_message("s1", "user", "u1")
        db.append_message("s1", "assistant", "a1")
        assert db.message_count("s1") == 2
        db.delete_session("s1")
        # s1 is gone; its cached count must not survive the delete.
        assert db.message_count("s1") == 0
        assert db.message_count() == 0

    def test_write_to_one_db_does_not_invalidate_another(self, tmp_path):
        # Two separate state.db files share the module-level cache. A write to
        # db A must NOT invalidate db B's cached count (the invalidate_prefix
        # fix). The original stash's global clear() would have nuked both.
        a = SessionDB(db_path=tmp_path / "a.db")
        b = SessionDB(db_path=tmp_path / "b.db")
        a.create_session("sa", "cli")
        b.create_session("sb", "cli")
        a.append_message("sa", "user", "x")
        b.append_message("sb", "user", "y")
        # Populate both caches.
        assert a.message_count("sa") == 1
        assert b.message_count("sb") == 1
        a_cached = _metadata_cache.get(f"{a.db_path}:msg_count:sa")
        b_cached = _metadata_cache.get(f"{b.db_path}:msg_count:sb")
        assert a_cached == 1 and b_cached == 1
        # Write to A only.
        a.append_message("sa", "user", "x2")
        # A's cache entry was invalidated...
        assert _metadata_cache.get(f"{a.db_path}:msg_count:sa") is None
        # ...but B's must survive (no cross-invalidation).
        assert _metadata_cache.get(f"{b.db_path}:msg_count:sb") == 1
        # And B still reads correctly from its intact cache.
        assert b.message_count("sb") == 1
        a.close(); b.close()

    def test_ttl_expiry_refetches(self, db, monkeypatch):
        # Force a near-zero TTL on the shared cache so the entry expires
        # immediately, then verify a fresh COUNT(*) runs on the next read.
        _metadata_cache._ttl = 0.0
        try:
            db.append_message("s1", "user", "u1")
            db.message_count("s1")  # populates (and instantly expires)
            time.sleep(0.001)
            calls = {"n": 0}
            orig = db._read_one

            def _counting(sql, params=()):
                if "COUNT(*)" in sql:
                    calls["n"] += 1
                return orig(sql, params)

            db._read_one = _counting
            try:
                assert db.message_count("s1") == 1
            finally:
                db._read_one = orig
            assert calls["n"] == 1, "expired entry should re-run COUNT(*)"
        finally:
            _metadata_cache._ttl = 3.0

    def test_cache_value_matches_uncached(self, db):
        # The cached value must equal what an uncached read returns.
        db.append_message("s1", "user", "u1")
        db.append_message("s1", "assistant", "a1")
        db.append_message("s1", "user", "u2")
        cached = db.message_count("s1")
        _metadata_cache.clear()
        uncached = db.message_count("s1")
        assert cached == uncached == 3
