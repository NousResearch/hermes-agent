"""Tests for agent/tool_cache.py — opt-in cross-tool result cache."""

import json
import os
import tempfile
import time

import pytest

from agent.tool_cache import (
    _DEFAULT_TTL,
    _cache_dir,
    _canonical_key,
    _enforce_budget,
    cache_stats,
    cached,
    clear_cache,
    reset_stats,
)


@pytest.fixture(autouse=True)
def tmp_cache(monkeypatch, tmp_path):
    """Redirect the cache to an isolated tmp dir, reset stats between tests."""
    monkeypatch.setenv("HERMES_TOOL_CACHE_DIR", str(tmp_path))
    clear_cache()
    reset_stats()
    yield tmp_path
    clear_cache()
    reset_stats()


def test_canonical_key_is_order_independent_for_kwargs():
    a = _canonical_key((), {"b": 2, "a": 1})
    b = _canonical_key((), {"a": 1, "b": 2})
    assert a == b


def test_canonical_key_distinguishes_args_vs_kwargs():
    a = _canonical_key((1, 2), {})
    b = _canonical_key((), {"first": 1, "second": 2})
    # Different shapes are different keys.
    assert a != b


def test_cached_hits_misses(tmp_cache):
    calls = []

    @cached(ttl_seconds=60)
    def expensive(x: int) -> int:
        calls.append(x)
        return x * 10

    assert expensive(5) == 50
    assert expensive(5) == 50  # second call hits cache
    assert expensive(6) == 60
    assert calls == [5, 6]  # 5 was computed once, 6 once

    stats = cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["stores"] == 2


def test_cached_respects_ttl(tmp_cache):
    calls = []

    @cached(ttl_seconds=1)
    def short_lived(x: int) -> int:
        calls.append(x)
        return x

    assert short_lived(7) == 7
    assert short_lived(7) == 7  # cached
    assert calls == [7]
    # Expire the entry by waiting + then mutating expires_at manually.
    path = _cache_dir() / "tool:tests.agent.test_tool_cache.test_cached_respects_ttl.<locals>.short_lived"
    files = list(path.rglob("*.json"))
    assert files, "cache file should exist"
    envelope = json.loads(files[0].read_text())
    envelope["expires_at"] = int(time.time()) - 1
    files[0].write_text(json.dumps(envelope))
    assert short_lived(7) == 7  # recomputed
    assert calls == [7, 7]


def test_cached_does_not_cache_exceptions(tmp_cache):
    state = {"n": 0}

    @cached(ttl_seconds=60)
    def flaky():
        state["n"] += 1
        raise ValueError("boom")

    with pytest.raises(ValueError):
        flaky()
    with pytest.raises(ValueError):
        flaky()
    # No entry should have been stored.
    stats = cache_stats()
    assert stats["stores"] == 0


def test_cached_does_not_cache_unserializable(tmp_cache):
    @cached(ttl_seconds=60)
    def returns_set():
        return {"a", "b"}  # sets are not JSON-native; cache must skip

    result = returns_set()
    # Result returned correctly even though we skipped caching.
    assert isinstance(result, set)
    stats = cache_stats()
    assert stats["stores"] == 0


def test_clear_cache_scoped(tmp_cache):
    @cached(ttl_seconds=60)
    def tool_a(x: int) -> int:
        return x

    @cached(ttl_seconds=60)
    def tool_b(x: int) -> int:
        return x * 100

    tool_a(1)
    tool_b(2)
    assert cache_stats()["entries"] == 2

    # Clear one tool only.
    removed = clear_cache("tool:tests.agent.test_tool_cache.test_clear_cache_scoped.<locals>.tool_a")
    assert removed == 1
    assert cache_stats()["entries"] == 1

    # Clear all.
    clear_cache()
    assert cache_stats()["entries"] == 0


def test_key_from_drops_noisy_arg(tmp_cache):
    calls = []

    @cached(ttl_seconds=60, key_from=lambda ts, payload: (payload,))
    def noised(ts: int, payload: str) -> str:
        calls.append(payload)
        return payload.upper()

    noised(1, "hello")
    noised(2, "hello")  # different ts but key_from drops it -> hit
    noised(3, "world")
    assert calls == ["hello", "world"]
    stats = cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2


def test_atomic_write_does_not_leave_partial_files(tmp_cache, monkeypatch):
    """If json.dump raises mid-write, no half-written file should remain."""
    @cached(ttl_seconds=60)
    def good_value():
        return {"ok": True}

    good_value()
    # Force a failure during write by patching json.dump to raise.
    import agent.tool_cache as tc

    original_dump = tc.json.dump
    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(tc.json, "dump", boom)
    @cached(ttl_seconds=60)
    def failing():
        return {"ok": True}

    # Should not raise; cache write fails silently.
    assert failing() == {"ok": True}
    # No .cache- temp files should be left behind.
    leftovers = list(_cache_dir().rglob(".cache-*"))
    assert leftovers == []


def test_budget_enforcement_caps_entries(tmp_cache, monkeypatch):
    from agent import tool_cache as tc

    monkeypatch.setattr(tc, "_MAX_ENTRIES", 3)
    monkeypatch.setattr(tc, "_MAX_BYTES", 10**12)  # effectively unlimited bytes

    @cached(ttl_seconds=60)
    def gen(i: int) -> int:
        return i

    for i in range(10):
        gen(i)

    # After budget enforcement, only the newest 3 should remain.
    assert cache_stats()["entries"] == 3
