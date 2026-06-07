"""Tests for gateway/session_model_pool.py"""

import time
import threading
import pytest

from gateway.session_model_pool import (
    PoolModelEntry,
    SessionModelPool,
    reset_session_model_pool,
)


# ---------------------------------------------------------------------------
# PoolModelEntry
# ---------------------------------------------------------------------------


class TestPoolModelEntry:
    def test_available_session_slots(self):
        e = PoolModelEntry(model="glm-5", provider="zai", max_concurrent=3, reserved_for_auxiliary=1)
        e.session_slots = {"s1": time.monotonic()}
        assert e.available_session_slots == 1  # 3 - 1(reserved) - 1(session) = 1

    def test_available_auxiliary_slots(self):
        e = PoolModelEntry(model="glm-5", provider="zai", max_concurrent=3, reserved_for_auxiliary=2)
        e.auxiliary_count = 1
        assert e.available_auxiliary_slots == 1

    def test_is_saturated(self):
        e = PoolModelEntry(model="glm-5", provider="zai", max_concurrent=2, reserved_for_auxiliary=1)
        e.session_slots = {"s1": time.monotonic()}
        e.auxiliary_count = 1
        assert e.is_saturated

    def test_not_saturated(self):
        e = PoolModelEntry(model="glm-5", provider="zai", max_concurrent=3, reserved_for_auxiliary=1)
        e.session_slots = {"s1": time.monotonic()}
        assert not e.is_saturated

    def test_pool_key(self):
        e = PoolModelEntry(model="glm-5-turbo", provider="zai")
        assert e.pool_key == "zai:glm-5-turbo"


# ---------------------------------------------------------------------------
# SessionModelPool.from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_disabled_returns_none(self):
        assert SessionModelPool.from_config({"enabled": False}) is None

    def test_empty_config_returns_none(self):
        assert SessionModelPool.from_config({}) is None

    def test_empty_pool_returns_none(self):
        assert SessionModelPool.from_config({"enabled": True, "pool": []}) is None

    def test_missing_pool_returns_none(self):
        assert SessionModelPool.from_config({"enabled": True}) is None

    def test_valid_config(self):
        config = {
            "enabled": True,
            "strategy": "round-robin",
            "inactive_timeout": 600,
            "pool": [
                {"model": "glm-5-turbo", "provider": "zai", "max_concurrent": 1},
                {"model": "glm-4.7", "provider": "zai", "max_concurrent": 2, "reserved_for_auxiliary": 1},
            ],
        }
        pool = SessionModelPool.from_config(config)
        assert pool is not None
        assert pool.enabled
        assert pool.strategy == "round-robin"
        assert pool.inactive_timeout == 600
        assert len(pool._entries) == 2
        assert pool._entries[0].model == "glm-5-turbo"
        assert pool._entries[1].reserved_for_auxiliary == 1

    def test_invalid_entries_skipped(self):
        config = {
            "enabled": True,
            "pool": [
                {"model": "", "provider": "zai"},  # no model
                {"provider": "zai"},  # no model
                {"model": "glm-5", "provider": "", "max_concurrent": 1},  # no provider
                "not-a-dict",
                {"model": "glm-5", "provider": "zai", "max_concurrent": 1},  # valid
            ],
        }
        pool = SessionModelPool.from_config(config)
        assert pool is not None
        assert len(pool._entries) == 1

    def test_unknown_strategy_defaults_to_round_robin(self):
        config = {
            "enabled": True,
            "strategy": "invalid",
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 1},
            ],
        }
        pool = SessionModelPool.from_config(config)
        assert pool.strategy == "round-robin"

    def test_reserved_capped_to_max_concurrent(self):
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 1, "reserved_for_auxiliary": 5},
            ],
        }
        pool = SessionModelPool.from_config(config)
        assert pool._entries[0].reserved_for_auxiliary == 1

    def test_priority_clamped_to_1_10(self):
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 1, "priority": 0},
                {"model": "glm-4", "provider": "zai", "max_concurrent": 1, "priority": 99},
            ],
        }
        pool = SessionModelPool.from_config(config)
        assert pool._entries[0].priority == 1
        assert pool._entries[1].priority == 10


# ---------------------------------------------------------------------------
# Session slot management
# ---------------------------------------------------------------------------


class TestSessionSlots:
    def _make_pool(self, strategy="round-robin"):
        config = {
            "enabled": True,
            "strategy": strategy,
            "pool": [
                {"model": "glm-5-turbo", "provider": "zai", "max_concurrent": 1, "priority": 10},
                {"model": "glm-4.7", "provider": "zai", "max_concurrent": 2, "priority": 8},
                {"model": "glm-4.6", "provider": "zai", "max_concurrent": 3, "reserved_for_auxiliary": 1, "priority": 5},
                {"model": "glm-4.5", "provider": "zai", "max_concurrent": 10, "priority": 1},
            ],
        }
        return SessionModelPool.from_config(config)

    def test_acquire_session_slot(self):
        pool = self._make_pool()
        result = pool.acquire_session_slot("sess-1")
        assert result is not None
        assert result["model"] == "glm-5-turbo"  # first entry, oldest activity → picked by round-robin
        assert result["provider"] == "zai"

    def test_acquire_returns_existing(self):
        pool = self._make_pool()
        r1 = pool.acquire_session_slot("sess-1")
        assert r1["model"] == "glm-5-turbo"
        r2 = pool.acquire_session_slot("sess-1")
        # Should return the same assignment
        assert r2["model"] == "glm-5-turbo"

    def test_acquires_next_when_saturated(self):
        pool = self._make_pool(strategy="priority")
        # glm-5-turbo has 1 session slot, 0 reserved → 1 available
        pool.acquire_session_slot("sess-1")  # takes glm-5-turbo
        # Next should get glm-4.7 (priority 8, 2 slots available)
        result = pool.acquire_session_slot("sess-2")
        assert result["model"] == "glm-4.7"

    def test_saturate_all_models(self):
        pool = self._make_pool(strategy="round-robin")
        # Available session slots: 1 + 2 + (3-1) + 10 = 15
        for i in range(15):
            result = pool.acquire_session_slot(f"sess-{i}")
            assert result is not None, f"Failed at session {i}"
        # 16th should fail
        result = pool.acquire_session_slot("sess-overflow")
        assert result is None

    def test_release_session_slot(self):
        pool = self._make_pool()
        pool.acquire_session_slot("sess-1")
        pool.release_session_slot("sess-1")
        # After release, glm-5-turbo should be available again
        stats = pool.get_pool_stats()
        glm5 = next(m for m in stats["models"] if m["model"] == "glm-5-turbo")
        assert glm5["session_count"] == 0

    def test_release_nonexistent_is_noop(self):
        pool = self._make_pool()
        pool.release_session_slot("nonexistent")  # should not raise

    def test_manual_override_prevents_assignment(self):
        pool = self._make_pool()
        pool.mark_manual_override("sess-manual")
        result = pool.acquire_session_slot("sess-manual")
        assert result is None  # manual override = pool won't assign

    def test_mark_manual_override_releases_existing(self):
        pool = self._make_pool()
        pool.acquire_session_slot("sess-1")
        pool.mark_manual_override("sess-1")
        stats = pool.get_pool_stats()
        glm5 = next(m for m in stats["models"] if m["model"] == "glm-5-turbo")
        assert glm5["session_count"] == 0

    def test_clear_manual_override_allows_reassignment(self):
        pool = self._make_pool()
        pool.mark_manual_override("sess-1")
        pool.clear_manual_override("sess-1")
        result = pool.acquire_session_slot("sess-1")
        assert result is not None


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestStrategies:
    def _make_pool(self, strategy):
        config = {
            "enabled": True,
            "strategy": strategy,
            "pool": [
                {"model": "high", "provider": "zai", "max_concurrent": 2, "priority": 10},
                {"model": "low", "provider": "zai", "max_concurrent": 2, "priority": 1},
            ],
        }
        return SessionModelPool.from_config(config)

    def test_priority_strategy_prefers_highest(self):
        pool = self._make_pool("priority")
        result = pool.acquire_session_slot("s1")
        assert result["model"] == "high"

    def test_least_loaded_strategy(self):
        pool = self._make_pool("least-loaded")
        r1 = pool.acquire_session_slot("s1")
        r2 = pool.acquire_session_slot("s2")
        # Both should go to different models (least loaded)
        assert r1["model"] != r2["model"]

    def test_round_robin_strategy(self):
        pool = self._make_pool("round-robin")
        r1 = pool.acquire_session_slot("s1")
        r2 = pool.acquire_session_slot("s2")
        # round-robin uses oldest last_activity → should distribute
        assert r1["model"] != r2["model"]

    def test_priority_tie_break_by_least_loaded(self):
        """When two models have same priority, pick the least loaded."""
        config = {
            "enabled": True,
            "strategy": "priority",
            "pool": [
                {"model": "a", "provider": "zai", "max_concurrent": 3, "priority": 5},
                {"model": "b", "provider": "zai", "max_concurrent": 3, "priority": 5},
            ],
        }
        pool = SessionModelPool.from_config(config)
        r1 = pool.acquire_session_slot("s1")  # fills model a (first)
        r2 = pool.acquire_session_slot("s2")  # tie-break → should pick b (0 sessions)
        assert r1["model"] == "a"
        assert r2["model"] == "b"

    def test_least_loaded_tie_break_by_priority(self):
        """When two models have same load, pick the highest priority."""
        config = {
            "enabled": True,
            "strategy": "least-loaded",
            "pool": [
                {"model": "lo-pri", "provider": "zai", "max_concurrent": 3, "priority": 1},
                {"model": "hi-pri", "provider": "zai", "max_concurrent": 3, "priority": 10},
            ],
        }
        pool = SessionModelPool.from_config(config)
        r1 = pool.acquire_session_slot("s1")
        r2 = pool.acquire_session_slot("s2")
        # Both have 1 session after r1 is on first; tie-break → hi-pri
        # Actually both start at 0 load → hi-pri (10) should be picked first
        assert r1["model"] == "hi-pri"
        # After s1 on hi-pri, lo-pri (0 load) wins over hi-pri (1 load)
        assert r2["model"] == "lo-pri"


# ---------------------------------------------------------------------------
# Auxiliary slot management
# ---------------------------------------------------------------------------


class TestAuxiliarySlots:
    def _make_pool(self):
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 2, "reserved_for_auxiliary": 1},
            ],
        }
        return SessionModelPool.from_config(config)

    def test_acquire_auxiliary_slot(self):
        pool = self._make_pool()
        assert pool.acquire_auxiliary_slot("glm-5", "zai") is True

    def test_auxiliary_blocked_when_full(self):
        pool = self._make_pool()
        pool.acquire_auxiliary_slot("glm-5", "zai")
        # reserved=1, now aux_count=1 → no more auxiliary slots
        assert pool.acquire_auxiliary_slot("glm-5", "zai", timeout=0.1) is False

    def test_release_auxiliary_slot(self):
        pool = self._make_pool()
        pool.acquire_auxiliary_slot("glm-5", "zai")
        pool.release_auxiliary_slot("glm-5", "zai")
        assert pool.acquire_auxiliary_slot("glm-5", "zai") is True  # freed

    def test_auxiliary_model_not_in_pool_allowed(self):
        pool = self._make_pool()
        assert pool.acquire_auxiliary_slot("qwen-local", "local") is True

    def test_auxiliary_release_nonexistent_noop(self):
        pool = self._make_pool()
        pool.release_auxiliary_slot("nonexistent", "unknown")  # no raise


# ---------------------------------------------------------------------------
# get_pool_stats
# ---------------------------------------------------------------------------


class TestPoolStats:
    def test_stats_structure(self):
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 2, "reserved_for_auxiliary": 1},
            ],
        }
        pool = SessionModelPool.from_config(config)
        stats = pool.get_pool_stats()
        assert stats["enabled"] is True
        assert stats["strategy"] == "round-robin"
        assert len(stats["models"]) == 1
        assert stats["models"][0]["max_concurrent"] == 2


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_session_model_pool(self):
        reset_session_model_pool()
        from gateway.session_model_pool import get_session_model_pool
        pool = get_session_model_pool({"session_model_pool": {"enabled": False}})
        assert pool is None
        reset_session_model_pool()

    def test_reset(self):
        reset_session_model_pool()
        from gateway.session_model_pool import _pool_instance
        assert _pool_instance is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_acquire(self):
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-4.5", "provider": "zai", "max_concurrent": 5},
            ],
        }
        pool = SessionModelPool.from_config(config)
        results = []
        errors = []

        def acquire(idx):
            try:
                r = pool.acquire_session_slot(f"sess-{idx}")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquire, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 5 slots available, 10 threads → 5 successes
        successes = [r for r in results if r is not None]
        assert len(successes) == 5


# ---------------------------------------------------------------------------
# Eviction (inactive session reaping)
# ---------------------------------------------------------------------------


class TestEviction:
    def _make_pool(self, timeout=2.0):
        config = {
            "enabled": True,
            "strategy": "priority",
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 2},
            ],
            "inactive_timeout": timeout,
        }
        return SessionModelPool.from_config(config)

    def test_active_session_not_evicted(self):
        pool = self._make_pool(timeout=10.0)
        pool.acquire_session_slot("sess-1")
        # Immediately after acquire, session should still be alive
        stats = pool.get_pool_stats()
        assert stats["total_sessions"] == 1

    def test_inactive_session_evicted(self):
        pool = self._make_pool(timeout=0.2)
        pool.acquire_session_slot("sess-1")
        # Wait longer than timeout so sess-1 becomes stale
        time.sleep(0.5)
        # Next acquire should trigger eviction of sess-1 and assign sess-2
        result = pool.acquire_session_slot("sess-2")
        assert result is not None
        # The old session should have been evicted, freeing the slot
        stats = pool.get_pool_stats()
        assert stats["total_sessions"] == 1  # only sess-2 remains

    def test_partial_eviction(self):
        """Only stale sessions are evicted; active ones remain."""
        pool = self._make_pool(timeout=0.5)
        pool.acquire_session_slot("sess-old")
        # Refresh sess-new by re-acquiring (touches timestamp)
        time.sleep(0.6)
        pool.acquire_session_slot("sess-old")  # re-acquire to refresh
        pool.acquire_session_slot("sess-new")
        stats = pool.get_pool_stats()
        assert stats["total_sessions"] == 2

    def test_eviction_frees_slots_for_new_sessions(self):
        """After eviction, the freed slot can be used by a new session."""
        pool = self._make_pool(timeout=0.2)
        # Fill both slots
        pool.acquire_session_slot("sess-1")
        pool.acquire_session_slot("sess-2")
        # Pool is now saturated
        assert pool.acquire_session_slot("sess-3") is None
        # Wait for eviction
        time.sleep(0.5)
        # Acquire should trigger eviction and succeed
        result = pool.acquire_session_slot("sess-3")
        assert result is not None


# ---------------------------------------------------------------------------
# Duplicate pool_key validation
# ---------------------------------------------------------------------------


class TestDuplicatePoolKey:
    def test_duplicate_pool_key_warns(self, caplog):
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 1},
                {"model": "glm-5", "provider": "zai", "max_concurrent": 2},  # duplicate pool_key
            ],
        }
        pool = SessionModelPool.from_config(config)
        assert pool is not None
        # Should have warned about duplicate
        assert any("duplicate pool_key" in r.message for r in caplog.records)

    def test_duplicate_pool_key_both_entries_retained(self):
        """Duplicate pool_key is deduplicated — last entry wins, first discarded."""
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 1},
                {"model": "glm-5", "provider": "zai", "max_concurrent": 3},
            ],
        }
        pool = SessionModelPool.from_config(config)
        # Only the last entry is kept (deduplication)
        assert len(pool._entries) == 1
        assert pool._entries[0].max_concurrent == 3


# ---------------------------------------------------------------------------
# Auxiliary slot blocking edge cases
# ---------------------------------------------------------------------------


class TestAuxSlotBlocking:
    def test_session_does_not_steal_aux_slot(self):
        """Sessions can only use session slots, not auxiliary reserved slots."""
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 3, "reserved_for_auxiliary": 2},
            ],
        }
        pool = SessionModelPool.from_config(config)
        # Only 1 session slot (3 - 2 reserved = 1)
        pool.acquire_session_slot("sess-1")
        # Should be saturated for sessions
        result = pool.acquire_session_slot("sess-2")
        assert result is None

    def test_aux_slot_independent_of_session_slots(self):
        """Auxiliary slots don't consume session capacity."""
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-5", "provider": "zai", "max_concurrent": 3, "reserved_for_auxiliary": 2},
            ],
        }
        pool = SessionModelPool.from_config(config)
        # Use all session slots (3 - 2 reserved = 1)
        pool.acquire_session_slot("sess-1")
        # Auxiliary should still have 2 reserved slots available
        assert pool.acquire_auxiliary_slot("glm-5", "zai") is True
        assert pool.acquire_auxiliary_slot("glm-5", "zai") is True
        # Third aux should fail (reserved=2)
        assert pool.acquire_auxiliary_slot("glm-5", "zai", timeout=0.1) is False

    def test_concurrent_aux_acquire(self):
        """Multiple auxiliary callers compete for limited slots."""
        config = {
            "enabled": True,
            "pool": [
                {"model": "glm-4.6", "provider": "zai", "max_concurrent": 4, "reserved_for_auxiliary": 2},
            ],
        }
        pool = SessionModelPool.from_config(config)
        results = []
        errors = []

        def acquire_aux(idx):
            try:
                ok = pool.acquire_auxiliary_slot("glm-4.6", "zai", timeout=0.5)
                results.append(ok)
                if ok:
                    time.sleep(0.2)
                    pool.release_auxiliary_slot("glm-4.6", "zai")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquire_aux, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # At least 2 should succeed (reserved=2), possibly more with timing
        successes = [r for r in results if r is True]
        assert len(successes) >= 2
