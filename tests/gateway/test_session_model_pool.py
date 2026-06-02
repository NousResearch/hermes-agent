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
        e.session_slots = ["s1"]
        assert e.available_session_slots == 1  # 3 - 1(reserved) - 1(session) = 1

    def test_available_auxiliary_slots(self):
        e = PoolModelEntry(model="glm-5", provider="zai", max_concurrent=3, reserved_for_auxiliary=2)
        e.auxiliary_count = 1
        assert e.available_auxiliary_slots == 1

    def test_is_saturated(self):
        e = PoolModelEntry(model="glm-5", provider="zai", max_concurrent=2, reserved_for_auxiliary=1)
        e.session_slots = ["s1"]
        e.auxiliary_count = 1
        assert e.is_saturated

    def test_not_saturated(self):
        e = PoolModelEntry(model="glm-5", provider="zai", max_concurrent=3, reserved_for_auxiliary=1)
        e.session_slots = ["s1"]
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
        assert result["model"] == "glm-5-turbo"  # highest priority
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

    def test_reconstruct_clears_state(self):
        pool = self._make_pool()
        pool.acquire_session_slot("sess-1")
        pool.acquire_session_slot("sess-2")
        pool.reconstruct_from_active_sessions(["sess-1"])
        assert pool.get_pool_stats()["total_sessions"] == 0


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
