"""Tests for agent.concurrency.ConcurrencySemaphore."""

import threading
import time

import pytest

from agent.concurrency import ConcurrencySemaphore, get_semaphore, reset_registry
from agent.concurrency import get_configured_max_concurrent
from agent.model_metadata import get_default_concurrency


class TestConcurrencySemaphore:

    def test_acquire_within_limit(self):
        sem = ConcurrencySemaphore(max_concurrent=2)
        assert sem.acquire() is True
        assert sem.acquire() is True
        assert sem.active == 2

    def test_acquire_blocks_at_limit(self):
        sem = ConcurrencySemaphore(max_concurrent=1)
        assert sem.acquire() is True
        assert sem.acquire(timeout=0.05) is False

    def test_release_unblocks_waiter(self):
        sem = ConcurrencySemaphore(max_concurrent=1)
        sem.acquire()

        result = [None]

        def waiter():
            result[0] = sem.acquire(timeout=2)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)  # let thread register as waiter
        sem.release()
        t.join(timeout=3)
        assert result[0] is True

    def test_release_below_zero_is_safe(self):
        sem = ConcurrencySemaphore(max_concurrent=1)
        sem.release()
        assert sem.active == 0

    def test_priority_waiter_served_before_non_priority(self):
        sem = ConcurrencySemaphore(max_concurrent=1)
        sem.acquire()

        order: list[str] = []

        def non_priority_waiter():
            sem.acquire(timeout=2)
            order.append("non-priority")
            sem.release()

        def priority_waiter():
            sem.acquire(priority=True, timeout=2)
            order.append("priority")
            sem.release()

        t_np = threading.Thread(target=non_priority_waiter)
        t_np.start()
        time.sleep(0.02)  # let non-priority register first

        t_p = threading.Thread(target=priority_waiter)
        t_p.start()
        time.sleep(0.02)  # let priority register

        sem.release()  # unblock one waiter
        t_p.join(timeout=3)
        t_np.join(timeout=3)

        assert order[0] == "priority"

    def test_slot_context_manager_releases_on_exception(self):
        sem = ConcurrencySemaphore(max_concurrent=1)
        with pytest.raises(RuntimeError):
            with sem.slot() as acquired:
                assert acquired is True
                raise RuntimeError("boom")
        assert sem.active == 0

    def test_zero_timeout_is_nonblocking(self):
        sem = ConcurrencySemaphore(max_concurrent=1)
        sem.acquire()
        assert sem.acquire(timeout=0) is False

    def test_invalid_max_concurrent_raises(self):
        with pytest.raises(ValueError):
            ConcurrencySemaphore(max_concurrent=0)
        with pytest.raises(ValueError):
            ConcurrencySemaphore(max_concurrent=-1)


@pytest.mark.asyncio
class TestConcurrencySemaphoreAsync:

    async def test_async_slot_acquire_and_release(self):
        sem = ConcurrencySemaphore(max_concurrent=1)
        async with sem.async_slot() as acquired:
            assert acquired is True
            assert sem.active == 1
        assert sem.active == 0

    async def test_async_slot_timeout(self):
        sem = ConcurrencySemaphore(max_concurrent=1)
        sem.acquire()
        async with sem.async_slot(timeout=0.05) as acquired:
            assert acquired is False
        sem.release()


class TestRegistry:
    def setup_method(self):
        reset_registry()

    def test_get_semaphore_creates_new(self):
        sem = get_semaphore("zai", "key-1", max_concurrent=1)
        assert isinstance(sem, ConcurrencySemaphore)
        assert sem.max_concurrent == 1

    def test_get_semaphore_returns_same_instance(self):
        sem1 = get_semaphore("zai", "key-1", max_concurrent=1)
        sem2 = get_semaphore("zai", "key-1", max_concurrent=5)
        assert sem1 is sem2
        assert sem1.max_concurrent == 1  # first call's value wins

    def test_different_keys_get_different_semaphores(self):
        sem1 = get_semaphore("zai", "key-1", max_concurrent=1)
        sem2 = get_semaphore("zai", "key-2", max_concurrent=2)
        assert sem1 is not sem2

    def test_different_providers_get_different_semaphores(self):
        sem1 = get_semaphore("zai", "key-1", max_concurrent=1)
        sem2 = get_semaphore("kimi", "key-1", max_concurrent=1)
        assert sem1 is not sem2


class TestConfiguredConcurrency:

    def test_model_config_override(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {
                    "provider": "zai",
                    "default": "glm-5.1",
                    "max_concurrent": 3,
                }
            },
        )

        assert get_configured_max_concurrent(
            provider="zai",
            model="glm-5.1",
        ) == 3

    def test_custom_provider_override_by_name(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "custom_providers": [
                    {
                        "name": "My ZAI",
                        "base_url": "https://api.z.ai/api/coding/paas/v4",
                        "max_concurrent": 2,
                    }
                ]
            },
        )

        assert get_configured_max_concurrent(
            provider="custom:my-zai",
            model="glm-5",
        ) == 2

    def test_custom_provider_model_override_wins(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "custom_providers": [
                    {
                        "name": "remote",
                        "base_url": "https://remote.example/v1",
                        "max_concurrent": 4,
                        "models": {
                            "glm-5.1": {
                                "max_concurrent": 1,
                            }
                        },
                    }
                ]
            },
        )

        assert get_configured_max_concurrent(
            provider="custom",
            model="glm-5.1",
            base_url="https://remote.example/v1",
        ) == 1

    def test_invalid_override_is_ignored(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"model": {"provider": "zai", "max_concurrent": 0}},
        )

        assert get_configured_max_concurrent(provider="zai") is None


class TestDefaultConcurrency:

    def test_zai_glm_5_1_returns_1(self):
        assert get_default_concurrency("zai", "glm-5.1") == 1

    def test_zai_glm_5_returns_2(self):
        assert get_default_concurrency("zai", "glm-5") == 2

    def test_zai_glm_4_5_returns_10(self):
        assert get_default_concurrency("zai", "glm-4.5") == 10

    def test_zai_unknown_model_returns_provider_default(self):
        assert get_default_concurrency("zai", "glm-999") == 10

    def test_kimi_returns_1(self):
        assert get_default_concurrency("kimi-coding", None) == 1

    def test_unknown_provider_returns_global_default(self):
        assert get_default_concurrency("openrouter", None) >= 8

    def test_none_provider_returns_global_default(self):
        assert get_default_concurrency(None, None) >= 1
