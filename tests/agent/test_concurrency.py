"""Tests for agent.concurrency.ConcurrencySemaphore."""

import threading
import time

import pytest

from agent.concurrency import ConcurrencySemaphore, get_semaphore, reset_registry


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
