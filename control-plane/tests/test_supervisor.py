"""Supervisor lifecycle tests against a fake backend (no Hermes, no network)."""
import asyncio

import pytest

from orchard.backends.base import WorkerBackend
from orchard.config import Settings
from orchard.models import Employee
from orchard.supervisor import CapacityFull, Supervisor


class FakeBackend(WorkerBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self.awake: set[str] = set()
        self.wake_calls: list[str] = []
        self.sleep_calls: list[str] = []

    async def is_ready(self, employee):
        return employee.id in self.awake

    async def ensure_ready(self, employee):
        self.wake_calls.append(employee.id)
        self.awake.add(employee.id)

    async def send(self, employee, session, message):
        assert employee.id in self.awake
        return f"[{employee.id}] {message}"

    async def sleep(self, employee):
        self.sleep_calls.append(employee.id)
        self.awake.discard(employee.id)


def _emp(i):
    return Employee(i, i, f"mm-{i}", 0.0)


@pytest.mark.asyncio
async def test_wakes_once_then_reuses():
    s = Settings()
    be = FakeBackend(s)
    sup = Supervisor(s, be)
    assert await sup.handle(_emp("alice"), "c1", "hi") == "[alice] hi"
    assert await sup.handle(_emp("alice"), "c1", "again") == "[alice] again"
    assert be.wake_calls == ["alice"]  # woken exactly once


@pytest.mark.asyncio
async def test_capacity_evicts_lru():
    s = Settings()
    s.supervisor.max_active_workers = 2
    be = FakeBackend(s)
    sup = Supervisor(s, be)
    await sup.handle(_emp("a"), "c", "1")
    await sup.handle(_emp("b"), "c", "1")
    await sup.handle(_emp("c"), "c", "1")  # exceeds cap -> evict LRU (a)
    assert "a" in be.sleep_calls
    assert be.awake == {"b", "c"}


@pytest.mark.asyncio
async def test_idle_reaper_sleeps_stale_workers():
    s = Settings()
    s.supervisor.idle_ttl_seconds = 0  # everything is immediately stale
    be = FakeBackend(s)
    sup = Supervisor(s, be)
    await sup.handle(_emp("a"), "c", "1")
    reaped = await sup._reap_once()
    assert reaped == ["a"]
    assert "a" in be.sleep_calls
    assert be.awake == set()
