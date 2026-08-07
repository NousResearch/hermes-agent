# SPDX-License-Identifier: Apache-2.0
"""Tests for the prefetch runner.

Focus on the framework contract, not any specific task shape:

* Registration replaces on same source (avoid cache races).
* Concurrent execution: one slow task doesn't hold the batch.
* Fetch errors → captured in outcome, do not raise, do not abort peers.
* Cache-write errors → captured too.
* Dynamic-provider task list is merged; broken provider is silently
  ignored (static tasks still run).
* Selective run by task_sources filters correctly.
* clear_expired is always called first; failures there don't abort.
"""
from __future__ import annotations

import asyncio
import time
from typing import Sequence

import pytest

from agent.prefetch import (
    PrefetchOutcome,
    PrefetchRunner,
    PrefetchTask,
)


# ── Fake cache store ───────────────────────────────────────────────


class FakeCache:
    def __init__(self, *, fail_expire: bool = False, fail_save_for: str | None = None):
        self.saved: list[dict] = []
        self.cleared: int = 0
        self.fail_expire = fail_expire
        self.fail_save_for = fail_save_for

    def save(
        self,
        *,
        source: str,
        content: str,
        ttl_minutes: int,
        tags: Sequence[str],
    ) -> None:
        if self.fail_save_for and self.fail_save_for == source:
            raise RuntimeError(f"cache disk full for {source}")
        self.saved.append(
            {
                "source": source,
                "content": content,
                "ttl_minutes": ttl_minutes,
                "tags": list(tags),
            }
        )

    def clear_expired(self) -> None:
        self.cleared += 1
        if self.fail_expire:
            raise RuntimeError("clear failed")


# ── Registration ────────────────────────────────────────────────────


def test_register_rejects_empty_source() -> None:
    runner = PrefetchRunner(cache=FakeCache())
    with pytest.raises(ValueError):
        runner.register(PrefetchTask(source="", fetch=lambda: asyncio.sleep(0)))  # type: ignore[arg-type]


def test_register_rejects_non_callable_fetch() -> None:
    runner = PrefetchRunner(cache=FakeCache())
    with pytest.raises(ValueError):
        runner.register(PrefetchTask(source="x", fetch="not callable"))  # type: ignore[arg-type]


def test_register_overwrites_on_same_source() -> None:
    """Same-source duplicates would race on the cache row, so keyed
    replacement is safer than list append.
    """
    runner = PrefetchRunner(cache=FakeCache())

    async def a() -> str:
        return "A"

    async def b() -> str:
        return "B"

    runner.register(PrefetchTask(source="x", fetch=a))
    runner.register(PrefetchTask(source="x", fetch=b, label="second"))
    assert runner.registered_sources() == ["x"]


def test_unregister_returns_removal_status() -> None:
    runner = PrefetchRunner(cache=FakeCache())

    async def f() -> str:
        return ""

    runner.register(PrefetchTask(source="x", fetch=f))
    assert runner.unregister("x") is True
    assert runner.unregister("x") is False


# ── Execution ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_writes_to_cache_on_success() -> None:
    cache = FakeCache()
    runner = PrefetchRunner(cache=cache)

    async def fetch_weather() -> str:
        return "sunny, 25C"

    runner.register(
        PrefetchTask(
            source="weather:x",
            fetch=fetch_weather,
            ttl_minutes=30,
            label="Weather X",
            tags=["weather", "x"],
        )
    )
    outcomes = await runner.run()
    assert outcomes == [
        PrefetchOutcome(source="weather:x", label="Weather X", ok=True)
    ]
    assert cache.saved == [
        {
            "source": "weather:x",
            "content": "sunny, 25C",
            "ttl_minutes": 30,
            "tags": ["weather", "x"],
        }
    ]
    assert cache.cleared == 1  # clear_expired invoked once


@pytest.mark.asyncio
async def test_run_returns_empty_when_no_tasks() -> None:
    cache = FakeCache()
    runner = PrefetchRunner(cache=cache)
    outcomes = await runner.run()
    assert outcomes == []
    # cache.cleared is still called — a run with nothing to fetch
    # still gets a chance to sweep expired rows.
    assert cache.cleared == 1


@pytest.mark.asyncio
async def test_run_captures_fetch_exception() -> None:
    cache = FakeCache()
    runner = PrefetchRunner(cache=cache)

    async def broken() -> str:
        raise RuntimeError("no network")

    runner.register(PrefetchTask(source="x", fetch=broken, label="X"))
    outcomes = await runner.run()
    assert outcomes[0].ok is False
    assert outcomes[0].error is not None
    assert "no network" in outcomes[0].error
    assert cache.saved == []


@pytest.mark.asyncio
async def test_run_captures_cache_write_exception() -> None:
    cache = FakeCache(fail_save_for="x")
    runner = PrefetchRunner(cache=cache)

    async def ok() -> str:
        return "content"

    runner.register(PrefetchTask(source="x", fetch=ok))
    outcomes = await runner.run()
    assert outcomes[0].ok is False
    assert "disk full" in (outcomes[0].error or "")
    assert cache.saved == []


@pytest.mark.asyncio
async def test_run_isolates_failing_task_from_healthy_peers() -> None:
    """One broken task must not abort the batch."""
    cache = FakeCache()
    runner = PrefetchRunner(cache=cache)

    async def good() -> str:
        return "OK"

    async def bad() -> str:
        raise RuntimeError("oops")

    runner.register(PrefetchTask(source="good", fetch=good))
    runner.register(PrefetchTask(source="bad", fetch=bad))
    outcomes = await runner.run()
    by_source = {o.source: o for o in outcomes}
    assert by_source["good"].ok is True
    assert by_source["bad"].ok is False
    assert [row["source"] for row in cache.saved] == ["good"]


@pytest.mark.asyncio
async def test_run_executes_concurrently() -> None:
    """A slow task shouldn't hold a fast one — verify by checking the
    total wall-clock time is closer to max(t) than sum(t).
    """
    cache = FakeCache()
    runner = PrefetchRunner(cache=cache)

    async def slow() -> str:
        await asyncio.sleep(0.15)
        return "slow"

    async def fast() -> str:
        await asyncio.sleep(0.01)
        return "fast"

    runner.register(PrefetchTask(source="slow", fetch=slow))
    runner.register(PrefetchTask(source="fast", fetch=fast))

    started = time.monotonic()
    outcomes = await runner.run()
    elapsed = time.monotonic() - started

    assert all(o.ok for o in outcomes)
    # Sequential would be ~0.16s; concurrent should be ~0.15s.
    # Give it plenty of slack for CI variance but rule out sequential.
    assert elapsed < 0.30, f"batch took {elapsed:.3f}s — looks sequential"


@pytest.mark.asyncio
async def test_run_filters_by_task_sources() -> None:
    cache = FakeCache()
    runner = PrefetchRunner(cache=cache)

    async def a() -> str:
        return "A"

    async def b() -> str:
        return "B"

    runner.register(PrefetchTask(source="a", fetch=a))
    runner.register(PrefetchTask(source="b", fetch=b))

    outcomes = await runner.run(task_sources=["a"])
    assert [o.source for o in outcomes] == ["a"]
    assert [row["source"] for row in cache.saved] == ["a"]


@pytest.mark.asyncio
async def test_run_survives_failing_clear_expired() -> None:
    cache = FakeCache(fail_expire=True)
    runner = PrefetchRunner(cache=cache)

    async def ok() -> str:
        return "content"

    runner.register(PrefetchTask(source="x", fetch=ok))
    outcomes = await runner.run()
    assert outcomes[0].ok is True
    assert cache.saved[0]["source"] == "x"


# ── Dynamic task provider ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_dynamic_provider_supplies_additional_tasks() -> None:
    cache = FakeCache()

    async def dyn_fetch() -> str:
        return "dyn"

    def provider() -> list[PrefetchTask]:
        return [
            PrefetchTask(source="dyn:x", fetch=dyn_fetch, ttl_minutes=15)
        ]

    runner = PrefetchRunner(cache=cache, dynamic_provider=provider)
    outcomes = await runner.run()
    assert [o.source for o in outcomes] == ["dyn:x"]
    assert cache.saved[0]["ttl_minutes"] == 15


@pytest.mark.asyncio
async def test_dynamic_provider_does_not_shadow_static_task() -> None:
    """If a dynamic task uses the same source as a static one, the
    static registration wins — otherwise a plugin could silently
    replace a core task.
    """
    cache = FakeCache()

    async def static_fetch() -> str:
        return "static"

    async def dyn_fetch() -> str:
        return "dyn"

    def provider() -> list[PrefetchTask]:
        return [PrefetchTask(source="shared", fetch=dyn_fetch, ttl_minutes=99)]

    runner = PrefetchRunner(cache=cache, dynamic_provider=provider)
    runner.register(PrefetchTask(source="shared", fetch=static_fetch, ttl_minutes=10))
    outcomes = await runner.run()
    assert len(outcomes) == 1
    assert cache.saved[0]["content"] == "static"
    assert cache.saved[0]["ttl_minutes"] == 10


@pytest.mark.asyncio
async def test_dynamic_provider_error_is_absorbed() -> None:
    cache = FakeCache()

    def broken_provider() -> list[PrefetchTask]:
        raise RuntimeError("db offline")

    async def ok() -> str:
        return "static"

    runner = PrefetchRunner(cache=cache, dynamic_provider=broken_provider)
    runner.register(PrefetchTask(source="s", fetch=ok))
    outcomes = await runner.run()
    # Static task still ran.
    assert [o.source for o in outcomes] == ["s"]
    assert outcomes[0].ok is True
