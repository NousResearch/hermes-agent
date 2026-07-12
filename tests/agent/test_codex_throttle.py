"""Unit tests for the Codex outbound-call burst limiter (``agent.codex_throttle``).

Covers the four behaviours the HEMP 2026-07-12 contract pins for the limiter
that paces ``openai-codex`` Responses API calls (the 13:14 429 credential
cascade fix):

1. **Rolling-window pacing** — under the cap ``acquire`` admits instantly; at
   the cap it waits *exactly* until the oldest in-window call ages out, then
   admits (drop-free). ``window_count`` prunes aged events.
2. **Off / unregistered no-op** — ``enabled: false`` or ``calls_per_hour <= 0``
   makes the limiter inert; the sync bridge is a no-op when unconfigured,
   disabled, or when no gateway loop is registered (CLI / cron / tests).
3. **Loop-thread guard** — ``throttle_codex_call_blocking`` returns immediately
   when already running on the gateway loop (``_running_loop_is``), so it never
   submits ``acquire`` back onto its own loop and deadlocks.
4. **FIFO order** — queued callers are admitted in arrival order.

The limiter takes injectable ``time_func`` / ``sleep_func`` so window maths can
be driven deterministically without real wall-clock waits.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

import agent.codex_throttle as ct
from agent.codex_throttle import (
    CodexRateLimiter,
    configure,
    configure_from_config,
    get_limiter,
    set_gateway_loop,
    throttle_codex_call_blocking,
)


@pytest.fixture(autouse=True)
def _reset_codex_throttle_singletons(monkeypatch):
    """Isolate the process-wide limiter / loop between tests in this file."""
    monkeypatch.setattr(ct, "_limiter", None)
    monkeypatch.setattr(ct, "_gateway_loop", None)
    yield


class _FakeClock:
    """Injectable deterministic clock + async sleep that advances it.

    Every ``await sleep(n)`` records ``n`` and moves the clock forward by ``n``,
    so a burst past the cap is driven with zero real wall-clock time.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# --------------------------------------------------------------------------- #
# 1. Rolling-window pacing
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_under_cap_admits_immediately_without_waiting():
    clock = _FakeClock()
    lim = CodexRateLimiter(
        calls_per_hour=3,
        window_seconds=100.0,
        max_wait_seconds=None,
        time_func=clock.time,
        sleep_func=clock.sleep,
    )
    for _ in range(3):
        await lim.acquire()
    assert lim.window_count() == 3
    assert clock.sleeps == []  # never waited while under the cap


@pytest.mark.asyncio
async def test_at_cap_waits_exactly_until_oldest_ages_out():
    clock = _FakeClock(start=1000.0)
    lim = CodexRateLimiter(
        calls_per_hour=3,
        window_seconds=100.0,
        max_wait_seconds=None,
        time_func=clock.time,
        sleep_func=clock.sleep,
    )
    for _ in range(3):  # fill the window at t=1000
        await lim.acquire()

    await lim.acquire()  # 4th: at the cap -> must wait one full window

    # oldest in-window call was at t=1000; it ages out at 1000+100 -> wait == 100
    assert clock.sleeps == [100.0]
    assert clock.now == 1100.0
    # the three t=1000 events aged out; only the freshly admitted 4th remains
    assert lim.window_count() == 1


@pytest.mark.asyncio
async def test_window_count_prunes_events_outside_window():
    clock = _FakeClock(start=500.0)
    lim = CodexRateLimiter(
        calls_per_hour=10,
        window_seconds=60.0,
        max_wait_seconds=None,
        time_func=clock.time,
        sleep_func=clock.sleep,
    )
    await lim.acquire()
    await lim.acquire()
    assert lim.window_count() == 2

    clock.now += 61.0  # both admitted calls now older than the 60s window
    assert lim.window_count() == 0


@pytest.mark.asyncio
async def test_max_wait_safety_valve_admits_drop_free_under_sustained_overload():
    """Under sustained > cap load, a single call never blocks longer than
    ``max_wait_seconds`` — it proceeds drop-free rather than pin its thread."""
    clock = _FakeClock(start=0.0)
    lim = CodexRateLimiter(
        calls_per_hour=1,
        window_seconds=1000.0,  # oldest call would otherwise block ~1000s
        max_wait_seconds=5.0,
        time_func=clock.time,
        sleep_func=clock.sleep,
    )
    await lim.acquire()  # fills the single slot at t=0
    await lim.acquire()  # at cap; wait capped at max_wait then admit drop-free

    assert sum(clock.sleeps) == pytest.approx(5.0)  # never waited the full window
    assert lim.window_count() == 2  # both admitted (drop-free), none dropped


# --------------------------------------------------------------------------- #
# 2. Off switch / no-op
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_disabled_by_zero_cap_is_noop():
    lim = CodexRateLimiter(calls_per_hour=0)
    assert lim.active is False
    await lim.acquire()  # completes through instantly, records nothing
    assert lim.window_count() == 0


@pytest.mark.asyncio
async def test_disabled_by_enabled_false_is_noop():
    lim = CodexRateLimiter(calls_per_hour=100, enabled=False)
    assert lim.active is False
    await lim.acquire()
    assert lim.window_count() == 0


def test_configure_from_config_missing_block_fails_safe_to_disabled():
    lim = configure_from_config({})
    assert lim.enabled is False
    assert lim.active is False
    assert get_limiter() is lim  # installed as the process-wide singleton


def test_configure_from_config_enables_with_values():
    lim = configure_from_config(
        {"throttle": {"codex": {"enabled": True, "calls_per_hour": 40}}}
    )
    assert lim.active is True
    assert lim.calls_per_hour == 40


def test_configure_from_config_malformed_cap_fails_safe_to_disabled():
    lim = configure_from_config(
        {"throttle": {"codex": {"enabled": True, "calls_per_hour": "oops"}}}
    )
    # a config typo must never wedge codex: malformed cap -> 0 -> inactive
    assert lim.calls_per_hour == 0
    assert lim.active is False


def test_bridge_noop_when_limiter_unconfigured():
    # _limiter is None (reset fixture) -> return without touching any loop.
    assert get_limiter() is None
    throttle_codex_call_blocking()  # must not raise / hang


def test_bridge_noop_when_limiter_disabled():
    configure(enabled=False, calls_per_hour=0)
    throttle_codex_call_blocking()  # inactive limiter -> no-op


def test_bridge_noop_when_no_gateway_loop_registered():
    configure(enabled=True, calls_per_hour=5)
    set_gateway_loop(None)  # CLI / cron / tests: no loop -> bridge is a no-op
    throttle_codex_call_blocking()


# --------------------------------------------------------------------------- #
# 3. Loop-thread guard (_running_loop_is)
# --------------------------------------------------------------------------- #
def test_running_loop_is_false_outside_event_loop():
    other = asyncio.new_event_loop()
    try:
        # No loop is running in this sync test -> get_running_loop() raises ->
        # guarded to False.
        assert ct._running_loop_is(other) is False
    finally:
        other.close()


@pytest.mark.asyncio
async def test_running_loop_is_true_for_the_current_running_loop():
    loop = asyncio.get_running_loop()
    assert ct._running_loop_is(loop) is True


@pytest.mark.asyncio
async def test_bridge_returns_early_when_called_on_gateway_loop_thread(monkeypatch):
    """If invoked from the gateway loop thread itself, the bridge must bail out
    *before* submitting ``acquire`` onto that loop — otherwise it would block
    the loop waiting on its own future (deadlock)."""
    lim = configure(enabled=True, calls_per_hour=1)  # active

    calls = {"n": 0}

    async def _spy_acquire():
        calls["n"] += 1

    monkeypatch.setattr(lim, "acquire", _spy_acquire)

    loop = asyncio.get_running_loop()
    set_gateway_loop(loop)  # gateway loop == the loop we're running on now

    throttle_codex_call_blocking(timeout=5)

    # The _running_loop_is guard returned before acquire could be scheduled.
    assert calls["n"] == 0


def test_bridge_admits_via_gateway_loop_from_worker_thread():
    """Happy path: called off-loop, the bridge hops onto the gateway loop and
    completes once admitted — exercising the run_coroutine_threadsafe wiring."""
    loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()
    try:
        deadline = time.time() + 5
        while not loop.is_running() and time.time() < deadline:
            time.sleep(0.005)
        assert loop.is_running()

        configure(enabled=True, calls_per_hour=5)  # under cap -> admit instantly
        set_gateway_loop(loop)

        # Called from THIS (worker) thread, not the loop thread.
        throttle_codex_call_blocking(timeout=5)

        # future.result() only returns after acquire() recorded the admission.
        assert get_limiter().window_count() == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


# --------------------------------------------------------------------------- #
# 4. FIFO order under contention
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_acquire_preserves_fifo_order_under_contention():
    """At the cap, queued callers are admitted strictly in arrival order."""
    lim = CodexRateLimiter(
        calls_per_hour=1, window_seconds=0.05, max_wait_seconds=None
    )
    admitted: list[str] = []

    await lim.acquire()  # seed takes the single slot
    admitted.append("seed")

    async def worker(name: str) -> None:
        await lim.acquire()
        admitted.append(name)

    tasks = []
    for name in ("a", "b", "c", "d"):
        tasks.append(asyncio.create_task(worker(name)))
        # Yield so this worker reaches & queues on the internal lock (in order)
        # before the next is created — pins deterministic arrival order.
        await asyncio.sleep(0)

    await asyncio.gather(*tasks)
    assert admitted == ["seed", "a", "b", "c", "d"]
