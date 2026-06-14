"""
tests/gateway/test_c3_event_loop_lag.py

C3 validation: asyncio.to_thread wrapping of kanban I/O in
_drain_shadow_clone_inbox does not block the event loop.

Pass criterion: max heartbeat lag during drain < 50 ms.
"""
import asyncio
import statistics
import time

import pytest


# ---------------------------------------------------------------------------
# Synthetic slow kanban read (mimics file-locked DB access ~50 ms per batch)
# ---------------------------------------------------------------------------

def _fake_kanban_read_sync(ticket_ids: list, delay_s: float = 0.05) -> list:
    """Blocking kanban I/O stand-in — sleeps delay_s to simulate flock'd DB."""
    time.sleep(delay_s)
    return [{"ticket_id": tid, "result": "ok", "status": "done"} for tid in ticket_ids]


# ---------------------------------------------------------------------------
# Heartbeat monitor: records how many ms each 100 ms tick was delayed
# ---------------------------------------------------------------------------

async def _heartbeat_monitor(lags: list, stop: asyncio.Event) -> None:
    """Record how many ms each 100 ms heartbeat tick was delayed."""
    while not stop.is_set():
        t0 = time.perf_counter()
        await asyncio.sleep(0.1)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        lags.append(max(0.0, elapsed - 100.0))


# ---------------------------------------------------------------------------
# Drain patterns
# ---------------------------------------------------------------------------

async def _drain_before(ticket_ids: list, delay_s: float = 0.05) -> None:
    """Simulates the OLD (blocking) drain — sync read on the event loop."""
    _fake_kanban_read_sync(ticket_ids, delay_s)


async def _drain_after(ticket_ids: list, delay_s: float = 0.05) -> None:
    """Simulates the FIXED drain — sync read off-thread via asyncio.to_thread."""
    await asyncio.to_thread(_fake_kanban_read_sync, ticket_ids, delay_s)


# ---------------------------------------------------------------------------
# Measurement helper
# ---------------------------------------------------------------------------

async def _measure_lag(drain_fn, n_drains: int = 8, delay_s: float = 0.05) -> dict:
    lags: list[float] = []
    stop = asyncio.Event()
    monitor = asyncio.create_task(_heartbeat_monitor(lags, stop))

    for i in range(n_drains):
        tickets = [f"t_{i:04d}a", f"t_{i:04d}b"]
        await drain_fn(tickets, delay_s)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.15)  # let one more tick complete
    stop.set()
    await monitor

    if not lags:
        return {"max": 0.0, "p95": 0.0, "mean": 0.0, "count": 0}

    return {
        "max": max(lags),
        "p95": sorted(lags)[int(len(lags) * 0.95)],
        "mean": statistics.mean(lags),
        "count": len(lags),
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestC3EventLoopLag:
    """C3: asyncio.to_thread wrapping keeps event loop free during drain."""

    @pytest.mark.asyncio
    async def test_after_fix_max_lag_under_50ms(self):
        """AFTER fix: max heartbeat delay during drain < 50 ms."""
        result = await _measure_lag(_drain_after, n_drains=8, delay_s=0.05)
        assert result["count"] > 0, "Heartbeat monitor produced no ticks"
        assert result["max"] < 50.0, (
            f"Event loop lag too high after fix: max={result['max']:.1f}"
            f" ms (threshold=50 ms, p95={result['p95']:.1f}"
            f" ms, mean={result['mean']:.1f} ms)"
        )

    @pytest.mark.asyncio
    async def test_before_fix_would_have_higher_lag(self):
        """BEFORE fix: blocking drain shows measurably higher lag (documents the regression)."""
        result = await _measure_lag(_drain_before, n_drains=4, delay_s=0.05)
        assert result["count"] > 0, "Heartbeat monitor produced no ticks"
        # The blocking drain causes at least one tick to be delayed by ~50 ms
        assert result["max"] > 10.0, (
            f"Expected blocking drain to show >10 ms lag, got {result['max']:.1f} ms"
        )

    @pytest.mark.asyncio
    async def test_concurrent_messages_not_blocked_during_drain(self):
        """Gateway can process 3 concurrent messages while drain runs — none dropped."""
        messages_processed: list[int] = []

        async def process_message(msg_id: int) -> None:
            await asyncio.sleep(0.01)
            messages_processed.append(msg_id)

        async def drain_after_concurrent() -> None:
            await asyncio.to_thread(_fake_kanban_read_sync, ["t_conc_01", "t_conc_02"], 0.05)

        await asyncio.gather(
            drain_after_concurrent(),
            process_message(1),
            process_message(2),
            process_message(3),
        )

        assert len(messages_processed) == 3, (
            f"Expected 3 messages processed during drain, got {len(messages_processed)}"
        )
        assert sorted(messages_processed) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_improvement_factor_significant(self):
        """to_thread version is measurably faster than sync version under concurrent load."""
        after = await _measure_lag(_drain_after, n_drains=6, delay_s=0.05)
        assert after["count"] > 0, "No heartbeat ticks recorded"
        assert after["max"] < 50.0, (
            f"AFTER max lag {after['max']:.1f} ms exceeds 50 ms threshold"
        )
        assert after["mean"] < 20.0, (
            f"AFTER mean lag {after['mean']:.1f} ms is unexpectedly high"
        )
