"""Tests for the SignalAttachmentScheduler token-bucket simulator."""
import asyncio
import threading
import time

import pytest

from gateway.platforms import signal_rate_limit as rate_limit_module
from gateway.platforms.signal_rate_limit import (
    SignalAttachmentScheduler,
    _reset_scheduler,
)


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    """Drop the process-wide scheduler so each test gets a clean bucket."""
    _reset_scheduler()
    yield
    _reset_scheduler()


def _patch_sleep_and_time(monkeypatch, capture: list):
    """Replace asyncio.sleep inside the scheduler module so tests don't
    actually wait and advances time.monotonic to simulate time passing.
    Captures the requested duration per call."""
    offset = 0.0
    async def _fake_sleep(seconds):
        capture.append(seconds)
        nonlocal offset
        offset += seconds

    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.asyncio.sleep", _fake_sleep
    )
    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.time.monotonic", lambda: offset
    )




class TestEstimateWait:

    def test_proportional_to_deficit_when_empty(self, monkeypatch):
        """Freeze monotonic so estimate_wait doesn't see fractional refill."""
        s = SignalAttachmentScheduler()
        s.tokens = 0.0
        frozen = s.last_refill
        monkeypatch.setattr(
            "gateway.platforms.signal_rate_limit.time.monotonic", lambda: frozen
        )
        # 32 tokens at 0.25 tokens/sec = 128s
        assert s.estimate_wait(32) == pytest.approx(32 / s.refill_rate)
        assert s.estimate_wait(1) == pytest.approx(1 / s.refill_rate)


class TestAcquire:

    @pytest.mark.asyncio
    async def test_acquire_within_capacity_no_sleep(self, monkeypatch):
        sleeps: list = []
        _patch_sleep_and_time(monkeypatch, sleeps)

        s = SignalAttachmentScheduler()
        wait = await s.acquire(10)
        await s.report_rpc_duration(0.001, 10)  # actually deduct tokens

        assert wait == 0.0
        assert sleeps == []
        assert s.tokens == s.capacity - 10

    @pytest.mark.asyncio
    async def test_acquire_when_empty_sleeps_for_deficit(self, monkeypatch):
        sleeps: list = []
        _patch_sleep_and_time(monkeypatch, sleeps)
        s = SignalAttachmentScheduler()

        s.tokens = 0.0
        wait = await s.acquire(32)
        await s.report_rpc_duration(1e-12, 32)

        # 32 tokens at default 0.25 tokens/sec = 128s
        expected = 32 / s.refill_rate
        assert wait == pytest.approx(expected)
        assert sleeps == [pytest.approx(expected)]
        # After sleep+acquire+rpc call, the bucket is empty again.
        assert s.tokens == pytest.approx(0.0)


class TestFeedback:
    def test_calibrates_refill_rate_from_retry_after(self):
        s = SignalAttachmentScheduler()
        original = s.refill_rate
        s.feedback(retry_after=42.0, n_attempted=1)
        assert s.refill_rate == pytest.approx(1.0 / 42.0)
        assert s.refill_rate != original


class TestRefillClamping:
    def test_refill_does_not_exceed_capacity(self, monkeypatch):
        """Even after a long elapsed window, refill clamps at capacity."""
        s = SignalAttachmentScheduler()
        s.tokens = 0.0
        # Pretend a year passed.
        monkeypatch.setattr(
            "gateway.platforms.signal_rate_limit.time.monotonic",
            lambda: s.last_refill + 365 * 24 * 3600,
        )
        s._refill()
        assert s.tokens == s.capacity


class TestFifoAcquire:
    @pytest.mark.asyncio
    async def test_concurrent_acquires_serialize(self, monkeypatch):
        """Two coroutines acquiring full capacity each: the second waits
        in the lock queue until the first finishes its bucket math + sleep.
        Demonstrates the FIFO fairness across sessions."""
        sleeps: list = []
        _patch_sleep_and_time(monkeypatch, sleeps)
        s = SignalAttachmentScheduler()

        results: list = []

        async def worker(label: str):
            wait = await s.acquire(int(s.capacity))
            await s.report_rpc_duration(1e-12, int(s.capacity))
            results.append((label, wait))

        # Launch in order; FIFO means A finishes first, then B.
        await asyncio.gather(worker("A"), worker("B"))

        assert [r[0] for r in results] == ["A", "B"]
        # A had a full bucket (no wait). B waited a full refill.
        assert results[0][1] == 0.0
        assert results[1][1] == pytest.approx(s.capacity / s.refill_rate)



    def test_concurrent_first_touch_returns_one_instance(self, monkeypatch):
        """Regression for #24744: racing first touches must converge on one scheduler."""
        workers = 16
        gate = threading.Barrier(workers)
        constructed: list = []
        real_cls = rate_limit_module.SignalAttachmentScheduler

        class SlowScheduler(real_cls):  # type: ignore[valid-type, misc]
            def __init__(self, *args, **kwargs):
                time.sleep(0.05)  # hold the check-then-act window open
                super().__init__(*args, **kwargs)
                constructed.append(1)

        monkeypatch.setattr(rate_limit_module, "SignalAttachmentScheduler", SlowScheduler)
        results: list = []

        def worker():
            gate.wait(timeout=10)  # release every thread into get_scheduler at once
            results.append(get_scheduler())

        threads = [threading.Thread(target=worker) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads)
        assert len(constructed) == 1
        assert len(results) == workers
        assert all(r is results[0] for r in results)

