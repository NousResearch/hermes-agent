"""Tests for the SignalAttachmentScheduler token-bucket simulator."""
import asyncio
import time

import pytest

from gateway.platforms.signal_rate_limit import (
    SIGNAL_MAX_ATTACHMENTS_PER_MSG,
    SIGNAL_RATE_LIMIT_BUCKET_CAPACITY,
    SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER,
    SignalAttachmentScheduler,
    get_scheduler,
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


class TestSchedulerInitialState:
    def test_default_capacity_matches_signal_cap(self):
        s = SignalAttachmentScheduler()
        assert s.capacity == SIGNAL_RATE_LIMIT_BUCKET_CAPACITY

    def test_default_refill_rate_from_default_retry_after(self):
        s = SignalAttachmentScheduler()
        assert s.refill_rate == pytest.approx(1.0 / SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER)

    def test_starts_full(self):
        s = SignalAttachmentScheduler()
        assert s.tokens == s.capacity


class TestEstimateWait:
    def test_zero_when_bucket_has_enough(self):
        s = SignalAttachmentScheduler()
        assert s.estimate_wait(10) == 0.0
        assert s.estimate_wait(int(s.capacity)) == 0.0

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
    async def test_acquire_zero_is_noop(self, monkeypatch):
        sleeps: list = []
        _patch_sleep_and_time(monkeypatch, sleeps)
        s = SignalAttachmentScheduler()
        original = s.tokens
        wait = await s.acquire(0)
        assert wait == 0.0
        assert sleeps == []
        assert s.tokens == original

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

    @pytest.mark.asyncio
    async def test_back_to_back_acquires_drain_then_wait(self, monkeypatch):
        """Two sequential acquires of capacity each: first immediate,
        second waits a full refill window."""
        sleeps: list = []
        _patch_sleep_and_time(monkeypatch, sleeps)
        s = SignalAttachmentScheduler()

        await s.acquire(int(s.capacity))
        await s.report_rpc_duration(1e-12, int(s.capacity))

        assert sleeps == []  # first batch had a full bucket

        await s.acquire(int(s.capacity))
        await s.report_rpc_duration(1e-12, int(s.capacity))
        # Second batch: no time elapsed (mocked sleep doesn't advance
        # monotonic), tokens still 0 → wait the full capacity / rate.
        assert sleeps == [pytest.approx(s.capacity / s.refill_rate)]

    @pytest.mark.asyncio
    async def test_acquire_more_tokens_than_capacity(self, monkeypatch):
        s = SignalAttachmentScheduler()

        with pytest.raises(Exception):
            await s.acquire(int(s.capacity) + 1)

class TestFeedback:
    def test_calibrates_refill_rate_from_retry_after(self):
        s = SignalAttachmentScheduler()
        original = s.refill_rate
        s.feedback(retry_after=42.0, n_attempted=1)
        assert s.refill_rate == pytest.approx(1.0 / 42.0)
        assert s.refill_rate != original

    def test_none_retry_after_leaves_rate(self):
        s = SignalAttachmentScheduler()
        original = s.refill_rate
        s.feedback(retry_after=None, n_attempted=5)
        assert s.refill_rate == original

    def test_zeros_tokens(self):
        s = SignalAttachmentScheduler()
        assert s.tokens > 0
        s.feedback(retry_after=4.0, n_attempted=1)
        assert s.tokens == 0.0

    @pytest.mark.asyncio
    async def test_acquire_after_feedback_uses_calibrated_rate(self, monkeypatch):
        """signal-cli ≥v0.14.3: server says 'retry_after=42 for one
        token' → next acquire(1) waits 42s. Drops the old defensive
        ``retry_after * 32`` heuristic in favor of the server's
        authoritative per-token value."""
        sleeps: list = []
        _patch_sleep_and_time(monkeypatch, sleeps)
        s = SignalAttachmentScheduler()

        # Initial acquire empties enough; 429 fires.
        await s.acquire(1)
        s.feedback(retry_after=42.0, n_attempted=1)

        # Re-acquire: bucket empty, calibrated rate = 1/42.
        await s.acquire(1)
        assert sleeps == [pytest.approx(42.0)]


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


class TestSingleton:
    def test_get_scheduler_returns_same_instance(self):
        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2

    def test_reset_scheduler_yields_new_instance(self):
        s1 = get_scheduler()
        _reset_scheduler()
        s2 = get_scheduler()
        assert s1 is not s2
