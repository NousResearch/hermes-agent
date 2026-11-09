"""Ticker loop period contract: the sleep is to a deadline, not for a duration.

Both ticker loops (single-profile and multiplex) must schedule tick k at
``start + k * interval`` regardless of how long each tick took, and a tick that
overruns the interval must re-baseline rather than fire a burst of zero-length waits.
Behaviour contract per ``cron/AGENTS.md``: loose bounds on a virtual clock, no sleeping.
"""
from unittest.mock import patch

import pytest

import cron.scheduler_provider as sp

INTERVAL = 60.0


class _FakeTime:
    now = 100.0

    @classmethod
    def monotonic(cls):
        return cls.now


class _RecordingStop:
    def __init__(self, cycles):
        self.waits = []
        self._cycles = cycles

    def is_set(self):
        return len(self.waits) >= self._cycles

    def wait(self, timeout=None):
        self.waits.append(timeout)
        _FakeTime.now += timeout
        return self.is_set()


def _drive(loop, tmp_path, monkeypatch, *, tick_cost, cycles):
    """Run the real ticker loop on a virtual clock; return (waits, elapsed virtual seconds)."""
    _FakeTime.now = 100.0
    monkeypatch.setattr(sp, "time", _FakeTime)
    stop = _RecordingStop(cycles)
    provider = sp.InProcessCronScheduler()

    def fake_tick(*args, **kwargs):
        _FakeTime.now += tick_cost
        return 0

    (tmp_path / "cron").mkdir(exist_ok=True)
    profile_homes = None if loop == "single" else [("p1", tmp_path)]
    with patch.object(provider, "recover_interrupted", return_value=0), \
         patch("cron.scheduler.tick", side_effect=fake_tick), \
         patch("cron.jobs.record_ticker_heartbeat"), \
         patch("cron.jobs.clear_ticker_error"):
        provider.start(stop, interval=INTERVAL, profile_homes=profile_homes)
    return stop.waits, _FakeTime.now - 100.0


@pytest.mark.parametrize("loop", ["single", "multiplex"])
def test_ticker_period_does_not_accumulate_tick_cost(loop, tmp_path, monkeypatch):
    """Scheduled instants stay at k * interval: error must not grow with k."""
    cycles = 20
    waits, elapsed = _drive(loop, tmp_path, monkeypatch, tick_cost=0.25, cycles=cycles)
    assert len(waits) == cycles
    assert all(0 < w < INTERVAL for w in waits), waits
    # Sleep-after-work would put this at cycles * (interval + 0.25) = +5s of drift.
    assert abs(elapsed - cycles * INTERVAL) < 0.1


@pytest.mark.parametrize("loop", ["single", "multiplex"])
def test_ticker_overrun_rebaselines_instead_of_bursting(loop, tmp_path, monkeypatch):
    """A tick longer than the interval is followed by normal waits, not zero-length ones."""
    waits, _ = _drive(loop, tmp_path, monkeypatch, tick_cost=1.7 * INTERVAL, cycles=3)
    assert all(w == pytest.approx(INTERVAL) for w in waits), waits
