"""NEEDS_ATTENTION after a suspended host (#126825): Windows' ``time.monotonic()`` keeps counting
through Modern Standby, so a platform queued just before the laptop slept looked hours-failed on
the first watcher pass after resume, before it had retried once. Both reconnect lanes (the launch
profile's watcher and a secondary's loop) are driven for real; only the clock and connect are fake."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import gateway.run as gateway_run
from gateway import run_adapters
from gateway import run_adapters_recovery
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from tests.gateway.test_platform_reconnect import StubAdapter, _make_runner

_SUSPEND = 13.1 * 3600


class _Clock:
    """A monotonic clock asyncio.sleep advances; ``suspend_at`` adds a host suspend inside that sleep."""

    def __init__(self, suspend_at: int):
        self.t, self.sleeps, self.suspend_at = 1000.0, 0, suspend_at

    def __call__(self) -> float:
        return self.t

    def sleep(self, n):
        self.sleeps += 1
        self.t += n
        if self.sleeps == self.suspend_at:
            self.t += _SUSPEND


def _set_threshold(monkeypatch, seconds: float) -> None:
    monkeypatch.setattr(gateway_run, "_reconnect_attention_after_secs", lambda: seconds)


def _install_clock(monkeypatch, clock: "_Clock") -> None:
    """The reconnect lanes read this clock; the event loop keeps the real one."""
    for module in (run_adapters, run_adapters_recovery):
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=clock))


@pytest.mark.parametrize("threshold, expect_flag", [(7200, False), (60, True)])
@pytest.mark.asyncio
async def test_primary_watcher_retries_before_escalating_across_a_suspend(monkeypatch, threshold, expect_flag):
    """The first pass after a 13 h suspend retries before any escalation; time the host was
    awake still escalates (control: a 60 s threshold flags after the first failed retry)."""
    _set_threshold(monkeypatch, threshold)
    runner = _make_runner()
    writes, events = [], []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda p, **kw: writes.append(kw))
    clock = _Clock(suspend_at=12)  # inside the watcher's first 10 s idle
    _install_clock(monkeypatch, clock)
    failing = StubAdapter(fatal_error="network down")
    real_connect = failing.connect

    async def connect(*, is_reconnect=False):
        events.append(("attempt", any(w.get("needs_attention") for w in writes)))
        return await real_connect(is_reconnect=is_reconnect)
    failing.connect = connect
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: failing)
    runner._failed_platforms[Platform.TELEGRAM] = runner._reconnect_queue_entry(
        Platform.TELEGRAM, StubAdapter(), PlatformConfig(enabled=True, token="t"), attempts=0, delay=30)

    real_sleep = asyncio.sleep

    async def fake_sleep(n):
        clock.sleep(n)
        if len(events) >= 2 or clock.sleeps > 400:
            runner._running = False
        await real_sleep(0)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        await asyncio.wait_for(runner._platform_reconnect_watcher(), timeout=10)

    assert events[0] == ("attempt", False), events
    flagged = [w for w in writes if w.get("needs_attention")]
    assert bool(flagged) is expect_flag, (events, flagged)
    info = runner._failed_platforms[Platform.TELEGRAM]
    assert clock() - info["queued_at"] < _SUSPEND, "the suspend must not count as queued time"


@pytest.mark.asyncio
async def test_secondary_loop_does_not_count_a_suspend_in_its_backoff(monkeypatch):
    """A secondary's reconnect loop measures the same awake-only time across its backoff sleep."""
    _set_threshold(monkeypatch, 7200)
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda key, **kw: writes.append((key, kw)))
    clock = _Clock(suspend_at=1)  # the laptop sleeps during the first backoff
    _install_clock(monkeypatch, clock)
    attempts = []

    class _RetryableAdapter:
        has_fatal_error = True
        fatal_error_retryable = True

    async def failing_attempt(profile_name, platform, predecessor=None):
        attempts.append(clock())
        return _RetryableAdapter(), False

    async def noop(*_a, **_k):
        return None

    monkeypatch.setattr(runner, "_secondary_reconnect_attempt", failing_attempt)
    monkeypatch.setattr(runner, "_safe_adapter_disconnect", noop)
    monkeypatch.setattr(runner, "_routed_profile_home", lambda name: None)
    monkeypatch.setattr(gateway_run, "_reconnect_backoff", lambda attempts: 30)
    real_sleep = asyncio.sleep

    async def fake_sleep(n):
        clock.sleep(n)
        if len(attempts) >= 3:
            runner._running = False
        await real_sleep(0)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        await asyncio.wait_for(runner._run_secondary_profile_reconnect("sec", Platform.DISCORD), timeout=10)

    assert len(attempts) >= 3
    assert not [kw for _k, kw in writes if kw.get("needs_attention")], writes


def test_no_escalation_before_the_first_failed_attempt(monkeypatch):
    """However long an entry has been queued, NEEDS_ATTENTION waits for one failed retry."""
    _set_threshold(monkeypatch, 10)
    runner = _make_runner()
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda p, **kw: writes.append(kw))
    info = {"queued_at": 0.0, "attempts": 0}
    assert runner._flag_reconnect_needs_attention(Platform.TELEGRAM, info, 99999.0) is False
    info["attempts"] = 1
    assert runner._flag_reconnect_needs_attention(Platform.TELEGRAM, info, 99999.0) is True
    assert [w for w in writes if w.get("needs_attention")]
