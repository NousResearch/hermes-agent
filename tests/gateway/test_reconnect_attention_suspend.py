"""NEEDS_ATTENTION after a suspended host (#126825): Windows' ``time.monotonic()`` keeps counting
through Modern Standby, so a platform queued just before the laptop slept looked hours-failed on
the first watcher pass after resume, before it had retried once. Both reconnect lanes (the launch
profile's watcher and a secondary's loop) are driven for real; only the clock and connect are fake."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_agent.gateway.run as gateway_run
from hermes_agent.gateway import run_adapters
from hermes_agent.gateway import run_adapters_recovery
from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.run import GatewayRunner
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


def _watch(monkeypatch, runner, clock, *, stop_when):
    """Drive the real watcher on ``clock``; every connect fails, ``stop_when()`` ends the run."""
    failing = StubAdapter(fatal_error="network down")
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: failing)
    real_sleep = asyncio.sleep

    async def fake_sleep(n):
        clock.sleep(n)
        if stop_when() or clock.sleeps > 400:
            runner._running = False
        await real_sleep(0)
    return failing, patch("asyncio.sleep", side_effect=fake_sleep)


@pytest.mark.asyncio
async def test_entry_queued_after_resume_keeps_its_own_clock(monkeypatch):
    """A platform that fails after the host wakes was not asleep: the suspend overrun of the idle
    it was queued in must not push its ``queued_at`` into the future (that delays a real outage's
    NEEDS_ATTENTION by the length of the suspend)."""
    _set_threshold(monkeypatch, 7200)
    runner = _make_runner()
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda p, **kw: None)
    clock = _Clock(suspend_at=12)  # inside the watcher's first post-pass idle
    _install_clock(monkeypatch, clock)
    cfg = PlatformConfig(enabled=True, token="t")
    runner._failed_platforms[Platform.TELEGRAM] = runner._reconnect_queue_entry(
        Platform.TELEGRAM, StubAdapter(), cfg, attempts=0, delay=30)
    seen = []
    real_discount = run_adapters.discount_suspended_wait

    def discount(entries, waited_from, planned):
        real_discount(entries, waited_from, planned)
        fresh = runner._failed_platforms.get(Platform.DISCORD)
        if fresh is not None:
            seen.append((fresh["queued_at"], clock()))
    monkeypatch.setattr(run_adapters, "discount_suspended_wait", discount)

    def stop_when():
        if clock.sleeps == clock.suspend_at:  # the host just woke; another platform fails now
            runner._failed_platforms[Platform.DISCORD] = runner._reconnect_queue_entry(
                Platform.DISCORD, StubAdapter(), cfg, attempts=0, delay=30)
        return bool(seen)
    _failing, sleeping = _watch(monkeypatch, runner, clock, stop_when=stop_when)
    with sleeping:
        await asyncio.wait_for(runner._platform_reconnect_watcher(), timeout=10)

    queued_at, now = seen[0]
    assert queued_at <= now, f"fresh entry moved {queued_at - now:.0f}s into the future"


@pytest.mark.asyncio
async def test_suspend_during_the_initial_delay_does_not_escalate_a_startup_entry(monkeypatch):
    """An entry startup queued (already one failed attempt) must not reach NEEDS_ATTENTION on the
    watcher's first pass because the host slept through the 10 s startup delay."""
    _set_threshold(monkeypatch, 7200)
    runner = _make_runner()
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda p, **kw: writes.append(kw))
    clock = _Clock(suspend_at=1)  # the initial asyncio.sleep(10)
    _install_clock(monkeypatch, clock)
    runner._failed_platforms[Platform.TELEGRAM] = runner._reconnect_queue_entry(
        Platform.TELEGRAM, StubAdapter(), PlatformConfig(enabled=True, token="t"), attempts=1, delay=30)
    attempts = []
    failing, sleeping = _watch(monkeypatch, runner, clock, stop_when=lambda: bool(attempts))
    real_connect = failing.connect

    async def connect(*, is_reconnect=False):
        attempts.append(any(w.get("needs_attention") for w in writes))
        return await real_connect(is_reconnect=is_reconnect)
    failing.connect = connect
    with sleeping:
        await asyncio.wait_for(runner._platform_reconnect_watcher(), timeout=10)

    assert attempts and attempts[0] is False, "flagged NEEDS_ATTENTION before the first retry"
    assert not [w for w in writes if w.get("needs_attention")], writes


@pytest.mark.asyncio
async def test_platform_queued_early_in_idle_is_not_aged_by_suspend(monkeypatch):
    """A fatal queued while the watcher is idling, just before host suspend, must not
    inherit that suspend as an outage. A late entry queued after wake stays unshifted."""
    _set_threshold(monkeypatch, 7200)  # the production two-hour default
    runner = _make_runner()
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda key, **kw: writes.append((key, kw)))
    clock = _Clock(suspend_at=-1)
    _install_clock(monkeypatch, clock)
    cfg = PlatformConfig(enabled=True, token="t")
    runner._failed_platforms[Platform.TELEGRAM] = runner._reconnect_queue_entry(
        Platform.TELEGRAM, StubAdapter(), cfg, attempts=0, delay=0)
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config:
                        StubAdapter(platform=platform, fatal_error="network down"))

    real_flag = runner._flag_reconnect_needs_attention
    checked = []

    def flag(platform, info, now, **kwargs):
        flagged = real_flag(platform, info, now, **kwargs)
        if platform == Platform.DISCORD and info["attempts"] >= 1:
            checked.append(flagged)
            runner._running = False
        return flagged

    monkeypatch.setattr(runner, "_flag_reconnect_needs_attention", flag)
    real_sleep = asyncio.sleep
    queued_during_idle = False

    async def fake_sleep(seconds):
        nonlocal queued_during_idle
        if seconds == 1 and not queued_during_idle:
            # The primary watcher entered its 10s idle with only Telegram queued.
            # Another fatal arrives 2s into the idle, before the laptop sleeps.
            clock.t += 2
            runner._failed_platforms[Platform.DISCORD] = runner._reconnect_queue_entry(
                Platform.DISCORD, StubAdapter(platform=Platform.DISCORD), cfg, attempts=0, delay=0)
            queued_during_idle = True
            clock.t += _SUSPEND + seconds
        else:
            clock.sleep(seconds)
        if clock.sleeps > 400:
            runner._running = False
        await real_sleep(0)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        await asyncio.wait_for(runner._platform_reconnect_watcher(), timeout=10)

    assert queued_during_idle, "test never queued a second platform during watcher idle"
    assert checked == [False], "a real two-hour outage must be awake time, not host suspend"
    assert not [(key, kw) for key, kw in writes if key == "discord" and kw.get("needs_attention")]


@pytest.mark.asyncio
async def test_platform_queued_during_the_empty_queue_wait_is_not_aged_by_suspend(monkeypatch):
    """The watcher idles up to 30 s while nothing is queued. A fatal queued in that wait just
    before the host sleeps must not count the suspend either (#153439 review)."""
    _set_threshold(monkeypatch, 7200)
    runner = _make_runner()
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda key, **kw: writes.append((key, kw)))
    clock = _Clock(suspend_at=-1)
    _install_clock(monkeypatch, clock)
    cfg = PlatformConfig(enabled=True, token="t")
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config:
                        StubAdapter(platform=platform, fatal_error="network down"))
    real_flag = runner._flag_reconnect_needs_attention
    checked = []

    def flag(platform, info, now, **kwargs):
        flagged = real_flag(platform, info, now, **kwargs)
        if info["attempts"] >= 1:
            checked.append(flagged)
            runner._running = False
        return flagged

    monkeypatch.setattr(runner, "_flag_reconnect_needs_attention", flag)
    real_sleep = asyncio.sleep
    queued = False

    async def fake_sleep(seconds):
        nonlocal queued
        clock.sleep(seconds)
        if seconds == 1 and clock.sleeps == 4 and not queued:
            # Past the startup delay, the queue is empty: a fatal lands, then the laptop sleeps.
            runner._failed_platforms[Platform.TELEGRAM] = runner._reconnect_queue_entry(
                Platform.TELEGRAM, StubAdapter(), cfg, attempts=0, delay=0)
            queued = True
            clock.t += _SUSPEND
        if clock.sleeps > 400:
            runner._running = False
        await real_sleep(0)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        await asyncio.wait_for(runner._platform_reconnect_watcher(), timeout=10)

    assert queued, "test never queued a platform during the empty-queue wait"
    assert checked == [False], "the suspend in the empty-queue wait counted as outage time"
    assert not [kw for _key, kw in writes if kw.get("needs_attention")]
