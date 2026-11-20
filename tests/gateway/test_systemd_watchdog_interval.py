"""systemd watchdog scoping (WATCHDOG_PID) and heartbeat cadence vs. the advertised deadline."""

from __future__ import annotations

import asyncio
import os
import types

import pytest

import gateway.systemd_notify as notify_mod


@pytest.fixture
def notify_env(monkeypatch):
    """Pretend systemd handed us a notify socket; no real datagrams are sent."""
    monkeypatch.setattr(notify_mod, "_notify_socket", lambda: "/run/hermes-test/notify")
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    monkeypatch.setenv("WATCHDOG_USEC", "30000000")
    return monkeypatch


def test_watchdog_enabled_when_pid_absent(notify_env):
    assert notify_mod.watchdog_interval_seconds() == 30.0


def test_watchdog_enabled_when_pid_matches(notify_env):
    notify_env.setenv("WATCHDOG_PID", str(os.getpid()))
    assert notify_mod.watchdog_interval_seconds() == 30.0
    assert notify_mod.SystemdWatchdog().enabled is True


@pytest.mark.parametrize("pid", [str(os.getpid() + 1), "not-a-pid"])
def test_watchdog_disabled_when_pid_mismatched_or_invalid(notify_env, pid):
    calls: list[str] = []
    notify_env.setattr(notify_mod, "notify", lambda message: calls.append(message) or True)
    notify_env.setenv("WATCHDOG_PID", pid)

    assert notify_mod.watchdog_interval_seconds() is None
    watchdog = notify_mod.SystemdWatchdog()
    assert watchdog.enabled is False
    assert watchdog.record_tick(scheduled_at=0.0, now=0.0) is False
    assert calls == []


@pytest.mark.parametrize("usec", [500_000, 1_500_000])
def test_sub_two_second_watchdog_pings_well_inside_deadline(notify_env, usec):
    """WatchdogSec below 2s must not be stretched by any heartbeat floor: every ping, including the
    first, lands within the advertised timeout even when each wake-up is late by the full lag budget."""
    notify_env.setenv("WATCHDOG_USEC", str(usec))
    deadline = usec / 1e6
    clock = types.SimpleNamespace(now=0.0)
    pings: list[float] = []

    watchdog = notify_mod.SystemdWatchdog()
    lag = watchdog._lag_tolerance() * 0.99

    async def _late_sleep(delay: float) -> None:
        clock.now += delay + lag  # worst case the watchdog still treats as healthy

    def _notify(message: str) -> bool:
        if message == "WATCHDOG=1":
            pings.append(clock.now)
            if len(pings) >= 5:
                watchdog._stopping = True
        return True

    fake_asyncio = types.SimpleNamespace(
        sleep=_late_sleep,
        get_running_loop=lambda: types.SimpleNamespace(time=lambda: clock.now),
        CancelledError=asyncio.CancelledError,
    )
    notify_env.setattr(notify_mod, "asyncio", fake_asyncio)
    notify_env.setattr(notify_mod, "notify", _notify)

    asyncio.run(watchdog._run())

    assert len(pings) == 5
    assert watchdog.unhealthy is False
    gaps = [pings[0]] + [b - a for a, b in zip(pings, pings[1:])]
    assert max(gaps) < deadline, (gaps, deadline)
