"""Phases of the systemd event-loop watchdog, observed through the datagrams it sends."""

from __future__ import annotations

import pytest

import gateway.systemd_notify as notify_mod

pytestmark = pytest.mark.platforms("linux")


@pytest.fixture
def sent(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    # A long interval: the feeding task stays asleep, so only record_tick() feeds.
    monkeypatch.setenv("WATCHDOG_USEC", "100000000")
    monkeypatch.setattr(notify_mod, "notify", lambda message: calls.append(message) or True)
    return calls


def test_a_late_tick_stops_feeding_for_good(sent):
    watchdog = notify_mod.SystemdWatchdog(lag_tolerance_seconds=1.0)

    assert watchdog.record_tick(scheduled_at=0.0, now=0.5) is True
    assert watchdog.record_tick(scheduled_at=0.0, now=5.0) is False
    assert watchdog.unhealthy is True
    assert watchdog.record_tick(scheduled_at=10.0, now=10.1) is False

    assert sent.count("WATCHDOG=1") == 1
    assert sent[-1].startswith("STATUS=watchdog unhealthy")


@pytest.mark.asyncio
async def test_stop_sends_stopping_once_and_ends_feeding(sent):
    watchdog = notify_mod.SystemdWatchdog(lag_tolerance_seconds=1.0)
    assert watchdog.start() is True

    await watchdog.stop()
    await watchdog.stop()

    assert sent.count("STOPPING=1") == 1
    assert watchdog.record_tick(scheduled_at=0.0, now=0.1) is False
    assert "WATCHDOG=1" not in sent


@pytest.mark.asyncio
async def test_an_unhealthy_watchdog_still_announces_its_stop(sent):
    watchdog = notify_mod.SystemdWatchdog(lag_tolerance_seconds=1.0)
    assert watchdog.start() is True
    assert watchdog.record_tick(scheduled_at=0.0, now=5.0) is False

    await watchdog.stop()

    assert sent[-1] == "STOPPING=1"


@pytest.mark.asyncio
async def test_a_restart_feeds_again_and_announces_the_next_stop(sent):
    watchdog = notify_mod.SystemdWatchdog(lag_tolerance_seconds=1.0)
    assert watchdog.start() is True
    assert watchdog.record_tick(scheduled_at=0.0, now=5.0) is False
    await watchdog.stop()

    assert watchdog.start() is True
    assert watchdog.unhealthy is False
    assert watchdog.record_tick(scheduled_at=0.0, now=0.1) is True
    await watchdog.stop()

    assert sent.count("STOPPING=1") == 2
    assert sent.count("WATCHDOG=1") == 1


def test_a_disabled_watchdog_sends_nothing(sent, monkeypatch):
    monkeypatch.delenv("WATCHDOG_USEC")
    watchdog = notify_mod.SystemdWatchdog()

    assert watchdog.start() is False
    assert watchdog.record_tick(scheduled_at=0.0, now=0.1) is False
    assert sent == []
