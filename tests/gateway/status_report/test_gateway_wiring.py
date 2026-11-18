"""The reporter inside a gateway: when the watcher starts, the goodbye before a self-suspend, and
that reporting is not activity."""

from __future__ import annotations

import asyncio
import threading

import pytest

from gateway.run import GatewayRunner
from gateway.status_report.reporter import StatusReporter
from gateway.status_report.target import STATUS_REPORT_URL_ENV
from tests.gateway.status_report.support import (
    Receiver,
    Reply,
    make_reporter,
    new_credential,
    settle,
    stamped_url,
)
from tests.waiting import async_wait_until

WATCHER = "status_report_watcher"


class _Relay:
    def __init__(self) -> None:
        self.redial: list[str] = []

    def hold_redial(self) -> bool:
        self.redial.append("hold")
        return True

    def release_redial(self) -> bool:
        self.redial.append("release")
        return True


def _bare_runner(monkeypatch) -> GatewayRunner:
    """A GatewayRunner that was never booted: the real methods run, only what they reach is stubbed."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._draining = False
    runner._scale_to_zero_cooldown_until = 0.0
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._failed_platforms = {}
    runner._startup_parked_platforms = False
    runner._last_inbound_at = 0.0  # inbound-quiet for hours
    runner.config = None
    runner.adapters = {}
    runner._profile_adapters = {}
    monkeypatch.setattr(runner, "_update_runtime_status", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(runner, "_scale_to_zero_idle_timeout_seconds", lambda: 300.0, raising=False)
    return runner


def _spawned_watchers(monkeypatch, runner: GatewayRunner) -> list[str]:
    """Names of the watchers a gateway start spawns, without running any of them."""
    names: list[str] = []
    monkeypatch.setattr(runner, "_spawn_supervised", lambda factory, name, **kwargs: names.append(name), raising=False)
    monkeypatch.setattr(runner, "_spawn_reconnect_watcher", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(runner, "_scale_to_zero_should_arm", lambda: False, raising=False)
    runner._start_spawn_background_watchers()
    return names


def test_without_the_url_no_watcher_starts(monkeypatch):
    monkeypatch.delenv(STATUS_REPORT_URL_ENV, raising=False)
    runner = _bare_runner(monkeypatch)

    names = _spawned_watchers(monkeypatch, runner)

    assert WATCHER not in names and "drain_control_watcher" in names
    assert getattr(runner, "_status_reporter", None) is None


def test_the_url_in_the_launch_environment_starts_one_supervised_watcher(monkeypatch):
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))
    runner = _bare_runner(monkeypatch)

    names = _spawned_watchers(monkeypatch, runner)

    assert names.count(WATCHER) == 1
    assert isinstance(runner._status_reporter, StatusReporter)


def test_a_gateway_started_for_a_named_profile_does_not_report(tmp_path, monkeypatch):
    """The image starts each profile's gateway as ``hermes -p <name> gateway run``: same container
    environment, so the same stamp, but its home is ``<root>/profiles/<name>``."""
    profile_home = tmp_path / "hermes_test" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))
    runner = _bare_runner(monkeypatch)

    names = _spawned_watchers(monkeypatch, runner)

    assert WATCHER not in names
    assert getattr(runner, "_status_reporter", None) is None


def _suspendable(monkeypatch, lever: str) -> tuple[GatewayRunner, _Relay, list[str]]:
    """A runner about to suspend itself by *lever* ("in-guest", "brokered" or "none")."""
    runner, relay, pulled = _bare_runner(monkeypatch), _Relay(), []
    monkeypatch.setattr(runner, "_relay_adapter_for_dormancy", lambda: relay, raising=False)
    monkeypatch.setattr("gateway.scale_to_zero.self_suspend_available", lambda *args, **kwargs: lever == "in-guest")
    monkeypatch.setattr("gateway.scale_to_zero.FLY_FREEZE_GRACE_S", 0.0)
    if lever == "brokered":
        monkeypatch.setenv("GATEWAY_RELAY_SLEEP_URL", "https://sleep.example/instance?t=s")
    else:
        monkeypatch.delenv("GATEWAY_RELAY_SLEEP_URL", raising=False)
    return runner, relay, pulled


def _levers(monkeypatch, pull) -> None:
    monkeypatch.setattr("gateway.scale_to_zero.suspend_self", pull)
    monkeypatch.setattr("gateway.scale_to_zero.request_brokered_suspend", pull)


async def _sender(receiver: Receiver) -> StatusReporter:
    reporter = make_reporter(receiver)
    await reporter.poll()  # the first beat makes it the sender
    return reporter


@pytest.mark.asyncio
@pytest.mark.parametrize("lever", ["in-guest", "brokered"])
async def test_a_goodbye_goes_out_before_either_suspend_lever_is_done(monkeypatch, lever):
    runner, _relay, pulled = _suspendable(monkeypatch, lever)
    receiver = Receiver()
    runner._status_reporter = await _sender(receiver)

    def pull(*args, **kwargs):
        # The lever runs in a worker thread; the goodbye must already be on its way without it.
        pulled.append("goodbye first" if receiver.goodbye_arrived.wait(timeout=5) else "no goodbye")
        return True

    _levers(monkeypatch, pull)

    await runner._scale_to_zero_self_suspend()
    await settle()

    assert pulled == ["goodbye first"]
    assert receiver.kinds == ["beat", "goodbye"]
    assert 2.0 <= receiver.requests[1].timeout <= 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize("lever,redial", [("in-guest", ["release"]), ("brokered", [])])
async def test_a_hanging_receiver_neither_delays_nor_stops_the_lever(monkeypatch, lever, redial):
    runner, relay, pulled = _suspendable(monkeypatch, lever)
    receiver, hang = Receiver(), threading.Event()
    runner._status_reporter = await _sender(receiver)
    receiver.reply_with(Reply(hang=hang))
    _levers(monkeypatch, lambda *args, **kwargs: pulled.append("lever") or True)

    try:
        # The receiver never answers; awaiting the goodbye would hang here until the bound below.
        await asyncio.wait_for(runner._scale_to_zero_self_suspend(), timeout=5)
        await async_wait_until(lambda: receiver.kinds == ["beat", "goodbye"], timeout=5, what="the goodbye")

        assert pulled == ["lever"]
        assert relay.redial == redial  # the suspend ran to its normal end
        assert not hang.is_set()  # ... while the receiver was still hanging
    finally:
        hang.set()
        await settle()


@pytest.mark.asyncio
async def test_a_goodbye_that_cannot_start_does_not_fail_the_suspend(monkeypatch):
    runner, relay, pulled = _suspendable(monkeypatch, "in-guest")

    class Broken:
        def goodbye(self) -> None:
            raise RuntimeError("no goodbye today")

    runner._status_reporter = Broken()
    _levers(monkeypatch, lambda *args, **kwargs: pulled.append("lever") or True)

    await runner._scale_to_zero_self_suspend()

    assert pulled == ["lever"]
    assert relay.redial == ["release"]


@pytest.mark.asyncio
async def test_no_goodbye_where_there_is_no_suspend_lever(monkeypatch):
    runner, relay, pulled = _suspendable(monkeypatch, "none")
    receiver = Receiver()
    runner._status_reporter = await _sender(receiver)
    _levers(monkeypatch, lambda *args, **kwargs: pulled.append("lever") or True)

    await runner._scale_to_zero_self_suspend()
    await settle()

    assert pulled == [] and relay.redial == ["release"]
    assert receiver.kinds == ["beat"]


@pytest.mark.asyncio
async def test_reports_do_not_count_as_activity(monkeypatch):
    """Scale-to-zero must still see an idle gateway: a report is outgoing, and neither the watcher
    nor a goodbye in flight is work."""
    monkeypatch.setattr("cron.scheduler.get_running_job_ids", set)
    runner = _bare_runner(monkeypatch)
    assert runner._scale_to_zero_is_idle() is True  # control: idle before any report
    receiver, hang = Receiver(), threading.Event()
    reporter = make_reporter(receiver)
    runner._spawn_supervised(lambda: reporter.run(lambda: runner._running, tick=0.01), WATCHER)
    try:
        await async_wait_until(lambda: receiver.kinds == ["beat"], timeout=5, what="the first report")
        receiver.reply_with(Reply(hang=hang))
        reporter.goodbye()
        await async_wait_until(lambda: receiver.kinds == ["beat", "goodbye"], timeout=5, what="the goodbye")

        assert runner._scale_to_zero_has_live_background_work() is False
        assert runner._scale_to_zero_is_idle() is True
        assert runner._last_inbound_at == 0.0
    finally:
        hang.set()
        runner._running = False
        await settle()
