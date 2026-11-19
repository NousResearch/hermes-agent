"""The reporter inside a gateway: when the watcher starts, the goodbye before a self-suspend, and
that reporting is not activity."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import types
import urllib.request

import pytest

from gateway import status as gateway_status
from gateway.run import GatewayRunner
from gateway.status_report.reporter import StatusReporter
from gateway.status_report.target import STATUS_REPORT_URL_ENV
from tests.gateway.status_report.support import (
    AFTER_INTERVAL_S,
    REPORT_URL,
    Clock,
    Receiver,
    Reply,
    assert_matches_contract,
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


def _start_the_status_watcher(monkeypatch, runner: GatewayRunner, receiver: Receiver) -> None:
    """A gateway start's watcher spawn, for real: the reporter is built from the launch environment
    and handed to the real supervisor. Only the other watchers are left out (they need a booted
    gateway), and *receiver* stands where urllib would connect."""
    supervise = runner._spawn_supervised
    monkeypatch.setattr(
        runner, "_spawn_supervised",
        lambda factory, name, **kwargs: supervise(factory, name, **kwargs) if name == WATCHER else None)
    monkeypatch.setattr(runner, "_spawn_reconnect_watcher", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(runner, "_scale_to_zero_should_arm", lambda: False, raising=False)
    monkeypatch.setattr(runner, "_supervised_backoff", lambda attempt: 0)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: types.SimpleNamespace(open=receiver))
    runner._start_spawn_background_watchers()


def _sender_locks() -> list:
    return list(gateway_status._get_lock_dir().glob("*.lock"))


def _reporter_threads() -> list[str]:
    return [thread.name for thread in threading.enumerate() if thread.name.startswith("status-report")]


@pytest.mark.asyncio
async def test_a_gateway_start_runs_the_watcher_to_its_first_beat_and_a_stop_ends_it(monkeypatch):
    credential = new_credential()
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(credential))
    runner, receiver = _bare_runner(monkeypatch), Receiver()

    _start_the_status_watcher(monkeypatch, runner, receiver)
    (watcher,) = runner._background_tasks
    try:
        await async_wait_until(lambda: receiver.kinds == ["beat"], timeout=5, what="the first beat")
        (beat,) = receiver.requests
        assert beat.url == REPORT_URL and beat.headers["Authorization"] == f"Bearer {credential}"
        assert_matches_contract(beat.body, "beat")
        assert beat.body["boot_id"] == runner._status_reporter.boot_id
        assert _sender_locks(), "reporting without the sender lock"

        runner._running = False  # the gateway stops: the watcher sees it at its next poll
        await asyncio.wait({watcher}, timeout=4)
        assert watcher.done() and not watcher.cancelled(), "the watcher outlived the gateway"
    finally:
        watcher.cancel()
        await settle()

    assert _sender_locks() == []
    await async_wait_until(lambda: not _reporter_threads(), timeout=5, what="the reporter's threads to end")


@pytest.mark.asyncio
async def test_a_watcher_that_dies_is_respawned_and_the_same_reporter_takes_the_lock_again(monkeypatch):
    """``run()`` gives the lock up on its way out, so the respawn has to take it again to report."""
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))
    clock = Clock(monkeypatch)
    runner, receiver = _bare_runner(monkeypatch), Receiver()
    run, deaths = StatusReporter.run, [RuntimeError("the watcher died")]

    async def run_that_dies_once(reporter, is_running, **kwargs):
        def still_running() -> bool:
            if receiver.requests and deaths:  # the first beat is out
                raise deaths.pop()
            return is_running()

        await run(reporter, still_running, tick=0.01)

    monkeypatch.setattr(StatusReporter, "run", run_that_dies_once)

    _start_the_status_watcher(monkeypatch, runner, receiver)
    try:
        await async_wait_until(lambda: receiver.kinds == ["beat"] and not deaths, timeout=5, what="a beat, then the death")
        clock.advance(AFTER_INTERVAL_S)
        await async_wait_until(lambda: receiver.kinds == ["beat", "beat"], timeout=5, what="a beat from the respawn")

        assert [seen.body["seq"] for seen in receiver.requests] == [1, 2]
        assert {seen.body["boot_id"] for seen in receiver.requests} == {runner._status_reporter.boot_id}
        assert _sender_locks(), "reporting without the sender lock"
    finally:
        runner._running = False
        await settle()

    assert _sender_locks() == []


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
        assert receiver.answered == 1  # ... while the goodbye was still waiting for its reply
    finally:
        hang.set()
        await settle()


@pytest.mark.asyncio
async def test_a_beat_on_the_wire_neither_delays_the_lever_nor_holds_the_goodbye_back(monkeypatch):
    """The lever runs on the event loop's default threads, and a beat can wait up to its timeout for
    a reply. Neither may queue behind the other: however few default threads there are, the lever
    runs at once, and the goodbye still goes out while the beat waits."""
    asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    clock = Clock(monkeypatch)
    runner, _relay, pulled = _suspendable(monkeypatch, "in-guest")
    receiver, hang = Receiver(), threading.Event()
    reporter = runner._status_reporter = await _sender(receiver)
    receiver.reply_with(Reply(hang=hang))
    _levers(monkeypatch, lambda *args, **kwargs: pulled.append("lever") or True)
    clock.advance(AFTER_INTERVAL_S)
    beat = asyncio.create_task(reporter.poll())
    try:
        await async_wait_until(lambda: receiver.kinds == ["beat", "beat"], timeout=5, what="a beat on the wire")

        await asyncio.wait_for(runner._scale_to_zero_self_suspend(), timeout=5)
        await async_wait_until(lambda: receiver.kinds == ["beat", "beat", "goodbye"], timeout=5, what="the goodbye")

        assert pulled == ["lever"]
        assert not beat.done(), "the beat was answered: this test no longer holds a thread"
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
