"""Gateway event-loop freeze backstops for issue #69089."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from gateway.shutdown_watchdog import (
    _arm_loop_floor_timer,
    start_loop_liveness_watchdog,
)


def _immediate_loop() -> MagicMock:
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.call_soon_threadsafe.side_effect = lambda callback: callback()
    return loop


def test_loop_liveness_watchdog_responsive_probe_does_not_fire():
    loop = _immediate_loop()
    exit_codes = []

    with (
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=exit_codes.append),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=2
        )
        assert handle is not None
        deadline = time.monotonic() + 2.0
        while loop.call_soon_threadsafe.call_count < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        handle.stop()
        handle.join(timeout=1.0)

    assert loop.call_soon_threadsafe.call_count >= 3
    assert not handle.is_alive()
    dump.assert_not_called()
    assert exit_codes == []


def test_loop_liveness_watchdog_exits_after_consecutive_misses():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    fired = threading.Event()
    exit_codes = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        fired.set()

    with (
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=2
        )
        assert handle is not None
        assert fired.wait(timeout=2.0), "loop liveness watchdog did not fire"
        handle.join(timeout=1.0)

    assert not handle.is_alive()
    assert loop.call_soon_threadsafe.call_count == 2
    dump.assert_called_once_with(all_threads=True)
    assert exit_codes == [75]


def test_loop_liveness_watchdog_stop_during_final_miss_disarms_hard_exit():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    probe_scheduled = threading.Event()
    release_probe = threading.Event()
    probe_event_ref = {}
    handle_ref = {}
    exit_codes = []

    class FinalStrikeLimit:
        def __gt__(self, _strikes: int) -> bool:
            # If strike evaluation is reached, keep recheck #2 from masking a
            # missing post-probe recheck #1 in this boundary test.
            handle_ref["handle"]._stop_event.clear()
            return False

    def hold_scheduled_probe(callback) -> None:
        probe_event_ref["event"] = callback.__self__
        probe_scheduled.set()
        assert release_probe.wait(timeout=2.0)

    loop.call_soon_threadsafe.side_effect = hold_scheduled_probe
    with (
        patch(
            "gateway.shutdown_watchdog._positive_int_env",
            return_value=FinalStrikeLimit(),
        ),
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=exit_codes.append),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        handle_ref["handle"] = handle
        assert probe_scheduled.wait(timeout=2.0), "watchdog did not schedule a probe"

        def stop_during_miss() -> bool:
            handle.stop()
            return False

        probe_event_ref["event"].is_set = stop_during_miss
        release_probe.set()
        handle.join(timeout=1.0)

    assert not handle.is_alive()
    assert exit_codes == []
    critical.assert_not_called()
    dump.assert_not_called()


def test_loop_liveness_watchdog_stop_after_first_recheck_skips_final_actions():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    probe_scheduled = threading.Event()
    release_probe = threading.Event()

    def hold_scheduled_probe(callback) -> None:
        probe_scheduled.set()
        assert release_probe.wait(timeout=2.0)

    loop.call_soon_threadsafe.side_effect = hold_scheduled_probe
    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit") as hard_exit,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        assert probe_scheduled.wait(timeout=2.0), "watchdog did not schedule a probe"

        original_is_set = handle._stop_event.is_set
        is_set_calls = 0

        def stop_on_final_recheck() -> bool:
            nonlocal is_set_calls
            is_set_calls += 1
            # With the forced immediate timeout: _wait_for_probe is call 1,
            # recheck #1 is call 2, and recheck #2 is call 3.
            if is_set_calls == 3:
                handle.stop()
            return original_is_set()

        handle._stop_event.is_set = stop_on_final_recheck
        with patch(
            "gateway.shutdown_watchdog.time.monotonic", side_effect=[0.0, 1.0]
        ):
            release_probe.set()
            handle.join(timeout=1.0)

    assert is_set_calls == 3
    assert not handle.is_alive()
    critical.assert_not_called()
    dump.assert_not_called()
    hard_exit.assert_not_called()


def test_loop_liveness_watchdog_recovery_resets_strikes():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    four_probes = threading.Event()

    def alternate_response(callback) -> None:
        count = loop.call_soon_threadsafe.call_count
        if count in {2, 4}:
            callback()
        if count >= 4:
            four_probes.set()

    loop.call_soon_threadsafe.side_effect = alternate_response
    with (
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit") as hard_exit,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=2
        )
        assert handle is not None
        assert four_probes.wait(
            timeout=2.0
        ), "watchdog did not complete recovery probes"
        handle.stop()
        handle.join(timeout=1.0)

    assert not handle.is_alive()
    dump.assert_not_called()
    hard_exit.assert_not_called()


def test_loop_liveness_watchdog_stop_exits_thread_and_stops_probes():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    first_probe = threading.Event()
    loop.call_soon_threadsafe.side_effect = lambda callback: first_probe.set()

    handle = start_loop_liveness_watchdog(
        loop, probe_interval=0.01, probe_timeout=0.5, max_strikes=10
    )
    assert handle is not None
    assert first_probe.wait(timeout=2.0)
    handle.stop()
    handle.join(timeout=1.0)
    calls_after_stop = loop.call_soon_threadsafe.call_count
    time.sleep(0.05)

    assert not handle.is_alive()
    assert loop.call_soon_threadsafe.call_count == calls_after_stop


def test_loop_liveness_watchdog_env_can_disable(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOOP_WATCHDOG", "0")
    loop = MagicMock(spec=asyncio.AbstractEventLoop)

    handle = start_loop_liveness_watchdog(
        loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
    )

    assert handle is None
    loop.call_soon_threadsafe.assert_not_called()


def test_loop_liveness_watchdog_env_overrides_probe_settings(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOOP_WATCHDOG_INTERVAL", "0.01")
    monkeypatch.setenv("HERMES_GATEWAY_LOOP_WATCHDOG_TIMEOUT", "0.01")
    monkeypatch.setenv("HERMES_GATEWAY_LOOP_WATCHDOG_STRIKES", "1")
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    fired = threading.Event()

    with (
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback"),
        patch(
            "gateway.shutdown_watchdog.os._exit",
            side_effect=lambda code: fired.set(),
        ),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=10.0, probe_timeout=10.0, max_strikes=10
        )
        assert handle is not None
        assert fired.wait(timeout=2.0), "env overrides were not applied"
        handle.join(timeout=1.0)

    assert not handle.is_alive()
    assert loop.call_soon_threadsafe.call_count == 1


@pytest.mark.asyncio
async def test_loop_liveness_watchdog_detects_real_loop_sync_freeze():
    loop = asyncio.get_running_loop()
    fired = threading.Event()
    exit_codes = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        fired.set()

    with (
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.02, probe_timeout=0.03, max_strikes=2
        )
        assert handle is not None
        await asyncio.sleep(0.03)
        time.sleep(0.25)
        assert fired.wait(timeout=1.0), "watchdog did not detect the frozen real loop"
        handle.stop()
        handle.join(timeout=1.0)

    dump.assert_called_once_with(all_threads=True)
    assert exit_codes == [75]


@pytest.mark.asyncio
async def test_loop_liveness_watchdog_leaves_responsive_real_loop_running():
    loop = asyncio.get_running_loop()
    with (
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit") as hard_exit,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.02, probe_timeout=0.03, max_strikes=2
        )
        assert handle is not None
        await asyncio.sleep(0.25)
        handle.stop()
        handle.join(timeout=1.0)

    assert not handle.is_alive()
    dump.assert_not_called()
    hard_exit.assert_not_called()


def test_loop_floor_timer_reschedules_until_cancelled():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    scheduled = []

    def fake_call_later(delay, callback):
        timer = MagicMock(spec=asyncio.TimerHandle)
        scheduled.append((delay, callback, timer))
        return timer

    loop.call_later.side_effect = fake_call_later
    handle = _arm_loop_floor_timer(loop, interval=5.0)

    assert len(scheduled) == 1
    assert scheduled[0][0] == 5.0
    scheduled[0][1]()
    assert len(scheduled) == 2
    assert scheduled[1][0] == 5.0

    handle.cancel()
    scheduled[1][2].cancel.assert_called_once_with()
    scheduled[1][1]()
    assert len(scheduled) == 2


def test_gateway_runner_liveness_guards_start_and_stop():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._loop_floor_timer_handle = None
    runner._loop_liveness_watchdog = None
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    floor_timer = MagicMock()
    watchdog = MagicMock()
    watchdog.is_alive.return_value = True

    with (
        patch(
            "gateway.run._arm_loop_floor_timer", return_value=floor_timer
        ) as arm_floor,
        patch(
            "gateway.run.start_loop_liveness_watchdog", return_value=watchdog
        ) as start_watchdog,
    ):
        runner._start_loop_liveness_guards(loop)

    arm_floor.assert_called_once_with(loop)
    start_watchdog.assert_called_once_with(loop)
    assert runner._loop_floor_timer_handle is floor_timer
    assert runner._loop_liveness_watchdog is watchdog

    runner._stop_loop_liveness_guards()

    watchdog.stop.assert_called_once_with()
    floor_timer.cancel.assert_called_once_with()
    assert runner._loop_liveness_watchdog is None
    assert runner._loop_floor_timer_handle is None
