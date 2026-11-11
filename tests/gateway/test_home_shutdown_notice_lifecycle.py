"""The home-channel shutdown broadcast stays truthful across restart loops, takeovers and supervisor restarts.

* A respawn loop posts at most ``restart_loop_guard.max_restarts`` notices, then goes quiet (#60438).
* A planned ``--replace`` takeover posts none: the replacing gateway serves the home on (#32008).
* A delivered goodbye owes the next boot a "Gateway online", whatever stopped the gateway (#121937).
"""

import asyncio
import signal
from unittest.mock import patch

import pytest

from gateway.config import HomeChannel, Platform
from gateway.restart_notice import planned_restart_marker_path
from tests.gateway.restart_test_helpers import make_restart_runner

HOME_CHAT = "home-42"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.drain_control.drain_notification_suppressed", lambda: False)
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id=HOME_CHAT, name="Family")
    runner._home_shutdown_notice = None
    runner._planned_takeover_shutdown = False
    return runner, adapter


def _home_sends(adapter) -> int:
    return sum(1 for chat_id, _text, _meta in adapter.sent_calls if chat_id == HOME_CHAT)


async def _stop_cycle(runner, adapter) -> None:
    """One process lifetime ending in the real ``stop()`` (a fresh runner state per respawn)."""
    runner._home_shutdown_notice = None
    runner._running = True
    runner._stop_task = None
    runner._shutdown_event = asyncio.Event()
    runner.adapters = {Platform.TELEGRAM: adapter}
    with patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
        await runner.stop()


@pytest.mark.asyncio
async def test_respawn_loop_stops_posting_and_owes_no_online_notice(runner):
    runner, adapter = runner
    for _ in range(10):
        await _stop_cycle(runner, adapter)

    assert _home_sends(adapter) == 3  # restart_loop_guard.DEFAULT_MAX_RESTARTS
    # The last (muted) cycle leaves no "back online" debt behind, so the loop is quiet on both ends.
    assert not planned_restart_marker_path().exists()


@pytest.mark.asyncio
async def test_supervisor_restart_that_said_goodbye_owes_the_next_boot_a_hello(runner):
    runner, adapter = runner
    await _stop_cycle(runner, adapter)

    assert _home_sends(adapter) == 1
    assert planned_restart_marker_path().exists()

    # Control: a stop that told the home chat nothing owes it nothing.
    planned_restart_marker_path().unlink()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    await _stop_cycle(runner, adapter)
    assert _home_sends(adapter) == 1
    assert not planned_restart_marker_path().exists()


@pytest.mark.asyncio
async def test_planned_replace_takeover_sends_no_home_notice(runner, monkeypatch):
    """The real SIGTERM handler, with the takeover marker naming this PID, then the real ``stop()``."""
    from gateway import run as gateway_run

    runner, adapter = runner
    monkeypatch.setattr("gateway.status.consume_takeover_marker_for_self", lambda: True)
    with patch("gateway.status.remove_pid_file"), patch("gateway.status.publish_runtime_status"):
        before = set(asyncio.all_tasks())
        gateway_run._start_gateway_make_shutdown_signal_handler(runner, [False])(signal.SIGTERM)
        stop_tasks = set(asyncio.all_tasks()) - before
        await asyncio.wait_for(asyncio.gather(*stop_tasks), timeout=30)

    assert runner._shutdown_event.is_set()
    assert _home_sends(adapter) == 0
    assert not planned_restart_marker_path().exists()
