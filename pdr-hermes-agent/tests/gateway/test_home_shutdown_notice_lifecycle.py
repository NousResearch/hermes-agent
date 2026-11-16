"""The home-channel shutdown broadcast stays truthful across restart loops, takeovers and supervisor restarts.

* A respawn loop posts at most ``restart_loop_guard.max_restarts`` notices, then goes quiet (#60438).
* A planned ``--replace`` takeover posts none: the replacing gateway serves the home on (#32008).
* A delivered goodbye owes the next boot a "Gateway online", whatever stopped the gateway (#121937).
"""

import asyncio
import signal
from unittest.mock import MagicMock, patch

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


async def _stop_cycle(runner, adapter, *, adapters=None) -> None:
    """One process lifetime ending in the real ``stop()`` (a fresh runner state per respawn)."""
    runner._home_shutdown_notice = None
    runner._running = True
    runner._stop_task = None
    runner._shutdown_event = asyncio.Event()
    runner.adapters = {Platform.TELEGRAM: adapter} if adapters is None else adapters
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

    # A task running IN the home chat gets the goodbye from the per-session pass instead: still owed.
    planned_restart_marker_path().unlink()
    agent = MagicMock()
    agent.interrupt.side_effect = lambda *a, **k: runner._running_agents.clear()
    runner._running_agents = {f"agent:main:telegram:dm:{HOME_CHAT}": agent}
    await _stop_cycle(runner, adapter)
    assert _home_sends(adapter) == 2
    assert planned_restart_marker_path().exists()

    # Control: a stop that told the home chat nothing owes it nothing.
    planned_restart_marker_path().unlink()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    await _stop_cycle(runner, adapter)
    assert _home_sends(adapter) == 2
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

@pytest.mark.asyncio
async def test_respawn_loop_caps_home_notices_with_running_home_task(runner):
    """An active task in the home chat must not bypass the three-notice loop cap."""
    runner, adapter = runner
    for _ in range(10):
        agent = MagicMock()
        agent.interrupt.side_effect = lambda *a, **k: runner._running_agents.clear()
        runner._running_agents = {f"agent:main:telegram:dm:{HOME_CHAT}": agent}
        await _stop_cycle(runner, adapter)

    assert _home_sends(adapter) == 3  # restart_loop_guard.DEFAULT_MAX_RESTARTS
    assert not planned_restart_marker_path().exists()



@pytest.mark.asyncio
async def test_loop_cap_counts_delivered_notices_not_attempts(runner):
    """A respawn that dies before Telegram connects tells nobody, so it must not spend a slot."""
    runner, adapter = runner
    for _ in range(10):
        await _stop_cycle(runner, adapter, adapters={})  # no adapter connected: nothing delivered
        await _stop_cycle(runner, adapter)

    assert _home_sends(adapter) == 3  # restart_loop_guard.DEFAULT_MAX_RESTARTS delivered goodbyes


@pytest.mark.asyncio
async def test_a_loop_longer_than_the_stored_history_stays_quiet(runner):
    """The notice log keeps only the newest stops. A continuous loop that outruns it must still
    remember the goodbyes it already delivered, or it posts three more every ~50 cycles."""
    from gateway import restart_loop_guard
    runner, adapter = runner
    for _ in range(restart_loop_guard._MAX_STORED_BOOTS * 2 + 10):
        await _stop_cycle(runner, adapter)

    assert _home_sends(adapter) == 3  # restart_loop_guard.DEFAULT_MAX_RESTARTS



class _KilledDuringDrain(Exception):
    """Stands in for the supervisor's SIGKILL at its stop deadline, while the drain is running."""


@pytest.mark.asyncio
async def test_a_process_killed_during_drain_still_owes_hello_and_spends_its_slot(runner, monkeypatch):
    """The goodbye is out before the drain; a lifetime that ends there must already have recorded it."""
    from gateway.run import GatewayRunner
    runner, adapter = runner

    async def _killed(*_a, **_k):
        raise _KilledDuringDrain
    monkeypatch.setattr(GatewayRunner, "_stop_drain_active_work", _killed)
    for respawn in range(8):
        with pytest.raises(_KilledDuringDrain):
            await _stop_cycle(runner, adapter)
        if respawn == 0:
            assert planned_restart_marker_path().exists()  # the delivered goodbye is owed a hello

    assert _home_sends(adapter) == 3  # restart_loop_guard.DEFAULT_MAX_RESTARTS
