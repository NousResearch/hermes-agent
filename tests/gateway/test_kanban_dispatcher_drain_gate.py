import asyncio
from unittest.mock import MagicMock, patch

from gateway.run import GatewayRunner


def _make_runner(*, external_drain=False, shutdown_drain=False):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._draining = shutdown_drain
    runner._external_drain_active = external_drain
    runner._kanban_dispatcher_lock_handle = None
    return runner


def _dispatch_config():
    return {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": True,
        }
    }


def _run_one_dispatcher_tick(runner):
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        # First sleep is the startup delay. On the next sleep the loop has
        # completed one body iteration, so stop the watcher.
        if len(sleep_calls) >= 2:
            runner._running = False

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("hermes_cli.config.load_config", return_value=_dispatch_config()):
        with patch("gateway.kanban_watchers._acquire_singleton_lock", return_value=(MagicMock(), "held")):
            with patch("gateway.kanban_watchers._release_singleton_lock"):
                with patch("asyncio.sleep", side_effect=fake_sleep):
                    with patch("asyncio.to_thread", side_effect=fake_to_thread):
                        asyncio.run(runner._kanban_dispatcher_watcher())


def test_kanban_dispatcher_spawns_ready_tasks_when_not_draining():
    runner = _make_runner()
    dispatch_once = MagicMock()

    with patch("hermes_cli.kanban_db.dispatch_once", dispatch_once):
        with patch("hermes_cli.kanban_db.list_boards", return_value=[{"slug": "default"}]):
            with patch("hermes_cli.kanban_db.connect"):
                _run_one_dispatcher_tick(runner)

    dispatch_once.assert_called_once()


def test_kanban_dispatcher_does_not_spawn_ready_tasks_during_external_drain():
    runner = _make_runner(external_drain=True)
    dispatch_once = MagicMock()

    with patch("hermes_cli.kanban_db.dispatch_once", dispatch_once):
        with patch("hermes_cli.kanban_db.list_boards", return_value=[{"slug": "default"}]):
            with patch("hermes_cli.kanban_db.connect"):
                _run_one_dispatcher_tick(runner)

    dispatch_once.assert_not_called()


def test_kanban_dispatcher_does_not_spawn_ready_tasks_during_shutdown_drain():
    runner = _make_runner(shutdown_drain=True)
    dispatch_once = MagicMock()

    with patch("hermes_cli.kanban_db.dispatch_once", dispatch_once):
        with patch("hermes_cli.kanban_db.list_boards", return_value=[{"slug": "default"}]):
            with patch("hermes_cli.kanban_db.connect"):
                _run_one_dispatcher_tick(runner)

    dispatch_once.assert_not_called()
