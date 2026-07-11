"""Notifier subscriptions are independent from dispatcher ownership."""

import asyncio
from unittest.mock import MagicMock, patch

from gateway.config import Platform
from gateway.run import GatewayRunner


def _make_runner(with_adapter=False):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: MagicMock()} if with_adapter else {}
    runner._kanban_sub_fail_counts = {}
    return runner


def _fake_config(dispatch_in_gateway):
    return {"kanban": {"dispatch_in_gateway": dispatch_in_gateway}}


def test_notifier_watcher_runs_when_dispatch_disabled():
    """A profile-only gateway must still deliver its own subscriptions."""
    runner = _make_runner(with_adapter=True)
    past_gate = []
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        if len(sleep_calls) >= 2:
            runner._running = False

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import hermes_cli.kanban_db as _kb

    with patch("hermes_cli.config.load_config", return_value=_fake_config(False)):
        with patch.object(
            _kb, "list_boards",
            side_effect=lambda *a, **kw: past_gate.append(True) or [],
        ):
            with patch("asyncio.sleep", side_effect=fake_sleep):
                with patch("asyncio.to_thread", side_effect=fake_to_thread):
                    asyncio.run(runner._kanban_notifier_watcher())

    assert past_gate


def test_notifier_watcher_runs_when_dispatch_enabled():
    """dispatch_in_gateway=true proceeds past the gate to the board fan-out."""
    runner = _make_runner(with_adapter=True)
    past_gate = []
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        # Stop after the initial delay + first per-interval sleep so the loop
        # body runs exactly once.
        if len(sleep_calls) >= 2:
            runner._running = False

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import hermes_cli.kanban_db as _kb

    with patch("hermes_cli.config.load_config", return_value=_fake_config(True)):
        with patch.object(
            _kb, "list_boards",
            side_effect=lambda *a, **kw: past_gate.append(True) or [],
        ):
            with patch("asyncio.sleep", side_effect=fake_sleep):
                with patch("asyncio.to_thread", side_effect=fake_to_thread):
                    asyncio.run(runner._kanban_notifier_watcher())

    assert past_gate, "list_boards should be called when dispatch_in_gateway=true"
