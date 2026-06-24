"""Tests for the notifier_in_gateway gate on _kanban_notifier_watcher.

AP-4028 decoupled the notifier from the dispatcher's singleton lock. The
gate now reads ``kanban.notifier_in_gateway`` (config) and
``HERMES_KANBAN_NOTIFIER_IN_GATEWAY`` (env), defaulting to enabled —
i.e. the notifier runs on every gateway that hasn't been explicitly
opted out, regardless of which host holds the dispatcher lock.

Coverage:
- Non-notifier gateways (notifier_in_gateway=false) exit before opening
  any DB.
- HERMES_KANBAN_NOTIFIER_IN_GATEWAY env var disables without loading
  config.
- Notifier-eligible gateways (notifier_in_gateway=true, default) proceed
  past the gate.
- The legacy dispatcher-gate keys (``dispatch_in_gateway``,
  ``HERMES_KANBAN_DISPATCH_IN_GATEWAY``) NO LONGER control the notifier —
  only the dispatcher.
"""

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


def _fake_config(notifier_in_gateway, dispatch_in_gateway=True):
    return {
        "kanban": {
            "notifier_in_gateway": notifier_in_gateway,
            # Legacy dispatcher-gate key MUST still be present so the test
            # proves the notifier no longer reads it.
            "dispatch_in_gateway": dispatch_in_gateway,
        }
    }


def test_notifier_watcher_skips_when_notifier_disabled():
    """notifier_in_gateway=false returns before opening any board DB."""
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_fake_config(False)):
        with patch("hermes_cli.kanban_db.connect") as mock_connect:
            asyncio.run(runner._kanban_notifier_watcher())
    mock_connect.assert_not_called()


def test_notifier_watcher_env_override_disables(monkeypatch):
    """HERMES_KANBAN_NOTIFIER_IN_GATEWAY=false skips config load entirely."""
    runner = _make_runner()
    monkeypatch.setenv("HERMES_KANBAN_NOTIFIER_IN_GATEWAY", "false")
    with patch("hermes_cli.config.load_config") as mock_load_config:
        with patch("hermes_cli.kanban_db.connect") as mock_connect:
            asyncio.run(runner._kanban_notifier_watcher())
    mock_load_config.assert_not_called()
    mock_connect.assert_not_called()


def test_notifier_watcher_runs_when_notifier_enabled():
    """notifier_in_gateway=true (default) proceeds past the gate."""
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

    assert past_gate, "list_boards should be called when notifier_in_gateway=true"


def test_notifier_watcher_runs_even_when_dispatcher_disabled():
    """AP-4028 invariant: the notifier-gate must NOT read dispatch_in_gateway.

    With dispatch_in_gateway=false (external ``hermes kanban daemon``
    holds the lock) the dispatcher watcher will refuse to start — but
    the notifier MUST still run, because that's the whole point of the
    decoupling. This test pins that contract: setting dispatch_in_gateway
    to false while leaving notifier_in_gateway at its default (True)
    must NOT prevent the notifier from reaching the board fan-out.
    """
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

    cfg = {
        "kanban": {
            "dispatch_in_gateway": False,  # legacy gate — must be ignored
            "notifier_in_gateway": True,   # new gate — must allow entry
        }
    }

    with patch("hermes_cli.config.load_config", return_value=cfg):
        with patch.object(
            _kb, "list_boards",
            side_effect=lambda *a, **kw: past_gate.append(True) or [],
        ):
            with patch("asyncio.sleep", side_effect=fake_sleep):
                with patch("asyncio.to_thread", side_effect=fake_to_thread):
                    asyncio.run(runner._kanban_notifier_watcher())

    assert past_gate, (
        "AP-4028 regression: notifier watcher bailed out when "
        "dispatch_in_gateway=false — the two gates must be independent."
    )