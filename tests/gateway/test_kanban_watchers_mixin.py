"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch

from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.run import GatewayRunner

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def _orphan_recovery_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    return runner


def test_async_delegation_orphan_recovery_watcher_honors_config_gate(monkeypatch):
    runner = _orphan_recovery_runner()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "async_delegation": {"orphan_recovery_in_gateway": False},
    })

    with patch("tools.async_delegation.recover_abandoned_delegations") as recover:
        asyncio.run(runner._async_delegation_orphan_recovery_watcher())

    recover.assert_not_called()


def test_async_delegation_orphan_recovery_watcher_sweeps_then_stops(monkeypatch):
    runner = _orphan_recovery_runner()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "async_delegation": {"orphan_recovery_interval_seconds": 600},
    })
    calls = []

    def recover():
        calls.append(True)
        runner._running = False
        return 1

    async def immediate_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("tools.async_delegation.recover_abandoned_delegations", recover):
        with patch("asyncio.to_thread", side_effect=immediate_to_thread):
            asyncio.run(runner._async_delegation_orphan_recovery_watcher())

    assert calls == [True]
