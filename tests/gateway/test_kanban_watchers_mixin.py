"""Behavior contracts for GatewayRunner's Kanban background watchers."""

from __future__ import annotations

import asyncio


def test_gateway_watchers_honor_the_dispatch_disabled_environment(monkeypatch):
    """Both background loops must exit before opening Kanban state when disabled."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "false")

    asyncio.run(runner._kanban_notifier_watcher())
    asyncio.run(runner._kanban_dispatcher_watcher())


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one dispatcher holder may own the machine-local advisory lock."""
    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    first_handle, first_state = _acquire_singleton_lock(lock)
    assert first_state == "held" and first_handle is not None

    second_handle, second_state = _acquire_singleton_lock(lock)
    assert second_state == "contended" and second_handle is None

    _release_singleton_lock(first_handle)
    third_handle, third_state = _acquire_singleton_lock(lock)
    assert third_state == "held" and third_handle is not None
    _release_singleton_lock(third_handle)
