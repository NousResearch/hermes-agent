"""Gateway session binding/delivery for external background-task completions.

An external task registered from a gateway-bound parent session must reach the
ORIGINAL platform / chat / thread session exactly once through the existing
``_async_delegation_watcher`` delivery rail, be durably claimed/acknowledged,
and never be injected twice.
"""

import asyncio
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.background_tasks import _create_external_background_tasks_service
from agent.host_context import bind_host_parent
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session_context import clear_session_vars, set_session_vars
from tools.async_delegation import get_durable_delegation
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_queue():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _drain_one():
    import time

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.01)
    return None


def _runner(adapter, *, live_parent):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={},
    )
    runner._session_sources = None
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048

    class _SessionDB:
        def __init__(self, live):
            self.live = live

        async def get_session(self, session_id):
            if session_id == self.live:
                return {"id": session_id, "ended_at": None}
            return None

        async def get_compression_tip(self, session_id):
            return None

    runner._session_db = _SessionDB(live_parent)
    return runner


def _stop_after_sleeps(monkeypatch, runner, count):
    sleep_calls = 0

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def _register_and_complete(*, plugin_id, parent_session_id, label, summary):
    parent = SimpleNamespace(session_id=parent_session_id)
    svc = _create_external_background_tasks_service(
        plugin_id=plugin_id,
        parent_agent_resolver=lambda: parent,
    )
    tokens = set_session_vars(
        platform="telegram",
        source="telegram",
        session_key="agent:main:telegram:dm:12345:678",
    )
    try:
        with bind_host_parent(parent):
            handle = svc.register_external(external_id="run-1", label=label)
        result = svc.complete(handle, event_id="e1", summary=summary)
    finally:
        clear_session_vars(tokens)
    assert result.accepted is True
    evt = _drain_one()
    assert evt is not None
    process_registry.completion_queue.put(evt)  # restore for the watcher under test
    return evt


def test_gateway_delivery_reaches_original_platform_chat_thread_once(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    evt = _register_and_complete(
        plugin_id="gw-plugin",
        parent_session_id="parent-telegram",
        label="gateway run",
        summary="gateway summary",
    )

    runner = _runner(adapter, live_parent="parent-telegram")
    _stop_after_sleeps(monkeypatch, runner, count=2)
    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    injected = adapter.handle_message.await_args.args[0]
    assert injected.source.platform is Platform.TELEGRAM
    assert injected.source.chat_id == "12345"
    assert injected.source.thread_id == "678"
    assert "gateway summary" in injected.text
    assert "BACKGROUND TASK COMPLETE" in injected.text

    info = get_durable_delegation(evt["delegation_id"])
    assert info is not None
    assert info["delivery_state"] == "delivered"


def test_gateway_delivery_is_delivered_once_for_duplicate_queue_replay(monkeypatch):
    """Byte-identical queue replays of one external completion inject once."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    evt = _register_and_complete(
        plugin_id="gw-plugin",
        parent_session_id="parent-telegram",
        label="gateway run",
        summary="gateway summary",
    )
    # Replay the same event (as restore_undelivered_completions would re-enqueue
    # the exact persisted payload after a restart).
    isolated = queue.Queue()
    isolated.put(dict(evt))
    isolated.put(dict(evt))
    import tools.process_registry as pr_module

    monkeypatch.setattr(
        pr_module, "process_registry", type("R", (), {"completion_queue": isolated})()
    )

    runner = _runner(adapter, live_parent="parent-telegram")
    _stop_after_sleeps(monkeypatch, runner, count=2)
    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()


def test_gateway_drop_when_parent_session_gone(monkeypatch):
    """A completion pinned to a permanently-gone session is terminally dropped."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    evt = _register_and_complete(
        plugin_id="gw-plugin",
        parent_session_id="parent-gone",
        label="gateway run",
        summary="gateway summary",
    )

    runner = _runner(adapter, live_parent="some-other-live")
    _stop_after_sleeps(monkeypatch, runner, count=2)
    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_not_awaited()
    info = get_durable_delegation(evt["delegation_id"])
    assert info is not None
    assert info["delivery_state"] == "dropped"
