"""Gateway /agents surfaces background delegations with live activity (#51690).

Drives the REAL GatewayRunner._handle_agents_command against a REAL
async-delegation registry dispatch (no mocked list function), so the test
covers the whole projection: registry record → list_async_delegations()
live sampling → /agents rendering.
"""

import threading
import time

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_key_for_source = lambda source: "agent:main:test:dm:1"
    return runner


class _Event:
    source = None


@pytest.mark.asyncio
async def test_agents_command_marks_stalling_delegation(monkeypatch):
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.1)
    # Long grace so the record stays in 'stalling' while we render.
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 30.0)
    gate = threading.Event()

    res = ad.dispatch_async_delegation(
        goal="wedged child", context=None, toolsets=None, role="leaf",
        model="m", session_key="agent:main:test:dm:1", max_async_children=1,
        runner=lambda: {} if gate.wait(timeout=10) else {},
        progress_fn=lambda: ((0, None), False),
    )
    assert res["status"] == "dispatched"

    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            items = ad.list_async_delegations()
            if any(
                d["delegation_id"] == res["delegation_id"]
                and d.get("status") == "stalling"
                for d in items
            ):
                break
            time.sleep(0.02)
        else:
            pytest.fail("delegation never reached stalling state")

        runner = _make_runner()
        out = await runner._handle_agents_command(_Event())
    finally:
        gate.set()

    assert res["delegation_id"] in out
    assert "stalling" in out
    assert "no progress" in out


@pytest.mark.asyncio
async def test_agents_command_lists_one_graph_handle_for_independent_clusters():
    gate = threading.Event()
    try:
        result = ad.dispatch_async_delegation_batches(
            graph_id="deleg_visible_graph", max_async_children=1,
            batches=[
                {
                    "goals": [f"task {i}"],
                    "runner": lambda: {} if gate.wait(10) else {},
                    "progress_fn": lambda: (((1, "terminal", time.time()),), True),
                }
                for i in range(3)
            ],
        )
        assert result["status"] == "dispatched"
        output = await _make_runner()._handle_agents_command(_Event())
        assert output.count("`deleg_visible_graph`") == 1
        for component in result["delegations"]:
            assert component["delegation_id"] not in output
        assert "child 3" in output
    finally:
        gate.set()

