"""Reserved executor lane for interactive turns (``gateway.interactive_executor_workers``).

Contract:
  * lane off (default / bare runner without config) -> ``_get_interactive_executor()`` IS the
    shared pool, byte-identical to the pre-lane gateway;
  * lane on -> a separate ``hermes-gw-interactive`` pool; shared pool untouched;
  * routing is an explicit human-chat allowlist — API/webhook/relay/unknown/malformed sources
    fail safe to the shared pool;
  * shutdown closes both pools; the shared pool saturated by batch work cannot delay a lane task;
  * the agent-turn call site routes by the turn's platform.
"""
import asyncio
import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner


def _make_runner(interactive_workers=0):
    runner = object.__new__(GatewayRunner)
    runner._executor = None
    runner._executor_lock = threading.Lock()
    runner._executor_closing = False
    runner.config = GatewayConfig.from_dict({"gateway": {"interactive_executor_workers": interactive_workers}})
    return runner


def _src(platform):
    return SimpleNamespace(platform=SimpleNamespace(value=platform))


def test_lane_off_is_the_shared_pool_including_bare_runner():
    runner = _make_runner(interactive_workers=0)
    bare = object.__new__(GatewayRunner)  # legacy/test runner: no ``config`` at all
    try:
        assert runner._get_interactive_executor() is runner._get_executor()
        assert bare._get_interactive_executor() is bare._get_executor()
    finally:
        runner._shutdown_executor()
        bare._shutdown_executor()


def test_lane_on_is_a_separate_memoized_pool():
    runner = _make_runner(interactive_workers=3)
    try:
        shared, lane = runner._get_executor(), runner._get_interactive_executor()
        assert lane is not shared
        assert lane._max_workers == 3
        assert lane._thread_name_prefix == "hermes-gw-interactive"
        assert runner._get_interactive_executor() is lane
    finally:
        runner._shutdown_executor()


def test_routing_is_a_human_chat_allowlist_that_fails_safe():
    for p in ("telegram", "discord", "slack", "whatsapp", "signal", "matrix", "email", "sms"):
        assert GatewayRunner._is_batch_platform(_src(p)) is False, p
    for p in ("webhook", "api_server", "relay", "msgraph_webhook", "mcp_http", "unknown", "cli"):
        assert GatewayRunner._is_batch_platform(_src(p)) is True, p
    assert GatewayRunner._is_batch_platform(SimpleNamespace(platform=Platform.TELEGRAM)) is False
    assert GatewayRunner._is_batch_platform(SimpleNamespace(platform=Platform.WEBHOOK)) is True
    assert GatewayRunner._is_batch_platform(SimpleNamespace(platform="telegram")) is False
    # malformed sources -> shared pool (pre-lane behavior), never reserved capacity
    for bad in (None, SimpleNamespace(), SimpleNamespace(platform=None), SimpleNamespace(platform=""),
                SimpleNamespace(platform=42), SimpleNamespace(platform=SimpleNamespace(value=None))):
        assert GatewayRunner._is_batch_platform(bad) is True


def test_shutdown_closes_both_pools_and_blocks_recreation():
    runner = _make_runner(interactive_workers=2)
    shared, lane = runner._get_executor(), runner._get_interactive_executor()
    assert runner._shutdown_executor(drain_timeout=2.0) == 0
    assert shared._shutdown and lane._shutdown
    with pytest.raises(RuntimeError):
        runner._get_executor()
    with pytest.raises(RuntimeError):
        runner._get_interactive_executor()


def test_interactive_lane_immune_to_batch_saturation():
    runner = _make_runner(interactive_workers=2)
    runner._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="hermes-gateway")
    try:
        shared, lane = runner._get_executor(), runner._get_interactive_executor()
        release = threading.Event()
        blockers = [shared.submit(release.wait) for _ in range(4)]  # fill every shared slot + queue
        t0 = time.monotonic()
        assert lane.submit(lambda: "fast").result(timeout=5) == "fast"
        assert time.monotonic() - t0 < 2.0
        release.set()
        concurrent.futures.wait(blockers, timeout=5)
    finally:
        runner._shutdown_executor()


@pytest.mark.parametrize("platform,expected_prefix", [
    ("telegram", "hermes-gw-interactive"),
    ("webhook", "hermes-gateway"),
])
def test_start_turn_worker_routes_by_platform(monkeypatch, platform, expected_prefix):
    """The agent-turn submission site (_run_agent_start_turn_worker) picks the pool by platform."""
    from gateway.turn_context import TurnContext

    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")  # no watchdog thread
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT_WARNING", "0")
    runner = _make_runner(interactive_workers=1)
    try:
        turn_ctx = TurnContext(source=_src(platform), session_key="k", session_id="sid")

        async def go():
            worker = runner._run_agent_start_turn_worker(turn_ctx, lambda: threading.current_thread().name)
            return await worker.executor_task

        assert asyncio.run(go()).startswith(expected_prefix)
    finally:
        runner._shutdown_executor()
