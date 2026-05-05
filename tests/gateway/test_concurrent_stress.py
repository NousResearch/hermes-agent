"""Stress tests for concurrent busy_input_mode.

Simulates real-world scenarios:
- Rapid-fire messages from a single user
- Multiple sessions hitting concurrency limits
- Mixed completion times
- Cancel during heavy load
- Shutdown under load
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class _TrackingAdapter:
    """Adapter that tracks all sent messages with timestamps."""

    def __init__(self):
        self.name = "telegram"
        self.sent = []

    async def _send_with_retry(self, chat_id, content, **kwargs):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "time": time.monotonic(),
        })

    async def _send_media(self, chat_id, path, **kwargs):
        pass

    def extract_media(self, text):
        return [], text

    def extract_images(self, text):
        return [], text


def _make_runner(max_concurrent=3):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = _TrackingAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.delivery_router = MagicMock()
    runner._concurrent_tasks = {}
    runner._concurrent_counter = 0
    runner._busy_input_mode = "concurrent"
    runner._busy_ack_ts = {}
    runner._max_concurrent_tasks = max_concurrent
    return runner, adapter


def _make_event(text="hello", chat_id="12345", user_id="u1"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm",
        user_id=user_id,
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


# ------------------------------------------------------------------
# Stress Test 1: Rapid-fire messages from single user
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rapid_fire_messages():
    """Simulate a user sending 10 messages in quick succession
    with max_concurrent=3. Only 3 should spawn, rest rejected."""
    runner, adapter = _make_runner(max_concurrent=3)
    event = _make_event()
    session_key = build_session_key(event.source)
    runner._running_agents[session_key] = MagicMock()

    spawned = 0
    rejected = 0

    async def mock_concurrent(ev, sk, tid):
        nonlocal spawned
        spawned += 1
        await asyncio.sleep(0.5)  # Simulate work

    with patch.object(runner, '_run_concurrent_agent', mock_concurrent):
        tasks = []
        for i in range(10):
            result = await runner._handle_active_session_busy_message(
                _make_event(f"msg-{i}"), session_key
            )
            if result:
                # Check if it was spawned or rejected
                pass

        # Wait for all tasks
        all_tasks = [
            t for tasks in runner._concurrent_tasks.values()
            for t in tasks
        ]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

    # Should have spawned exactly 3
    assert spawned == 3
    # Should have rejection messages
    rejections = [m for m in adapter.sent if "已满" in m["content"]]
    assert len(rejections) >= 1  # At least some rejections


# ------------------------------------------------------------------
# Stress Test 2: Multiple sessions competing
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_sessions_competing():
    """Multiple sessions each trying to spawn tasks, sharing the global limit."""
    runner, adapter = _make_runner(max_concurrent=5)

    # Create 3 sessions with running agents
    sessions = []
    for i in range(3):
        event = _make_event(chat_id=f"chat-{i}", user_id=f"user-{i}")
        sk = build_session_key(event.source)
        runner._running_agents[sk] = MagicMock()
        sessions.append((event, sk))

    spawned = 0

    async def mock_concurrent(ev, sk, tid):
        nonlocal spawned
        spawned += 1
        await asyncio.sleep(0.3)

    with patch.object(runner, '_run_concurrent_agent', mock_concurrent):
        # Each session spawns 2 tasks = 6 total, but limit is 5
        for event, sk in sessions:
            for i in range(2):
                await runner._handle_active_session_busy_message(
                    _make_event(f"msg-{i}", event.source.chat_id, event.source.user_id),
                    sk
                )

        all_tasks = [
            t for tasks in runner._concurrent_tasks.values()
            for t in tasks
        ]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

    # Should have spawned exactly 5 (the limit)
    assert spawned == 5
    # Should have at least one rejection
    rejections = [m for m in adapter.sent if "已满" in m["content"]]
    assert len(rejections) >= 1


# ------------------------------------------------------------------
# Stress Test 3: Mixed completion times
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_completion_times():
    """Tasks with different completion times should all complete properly."""
    runner, adapter = _make_runner(max_concurrent=5)
    event = _make_event()
    session_key = build_session_key(event.source)
    runner._running_agents[session_key] = MagicMock()

    completed = []

    async def variable_task(ev, sk, tid):
        delay = float(ev.text) if ev.text.replace(".", "").isdigit() else 0.1
        await asyncio.sleep(delay)
        completed.append(tid)

    with patch.object(runner, '_run_concurrent_agent', variable_task):
        for delay in ["0.05", "0.1", "0.2", "0.15", "0.08"]:
            await runner._handle_active_session_busy_message(
                _make_event(delay), session_key
            )

        all_tasks = [
            t for tasks in runner._concurrent_tasks.values()
            for t in tasks
        ]
        await asyncio.gather(*all_tasks, return_exceptions=True)

    assert len(completed) == 5
    # All tasks should have unique IDs
    assert len(set(completed)) == 5


# ------------------------------------------------------------------
# Stress Test 4: Cancel under heavy load
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_under_heavy_load():
    """Cancel all tasks while many are running."""
    runner, _ = _make_runner(max_concurrent=10)

    async def long_task():
        await asyncio.sleep(100)

    # Spawn 10 tasks
    tasks = [asyncio.create_task(long_task()) for _ in range(10)]
    runner._concurrent_tasks["heavy-session"] = tasks

    # Cancel all
    cancelled = runner.cancel_all_concurrent_tasks()
    assert cancelled == 10

    await asyncio.sleep(0.1)
    for task in tasks:
        assert task.cancelled()


# ------------------------------------------------------------------
# Stress Test 5: Drain with partial completion
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drain_partial_completion():
    """Some tasks finish fast, some slow. Drain should wait for all
    within timeout, then cancel the rest."""
    runner, _ = _make_runner()

    fast_done = False
    slow_done = False

    async def fast_task():
        nonlocal fast_done
        await asyncio.sleep(0.1)
        fast_done = True

    async def slow_task():
        nonlocal slow_done
        await asyncio.sleep(100)
        slow_done = True

    tasks = [
        asyncio.create_task(fast_task()),
        asyncio.create_task(fast_task()),
        asyncio.create_task(slow_task()),
        asyncio.create_task(slow_task()),
    ]
    runner._concurrent_tasks["mixed-session"] = tasks

    finished = await runner._drain_concurrent_tasks(timeout=0.5)
    assert finished is False  # Should timeout
    assert fast_done is True  # Fast tasks should have completed
    assert slow_done is False  # Slow tasks should have been cancelled


# ------------------------------------------------------------------
# Stress Test 6: Shutdown under load
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shutdown_under_load():
    """Shutdown should drain concurrent tasks before active agents."""
    runner, adapter = _make_runner(max_concurrent=5)

    shutdown_order = []

    async def track_task(ev, sk, tid):
        await asyncio.sleep(0.2)
        shutdown_order.append(f"concurrent-{tid}")

    # Simulate running agents
    event = _make_event()
    session_key = build_session_key(event.source)
    runner._running_agents[session_key] = MagicMock()

    async def mock_concurrent(ev, sk, tid):
        shutdown_order.append(f"start-{tid}")
        await asyncio.sleep(0.3)
        shutdown_order.append(f"end-{tid}")

    with patch.object(runner, '_run_concurrent_agent', mock_concurrent):
        # Spawn 3 concurrent tasks
        for i in range(3):
            await runner._handle_active_session_busy_message(
                _make_event(f"msg-{i}"), session_key
            )

        # Start shutdown
        runner._draining = True
        await runner._drain_concurrent_tasks(timeout=2.0)

    # All tasks should have completed
    assert len([x for x in shutdown_order if x.startswith("end-")]) == 3


# ------------------------------------------------------------------
# Stress Test 7: Task ID uniqueness
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_id_uniqueness():
    """All task IDs should be unique even under rapid spawning."""
    runner, _ = _make_runner(max_concurrent=100)
    event = _make_event()
    session_key = build_session_key(event.source)
    runner._running_agents[session_key] = MagicMock()

    task_ids = []

    async def capture_task(ev, sk, tid):
        task_ids.append(tid)
        await asyncio.sleep(0.01)

    with patch.object(runner, '_run_concurrent_agent', capture_task):
        for i in range(20):
            await runner._handle_active_session_busy_message(
                _make_event(f"msg-{i}"), session_key
            )

        all_tasks = [
            t for tasks in runner._concurrent_tasks.values()
            for t in tasks
        ]
        await asyncio.gather(*all_tasks, return_exceptions=True)

    assert len(task_ids) == 20
    assert len(set(task_ids)) == 20  # All unique


# ------------------------------------------------------------------
# Stress Test 8: Memory isolation
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_tasks_dont_share_state():
    """Concurrent tasks should not interfere with each other's tracking."""
    runner, _ = _make_runner(max_concurrent=5)

    async def quick_task():
        await asyncio.sleep(0.05)

    # Spawn tasks for multiple sessions
    for i in range(3):
        sk = f"session-{i}"
        tasks = [asyncio.create_task(quick_task()) for _ in range(2)]
        runner._concurrent_tasks[sk] = tasks

    # Attach cleanup callbacks (like the real code does)
    for sk, tasks in runner._concurrent_tasks.items():
        for task in tasks:
            task.add_done_callback(
                lambda t, s=sk: runner._on_concurrent_done(t, s)
            )

    # Cancel session-0's tasks only
    cancelled = runner.cancel_concurrent_tasks("session-0")
    assert cancelled == 2

    await asyncio.sleep(0.1)

    # session-0 should be cleaned up via callback
    assert "session-0" not in runner._concurrent_tasks
    # Other sessions' tasks completed naturally and cleaned up too
    assert "session-1" not in runner._concurrent_tasks
    assert "session-2" not in runner._concurrent_tasks
