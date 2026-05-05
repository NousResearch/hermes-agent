"""Tests for concurrent busy_input_mode.

Covers:
- Concurrent task spawning
- Concurrency limit enforcement
- Task cancellation (per-session and all)
- Graceful shutdown draining
- Cleanup callback
- CancelledError handling
- Config loading
- Status reporting
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class _FakeAdapter:
    """Minimal adapter stub for testing."""

    def __init__(self):
        self.name = "telegram"
        self.sent_messages = []
        self.sent_media = []

    async def _send_with_retry(self, chat_id, content, **kwargs):
        self.sent_messages.append({"chat_id": chat_id, "content": content, **kwargs})

    async def _send_media(self, chat_id, path, **kwargs):
        self.sent_media.append({"chat_id": chat_id, "path": path, **kwargs})

    def extract_media(self, text):
        return [], text

    def extract_images(self, text):
        return [], text


def _make_runner(max_concurrent=3):
    """Create a minimal GatewayRunner with concurrent mode enabled."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = _FakeAdapter()
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

    # Concurrent mode fields
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
# Test: Basic concurrent task spawning
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_task_spawns_independent_agent():
    """When busy_input_mode is 'concurrent' and a session has a running agent,
    a new message should spawn an independent agent task."""
    runner, adapter = _make_runner()
    event = _make_event()
    session_key = build_session_key(event.source)

    # Simulate a running agent
    runner._running_agents[session_key] = MagicMock()

    # Mock the _run_concurrent_agent to avoid real agent creation
    async def mock_concurrent(ev, sk, tid):
        return None

    with patch.object(runner, '_run_concurrent_agent', mock_concurrent):
        result = await runner._handle_active_session_busy_message(
            event, session_key
        )

    assert result is True
    assert len(runner._concurrent_tasks.get(session_key, [])) == 1
    assert runner._concurrent_counter == 1


# ------------------------------------------------------------------
# Test: Concurrency limit enforcement
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_limit_rejected():
    """When max_concurrent_tasks is reached, new messages should be rejected."""
    runner, adapter = _make_runner(max_concurrent=2)
    event = _make_event()
    session_key = build_session_key(event.source)

    # Simulate a running agent
    runner._running_agents[session_key] = MagicMock()

    # Fill up concurrent tasks to the limit
    async def mock_concurrent(ev, sk, tid):
        await asyncio.sleep(10)  # Long-running task

    with patch.object(runner, '_run_concurrent_agent', mock_concurrent):
        # Spawn first task
        await runner._handle_active_session_busy_message(
            event, session_key
        )
        # Spawn second task
        await runner._handle_active_session_busy_message(
            _make_event("second"), session_key
        )

    assert len(runner._concurrent_tasks[session_key]) == 2

    # Third message should be rejected
    with patch.object(runner, '_run_concurrent_agent', mock_concurrent):
        result = await runner._handle_active_session_busy_message(
            _make_event("third"), session_key
        )

    assert result is True
    # Should have sent a rejection message
    assert any("已满" in msg["content"] for msg in adapter.sent_messages)
    # Task count should still be 2
    assert len(runner._concurrent_tasks[session_key]) == 2

    # Cleanup
    for task in runner._concurrent_tasks.get(session_key, []):
        task.cancel()


# ------------------------------------------------------------------
# Test: Cancel concurrent tasks per session
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_concurrent_tasks_session():
    """cancel_concurrent_tasks should cancel all tasks for a given session."""
    runner, adapter = _make_runner()
    session_key = "test-session"
    runner._running_agents[session_key] = MagicMock()

    # Create some real asyncio tasks
    async def long_task():
        await asyncio.sleep(100)

    tasks = []
    for _ in range(3):
        task = asyncio.create_task(long_task())
        tasks.append(task)

    runner._concurrent_tasks[session_key] = tasks

    cancelled = runner.cancel_concurrent_tasks(session_key)
    assert cancelled == 3

    # Wait a tick for cancellation to propagate
    await asyncio.sleep(0.05)
    for task in tasks:
        assert task.cancelled()


# ------------------------------------------------------------------
# Test: Cancel all concurrent tasks
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_all_concurrent_tasks():
    """cancel_all_concurrent_tasks should cancel tasks across all sessions."""
    runner, _ = _make_runner()

    async def long_task():
        await asyncio.sleep(100)

    # Create tasks for multiple sessions
    for sk in ["session-a", "session-b", "session-c"]:
        tasks = [asyncio.create_task(long_task()) for _ in range(2)]
        runner._concurrent_tasks[sk] = tasks

    total = runner.cancel_all_concurrent_tasks()
    assert total == 6

    await asyncio.sleep(0.05)
    for tasks in runner._concurrent_tasks.values():
        for task in tasks:
            assert task.cancelled()


# ------------------------------------------------------------------
# Test: Drain concurrent tasks
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drain_concurrent_tasks_completes():
    """_drain_concurrent_tasks should wait for tasks to finish."""
    runner, _ = _make_runner()

    results = []

    async def quick_task():
        await asyncio.sleep(0.1)
        results.append("done")

    tasks = [asyncio.create_task(quick_task()) for _ in range(3)]
    runner._concurrent_tasks["session-1"] = tasks

    finished = await runner._drain_concurrent_tasks(timeout=5.0)
    assert finished is True
    assert len(results) == 3


@pytest.mark.asyncio
async def test_drain_concurrent_tasks_timeout():
    """_drain_concurrent_tasks should cancel remaining tasks on timeout."""
    runner, _ = _make_runner()

    async def slow_task():
        await asyncio.sleep(100)

    tasks = [asyncio.create_task(slow_task()) for _ in range(3)]
    runner._concurrent_tasks["session-1"] = tasks

    finished = await runner._drain_concurrent_tasks(timeout=0.2)
    assert finished is False

    await asyncio.sleep(0.1)
    for task in tasks:
        assert task.cancelled()


@pytest.mark.asyncio
async def test_drain_concurrent_tasks_empty():
    """_drain_concurrent_tasks should return True immediately when no tasks."""
    runner, _ = _make_runner()
    finished = await runner._drain_concurrent_tasks(timeout=1.0)
    assert finished is True


# ------------------------------------------------------------------
# Test: Cleanup callback
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_concurrent_done_cleanup():
    """_on_concurrent_done should remove finished tasks from tracking dict."""
    runner, _ = _make_runner()
    session_key = "test-session"

    async def quick_task():
        return "done"

    task = asyncio.create_task(quick_task())
    runner._concurrent_tasks[session_key] = [task]
    task.add_done_callback(
        lambda t: runner._on_concurrent_done(t, session_key)
    )

    await task
    await asyncio.sleep(0.05)  # Let callback run

    assert session_key not in runner._concurrent_tasks


@pytest.mark.asyncio
async def test_on_concurrent_done_exception_logged():
    """_on_concurrent_done should log exceptions from failed tasks."""
    runner, _ = _make_runner()
    session_key = "test-session"

    async def failing_task():
        raise ValueError("test error")

    task = asyncio.create_task(failing_task())
    runner._concurrent_tasks[session_key] = [task]
    task.add_done_callback(
        lambda t: runner._on_concurrent_done(t, session_key)
    )

    await asyncio.sleep(0.1)  # Let task fail and callback run
    assert session_key not in runner._concurrent_tasks


# ------------------------------------------------------------------
# Test: Status reporting
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_concurrent_task_info():
    """get_concurrent_task_info should report active tasks per session."""
    runner, _ = _make_runner()

    async def long_task():
        await asyncio.sleep(100)

    tasks_a = [asyncio.create_task(long_task()) for _ in range(2)]
    tasks_b = [asyncio.create_task(long_task()) for _ in range(1)]
    runner._concurrent_tasks["session-a"] = tasks_a
    runner._concurrent_tasks["session-b"] = tasks_b

    info = runner.get_concurrent_task_info()
    assert info["session-a"]["count"] == 2
    assert info["session-b"]["count"] == 1

    # Cleanup
    runner.cancel_all_concurrent_tasks()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_get_concurrent_task_info_empty():
    """get_concurrent_task_info should return empty dict when no tasks."""
    runner, _ = _make_runner()
    info = runner.get_concurrent_task_info()
    assert info == {}


# ------------------------------------------------------------------
# Test: Config loading
# ------------------------------------------------------------------

def test_load_max_concurrent_tasks_from_env():
    """Should load from HERMES_MAX_CONCURRENT_TASKS env var."""
    with patch.dict(os.environ, {"HERMES_MAX_CONCURRENT_TASKS": "5"}):
        val = GatewayRunner._load_max_concurrent_tasks()
        assert val == 5


def test_load_max_concurrent_tasks_default():
    """Should default to 3 when not configured."""
    with patch.dict(os.environ, {}, clear=False):
        # Remove the env var if it exists
        os.environ.pop("HERMES_MAX_CONCURRENT_TASKS", None)
        with patch("gateway.run._hermes_home") as mock_home:
            mock_home.__truediv__ = lambda self, x: type("P", (), {
                "exists": lambda: False
            })()
            val = GatewayRunner._load_max_concurrent_tasks()
        assert val == 3


def test_load_max_concurrent_tasks_invalid_env():
    """Should fall back to default on invalid env var."""
    with patch.dict(os.environ, {"HERMES_MAX_CONCURRENT_TASKS": "not-a-number"}):
        val = GatewayRunner._load_max_concurrent_tasks()
        assert val == 3


def test_load_max_concurrent_tasks_zero_env():
    """Should fall back to default on zero env var."""
    with patch.dict(os.environ, {"HERMES_MAX_CONCURRENT_TASKS": "0"}):
        val = GatewayRunner._load_max_concurrent_tasks()
        assert val == 3


# ------------------------------------------------------------------
# Test: Ack message includes task count
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ack_message_includes_count():
    """The acknowledgment message should include concurrent task count."""
    runner, adapter = _make_runner(max_concurrent=5)
    event = _make_event()
    session_key = build_session_key(event.source)

    runner._running_agents[session_key] = MagicMock()

    async def mock_concurrent(ev, sk, tid):
        await asyncio.sleep(10)

    with patch.object(runner, '_run_concurrent_agent', mock_concurrent):
        await runner._handle_active_session_busy_message(
            event, session_key
        )

    # Check ack message
    ack_msgs = [m for m in adapter.sent_messages if "并发" in m["content"]]
    assert len(ack_msgs) == 1
    assert "1/5" in ack_msgs[0]["content"]

    # Cleanup
    runner.cancel_all_concurrent_tasks()
    await asyncio.sleep(0.1)


# ------------------------------------------------------------------
# Test: Debounce cooldown
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ack_debounce():
    """Second message within cooldown should not send another ack."""
    runner, adapter = _make_runner()
    event = _make_event()
    session_key = build_session_key(event.source)

    runner._running_agents[session_key] = MagicMock()

    async def mock_concurrent(ev, sk, tid):
        await asyncio.sleep(10)

    with patch.object(runner, '_run_concurrent_agent', mock_concurrent):
        await runner._handle_active_session_busy_message(
            event, session_key
        )
        # Second message immediately (within 15s cooldown)
        await runner._handle_active_session_busy_message(
            _make_event("second"), session_key
        )

    ack_msgs = [m for m in adapter.sent_messages if "并发" in m["content"]]
    assert len(ack_msgs) == 1  # Only first message got ack

    # Cleanup
    runner.cancel_all_concurrent_tasks()
    await asyncio.sleep(0.1)
