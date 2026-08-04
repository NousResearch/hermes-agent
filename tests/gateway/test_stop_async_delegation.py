"""Regression coverage for /stop and background async delegations.

The gateway owns two independent execution lanes for one messaging session:
the foreground AIAgent and detached ``delegate_task`` workers. ``/stop`` must
cancel both lanes, drop an already-queued completion, and clear any restart
auto-resume marker without affecting another thread.
"""

import asyncio
import queue
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    Platform,
    _COMPLETION_ADMISSION_REJECTED_KEY,
    _COMPLETION_ADMISSION_TOKEN_KEY,
)
from gateway.run import GatewayRunner, _INTERRUPT_REASON_STOP
from gateway.session import SessionSource, build_session_key
from tools import async_delegation as ad
from tools.process_registry import ProcessSession, process_registry


@pytest.fixture(autouse=True)
def _clean_async_delegations():
    ad._reset_for_tests()
    with process_registry._lock:
        process_registry._running.clear()
        process_registry._finished.clear()
    process_registry.pending_watchers = []
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._reset_for_tests()
    with process_registry._lock:
        process_registry._running.clear()
        process_registry._finished.clear()
    process_registry.pending_watchers = []
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


class _SessionEntry:
    def __init__(self, session_key, session_id, *, resume_pending=True):
        self.session_key = session_key
        self.session_id = session_id
        self.resume_pending = resume_pending


class _SessionStore:
    def __init__(self, entry):
        self.entry = entry
        self.clear_calls = []

    def get_or_create_session(self, _source):
        return self.entry

    def clear_resume_pending(self, session_key):
        self.clear_calls.append(session_key)
        changed = self.entry.resume_pending
        self.entry.resume_pending = False
        return changed


class _AdmissionAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _source(
    thread_id,
    *,
    platform=Platform.DISCORD,
    chat_type="forum",
):
    return SessionSource(
        platform=platform,
        chat_type=chat_type,
        chat_id="channel",
        thread_id=thread_id,
        user_id="user",
    )


def _wait_for_inactive(timeout=5):
    deadline = time.monotonic() + timeout
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)


def _take_completion(delegation_id):
    while not process_registry.completion_queue.empty():
        candidate = process_registry.completion_queue.get_nowait()
        if candidate.get("delegation_id") == delegation_id:
            return candidate
    return None


def _delivery_runner(*, mock_injection=True):
    runner = object.__new__(GatewayRunner)
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 16
    if mock_injection:
        runner._inject_watch_notification = AsyncMock(return_value=True)
    session_db = MagicMock()
    session_db.get_session = AsyncMock(
        return_value={"id": "parent-a", "ended_at": None}
    )
    runner._session_db = session_db
    return runner


def _admission_adapter(runner, *, platform=Platform.DISCORD):
    adapter = _AdmissionAdapter(
        PlatformConfig(enabled=True, token="test-token"),
        platform,
    )
    handler = AsyncMock(return_value=None)
    adapter.set_message_handler(handler)
    adapter.set_completion_admission_validator(
        runner._validate_completion_admission
    )
    return adapter, handler


def _tokenized_completion(runner, source):
    session_key = build_session_key(source)
    return MessageEvent(
        text="[SYSTEM: background delegation completed]",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={
            _COMPLETION_ADMISSION_TOKEN_KEY: (
                runner._capture_completion_admission(session_key)
            )
        },
    )


@pytest.mark.asyncio
async def test_interrupt_and_clear_session_cancels_foreground_and_background(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = _source("thread-a")
    session_key = build_session_key(source)
    entry = _SessionEntry(session_key, "parent-a")
    store = _SessionStore(entry)

    gate = threading.Event()
    child_interrupted = MagicMock(side_effect=gate.set)

    def background_runner():
        gate.wait(timeout=60)
        return {"status": "interrupted", "summary": None}

    dispatched = ad.dispatch_async_delegation(
        goal="background work",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key=session_key,
        parent_session_id=entry.session_id,
        runner=background_runner,
        interrupt_fn=child_interrupted,
    )

    runner = object.__new__(GatewayRunner)
    foreground = MagicMock()
    foreground.session_id = entry.session_id
    runner._running_agents = {session_key: foreground}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner.session_store = store
    runner._adapter_for_source = lambda _source: None
    runner._release_running_agent_state = lambda key: runner._running_agents.pop(
        key, None
    )
    runner._evict_cached_agent = lambda _key: None

    await runner._interrupt_and_clear_session(
        session_key,
        source,
        interrupt_reason=_INTERRUPT_REASON_STOP,
        invalidation_reason="stop_command",
    )

    foreground.interrupt.assert_called_once_with(_INTERRUPT_REASON_STOP)
    child_interrupted.assert_called_once()
    assert store.clear_calls == [session_key]
    assert entry.resume_pending is False
    assert session_key not in runner._running_agents

    _wait_for_inactive()
    assert ad.active_count() == 0
    durable = ad.get_durable_delegation(dispatched["delegation_id"])
    assert durable["state"] == "cancelled"
    assert durable["delivery_state"] == "dropped"


@pytest.mark.asyncio
async def test_stop_with_only_background_work_is_scoped_and_idempotent(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source_a = _source("thread-a")
    source_b = _source("thread-b")
    key_a = build_session_key(source_a)
    key_b = build_session_key(source_b)
    entry = _SessionEntry(key_a, "parent-a")
    store = _SessionStore(entry)

    gate_a = threading.Event()
    gate_b = threading.Event()
    interrupted_a = MagicMock(side_effect=gate_a.set)
    interrupted_b = MagicMock(side_effect=gate_b.set)

    def runner_a():
        gate_a.wait(timeout=60)
        return {"status": "interrupted", "summary": None}

    def runner_b():
        gate_b.wait(timeout=60)
        return {"status": "completed", "summary": "other"}

    mine = ad.dispatch_async_delegation(
        goal="mine",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key=key_a,
        parent_session_id="parent-a",
        runner=runner_a,
        interrupt_fn=interrupted_a,
    )
    other = ad.dispatch_async_delegation(
        goal="other",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key=key_b,
        parent_session_id="parent-b",
        runner=runner_b,
        interrupt_fn=interrupted_b,
    )

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner.session_store = store
    runner.adapters = {}
    runner._is_user_authorized = lambda _source: True

    event = MessageEvent(
        text="/stop",
        message_type=MessageType.TEXT,
        source=source_a,
    )
    try:
        first = await runner._handle_stop_command(event)
        assert "no active" not in str(first).lower()
        interrupted_a.assert_called_once()
        interrupted_b.assert_not_called()
        assert entry.resume_pending is False

        second = await runner._handle_stop_command(event)
        assert "no active" in str(second).lower()
        interrupted_a.assert_called_once()

        mine_row = ad.get_durable_delegation(mine["delegation_id"])
        other_row = ad.get_durable_delegation(other["delegation_id"])
        assert mine_row["state"] == "cancelled"
        assert mine_row["delivery_state"] == "dropped"
        assert other_row["state"] == "running"
        assert other_row["delivery_state"] == "pending"
    finally:
        gate_b.set()


@pytest.mark.asyncio
async def test_stop_cancels_terminal_process_and_drops_its_notifications(
    tmp_path, monkeypatch
):
    """A stopped chat must not be revived by terminal notify_on_complete."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source_a = _source("thread-a")
    source_b = _source("thread-b")
    key_a = build_session_key(source_a)
    key_b = build_session_key(source_b)
    entry = _SessionEntry(key_a, "parent-a")
    store = _SessionStore(entry)

    mine = ProcessSession(
        id="proc_mine",
        command="long-running-test",
        session_key=key_a,
        started_at=time.time(),
        pid=1001,
        process=MagicMock(pid=1001),
        watcher_interval=5,
        notify_on_complete=True,
    )
    other = ProcessSession(
        id="proc_other",
        command="other-session-test",
        session_key=key_b,
        started_at=time.time(),
        pid=1002,
        process=MagicMock(pid=1002),
        watcher_interval=5,
        notify_on_complete=True,
    )
    with process_registry._lock:
        process_registry._running[mine.id] = mine
        process_registry._running[other.id] = other
    process_registry.pending_watchers = [
        {"session_id": mine.id, "session_key": key_a},
        {"session_id": other.id, "session_key": key_b},
    ]
    process_registry.completion_queue.put(
        {
            "type": "completion",
            "session_id": mine.id,
            "session_key": key_a,
            "started_at": mine.started_at,
        }
    )
    process_registry.completion_queue.put(
        {
            "type": "completion",
            "session_id": other.id,
            "session_key": key_b,
            "started_at": other.started_at,
        }
    )
    terminate = MagicMock()
    monkeypatch.setattr(process_registry, "_terminate_host_pid", terminate)

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner.session_store = store
    runner.adapters = {}
    runner._is_user_authorized = lambda _source: True
    event = MessageEvent(
        text="/stop",
        message_type=MessageType.TEXT,
        source=source_a,
    )

    first = await runner._handle_stop_command(event)

    assert "no active" not in str(first).lower()
    assert mine.exited is True
    assert mine.completion_reason == "killed"
    assert other.exited is False
    terminate.assert_called_once_with(1001, None)
    assert process_registry.pending_watchers == [
        {"session_id": other.id, "session_key": key_b}
    ]
    queued = []
    while not process_registry.completion_queue.empty():
        queued.append(process_registry.completion_queue.get_nowait())
    assert [item["session_id"] for item in queued] == [other.id]

    second = await runner._handle_stop_command(event)
    assert "no active" in str(second).lower()
    terminate.assert_called_once()


@pytest.mark.asyncio
async def test_stop_drops_already_finished_terminal_completion(
    tmp_path, monkeypatch
):
    """A queued completion remains cancellable after its process has exited."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = _source("thread-a")
    session_key = build_session_key(source)
    entry = _SessionEntry(session_key, "parent-a")
    store = _SessionStore(entry)
    finished = ProcessSession(
        id="proc_finished",
        command="already-done",
        session_key=session_key,
        started_at=time.time(),
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    with process_registry._lock:
        process_registry._finished[finished.id] = finished
    process_registry.completion_queue.put(
        {
            "type": "completion",
            "session_id": finished.id,
            "session_key": session_key,
            "started_at": finished.started_at,
        }
    )

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner.session_store = store
    runner.adapters = {}
    runner._is_user_authorized = lambda _source: True
    event = MessageEvent(
        text="/stop",
        message_type=MessageType.TEXT,
        source=source,
    )

    result = await runner._handle_stop_command(event)

    assert "no active" not in str(result).lower()
    assert finished.completion_suppressed is True
    assert finished.notify_on_complete is False
    assert process_registry.completion_queue.empty()


@pytest.mark.asyncio
async def test_gateway_drops_completion_cancelled_by_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    dispatched = ad.dispatch_async_delegation(
        goal="race",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="discord-thread-a",
        parent_session_id="parent-a",
        runner=lambda: {"status": "completed", "summary": "must not re-enter"},
    )
    _wait_for_inactive()
    assert ad.active_count() == 0
    assert ad.interrupt_for_session(
        session_key="discord-thread-a",
        parent_session_id="parent-a",
        reason="stop_command",
    ) == 1

    evt = _take_completion(dispatched["delegation_id"])
    assert evt is not None

    runner = _delivery_runner()
    assert await runner._deliver_completion_notification("synthetic", evt) is None
    runner._inject_watch_notification.assert_not_awaited()
    assert ad.restore_undelivered_completions(queue.Queue()) == 0


@pytest.mark.asyncio
async def test_gateway_revalidates_claim_when_stop_wins_delivery_race(
    tmp_path, monkeypatch
):
    """A stop arriving just after claim must still block synthetic injection."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    dispatched = ad.dispatch_async_delegation(
        goal="claim race",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="discord-thread-a",
        parent_session_id="parent-a",
        runner=lambda: {"status": "completed", "summary": "must not re-enter"},
    )
    _wait_for_inactive()
    assert ad.active_count() == 0

    evt = _take_completion(dispatched["delegation_id"])
    assert evt is not None

    real_claim = ad.claim_completion_delivery

    def claim_then_stop(delegation_id, claim_id):
        assert real_claim(delegation_id, claim_id) is True
        assert ad.interrupt_for_session(
            session_key="discord-thread-a",
            parent_session_id="parent-a",
            reason="stop_command",
        ) == 1
        return True

    monkeypatch.setattr(ad, "claim_completion_delivery", claim_then_stop)

    runner = _delivery_runner()
    assert await runner._deliver_completion_notification("synthetic", evt) is None
    runner._inject_watch_notification.assert_not_awaited()
    durable = ad.get_durable_delegation(dispatched["delegation_id"])
    assert durable["state"] == "cancelled"
    assert durable["delivery_state"] == "dropped"


@pytest.mark.asyncio
async def test_stop_after_claim_revalidation_blocks_adapter_admission(
    tmp_path, monkeypatch
):
    """A stop during Telegram topic recovery wins after the final claim read."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = _source(
        "thread-a",
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    session_key = build_session_key(source)
    dispatched = ad.dispatch_async_delegation(
        goal="post-claim race",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key=session_key,
        parent_session_id="parent-a",
        runner=lambda: {"status": "completed", "summary": "must not re-enter"},
    )
    _wait_for_inactive()
    evt = _take_completion(dispatched["delegation_id"])
    assert evt is not None

    runner = _delivery_runner(mock_injection=False)
    adapter, handler = _admission_adapter(runner, platform=Platform.TELEGRAM)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={session_key: SimpleNamespace(origin=source)},
    )
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        clear_resume_pending=AsyncMock(return_value=False),
    )

    recovery_entered = threading.Event()
    release_recovery = threading.Event()

    def blocking_topic_recovery(_source):
        recovery_entered.set()
        release_recovery.wait(timeout=5)
        return None

    adapter.set_topic_recovery_fn(blocking_topic_recovery)
    delivery_task = asyncio.create_task(
        runner._deliver_completion_notification("synthetic", evt)
    )

    entered = await asyncio.wait_for(
        asyncio.to_thread(recovery_entered.wait, 3),
        timeout=4,
    )
    cancelled = -1
    if entered:
        try:
            cancelled = await runner._cancel_session_background_work(
                session_key,
                parent_session_id="parent-a",
                reason="stop_command",
            )
        finally:
            release_recovery.set()
    else:
        release_recovery.set()

    result = await asyncio.wait_for(delivery_task, timeout=5)
    await adapter.cancel_background_tasks()

    assert entered is True
    assert cancelled == 1
    assert result is None
    handler.assert_not_awaited()
    assert session_key not in adapter._active_sessions
    durable = ad.get_durable_delegation(dispatched["delegation_id"])
    assert durable["state"] == "cancelled"
    assert durable["delivery_state"] == "dropped"


@pytest.mark.asyncio
async def test_terminal_completion_stop_race_blocks_adapter_admission(
    tmp_path, monkeypatch
):
    """A terminal completion in Telegram topic recovery loses to /stop."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = _source(
        "thread-a",
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    session_key = build_session_key(source)
    finished = ProcessSession(
        id="proc_admission_race",
        command="race-test",
        session_key=session_key,
        started_at=time.time(),
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    with process_registry._lock:
        process_registry._finished[finished.id] = finished
    evt = {
        "type": "completion",
        "session_id": finished.id,
        "session_key": session_key,
        "platform": source.platform.value,
        "chat_type": source.chat_type,
        "chat_id": source.chat_id,
        "thread_id": source.thread_id,
        "user_id": source.user_id,
        "started_at": finished.started_at,
        "command": finished.command,
        "exit_code": 0,
        "output": "done",
    }

    runner = _delivery_runner(mock_injection=False)
    adapter, handler = _admission_adapter(runner, platform=Platform.TELEGRAM)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={session_key: SimpleNamespace(origin=source)},
    )
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        clear_resume_pending=AsyncMock(return_value=False),
    )

    recovery_entered = threading.Event()
    release_recovery = threading.Event()

    def blocking_topic_recovery(_source):
        recovery_entered.set()
        release_recovery.wait(timeout=5)
        return None

    adapter.set_topic_recovery_fn(blocking_topic_recovery)
    delivery_task = asyncio.create_task(
        runner._deliver_completion_notification("synthetic", evt)
    )

    entered = await asyncio.wait_for(
        asyncio.to_thread(recovery_entered.wait, 3),
        timeout=4,
    )
    if entered:
        try:
            await runner._cancel_session_background_work(
                session_key,
                parent_session_id="parent-a",
                reason="stop_command",
            )
        finally:
            release_recovery.set()
    else:
        release_recovery.set()

    result = await asyncio.wait_for(delivery_task, timeout=5)
    await adapter.cancel_background_tasks()

    assert entered is True
    assert result is None
    assert finished.completion_suppressed is True
    handler.assert_not_awaited()
    assert session_key not in adapter._active_sessions


@pytest.mark.asyncio
async def test_other_session_terminal_completion_is_admitted():
    """An unrelated process completion keeps its own admission epoch."""
    source = _source("thread-b")
    session_key = build_session_key(source)
    finished = ProcessSession(
        id="proc_other_completion",
        command="other-test",
        session_key=session_key,
        started_at=time.time(),
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    with process_registry._lock:
        process_registry._finished[finished.id] = finished
    evt = {
        "type": "completion",
        "session_id": finished.id,
        "session_key": session_key,
        "platform": Platform.DISCORD.value,
        "chat_type": source.chat_type,
        "chat_id": source.chat_id,
        "thread_id": source.thread_id,
        "user_id": source.user_id,
        "started_at": finished.started_at,
        "command": finished.command,
        "exit_code": 0,
        "output": "done",
    }

    runner = _delivery_runner(mock_injection=False)
    adapter, handler = _admission_adapter(runner)
    runner.adapters = {Platform.DISCORD: adapter}
    processed = asyncio.Event()

    async def mark_processed(_event):
        processed.set()

    handler.side_effect = mark_processed

    result = await runner._deliver_completion_notification("synthetic", evt)
    await asyncio.wait_for(processed.wait(), timeout=2)
    await adapter.cancel_background_tasks()

    assert result is True
    handler.assert_awaited_once()
    delivered_event = handler.await_args.args[0]
    admission = delivered_event.metadata[_COMPLETION_ADMISSION_TOKEN_KEY]
    assert admission["session_key"] == session_key
    assert admission["epoch"] == runner._current_completion_admission_epoch(
        session_key
    )
    assert _COMPLETION_ADMISSION_REJECTED_KEY not in delivered_event.metadata


@pytest.mark.asyncio
async def test_stale_queued_completion_is_not_drained_after_stop_epoch():
    runner = object.__new__(GatewayRunner)
    source = _source("thread-a")
    session_key = build_session_key(source)
    adapter, handler = _admission_adapter(runner)
    command_guard = asyncio.Event()
    adapter._active_sessions[session_key] = command_guard
    event = _tokenized_completion(runner, source)

    await adapter.handle_message(event)
    assert adapter._pending_messages[session_key] is event

    runner._invalidate_completion_admission_epoch(
        session_key,
        reason="stop_command",
    )
    await adapter._drain_pending_after_session_command(
        session_key,
        command_guard,
    )
    await asyncio.sleep(0)

    handler.assert_not_awaited()
    assert event.metadata[_COMPLETION_ADMISSION_REJECTED_KEY] is True
    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._session_tasks


@pytest.mark.asyncio
async def test_current_queued_completion_drains_normally():
    runner = object.__new__(GatewayRunner)
    source = _source("thread-a")
    session_key = build_session_key(source)
    adapter, handler = _admission_adapter(runner)
    processed = asyncio.Event()

    async def mark_processed(_event):
        processed.set()

    handler.side_effect = mark_processed
    command_guard = asyncio.Event()
    adapter._active_sessions[session_key] = command_guard
    event = _tokenized_completion(runner, source)

    await adapter.handle_message(event)
    assert adapter._pending_messages[session_key] is event

    await adapter._drain_pending_after_session_command(
        session_key,
        command_guard,
    )
    await asyncio.wait_for(processed.wait(), timeout=2)
    await adapter.cancel_background_tasks()

    handler.assert_awaited_once_with(event)
    assert _COMPLETION_ADMISSION_REJECTED_KEY not in event.metadata
