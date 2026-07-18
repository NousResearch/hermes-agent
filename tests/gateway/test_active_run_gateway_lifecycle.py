"""Gateway lifecycle integration tests for exact active-run recovery."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.platforms.base import MessageEvent, MessageType
from gateway.recovery import PHASE_EXECUTING, PHASE_RESPONSE_READY, ActiveRunStore
from gateway.run import GatewayRunner
from gateway.session import (
    AsyncSessionStore,
    SessionEntry,
    SessionSource,
    SessionStore,
    build_session_key,
)
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _source(chat_id: str = "chat-1") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="dm",
        user_id="owner",
        thread_id="thread-1",
    )


def _entry(
    source: SessionSource, *, updated_at: datetime | None = None
) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=f"sid-{source.chat_id}",
        created_at=now - timedelta(hours=3),
        updated_at=updated_at or now,
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _minimal_store(tmp_path, entry: SessionEntry) -> SessionStore:
    config = GatewayConfig(
        sessions_dir=tmp_path,
        write_sessions_json=False,
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=1),
    )
    store = SessionStore(tmp_path, config)
    store._db = None
    store._entries = {entry.session_key: entry}
    store._loaded = True
    return store


@pytest.mark.asyncio
async def test_old_exact_run_recovers_even_with_clean_marker_and_idle_policy(
    tmp_path, monkeypatch
):
    source = _source()
    entry = _entry(source, updated_at=datetime.now() - timedelta(hours=2))
    store = _minimal_store(tmp_path, entry)
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._active_run_store = ActiveRunStore(tmp_path)
    runner._recovery_boot_id = "boot-a"
    record = runner._active_run_store.begin(
        entry.session_key,
        trigger_message_id="message-1",
        started_at=time.time() - 7200,
    )
    # The exact journal is authoritative even if a graceful marker exists.
    (tmp_path / ".clean_shutdown").write_text("clean", encoding="utf-8")

    facade = AsyncSessionStore(store)
    facade.load_transcript = AsyncMock(
        return_value=[
            {
                "role": "user",
                "content": "long-running request",
                "message_id": "message-1",
                "timestamp": time.time() - 7200,
            }
        ]
    )

    async def _mark(session_key, reason):
        return store.mark_resume_pending(session_key, reason)

    facade.mark_resume_pending = _mark
    runner._async_session_store = facade
    monkeypatch.setattr(
        "tools.process_registry.process_registry.has_active_for_session",
        lambda _key: False,
    )

    exact_keys = await runner._prepare_active_run_recovery()

    assert exact_keys == {entry.session_key}
    recovered = store._entries[entry.session_key]
    assert recovered.resume_pending is True
    assert recovered.resume_reason == "restart_interrupted_exact"
    assert runner._active_recovery_reasons[entry.session_key] == (
        "restart_interrupted_exact"
    )
    assert runner._active_run_store.get(entry.session_key).run_id == record.run_id
    assert runner._active_run_store.get(entry.session_key).recovery_attempts == 1
    # Durable exact recovery also bypasses the ordinary idle reset policy.
    assert store._is_session_expired(recovered) is False
    assert store.get_or_create_session(source).session_id == entry.session_id


@pytest.mark.asyncio
async def test_missing_journal_clears_stale_exact_resume_marker(tmp_path):
    source = _source()
    entry = _entry(source)
    entry.resume_pending = True
    entry.resume_reason = "restart_interrupted_exact"
    store = _minimal_store(tmp_path, entry)
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._active_run_store = ActiveRunStore(tmp_path)

    exact_keys = await runner._prepare_active_run_recovery()

    assert exact_keys == set()
    assert entry.resume_pending is False
    assert entry.resume_reason is None


def test_durable_pause_is_not_expired_or_pruned(tmp_path):
    source = _source()
    entry = _entry(source, updated_at=datetime.now() - timedelta(days=2))
    entry.resume_pending = True
    entry.resume_reason = "side_effect_unknown"
    store = _minimal_store(tmp_path, entry)

    assert store._is_session_expired(entry) is False
    assert store.prune_old_entries(max_age_days=1) == 0
    assert store._entries[entry.session_key] is entry


@pytest.mark.asyncio
async def test_explicitly_suspended_session_discards_orphan_journal(tmp_path):
    source = _source()
    entry = _entry(source)
    entry.suspended = True
    store = _minimal_store(tmp_path, entry)
    active_store = ActiveRunStore(tmp_path)
    active_store.begin(entry.session_key, trigger_message_id="message-1")
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._active_run_store = active_store
    runner._recovery_boot_id = "boot-a"

    exact_keys = await runner._prepare_active_run_recovery()

    assert exact_keys == set()
    assert active_store.get(entry.session_key) is None


def test_legacy_120_second_heuristic_excludes_exact_journal_keys(tmp_path):
    source_exact = _source("exact")
    source_legacy = _source("legacy")
    exact = _entry(source_exact)
    legacy = _entry(source_legacy)
    store = _minimal_store(tmp_path, exact)
    store._entries[legacy.session_key] = legacy

    count = store.suspend_recently_active(
        max_age_seconds=120,
        exclude_session_keys={exact.session_key},
    )

    assert count == 1
    assert exact.resume_pending is False
    assert legacy.resume_reason == "restart_interrupted"


class _DeliveryAdapter:
    def __init__(self):
        self.callback = None
        self.generation = None

    def register_post_delivery_callback(
        self, session_key, callback, *, generation=None
    ):
        self.callback = callback
        self.generation = generation


@pytest.mark.asyncio
async def test_delivery_callback_clears_only_response_ready_run(tmp_path):
    source = _source()
    entry = _entry(source)
    store = _minimal_store(tmp_path, entry)
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._active_run_store = ActiveRunStore(tmp_path)
    runner._clear_restart_failure_count = MagicMock()
    adapter = _DeliveryAdapter()
    runner._adapter_for_source = lambda _source: adapter

    facade = MagicMock(spec=AsyncSessionStore)
    facade._store = store
    facade.clear_resume_pending = AsyncMock(return_value=True)
    runner._async_session_store = facade
    event = MessageEvent(
        text="work",
        message_type=MessageType.TEXT,
        source=source,
        message_id="message-1",
    )

    run_id = await runner._begin_active_run(event=event, session_key=entry.session_key)
    assert runner._active_run_store.get(entry.session_key).phase == PHASE_EXECUTING
    await runner._mark_active_run_response_ready(
        session_key=entry.session_key,
        run_id=run_id,
        source=source,
        run_generation=7,
    )
    assert runner._active_run_store.get(entry.session_key).phase == PHASE_RESPONSE_READY
    assert adapter.generation == 7

    await adapter.callback()

    assert runner._active_run_store.get(entry.session_key) is None
    facade.clear_resume_pending.assert_awaited_once_with(entry.session_key)
    runner._clear_restart_failure_count.assert_called_once_with(entry.session_key)


@pytest.mark.asyncio
async def test_stale_delivery_callback_cannot_clear_newer_run(tmp_path):
    source = _source()
    entry = _entry(source)
    store = _minimal_store(tmp_path, entry)
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._active_run_store = ActiveRunStore(tmp_path)
    runner._clear_restart_failure_count = MagicMock()
    adapter = _DeliveryAdapter()
    runner._adapter_for_source = lambda _source: adapter
    facade = MagicMock(spec=AsyncSessionStore)
    facade._store = store
    facade.clear_resume_pending = AsyncMock(return_value=True)
    runner._async_session_store = facade

    old = runner._active_run_store.begin(
        entry.session_key,
        trigger_message_id="message-1",
    )
    await runner._mark_active_run_response_ready(
        session_key=entry.session_key,
        run_id=old.run_id,
        source=source,
        run_generation=7,
    )
    newer = runner._active_run_store.begin(
        entry.session_key,
        trigger_message_id="message-2",
    )

    await adapter.callback()

    assert runner._active_run_store.get(entry.session_key) == newer
    facade.clear_resume_pending.assert_not_awaited()
    runner._clear_restart_failure_count.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_before_response_ready_leaves_executing_run(tmp_path):
    source = _source()
    entry = _entry(source)
    store = _minimal_store(tmp_path, entry)
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._active_run_store = ActiveRunStore(tmp_path)
    event = MessageEvent(
        text="work",
        message_type=MessageType.TEXT,
        source=source,
        message_id="message-1",
    )

    run_id = await runner._begin_active_run(event=event, session_key=entry.session_key)

    record = runner._active_run_store.get(entry.session_key)
    assert record.run_id == run_id
    assert record.phase == PHASE_EXECUTING


@pytest.mark.asyncio
async def test_recovered_watchers_start_before_session_replay(monkeypatch):
    runner = object.__new__(GatewayRunner)
    order: list[str] = []
    runner._spawn_supervised = lambda factory, name, **kwargs: order.append(name)
    runner._schedule_resume_pending_sessions = lambda: order.append("session-replay")
    monkeypatch.setattr(
        "tools.process_registry.process_registry.pending_watchers",
        [{"session_id": "process-1"}],
    )

    await runner._start_recovery_consumers()

    assert order == [
        "process_watcher:process-1",
        "async_delegation_watcher",
        "session-replay",
    ]


@pytest.mark.asyncio
async def test_internal_event_cannot_resume_safety_paused_session(tmp_path):
    source = _source()
    entry = _entry(source)
    entry.resume_pending = True
    entry.resume_reason = "side_effect_unknown"
    store = _minimal_store(tmp_path, entry)
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._active_recovery_reasons = {entry.session_key: "side_effect_unknown"}

    event = MessageEvent(
        text="background completion",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
    )

    assert await runner._handle_message(event) is None
    assert entry.resume_pending is True


@pytest.mark.asyncio
async def test_trigger_message_redelivery_cannot_resume_safety_pause(tmp_path):
    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="paused-redelivery")
    session_key = build_session_key(source)
    entry = SessionEntry(
        session_key=session_key,
        session_id="sid-paused-redelivery",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="side_effect_unknown",
    )
    runner.session_store._entries = {session_key: entry}
    runner._active_recovery_reasons = {session_key: "side_effect_unknown"}
    runner._active_run_store = ActiveRunStore(tmp_path)
    runner._active_run_store.begin(
        session_key,
        trigger_message_id="original-message",
    )
    runner._handle_message_with_agent = AsyncMock(return_value="must not run")
    event = MessageEvent(
        text="original request",
        message_type=MessageType.TEXT,
        source=source,
        message_id="original-message",
    )

    assert await runner._handle_message(event) is None
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_recovery_concurrency_is_capped_at_four(tmp_path):
    runner, adapter = make_restart_runner()
    runner._active_run_store = ActiveRunStore(tmp_path)
    entries = []
    for index in range(7):
        source = make_restart_source(chat_id=f"recover-{index}")
        session_key = f"agent:main:telegram:dm:recover-{index}"
        entries.append(
            SessionEntry(
                session_key=session_key,
                session_id=f"sid-{index}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                origin=source,
                platform=Platform.TELEGRAM,
                chat_type="dm",
                resume_pending=True,
                resume_reason="restart_interrupted_exact",
                last_resume_marked_at=datetime.now(),
            )
        )
        runner._active_run_store.begin(
            session_key,
            trigger_message_id=f"message-{index}",
        )
    runner.session_store._entries = {entry.session_key: entry for entry in entries}
    runner._recovery_semaphore = asyncio.Semaphore(4)
    gate = asyncio.Event()
    active = 0
    maximum = 0

    async def _blocked_handle(_event):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await gate.wait()
        finally:
            active -= 1

    adapter.handle_message = _blocked_handle
    assert runner._schedule_resume_pending_sessions() == 7
    for _ in range(20):
        if maximum == 4:
            break
        await asyncio.sleep(0.01)

    assert maximum == 4
    gate.set()
    await asyncio.sleep(0.1)
    assert active == 0


def test_exact_resume_without_journal_is_never_scheduled(tmp_path):
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="missing-journal")
    entry = SessionEntry(
        session_key="agent:main:telegram:dm:missing-journal",
        session_id="sid-missing",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted_exact",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {entry.session_key: entry}
    runner._active_run_store = ActiveRunStore(tmp_path)
    adapter.handle_message = AsyncMock()

    assert runner._schedule_resume_pending_sessions() == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_pause_notification_is_sent_once_per_boot_to_original_thread(tmp_path):
    source = _source()
    entry = _entry(source)
    entry.resume_pending = True
    entry.resume_reason = "side_effect_unknown"
    store = _minimal_store(tmp_path, entry)
    active_store = ActiveRunStore(tmp_path)
    active_store.begin(entry.session_key, trigger_message_id="message-1")
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._active_run_store = active_store
    runner._recovery_pause_notified = set()
    runner._background_tasks = set()
    runner._running_agents = {}
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=None)
    runner._adapter_for_source = lambda _source: adapter
    runner._is_user_authorized = lambda _source: True
    runner._thread_metadata_for_source = lambda _source: {
        "thread_id": _source.thread_id
    }

    assert runner._schedule_recovery_pause_notifications() == 1
    assert runner._schedule_recovery_pause_notifications() == 0
    await asyncio.gather(*list(runner._background_tasks))

    adapter.send.assert_awaited_once()
    args, kwargs = adapter.send.await_args
    assert args[0] == source.chat_id
    assert "will not replay" in args[1]
    assert kwargs["metadata"] == {"thread_id": source.thread_id}


@pytest.mark.asyncio
async def test_stale_pause_notification_is_suppressed_after_manual_recovery(tmp_path):
    source = _source()
    entry = _entry(source)
    entry.resume_pending = True
    entry.resume_reason = "side_effect_unknown"
    store = _minimal_store(tmp_path, entry)
    active_store = ActiveRunStore(tmp_path)
    record = active_store.begin(entry.session_key, trigger_message_id="message-1")
    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._active_run_store = active_store
    runner._recovery_pause_notified = set()
    runner._running_agents = {}
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=None)
    runner._adapter_for_source = lambda _source: adapter

    # A new user turn replaces the paused run before the queued notification
    # task gets CPU time.
    active_store.begin(entry.session_key, trigger_message_id="message-2")
    await runner._send_recovery_pause_notification(entry=entry, run_id=record.run_id)

    adapter.send.assert_not_awaited()
