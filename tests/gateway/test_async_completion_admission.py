"""Durable completion ownership through real gateway admission and cold turns.

Only model execution and transport I/O are replaced. The adapter queue, runner
callbacks, session store, SQLite ledger and restart recovery are production code.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from tools import async_delegation as delegation


class LocalAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
        self.sent = []
        self.fail_send = False

    @property
    def name(self):
        return "telegram"

    async def connect(self, *, is_reconnect=False):
        self._mark_connected()
        return True

    async def disconnect(self):
        self._running = False

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, dict(metadata or {})))
        return SendResult(success=not self.fail_send, message_id="local-reply",
                          error="offline" if self.fail_send else None,
                          retryable=False)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


class LocalModel:
    messages = []
    releases = {}
    failure = False
    instances = []
    result_flags = {}
    async_releases = []

    def __init__(self, **kwargs):
        self.tools = []
        self.model, self.provider = "fixture-model", "openai"
        self._interrupt_requested = False
        self._interrupt_message = None
        self.session_id = kwargs.get("session_id")
        type(self).instances.append(self)

    @property
    def is_interrupted(self):
        return self._interrupt_requested

    def interrupt(self, message=None, **kwargs):
        self._interrupt_requested = True
        self._interrupt_message = message
        return True

    def run_conversation(self, message, conversation_history=None, **kwargs):
        type(self).messages.append((message, str(get_hermes_home()), self.session_id))
        for token, release in type(self).releases.items():
            if token in message:
                assert release.wait(20), f"test did not release {token}"
        if type(self).failure and "result-" in message:
            raise RuntimeError("model failed after starting")
        return {"final_response": "local answer", "messages": [], "api_calls": 1,
                "completed": True, "interrupted": self._interrupt_requested,
                "interrupt_message": self._interrupt_message,
                **(type(self).result_flags if "result-" in message else {})}


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), 12)


async def drain(adapter):
    async def wait():
        while adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
    await asyncio.wait_for(wait(), 20)


def ledger(home, name):
    with sqlite3.connect(home / "state.db") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM async_delegations WHERE delegation_id=?", (name,)).fetchone()
        return dict(row) if row else None


def claim_all(events, consumer) -> tuple[tuple[str, str], ...]:
    receipts = []
    for event in events:
        claim = delegation.claim_event_delivery(event, consumer)
        assert isinstance(claim, str) and claim
        receipts.append((str(event["delegation_id"]), claim))
    return tuple(receipts)


def seed(runner, source, home, name, *, parent=None):
    with _profile_runtime_scope(home):
        entry = runner.session_store.get_or_create_session(source)
        runner.session_store._db.create_session(parent or entry.session_id, "telegram")
        evt = {"type": "async_delegation", "session_key": runner._session_key_for_source(source),
               "delegation_id": name, "summary": "result-" + name,
               "status": "completed", "goal": "inspect " + name,
               "parent_session_id": parent or entry.session_id,
               "platform": "telegram", "chat_type": source.chat_type,
               "chat_id": source.chat_id, "thread_id": source.thread_id,
               "origin_profile": source.profile or "default",
               "origin_hermes_home": str(home), "user_id": source.user_id,
               "dispatched_at": time.time(), "completed_at": time.time()}
        delegation._persist_dispatch(evt)
        delegation._persist_completion(evt, {"status": "completed", "summary": evt["summary"]})
    return evt


@asynccontextmanager
async def gateway(monkeypatch, root, profile):
    default_home = root / ".hermes"
    home = default_home if profile == "default" else default_home / "profiles" / profile
    for directory in {default_home, home}:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(
            "model:\n  default: fixture-model\n  provider: openai\n"
            "display:\n  tool_progress: off\n  busy_ack_enabled: false\n"
            "  busy_input_mode: interrupt\n  busy_text_mode: queue\n"
            "session_reset:\n  mode: none\n")
        (directory / ".env").write_text("TELEGRAM_ALLOWED_USERS=123\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    monkeypatch.setenv("HERMES_GATEWAY_NOTIFY_INTERVAL", "0")
    monkeypatch.setattr(Path, "home", lambda: root)
    monkeypatch.setattr("gateway.run._hermes_home", default_home)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda **kw: {"api_key": "fixture"})
    monkeypatch.setattr("run_agent.AIAgent", LocalModel)
    monkeypatch.setattr("tools.tirith_security._install_tirith", lambda **kw: (None, "offline test"))
    from tools.process_registry import process_registry
    monkeypatch.setattr(process_registry, "completion_queue", queue.Queue())
    LocalModel.messages, LocalModel.releases, LocalModel.instances = [], {}, []
    LocalModel.failure = False
    LocalModel.result_flags = {}
    LocalModel.async_releases = []
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True, sessions_dir=default_home / "sessions"))
    monkeypatch.setattr(runner, "_instantiate_adapter", lambda *_args: LocalAdapter())
    adapter = runner._create_adapter(Platform.TELEGRAM, PlatformConfig(enabled=True))
    assert isinstance(adapter, LocalAdapter)
    if profile == "default":
        runner.adapters[Platform.TELEGRAM] = adapter
        runner._wire_adapter_handlers(adapter)
    else:
        runner._profile_adapters[profile] = {Platform.TELEGRAM: adapter}
        runner._configure_profile_adapter(adapter, profile, Platform.TELEGRAM)
    await adapter.connect()
    source = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="123",
                           user_id="123", thread_id="17", profile=None if profile == "default" else profile)
    try:
        yield runner, adapter, source, home
    finally:
        for release in LocalModel.releases.values():
            release.set()
        for release in LocalModel.async_releases:
            release.set()
        await drain(adapter)
        await runner._cancel_process_completion_batch_tasks()
        runner.close_all_session_db_handles()
        runner.session_store.close_all_db_handles()
        shutil.rmtree(default_home)


async def retry_after_adapter_cleanup(case, profile, runner, adapter, source, home, events, monkeypatch):
    key = runner._session_key_for_source(source)
    human_done = threading.Event()
    LocalModel.releases["human-active"] = human_done
    entered, finish = asyncio.Event(), asyncio.Event()
    LocalModel.async_releases.append(finish)
    processing_start, handle = adapter.on_processing_start, adapter.handle_message
    if case == "shutdown_start":
        async def blocked_start(event):
            if event.internal:
                entered.set()
                await finish.wait()
            await processing_start(event)
        monkeypatch.setattr(adapter, "on_processing_start", blocked_start)
    await adapter.handle_message(MessageEvent(text="human-active", source=source, message_id="human-1"))
    await until(lambda: bool(LocalModel.messages))
    if case == "late_admission_return":
        async def delayed_return(event):
            await handle(event)
            if event.internal:
                entered.set()
                await finish.wait()
        monkeypatch.setattr(adapter, "handle_message", delayed_return)
        admitted = asyncio.create_task(runner._deliver_async_delegation_group(events))
        await asyncio.wait_for(entered.wait(), 10)
    else:
        assert await runner._deliver_async_delegation_group(events) is True
    if case == "replace_queued":
        extra = seed(runner, source, home, "unit-overflow")
        assert await runner._deliver_async_delegation_group([extra]) is True
        events.append(extra)
        assert runner._queue_depth(key, adapter=adapter) == 2
    if case == "shutdown_start":
        human_done.set()
        await asyncio.wait_for(entered.wait(), 10)
    assert all(ledger(home, evt["delegation_id"])["delivery_state"] == "queued" for evt in events)
    cleanup = asyncio.create_task(adapter.cancel_background_tasks())
    await asyncio.sleep(0)
    human_done.set()
    await cleanup
    assert all((ledger(home, evt["delegation_id"])["delivery_state"],
                ledger(home, evt["delegation_id"])["delivery_attempts"]) == ("pending", 0) for evt in events)
    assert not any("result-" in row[0] for row in LocalModel.messages)
    assert not adapter._pending_messages and not runner._overflow_queue(key)
    if case == "late_admission_return":
        finish.set()
        assert await admitted is True
        monkeypatch.setattr(adapter, "handle_message", handle)
        assert all(ledger(home, evt["delegation_id"])["delivery_state"] == "pending" for evt in events)
    if case == "shutdown_start":
        monkeypatch.setattr(adapter, "on_processing_start", processing_start)
    target = adapter
    if case == "replace_queued":
        target = runner._create_adapter(Platform.TELEGRAM, PlatformConfig(enabled=True))
        assert isinstance(target, LocalAdapter)
        if profile == "default":
            runner.adapters[Platform.TELEGRAM] = target
            runner._wire_adapter_handlers(target)
        else:
            runner._profile_adapters[profile][Platform.TELEGRAM] = target
            runner._configure_profile_adapter(target, profile, Platform.TELEGRAM)
        await target.connect()
    assert await runner._deliver_async_delegation_group(events) is True
    await drain(target)
    processed = [row for row in LocalModel.messages if "result-" in row[0]]
    assert len(processed) == 1
    assert all(processed[0][0].count(evt["summary"]) == 1 for evt in events)
    assert processed[0][1:] == (str(home), events[0]["parent_session_id"])
    assert all(ledger(home, evt["delegation_id"])["delivery_state"] == "delivered" for evt in events)
    assert await runner._deliver_async_delegation_group(events) is None
    assert all(chat == source.chat_id and metadata.get("thread_id") == source.thread_id
               for chat, _text, metadata in target.sent)


async def cancel_before_first_step(case, runner, adapter, source, home, events, monkeypatch):
    key = runner._session_key_for_source(source)
    entered = []
    processing_start = adapter.on_processing_start
    async def observe_start(event):
        entered.append(event)
        await processing_start(event)
    monkeypatch.setattr(adapter, "on_processing_start", observe_start)
    if "drain" in case:
        # A finished owner's retained guard is the normal drain handoff state.
        # Use real busy admission to obtain the event before creating its drain task.
        adapter._active_sessions[key] = asyncio.Event()
    assert await runner._deliver_async_delegation_group(events) is True
    if "drain" in case:
        event = adapter._pending_messages.pop(key)
        adapter._spawn_drain_task(event, key)
    # No yield separates task creation from cancellation. A coroutine canceled in
    # this window never enters its body or any of its finally blocks.
    assert not entered and not LocalModel.messages
    task = adapter._session_tasks[key]
    assert not task.done()
    if case.endswith("explicit"):
        await adapter.cancel_session_processing(key)
        expected = "dropped"
    else:
        await adapter.cancel_background_tasks()
        expected = "pending"
    assert task.cancelled() and not entered and not LocalModel.messages
    row = ledger(home, events[0]["delegation_id"])
    assert row["delivery_state"] == expected
    if expected == "dropped":
        assert await runner._deliver_async_delegation_group(events) is None
        return
    assert row["delivery_attempts"] == 0
    assert await runner._deliver_async_delegation_group(events) is True
    await drain(adapter)
    assert len(LocalModel.messages) == 1
    assert LocalModel.messages[0][0].count(events[0]["summary"]) == 1
    assert LocalModel.messages[0][1:] == (str(home), events[0]["parent_session_id"])
    assert ledger(home, events[0]["delegation_id"])["delivery_state"] == "delivered"


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("case", [
    "busy", "idle", "batch", "fifo", "send_failure", "new", "reset", "compression",
    "prestart_error", "refused", "runtime_refused", "storage_error", "model_error", "canceled", "foreign_profile",
    "orphan_fifo", "partial_result", "error_result", "incomplete_result", "hook_reset", "start_canceled",
    "shutdown_start", "replace_queued", "late_admission_return",
    "prestep_cold_shutdown", "prestep_cold_explicit", "prestep_drain_shutdown", "prestep_drain_explicit",
])
async def test_completion_receipt_follows_processing_not_adapter_acceptance(case, profile, tmp_path, monkeypatch):
    async with gateway(monkeypatch, tmp_path, profile) as (runner, adapter, source, home):
        events = [seed(runner, source, home, f"unit-{i}")
                  for i in range(2 if case in {"batch", "storage_error", "model_error", "new", "reset"} else 1)]
        if case.startswith("prestep_"):
            await cancel_before_first_step(case, runner, adapter, source, home, events, monkeypatch)
            return
        if case in {"shutdown_start", "replace_queued", "late_admission_return"}:
            await retry_after_adapter_cleanup(case, profile, runner, adapter, source, home, events, monkeypatch)
            return
        key = runner._session_key_for_source(source)
        release = threading.Event()
        LocalModel.releases["human-active"] = release
        hook_started, hook_release = asyncio.Event(), asyncio.Event()
        LocalModel.async_releases.append(hook_release)
        if case == "hook_reset":
            emit = runner.hooks.emit
            async def blocked_start(name, context):
                await emit(name, context)
                if name == "agent:start" and "unit-0" in context.get("message", ""):
                    hook_started.set()
                    await hook_release.wait()
            monkeypatch.setattr(runner.hooks, "emit", blocked_start)
        if case == "start_canceled":
            processing_start = adapter.on_processing_start
            async def blocked_processing_start(event):
                if event.internal:
                    hook_started.set()
                    await hook_release.wait()
                await processing_start(event)
            monkeypatch.setattr(adapter, "on_processing_start", blocked_processing_start)
        if case == "refused":
            handler = adapter._message_handler
            adapter._message_handler = None
            for _ in range(10):
                assert await runner._deliver_async_delegation_group(events) is False
            adapter._message_handler = handler
            assert ledger(home, events[0]["delegation_id"])["delivery_attempts"] == 0
        if case == "runtime_refused":
            claim_slot = runner._claim_active_session_slot
            monkeypatch.setattr(runner, "_claim_active_session_slot", lambda *_args: (None, "at capacity"))
            for _ in range(3):
                assert await runner._deliver_async_delegation_group(events) is True
                await drain(adapter)
                row = ledger(home, events[0]["delegation_id"])
                assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
                assert not LocalModel.messages
            monkeypatch.setattr(runner, "_claim_active_session_slot", claim_slot)
        if case == "storage_error":
            with sqlite3.connect(home / "state.db") as conn:
                conn.execute("""CREATE TRIGGER fail_admission BEFORE UPDATE ON async_delegations
                                WHEN NEW.delivery_state='queued' AND NEW.delegation_id='unit-1'
                                BEGIN SELECT RAISE(ABORT, 'storage write failed'); END""")
            assert await runner._deliver_async_delegation_group(events) is False
            assert not adapter._pending_messages and not adapter._background_tasks
            assert not runner._completion_deliveries_delivered
            assert all((ledger(home, evt["delegation_id"])["delivery_state"],
                        ledger(home, evt["delegation_id"])["delivery_attempts"]) == ("pending", 0) for evt in events)
            with sqlite3.connect(home / "state.db") as conn:
                conn.execute("DROP TRIGGER fail_admission")
        foreign_home = tmp_path / "foreign"
        if case == "foreign_profile":
            foreign_home.mkdir()
            seed(runner, source, foreign_home, events[0]["delegation_id"])
        if case != "idle":
            await adapter.handle_message(MessageEvent(text="human-active", source=source, message_id="human-1"))
            await until(lambda: bool(LocalModel.messages))
        assert await runner._deliver_async_delegation_group(events) is True
        if case != "idle":
            for evt in events:
                assert ledger(home, evt["delegation_id"])["delivery_state"] == "queued"
            assert not adapter._active_sessions[key].is_set()
            assert not any(model.is_interrupted for model in LocalModel.instances)
        if case == "fifo":
            extra = seed(runner, source, home, "unit-later")
            assert await runner._deliver_async_delegation_group([extra]) is True
            events.append(extra)
        assert await runner._deliver_async_delegation_group(events) is None
        adapter.fail_send = case == "send_failure"
        if case in {"new", "reset", "compression"}:
            with _profile_runtime_scope(home):
                db = runner.session_store._db
                parent = events[0]["parent_session_id"]
                db.end_session(parent, end_reason={"new": "new_session", "reset": "session_reset",
                                                  "compression": "compression"}[case])
                successor = "replacement-session"
                db.create_session(successor, "telegram", parent_session_id=parent if case == "compression" else None)
                if case == "compression":
                    # The compressor rotates its executing agent together with the DB
                    # lineage; a transport-side route change alone models /resume.
                    LocalModel.instances[-1].session_id = successor
                    runner.session_store.advance_compression_session(key, parent, successor)
                else:
                    runner.session_store.switch_session(key, successor)
        resolve = runner._hmwa_resolve_session
        if case == "prestart_error":
            async def temporary_failure(event, inbound_source):
                if event.internal:
                    raise sqlite3.OperationalError("temporary session lookup failure")
                return await resolve(event, inbound_source)
            monkeypatch.setattr(runner, "_hmwa_resolve_session", temporary_failure)
        LocalModel.failure = case == "model_error"
        LocalModel.result_flags = {
            "partial_result": {"partial": True}, "error_result": {"error": "provider error"},
            "incomplete_result": {"completed": False},
        }.get(case, {})
        completion_release = threading.Event()
        if case == "canceled":
            LocalModel.releases["result-"] = completion_release
        if case == "orphan_fifo":
            # A previous handler may leave only an overflow tail. Preserve the
            # actually admitted receipt while arranging that existing rescue precondition.
            orphan = adapter._pending_messages.pop(key)
            previous = adapter._session_tasks[key]
            previous.cancel()
            release.set()
            await asyncio.gather(previous, return_exceptions=True)
            runner._session_state(key).conversation.queued_events.append(orphan)
            await adapter.handle_message(MessageEvent(text="human-rescue", source=source, message_id="human-2"))
        else:
            release.set()
        if case == "hook_reset":
            await asyncio.wait_for(hook_started.wait(), 10)
            with _profile_runtime_scope(home):
                db = runner.session_store._db
                db.end_session(events[0]["parent_session_id"], end_reason="session_reset")
                db.create_session("replacement-session", "telegram")
                runner.session_store.switch_session(key, "replacement-session")
            hook_release.set()
        if case == "start_canceled":
            await asyncio.wait_for(hook_started.wait(), 10)
            task = adapter._session_tasks[key]
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert ledger(home, events[0]["delegation_id"])["delivery_state"] == "pending"
            assert not any("result-" in row[0] for row in LocalModel.messages)
            monkeypatch.setattr(adapter, "on_processing_start", processing_start)
            assert await runner._deliver_async_delegation_group(events) is True
        if case == "canceled":
            await until(lambda: any("result-" in row[0] for row in LocalModel.messages))
            assert ledger(home, events[0]["delegation_id"])["delivery_state"] == "processing"
            task = adapter._session_tasks[key]
            task.cancel()
            completion_release.set()
            await asyncio.gather(task, return_exceptions=True)
        await drain(adapter)
        processed = [record for record in LocalModel.messages if "result-" in record[0]]
        if case in {"new", "reset", "hook_reset", "prestart_error"}:
            assert not processed
            expected = "pending" if case == "prestart_error" else "dropped"
            assert all(ledger(home, evt["delegation_id"])["delivery_state"] == expected for evt in events)
            if case != "prestart_error":
                return
            monkeypatch.setattr(runner, "_hmwa_resolve_session", resolve)
            from tools.process_registry import process_registry
            assert not process_registry.completion_queue.empty()
            assert await runner._deliver_async_delegation_group(events) is True
            await drain(adapter)
            processed = [record for record in LocalModel.messages if "result-" in record[0]]
        assert len(processed) == (2 if case == "fifo" else 1)
        text = "\n".join(record[0] for record in processed)
        assert all(text.count(evt["summary"]) == 1 for evt in events)
        assert all(record[1] == str(home) for record in processed)
        expected_parent = "replacement-session" if case == "compression" else events[0]["parent_session_id"]
        assert all(record[2] == expected_parent for record in processed), processed
        outcome = "unknown" if case in {"model_error", "canceled", "partial_result", "error_result", "incomplete_result"} else "delivered"
        assert all(ledger(home, evt["delegation_id"])["delivery_state"] == outcome for evt in events)
        before = len(LocalModel.messages)
        assert await runner._deliver_async_delegation_group(events) is None
        await drain(adapter)
        assert len(LocalModel.messages) == before
        if case == "foreign_profile":
            assert ledger(foreign_home, events[0]["delegation_id"])["delivery_state"] == "pending"
        if case == "orphan_fifo":
            assert LocalModel.messages[-1][0] == "human-rescue"
        assert all(chat == source.chat_id and metadata.get("thread_id") == source.thread_id
                   for chat, _text, metadata in adapter.sent)


async def process_phase(root, profile, phase, processing):
    """Executed in genuine short-lived interpreters; os._exit bypasses task finalizers."""
    with pytest.MonkeyPatch.context() as patch:
        async with gateway(patch, root, profile) as (runner, adapter, source, home):
            if phase == "admit":
                events = [seed(runner, source, home, f"restart-{i}") for i in range(2)]
                release = threading.Event()
                LocalModel.releases["result-" if processing else "human-active"] = release
                if not processing:
                    await adapter.handle_message(MessageEvent(text="human-active", source=source))
                    await until(lambda: bool(LocalModel.messages))
                assert await runner._deliver_async_delegation_group(events) is True
                if processing:
                    await until(lambda: any("result-" in row[0] for row in LocalModel.messages))
                result = {"rows": [ledger(home, evt["delegation_id"]) for evt in events],
                          "events": events, "model": LocalModel.messages}
                (root / "admit.json").write_text(json.dumps(result))
                os._exit(0)
            from tools.process_registry import process_registry
            with _profile_runtime_scope(home):
                restored = delegation.restore_undelivered_completions(process_registry.completion_queue)
            runner._running = True
            watcher = asyncio.create_task(runner._async_delegation_watcher(interval=0.01))
            if restored:
                await until(lambda: all(ledger(home, f"restart-{i}")["delivery_state"] == "delivered"
                                        for i in range(2)))
            runner._running = False
            await watcher
            await drain(adapter)
            (root / "restore.json").write_text(json.dumps({
                "restored": restored, "model": LocalModel.messages, "sent": adapter.sent,
                "rows": [ledger(home, f"restart-{i}") for i in range(2)],
            }))


async def child(root, profile, phase, processing):
    code = (
        "import asyncio, runpy, sys; from pathlib import Path; "
        "ns=runpy.run_path(sys.argv[1]); "
        "asyncio.run(ns['process_phase'](Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]=='1'))"
    )
    env = dict(os.environ, HOME=str(root / "platform-home"), HERMES_HOME=str(root / ".hermes"))
    result = await asyncio.to_thread(
        subprocess.run, [sys.executable, "-c", code, str(Path(__file__).resolve()), str(root),
                         profile, phase, "1" if processing else "0"],
        env=env, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return json.loads((root / f"{phase}.json").read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("case", [
    "restart_queued", "restart_processing", "live_owner", "stale_receipt", "batch_atomic", "refund",
    "legacy_schema", "retry_budget",
])
async def test_completion_ownership_survives_process_boundaries(case, profile, tmp_path, monkeypatch):
    if case.startswith("restart_"):
        processing = case == "restart_processing"
        admitted = await child(tmp_path, profile, "admit", processing)
        recovered = await child(tmp_path, profile, "restore", processing)
        assert {row["delivery_state"] for row in admitted["rows"]} == {"processing" if processing else "queued"}, recovered
        assert {row["delivery_state"] for row in recovered["rows"]} == {"unknown" if processing else "delivered"}
        processed = [row for row in recovered["model"] if "result-" in row[0]]
        assert len(processed) == (0 if processing else 1)
        assert recovered["restored"] == (0 if processing else 2)
        if processed:
            assert all(processed[0][0].count(evt["summary"]) == 1 for evt in admitted["events"])
            assert processed[0][1] == admitted["events"][0]["origin_hermes_home"]
            assert processed[0][2] == admitted["events"][0]["parent_session_id"]
            assert recovered["sent"] and all(chat == "123" and metadata.get("thread_id") == "17"
                                              for chat, _text, metadata in recovered["sent"])
        return

    from tools.async_delegation_admission import (
        begin_queued_completion_deliveries, queue_completion_deliveries,
        recover_queued_completion_deliveries, settle_queued_completion_deliveries,
    )
    async with gateway(monkeypatch, tmp_path, profile) as (runner, _adapter, source, home):
        events = [seed(runner, source, home, f"ownership-{i}") for i in range(2)]
        with _profile_runtime_scope(home):
            if case == "legacy_schema":
                # A real pre-admission database has no owner columns. Reconciliation must
                # add them without replaying its already delivered rows.
                assert delegation.mark_completion_delivered(events[0]["delegation_id"])
                prior = ledger(home, events[0]["delegation_id"])
                with sqlite3.connect(home / "state.db") as conn:
                    conn.execute("ALTER TABLE async_delegations DROP COLUMN delivery_owner_pid")
                    conn.execute("ALTER TABLE async_delegations DROP COLUMN delivery_owner_started_at")
                queued = queue.Queue()
                delegation.restore_undelivered_completions(queued)
                after = ledger(home, events[0]["delegation_id"])
                assert (after["delivery_state"], after["delivered_at"], after["result_json"]) == (
                    "delivered", prior["delivered_at"], prior["result_json"])
                assert [queued.get_nowait()["delegation_id"] for _ in range(queued.qsize())] == [events[1]["delegation_id"]]
                return
            receipts = claim_all(events, "test")
            if case == "retry_budget":
                for attempt in range(delegation._MAX_DELIVERY_ATTEMPTS):
                    queue_completion_deliveries(receipts)
                    assert settle_queued_completion_deliveries(receipts, "retry") == len(receipts)
                    expected = "dropped" if attempt + 1 == delegation._MAX_DELIVERY_ATTEMPTS else "pending"
                    assert all((ledger(home, name)["delivery_state"], ledger(home, name)["delivery_attempts"])
                               == (expected, attempt + 1) for name, _token in receipts)
                    if expected == "pending":
                        receipts = claim_all(events, "retry")
                assert all(delegation.claim_event_delivery(evt, "too-late") is None for evt in events)
                return
            if case == "batch_atomic":
                stale = (receipts[0], (receipts[1][0], "stale"))
                with pytest.raises(RuntimeError):
                    queue_completion_deliveries(stale)
                assert all(ledger(home, name)["delivery_state"] == "pending" for name, _token in receipts)
            assert queue_completion_deliveries(receipts) == receipts
            if case == "live_owner":
                with sqlite3.connect(home / "state.db") as conn:
                    conn.execute("UPDATE async_delegations SET delivery_claimed_at=?", (time.time() - 3600,))
                assert not delegation.claim_completion_delivery(receipts[0][0], "competing-consumer")
                competing = await asyncio.to_thread(
                    subprocess.run, [sys.executable, "-c",
                        "import json,sys; from tools.async_delegation import claim_completion_delivery; "
                        "from tools.async_delegation_admission import recover_queued_completion_deliveries; "
                        "print(json.dumps([claim_completion_delivery(sys.argv[1],'other-process'), "
                        "recover_queued_completion_deliveries()]))", receipts[0][0]],
                    env=dict(os.environ, HERMES_HOME=str(home)), capture_output=True, text=True, timeout=15,
                )
                assert competing.returncode == 0, competing.stderr
                assert json.loads(competing.stdout) == [False, 0]
                assert recover_queued_completion_deliveries() == 0
                assert all(ledger(home, name)["delivery_state"] == "queued" for name, _token in receipts)
            if case in {"stale_receipt", "batch_atomic"}:
                stale = (receipts[0], (receipts[1][0], "stale"))
                assert not begin_queued_completion_deliveries(stale)
                assert all(ledger(home, name)["delivery_state"] == "queued" for name, _token in receipts)
                assert settle_queued_completion_deliveries([(receipts[0][0], "stale")], "pending") == 0
            if case == "refund":
                assert settle_queued_completion_deliveries(receipts, "pending") == 2
                assert all((ledger(home, name)["delivery_state"], ledger(home, name)["delivery_attempts"])
                           == ("pending", 0) for name, _token in receipts)
                fresh = claim_all(events, "next")
                assert fresh != receipts
                queue_completion_deliveries(fresh)
                assert not begin_queued_completion_deliveries(receipts)
                receipts = fresh
            assert begin_queued_completion_deliveries(receipts)
            assert not begin_queued_completion_deliveries(receipts)
            assert recover_queued_completion_deliveries() == 0
            assert settle_queued_completion_deliveries(receipts, "delivered") == 2
            assert settle_queued_completion_deliveries(receipts, "delivered") == 0
            assert all(ledger(home, name)["delivery_state"] == "delivered" for name, _token in receipts)
