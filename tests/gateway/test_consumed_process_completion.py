"""Consumed producers must not wake a model after completion queue admission."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from tools.process_registry import ProcessRegistry


class Transport(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, **kwargs):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def stop_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


async def eventually(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import tools.process_registry as module
    import gateway.run as gateway_run

    monkeypatch.setattr(module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = ProcessRegistry()
    monkeypatch.setattr(module, "process_registry", registry)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    adapter = Transport()
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    source = adapter.build_source("123", user_id="123")
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    return runner, adapter, registry, source, key


async def producer(registry, key, tmp_path, exit_code):
    session = registry.spawn_local(
        subprocess.list2cmdline([
            sys.executable, "-c",
            f"import sys; print('first-line'); print('final-line'); sys.exit({exit_code})",
        ]),
        cwd=str(tmp_path), session_key=key,
    )
    assert await asyncio.to_thread(session._completion_event.wait, 10)
    assert session.exited and session.exit_code == exit_code
    assert "final-line" in session.output_buffer
    return session


def observe(registry, session, action):
    if action == "wait":
        assert registry.wait(session.id, timeout=1)["exit_code"] == session.exit_code
    elif action == "log":
        assert "final-line" in registry.read_log(session.id)["output"]
    elif action == "head":
        registry.read_log(session.id, offset=0, limit=1)
    elif action == "poll":
        registry.poll(session.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["batch", "adapter", "recursive"])
@pytest.mark.parametrize("action", ["wait", "log", "poll", "head", "unread"])
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_consumption_controls_queued_execution(rig, tmp_path, stage, action, exit_code):
    runner, adapter, registry, source, key = rig
    session = await producer(registry, key, tmp_path, exit_code)
    runner._run_agent = AsyncMock(return_value={"final_response": "completion reply", "messages": []})
    # A real adapter busy guard admits the watcher into its production pending slot.
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = asyncio.current_task()
    watcher = dict(session_id=session.id, session_key=key, platform="telegram", chat_id="123",
                   chat_type="dm", check_interval=0, notify_on_complete=True)
    task = asyncio.create_task(runner._run_process_watcher(watcher))
    if stage == "batch":
        await eventually(lambda: bool(getattr(runner, "_completion_notification_batches", {})))
    else:
        await asyncio.wait_for(task, 10)
        assert key in adapter._pending_messages
    observe(registry, session, action)
    await asyncio.wait_for(task, 10)
    consumed = action in {"wait", "log"}
    try:
        if stage == "recursive":
            result = {"final_response": "foreground done", "messages": []}
            event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
            if event or text:
                ctx = SimpleNamespace(
                    source=source, session_id="foreground", session_key=key, run_generation=1,
                    _interrupt_depth=0, history=[], _status_thread_metadata=None,
                    context_prompt="", stream_consumer_holder=[None], event_message_id=None,
                    inbound_message_id=None,
                )
                await runner._run_agent_queued_followup(ctx, adapter, text, event, result, result, None)
        elif key in adapter._pending_messages:
            event = adapter.get_pending_message(key)
            adapter._spawn_drain_task(event, key)
            await eventually(lambda: key not in adapter._session_tasks)
        assert runner._run_agent.await_count == (0 if consumed else 1)
        if consumed:
            assert not adapter.sent
        else:
            assert session.id in runner._run_agent.call_args.kwargs["message"]
    finally:
        adapter._session_tasks.pop(key, None)
        adapter._active_sessions.pop(key, None)
        await runner._cancel_process_completion_batch_tasks()


def followup_context(source, key):
    return SimpleNamespace(
        source=source, session_id="foreground", session_key=key, run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None,
        context_prompt="", stream_consumer_holder=[None], event_message_id=None,
        inbound_message_id=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["batch", "adapter", "recursive", "followup"])
@pytest.mark.parametrize("case", [
    "primary", "last", "single", "all", "missing_epoch", "old_epoch", "unknown_id",
    "other_session", "user", "debounced_user", "watch", "watch_media", "async_delegation", "refused",
    "malformed_metadata",
])
async def test_filter_preserves_unconsumed_siblings_and_unrelated_work(rig, tmp_path, stage, case):
    from tools.process_registry_notifications import format_process_notification

    runner, adapter, registry, source, key = rig
    adapter.config.extra["allow_from"] = ["123"]
    runner._busy_input_mode = runner._busy_text_mode = "queue"
    runner._run_agent = AsyncMock(return_value={"final_response": "completion reply", "messages": []})
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = asyncio.current_task()
    sessions = [await producer(registry, key, tmp_path, code) for code in (0, 7, 0)]
    route = dict(session_key=key, platform="telegram", chat_id="123", chat_type="dm", user_id="123")
    events = [runner._build_process_completion_event(route, session, session.id) for session in sessions]
    removed = {"primary": [0], "last": [2], "single": [0, 1], "all": [0, 1, 2]}.get(case, [0])
    if case in {"user", "debounced_user", "watch", "watch_media", "async_delegation", "malformed_metadata"}:
        removed = [0, 1, 2]
    if case == "missing_epoch":
        events[0].pop("started_at")
    elif case == "old_epoch":
        events[0]["started_at"] -= 1
    elif case == "unknown_id":
        events[0]["session_id"] = "proc_unknown"
    elif case == "other_session":
        sessions[0].session_key = key + ":different-owner"

    extra_text = "independent follow-up"
    if case == "debounced_user":
        adapter._busy_text_mode = "queue"
        adapter._busy_text_debounce_seconds = adapter._busy_text_hard_cap_seconds = 60
        await adapter.handle_message(MessageEvent(text=extra_text, source=source))
        assert key in adapter._text_debounce_store()

    # Real admission refusal, with no handler installed, must release the batch for retry.
    if case == "refused":
        adapter.set_message_handler(None)
    tasks = [asyncio.create_task(runner._enqueue_process_completion_notification(
        format_process_notification(evt), evt,
    )) for evt in events]
    await eventually(lambda: sum(map(len, runner._completion_notification_batches.values())) == len(events))
    if stage == "batch":
        for index in removed:
            observe(registry, sessions[index], "wait")
    receipts = await asyncio.wait_for(asyncio.gather(*tasks), 10)
    if case == "refused":
        assert receipts == ([None, False, False] if stage == "batch" else [False] * 3)
        adapter.set_message_handler(runner._handle_message)
        receipts = await asyncio.wait_for(asyncio.gather(*[
            runner._enqueue_process_completion_notification(format_process_notification(evt), evt)
            for evt in events
        ]), 10)
    assert receipts == ([None if i in removed else True for i in range(3)]
                        if stage == "batch" and case not in {
                            "missing_epoch", "old_epoch", "unknown_id", "other_session",
                        } else [True] * 3)

    # A later user/watch/delegation event must remain a separate queued unit.
    extra = MessageEvent(text=extra_text, source=source, internal=case != "user")
    if case == "user":
        extra.metadata["process_completion_entries"] = [(format_process_notification(events[0]), events[0])]
    elif case == "malformed_metadata":
        extra.metadata["process_completion_entries"] = ["invalid"]
    elif case == "watch_media":
        # Photo type alone activates the album merge path; no vision I/O is needed.
        extra.message_type = MessageType.PHOTO
    if case == "debounced_user":
        await adapter._flush_text_debounce_now(key)
    elif case == "watch":
        await runner._inject_watch_notification(extra_text, dict(events[0], type="watch_match"))
    elif case == "async_delegation":
        await runner._inject_watch_notification(extra_text, dict(events[0], type="async_delegation"))
    else:
        await adapter.handle_message(extra)
        assert extra._gateway_accepted

    pending_event = pending_text = None
    result = {"final_response": "foreground done", "messages": []}
    if stage == "followup":
        pending_event, pending_text = await runner._run_agent_drain_pending(result, adapter, source, key)
    if stage != "batch":
        for index in removed:
            observe(registry, sessions[index], "log")
    try:
        if pending_event:
            await runner._run_agent_queued_followup(
                followup_context(source, key), adapter, pending_text, pending_event, result, result, None,
            )
        # Each drain uses the actual slot + overflow promotion; only the model
        # boundary is replaced, so emulate its return to the production drain.
        while key in adapter._pending_messages or runner._overflow_queue(key):
            if stage in {"batch", "adapter"}:
                event = adapter.get_pending_message(key)
                if event is None:
                    event = runner._promote_queued_event(key, adapter, None)
                adapter._spawn_drain_task(event, key)
                await eventually(lambda: key not in adapter._session_tasks)
            else:
                event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
                if event or text:
                    await runner._run_agent_queued_followup(
                        followup_context(source, key), adapter, text, event, result, result, None,
                    )
        messages = [call.kwargs["message"] for call in runner._run_agent.call_args_list]
        assert sum(extra_text in message for message in messages) == 1
        unverifiable = case in {"missing_epoch", "old_epoch", "unknown_id", "other_session"}
        retained = [i for i in range(3) if i not in removed or unverifiable]
        completion_messages = [message for message in messages if any(evt["session_id"] in message for evt in events)]
        assert len(completion_messages) == bool(retained)
        if retained:
            message = completion_messages[0]
            for index, evt in enumerate(events):
                assert (evt["session_id"] in message) == (index in retained)
            if len(retained) > 1:
                assert f"{len(retained)} background processes completed" in message
            if 1 in retained:
                assert "exit_code=7" in message
    finally:
        adapter._session_tasks.pop(key, None)
        adapter._active_sessions.pop(key, None)
        await runner._cancel_process_completion_batch_tasks()
