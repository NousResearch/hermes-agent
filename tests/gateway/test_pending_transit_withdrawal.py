"""Deletion remains effective while queued voice preparation awaits STT."""

import asyncio
import threading
from dataclasses import replace

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base_pending_merge import merge_pending_message_event
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from tests.gateway.test_telegram_voice_v0_regressions import _PendingVoiceAdapter, _source


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("pause", ["transcription", "echo"])
async def test_deleted_voice_is_not_dispatched_after_pending_drain_stt(monkeypatch, overlap, partial, pause):
    adapter = _PendingVoiceAdapter()
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[adapter.platform] = adapter
    adapter.gateway_runner = runner
    runner._draining = False
    monkeypatch.setattr(runner, "_should_echo_stt_transcripts", lambda: True)
    adapter.set_queued_withdrawal_handler(runner._withdraw_queued_followups)
    source = replace(_source(), user_id="alice")
    event = MessageEvent(text="", message_type=MessageType.VOICE, source=source,
                         message_id="voice", media_urls=["/cache/voice.ogg"], media_types=["audio/ogg"])
    key = adapter._event_session_key(event)
    runner._enqueue_fifo(key, event, adapter)
    retained = MessageEvent(text="", message_type=MessageType.VOICE, source=source,
                            message_id="remaining", media_urls=["/cache/remaining.ogg"], media_types=["audio/ogg"])
    if partial:
        merge_pending_message_event(adapter._pending_messages, key, retained)
    busy_entered = asyncio.Event()
    busy_resume = asyncio.Event()
    busy_event = MessageEvent(text="later deleted input", source=source, message_id="later")
    busy_task = None

    async def busy_handler(event, session_key):
        busy_entered.set()
        await busy_resume.wait()
        return False

    adapter.set_busy_session_handler(busy_handler)
    adapter._active_sessions[key] = asyncio.Event()
    entered = asyncio.Event()
    echo_resume = asyncio.Event()
    original_send = adapter.send

    async def send(chat_id, content, reply_to=None, metadata=None):
        if pause == "echo" and "withdrawn words" in content:
            entered.set()
            await echo_resume.wait()
        return await original_send(chat_id, content, reply_to=reply_to, metadata=metadata)

    monkeypatch.setattr(adapter, "send", send)
    resume = threading.Event()
    loop = asyncio.get_running_loop()

    def transcribe(path, *args):
        if pause == "transcription" and path.endswith("/voice.ogg"):
            loop.call_soon_threadsafe(entered.set)
            assert resume.wait(timeout=5), "STT test was not released"
        return {"success": True, "transcript": "remaining words" if path.endswith("/remaining.ogg") else "withdrawn words"}

    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    drain = asyncio.create_task(runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, source, key,
    ))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        if overlap:
            busy_task = asyncio.create_task(adapter._handle_message_while_active(busy_event, key))
            await asyncio.wait_for(busy_entered.wait(), timeout=5)
        busy_found = (adapter.withdraw_pending_message("later", chat_id=source.chat_id, sender_id="alice")
                      if overlap else None)
        found = adapter.withdraw_pending_message("voice", chat_id=source.chat_id, sender_id="alice")
    finally:
        resume.set()
        busy_resume.set()
        echo_resume.set()
        if busy_task is not None:
            await busy_task
    dispatched = await drain

    echoes = ([(source.chat_id, '🎙️ "withdrawn words"', None)] if pause == "echo" else [])
    if partial:
        echoes.append((source.chat_id, '🎙️ "remaining words"', None))
    assert (found, busy_found, dispatched, adapter.sent, adapter._pending_messages) == (
        True, True if overlap else None,
        (retained, '"remaining words"') if partial else (None, None),
        echoes, {},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["admission", "prepare", "hook", "claimed"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_followup_withdrawal_stops_at_claim_and_cancellation_keeps_unclaimed_input(
    monkeypatch, phase, partial, cancel,
):
    from gateway.turn_context import TurnContext
    from tests.gateway.test_queued_followup_processing_hooks import HookRecordingAdapter

    adapter = HookRecordingAdapter()
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[adapter.platform] = adapter
    adapter.gateway_runner = runner
    adapter.set_queued_withdrawal_handler(runner._withdraw_queued_followups)
    source = replace(_source(), user_id="alice")
    event = MessageEvent(text="withdrawn", source=source, message_id="withdrawn")
    retained = MessageEvent(text="retained", source=source, message_id="remaining")
    key = adapter._event_session_key(event)
    runner._enqueue_fifo(key, event, adapter)
    if partial:
        merge_pending_message_event(adapter._pending_messages, key, retained, merge_text=True)
    accepted_text = event.text
    pending_event, pending = await runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, source, key,
    )
    entered = asyncio.Event()
    resume = asyncio.Event()
    calls = []

    async def prepare(*, event, **kwargs):
        text = event.text
        if phase == "prepare" and not entered.is_set():
            entered.set()
            await resume.wait()
        return text

    async def start(event):
        adapter.started.append(event.message_id)
        if phase == "hook":
            entered.set()
            await resume.wait()

    async def refresh(*args):
        return None

    async def run(**kwargs):
        calls.append(kwargs["message"])
        if phase == "claimed":
            entered.set()
            await resume.wait()
        return {"final_response": "followup", "messages": []}

    monkeypatch.setattr(runner, "_prepare_profile_scoped_inbound_message_text", prepare)
    monkeypatch.setattr(runner, "_persist_prompt_pins", refresh)
    monkeypatch.setattr(runner, "_refresh_agent_cache_message_count", refresh)
    monkeypatch.setattr(runner, "_run_agent", run)
    monkeypatch.setattr(adapter, "on_processing_start", start)
    turn = TurnContext(source=source, session_key=key, session_id="session", history=[], context_prompt="")
    if phase == "admission":
        async def rewrite(hook, *, event, **kwargs):
            text = event.text
            if not entered.is_set():
                entered.set()
                await resume.wait()
            return [{"action": "rewrite", "text": "rewritten " + text}]

        async def handle(event, *args):
            calls.append(event.text)
            return None

        monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
        monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", rewrite)
        monkeypatch.setattr(runner, "_handle_message_with_agent", handle)
        adapter.set_message_handler(runner._handle_message)
        adapter._start_session_processing(event, key)
        task = adapter._session_tasks[key]
    else:
        task = asyncio.create_task(runner._run_agent_queued_followup(
            turn, adapter, pending, pending_event, "done", {"interrupted": True, "messages": []}, None,
        ))
        adapter._track_session_task(key, task)
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        found = adapter.withdraw_pending_message("withdrawn", chat_id=source.chat_id, sender_id="alice")
        if cancel:
            await adapter.cancel_session_processing(key, discard_pending=False)
        else:
            resume.set()
            await task
    finally:
        resume.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    claimed = phase == "claimed"
    expected_calls = [accepted_text] if claimed else (
        ["rewritten retained" if phase == "admission" else "retained"] if partial and not cancel else []
    )
    expected_pending = {key: retained} if cancel and partial and not claimed else {}
    assert (found, calls, adapter._pending_messages, task.cancelled()) == (
        not claimed, expected_calls, expected_pending, cancel,
    )
