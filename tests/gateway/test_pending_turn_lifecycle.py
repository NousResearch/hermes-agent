"""Queued work belongs to the finishing turn across the TTS and delivery awaits."""

import asyncio
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.turn_context import TurnContext
from tests.gateway.test_drained_event_conservation import (
    KEY,
    _admit,
    _make_runner,
    _overflow,
    _source,
    _SttBarrier,
    _text_event,
    _voice_event,
)


class _TtsBarrier:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.done = False
        self.suppress_whole_file = False

    def finish(self):
        pass

    async def wait_complete(self, *, timeout):
        self.entered.set()
        await self.release.wait()
        self.done = True


def _finishing_turn(runner, monkeypatch, generation, result, barrier):
    """Keep the real _run_agent orchestration, TTS wait, drain and task cleanup."""
    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=object))
    ctx = TurnContext(
        source=_source(), session_key=KEY, session_id="sess-1", run_generation=generation,
        result_holder=[result], streaming_tts_consumer_holder=[barrier],
    )
    disp = SimpleNamespace(
        needs_progress_queue=False, log_mode_enabled=False, _native_slack_task_cards=False,
    )
    runner._get_proxy_url = lambda: None
    runner._run_agent_display_settings = lambda source: disp
    runner._run_agent_build_turn_context = MagicMock(
        return_value=(ctx, SimpleNamespace(run_sync=lambda: None), None),
    )
    runner._run_agent_bind_turn_wiring = MagicMock()
    runner._run_agent_start_streaming_tts = MagicMock()
    for name in ("stream_consumer_task", "track_agent", "monitor_for_interrupt", "notify_long_running"):
        setattr(runner, f"_run_agent_{name}", AsyncMock())
    runner._run_agent_start_turn_worker = MagicMock(return_value=SimpleNamespace(executor_task=None))
    runner._run_agent_await_turn_worker = AsyncMock(return_value=result)
    runner._run_agent_evict_on_fallback = MagicMock()
    runner._run_agent_queued_followup = AsyncMock(return_value={"final_response": "stale follow-up"})
    runner._run_agent_mark_streamed_delivery = AsyncMock()
    runner._run_agent_schedule_bubble_cleanup = MagicMock()


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", ["successor_queue", "steer", "interrupt"])
async def test_finishing_turn_does_not_adopt_generation_invalidated_during_tts(monkeypatch, followup):
    runner, adapter = _make_runner(monkeypatch)
    generation = runner._begin_session_run_generation(KEY)
    result = {"final_response": "old final", "messages": []}
    if followup == "interrupt":
        result.update(interrupted=True, interrupt_message="old interrupt")
    else:
        result["pending_steer"] = "old correction"
    barrier = _TtsBarrier()
    _finishing_turn(runner, monkeypatch, generation, result, barrier)
    runner._transcribe_and_echo_pending_voice = AsyncMock(return_value=("successor transcript", []))
    task = asyncio.create_task(runner._run_agent(
        "old input", "context", [], _source(), "sess-1", session_key=KEY, run_generation=generation,
    ))
    try:
        await asyncio.wait_for(barrier.entered.wait(), timeout=5)
        runner._invalidate_session_run_generation(KEY, reason="user_stop")
        runner._clear_conversation_scope(KEY, reason="user_stop")
        successor, later = _voice_event("successor", "D1"), _text_event("later", "E1")
        if followup == "successor_queue":
            _admit(runner, adapter, successor, later)
        barrier.release.set()
        assert await asyncio.wait_for(task, timeout=5) is result
        runner._run_agent_queued_followup.assert_not_awaited()
        runner._transcribe_and_echo_pending_voice.assert_not_awaited()
        if followup == "successor_queue":
            assert adapter._pending_messages[KEY] is successor
            assert _overflow(runner) == [later]
        else:
            assert KEY not in adapter._pending_messages
            assert _overflow(runner) == []
    finally:
        barrier.release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["entry", "transcription", "cancelled", "error"])
async def test_restart_retains_accepted_steer_before_queued_events(monkeypatch, tmp_path, boundary):
    runner, adapter = _make_runner(monkeypatch)
    a, b, c = _voice_event("A", "A1"), _text_event("B", "B1"), _text_event("C", "C1")
    _admit(runner, adapter, a, b, c)
    result = {"final_response": "done", "messages": [], "pending_steer": "accepted correction"}
    barrier = _SttBarrier(fail=boundary == "error")
    runner._transcribe_and_echo_pending_voice = barrier
    runner._draining = boundary == "entry"
    task = asyncio.create_task(runner._run_agent_drain_pending(result, adapter, _source(), KEY))
    try:
        if boundary != "entry":
            await asyncio.wait_for(barrier.entered.wait(), timeout=5)
            runner._draining = True
        if boundary == "cancelled":
            assert task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            barrier.release.set()
            if boundary == "error":
                with pytest.raises(RuntimeError, match="transcription blew up"):
                    await task
            else:
                assert await asyncio.wait_for(task, timeout=5) == (None, None)
        retained = [adapter._pending_messages[KEY], *_overflow(runner)]
        assert [event.text for event in retained] == ["accepted correction", "A", "B", "C"]
        assert retained[1:] == [a, b, c]
        assert not retained[0].internal
        assert retained[0].preserve_prompt_pins
        await _assert_steer_spool_order(tmp_path, runner, adapter)
    finally:
        barrier.release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


async def _assert_steer_spool_order(tmp_path, runner, adapter):
    from gateway.shutdown_flush import (
        flush_overflow_to_file,
        flush_pending_to_file,
        recover_pending_spool,
    )
    from hermes_state import SessionDB

    assert flush_pending_to_file(dict(adapter._pending_messages), reason="shutdown") == 1
    assert flush_overflow_to_file({KEY: list(_overflow(runner))}, reason="shutdown") == 3
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.ensure_session("sess-1", "gateway")
        recovered, held = recover_pending_spool(
            db, session_resolver=lambda key, not_after=None: ("sess-1", db),
        )
        assert (recovered, held) == (4, set())
        assert [message["content"] for message in db.get_messages("sess-1")] == [
            "accepted correction", "A", "B", "C",
        ]
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary, outcome", [
    (boundary, "invalidate") for boundary in ("delivery", "prepare", "persist", "typing", "refresh")
] + [
    (boundary, outcome) for boundary in ("delivery", "prepare", "refresh")
    for outcome in ("draining", "cancelled", "error")
])
async def test_queued_followup_retains_owner_and_accepted_work_across_awaits(monkeypatch, boundary, outcome):
    runner, adapter = _make_runner(monkeypatch)
    generation = runner._begin_session_run_generation(KEY)
    event = _text_event("old accepted follow-up", "A1")
    event.channel_prompt = "old channel prompt"
    result = {"final_response": "old final", "messages": []}
    _admit(runner, adapter, event)
    drained, pending = await runner._run_agent_drain_pending(result, adapter, _source(), KEY)
    assert drained is event and pending == event.text
    entered, release = asyncio.Event(), asyncio.Event()

    async def pause_at(name, *args, **kwargs):
        if boundary == name:
            entered.set()
            await release.wait()
            if outcome == "error":
                raise RuntimeError("follow-up preparation failed")
        return event.text if name == "prepare" else None

    runner._run_agent_deliver_first_response = AsyncMock(side_effect=partial(pause_at, "delivery"))
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(side_effect=partial(pause_at, "prepare"))
    runner._persist_prompt_pins = AsyncMock(side_effect=partial(pause_at, "persist"))
    adapter.send_typing = AsyncMock(side_effect=partial(pause_at, "typing"))
    runner._refresh_agent_cache_message_count = AsyncMock(side_effect=partial(pause_at, "refresh"))
    runner._session_key_for_source = lambda source: KEY
    runner._run_agent = AsyncMock(return_value={"final_response": "stale recursive turn"})
    ctx = TurnContext(
        source=_source(), session_id="sess-1", session_key=KEY, run_generation=generation,
        history=[], result_holder=[result],
    )
    task = asyncio.create_task(runner._run_agent_queued_followup(
        ctx, adapter, event.text, event, result, result, None,
    ))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        if outcome == "invalidate":
            runner._invalidate_session_run_generation(KEY, reason="user_stop")
            runner._clear_conversation_scope(KEY, reason="user_stop")
            successor, later = _text_event("successor", "D1"), _text_event("later", "E1")
            _admit(runner, adapter, successor, later)
            runner._pinned_channel_inputs(KEY, "successor channel prompt", _source(), preserve_pin=False)
        else:
            runner._draining = True
        if outcome == "cancelled":
            assert task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            if outcome == "error":
                with pytest.raises(RuntimeError, match="follow-up preparation failed"):
                    await task
            else:
                await asyncio.wait_for(task, timeout=5)
        runner._run_agent.assert_not_awaited()
        if outcome == "invalidate":
            assert runner._peek_session_state(KEY).conversation.channel_pin[0] == "successor channel prompt"
            assert adapter._pending_messages[KEY] is successor
            assert _overflow(runner) == [later]
        else:
            assert adapter._pending_messages.get(KEY) is event
            assert _overflow(runner) == []
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
