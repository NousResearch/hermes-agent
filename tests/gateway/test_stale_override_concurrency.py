"""Ingress must retain session ownership while a stale-notice read is suspended."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stale_override_notice import StaleOverrideNoticeConfig
from gateway.session import AsyncSessionStore, SessionStore, build_session_key
from gateway.run import _AGENT_PENDING_SENTINEL
from tests.gateway.test_session_race_guard import _make_event, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", ["second message", "/stop"])
async def test_notice_read_preserves_busy_and_stop_contract(followup, tmp_path):
    runner = _make_runner()
    runner.config.stale_override_notice = StaleOverrideNoticeConfig(
        mode="info_only", channels=("*",)
    )
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner._session_db = SimpleNamespace(_db=runner.session_store._db)
    runner._async_session_store = AsyncSessionStore(runner.session_store)
    original_read = runner.async_session_store.get_session_metadata
    entered = asyncio.Event()
    release = asyncio.Event()
    reads = 0

    async def read_clock(*args):
        nonlocal reads
        reads += 1
        if reads == 1:
            entered.set()
            await release.wait()
        return await original_read(*args)

    runner.async_session_store.get_session_metadata = AsyncMock(side_effect=read_clock)
    runner._handle_message_with_agent = AsyncMock(return_value="answer")
    runner._run_post_turn_hooks = AsyncMock()
    first = _make_event("first message")
    key = build_session_key(first.source)
    task = asyncio.create_task(runner._handle_message(first))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert runner._running_agents.get(key) is _AGENT_PENDING_SENTINEL
        response = await asyncio.wait_for(
            runner._handle_message(_make_event(followup)), timeout=5
        )
        runner._handle_message_with_agent.assert_not_awaited()
        if followup == "/stop":
            assert "stopped" in response.lower()
            assert key not in runner._running_agents
        else:
            assert runner.adapters[first.source.platform]._pending_messages[key].text == followup
        release.set()
        await asyncio.wait_for(task, timeout=5)
        if followup == "/stop":
            runner._handle_message_with_agent.assert_not_awaited()
        else:
            runner._handle_message_with_agent.assert_awaited_once()
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=5)
    assert key not in runner._running_agents


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["clock", "session", "picker"])
async def test_invalidated_notice_cannot_hold_or_replay_after_suspended_io(phase):
    from tests.gateway.test_stale_override_notice import _event, _runner

    runner = _runner("confirm")
    key = "held-session"
    generation = runner._begin_session_run_generation(key)
    entered, release = asyncio.Event(), asyncio.Event()
    event = _event()

    async def suspend(result):
        entered.set()
        await release.wait()
        return result

    if phase == "clock":
        import time
        async def read_clock(*args):
            return await suspend(time.time() - 3700)
        runner.async_session_store.get_session_metadata.side_effect = read_clock
    elif phase == "session":
        async def open_session(*args, **kwargs):
            return await suspend(SimpleNamespace(was_auto_reset=False))
        runner.async_session_store.get_or_create_session.side_effect = open_session
    else:
        real_picker = runner._adapter.send_choice_picker
        async def send_picker(**kwargs):
            result = await real_picker(**kwargs)
            return await suspend(result)
        runner._adapter.send_choice_picker = send_picker

    task = asyncio.create_task(runner._maybe_handle_stale_override_notice(
        event, key, run_generation=generation,
    ))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        runner._invalidate_session_run_generation(key, reason="stop")
        successor = runner._begin_session_run_generation(key)
        release.set()
        assert await asyncio.wait_for(task, timeout=5) == (True, None)
        assert runner._current_session_run_generation(key) == successor
        assert not runner._stale_override_pending
        runner._clear_stale_override_selection.assert_not_awaited()
        runner._adapter.handle_message.assert_not_awaited()
        runner._schedule_stale_override_prompt_expiry.assert_not_called()
        if runner._adapter.picker_call:
            result = await runner._adapter.picker_call["on_choice_selected"]("123", "continue")
            assert "expired" in result.lower()
            runner._adapter.handle_message.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_stop_during_picker_reset_cannot_mutate_or_replay_successor():
    from gateway.run import GatewayRunner
    from tests.gateway.test_stale_override_notice import _event, _runner

    runner = _runner("confirm")
    key = "held-session"
    runner._begin_session_run_generation(key)
    runner._clear_stale_override_selection = (
        GatewayRunner._clear_stale_override_selection.__get__(runner)
    )
    runner._evict_cached_agent = MagicMock()
    entered, release = asyncio.Event(), asyncio.Event()

    async def persist(*args):
        entered.set()
        await release.wait()
    runner.async_session_store.set_model_override.side_effect = persist
    assert await runner._maybe_handle_stale_override_notice(_event(), key) == (True, None)
    callback = runner._adapter.picker_call["on_choice_selected"]
    task = asyncio.create_task(callback("123", "defaults"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        runner._invalidate_session_run_generation(key, reason="stop")
        runner._begin_session_run_generation(key)
        successor_reasoning = {"enabled": True, "effort": "high"}
        runner._set_session_reasoning_override(key, successor_reasoning)
        release.set()
        result = await asyncio.wait_for(task, timeout=5)
        assert "expired" in result.lower()
        assert runner._session_state(key).conversation.reasoning_override == successor_reasoning
        runner._adapter.handle_message.assert_not_awaited()
        runner._evict_cached_agent.assert_not_called()
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=5)
