"""Ingress must retain session ownership while a stale-notice read is suspended."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
