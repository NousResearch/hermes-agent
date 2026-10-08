"""Run limits must be enforceable; a cancelled transport cannot release a live writer."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import api_server, api_server_runs as runs
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_live_owner_handoff_rejects_limits_before_mailbox_delivery(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._conversation_history_for_session = AsyncMock(return_value=[])
    adapter._admit_to_live_bot_chat = AsyncMock()
    monkeypatch.setattr(runs, '_resolve_live_session_id', AsyncMock(return_value='owned'))
    monkeypatch.setattr(runs, '_acquire_run_lease_or_response', AsyncMock(return_value=(None, None)))
    request = MagicMock()
    request.headers = {}
    request.json = AsyncMock(return_value={
        'input': 'hello', 'session_id': 'owned', 'execution_policy': {'max_turns': 1}})
    response = await runs._handle_runs(adapter, request, _api_server=api_server)
    assert response.status == 409
    assert 'execution_policy_handoff_unsupported' in response.text
    adapter._admit_to_live_bot_chat.assert_not_awaited()
    assert not adapter._active_run_tasks
    runs._close_run_state(adapter)


@pytest.mark.asyncio
async def test_cancelled_run_retains_lease_until_worker_exits(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agent = SimpleNamespace(event_callback=None)
    adapter._create_agent = MagicMock(return_value=agent)
    started, unblock = threading.Event(), threading.Event()
    lease = object()
    releases = []
    monkeypatch.setattr(runs, '_release_run_lease', lambda value: releases.append(value) if value else None)

    def worker(*args, **kwargs):
        started.set()
        assert unblock.wait(5), 'test worker was not released'
        return {}, {}, {}

    monkeypatch.setattr(runs, '_run_agent_sync_owned', worker)
    run = runs._RunLaunch(
        adapter, 'run_test', runs._RunStream(), 'session_test', None, False, 'hello', [], True,
        agent_kwargs={}, request_profile=None, browser_control_principal=None,
        browser_control_transport_family=None, session_lease=lease)
    task = asyncio.create_task(runs._execute_run(adapter, run, _api_server=api_server))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert releases == []
        assert run.session_lease is lease
    finally:
        unblock.set()
        if run.worker_future is not None:
            await asyncio.wait_for(asyncio.shield(run.worker_future), 3)
        runs._close_run_state(adapter)
    assert releases == [lease]
    assert run.session_lease is None
