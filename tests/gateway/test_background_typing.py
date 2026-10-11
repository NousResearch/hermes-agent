"""Real process liveness controls background presence; no new model turn required."""
import asyncio
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.background_typing import refresh_background_typing
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools.process_registry import process_registry


def runner_for(source):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    bot = SimpleNamespace(send_chat_action=AsyncMock())
    adapter._bot = bot
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._draining = runner._restart_requested = False
    runner._delivery_adapter_for = lambda source: adapter
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value={"ended_at": None}))
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None, _entries={"owner": SimpleNamespace(origin=source)})
    return runner, adapter, bot


@pytest.mark.asyncio
async def test_real_process_keeps_typing_until_exit_without_tracking_daemons():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", thread_id="10")
    runner, _adapter, bot = runner_for(source)
    command = f'{shlex.quote(sys.executable)} -c "import time; time.sleep(60)"'
    processes = [process_registry.spawn_local(command, session_key="owner") for _ in range(2)]
    try:
        for p in processes:
            p.parent_session_id = "parent"
            p.notify_on_complete = True
        await refresh_background_typing(runner)
        # Two jobs in one topic still generate just one refresh.
        bot.send_chat_action.assert_awaited_once_with(chat_id=123, action="typing", message_thread_id=10)
        await refresh_background_typing(runner)
        assert bot.send_chat_action.await_count == 2
        for p in processes:
            p.notify_on_complete = False
        await refresh_background_typing(runner)
        assert bot.send_chat_action.await_count == 2
        for p in processes:
            p.notify_on_complete = True
            await asyncio.to_thread(process_registry.kill_process, p.id)
        await refresh_background_typing(runner)
        assert bot.send_chat_action.await_count == 2
    finally:
        for p in processes:
            await asyncio.to_thread(process_registry.kill_process, p.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ["foreground", "prompt", "closed", "disabled", "draining", "completed"])
async def test_background_presence_respects_owner_lifecycle(monkeypatch, blocked):
    from tools import async_delegation
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", thread_id="10")
    runner, adapter, bot = runner_for(source)
    record = {"session_key": "owner", "parent_session_id": "parent", "status": "running"}
    monkeypatch.setattr(async_delegation, "list_async_delegations", lambda: [record])
    from gateway.run import _USER_BOUNDARY_END_REASONS
    configure = {
        "foreground": lambda: adapter._active_sessions.update({"owner": asyncio.Event()}),
        "prompt": lambda: adapter.pause_typing_for_chat("123", metadata={"thread_id": "10"}),
        "closed": lambda: setattr(runner._session_db.get_session, "return_value",
                                  {"ended_at": 1, "end_reason": next(iter(_USER_BOUNDARY_END_REASONS))}),
        "disabled": lambda: setattr(adapter.config, "typing_indicator", False),
        "draining": lambda: setattr(runner, "_draining", True),
        "completed": lambda: record.update(status="completed"),
    }
    configure[blocked]()
    await refresh_background_typing(runner)
    bot.send_chat_action.assert_not_awaited()
