"""Adapter plugins can query background work without importing tool internals."""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter
from tools import process_registry as processes


class Adapter(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        pass

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.parametrize("notify,exited,expected", [(True, False, True), (False, False, False), (True, True, False)])
def test_background_activity_requires_same_session_completion_work(monkeypatch, notify, exited, expected):
    from gateway.config import Platform, PlatformConfig

    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    key = "agent:alpha:discord:group:100:1"
    worker = processes.ProcessSession(id="job", command="build", session_key=key,
                                      notify_on_complete=notify, exited=exited)
    registry._running[worker.id] = worker
    adapter = Adapter(Platform.DISCORD, PlatformConfig(enabled=True))
    assert adapter.has_background_work(key) is expected
    assert not adapter.has_background_work("agent:beta:discord:group:100:1")
    assert not adapter.has_background_work("")


@pytest.mark.asyncio
async def test_adapter_interrupt_preserves_other_sessions():
    import asyncio
    from gateway.config import Platform, PlatformConfig

    adapter = Adapter(Platform.DISCORD, PlatformConfig(enabled=True))
    mine, other = asyncio.Event(), asyncio.Event()
    adapter._active_sessions.update(mine=mine, other=other)
    adapter._stop_typing_with_metadata = AsyncMock()
    await adapter.interrupt_session_activity("mine", "100", {"thread_id": "100"})
    assert mine.is_set()
    assert not other.is_set()
    adapter._stop_typing_with_metadata.assert_awaited_once_with("100", {"thread_id": "100"})
