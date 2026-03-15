import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _source(chat_id="123456", chat_type="dm"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
    )


@pytest.mark.asyncio
async def test_cancel_background_tasks_cancels_inflight_message_processing():
    adapter = StubAdapter()
    release = asyncio.Event()

    async def block_forever(_event):
        await release.wait()
        return None

    adapter.set_message_handler(block_forever)
    event = MessageEvent(text="work", source=_source(), message_id="1")

    await adapter.handle_message(event)
    await asyncio.sleep(0)

    session_key = build_session_key(event.source)
    assert session_key in adapter._active_sessions
    assert adapter._background_tasks

    await adapter.cancel_background_tasks()

    assert adapter._background_tasks == set()
    assert adapter._active_sessions == {}
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_gateway_stop_interrupts_running_agents_and_cancels_adapter_tasks():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._pending_messages = {"session": "pending text"}
    runner._pending_approvals = {"session": {"command": "rm -rf /tmp/x"}}
    runner._shutdown_all_gateway_honcho = lambda: None

    adapter = StubAdapter()
    release = asyncio.Event()

    async def block_forever(_event):
        await release.wait()
        return None

    adapter.set_message_handler(block_forever)
    event = MessageEvent(text="work", source=_source(), message_id="1")
    await adapter.handle_message(event)
    await asyncio.sleep(0)

    disconnect_mock = AsyncMock()
    adapter.disconnect = disconnect_mock

    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents = {session_key: running_agent}
    runner.adapters = {Platform.TELEGRAM: adapter}

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    running_agent.interrupt.assert_called_once_with("Gateway shutting down")
    disconnect_mock.assert_awaited_once()
    assert runner.adapters == {}
    assert runner._running_agents == {}
    assert runner._pending_messages == {}
    assert runner._pending_approvals == {}
    assert runner._shutdown_event.is_set() is True
