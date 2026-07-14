import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL


class _PendingAdapter:
    def __init__(self):
        self._pending_messages = {}


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self, config):
        super().__init__(config, Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: _PendingAdapter()}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._busy_ack_ts = {}
    runner._voice_mode = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner.session_store = None
    runner._is_user_authorized = lambda _source: True
    return runner


@pytest.mark.asyncio
async def test_handle_message_does_not_priority_interrupt_photo_followup():
    runner = _make_runner()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1")
    session_key = build_session_key(source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    event = MessageEvent(
        text="caption",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo-a.jpg"],
        media_types=["image/jpeg"],
    )

    result = await runner._handle_message(event)

    assert result is None
    running_agent.interrupt.assert_not_called()
    assert runner.adapters[Platform.TELEGRAM]._pending_messages[session_key] is event


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_start", [True, False])
async def test_base_adapter_busy_route_queues_photo_without_interrupt(pending_start):
    runner = _make_runner()
    config = PlatformConfig(enabled=True, token="***", typing_indicator=False)
    adapter = _RecordingAdapter(config)
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters = {Platform.TELEGRAM: adapter}

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="photo-busy-route",
        chat_type="dm",
        user_id="u1",
    )
    session_key = build_session_key(source)
    interrupt_event = asyncio.Event()
    adapter._active_sessions[session_key] = interrupt_event
    runner._session_run_generation[session_key] = 1
    running_agent = _AGENT_PENDING_SENTINEL if pending_start else MagicMock()
    runner._running_agents[session_key] = running_agent

    event = MessageEvent(
        text="album caption",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo-b.jpg"],
        media_types=["image/jpeg"],
    )

    await adapter.handle_message(event)

    assert adapter._pending_messages[session_key] is event
    assert runner._session_run_generation[session_key] == 1
    assert interrupt_event.is_set() is False
    assert adapter.sent == []
    if not pending_start:
        running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_base_adapter_interrupt_replaces_startup_turn_exactly_once(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _make_runner()
    config = PlatformConfig(enabled=True, token="***", typing_indicator=False)
    adapter = _RecordingAdapter(config)
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="startup-interrupt-route",
        chat_type="dm",
        user_id="u1",
    )
    session_key = build_session_key(source)
    obsolete_started = asyncio.Event()
    release_obsolete = asyncio.Event()
    handled = []

    async def handler(event):
        handled.append(event.text)
        if event.text == "obsolete prompt":
            runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
            runner._session_run_generation[session_key] = 1
            obsolete_started.set()
            await release_obsolete.wait()
            runner._running_agents.pop(session_key, None)
            return "obsolete answer"
        return "replacement answer"

    adapter.set_message_handler(handler)
    first = MessageEvent(
        text="obsolete prompt",
        message_type=MessageType.TEXT,
        source=source,
    )
    replacement = MessageEvent(
        text="replacement prompt",
        message_type=MessageType.TEXT,
        source=source,
    )

    await adapter.handle_message(first)
    await asyncio.wait_for(obsolete_started.wait(), timeout=2.0)
    await adapter.handle_message(replacement)

    assert adapter._pending_messages[session_key] is replacement
    assert runner._session_run_generation[session_key] == 2
    assert adapter._active_sessions[session_key].is_set()

    release_obsolete.set()
    for _ in range(200):
        if adapter.sent == ["replacement answer"] and session_key not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)

    assert handled == ["obsolete prompt", "replacement prompt"]
    assert adapter.sent == ["replacement answer"]
    assert session_key not in adapter._pending_messages
    assert session_key not in adapter._active_sessions
