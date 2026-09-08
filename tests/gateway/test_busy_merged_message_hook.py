"""``on_message_merged`` fires exactly once when a busy follow-up is folded into the running turn
(steer, or interrupt redirect) and never when it is queued for a turn of its own.

An adapter that keeps per-event state (an ack/receipt row per inbound message) otherwise cannot
learn that a merged message was consumed: ``on_processing_start``/``on_processing_complete`` only
fire for events that get their own turn."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, SessionSource, build_session_key
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test-token"), Platform.TELEGRAM)
        self.merged = []
        self.started = []
        self.completed = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}

    async def on_message_merged(self, event):
        self.merged.append(event)

    async def on_processing_start(self, event):
        self.started.append(event)

    async def on_processing_complete(self, event, outcome):
        self.completed.append((event, outcome))


def _event(text="follow up") -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, message_id="message-1",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm", user_id="user-1"),
    )


@pytest.fixture
def runner(monkeypatch):
    # No busy ack: the handler returns right after the steer/redirect/queue decision.
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "0")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._profile_adapters = {}
    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner._sessions = {}
    runner._draining = False
    runner._restart_requested = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda source: True
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    return runner


def _running_agent(runner, session_key, *, steer=True, redirect=True):
    agent = MagicMock()
    agent._active_children = []
    agent._supports_active_turn_redirect = True
    agent.steer = MagicMock(return_value=steer)
    agent.redirect = MagicMock(return_value=redirect)
    runner._running_agents[session_key] = agent
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
@pytest.mark.parametrize("path", ["busy_handler", "priority"])
async def test_merged_follow_up_fires_on_message_merged_once(runner, mode, path):
    runner._busy_input_mode = mode
    adapter = runner.adapters[Platform.TELEGRAM]
    event = _event()
    session_key = build_session_key(event.source)
    agent = _running_agent(runner, session_key)

    if path == "busy_handler":
        assert await runner._handle_active_session_busy_message(event, session_key) is True
    else:
        assert await runner._hm_handle_running_session_message(event, event.source, session_key) is None

    verb = agent.redirect if mode == "interrupt" else agent.steer
    verb.assert_called_once()
    agent.interrupt.assert_not_called()
    assert adapter.merged == [event]
    assert adapter.started == [] and adapter.completed == []
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["busy_handler", "priority"])
async def test_queued_follow_up_does_not_fire_on_message_merged(runner, path):
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "queue"
    adapter = runner.adapters[Platform.TELEGRAM]
    event = _event()
    session_key = build_session_key(event.source)
    agent = _running_agent(runner, session_key)

    if path == "busy_handler":
        # False: the base adapter queues the event itself; nothing was merged.
        assert await runner._handle_active_session_busy_message(event, session_key) is False
    else:
        assert await runner._hm_handle_running_session_message(event, event.source, session_key) is None
        assert adapter._pending_messages[session_key] is event

    agent.steer.assert_not_called()
    agent.redirect.assert_not_called()
    assert adapter.merged == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
@pytest.mark.parametrize("path", ["busy_handler", "priority"])
async def test_rejected_steer_or_redirect_falls_back_without_on_message_merged(runner, mode, path):
    """The agent refuses the mid-run text, so the event is queued (steer) or interrupts (redirect)
    and later gets its own turn — it was not merged."""
    runner._busy_input_mode = mode
    adapter = runner.adapters[Platform.TELEGRAM]
    event = _event()
    session_key = build_session_key(event.source)
    agent = _running_agent(runner, session_key, steer=False, redirect=False)

    if path == "busy_handler":
        assert await runner._handle_active_session_busy_message(event, session_key) is True
    else:
        assert await runner._hm_handle_running_session_message(event, event.source, session_key) is None

    verb = agent.redirect if mode == "interrupt" else agent.steer
    verb.assert_called_once()
    assert adapter.merged == []
    if mode == "steer":
        assert adapter._pending_messages[session_key] is event
    else:
        agent.interrupt.assert_called_once_with("follow up")
