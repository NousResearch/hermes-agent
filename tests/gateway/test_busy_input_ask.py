from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key
from gateway.run import GatewayRunner


class _PromptAdapter:
    def __init__(self):
        self._pending_messages = {}
        self._active_sessions = {}
        self.prompts = []
        self.started = []

    async def send_busy_prompt(self, **kwargs):
        self.prompts.append(kwargs)
        return SendResult(success=True, message_id="prompt-1")

    def _start_session_processing(self, event, session_key):
        self.started.append((session_key, event))
        return True


def _make_runner(adapter=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = adapter or _PromptAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._busy_prompt_events = {}
    runner._busy_prompt_counter = iter(range(1, 100))
    runner._busy_input_mode = "ask"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._restart_requested = False
    runner._is_user_authorized = lambda _source: True
    return runner, adapter


def _event(text="follow up"):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="u1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="m2",
    )


def test_busy_input_mode_accepts_ask(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "ask")
    assert GatewayRunner._load_busy_input_mode() == "ask"


def test_busy_text_mode_queue_does_not_bypass_ask(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "ask")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "queue")
    assert GatewayRunner._load_busy_text_mode() == "interrupt"


@pytest.mark.asyncio
async def test_busy_ask_sends_prompt_instead_of_queueing_immediately():
    runner, adapter = _make_runner()
    event = _event()
    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {"current_tool": "terminal"}
    runner._running_agents[session_key] = running_agent

    handled = await runner._handle_active_session_busy_message(event, session_key)

    assert handled is True
    assert adapter.prompts
    assert adapter.prompts[0]["prompt_id"] == "1"
    assert "still working" in adapter.prompts[0]["prompt"]
    assert session_key not in adapter._pending_messages
    assert runner._busy_prompt_events["1"] == (session_key, event)
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_busy_prompt_queue_choice_queues_when_session_active():
    runner, adapter = _make_runner()
    event = _event()
    session_key = build_session_key(event.source)
    runner._running_agents[session_key] = MagicMock()
    runner._busy_prompt_events["7"] = (session_key, event)

    resolved = await runner.resolve_busy_prompt_choice("7", "queue")

    assert resolved is True
    assert adapter._pending_messages[session_key] is event
    assert "7" not in runner._busy_prompt_events


@pytest.mark.asyncio
async def test_busy_prompt_queue_choice_starts_if_original_run_finished():
    runner, adapter = _make_runner()
    event = _event()
    session_key = build_session_key(event.source)
    runner._busy_prompt_events["8"] = (session_key, event)

    resolved = await runner.resolve_busy_prompt_choice("8", "queue")

    assert resolved is True
    assert adapter.started == [(session_key, event)]
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_busy_prompt_interrupt_choice_interrupts_running_agent():
    runner, adapter = _make_runner()
    event = _event("stop and do this")
    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent
    runner._busy_prompt_events["9"] = (session_key, event)

    resolved = await runner.resolve_busy_prompt_choice("9", "interrupt")

    assert resolved is True
    assert adapter._pending_messages[session_key] is event
    running_agent.interrupt.assert_called_once_with("stop and do this")


@pytest.mark.asyncio
async def test_busy_prompt_ignore_drops_event():
    runner, adapter = _make_runner()
    event = _event()
    session_key = build_session_key(event.source)
    runner._busy_prompt_events["10"] = (session_key, event)

    resolved = await runner.resolve_busy_prompt_choice("10", "ignore")

    assert resolved is True
    assert session_key not in adapter._pending_messages
    assert adapter.started == []
