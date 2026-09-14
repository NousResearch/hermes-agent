"""Processing-hook parity for queued follow-up turns.

A message that arrives while a turn is already running is parked in the
adapter's ``_pending_messages`` slot and drained *in-band* by
``GatewayRunner._run_agent`` rather than by
``BasePlatformAdapter._process_message_background``.  The runner-side drain
must still fire the ``on_processing_start`` / ``on_processing_complete``
lifecycle hooks, otherwise every platform that renders a read-receipt
reaction from those hooks (Slack 👀, Discord, Telegram, Feishu, Matrix,
Signal, ...) silently skips the acknowledgement for mid-turn messages.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionSource


class HookRecordingAdapter(BasePlatformAdapter):
    """Adapter that records the processing-hook lifecycle it is driven through."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.started: list = []
        self.completed: list = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def on_processing_start(self, event: MessageEvent) -> None:
        self.started.append(getattr(event, "message_id", None))

    async def on_processing_complete(self, event, outcome) -> None:
        self.completed.append((getattr(event, "message_id", None), outcome))


class _TwoTurnAgent:
    calls: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


class _RaisingSecondTurnAgent:
    calls: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        if len(type(self).calls) >= 2:
            raise RuntimeError("boom in the queued follow-up turn")
        return {
            "final_response": "done-1",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _install_fake_agent(monkeypatch, tmp_path, agent_cls):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )


SESSION_KEY = "agent:main:telegram:dm:4242"


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm")


@pytest.mark.asyncio
async def test_queued_followup_fires_processing_hooks(monkeypatch, tmp_path):
    """The runner-drained follow-up gets the same start/complete hooks as a
    message that arrives while the session is idle."""
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks",
        session_key=SESSION_KEY,
    )

    # The follow-up really did run in-band.
    assert result["final_response"] == "done-2"
    assert _TwoTurnAgent.calls == ["the first turn", "the follow-up"]

    # ...and it was acknowledged through the lifecycle hooks.
    assert adapter.started == ["queued-1"]
    assert adapter.completed == [("queued-1", ProcessingOutcome.SUCCESS)]


@pytest.mark.asyncio
async def test_queued_followup_failure_completes_the_hook(monkeypatch, tmp_path):
    """A follow-up turn that blows up still closes its hook, so a platform
    never strands a 'still working' marker on the user's message."""
    _RaisingSecondTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _RaisingSecondTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the doomed follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="queued-2",
    )

    with pytest.raises(RuntimeError):
        await runner._run_agent(
            message="the first turn",
            context_prompt="",
            history=[],
            source=_source(),
            session_id="sess-hooks-failure",
            session_key=SESSION_KEY,
        )

    assert adapter.started == ["queued-2"]
    assert adapter.completed == [("queued-2", ProcessingOutcome.FAILURE)]


@pytest.mark.asyncio
async def test_synthetic_followup_is_not_acknowledged(monkeypatch, tmp_path):
    """Drains with no inbound platform message — /goal continuations, wake-ups,
    CLI hand-offs — carry no message_id and must stay silent: there is nothing
    on the platform to react to."""
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="synthetic continuation",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id=None,
    )

    result = await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-synthetic",
        session_key=SESSION_KEY,
    )

    # It still ran — we only suppressed the acknowledgement, not the turn.
    assert result["final_response"] == "done-2"
    assert _TwoTurnAgent.calls == ["the first turn", "synthetic continuation"]

    assert adapter.started == []
    assert adapter.completed == []


@pytest.mark.asyncio
async def test_raw_envelope_only_followup_is_acknowledged(monkeypatch, tmp_path):
    """Signal never sets message_id — its hook keys off the raw envelope
    (sender + timestamp_ms) — and Discord's reads raw_message. An event
    carrying only a raw envelope is still a real inbound message."""
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="signal-shaped follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id=None,
        raw_message={"sender": "+15550100", "timestamp_ms": 1700000000000},
    )

    await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-raw",
        session_key=SESSION_KEY,
    )

    assert adapter.started == [None]
    assert adapter.completed == [(None, ProcessingOutcome.SUCCESS)]


class CompleteOnlyAdapter(HookRecordingAdapter):
    """Google Chat and webhook implement on_processing_complete WITHOUT
    on_processing_start; theirs is end-of-cycle teardown (reap the typing
    card / end the delivery session), not a reaction."""

    on_processing_start = BasePlatformAdapter.on_processing_start


@pytest.mark.asyncio
async def test_complete_only_adapter_is_left_alone(monkeypatch, tmp_path):
    """We bracket, so both halves must belong to us. An adapter that only
    implements the completion half must not be handed a completion here: at
    this point the follow-up's reply has not been delivered yet, so its
    teardown would fire against a live turn."""
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = CompleteOnlyAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="queued-3",
    )

    result = await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-complete-only",
        session_key=SESSION_KEY,
    )

    assert result["final_response"] == "done-2"
    assert adapter.completed == []
