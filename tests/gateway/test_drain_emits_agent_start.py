"""Regression: the interrupt/drain follow-up path must emit ``agent:start``.

In ``busy_input_mode: interrupt`` a message typed while the agent is busy
interrupts the run, is queued in ``_pending_messages``, and is later promoted
via the drain path inside ``GatewayRunner._run_agent`` (the recursive
follow-up turn).  The MAIN dispatch emits ``agent:start`` before running the
agent, but the drain/follow-up path historically did NOT — so every hook
listening on ``agent:start`` (SessionStart-style integrations, activity
loggers, visualizers) silently missed interrupt/drain follow-up turns.

These tests drive ``_run_agent`` directly with a frozen pending event (the
same seam used by ``test_run_progress_interrupt.py``) and assert:

  * a plain-text follow-up emits ``agent:start`` once, with a payload shaped
    like the main-path one (platform, user_id, chat_id, session_id, message);
  * a voice follow-up emits the FINAL transcribed text — not the raw audio
    placeholder — because the payload is built from ``next_message`` (the
    output of ``_prepare_inbound_message_text``), not the queued placeholder;
  * the message is truncated to 500 chars, mirroring the main path;
  * NO ``agent:start`` is emitted when the follow-up is discarded before it
    reaches ``_run_agent`` (transcription yields ``None``; a stale /goal
    continuation is dropped).
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
    SendResult,
)
from gateway.session import SessionSource

SESSION_ID = "sess-drain"
SESSION_KEY = "agent:main:telegram:dm:9001"


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="x")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _RecordingHooks:
    """Records every emit so tests can assert agent:start (or its absence)."""

    def __init__(self):
        self.calls = []
        self.loaded_hooks = False

    async def emit(self, event_type, context=None):
        self.calls.append((event_type, context))

    def agent_start_payloads(self):
        return [ctx for (etype, ctx) in self.calls if etype == "agent:start"]


class _DrainAgent:
    """Fake AIAgent whose run returns an interrupted result with a queued
    follow-up still pending — exactly what the interrupt drain path consumes."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupted = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        return {
            "final_response": "partial answer",
            "messages": [{"role": "user", "content": "original turn"}],
            "api_calls": 1,
            "interrupted": True,
        }


def _make_runner(adapter, hooks):
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
    runner.hooks = hooks
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _source(user_id="userA"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="9001",
        chat_type="dm",
        user_id=user_id,
    )


async def _drive_drain(
    monkeypatch,
    tmp_path,
    pending_event,
    *,
    prepared_text,
    goal_active=None,
    interrupt_depth=0,
):
    """Run _run_agent with one frozen pending event queued for the drain.

    ``prepared_text`` is what the (stubbed) preprocessing pipeline yields for
    the pending event — i.e. the FINAL text the follow-up turn runs on (a
    transcription for a voice note).  ``None`` means the follow-up is dropped.
    """
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _DrainAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = _CaptureAdapter()
    hooks = _RecordingHooks()
    runner = _make_runner(adapter, hooks)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )

    # Stub the preprocessing pipeline so the drain's transcription step is
    # deterministic and isolated.  This stands in for the real STT/vision
    # pipeline; the contract under test is that the emit uses ITS result.
    async def _fake_prepare(*, event, source, history):
        return prepared_text

    monkeypatch.setattr(runner, "_prepare_inbound_message_text", _fake_prepare)

    if goal_active is not None:
        monkeypatch.setattr(
            runner, "_goal_still_active_for_session", lambda sid: goal_active
        )

    # Queue the follow-up exactly as an interrupt would have.
    adapter._pending_messages[SESSION_KEY] = pending_event

    await runner._run_agent(
        message="original turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id=SESSION_ID,
        session_key=SESSION_KEY,
        _interrupt_depth=interrupt_depth,
    )
    return hooks


@pytest.mark.asyncio
async def test_text_followup_emits_agent_start(monkeypatch, tmp_path):
    """A plain-text drained follow-up emits one agent:start with a main-path
    shaped payload built from the FOLLOW-UP's source."""
    followup = MessageEvent(
        text="follow up text",
        message_type=MessageType.TEXT,
        source=_source(user_id="userB"),
        message_id="m2",
    )
    hooks = await _drive_drain(
        monkeypatch, tmp_path, followup, prepared_text="follow up text"
    )

    payloads = hooks.agent_start_payloads()
    assert len(payloads) == 1, (
        "drain follow-up must emit agent:start exactly once "
        f"(emitted {len(payloads)})"
    )
    assert payloads[0] == {
        "platform": "telegram",
        "user_id": "userB",
        "chat_id": "9001",
        "session_id": SESSION_ID,
        "message": "follow up text",
        "trigger": "interrupt",
        "interrupt_depth": 1,
    }


@pytest.mark.asyncio
async def test_voice_followup_emits_transcribed_text_not_audio(monkeypatch, tmp_path):
    """A voice drained follow-up emits the transcribed text, never the raw
    audio path / media placeholder."""
    followup = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=["/tmp/voice-xyz.ogg"],
        media_types=["audio/ogg"],
        message_id="m3",
    )
    transcript = "remind me to water the plants at six"
    hooks = await _drive_drain(
        monkeypatch, tmp_path, followup, prepared_text=transcript
    )

    payloads = hooks.agent_start_payloads()
    assert len(payloads) == 1
    assert payloads[0]["message"] == transcript
    assert "/tmp/voice-xyz.ogg" not in payloads[0]["message"]
    assert "User sent audio" not in payloads[0]["message"]


@pytest.mark.asyncio
async def test_agent_start_message_truncated_to_500(monkeypatch, tmp_path):
    """The payload message is capped at 500 chars, like the main path."""
    followup = MessageEvent(
        text="x" * 600,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m4",
    )
    hooks = await _drive_drain(
        monkeypatch, tmp_path, followup, prepared_text="x" * 600
    )

    payloads = hooks.agent_start_payloads()
    assert len(payloads) == 1
    assert len(payloads[0]["message"]) == 500


@pytest.mark.asyncio
async def test_no_emit_when_followup_text_is_none(monkeypatch, tmp_path):
    """If preprocessing yields None the follow-up never reaches _run_agent —
    no agent:start may fire."""
    followup = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=["/tmp/silent.ogg"],
        media_types=["audio/ogg"],
        message_id="m5",
    )
    hooks = await _drive_drain(
        monkeypatch, tmp_path, followup, prepared_text=None
    )

    assert hooks.agent_start_payloads() == []


@pytest.mark.asyncio
async def test_no_emit_when_goal_continuation_inactive(monkeypatch, tmp_path):
    """A stale /goal continuation is discarded before _run_agent — no
    agent:start may fire."""
    followup = MessageEvent(
        text="[Continuing toward your standing goal]\nGoal: ship the thing",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m6",
    )
    hooks = await _drive_drain(
        monkeypatch,
        tmp_path,
        followup,
        prepared_text="should never be used",
        goal_active=False,
    )

    assert hooks.agent_start_payloads() == []


@pytest.mark.asyncio
async def test_followup_carries_interrupt_trigger_and_depth(monkeypatch, tmp_path):
    """A first-level drained follow-up is tagged trigger="interrupt", depth 1."""
    followup = MessageEvent(
        text="follow up text",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m7",
    )
    hooks = await _drive_drain(
        monkeypatch, tmp_path, followup, prepared_text="follow up text"
    )

    payloads = hooks.agent_start_payloads()
    assert len(payloads) == 1
    assert payloads[0]["trigger"] == "interrupt"
    assert payloads[0]["interrupt_depth"] == 1


@pytest.mark.asyncio
async def test_nested_interrupt_increments_depth(monkeypatch, tmp_path):
    """An interrupt-of-an-interrupt reports depth _interrupt_depth + 1.

    Driving ``_run_agent`` with ``_interrupt_depth=1`` (a turn that is itself a
    follow-up) makes its drained follow-up the second level — the emit must use
    ``_interrupt_depth + 1`` == 2, not a hard-coded 1.
    """
    followup = MessageEvent(
        text="deeper follow up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m8",
    )
    hooks = await _drive_drain(
        monkeypatch,
        tmp_path,
        followup,
        prepared_text="deeper follow up",
        interrupt_depth=1,
    )

    payloads = hooks.agent_start_payloads()
    assert len(payloads) == 1
    assert payloads[0]["trigger"] == "interrupt"
    assert payloads[0]["interrupt_depth"] == 2
