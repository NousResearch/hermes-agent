"""Regression tests for voice messages arriving while a gateway session is busy."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakeAgent:
    def __init__(self):
        self.steered = []

    def steer(self, text):
        self.steered.append(text)
        return True

    def get_activity_summary(self):
        return {
            "api_call_count": 1,
            "max_iterations": 10,
            "current_tool": "terminal",
        }


class _FakeAdapter:
    def __init__(self):
        self.sent = []

    async def _send_with_retry(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SimpleNamespace(success=True)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return await self._send_with_retry(
            chat_id,
            content,
            reply_to=reply_to,
            metadata=metadata,
        )

    def extract_media(self, response):
        return [], response

    def extract_images(self, response):
        return [], response


@pytest.mark.asyncio
async def test_busy_steer_transcribes_voice_before_queue_fallback(monkeypatch):
    """In steer mode, a captionless voice note should steer its transcript.

    Before this regression guard, _handle_active_session_busy_message looked only
    at event.text. Telegram voice notes have no text until STT runs, so steer mode
    silently degraded to queue mode even though STT was enabled and an agent was
    available to accept steer input.
    """
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5767139348",
        chat_type="dm",
        user_id="5767139348",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/busy-steer.ogg"],
        media_types=["audio/ogg"],
        message_id="voice-1",
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True)
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._draining = False
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_input_mode = "steer"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._agent_has_active_subagents = lambda _agent: False
    runner._is_user_authorized = lambda _source: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    queued = []
    runner._queue_or_replace_pending_event = lambda _session_key, _event: queued.append(_event)

    agent = _FakeAgent()
    session_key = "agent:main:telegram:dm:5767139348"
    runner._running_agents[session_key] = agent
    runner._running_agents_ts[session_key] = 123.0

    import agent.onboarding as onboarding
    monkeypatch.setattr(onboarding, "is_seen", lambda *_args, **_kwargs: True)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "nouvelle idée à traiter directement",
            "provider": "whisper",
        },
    ) as transcribe:
        handled = await runner._handle_active_session_busy_message(event, session_key)

    assert handled is True
    transcribe.assert_called_once_with("/tmp/busy-steer.ogg")
    assert agent.steered == ['"nouvelle idée à traiter directement"']
    assert queued == []
    assert runner.adapters[Platform.TELEGRAM].sent
    assert "Steered into current run" in runner.adapters[Platform.TELEGRAM].sent[0]["content"]


@pytest.mark.asyncio
async def test_busy_steer_fallback_captures_background_instead_of_queue(monkeypatch):
    """If a busy message cannot be truly steered, front-desk captures it.

    The previous steer fallback stored the message in the session pending queue,
    so Pierre saw the idea as blocked behind the current long run.  Front-desk
    mode should ACK immediately and launch a separate background session instead.
    """
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5767139348",
        chat_type="dm",
        user_id="5767139348",
    )
    event = MessageEvent(
        text="note indépendante pendant le long run",
        message_type=MessageType.TEXT,
        source=source,
        message_id="text-1",
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True)
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._background_tasks = set()
    runner._draining = False
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_input_mode = "steer"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._agent_has_active_subagents = lambda _agent: False
    runner._is_user_authorized = lambda _source: True
    runner._reply_anchor_for_event = lambda _event: "text-1"
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._persist_frontdesk_capture = lambda **_kwargs: None
    queued = []
    runner._queue_or_replace_pending_event = lambda _session_key, _event: queued.append(_event)

    background_calls = []

    async def fake_background_task(prompt, source, task_id, **kwargs):
        background_calls.append({
            "prompt": prompt,
            "source": source,
            "task_id": task_id,
            "kwargs": kwargs,
        })

    runner._run_background_task = fake_background_task
    # Agent exists but cannot accept steer(), which used to force queue fallback.
    runner._running_agents["agent:main:telegram:dm:5767139348"] = SimpleNamespace()
    runner._running_agents_ts["agent:main:telegram:dm:5767139348"] = 123.0

    import agent.onboarding as onboarding
    monkeypatch.setattr(onboarding, "is_seen", lambda *_args, **_kwargs: True)

    handled = await runner._handle_active_session_busy_message(
        event,
        "agent:main:telegram:dm:5767139348",
    )
    await asyncio.sleep(0)

    assert handled is True
    assert queued == []
    assert len(background_calls) == 1
    assert "note indépendante pendant le long run" in background_calls[0]["prompt"]
    assert background_calls[0]["kwargs"]["event_message_id"] == "text-1"
    assert runner.adapters[Platform.TELEGRAM].sent
    assert "background task" in runner.adapters[Platform.TELEGRAM].sent[0]["content"]
    assert "Queued for the next turn" not in runner.adapters[Platform.TELEGRAM].sent[0]["content"]


@pytest.mark.asyncio
async def test_busy_frontdesk_mode_captures_text_without_steering_or_queueing(monkeypatch):
    """frontdesk mode turns busy follow-ups into isolated background work."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5767139348",
        chat_type="dm",
        user_id="5767139348",
    )
    event = MessageEvent(
        text="nouvelle idée pendant un chantier long",
        message_type=MessageType.TEXT,
        source=source,
        message_id="frontdesk-text-1",
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True)
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._background_tasks = set()
    runner._draining = False
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_input_mode = "frontdesk"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._agent_has_active_subagents = lambda _agent: False
    runner._is_user_authorized = lambda _source: True
    runner._reply_anchor_for_event = lambda _event: "frontdesk-text-1"
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._persist_frontdesk_capture = lambda **_kwargs: None
    queued = []
    runner._queue_or_replace_pending_event = lambda _session_key, _event: queued.append(_event)

    background_calls = []

    async def fake_background_task(prompt, source, task_id, **kwargs):
        background_calls.append({"prompt": prompt, "source": source, "task_id": task_id, "kwargs": kwargs})

    runner._run_background_task = fake_background_task
    agent = _FakeAgent()
    session_key = "agent:main:telegram:dm:5767139348"
    runner._running_agents[session_key] = agent
    runner._running_agents_ts[session_key] = 123.0

    import agent.onboarding as onboarding
    monkeypatch.setattr(onboarding, "is_seen", lambda *_args, **_kwargs: True)

    handled = await runner._handle_active_session_busy_message(event, session_key)
    await asyncio.sleep(0)

    assert handled is True
    assert agent.steered == []
    assert queued == []
    assert len(background_calls) == 1
    assert "nouvelle idée pendant un chantier long" in background_calls[0]["prompt"]
    assert background_calls[0]["kwargs"]["event_message_id"] == "frontdesk-text-1"
    assert "background task" in runner.adapters[Platform.TELEGRAM].sent[0]["content"]
    assert "Queued for the next turn" not in runner.adapters[Platform.TELEGRAM].sent[0]["content"]
    assert "Steered into current run" not in runner.adapters[Platform.TELEGRAM].sent[0]["content"]


@pytest.mark.asyncio
async def test_busy_frontdesk_mode_passes_voice_and_image_media_to_background_task(monkeypatch):
    """frontdesk mode must not strand media in the same-session pending queue."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5767139348",
        chat_type="dm",
        user_id="5767139348",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/frontdesk-voice.ogg", "/tmp/frontdesk-image.png"],
        media_types=["audio/ogg", "image/png"],
        message_id="frontdesk-media-1",
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True)
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._background_tasks = set()
    runner._draining = False
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_input_mode = "frontdesk"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._agent_has_active_subagents = lambda _agent: False
    runner._is_user_authorized = lambda _source: True
    runner._reply_anchor_for_event = lambda _event: "frontdesk-media-1"
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._persist_frontdesk_capture = lambda **_kwargs: None
    queued = []
    runner._queue_or_replace_pending_event = lambda _session_key, _event: queued.append(_event)

    background_calls = []

    async def fake_background_task(prompt, source, task_id, **kwargs):
        background_calls.append({"prompt": prompt, "source": source, "task_id": task_id, "kwargs": kwargs})

    runner._run_background_task = fake_background_task
    session_key = "agent:main:telegram:dm:5767139348"
    runner._running_agents[session_key] = _FakeAgent()
    runner._running_agents_ts[session_key] = 123.0

    import agent.onboarding as onboarding
    monkeypatch.setattr(onboarding, "is_seen", lambda *_args, **_kwargs: True)

    handled = await runner._handle_active_session_busy_message(event, session_key)
    await asyncio.sleep(0)

    assert handled is True
    assert queued == []
    assert len(background_calls) == 1
    assert background_calls[0]["kwargs"]["media_urls"] == [
        "/tmp/frontdesk-voice.ogg",
        "/tmp/frontdesk-image.png",
    ]
    assert background_calls[0]["kwargs"]["media_types"] == ["audio/ogg", "image/png"]
    assert "background task" in runner.adapters[Platform.TELEGRAM].sent[0]["content"]


@pytest.mark.asyncio
async def test_leftover_steer_is_captured_as_frontdesk_background(monkeypatch):
    """A steer that arrives after the final tool batch must not recurse/queue.

    This covers the final-stream tail: the current run is allowed to finish, and
    the leftover steer becomes an independent background task instead of another
    recursive same-session turn.
    """
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5767139348",
        chat_type="dm",
        user_id="5767139348",
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._background_tasks = set()
    runner._reply_anchor_for_event = lambda _event: None
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._persist_frontdesk_capture = lambda **_kwargs: None

    background_calls = []

    async def fake_background_task(prompt, source, task_id, **kwargs):
        background_calls.append({"prompt": prompt, "source": source, "task_id": task_id, "kwargs": kwargs})

    runner._run_background_task = fake_background_task

    captured = await runner._capture_leftover_steer_as_frontdesk_background(
        source=source,
        steer_text="idée arrivée pendant la finalisation",
        session_key="agent:main:telegram:dm:5767139348",
        adapter=runner.adapters[Platform.TELEGRAM],
        metadata={},
    )
    await asyncio.sleep(0)

    assert captured is True
    assert len(background_calls) == 1
    assert "idée arrivée pendant la finalisation" in background_calls[0]["prompt"]
    assert runner.adapters[Platform.TELEGRAM].sent
    assert "after the last tool call" in runner.adapters[Platform.TELEGRAM].sent[0]["content"]


@pytest.mark.asyncio
async def test_frontdesk_background_audio_echoes_transcript_before_result(monkeypatch):
    """Voice sent to a frontdesk background worker must still be visible to Pierre.

    Frontdesk mode schedules the worker before the normal inbound STT echo path.
    The background enrichment step therefore owns the transcript echo.
    """
    import gateway.run as gateway_run

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5767139348",
        chat_type="dm",
        user_id="5767139348",
    )
    adapter = _FakeAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._service_tier = None
    runner._session_db = SimpleNamespace(_db=None)
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {"thread_id": "test"}
    runner._resolve_session_agent_runtime = lambda **_kwargs: ("test-model", {"api_key": "ok"})
    runner._resolve_session_reasoning_config = lambda **_kwargs: None
    runner._load_service_tier = lambda: None
    runner._resolve_turn_agent_config = lambda prompt, model, runtime: {
        "model": model,
        "runtime": runtime,
        "request_overrides": None,
    }

    async def fake_executor(_fn):
        return {"final_response": "fait"}

    runner._run_in_executor_with_context = fake_executor
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"agent": {}, "tools": {}, "display": {}},
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "texte brut du vocal frontdesk",
            "provider": "whisper",
        },
    ) as transcribe:
        await runner._run_background_task(
            prompt="Frontdesk capture:\nCaptured user message:\n[media]",
            source=source,
            task_id="fd_voice_echo",
            event_message_id="voice-42",
            media_urls=["/tmp/frontdesk-voice.ogg"],
            media_types=["audio/ogg"],
        )

    transcribe.assert_called_once_with("/tmp/frontdesk-voice.ogg")
    contents = [item["content"] for item in adapter.sent]
    assert '🎙️ "texte brut du vocal frontdesk"' in contents
    assert any("Background task complete" in item for item in contents)
