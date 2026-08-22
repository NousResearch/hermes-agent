"""Regression tests for session-aware native audio routing in the gateway."""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="1",
        chat_type="dm",
    )


def _bare_runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "google/gemini-test"
    runner._base_url = None
    runner._has_setup_skill = lambda: False
    return runner


def test_audio_routing_uses_native_only_for_audio_capable_model(monkeypatch):
    runner = _bare_runner()
    media_routing = importlib.import_module("agent.media_routing")

    monkeypatch.setattr(
        media_routing,
        "supported_input_modalities",
        lambda provider, model: {"text", "audio"},
    )
    assert runner._decide_audio_input_mode(
        provider="openrouter",
        model="google/gemini-test",
        user_config={},
    ) == "native"

    monkeypatch.setattr(
        media_routing,
        "supported_input_modalities",
        lambda provider, model: {"text", "image"},
    )
    assert runner._decide_audio_input_mode(
        provider="openrouter",
        model="text-only-test",
        user_config={},
    ) == "stt"

    # User explicit config overrides
    assert runner._decide_audio_input_mode(
        provider="openrouter",
        model="text-only-test",
        user_config={"gateway": {"audio_mode": "native"}},
    ) == "native"

    assert runner._decide_audio_input_mode(
        provider="openrouter",
        model="google/gemini-test",
        user_config={"gateway": {"audio_mode": "stt"}},
    ) == "stt"


@pytest.mark.asyncio
async def test_voice_message_stages_native_audio_without_stt(tmp_path):
    runner = _bare_runner()
    runner._decide_audio_input_mode = lambda **_: "native"
    source = _source()
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"OggS-test-audio")
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=[str(audio_path)],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("native audio must not invoke STT"),
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result == ""
    attachments = runner._consume_pending_native_audio_attachments(
        build_session_key(source)
    )
    assert attachments == [{
        "path": str(audio_path),
        "mime_type": "audio/ogg",
        "modality": "audio",
    }]
    assert runner._consume_pending_native_audio_attachments(build_session_key(source)) == []


class _CaptureAgent:
    calls = []

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, **kwargs):
        type(self).calls.append((message, kwargs))
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _pipeline_runner():
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
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
        stt_enabled=True,
    )
    runner._model = "google/gemini-test"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_agent_pipeline_sends_input_audio_and_persists_compact_marker(
    monkeypatch,
    tmp_path,
):
    _CaptureAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CaptureAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***", "provider": "openrouter"},
    )

    runner = _pipeline_runner()
    source = _source()
    session_key = build_session_key(source)
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"OggS-test-audio")
    runner._session_state(session_key).persistent.native_audio_attachments = [{
        "path": str(audio_path),
        "mime_type": "audio/ogg",
        "modality": "audio",
    }]

    result = await runner._run_agent(
        message="",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-native-audio",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    assert len(_CaptureAgent.calls) == 1
    message, kwargs = _CaptureAgent.calls[0]
    assert isinstance(message, list)
    assert any(part.get("type") == "input_audio" for part in message)
    audio_part = next(part for part in message if part.get("type") == "input_audio")
    assert audio_part["input_audio"]["format"] == "ogg"
    assert kwargs["persist_user_message"] == "[Voice message attached natively]"
    assert runner._consume_pending_native_audio_attachments(session_key) == []
