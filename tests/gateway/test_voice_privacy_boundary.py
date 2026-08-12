"""End-to-end synthetic-canary regressions for gateway voice privacy."""

import sys
import threading
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionEntry, SessionSource


SESSION_KEY = "agent:main:telegram:dm:voice-user"
SESSION_ID = "sess-voice-privacy"


class _QueuedVoiceAgent:
    messages: list[str] = []

    def __init__(self, **_kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).messages.append(message)
        return {
            "final_response": f"synthetic response {len(type(self).messages)}",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


class _QueueAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="synthetic-token"),
            Platform.TELEGRAM,
        )
        self.sent: list[str] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="synthetic-message")


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="voice-user",
        chat_type="dm",
        user_id="voice-user",
    )


def _voice_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.VOICE,
        source=_source(),
        message_id="voice-msg-1",
    )


def _bootstrap_runner(monkeypatch, tmp_path) -> gateway_run.GatewayRunner:
    """Run the real gateway handler while stubbing external agent/storage leaves."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    now = datetime.now()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY,
        session_id=SESSION_ID,
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "synthetic-test-key"},
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _queued_run_agent_runner(adapter: _QueueAdapter) -> gateway_run.GatewayRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._draining = False
    runner._gateway_loop = None
    runner.session_store = SimpleNamespace(
        _entries={},
        _save=lambda: None,
        get_model_override=lambda _session_key: None,
    )
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=True,
        streaming=None,
        multiplex_profiles=False,
    )
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_or_create_gateway_honcho = lambda _session_key: (None, None)
    runner._update_runtime_status = lambda _status: None
    return runner


@pytest.mark.asyncio
async def test_normal_voice_handler_logs_only_bounded_metadata(
    monkeypatch, tmp_path, caplog
):
    transcript_canary = "SYNTHETIC_NORMAL_VOICE_TRANSCRIPT_CANARY_7f14"
    reply_canary = "SYNTHETIC_VOICE_DERIVED_REPLY_CANARY_1d92"
    runner = _bootstrap_runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": reply_canary,
            "messages": [
                {"role": "user", "content": transcript_canary},
                {"role": "assistant", "content": reply_canary},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    caplog.set_level("DEBUG", logger="gateway.run")

    await runner._handle_message_with_agent(
        _voice_event(transcript_canary), _source(), SESSION_KEY, 1
    )

    assert transcript_canary not in caplog.text
    assert reply_canary not in caplog.text
    assert "message_type=voice" in caplog.text
    assert f"text_chars={len(transcript_canary)}" in caplog.text


@pytest.mark.asyncio
async def test_ordinary_text_handler_keeps_existing_preview_logging(
    monkeypatch, tmp_path, caplog
):
    text_canary = "SYNTHETIC_ORDINARY_TEXT_PREVIEW_CANARY_02b1"
    runner = _bootstrap_runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "synthetic text response",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text=text_canary,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="text-msg-1",
    )
    caplog.set_level("INFO", logger="gateway.run")

    await runner._handle_message_with_agent(event, _source(), SESSION_KEY, 1)

    assert text_canary in caplog.text
    assert "inbound message:" in caplog.text


@pytest.mark.asyncio
async def test_queued_voice_preview_logs_only_bounded_metadata(
    monkeypatch, tmp_path, caplog
):
    transcript_canary = "VOICE_QUEUE_CANARY_a38c"
    fake_run_agent = types.ModuleType("run_agent")
    _QueuedVoiceAgent.messages = []
    fake_run_agent.AIAgent = _QueuedVoiceAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = _QueueAdapter()
    runner = _queued_run_agent_runner(adapter)
    pending_event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=["/synthetic/private-voice.ogg"],
        media_types=["audio/ogg"],
        message_id="queued-voice-1",
    )
    adapter._pending_messages[SESSION_KEY] = pending_event
    runner._transcribe_and_echo_pending_voice = AsyncMock(
        return_value=(transcript_canary, [transcript_canary])
    )
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value=transcript_canary
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "synthetic-test-key"},
    )
    caplog.set_level("DEBUG", logger="gateway.run")

    result = await runner._run_agent(
        message="synthetic initial text turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id=SESSION_ID,
        session_key=SESSION_KEY,
    )

    assert result["final_response"] == "synthetic response 2"
    assert _QueuedVoiceAgent.messages[-1] == transcript_canary
    assert transcript_canary not in caplog.text
    assert "stage=queued" in caplog.text
    assert "stage=pending" in caplog.text
    assert f"text_chars={len(transcript_canary)}" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_metadata"),
    [
        (
            {
                "success": False,
                "transcript": "",
                "provider": "synthetic-provider",
                "error_code": "rate_limited",
                "error": {
                    "message": "STT_BODY_CANARY_STRUCTURED_41be",
                    "body": "STT_BODY_CANARY_STRUCTURED_41be",
                },
            },
            "provider=synthetic-provider stage=transcribe error_code=rate_limited",
        ),
        (
            {
                "success": False,
                "transcript": "",
                "provider": "legacy-provider",
                "error": "STT_BODY_CANARY_LEGACY_b75a",
            },
            "provider=legacy-provider stage=transcribe error_code=provider_failure",
        ),
        (
            RuntimeError("STT_BODY_CANARY_EXCEPTION_90d3"),
            "stage=transcribe error_code=stt_exception exception_type=RuntimeError",
        ),
    ],
)
async def test_gateway_stt_failures_expose_only_bounded_metadata(
    failure, expected_metadata, caplog
):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    audio_path = "/synthetic/AUDIO_PATH_CANARY_e42f.ogg"
    if isinstance(failure, Exception):
        transcribe = MagicMock(side_effect=failure)
        canary = str(failure)
    else:
        transcribe = MagicMock(return_value=failure)
        raw_error = failure["error"]
        canary = raw_error["message"] if isinstance(raw_error, dict) else raw_error
    local_fallback = MagicMock(
        return_value={
            "success": False,
            "transcript": "",
            "provider": "local",
            "error_code": "fallback_failed",
            "error": "synthetic local fallback failed",
        }
    )
    caplog.set_level("DEBUG", logger="gateway.run")

    with (
        patch("tools.transcription_tools.transcribe_audio", transcribe),
        patch(
            "tools.transcription_tools.transcribe_audio_local_fallback",
            local_fallback,
        ),
    ):
        visible_marker, transcripts = await runner._enrich_message_with_transcription(
            "", [audio_path]
        )

    assert transcripts == []
    assert canary not in caplog.text
    assert canary not in visible_marker
    assert audio_path not in caplog.text
    assert audio_path not in visible_marker
    assert expected_metadata in caplog.text
    assert "[voice message could not be transcribed automatically; error_code=" in visible_marker


@pytest.mark.asyncio
async def test_pending_voice_transcription_exception_logs_only_metadata(caplog):
    exception_canary = "PENDING_STT_EXCEPTION_CANARY_c83d"
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._transcribe_pending_audio_event_once = AsyncMock(
        side_effect=RuntimeError(exception_canary)
    )
    event = MessageEvent(
        text="safe caption",
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=["/synthetic/pending.ogg"],
        media_types=["audio/ogg"],
    )
    caplog.set_level("DEBUG", logger="gateway.run")

    text, transcripts = await runner._transcribe_and_echo_pending_voice(
        event,
        None,
        _source(),
        "safe caption",
        log_context="Voice-drain",
    )

    assert text == "safe caption"
    assert transcripts == []
    assert exception_canary not in caplog.text
    assert (
        "stage=pending error_code=stt_exception exception_type=RuntimeError"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_pending_voice_echo_exception_logs_only_metadata(caplog):
    exception_canary = "VOICE_ECHO_EXCEPTION_CANARY_6c21"
    transcript_canary = "VOICE_ECHO_TRANSCRIPT_CANARY_21d9"
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._should_echo_stt_transcripts = lambda: True
    adapter = SimpleNamespace(
        send=AsyncMock(side_effect=RuntimeError(exception_canary))
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=_source(),
    )
    caplog.set_level("DEBUG", logger="gateway.run")

    await runner._echo_pending_stt_transcripts_once(
        event,
        adapter,
        _source(),
        [transcript_canary],
        log_context="Voice-drain",
    )

    assert exception_canary not in caplog.text
    assert transcript_canary not in caplog.text
    assert (
        "stage=echo error_code=delivery_exception exception_type=RuntimeError"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_fresh_voice_echo_exception_logs_only_metadata(caplog):
    exception_canary = "FRESH_VOICE_ECHO_EXCEPTION_CANARY_48ca"
    transcript_canary = "FRESH_VOICE_TRANSCRIPT_CANARY_d112"
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._consume_pending_native_image_paths = lambda _session_key: []
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=(f'"{transcript_canary}"', [transcript_canary])
    )
    runner._should_echo_stt_transcripts = lambda: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._thread_metadata_for_source = lambda _source, _reply=None: None
    runner.adapters = {
        Platform.TELEGRAM: SimpleNamespace(
            send=AsyncMock(side_effect=RuntimeError(exception_canary))
        )
    }
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=["/synthetic/fresh-voice.ogg"],
        media_types=["audio/ogg"],
    )
    caplog.set_level("DEBUG", logger="gateway.run")

    prepared = await runner._prepare_inbound_message_text(
        event=event,
        source=_source(),
        history=[],
        session_key=SESSION_KEY,
    )

    assert transcript_canary in prepared
    assert transcript_canary not in caplog.text
    assert exception_canary not in caplog.text
    assert (
        "stage=echo error_code=delivery_exception exception_type=RuntimeError"
        in caplog.text
    )
