import asyncio
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")


def _runner(adapter=None):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        stt_enabled=True,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._consume_pending_native_image_paths = lambda _key: []
    runner._session_key_for_source = lambda _source: "telegram:dm:12345"
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._reply_anchor_for_event = lambda _event: None
    return runner


class _PendingVoiceAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="voice-echo")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _PendingVoiceAgent:
    messages = []

    def __init__(self, **kwargs):
        self.tools = []
        self.model = "test-model"
        self.provider = "test-provider"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._interrupted = threading.Event()

    @property
    def is_interrupted(self):
        return self._interrupt_requested

    def interrupt(self, message):
        self._interrupt_requested = True
        self._interrupt_message = message
        self._interrupted.set()

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).messages.append(message)
        if len(type(self).messages) == 1:
            assert self._interrupted.wait(timeout=3), "pending voice interrupt was not delivered"
            return {
                "final_response": "interrupted",
                "messages": [],
                "api_calls": 1,
                "interrupted": True,
                "interrupt_message": self._interrupt_message,
            }
        return {
            "final_response": "follow-up complete",
            "messages": [],
            "api_calls": 1,
            "interrupted": False,
        }


def _run_agent_runner(adapter):
    runner = _runner(adapter)
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._queued_events = {}
    runner._draining = False
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._should_echo_stt_transcripts = lambda: True
    return runner


@pytest.mark.asyncio
async def test_pending_voice_interrupt_reuses_transcript_and_echo():
    adapter = SimpleNamespace(send=AsyncMock())
    runner = _runner(adapter)
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/telegram-voice.ogg"],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello once", "provider": "mock"},
    ) as mock_transcribe:
        interrupt_text, interrupt_transcripts = await runner._transcribe_pending_audio_event_once(
            event,
            event.text,
        )
        await runner._echo_pending_stt_transcripts_once(
            event,
            adapter,
            source,
            interrupt_transcripts,
        )

        drain_text, drain_transcripts = await runner._transcribe_pending_audio_event_once(
            event,
            event.text,
        )
        await runner._echo_pending_stt_transcripts_once(
            event,
            adapter,
            source,
            drain_transcripts,
        )

    assert interrupt_text == '"hello once"'
    assert drain_text == interrupt_text
    assert drain_transcripts == interrupt_transcripts == ["hello once"]
    mock_transcribe.assert_called_once_with("/tmp/telegram-voice.ogg", None, "gateway")
    adapter.send.assert_awaited_once_with(
        "12345",
        '🎙️ "hello once"',
        metadata=None,
    )


@pytest.mark.asyncio
async def test_monitor_to_drain_transcribes_and_echoes_pending_voice_once(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_GATEWAY_NOTIFY_INTERVAL", "0")
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=_PendingVoiceAgent))

    adapter = _PendingVoiceAdapter()
    runner = _run_agent_runner(adapter)
    source = _source()
    session_key = "telegram:dm:12345"
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/telegram-pending-voice.ogg"],
        media_types=["audio/ogg"],
    )
    adapter._pending_messages[session_key] = event
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._active_sessions[session_key].set()
    _PendingVoiceAgent.messages = []

    with (
        patch("gateway.run._hermes_home", tmp_path),
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "fake"}),
        patch(
            "tools.transcription_tools.transcribe_audio",
            return_value={"success": True, "transcript": "hello once", "provider": "mock"},
        ) as mock_transcribe,
    ):
        result = await runner._run_agent(
            message="initial turn",
            context_prompt="",
            history=[],
            source=source,
            session_id="pending-voice-session",
            session_key=session_key,
        )

    assert result["final_response"] == "follow-up complete"
    assert _PendingVoiceAgent.messages == ["initial turn", '"hello once"']
    mock_transcribe.assert_called_once_with("/tmp/telegram-pending-voice.ogg", None, "gateway")
    assert adapter.sent == [("12345", '🎙️ "hello once"', None)]


@pytest.mark.asyncio
async def test_queued_voice_transcribes_immediately_and_drain_reuses_it():
    """A voice note queued behind an active run is transcribed and echoed right away, stays
    queued, and the drain (even one racing the in-flight STT) reuses that single STT call."""
    adapter = _PendingVoiceAdapter()
    runner = _run_agent_runner(adapter)
    runner._queued_events = {}
    source = _source()
    session_key = "telegram:dm:12345"
    event = MessageEvent(
        text="", message_type=MessageType.VOICE, source=source,
        media_urls=["/tmp/queued-voice.ogg"], media_types=["audio/ogg"],
    )
    stt_started, release_stt = threading.Event(), threading.Event()

    def _transcribe(*_args):
        stt_started.set()
        assert release_stt.wait(timeout=3)
        return {"success": True, "transcript": "queued hello", "provider": "mock"}

    with patch("tools.transcription_tools.transcribe_audio", side_effect=_transcribe) as mock_transcribe:
        runner._queue_or_replace_pending_event(session_key, event)
        assert adapter._pending_messages[session_key] is event  # still queued, FIFO unchanged
        assert await asyncio.to_thread(stt_started.wait, 3)  # STT began before any drain

        # Current task finishes while STT is still running: the drain joins the in-flight call.
        drain = asyncio.create_task(
            runner._run_agent_drain_pending({"final_response": "done"}, adapter, source, session_key)
        )
        await asyncio.sleep(0.05)
        release_stt.set()
        pending_event, pending = await drain
        await event._gateway_pending_stt_prefetch

    assert pending_event is event
    assert pending == '"queued hello"'
    mock_transcribe.assert_called_once_with("/tmp/queued-voice.ogg", None, "gateway")
    assert [content for _chat, content, _meta in adapter.sent] == ['🎙️ "queued hello"']
    assert adapter.sent[0][2].get("_interim_send") is True  # mid-turn: must not seal a live stream


@pytest.mark.asyncio
async def test_text_merged_during_inflight_stt_is_not_lost():
    """A typed follow-up merged into the queued voice event while STT runs must survive into the
    transcript the drain reuses."""
    from gateway.platforms.base import merge_pending_message_event

    adapter = _PendingVoiceAdapter()
    runner = _run_agent_runner(adapter)
    runner._queued_events = {}
    source = _source()
    session_key = "telegram:dm:12345"
    event = MessageEvent(
        text="", message_type=MessageType.VOICE, source=source,
        media_urls=["/tmp/merge-voice.ogg"], media_types=["audio/ogg"],
    )
    stt_started, release_stt = threading.Event(), threading.Event()

    def _transcribe(*_args):
        stt_started.set()
        assert release_stt.wait(timeout=3)
        return {"success": True, "transcript": "spoken part", "provider": "mock"}

    with patch("tools.transcription_tools.transcribe_audio", side_effect=_transcribe):
        runner._queue_or_replace_pending_event(session_key, event)
        assert await asyncio.to_thread(stt_started.wait, 3)
        merge_pending_message_event(
            adapter._pending_messages, session_key,
            MessageEvent(text="typed part", message_type=MessageType.TEXT, source=source),
            merge_text=True,
        )
        release_stt.set()
        await event._gateway_pending_stt_prefetch
        _pending_event, pending = await runner._run_agent_drain_pending(
            {"final_response": "done"}, adapter, source, session_key,
        )

    assert "spoken part" in pending and "typed part" in pending


@pytest.mark.asyncio
async def test_telegram_video_size_gate_rejects_oversized_media_before_download():
    adapter = object.__new__(TelegramAdapter)
    adapter._max_doc_bytes = 1024
    adapter._should_process_message = lambda _message: True
    adapter._build_message_event = lambda _message, _type, update_id=None: SimpleNamespace(
        text="caption",
        media_urls=[],
        media_types=[],
    )
    adapter._apply_telegram_group_observe_attribution = lambda event: event

    handled = []

    async def handle_message(event):
        handled.append(event)

    adapter.handle_message = handle_message

    class OversizedVideo:
        file_size = 2048

        async def get_file(self):  # pragma: no cover - failure path assertion
            pytest.fail("oversized videos must not be downloaded")

    msg = SimpleNamespace(
        caption=None,
        sticker=None,
        photo=None,
        voice=None,
        audio=None,
        video=OversizedVideo(),
        document=None,
        media_group_id=None,
    )
    update = SimpleNamespace(message=msg, update_id=1)

    await TelegramAdapter._handle_media_message(adapter, update, SimpleNamespace())

    assert len(handled) == 1
    assert handled[0].media_urls == []
    assert handled[0].media_types == []
    assert "video file" in handled[0].text
    assert "exceeds" in handled[0].text


def _voice_event(source, urls):
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=list(urls),
        media_types=["audio/ogg"] * len(urls),
    )


