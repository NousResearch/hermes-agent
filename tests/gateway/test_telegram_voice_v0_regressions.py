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

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
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


def _pending_audio_runner():
    """Build an opted in pending audio runner."""
    runner = _runner()
    runner.config = GatewayConfig(
        stt_enabled=True,
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                extra={"transcribe_audio_attachment_channels": ["12345"]}
            )
        },
    )
    return runner


@pytest.mark.asyncio
async def test_pending_audio_waiters_share_task_after_one_waiter_is_cancelled():
    runner = _pending_audio_runner()
    source = _source()
    event = MessageEvent(
        text="current caption",
        message_type=MessageType.AUDIO,
        source=source,
        media_urls=["/tmp/shared.mp3"],
        media_types=["audio/mpeg"],
    )
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def transcribe(path, model, context):
        """Hold one shared transcription task."""
        calls.append((path, model, context))
        entered.set()
        assert release.wait(timeout=5.0)
        return {"success": True, "transcript": "shared"}

    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        first_waiter = asyncio.create_task(
            runner._transcribe_pending_audio_event_once(event, "first caller")
        )
        assert await asyncio.to_thread(entered.wait, 5.0)
        cancelled_waiter = asyncio.create_task(
            runner._transcribe_pending_audio_event_once(event, "cancelled caller")
        )
        await asyncio.sleep(0)
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter

        shared_waiter = asyncio.create_task(
            runner._transcribe_pending_audio_event_once(event, "shared caller")
        )
        await asyncio.sleep(0)
        assert calls == [("/tmp/shared.mp3", None, "gateway")]
        release.set()
        first_result, shared_result = await asyncio.gather(first_waiter, shared_waiter)
        cached_result = await runner._transcribe_pending_audio_event_once(
            event, "cached caller"
        )
        prepared = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[]
        )

    prefix = '"shared"'
    assert first_result[0] == f"{prefix}\n\nfirst caller"
    assert shared_result[0] == f"{prefix}\n\nshared caller"
    assert cached_result[0] == f"{prefix}\n\ncached caller"
    assert event._gateway_pending_stt_text == prefix
    assert prepared == f"{prefix}\n\ncurrent caption"
    assert "audio file attachment" not in prepared
    assert calls == [("/tmp/shared.mp3", None, "gateway")]


@pytest.mark.asyncio
async def test_pending_audio_empty_cached_prefix_preserves_caller_text():
    runner = _pending_audio_runner()
    event = MessageEvent(
        text="event caption",
        message_type=MessageType.AUDIO,
        source=_source(),
        media_urls=["/tmp/empty-prefix.mp3"],
        media_types=["audio/mpeg"],
    )
    event._gateway_pending_stt_text = ""
    event._gateway_pending_stt_transcripts = []

    result = await runner._transcribe_pending_audio_event_once(
        event, "caller text"
    )

    assert result == ("caller text", [])


@pytest.mark.asyncio
async def test_pending_audio_replacement_task_keeps_cleanup_ownership():
    runner = _pending_audio_runner()
    source = _source()
    event = MessageEvent(
        text="current caption",
        message_type=MessageType.AUDIO,
        source=source,
        media_urls=["/tmp/old.mp3"],
        media_types=["audio/mpeg"],
    )
    old_entered = threading.Event()
    old_release = threading.Event()
    replacement_entered = threading.Event()
    replacement_release = threading.Event()
    calls = []

    def transcribe(path, model, context):
        """Hold old and replacement snapshots independently."""
        calls.append((path, model, context))
        if len(calls) == 1:
            old_entered.set()
            assert old_release.wait(timeout=5.0)
            transcript = "old"
        elif len(calls) == 2:
            replacement_entered.set()
            assert replacement_release.wait(timeout=5.0)
            transcript = "new old"
        else:
            transcript = "new attachment"
        return {"success": True, "transcript": transcript}

    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        old_waiter = asyncio.create_task(
            runner._transcribe_pending_audio_event_once(event, "old caller")
        )
        assert await asyncio.to_thread(old_entered.wait, 5.0)
        event.media_urls.append("/tmp/new.mp3")
        event.media_types.append("audio/mpeg")
        replacement_waiter = asyncio.create_task(
            runner._transcribe_pending_audio_event_once(event, "replacement caller")
        )
        assert await asyncio.to_thread(replacement_entered.wait, 5.0)
        replacement_task = event._gateway_pending_stt_task
        old_release.set()
        old_result = await old_waiter

        assert event._gateway_pending_stt_task is replacement_task
        replacement_release.set()
        replacement_result = await replacement_waiter

    prefix = '"new old"\n\n"new attachment"'
    assert old_result[0] == '"old"\n\nold caller'
    assert replacement_result[0] == f"{prefix}\n\nreplacement caller"
    assert event._gateway_pending_stt_text == prefix
    assert not hasattr(event, "_gateway_pending_stt_task")
    assert not hasattr(event, "_gateway_pending_stt_task_paths")
    assert calls == [
        ("/tmp/old.mp3", None, "gateway"),
        ("/tmp/old.mp3", None, "gateway"),
        ("/tmp/new.mp3", None, "gateway"),
    ]


@pytest.mark.asyncio
async def test_pending_audio_stale_snapshot_is_not_published_and_retries():
    runner = _pending_audio_runner()
    event = MessageEvent(
        text="stale caption",
        message_type=MessageType.AUDIO,
        source=_source(),
        media_urls=["/tmp/stale.mp3"],
        media_types=["audio/mpeg"],
    )
    stale_entered = threading.Event()
    stale_release = threading.Event()
    stale_calls = []

    def transcribe_stale(path, model, context):
        """Hold the original snapshot until its media changes."""
        stale_calls.append((path, model, context))
        if len(stale_calls) == 1:
            stale_entered.set()
            assert stale_release.wait(timeout=5.0)
        return {"success": True, "transcript": path}

    with patch(
        "tools.transcription_tools.transcribe_audio", side_effect=transcribe_stale
    ):
        stale_waiter = asyncio.create_task(
            runner._transcribe_pending_audio_event_once(event, "stale caller")
        )
        assert await asyncio.to_thread(stale_entered.wait, 5.0)
        event.media_urls.append("/tmp/fresh.mp3")
        event.media_types.append("audio/mpeg")
        stale_release.set()
        stale_result = await stale_waiter
        assert not hasattr(event, "_gateway_pending_stt_text")
        retry_result = await runner._transcribe_pending_audio_event_once(
            event, "retry caller"
        )

    assert stale_result[0] == '"/tmp/stale.mp3"\n\nstale caller'
    assert retry_result[0] == (
        '"/tmp/stale.mp3"\n\n"/tmp/fresh.mp3"\n\nretry caller'
    )
    assert stale_calls == [
        ("/tmp/stale.mp3", None, "gateway"),
        ("/tmp/stale.mp3", None, "gateway"),
        ("/tmp/fresh.mp3", None, "gateway"),
    ]


@pytest.mark.asyncio
async def test_media_only_audio_preserves_pending_clarify_for_normal_routing():
    from tools import clarify_gateway

    runner = _runner()
    runner.config = GatewayConfig(
        stt_enabled=True,
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                extra={"transcribe_audio_attachment_channels": []}
            )
        },
    )
    runner._prepare_clarify_reply_text = AsyncMock(return_value="")
    runner._queue_or_replace_pending_event = MagicMock()
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.AUDIO,
        source=source,
        media_urls=["/tmp/clarify.mp3"],
        media_types=["audio/mpeg"],
    )
    session_key = "telegram:dm:12345"
    entry = clarify_gateway.register("audio-clarify", session_key, "What next", None)
    try:
        result = await runner._hm_clarify_reply(event, source, session_key)

        assert result == ""
        assert entry.event.is_set() is False
        runner._queue_or_replace_pending_event.assert_called_once_with(session_key, event)
    finally:
        clarify_gateway.clear_session(session_key)


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
