"""Tests for BasePlatformAdapter topic-aware session handling."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


class DummyTelegramAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []
        self.typing = []
        self.processing_hooks = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def on_processing_start(self, event: MessageEvent) -> None:
        self.processing_hooks.append(("start", event.message_id))

    async def on_processing_complete(self, event: MessageEvent, success: bool) -> None:
        self.processing_hooks.append(("complete", event.message_id, success))


class DummySignalAdapter(DummyTelegramAdapter):
    def __init__(self):
        super().__init__()
        self.platform = Platform.SIGNAL
        self.voices = []

    def get_default_reply_target(self, event: MessageEvent):
        return None

    def requires_reply_context_metadata(self) -> bool:
        return True

    def should_send_text_after_auto_tts(self, event: MessageEvent) -> bool:
        return False

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.voices.append(
            {
                "chat_id": chat_id,
                "audio_path": audio_path,
                "caption": caption,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="voice-1")


def _make_event(chat_id: str, thread_id: str, message_id: str = "1", message_type: MessageType = MessageType.TEXT) -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="group",
            thread_id=thread_id,
            user_id="+15550001111",
        ),
        message_id=message_id,
        message_type=message_type,
    )


class TestBasePlatformTopicSessions:
    @pytest.mark.asyncio
    async def test_handle_message_does_not_interrupt_different_topic(self, monkeypatch):
        adapter = DummyTelegramAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        active_event = _make_event("-1001", "10")
        adapter._active_sessions[build_session_key(active_event.source)] = asyncio.Event()

        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(asyncio, "create_task", fake_create_task)

        await adapter.handle_message(_make_event("-1001", "11"))

        assert len(scheduled) == 1
        assert adapter._pending_messages == {}

    @pytest.mark.asyncio
    async def test_handle_message_interrupts_same_topic(self, monkeypatch):
        adapter = DummyTelegramAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        active_event = _make_event("-1001", "10")
        adapter._active_sessions[build_session_key(active_event.source)] = asyncio.Event()

        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(asyncio, "create_task", fake_create_task)

        pending_event = _make_event("-1001", "10", message_id="2")
        await adapter.handle_message(pending_event)

        assert scheduled == []
        assert adapter.get_pending_message(build_session_key(pending_event.source)) == pending_event

    @pytest.mark.asyncio
    async def test_process_message_background_replies_in_same_topic(self):
        adapter = DummyTelegramAdapter()
        typing_calls = []

        async def handler(_event):
            await asyncio.sleep(0)
            return "ack"

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            typing_calls.append({"chat_id": _chat_id, "metadata": metadata})
            await asyncio.Event().wait()

        adapter.set_message_handler(handler)
        adapter._keep_typing = hold_typing

        event = _make_event("-1001", "17585")
        await adapter._process_message_background(event, build_session_key(event.source))

        assert adapter.sent == [
            {
                "chat_id": "-1001",
                "content": "ack",
                "reply_to": "1",
                "metadata": {"thread_id": "17585"},
            }
        ]
        assert typing_calls == [
            {
                "chat_id": "-1001",
                "metadata": {"thread_id": "17585"},
            }
        ]
        assert adapter.processing_hooks == [
            ("start", "1"),
            ("complete", "1", True),
        ]

    @pytest.mark.asyncio
    async def test_process_message_background_signal_does_not_auto_reply(self):
        adapter = DummySignalAdapter()

        async def handler(_event):
            await asyncio.sleep(0)
            return "ack"

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        adapter.set_message_handler(handler)
        adapter._keep_typing = hold_typing

        event = _make_event("-1001", "17585")
        await adapter._process_message_background(event, build_session_key(event.source))

        assert adapter.sent == [
            {
                "chat_id": "-1001",
                "content": "ack",
                "reply_to": None,
                "metadata": {"thread_id": "17585"},
            }
        ]

    @pytest.mark.asyncio
    async def test_process_message_background_signal_reply_directive_forces_reply(self):
        adapter = DummySignalAdapter()

        async def handler(_event):
            await asyncio.sleep(0)
            return "[[reply_to_current]]\n\nack"

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        adapter.set_message_handler(handler)
        adapter._keep_typing = hold_typing

        event = _make_event("-1001", "17585")
        await adapter._process_message_background(event, build_session_key(event.source))

        assert adapter.sent == [
            {
                "chat_id": "-1001",
                "content": "ack",
                "reply_to": "1",
                "metadata": {
                    "thread_id": "17585",
                    "reply_to_message_id": "1",
                    "reply_to_author": "+15550001111",
                    "reply_to_text": "hello",
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_process_message_background_signal_voice_input_sends_only_voice_reply(self, monkeypatch, tmp_path):
        adapter = DummySignalAdapter()

        async def handler(_event):
            await asyncio.sleep(0)
            return "spoken reply"

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        tts_path = tmp_path / "reply.ogg"
        tts_path.write_bytes(b"fake-voice")

        monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
        monkeypatch.setattr(
            "tools.tts_tool.text_to_speech_tool",
            lambda text: json.dumps({"success": True, "file_path": str(tts_path)}),
        )

        adapter.set_message_handler(handler)
        adapter._keep_typing = hold_typing

        event = _make_event("-1001", "17585", message_type=MessageType.VOICE)
        await adapter._process_message_background(event, build_session_key(event.source))

        assert adapter.sent == []
        assert adapter.voices == [
            {
                "chat_id": "-1001",
                "audio_path": str(tts_path),
                "caption": None,
                "reply_to": None,
                "metadata": {"thread_id": "17585"},
            }
        ]

    @pytest.mark.asyncio
    async def test_process_message_background_marks_total_send_failure_unsuccessful(self):
        adapter = DummyTelegramAdapter()

        async def handler(_event):
            await asyncio.sleep(0)
            return "ack"

        async def failing_send(*_args, **_kwargs):
            return SendResult(success=False, error="send failed")

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        adapter.set_message_handler(handler)
        adapter.send = failing_send
        adapter._keep_typing = hold_typing

        event = _make_event("-1001", "17585")
        await adapter._process_message_background(event, build_session_key(event.source))

        assert adapter.processing_hooks == [
            ("start", "1"),
            ("complete", "1", False),
        ]

    @pytest.mark.asyncio
    async def test_process_message_background_marks_exception_unsuccessful(self):
        adapter = DummyTelegramAdapter()

        async def handler(_event):
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        adapter.set_message_handler(handler)
        adapter._keep_typing = hold_typing

        event = _make_event("-1001", "17585")
        await adapter._process_message_background(event, build_session_key(event.source))

        assert adapter.processing_hooks == [
            ("start", "1"),
            ("complete", "1", False),
        ]

    @pytest.mark.asyncio
    async def test_process_message_background_marks_cancellation_unsuccessful(self):
        adapter = DummyTelegramAdapter()
        release = asyncio.Event()

        async def handler(_event):
            await release.wait()
            return "ack"

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        adapter.set_message_handler(handler)
        adapter._keep_typing = hold_typing

        event = _make_event("-1001", "17585")
        task = asyncio.create_task(adapter._process_message_background(event, build_session_key(event.source)))
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert adapter.processing_hooks == [
            ("start", "1"),
            ("complete", "1", False),
        ]
