"""Tests for BasePlatformAdapter topic-aware session handling."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource, build_session_key


class DummyTelegramAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform = Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self._busy_text_mode = ""
        self.sent = []
        self.typing = []
        self.processing_hooks = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
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

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "stopped": True, "metadata": metadata})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def on_processing_start(self, event: MessageEvent) -> None:
        self.processing_hooks.append(("start", event.message_id))

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        self.processing_hooks.append(("complete", event.message_id, outcome))


def _make_event(chat_id: str, thread_id: str, message_id: str = "1") -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="group",
            thread_id=thread_id,
        ),
        message_id=message_id,
    )


class TestBasePlatformTopicSessions:

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
                "reply_to": None,
                "metadata": {"thread_id": "17585", "notify": True},
            }
        ]
        assert typing_calls == [
            {
                "chat_id": "-1001",
                "metadata": {"thread_id": "17585"},
            }
        ]
        assert {
            "chat_id": "-1001",
            "stopped": True,
            "metadata": {"thread_id": "17585"},
        } in adapter.typing
        assert adapter.processing_hooks == [
            ("start", "1"),
            ("complete", "1", ProcessingOutcome.SUCCESS),
        ]


class TestTelegramAutoTtsCaptionDelivery:
    @staticmethod
    def _make_voice_event(chat_id: str = "-1001", thread_id: str = "17585") -> MessageEvent:
        return MessageEvent(
            text="hello",
            message_type=MessageType.VOICE,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=chat_id,
                chat_type="group",
                thread_id=thread_id,
            ),
            message_id="voice-1",
        )

    @staticmethod
    def _hold_typing():
        async def hold(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        return hold


    @pytest.mark.asyncio
    async def test_long_original_with_short_spoken_script_still_sends_full_reply(self, tmp_path):
        adapter = DummyTelegramAdapter()
        adapter._keep_typing = self._hold_typing()
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="tts-1"))
        # Markdown-heavy reply: over the 1024-char caption limit as written,
        # but the normalized spoken script (markdown and URLs removed) is far
        # below it. Caption eligibility must follow the ORIGINAL reply, so the
        # full formatted text is still delivered as its own message instead of
        # being swallowed into a lossy caption.
        long_reply = "\n".join(
            f"- **item {i}** [details](https://example.com/some/very/long/path/{i:04d})"
            for i in range(20)
        )
        assert len(long_reply) > 1024
        assert len(adapter.prepare_tts_text(long_reply)) <= 1024
        adapter.set_message_handler(lambda _event: asyncio.sleep(0, result=long_reply))

        tts_path = tmp_path / "reply.ogg"
        tts_path.write_text("audio", encoding="utf-8")
        event = self._make_voice_event()

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
            "tools.tts_tool.text_to_speech_tool",
            return_value=json.dumps({"file_path": str(tts_path)}),
        ):
            await adapter._process_message_background(event, build_session_key(event.source))

        adapter.play_tts.assert_awaited_once()
        assert adapter.play_tts.await_args.kwargs["caption"] is None
        assert adapter.sent == [
            {
                "chat_id": "-1001",
                "content": long_reply,
                "reply_to": None,
                "metadata": {"thread_id": "17585", "notify": True},
            }
        ]


class DummyVoiceOutAdapter(DummyTelegramAdapter):
    """Adapter whose voice bubbles inherently render the spoken text (Carbon Voice-style
    server-side STT). Non-Telegram so the caption path stays out of the way."""

    voice_out_carries_text = True

    def __init__(self):
        super().__init__(platform=Platform.DISCORD)


class TestVoiceOutCarriesTextDelivery:
    """Behavioral coverage for the auto-TTS dispatch on adapters that opt into
    ``voice_out_carries_text``: the follow-up text bubble is suppressed only when the spoken text
    covers the complete response, and kept otherwise (formatting stripped, or the voice send fails).
    Drives ``_process_message_background`` end-to-end."""

    @staticmethod
    def _make_voice_event(chat_id: str = "chan-1", message_id: str = "voice-1") -> MessageEvent:
        return MessageEvent(
            text="hello",
            message_type=MessageType.VOICE,
            source=SessionSource(
                platform=Platform.DISCORD,
                chat_id=chat_id,
                chat_type="group",
            ),
            message_id=message_id,
        )

    @staticmethod
    async def _run_auto_tts(adapter, reply: str, tmp_path, tts_success: bool = True) -> None:
        async def hold(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        adapter._keep_typing = hold
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        adapter.play_tts = AsyncMock(
            return_value=SendResult(
                success=tts_success, message_id="tts-1", error=None if tts_success else "boom"))
        adapter.set_message_handler(lambda _event: asyncio.sleep(0, result=reply))

        tts_path = tmp_path / "reply.ogg"
        tts_path.write_text("audio", encoding="utf-8")
        event = TestVoiceOutCarriesTextDelivery._make_voice_event()

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
            "tools.tts_tool.text_to_speech_tool",
            return_value=json.dumps({"file_path": str(tts_path)}),
        ):
            await adapter._process_message_background(event, build_session_key(event.source))

    @pytest.mark.asyncio
    async def test_plain_reply_suppresses_followup_text(self, tmp_path):
        adapter = DummyVoiceOutAdapter()
        reply = "Short reply"
        assert adapter.prepare_tts_text(reply) == reply  # spoken text IS the whole response
        await self._run_auto_tts(adapter, reply, tmp_path)

        adapter.play_tts.assert_awaited_once()
        assert adapter.play_tts.await_args.kwargs["caption"] is None
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_formatted_reply_keeps_followup_text_when_tts_strips_markdown(self, tmp_path):
        adapter = DummyVoiceOutAdapter()
        formatted_reply = "**Bold** with a [link](https://example.com)"
        assert adapter.prepare_tts_text(formatted_reply) != formatted_reply
        await self._run_auto_tts(adapter, formatted_reply, tmp_path)

        adapter.play_tts.assert_awaited_once()
        assert len(adapter.sent) == 1
        assert adapter.sent[0]["content"] == formatted_reply

    @pytest.mark.asyncio
    async def test_failed_tts_send_keeps_followup_text(self, tmp_path):
        adapter = DummyVoiceOutAdapter()
        await self._run_auto_tts(adapter, "Short reply", tmp_path, tts_success=False)

        adapter.play_tts.assert_awaited_once()
        assert len(adapter.sent) == 1
        assert adapter.sent[0]["content"] == "Short reply"

    @pytest.mark.asyncio
    async def test_non_opted_in_adapter_keeps_followup_text(self, tmp_path):
        adapter = DummyTelegramAdapter(platform=Platform.DISCORD)
        assert adapter.voice_out_carries_text is False
        await self._run_auto_tts(adapter, "Short reply", tmp_path)

        adapter.play_tts.assert_awaited_once()
        assert len(adapter.sent) == 1
        assert adapter.sent[0]["content"] == "Short reply"

