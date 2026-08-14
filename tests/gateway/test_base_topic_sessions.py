"""Tests for BasePlatformAdapter topic-aware session handling."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult
from gateway.session import SessionSource, build_session_key


class DummyTelegramAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
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

    @pytest.mark.asyncio
    async def test_final_reply_uses_anchor_added_by_handler(self):
        """A transcript echo created during handling becomes the final anchor."""
        adapter = DummyTelegramAdapter()
        adapter.config.typing_indicator = False

        async def handler(event):
            event._gateway_stt_reply_anchor = "transcript-echo-7"
            return "ack"

        adapter.set_message_handler(handler)
        event = MessageEvent(
            text="",
            message_type=MessageType.VOICE,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="dm",
                thread_id="42",
            ),
            message_id="voice-note-6",
        )

        await adapter._process_message_background(
            event,
            build_session_key(event.source),
        )

        assert adapter.sent == [
            {
                "chat_id": "12345",
                "content": "ack",
                "reply_to": "transcript-echo-7",
                "metadata": {
                    "thread_id": "42",
                    "telegram_dm_topic_reply_fallback": True,
                    "direct_messages_topic_id": "42",
                    "telegram_reply_to_message_id": "transcript-echo-7",
                    "notify": True,
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_handler_error_replies_to_anchor_added_by_handler(self):
        """A user-facing handler error replies to the successful transcript echo."""
        adapter = DummyTelegramAdapter()
        adapter.config.typing_indicator = False

        async def handler(event):
            event._gateway_stt_reply_anchor = "transcript-echo-7"
            raise RuntimeError("provider failed")

        adapter.set_message_handler(handler)
        event = MessageEvent(
            text="",
            message_type=MessageType.VOICE,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="dm",
            ),
            message_id="voice-note-6",
        )

        await adapter._process_message_background(
            event,
            build_session_key(event.source),
        )

        assert len(adapter.sent) == 1
        assert adapter.sent[0]["reply_to"] == "transcript-echo-7"
        assert adapter.sent[0]["metadata"] is None
        assert "provider failed" in adapter.sent[0]["content"]

    @pytest.mark.asyncio
    async def test_explicit_anchor_tombstone_does_not_restore_outer_message_id(self):
        adapter = DummyTelegramAdapter()
        adapter.config.typing_indicator = False

        async def handler(event):
            event._gateway_stt_reply_anchor = None
            return "ack"

        adapter.set_message_handler(handler)
        event = MessageEvent(
            text="",
            message_type=MessageType.VOICE,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="dm",
            ),
            message_id="outer-voice-note-6",
        )

        await adapter._process_message_background(
            event,
            build_session_key(event.source),
        )

        assert adapter.sent == [
            {
                "chat_id": "12345",
                "content": "ack",
                "reply_to": None,
                "metadata": {"notify": True},
            }
        ]

    @pytest.mark.asyncio
    async def test_media_follow_on_replies_to_transcript_echo_in_ordinary_dm(
        self,
        tmp_path,
    ):
        adapter = DummyTelegramAdapter()
        adapter.config.typing_indicator = False
        adapter.send_voice = AsyncMock(
            return_value=SendResult(success=True, message_id="voice-8")
        )
        audio_path = tmp_path / "reply.mp3"
        audio_path.write_bytes(b"audio")

        async def handler(event):
            event._gateway_stt_reply_anchor = "transcript-echo-7"
            return f"answer\nMEDIA:{audio_path}"

        adapter.set_message_handler(handler)
        event = MessageEvent(
            text="",
            message_type=MessageType.VOICE,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="dm",
            ),
            message_id="voice-note-6",
        )

        await adapter._process_message_background(
            event,
            build_session_key(event.source),
        )

        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.await_args.kwargs["reply_to"] == "transcript-echo-7"


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

    @pytest.mark.asyncio
    async def test_short_caption_replies_to_transcript_echo_in_ordinary_dm(self, tmp_path):
        adapter = DummyTelegramAdapter()
        adapter.config.typing_indicator = False
        adapter._should_auto_tts_for_chat = lambda chat_id: True
        adapter.play_tts = AsyncMock(
            return_value=SendResult(success=True, message_id="tts-1")
        )

        async def handler(event):
            event._gateway_stt_reply_anchor = "transcript-echo-7"
            return "short answer"

        adapter.set_message_handler(handler)
        tts_path = tmp_path / "reply.ogg"
        tts_path.write_text("audio", encoding="utf-8")
        event = MessageEvent(
            text="",
            message_type=MessageType.VOICE,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="dm",
            ),
            message_id="voice-note-6",
        )

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
            "tools.tts_tool.text_to_speech_tool",
            return_value=json.dumps({"file_path": str(tts_path)}),
        ):
            await adapter._process_message_background(
                event,
                build_session_key(event.source),
            )

        adapter.play_tts.assert_awaited_once()
        assert adapter.play_tts.await_args is not None
        assert adapter.play_tts.await_args.kwargs == {
            "chat_id": "12345",
            "audio_path": str(tts_path),
            "caption": "short answer",
            "reply_to": "transcript-echo-7",
            "metadata": {"notify": True},
        }
        # A successful Telegram caption is the only final answer in this lane.
        assert adapter.sent == []

