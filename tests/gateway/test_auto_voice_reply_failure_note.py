"""Auto voice reply failures must surface to the user once per failure streak (#133134).

The three failure paths in ``_send_voice_reply`` (invalid TTS JSON, ``success: false`` with a
provider error envelope, any exception) used to be log-only: the text reply still went out, so
with a paid TTS provider users silently stopped getting voice (402/401/429) with nothing in chat
and no access to the server log on hosted installs.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._voice_mode = {}
    runner._voice_fail_noted = set()
    runner.adapters = {}
    runner._voice_key_for_source = lambda source: f"{source.platform.value}:{source.chat_id}"
    return runner


def _make_event(platform: Platform = Platform.TELEGRAM, chat_id: str = "123") -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(
            platform=platform, chat_id=chat_id, user_id="u1", user_name="User",
        ),
        message_type=MessageType.TEXT,
        message_id="456",
    )


class TestSendVoiceReplyReturnsFailureReason:

    @pytest.mark.asyncio
    async def test_provider_error_envelope_is_returned(self):
        """A paid provider's failure envelope (HTTP 402/401/429 ...) reaches the caller instead
        of dying in the server log."""
        runner = _make_runner()
        envelope = "Fish Audio: out of API credits. Top up at example.com"

        def failing_tts(*, text, output_path):
            return json.dumps({"success": False, "error": envelope})

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=failing_tts):
            reason = await runner._send_voice_reply(_make_event(), "hello")

        assert reason == envelope

    @pytest.mark.asyncio
    async def test_invalid_tts_json_returns_reason(self):
        runner = _make_runner()
        with patch("tools.tts_tool.text_to_speech_tool", return_value="not json"):
            reason = await runner._send_voice_reply(_make_event(), "hello")
        assert reason == "TTS returned an invalid response"

    @pytest.mark.asyncio
    async def test_tts_exception_returns_reason(self):
        runner = _make_runner()
        with patch("tools.tts_tool.text_to_speech_tool", side_effect=RuntimeError("connection reset")):
            reason = await runner._send_voice_reply(_make_event(), "hello")
        assert reason == "connection reset"

    @pytest.mark.asyncio
    async def test_successful_delivery_rearms_the_note(self, tmp_path):
        """A delivered voice reply clears the chat's noted flag, so the NEXT failure after a
        recovery is announced again instead of staying silent forever."""
        runner = _make_runner()
        runner._voice_fail_noted.add("telegram:123")
        runner._deliver_voice_reply = AsyncMock()

        def ok_tts(*, text, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake ogg")
            return json.dumps({"success": True, "file_path": output_path})

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=ok_tts):
            reason = await runner._send_voice_reply(_make_event(), "hello")

        assert reason is None
        assert "telegram:123" not in runner._voice_fail_noted


class TestVoiceUnavailableNote:

    def test_note_fires_once_per_chat(self):
        runner = _make_runner()
        event = _make_event()

        first = runner._voice_unavailable_note(event, "Fish Audio: out of API credits")
        second = runner._voice_unavailable_note(event, "Fish Audio: out of API credits")

        assert first == "(Voice reply unavailable: Fish Audio: out of API credits)"
        # Rate-limited: a persistent outage must not annotate every message.
        assert second is None

    def test_other_chats_are_not_suppressed(self):
        runner = _make_runner()
        assert runner._voice_unavailable_note(_make_event(chat_id="1"), "boom")
        assert runner._voice_unavailable_note(_make_event(chat_id="2"), "boom")

    def test_multiline_error_is_collapsed_and_capped(self):
        runner = _make_runner()
        note = runner._voice_unavailable_note(_make_event(), "line1\n  line2\n" + "x" * 500)
        assert "\n" not in note
        assert len(note) <= len("(Voice reply unavailable: ") + 200 + 1


class TestDeliverTurnResponseSurfacesNote:

    async def _run_seam(self, runner, agent_result, response="the answer"):
        event = _make_event()
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter._streaming_tts_turn_completed = lambda *a, **k: False
        runner._delivery_adapter_for = lambda source: adapter
        runner._event_thread_metadata = lambda event, source: None
        runner._should_send_voice_reply = lambda *a, **k: True

        async def failing_voice(event, text):
            return "Fish Audio: out of API credits"

        runner._send_voice_reply = failing_voice
        out = await runner._hmwa_deliver_turn_response(
            event, event.source, SimpleNamespace(session_id="s1"),
            "agent:main:telegram:123", None, agent_result, [], response, None, False,
        )
        return out, adapter

    @pytest.mark.asyncio
    async def test_unstreamed_reply_gets_note_appended(self):
        """The note rides along the text reply the caller is about to send."""
        out, adapter = await self._run_seam(_make_runner(), agent_result={})
        assert out == "the answer\n\n(Voice reply unavailable: Fish Audio: out of API credits)"

    @pytest.mark.asyncio
    async def test_streamed_reply_gets_note_as_its_own_line(self):
        """The streamed body cannot be amended, so the note is sent as its own line."""
        out, adapter = await self._run_seam(
            _make_runner(), agent_result={"already_sent": True, "media_already_delivered": True})
        assert out is None
        adapter.send.assert_awaited_once_with(
            "123", "(Voice reply unavailable: Fish Audio: out of API credits)", metadata=None)

    @pytest.mark.asyncio
    async def test_second_failure_in_the_same_streak_stays_silent(self):
        """The whole streak after the first failure is note-free: the second turn's text goes
        out unchanged."""
        runner = _make_runner()
        first, _ = await self._run_seam(runner, agent_result={})
        second, _ = await self._run_seam(runner, agent_result={})
        assert first.endswith("(Voice reply unavailable: Fish Audio: out of API credits)")
        assert second == "the answer"
