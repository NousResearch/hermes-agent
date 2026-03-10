"""Tests for the /voice command and auto voice reply in the gateway."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.platforms.base import MessageEvent, MessageType, SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text: str = "", message_type=MessageType.TEXT, chat_id="123") -> MessageEvent:
    source = SessionSource(
        chat_id=chat_id,
        user_id="user1",
        platform=MagicMock(),
    )
    source.platform.value = "telegram"
    source.thread_id = None
    event = MessageEvent(text=text, message_type=message_type, source=source)
    event.message_id = "msg42"
    return event


def _make_runner(tmp_path):
    """Create a bare GatewayRunner without calling __init__."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db = None
    runner.session_store = MagicMock()
    return runner


# =====================================================================
# /voice command handler
# =====================================================================

class TestHandleVoiceCommand:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_voice_on(self, runner):
        event = _make_event("/voice on")
        result = await runner._handle_voice_command(event)
        assert "enabled" in result.lower()
        assert runner._voice_mode["123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_voice_off(self, runner):
        runner._voice_mode["123"] = "voice_only"
        event = _make_event("/voice off")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert "123" not in runner._voice_mode

    @pytest.mark.asyncio
    async def test_voice_tts(self, runner):
        event = _make_event("/voice tts")
        result = await runner._handle_voice_command(event)
        assert "tts" in result.lower()
        assert runner._voice_mode["123"] == "all"

    @pytest.mark.asyncio
    async def test_voice_status_off(self, runner):
        event = _make_event("/voice status")
        result = await runner._handle_voice_command(event)
        assert "off" in result.lower()

    @pytest.mark.asyncio
    async def test_voice_status_on(self, runner):
        runner._voice_mode["123"] = "voice_only"
        event = _make_event("/voice status")
        result = await runner._handle_voice_command(event)
        assert "voice reply" in result.lower()

    @pytest.mark.asyncio
    async def test_toggle_off_to_on(self, runner):
        event = _make_event("/voice")
        result = await runner._handle_voice_command(event)
        assert "enabled" in result.lower()
        assert runner._voice_mode["123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_toggle_on_to_off(self, runner):
        runner._voice_mode["123"] = "voice_only"
        event = _make_event("/voice")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert "123" not in runner._voice_mode

    @pytest.mark.asyncio
    async def test_persistence_saved(self, runner):
        event = _make_event("/voice on")
        await runner._handle_voice_command(event)
        assert runner._VOICE_MODE_PATH.exists()
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
        assert data["123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_persistence_loaded(self, runner):
        runner._VOICE_MODE_PATH.write_text(json.dumps({"456": "all"}))
        loaded = runner._load_voice_modes()
        assert loaded == {"456": "all"}

    @pytest.mark.asyncio
    async def test_per_chat_isolation(self, runner):
        e1 = _make_event("/voice on", chat_id="aaa")
        e2 = _make_event("/voice tts", chat_id="bbb")
        await runner._handle_voice_command(e1)
        await runner._handle_voice_command(e2)
        assert runner._voice_mode["aaa"] == "voice_only"
        assert runner._voice_mode["bbb"] == "all"


# =====================================================================
# Auto voice reply decision logic
# =====================================================================

class TestAutoVoiceReply:
    """Test the should_voice_reply decision logic (extracted from _handle_message)."""

    def _should_reply(self, voice_mode, message_type, agent_messages=None, response="Hello!"):
        """Replicate the auto voice reply decision from _handle_message."""
        if not response or response.startswith("Error:"):
            return False

        is_voice_input = (message_type == MessageType.VOICE)
        should = (
            (voice_mode == "all")
            or (voice_mode == "voice_only" and is_voice_input)
        )
        if not should:
            return False

        # Dedup check
        if agent_messages:
            has_agent_tts = any(
                msg.get("role") == "assistant"
                and any(
                    tc.get("function", {}).get("name") == "text_to_speech"
                    for tc in (msg.get("tool_calls") or [])
                )
                for msg in agent_messages
            )
            if has_agent_tts:
                return False

        return True

    def test_voice_only_voice_input(self):
        assert self._should_reply("voice_only", MessageType.VOICE) is True

    def test_voice_only_text_input(self):
        assert self._should_reply("voice_only", MessageType.TEXT) is False

    def test_all_mode_text_input(self):
        assert self._should_reply("all", MessageType.TEXT) is True

    def test_all_mode_voice_input(self):
        assert self._should_reply("all", MessageType.VOICE) is True

    def test_off_mode(self):
        assert self._should_reply("off", MessageType.VOICE) is False
        assert self._should_reply("off", MessageType.TEXT) is False

    def test_error_response_skipped(self):
        assert self._should_reply("all", MessageType.TEXT, response="Error: boom") is False

    def test_empty_response_skipped(self):
        assert self._should_reply("all", MessageType.TEXT, response="") is False

    def test_dedup_skips_when_agent_called_tts(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "text_to_speech", "arguments": "{}"},
            }],
        }]
        assert self._should_reply("all", MessageType.TEXT, agent_messages=messages) is False

    def test_no_dedup_for_other_tools(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }],
        }]
        assert self._should_reply("all", MessageType.TEXT, agent_messages=messages) is True


# =====================================================================
# _send_voice_reply
# =====================================================================

class TestSendVoiceReply:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_calls_tts_and_send_voice(self, runner):
        mock_adapter = AsyncMock()
        mock_adapter.send_voice = AsyncMock()
        event = _make_event()
        runner.adapters[event.source.platform] = mock_adapter

        tts_result = json.dumps({"success": True, "file_path": "/tmp/test.ogg"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=True), \
             patch("os.unlink"), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello world")

        mock_adapter.send_voice.assert_called_once()
        call_args = mock_adapter.send_voice.call_args
        assert call_args.kwargs.get("chat_id") == "123"

    @pytest.mark.asyncio
    async def test_empty_text_after_strip_skips(self, runner):
        event = _make_event()

        with patch("tools.tts_tool.text_to_speech_tool") as mock_tts, \
             patch("tools.tts_tool._strip_markdown_for_tts", return_value=""):
            await runner._send_voice_reply(event, "```code only```")

        mock_tts.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_failure_no_crash(self, runner):
        event = _make_event()
        mock_adapter = AsyncMock()
        runner.adapters[event.source.platform] = mock_adapter
        tts_result = json.dumps({"success": False, "error": "API error"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=False), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello")

        mock_adapter.send_voice.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_caught(self, runner):
        event = _make_event()
        with patch("tools.tts_tool.text_to_speech_tool", side_effect=RuntimeError("boom")), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.makedirs"):
            # Should not raise
            await runner._send_voice_reply(event, "Hello")


# =====================================================================
# Help text + known commands
# =====================================================================

class TestVoiceInHelp:

    def test_voice_in_help_output(self):
        from gateway.run import GatewayRunner
        import inspect
        source = inspect.getsource(GatewayRunner._handle_help_command)
        assert "/voice" in source

    def test_voice_is_known_command(self):
        from gateway.run import GatewayRunner
        import inspect
        source = inspect.getsource(GatewayRunner._handle_message)
        assert '"voice"' in source
