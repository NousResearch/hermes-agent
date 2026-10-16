"""Voice capability declarations apply to every automatic reply path (#132441)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, StreamingTTSHandle
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.yuanbao import YuanbaoAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_constants import get_hermes_home
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.dingtalk.adapter import DingTalkAdapter
from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self, config):
        super().__init__(config, Platform.SLACK)
        self.audio = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="reply")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, metadata=None):
        self.audio.append(Path(audio_path).read_bytes())
        return SendResult(success=True, message_id="audio")

    def supports_streaming_tts(self, chat_id, audio_format):
        return True

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        return StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)

    async def write_streaming_tts(self, handle, chunk):
        self.audio.append(chunk)
        handle.audible = True


def _voice_turn(adapter, mode):
    get_hermes_home().joinpath("config.yaml").write_text(
        f"voice:\n  auto_tts: {str(mode != 'all').lower()}\n", encoding="utf-8"
    )
    source = SessionSource(platform=adapter.platform, chat_id="chat", chat_type="dm")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {} if mode == "global" else {f"{adapter.platform.value}:chat": mode}
    runner._gateway_loop = asyncio.get_running_loop()
    runner._sync_voice_mode_state_to_adapter(adapter)
    event = MessageEvent(
        text="hello", message_type=MessageType.VOICE, source=source, message_id="voice-1"
    )
    return runner, event


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["global", "all"])
@pytest.mark.parametrize(
    "adapter_class", [A2AAdapter, DingTalkAdapter, YuanbaoAdapter, WecomCallbackAdapter, _RecordingAdapter]
)
async def test_voice_input_delivers_text_without_audio_on_text_only_adapters(
    adapter_class, mode, monkeypatch
):
    adapter = adapter_class(PlatformConfig(enabled=True))
    _, event = _voice_turn(adapter, mode)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reply"))
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="The answer is 42."))
    synthesized = []

    def synthesize(*, text, output_path):
        synthesized.append(text)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"audio")
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", synthesize)
    await adapter._process_message_background(event, build_session_key(event.source))

    assert [
        call.kwargs["content"] if "content" in call.kwargs else call.args[1]
        for call in adapter.send.await_args_list
    ] == ["The answer is 42."]
    if adapter_class is _RecordingAdapter:
        assert synthesized == ["The answer is 42."]
        assert adapter.audio == [b"audio"]
    else:
        assert synthesized == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["global", "all", "off"])
async def test_streaming_voice_reply_obeys_capability_before_chat_preferences(mode, monkeypatch):
    class Streamer:
        def stream(self, text):
            yield b"pcm"

    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda _config: Streamer())
    delivered = []
    for supports_voice in (False, True):
        adapter = _RecordingAdapter(PlatformConfig(enabled=True))
        adapter.supports_voice_replies = supports_voice
        runner, event = _voice_turn(adapter, mode)
        holder = [None]
        runner._run_agent_start_streaming_tts(event.source, event.message_type, None, holder)
        if holder[0] is not None:
            holder[0].on_delta("The answer is 42.")
            holder[0].finish()
            await asyncio.wait_for(holder[0].start(), timeout=5)
        delivered.append(adapter.audio)

    assert delivered == [[], [] if mode == "off" else [b"pcm"]]
