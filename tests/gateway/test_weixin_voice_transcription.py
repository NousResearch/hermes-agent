"""A received Weixin voice note must reach SILK decoding and the gateway's configured STT backend."""

import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platforms import weixin
from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner
from hermes_cli.config import atomic_config_write
from tools import transcription_tools as stt


@pytest.mark.asyncio
async def test_weixin_voice_reaches_configured_stt_and_cleans_decoded_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    extra = {
        "account_id": "voice-bot", "dm_policy": "allowlist", "allow_from": ["speaker"],
        "use_platform_transcription": False,
    }
    atomic_config_write(tmp_path / "config.yaml", {
        "stt": {"enabled": True, "provider": "local", "language": "zh"},
        "platforms": {"weixin": {"enabled": True, "extra": extra}},
    })
    adapter = weixin.WeixinAdapter(load_gateway_config().platforms[Platform.WEIXIN])
    adapter._poll_session = Mock()
    adapter._token = ""
    adapter.handle_message = AsyncMock()
    raw_audio = b"\x02#!SILK_V3test-audio"
    monkeypatch.setattr(weixin, "_download_and_decrypt_media", AsyncMock(return_value=raw_audio))
    decoded_paths = []

    def decode(source, target):
        assert Path(source).read_bytes() == raw_audio
        decoded_paths.append(Path(target))
        with wave.open(target, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 320)

    def recognize(path, **kwargs):
        with wave.open(path, "rb") as wav:
            assert wav.getframerate() == 16000
        assert kwargs["language"] == "zh"
        segments = [SimpleNamespace(text="测试微信语音识别", no_speech_prob=0.0, avg_logprob=0.0)]
        return segments, SimpleNamespace(language="zh", duration=0.02)

    monkeypatch.setitem(sys.modules, "pilk", SimpleNamespace(silk_to_wav=decode))
    monkeypatch.setattr(stt, "_HAS_PILK", True)
    monkeypatch.setattr(stt, "_HAS_FASTER_WHISPER", True)
    backend = Mock(side_effect=recognize)
    model = SimpleNamespace(supported_languages=["zh"], transcribe=backend)
    monkeypatch.setattr(stt, "_get_or_load_local_model", lambda *_: model)
    await adapter._process_message({
        "from_user_id": "speaker", "to_user_id": "voice-bot", "message_id": "voice-message", "msg_type": 1,
        "item_list": [{"type": weixin.ITEM_VOICE, "voice_item": {
            "text": "untrusted Tencent transcript", "media": {"encrypt_query_param": "test-query"},
        }}],
    })
    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == MessageType.VOICE
    assert event.media_types == ["audio/silk"]
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    images, audio, files, videos = runner._classify_inbound_media(event, False)
    assert not (images or files or videos)
    enriched, transcripts = await runner._enrich_message_with_transcription(event.text, audio)

    assert transcripts == ["测试微信语音识别"]
    assert "测试微信语音识别" in enriched
    assert "untrusted Tencent transcript" not in enriched
    backend.assert_called_once()
    assert decoded_paths and all(not path.exists() for path in decoded_paths)
    assert Path(event.media_urls[0]).exists()
