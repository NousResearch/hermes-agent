"""``gateway.tts_reply_text``: voice-only auto-TTS replies (text only as the fallback)."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter


def test_tts_reply_text_defaults_on_for_backwards_compatibility():
    cfg = GatewayConfig.from_dict({})

    assert cfg.tts_reply_text is True
    assert cfg.to_dict()["tts_reply_text"] is True


def test_tts_reply_text_can_be_switched_off():
    cfg = GatewayConfig.from_dict({"tts_reply_text": False})

    assert cfg.tts_reply_text is False
    assert cfg.to_dict()["tts_reply_text"] is False


class _Adapter(BasePlatformAdapter):
    """Minimal concrete adapter: records the play_tts call, returns a canned result."""

    def __init__(self, *, tts_reply_text: bool, voice_success: bool):
        self.platform = Platform.TELEGRAM
        self.gateway_runner = SimpleNamespace(config=SimpleNamespace(tts_reply_text=tts_reply_text))
        self._voice_success = voice_success
        self.calls = []

    async def connect(self): ...
    async def disconnect(self): ...
    async def get_chat_info(self, *a, **k): return {}
    async def send(self, *a, **k): ...

    async def play_tts(self, chat_id, audio_path, caption=None, **kwargs):
        self.calls.append(caption)
        return SimpleNamespace(success=self._voice_success)


def _play(adapter, text="the reply"):
    event = SimpleNamespace(source=SimpleNamespace(chat_id="42"))
    return asyncio.run(BasePlatformAdapter._play_tts_file(
        adapter, event, text, "/tmp/reply.ogg", True, {}, lambda r: None))


def test_default_keeps_text_as_telegram_caption():
    adapter = _Adapter(tts_reply_text=True, voice_success=True)

    assert _play(adapter) is True          # text rode along as the caption → text send skipped
    assert adapter.calls == ["the reply"]


def test_voice_only_skips_text_when_voice_lands():
    adapter = _Adapter(tts_reply_text=False, voice_success=True)

    assert _play(adapter) is True          # voice delivered → no caption, no text message
    assert adapter.calls == [None]


def test_voice_only_falls_back_to_text_when_voice_fails():
    adapter = _Adapter(tts_reply_text=False, voice_success=False)

    assert _play(adapter) is False         # voice failed → the caller sends the text as usual
    assert adapter.calls == [None]


def _delivered(adapter, parts):
    return BasePlatformAdapter._tts_text_delivered(adapter, parts)


def test_voice_only_multipart_falls_back_to_text_when_any_part_fails():
    adapter = _Adapter(tts_reply_text=False, voice_success=True)

    assert _delivered(adapter, [False, True]) is False   # part 1 never landed → text still goes out
    assert _delivered(adapter, [True, True]) is True     # every part landed → voice-only reply
    assert _delivered(adapter, []) is False


def test_default_mode_multipart_keeps_first_caption_semantics():
    adapter = _Adapter(tts_reply_text=True, voice_success=True)

    assert _delivered(adapter, [True, False]) is True    # caption rode on part 1 → text skipped
    assert _delivered(adapter, [False, False]) is False
