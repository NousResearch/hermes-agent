import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.config import Platform
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


def test_telegram_audio_size_gate_rejects_oversized_media_before_download():
    adapter = object.__new__(TelegramAdapter)
    adapter._max_doc_bytes = 1024

    allowed, note = adapter._telegram_media_size_allowed(
        SimpleNamespace(file_size=2048),
        "voice message",
    )

    assert allowed is False
    assert "exceeds" in note
    assert "voice message" in note


@pytest.mark.asyncio
async def test_voice_tts_is_explicit_audio_reply_opt_in():
    adapter = SimpleNamespace(
        _auto_tts_disabled_chats=set(),
        _auto_tts_enabled_chats=set(),
    )
    runner = _runner(adapter)
    runner._voice_mode = {}
    runner._voice_provider_mode = {}
    runner._save_voice_modes = lambda: None
    runner._save_voice_provider_modes = lambda: None

    event = SimpleNamespace(
        source=_source(),
        get_command_args=lambda: "tts",
    )
    result = await GatewayRunner._handle_voice_command(runner, event)

    assert runner._voice_mode["telegram:12345"] == "all"
    assert "12345" in adapter._auto_tts_enabled_chats
    assert result
