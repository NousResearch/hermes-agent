"""Base-adapter auto-TTS must pass a platform-aware explicit output path.

Regression tests for the cleared-contextvar bug (#57049, #36685): the
post-handler auto-TTS block in ``BasePlatformAdapter._process_message_background``
runs AFTER ``_clear_session_env`` wiped ``HERMES_SESSION_PLATFORM``, so the
TTS tool's contextvar-based ``want_opus`` detection always resolved False on
that path and Opus platforms received MP3 (audio attachment, not a native
voice bubble). The fix passes an explicit output path from
``build_auto_tts_output_path(platform)``, which consults the TTS tool's
``OPUS_VOICE_PLATFORMS`` set — the single source of truth.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    build_auto_tts_output_path,
)
from gateway.session import SessionSource, build_session_key
from tools.tts_tool import OPUS_VOICE_PLATFORMS


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_voice_event(platform: Platform) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=platform,
            chat_id="-1001",
            chat_type="group",
        ),
        message_id="voice-1",
    )


def _hold_typing():
    async def hold(*_args, **_kwargs):
        await asyncio.Event().wait()

    return hold


# ---------------------------------------------------------------------------
# build_auto_tts_output_path: OPUS_VOICE_PLATFORMS is the single source of truth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform_name", sorted(OPUS_VOICE_PLATFORMS))
def test_output_path_is_ogg_for_every_opus_voice_platform(platform_name):
    path = build_auto_tts_output_path(platform_name)
    assert path.endswith(".ogg"), path


@pytest.mark.parametrize(
    "platform", [Platform.DISCORD, Platform.SLACK, "irc", None]
)
def test_output_path_is_mp3_for_non_opus_platforms(platform):
    path = build_auto_tts_output_path(platform)
    assert path.endswith(".mp3"), path


def test_output_path_accepts_platform_enum():
    assert build_auto_tts_output_path(Platform.TELEGRAM).endswith(".ogg")
    assert build_auto_tts_output_path(Platform.MATRIX).endswith(".ogg")
    assert build_auto_tts_output_path(Platform.FEISHU).endswith(".ogg")


def test_output_paths_are_unique():
    assert build_auto_tts_output_path("telegram") != build_auto_tts_output_path("telegram")


# ---------------------------------------------------------------------------
# Base-adapter auto-TTS block: explicit output_path, no contextvar reliance
# ---------------------------------------------------------------------------

async def _run_auto_tts(adapter: _DummyAdapter, platform: Platform):
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="tts-1"))
    long_reply = "x" * 2000  # avoid the telegram caption-collapse path
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result=long_reply))
    event = _make_voice_event(platform)
    requested = []

    def fake_tts(*, text, output_path=None):
        requested.append(output_path)
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake audio")
        return json.dumps({"success": True, "file_path": output_path})

    with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=fake_tts
    ):
        await adapter._process_message_background(
            event, build_session_key(event.source)
        )
    return requested, adapter


@pytest.mark.asyncio
async def test_base_auto_tts_requests_ogg_on_opus_platform():
    """Telegram (opus platform) gets an explicit .ogg path even though the
    HERMES_SESSION_PLATFORM contextvar is cleared by the time the block runs."""
    adapter = _DummyAdapter(Platform.TELEGRAM)
    requested, adapter = await _run_auto_tts(adapter, Platform.TELEGRAM)

    assert requested and requested[0] is not None
    assert requested[0].endswith(".ogg")
    adapter.play_tts.assert_awaited_once()
    assert adapter.play_tts.await_args.kwargs["audio_path"].endswith(".ogg")


@pytest.mark.asyncio
async def test_base_auto_tts_keeps_mp3_on_non_opus_platform():
    adapter = _DummyAdapter(Platform.DISCORD)
    requested, adapter = await _run_auto_tts(adapter, Platform.DISCORD)

    assert requested and requested[0] is not None
    assert requested[0].endswith(".mp3")
    adapter.play_tts.assert_awaited_once()
    assert adapter.play_tts.await_args.kwargs["audio_path"].endswith(".mp3")


@pytest.mark.asyncio
async def test_base_auto_tts_skips_playback_when_tool_reports_failure():
    """A success=False tool result must not deliver a stale/partial file."""
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="tts-1"))
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="reply text"))
    event = _make_voice_event(Platform.TELEGRAM)

    def fake_tts(*, text, output_path=None):
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"partial")
        return json.dumps({"success": False, "error": "backend exploded"})

    with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=fake_tts
    ):
        await adapter._process_message_background(
            event, build_session_key(event.source)
        )

    adapter.play_tts.assert_not_awaited()
    # Text reply still goes out.
    assert adapter.sent and adapter.sent[0]["content"] == "reply text"
