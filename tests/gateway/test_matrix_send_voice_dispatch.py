"""Regression tests for Matrix ``send_voice`` dispatch compatibility (#116776).

The base media dispatch calls ``send_voice(..., is_voice=...)`` for every audio
MEDIA attachment. The Matrix adapter used to reject that kwarg, so all non-image
MEDIA delivery raised ``TypeError: MatrixAdapter.send_voice() got an unexpected
keyword argument 'is_voice'`` and the attachment was silently dropped.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig


def _make_adapter(**extra):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
            **extra,
        },
    )
    return MatrixAdapter(config)


def _spy_send_local_file(adapter):
    adapter._send_local_file = AsyncMock(return_value=MagicMock(success=True))
    return adapter._send_local_file


def test_dispatch_call_with_is_voice_false_sends_plain_audio():
    """The exact dispatcher call shape (base.py ``_send_one``) must not raise
    TypeError, and an audio-ext attachment must keep its original format."""
    adapter = _make_adapter()
    spy = _spy_send_local_file(adapter)

    result = asyncio.run(adapter.send_voice(
        chat_id="!room:example.org", audio_path="/tmp/clip.mp3", metadata=None, is_voice=False))

    assert result.success is True
    spy.assert_awaited_once()
    args, kwargs = spy.call_args
    assert args[0] == "!room:example.org"
    assert args[1] == "/tmp/clip.mp3"  # original file, no transcode
    assert args[2] == "m.audio"
    assert kwargs["is_voice"] is False


def test_voice_tagged_attachment_gets_voice_bubble():
    adapter = _make_adapter()
    spy = _spy_send_local_file(adapter)

    asyncio.run(adapter.send_voice(
        chat_id="!room:example.org", audio_path="/tmp/clip.ogg", metadata=None, is_voice=True))

    args, kwargs = spy.call_args
    assert args[1] == "/tmp/clip.ogg"
    assert kwargs["is_voice"] is True  # MSC3245 voice metadata path


def test_legacy_caller_without_flag_keeps_voice_bubble():
    """``play_audio`` (base default) calls send_voice without the flag; TTS
    playback must stay a voice bubble."""
    adapter = _make_adapter()
    spy = _spy_send_local_file(adapter)

    asyncio.run(adapter.send_voice(chat_id="!room:example.org", audio_path="/tmp/tts.ogg"))

    args, kwargs = spy.call_args
    assert kwargs["is_voice"] is True


def test_voice_bubble_transcodes_non_ogg():
    adapter = _make_adapter()
    spy = _spy_send_local_file(adapter)
    with patch("plugins.platforms.matrix.adapter.transcode_to_ogg_opus",
               MagicMock(return_value="/tmp/clip_converted.ogg")):
        asyncio.run(adapter.send_voice(
            chat_id="!room:example.org", audio_path="/tmp/clip.mp3", is_voice=True))

    args, kwargs = spy.call_args
    assert args[1] == "/tmp/clip_converted.ogg"
    assert kwargs["file_name"] == "clip.ogg"  # keep the caller's basename
