from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_preprocess_prefixes_sender_for_shared_non_thread_group_session():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Alice] hello"


@pytest.mark.asyncio
async def test_shared_sender_prefix_wraps_enriched_turn_before_channel_context():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Ali\nce",
    )
    event = MessageEvent(
        text="please review",
        message_type=MessageType.AUDIO,
        source=source,
        media_urls=["/tmp/cache_123_recording.mp3"],
        media_types=["audio/mpeg"],
        reply_to_message_id="42",
        reply_to_text="Earlier message",
        channel_context="[Recent channel context]\nBob: background",
    )

    with patch(
        "tools.credential_files.to_agent_visible_cache_path",
        side_effect=lambda path: path,
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    media_note = (
        "[The user sent an audio file attachment: 'recording.mp3'. "
        "It is saved at: /tmp/cache_123_recording.mp3. "
        "Its content is not inlined here. If the user's request involves "
        "what the audio contains, transcribe or process it yourself — for "
        "example by passing the path to a transcription or media tool — "
        "instead of asking the user to describe it. Only ask what to do "
        "with it if their intent is genuinely unclear.]"
    )
    assert result == (
        "[Recent channel context]\nBob: background\n\n[New message]\n"
        "[Ali ce] [Replying to: \"Earlier message\"]\n\n"
        f"{media_note}\n\nplease review"
    )
    assert result.count("[Ali ce] ") == 1


@pytest.mark.asyncio
async def test_preprocess_keeps_plain_text_for_default_group_sessions():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"


@pytest.mark.asyncio
async def test_preprocess_keeps_shared_group_untagged_without_sender_name():
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_type="group",
        user_name=None,
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"
