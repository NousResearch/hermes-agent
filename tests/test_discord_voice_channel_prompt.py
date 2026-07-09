from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _NullClient:
    def get_channel(self, channel_id: int):
        return None


class _FakeVoiceChannel:
    name = "General"


class _FakeDiscordAdapter:
    def __init__(self, *, stored_prompt: str | None = None) -> None:
        self._voice_text_channels = {42: 123}
        self._voice_sources = {
            42: SessionSource(
                platform=Platform.DISCORD,
                chat_id="123",
                chat_type="channel",
                user_id="join-user",
                parent_chat_id="999",
            ).to_dict()
        }
        self._voice_channel_prompts = {42: stored_prompt} if stored_prompt is not None else {}
        self._client = _NullClient()
        self._voice_input_callback = None
        self._on_voice_disconnect = None
        self._voice_mode_getter = None
        self.handled_events = []
        self.resolve_calls = []

    async def get_user_voice_channel(self, guild_id: int, user_id: str) -> _FakeVoiceChannel:
        return _FakeVoiceChannel()

    async def join_voice_channel(self, voice_channel: _FakeVoiceChannel) -> bool:
        return True

    def _resolve_channel_prompt(self, channel_id: str, parent_id: str | None = None) -> str | None:
        self.resolve_calls.append((channel_id, parent_id))
        if channel_id == "123" and parent_id == "999":
            return "DRIVE MODE PROMPT"
        return None

    async def handle_message(self, event):
        self.handled_events.append(event)


def _voice_runner(adapter: _FakeDiscordAdapter):
    runner = cast(Any, GatewayRunner.__new__(GatewayRunner))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_mode = {}
    runner._save_voice_modes = lambda: None
    runner._set_adapter_auto_tts_enabled = lambda adapter, chat_id, enabled=True: None
    runner._set_adapter_auto_tts_disabled = lambda adapter, chat_id, disabled=True: None
    return runner


def test_discord_voice_join_clears_stale_prompt_when_rebound_to_unprompted_channel() -> None:
    adapter = _FakeDiscordAdapter(stored_prompt="OLD CHANNEL PROMPT")
    runner = _voice_runner(adapter)
    event = MessageEvent(
        text="/voice join",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="456",
            chat_type="channel",
            user_id="join-user",
        ),
        raw_message=SimpleNamespace(guild_id=42),
        channel_prompt=None,
    )

    result = asyncio.run(GatewayRunner._handle_voice_channel_join(runner, event))

    assert "Joined voice channel" in result
    assert adapter._voice_text_channels[42] == 456
    assert 42 not in adapter._voice_channel_prompts


def test_discord_voice_input_carries_resolved_channel_prompt() -> None:
    adapter = _FakeDiscordAdapter()
    runner = _voice_runner(adapter)
    runner._is_user_authorized = lambda source: True
    runner._is_duplicate_voice_transcript = lambda guild_id, user_id, transcript: False

    asyncio.run(
        GatewayRunner._handle_voice_channel_input(
            runner,
            guild_id=42,
            user_id=777,
            transcript="give me a ten second systems check",
        )
    )

    assert len(adapter.handled_events) == 1
    event = adapter.handled_events[0]
    assert event.message_type is MessageType.VOICE
    assert event.channel_prompt == "DRIVE MODE PROMPT"
    assert adapter.resolve_calls == [("123", "999")]
    assert event.raw_message.guild_id == 42


def test_discord_voice_input_prefers_prompt_captured_at_voice_join() -> None:
    runner = cast(Any, GatewayRunner.__new__(GatewayRunner))
    adapter = _FakeDiscordAdapter(stored_prompt="STORED DRIVE MODE PROMPT")
    runner.adapters = {Platform.DISCORD: adapter}
    runner._is_user_authorized = lambda source: True
    runner._is_duplicate_voice_transcript = lambda guild_id, user_id, transcript: False

    asyncio.run(
        GatewayRunner._handle_voice_channel_input(
            runner,
            guild_id=42,
            user_id=777,
            transcript="repeat that shorter",
        )
    )

    assert len(adapter.handled_events) == 1
    event = adapter.handled_events[0]
    assert event.message_type is MessageType.VOICE
    assert event.channel_prompt == "STORED DRIVE MODE PROMPT"
    assert adapter.resolve_calls == []
    assert event.raw_message.guild_id == 42
