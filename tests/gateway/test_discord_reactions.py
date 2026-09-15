"""Tests for Discord message reactions tied to processing lifecycle hooks."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource, build_session_key


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._allowed_user_ids = {"42"}
    return adapter


def _make_event(message_id: str, raw_message) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            chat_type="dm",
            user_id="42",
            user_name="Jezza",
        ),
        raw_message=raw_message,
        message_id=message_id,
    )


@pytest.mark.asyncio
async def test_process_message_background_adds_and_swaps_reactions(adapter):
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("1", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    assert raw_message.add_reaction.await_args_list[0].args == ("👀",)
    assert raw_message.remove_reaction.await_args_list[0].args == ("👀", adapter._client.user)
    assert raw_message.add_reaction.await_args_list[1].args == ("✅",)


@pytest.mark.asyncio
async def test_reactions_disabled_via_env(adapter, monkeypatch):
    """When DISCORD_REACTIONS=false, no reactions should be added."""
    monkeypatch.setenv("DISCORD_REACTIONS", "false")

    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("4", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    raw_message.add_reaction.assert_not_awaited()
    raw_message.remove_reaction.assert_not_awaited()
    # Response should still be sent
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_speaker_reaction_fetches_one_bot_reply_and_requests_audio(adapter, monkeypatch):
    """An authorized speaker reaction reads exactly the reacted-to bot response."""
    channel = SimpleNamespace(
        id=123,
        fetch_message=AsyncMock(return_value=SimpleNamespace(
            author=SimpleNamespace(id=99999), content="**Visible** reply",
        )),
    )
    adapter._client.get_channel = lambda _id: channel
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="status"))
    speak = AsyncMock()
    monkeypatch.setattr(adapter, "_send_tts_reaction_audio", speak)
    payload = SimpleNamespace(message_id=88, channel_id=123, user_id=42, emoji="🔊")

    assert await adapter._on_tts_reaction(payload) is True
    assert await adapter._on_tts_reaction(payload) is False
    adapter.send.assert_awaited_once_with(
        "123", "🎙️ Generating audio…", reply_to="88", metadata={"non_conversational": True},
    )
    speak.assert_awaited_once_with(chat_id="123", text="**Visible** reply", reply_to="88")


def test_speaker_reaction_strips_transport_only_media_directives(adapter):
    assert adapter._visible_tts_reaction_text(
        "Visible text\n[[audio_as_voice]]\nMEDIA:/tmp/voice-message.ogg\n[[as_document]]"
    ) == "Visible text"


@pytest.mark.asyncio
async def test_speaker_reaction_retries_tts_six_times_before_one_failure_notice(adapter, monkeypatch):
    """Transient TTS faults wait ten seconds and retry without making the user re-react."""
    attempt = AsyncMock(side_effect=[RuntimeError("provider unavailable")] * 6)
    sleep = AsyncMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="failure"))
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr(adapter, "_attempt_tts_reaction_delivery", attempt, raising=False)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    await adapter._send_tts_reaction_audio(chat_id="123", text="Visible reply", reply_to="88")

    assert attempt.await_count == 6
    assert sleep.await_args_list == [((10,),)] * 5
    adapter.send.assert_awaited_once_with(
        "123", "🎙️ Audio generation failed after 6 attempts. Please try again later.",
        reply_to="88", metadata={"non_conversational": True},
    )


@pytest.mark.asyncio
async def test_speaker_reaction_during_processing_voices_the_complete_unsplit_response(adapter, monkeypatch):
    """Reacting to the source message before ✅ arms one TTS job for the full final response."""
    raw_message = SimpleNamespace(
        author=SimpleNamespace(id=42), add_reaction=AsyncMock(), remove_reaction=AsyncMock(),
    )
    event = _make_event("1", raw_message)
    monkeypatch.setattr(adapter, "_record_discord_processing_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "_record_discord_processing_complete", lambda *args, **kwargs: None)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="status"))
    speak = AsyncMock()
    monkeypatch.setattr(adapter, "_send_tts_reaction_audio", speak)

    await adapter.on_processing_start(event)
    payload = SimpleNamespace(message_id=1, channel_id=123, user_id=42, emoji="🔈")
    assert await adapter._on_tts_reaction(payload) is True
    adapter._remember_tts_reaction_response(reply_to="1", content="full response across all Discord chunks")
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    speak.assert_awaited_once_with(
        chat_id="123", text="full response across all Discord chunks", reply_to="1",
    )


