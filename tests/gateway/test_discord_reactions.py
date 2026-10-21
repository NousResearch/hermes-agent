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

from plugins.platforms.discord.adapter import DiscordAdapter


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




def _ack_raw_message(message_id):
    return SimpleNamespace(id=message_id, add_reaction=AsyncMock(), remove_reaction=AsyncMock())


@pytest.mark.asyncio
async def test_ack_mode_keeps_eyes_on_answered_message_and_never_ticks(adapter, monkeypatch):
    """reactions=ack: 👀 stays as a "seen it" marker once the bot replied; no ✅."""
    monkeypatch.setenv("DISCORD_REACTIONS", "ack")
    raw_message = _ack_raw_message(11)
    event = _make_event("11", raw_message)

    await adapter.on_processing_start(event)
    adapter._last_conversational_send_at["123"] = __import__("time").monotonic()
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert [c.args for c in raw_message.add_reaction.await_args_list] == [("👀",)]
    raw_message.remove_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_ack_mode_removes_eyes_when_turn_was_silent(adapter, monkeypatch):
    """A silent turn means the message was not for the bot: take the 👀 back off."""
    monkeypatch.setenv("DISCORD_REACTIONS", "ack")
    raw_message = _ack_raw_message(12)
    event = _make_event("12", raw_message)
    adapter._last_conversational_send_at["123"] = 0.0  # an older reply must not count

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert [c.args for c in raw_message.add_reaction.await_args_list] == [("👀",)]
    assert raw_message.remove_reaction.await_args_list[0].args == ("👀", adapter._client.user)


@pytest.mark.asyncio
async def test_ack_mode_failure_swaps_to_cross(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REACTIONS", "ack")
    raw_message = _ack_raw_message(13)
    event = _make_event("13", raw_message)

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert [c.args for c in raw_message.add_reaction.await_args_list] == [("👀",), ("❌",)]
    assert raw_message.remove_reaction.await_args_list[0].args == ("👀", adapter._client.user)


@pytest.mark.asyncio
async def test_ack_mode_end_to_end_with_reply(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REACTIONS", "ack")
    raw_message = _ack_raw_message(14)

    async def handler(_event):
        await asyncio.sleep(0)
        return "on it"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    async def fake_send(chat_id, *_a, **_k):
        adapter._last_conversational_send_at[str(chat_id)] = __import__("time").monotonic()
        return SendResult(success=True, message_id="999")

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(side_effect=fake_send)
    adapter._keep_typing = hold_typing

    event = _make_event("14", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    assert [c.args for c in raw_message.add_reaction.await_args_list] == [("👀",)]
    raw_message.remove_reaction.assert_not_awaited()


@pytest.mark.parametrize("value,mode", [
    ("ack", "ack"), ("seen", "ack"), ("true", "full"), ("false", "off"), ("off", "off"),
])
def test_reaction_mode_parsing(adapter, monkeypatch, value, mode):
    monkeypatch.setenv("DISCORD_REACTIONS", value)
    assert adapter._reaction_mode() == mode
    assert adapter._reactions_enabled() is (mode == "full")
