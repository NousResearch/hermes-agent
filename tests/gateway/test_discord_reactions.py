"""Tests for Discord message reactions tied to processing lifecycle hooks."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome, SendResult
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


# --- Agent-facing reactions (send_message action="react") -------------------


class FakeChannel:
    """Minimal channel double: fetch_message succeeds only for known ids."""

    def __init__(self, channel_id, messages=None, parent=None):
        self.id = int(channel_id)
        self._messages = messages or {}
        self.parent = parent

    async def fetch_message(self, message_id):
        message = self._messages.get(int(message_id))
        if message is None:
            raise RuntimeError("404 Not Found (error code: 10008): Unknown Message")
        return message


def _fake_message(message_id):
    return SimpleNamespace(
        id=int(message_id),
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )


def _wire_channels(adapter, *channels):
    by_id = {channel.id: channel for channel in channels}
    adapter._client.get_channel = lambda cid: by_id.get(int(cid))


async def _run_turn(adapter, event):
    """Drive one full inbound turn so lifecycle state is recorded."""

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing
    await adapter._process_message_background(event, build_session_key(event.source))


@pytest.mark.asyncio
async def test_add_reaction_targets_explicit_message_id(adapter):
    message = _fake_message(555)
    _wire_channels(adapter, FakeChannel(123, {555: message}))

    result = await adapter.add_reaction(chat_id="123", emoji="🟢", message_id="555")

    assert result == {"success": True, "message_id": "555"}
    message.add_reaction.assert_awaited_once_with("🟢")


@pytest.mark.asyncio
async def test_add_reaction_defaults_to_the_triggering_message(adapter):
    """No message_id: react to whatever started this turn (photon precedent)."""
    message = _fake_message(777)
    _wire_channels(adapter, FakeChannel(123, {777: message}))
    await _run_turn(adapter, _make_event("777", _fake_message(777)))

    result = await adapter.add_reaction(chat_id="123", emoji="🟡")

    assert result == {"success": True, "message_id": "777"}
    message.add_reaction.assert_awaited_once_with("🟡")


@pytest.mark.asyncio
async def test_add_reaction_falls_back_to_the_thread_parent(adapter):
    """An auto-threaded turn runs in the thread; its trigger is in the parent."""
    message = _fake_message(777)
    parent = FakeChannel(123, {777: message})
    _wire_channels(adapter, parent, FakeChannel(888, {}, parent=parent))

    result = await adapter.add_reaction(chat_id="888", emoji="🔴", message_id="777")

    assert result == {"success": True, "message_id": "777"}
    message.add_reaction.assert_awaited_once_with("🔴")


@pytest.mark.asyncio
async def test_add_reaction_is_not_gated_by_the_lifecycle_env(adapter, monkeypatch):
    """DISCORD_REACTIONS mutes the automatic ack, not deliberate agent intent."""
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    message = _fake_message(555)
    _wire_channels(adapter, FakeChannel(123, {555: message}))

    result = await adapter.add_reaction(chat_id="123", emoji="🟢", message_id="555")

    assert result["success"] is True
    message.add_reaction.assert_awaited_once_with("🟢")


@pytest.mark.asyncio
async def test_add_reaction_without_a_target_reports_why(adapter):
    _wire_channels(adapter, FakeChannel(123, {}))

    result = await adapter.add_reaction(chat_id="123", emoji="🟢")

    assert result["success"] is False
    assert "message_id" in result["error"]


@pytest.mark.asyncio
async def test_add_reaction_reports_an_unknown_message(adapter):
    _wire_channels(adapter, FakeChannel(123, {}))

    result = await adapter.add_reaction(chat_id="123", emoji="🟢", message_id="555")

    assert result["success"] is False
    assert "555" in result["error"]


@pytest.mark.asyncio
async def test_remove_reaction_retracts_the_emoji_we_added(adapter):
    message = _fake_message(555)
    _wire_channels(adapter, FakeChannel(123, {555: message}))
    await adapter.add_reaction(chat_id="123", emoji="🟢", message_id="555")

    result = await adapter.remove_reaction(chat_id="123", message_id="555")

    assert result == {"success": True, "message_id": "555"}
    message.remove_reaction.assert_awaited_once_with("🟢", adapter._client.user)


@pytest.mark.asyncio
async def test_remove_reaction_without_a_tracked_emoji_reports_why(adapter):
    message = _fake_message(555)
    _wire_channels(adapter, FakeChannel(123, {555: message}))

    result = await adapter.remove_reaction(chat_id="123", message_id="555")

    assert result["success"] is False
    message.remove_reaction.assert_not_awaited()
