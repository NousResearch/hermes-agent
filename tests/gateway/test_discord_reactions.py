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
async def test_interaction_backed_events_do_not_attempt_reactions(adapter):
    interaction = SimpleNamespace(guild_id=123456789)

    async def handler(_event):
        await asyncio.sleep(0)
        return None

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter._add_reaction = AsyncMock()
    adapter._remove_reaction = AsyncMock()
    adapter._keep_typing = hold_typing

    event = MessageEvent(
        text="/status",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            chat_type="dm",
            user_id="42",
            user_name="Jezza",
        ),
        raw_message=interaction,
        message_id="2",
    )

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter._add_reaction.assert_not_awaited()
    adapter._remove_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_reaction_helper_failures_do_not_break_message_flow(adapter):
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(side_effect=[RuntimeError("no perms"), RuntimeError("no perms")]),
        remove_reaction=AsyncMock(side_effect=RuntimeError("no perms")),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("3", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send.assert_awaited_once()


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
async def test_reactions_disabled_via_env_zero(adapter, monkeypatch):
    """DISCORD_REACTIONS=0 should also disable reactions."""
    monkeypatch.setenv("DISCORD_REACTIONS", "0")

    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    event = _make_event("5", raw_message)
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    raw_message.add_reaction.assert_not_awaited()
    raw_message.remove_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_reactions_enabled_by_default(adapter, monkeypatch):
    """When DISCORD_REACTIONS is unset, reactions should still work (default: true)."""
    monkeypatch.delenv("DISCORD_REACTIONS", raising=False)

    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    event = _make_event("6", raw_message)
    await adapter.on_processing_start(event)

    raw_message.add_reaction.assert_awaited_once_with("👀")


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_removes_eyes_without_terminal_reaction(adapter):
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    event = _make_event("7", raw_message)
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    raw_message.remove_reaction.assert_awaited_once_with("👀", adapter._client.user)
    raw_message.add_reaction.assert_not_awaited()


# ---------------------------------------------------------------------------
# Inbound reaction event tests
# ---------------------------------------------------------------------------


class _MockEmoji:
    """Mock emoji that stringifies to the given value."""
    def __init__(self, value: str):
        self._value = value
    def __str__(self):
        return self._value


def _make_reaction_payload(
    user_id: int = 42,
    channel_id: int = 123,
    message_id: int = 1,
    emoji: str = "👍",
    member=None,
    guild_id: int | None = None,
) -> SimpleNamespace:
    """Create a mock RawReactionActionEvent payload."""
    return SimpleNamespace(
        user_id=user_id,
        channel_id=channel_id,
        message_id=message_id,
        emoji=_MockEmoji(emoji),
        member=member,
        guild_id=guild_id,
    )


def _make_mock_channel(
    bot_user,
    channel_id: int = 123,
    guild=None,
    channel_name: str = "test-channel",
) -> SimpleNamespace:
    """Create a mock Discord channel whose fetch_message returns a bot-authored message."""
    bot_message = SimpleNamespace(
        author=bot_user,
        id=1,
    )
    channel = SimpleNamespace(
        id=channel_id,
        name=channel_name,
        guild=guild,
        fetch_message=AsyncMock(return_value=bot_message),
    )
    return channel


@pytest.mark.asyncio
async def test_inbound_reaction_routes_as_synthetic_text_event(adapter):
    """A reaction on a bot message should be routed as a synthetic text event."""
    adapter.handle_message = AsyncMock()
    bot_user = adapter._client.user

    channel = _make_mock_channel(bot_user)
    adapter._client.get_channel = MagicMock(return_value=channel)

    payload = _make_reaction_payload(user_id=42, emoji="👍")

    await adapter._handle_inbound_reaction(payload, "added")

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert isinstance(event, MessageEvent)
    assert event.text == "reaction:added:👍"
    assert event.message_type == MessageType.TEXT
    assert event.source.user_id == "42"


@pytest.mark.asyncio
async def test_inbound_reaction_on_non_bot_message_ignored(adapter):
    """Reactions on messages NOT authored by the bot should be ignored."""
    adapter.handle_message = AsyncMock()

    non_bot_message = SimpleNamespace(
        author=SimpleNamespace(id=12345, name="SomeOtherUser"),
        id=1,
    )
    channel = SimpleNamespace(
        id=123,
        guild=None,
        fetch_message=AsyncMock(return_value=non_bot_message),
    )
    adapter._client.get_channel = MagicMock(return_value=channel)

    payload = _make_reaction_payload()
    await adapter._handle_inbound_reaction(payload, "added")

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_reaction_from_bot_ignored(adapter):
    """The bot's own reactions should be ignored to prevent feedback loops."""
    adapter.handle_message = AsyncMock()

    # Bot reacting on its own message — user_id matches client.user.id
    payload = _make_reaction_payload(user_id=99999)

    await adapter._handle_inbound_reaction(payload, "added")

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_reaction_disallowed_user_ignored(adapter):
    """Reactions from users not in the allowlist should be ignored."""
    adapter.handle_message = AsyncMock()
    adapter._allowed_user_ids = {"100", "200"}

    payload = _make_reaction_payload(user_id=42)  # 42 not in allowlist

    await adapter._handle_inbound_reaction(payload, "added")

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_reaction_disabled_via_env(adapter, monkeypatch):
    """DISCORD_REACTIONS=false should block inbound reaction routing."""
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    adapter.handle_message = AsyncMock()

    payload = _make_reaction_payload()

    await adapter._handle_inbound_reaction(payload, "added")

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_reaction_remove_routes_correctly(adapter):
    """Reaction removal events should use 'removed' in the synthetic text."""
    adapter.handle_message = AsyncMock()
    bot_user = adapter._client.user

    channel = _make_mock_channel(bot_user)
    adapter._client.get_channel = MagicMock(return_value=channel)

    payload = _make_reaction_payload(emoji="👍")

    await adapter._handle_inbound_reaction(payload, "removed")

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "reaction:removed:👍"


@pytest.mark.asyncio
async def test_inbound_reaction_fetches_uncached_channel(adapter):
    """When get_channel returns None, fetch_channel should be called."""
    adapter.handle_message = AsyncMock()
    bot_user = adapter._client.user

    channel = _make_mock_channel(bot_user)
    adapter._client.get_channel = MagicMock(return_value=None)
    adapter._client.fetch_channel = AsyncMock(return_value=channel)

    payload = _make_reaction_payload()

    await adapter._handle_inbound_reaction(payload, "added")

    adapter._client.fetch_channel.assert_awaited_once_with(123)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_inbound_reaction_with_guild_resolves_chat_name(adapter):
    """Guild reactions should include server name in the source chat_name."""
    adapter.handle_message = AsyncMock()
    bot_user = adapter._client.user

    guild = SimpleNamespace(id=999, name="Test Server")
    channel = _make_mock_channel(bot_user, guild=guild, channel_name="general")
    adapter._client.get_channel = MagicMock(return_value=channel)

    payload = _make_reaction_payload(guild_id=999)

    await adapter._handle_inbound_reaction(payload, "added")

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "group"
    assert "Test Server" in event.source.chat_name
    assert "#general" in event.source.chat_name


@pytest.mark.asyncio
async def test_inbound_reaction_member_display_name_used(adapter):
    """When payload.member has display_name, it should be used as user_name."""
    adapter.handle_message = AsyncMock()
    bot_user = adapter._client.user

    channel = _make_mock_channel(bot_user)
    adapter._client.get_channel = MagicMock(return_value=channel)

    member = SimpleNamespace(display_name="CoolUser")
    payload = _make_reaction_payload(member=member)

    await adapter._handle_inbound_reaction(payload, "added")

    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_name == "CoolUser"


@pytest.mark.asyncio
async def test_inbound_reaction_channel_fetch_failure_handled(adapter):
    """If fetch_channel raises, the handler should not crash."""
    adapter.handle_message = AsyncMock()
    adapter._client.get_channel = MagicMock(return_value=None)
    adapter._client.fetch_channel = AsyncMock(side_effect=Exception("channel not found"))

    payload = _make_reaction_payload()

    # Should not raise
    await adapter._handle_inbound_reaction(payload, "added")

    adapter.handle_message.assert_not_awaited()
