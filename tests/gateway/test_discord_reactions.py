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

from plugins.platforms.discord import adapter as discord_adapter_module  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from plugins.platforms.discord.reaction_actions import dispatch_raw_reaction_add  # noqa: E402


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
    adapter._allowed_user_ids = {"42"}
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


class _ReactionChannel:
    def __init__(self, channel_id, messages=None, parent=None):
        self.id = int(channel_id)
        self._messages = messages or {}
        self.parent = parent

    async def fetch_message(self, message_id):
        message = self._messages.get(int(message_id))
        if message is None:
            raise RuntimeError("Unknown Message")
        return message


def _reaction_target(message_id):
    return SimpleNamespace(
        id=int(message_id), add_reaction=AsyncMock(), remove_reaction=AsyncMock()
    )


@pytest.mark.asyncio
async def test_adapter_add_reaction_targets_explicit_message(adapter):
    message = _reaction_target(555)
    channel = _ReactionChannel(123, {555: message})
    adapter._client.get_channel = lambda channel_id: channel if channel_id == 123 else None

    result = await adapter.add_reaction("123", "🟢", message_id="555")

    assert result == {"success": True, "message_id": "555"}
    message.add_reaction.assert_awaited_once_with("🟢")


@pytest.mark.asyncio
async def test_adapter_reaction_uses_thread_parent_and_ignores_lifecycle_toggle(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    message = _reaction_target(777)
    parent = _ReactionChannel(123, {777: message})
    thread = _ReactionChannel(888, parent=parent)
    adapter._client.get_channel = lambda channel_id: {123: parent, 888: thread}.get(channel_id)

    result = await adapter.add_reaction("888", "🔴", message_id="777")

    assert result == {"success": True, "message_id": "777"}
    message.add_reaction.assert_awaited_once_with("🔴")


@pytest.mark.asyncio
async def test_adapter_remove_reaction_removes_bot_own_tracked_emoji(adapter):
    message = _reaction_target(555)
    channel = _ReactionChannel(123, {555: message})
    adapter._client.get_channel = lambda channel_id: channel if channel_id == 123 else None

    await adapter.add_reaction("123", "🟢", message_id="555")
    result = await adapter.remove_reaction("123", message_id="555")

    assert result == {"success": True, "message_id": "555"}
    message.remove_reaction.assert_awaited_once_with("🟢", adapter._client.user)


# ---------------------------------------------------------------------------
# on_discord_reaction_add plugin hook (raw reaction events)
# ---------------------------------------------------------------------------


def _make_reaction_payload(user_id=42, member=None, emoji="\u2705"):
    return SimpleNamespace(
        user_id=user_id,
        channel_id="123",
        message_id="456",
        guild_id=None,
        emoji=emoji,
        member=member,
    )


async def _dispatch_reaction(adapter, payload):
    await dispatch_raw_reaction_add(
        adapter,
        payload,
        discord_module=discord_adapter_module.discord,
    )


def _fake_reaction_channel(message, *, channel_id=123, name="email-important"):
    return SimpleNamespace(
        id=channel_id,
        name=name,
        fetch_message=AsyncMock(return_value=message),
        send=AsyncMock(),
    )


def _fake_bot_message(content="bot message"):
    return SimpleNamespace(
        author=SimpleNamespace(id=99999, bot=True),
        content=content,
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )


def _install_reaction_hook(monkeypatch, results):
    """Point the adapter's in-function imports at controllable fakes."""
    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(
        lifecycle, "has_hook", lambda name: name == "on_discord_reaction_add"
    )
    invoke = MagicMock(return_value=results)
    monkeypatch.setattr(
        lifecycle, "invoke_hook", lambda name, **payload: invoke(**payload)
    )
    return invoke


def _client_with_channel(channel, *, get_user=None):
    return SimpleNamespace(
        user=SimpleNamespace(id=99999, name="HermesBot"),
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(),
        get_user=get_user,
    )


@pytest.mark.asyncio
async def test_raw_reaction_add_no_hook_subscribed_does_nothing(adapter, monkeypatch):
    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: False)
    fetch_channel = AsyncMock()
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=99999, name="HermesBot"),
        get_channel=lambda _id: None,
        fetch_channel=fetch_channel,
    )

    await _dispatch_reaction(adapter, _make_reaction_payload())

    fetch_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_reaction_add_bot_self_reaction_ignored(adapter, monkeypatch):
    _install_reaction_hook(monkeypatch, [])
    fetch_channel = AsyncMock()
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=99999, name="HermesBot"),
        get_channel=lambda _id: None,
        fetch_channel=fetch_channel,
    )

    await _dispatch_reaction(adapter, _make_reaction_payload(user_id=99999))

    fetch_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_reaction_add_user_on_bot_message_invokes_hook(adapter, monkeypatch):
    invoke = _install_reaction_hook(monkeypatch, [])
    channel = _fake_reaction_channel(_fake_bot_message())
    adapter._client = _client_with_channel(channel)

    await _dispatch_reaction(adapter, _make_reaction_payload())

    invoke.assert_called_once()
    kwargs = invoke.call_args.kwargs
    assert kwargs["emoji"] == "\u2705"
    assert kwargs["user_id"] == "42"
    assert kwargs["channel_id"] == "123"
    assert kwargs["message_id"] == "456"
    assert kwargs["message_content"] == "bot message"
    assert kwargs["message_author_id"] == "99999"


@pytest.mark.asyncio
async def test_raw_reaction_add_non_allowed_channel_does_not_fetch_message(
    adapter, monkeypatch
):
    invoke = _install_reaction_hook(monkeypatch, [])
    channel = _fake_reaction_channel(_fake_bot_message(), channel_id=123)
    adapter._client = _client_with_channel(channel)
    adapter._allowed_user_ids = set()
    monkeypatch.setattr(adapter, "_get_allowed_channels", lambda: {"999"})
    monkeypatch.setattr(adapter, "_get_ignored_channels", lambda: set())

    await _dispatch_reaction(adapter, _make_reaction_payload())

    invoke.assert_not_called()
    channel.fetch_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_reaction_add_allowed_channel_invokes_without_user_allowlist(
    adapter, monkeypatch
):
    invoke = _install_reaction_hook(monkeypatch, [])
    channel = _fake_reaction_channel(_fake_bot_message(), channel_id=123)
    adapter._client = _client_with_channel(channel)
    adapter._allowed_user_ids = set()
    monkeypatch.setattr(adapter, "_get_allowed_channels", lambda: {"123"})
    monkeypatch.setattr(adapter, "_get_ignored_channels", lambda: set())

    await _dispatch_reaction(adapter, _make_reaction_payload())

    invoke.assert_called_once()
    channel.fetch_message.assert_awaited_once_with(456)


@pytest.mark.asyncio
async def test_raw_reaction_add_ignored_channel_does_not_fetch_message(
    adapter, monkeypatch
):
    invoke = _install_reaction_hook(monkeypatch, [])
    channel = _fake_reaction_channel(_fake_bot_message(), channel_id=123)
    adapter._client = _client_with_channel(channel)
    monkeypatch.setattr(adapter, "_get_allowed_channels", lambda: {"123"})
    monkeypatch.setattr(adapter, "_get_ignored_channels", lambda: {"123"})

    await _dispatch_reaction(adapter, _make_reaction_payload())

    invoke.assert_not_called()
    channel.fetch_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_reaction_add_hook_send_message_sends_followup(adapter, monkeypatch):
    _install_reaction_hook(
        monkeypatch,
        [{"action": "send_message", "content": "follow-up!"}],
    )
    channel = _fake_reaction_channel(_fake_bot_message())
    adapter._client = _client_with_channel(channel)

    await _dispatch_reaction(adapter, _make_reaction_payload())

    channel.send.assert_awaited_once_with(content="follow-up!")


@pytest.mark.asyncio
async def test_raw_reaction_add_hook_remove_user_reaction_removes_when_resolvable(
    adapter, monkeypatch
):
    member = SimpleNamespace(id=42, name="Jezza")
    _install_reaction_hook(
        monkeypatch,
        [{"action": "remove_user_reaction", "emoji": "\u2705"}],
    )
    message = _fake_bot_message()
    channel = _fake_reaction_channel(message)
    adapter._client = _client_with_channel(channel)

    await _dispatch_reaction(adapter, _make_reaction_payload(member=member))

    message.remove_reaction.assert_awaited_once_with("\u2705", member)
