"""Tests for native Discord slash command fast-paths (thread creation & auto-thread)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from gateway.config import PlatformConfig


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

from gateway.platforms.discord import DiscordAdapter  # noqa: E402


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


# ------------------------------------------------------------------
# /thread slash command registration
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registers_native_thread_slash_command(adapter):
    adapter._handle_thread_create_slash = AsyncMock()
    adapter._register_slash_commands()

    command = adapter._client.tree.commands["thread"]
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock()),
    )

    await command(interaction, name="Planning", message="", auto_archive_duration=1440)

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    adapter._handle_thread_create_slash.assert_awaited_once_with(interaction, "Planning", "", 1440)


# ------------------------------------------------------------------
# _handle_thread_create_slash — success, session dispatch, failure
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_thread_create_slash_reports_success(adapter):
    created_thread = SimpleNamespace(id=555, name="Planning", send=AsyncMock())
    parent_channel = SimpleNamespace(create_thread=AsyncMock(return_value=created_thread), send=AsyncMock())
    interaction_channel = SimpleNamespace(parent=parent_channel)
    interaction = SimpleNamespace(
        channel=interaction_channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "Kickoff", 1440)

    parent_channel.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )
    created_thread.send.assert_awaited_once_with("Kickoff")
    # Thread link shown to user
    interaction.followup.send.assert_awaited()
    args, kwargs = interaction.followup.send.await_args
    assert "<#555>" in args[0]
    assert kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_handle_thread_create_slash_dispatches_session_when_message_provided(adapter):
    """When a message is given, _dispatch_thread_session should be called."""
    created_thread = SimpleNamespace(id=555, name="Planning", send=AsyncMock())
    parent_channel = SimpleNamespace(create_thread=AsyncMock(return_value=created_thread))
    interaction = SimpleNamespace(
        channel=SimpleNamespace(parent=parent_channel),
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    adapter._dispatch_thread_session = AsyncMock()

    await adapter._handle_thread_create_slash(interaction, "Planning", "Hello Hermes", 1440)

    adapter._dispatch_thread_session.assert_awaited_once_with(
        interaction, "555", "Planning", "Hello Hermes",
    )


@pytest.mark.asyncio
async def test_handle_thread_create_slash_no_dispatch_without_message(adapter):
    """Without a message, no session dispatch should occur."""
    created_thread = SimpleNamespace(id=555, name="Planning", send=AsyncMock())
    parent_channel = SimpleNamespace(create_thread=AsyncMock(return_value=created_thread))
    interaction = SimpleNamespace(
        channel=SimpleNamespace(parent=parent_channel),
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    adapter._dispatch_thread_session = AsyncMock()

    await adapter._handle_thread_create_slash(interaction, "Planning", "", 1440)

    adapter._dispatch_thread_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_thread_create_slash_falls_back_to_seed_message(adapter):
    created_thread = SimpleNamespace(id=555, name="Planning")
    seed_message = SimpleNamespace(id=777, create_thread=AsyncMock(return_value=created_thread))
    channel = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("direct failed")),
        send=AsyncMock(return_value=seed_message),
    )
    interaction = SimpleNamespace(
        channel=channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "Kickoff", 1440)

    channel.send.assert_awaited_once_with("Kickoff")
    seed_message.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )
    interaction.followup.send.assert_awaited()


@pytest.mark.asyncio
async def test_handle_thread_create_slash_reports_failure(adapter):
    channel = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("direct failed")),
        send=AsyncMock(side_effect=RuntimeError("nope")),
    )
    interaction = SimpleNamespace(
        channel=channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "", 1440)

    interaction.followup.send.assert_awaited_once()
    args, kwargs = interaction.followup.send.await_args
    assert "Failed to create thread:" in args[0]
    assert "nope" in args[0]
    assert kwargs["ephemeral"] is True


# ------------------------------------------------------------------
# _dispatch_thread_session — builds correct event and routes it
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_thread_session_builds_thread_event(adapter):
    """Dispatched event should have chat_type=thread and chat_id=thread_id."""
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
    )

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    await adapter._dispatch_thread_session(interaction, "555", "Planning", "Hello!")

    assert len(captured_events) == 1
    event = captured_events[0]
    assert event.text == "Hello!"
    assert event.source.chat_id == "555"
    assert event.source.chat_type == "thread"
    assert event.source.thread_id == "555"
    assert "TestGuild" in event.source.chat_name


# ------------------------------------------------------------------
# Auto-thread: _auto_create_thread
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_create_thread_uses_message_content_as_name(adapter):
    thread = SimpleNamespace(id=999, name="Hello world")
    message = SimpleNamespace(
        content="Hello world, how are you?",
        create_thread=AsyncMock(return_value=thread),
    )

    result = await adapter._auto_create_thread(message)

    assert result is thread
    message.create_thread.assert_awaited_once()
    call_kwargs = message.create_thread.await_args[1]
    assert call_kwargs["name"] == "Hello world, how are you?"
    assert call_kwargs["auto_archive_duration"] == 1440


@pytest.mark.asyncio
async def test_auto_create_thread_truncates_long_names(adapter):
    long_text = "a" * 200
    thread = SimpleNamespace(id=999, name="truncated")
    message = SimpleNamespace(
        content=long_text,
        create_thread=AsyncMock(return_value=thread),
    )

    result = await adapter._auto_create_thread(message)

    assert result is thread
    call_kwargs = message.create_thread.await_args[1]
    assert len(call_kwargs["name"]) <= 80
    assert call_kwargs["name"].endswith("...")


@pytest.mark.asyncio
async def test_auto_create_thread_returns_none_on_failure(adapter):
    message = SimpleNamespace(
        content="Hello",
        create_thread=AsyncMock(side_effect=RuntimeError("no perms")),
    )

    result = await adapter._auto_create_thread(message)
    assert result is None


# ------------------------------------------------------------------
# Auto-thread integration in _handle_message
# ------------------------------------------------------------------


import discord as _discord_mod  # noqa: E402 — mock or real, used below


class _FakeTextChannel:
    """A channel that is NOT a discord.Thread or discord.DMChannel."""

    def __init__(self, channel_id=100, name="general", guild_name="TestGuild"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None


class _FakeThreadChannel(_discord_mod.Thread):
    """isinstance(ch, discord.Thread) → True."""

    def __init__(self, channel_id=200, name="existing-thread", guild_name="TestGuild", parent_id=100):
        # Don't call super().__init__ — mock Thread is just an empty type
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None
        self.parent = SimpleNamespace(id=parent_id, name="general", guild=SimpleNamespace(name=guild_name, id=1))


def _fake_message(channel, *, content="Hello", author_id=42, display_name="Jezza"):
    return SimpleNamespace(
        author=SimpleNamespace(id=author_id, display_name=display_name, bot=False),
        content=content,
        channel=channel,
        attachments=[],
        mentions=[],
        reference=None,
        created_at=None,
        id=12345,
    )


@pytest.mark.asyncio
async def test_auto_thread_creates_thread_and_redirects(adapter, monkeypatch):
    """When DISCORD_AUTO_THREAD=true, a new thread is created and the event routes there."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    thread = SimpleNamespace(id=999, name="Hello")
    adapter._auto_create_thread = AsyncMock(return_value=thread)

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    msg = _fake_message(_FakeTextChannel(), content="Hello world")

    await adapter._handle_message(msg)

    adapter._auto_create_thread.assert_awaited_once_with(msg)
    assert len(captured_events) == 1
    event = captured_events[0]
    assert event.source.chat_id == "999"  # redirected to thread
    assert event.source.chat_type == "thread"
    assert event.source.thread_id == "999"


@pytest.mark.asyncio
async def test_auto_thread_enabled_by_default_slash_commands(adapter, monkeypatch):
    """Without DISCORD_AUTO_THREAD env var, auto-threading is enabled (default: true)."""
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    fake_thread = _FakeThreadChannel(channel_id=999, name="auto-thread")
    adapter._auto_create_thread = AsyncMock(return_value=fake_thread)

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    msg = _fake_message(_FakeTextChannel())

    await adapter._handle_message(msg)

    adapter._auto_create_thread.assert_awaited_once()
    assert len(captured_events) == 1
    assert captured_events[0].source.chat_id == "999"  # redirected to thread
    assert captured_events[0].source.chat_type == "thread"


@pytest.mark.asyncio
async def test_auto_thread_can_be_disabled(adapter, monkeypatch):
    """Setting DISCORD_AUTO_THREAD=false keeps messages in the channel."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    adapter._auto_create_thread = AsyncMock()

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    msg = _fake_message(_FakeTextChannel())

    await adapter._handle_message(msg)

    adapter._auto_create_thread.assert_not_awaited()
    assert len(captured_events) == 1
    assert captured_events[0].source.chat_id == "100"  # stays in channel


@pytest.mark.asyncio
async def test_auto_thread_skips_threads_and_dms(adapter, monkeypatch):
    """Auto-thread should not create threads inside existing threads."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    adapter._auto_create_thread = AsyncMock()

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    msg = _fake_message(_FakeThreadChannel())

    await adapter._handle_message(msg)

    adapter._auto_create_thread.assert_not_awaited()  # should NOT auto-thread


# ------------------------------------------------------------------
# Config bridge
# ------------------------------------------------------------------


def test_discord_auto_thread_config_bridge(monkeypatch, tmp_path):
    """discord.auto_thread in config.yaml should be bridged to DISCORD_AUTO_THREAD env var."""
    import yaml
    from pathlib import Path

    # Write a config.yaml the loader will find
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()
    config_path = hermes_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "discord": {"auto_thread": True},
    }))

    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_dir))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("DISCORD_AUTO_THREAD") == "true"
