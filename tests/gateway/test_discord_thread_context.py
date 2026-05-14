"""Tests for Discord thread starter/body context injection."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, green=1, grey=2, blurple=2)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.MessageType = SimpleNamespace(reply="reply")
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

import gateway.platforms.discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server", messages=None):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(id=10, name=guild_name)
        self.topic = None
        self._messages = {str(getattr(message, "id", "")): message for message in (messages or [])}

    async def fetch_message(self, message_id):
        return self._messages[str(message_id)]


class FakeThread:
    def __init__(self, channel_id: int = 2, name: str = "cli coding agents", parent=None, history_messages=None, starter_message=None):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(id=10, name="Hermes Server")
        self.topic = None
        self.starter_message = starter_message
        self._history_messages = list(history_messages or [])

    def history(self, *, limit=100, oldest_first=None):
        async def gen():
            messages = self._history_messages[:limit]
            if oldest_first is False:
                messages = list(reversed(messages))
            for message in messages:
                yield message

        return gen()


def fake_message(message_id: int, content: str, *, author_name: str = "probe", channel=None):
    return SimpleNamespace(
        id=message_id,
        content=content,
        mentions=[],
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=SimpleNamespace(id=message_id + 1000, display_name=author_name, name=author_name),
        guild=getattr(channel, "guild", None),
        type=None,
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", type("FakeDMChannel", (), {}), raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_THREAD_CONTEXT_MESSAGES", "3")

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_thread_starter_and_history_are_injected(adapter):
    parent = FakeTextChannel(channel_id=100, name="hermes-home")
    starter = fake_message(1, "CLI coding agents should read the thread body", author_name="probe")
    prior = fake_message(2, "Implementation should inspect Discord thread context", author_name="probe")
    current = fake_message(3, "이거 구현 가능한지 검토해줘", author_name="probe")
    thread = FakeThread(parent=parent, starter_message=starter, history_messages=[starter, prior, current])
    current.channel = thread
    current.guild = thread.guild

    await adapter._handle_message(current)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "thread"
    assert event.source.thread_id == str(thread.id)
    assert "[Discord thread context]" in event.text
    assert "Thread: cli coding agents" in event.text
    assert "Starter from probe: CLI coding agents should read the thread body" in event.text
    assert "Message from probe: Implementation should inspect Discord thread context" in event.text
    assert event.text.rstrip().endswith("이거 구현 가능한지 검토해줘")
    assert "Message from probe: 이거 구현 가능한지 검토해줘" not in event.text


@pytest.mark.asyncio
async def test_thread_commands_keep_slash_prefix(adapter):
    parent = FakeTextChannel(channel_id=100, name="hermes-home")
    starter = fake_message(1, "Thread context should not break slash commands", author_name="probe")
    current = fake_message(2, "/status", author_name="probe")
    thread = FakeThread(parent=parent, starter_message=starter, history_messages=[starter, current])
    current.channel = thread
    current.guild = thread.guild

    await adapter._handle_message(current)

    event = adapter.handle_message.await_args.args[0]
    assert event.text == "/status"
    assert event.get_command() == "status"


@pytest.mark.asyncio
async def test_parent_starter_message_is_fetched_when_thread_starter_is_not_cached(adapter):
    starter = fake_message(200, "Parent message body that created this thread", author_name="probe")
    parent = FakeTextChannel(channel_id=100, name="hermes-home", messages=[starter])
    current = fake_message(201, "follow up", author_name="probe")
    thread = FakeThread(channel_id=200, parent=parent, starter_message=None, history_messages=[current])
    current.channel = thread
    current.guild = thread.guild

    await adapter._handle_message(current)

    event = adapter.handle_message.await_args.args[0]
    assert "Starter from probe: Parent message body that created this thread" in event.text
    assert event.text.rstrip().endswith("follow up")


@pytest.mark.asyncio
async def test_thread_context_can_be_disabled(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_CONTEXT_MESSAGES", "0")
    parent = FakeTextChannel(channel_id=100, name="hermes-home")
    starter = fake_message(1, "Hidden thread body", author_name="probe")
    current = fake_message(2, "follow up", author_name="probe")
    thread = FakeThread(parent=parent, starter_message=starter, history_messages=[starter])
    current.channel = thread
    current.guild = thread.guild

    await adapter._handle_message(current)

    event = adapter.handle_message.await_args.args[0]
    assert "[Discord thread context]" not in event.text
    assert event.text == "follow up"
