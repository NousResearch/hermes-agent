"""Tests for Discord category-aware channel gating.

A Discord category is the natural unit for scoping a bot on a shared server: one
category ID in ``allowed_channels`` should gate every channel filed under it, and a
channel created later must follow automatically with no config change.

discord.py exposes the owning category natively (``TextChannel.category`` and
``Thread.category``); a plain text channel has no ``parent_id`` — only threads do —
so the category cannot be derived from the parent chain alone.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


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
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
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

from gateway.config import PlatformConfig

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter

CATEGORY_ID = 1570448935372464272
CATEGORY_NAME = "LABS-X"


class FakeCategory:
    """Stands in for discord.CategoryChannel (the adapter only reads id/name)."""

    def __init__(self, category_id: int = CATEGORY_ID, name: str = CATEGORY_NAME):
        self.id = category_id
        self.name = name


# The adapter resolves the category class at call time, so the mock module has to
# expose a real class here for isinstance() to work.
discord_platform.discord.CategoryChannel = FakeCategory


class FakeTextChannel:
    """A text channel filed under a category, like a real TextChannel."""

    def __init__(self, channel_id: int = 1, name: str = "labs-workspace", category=None):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name="Hermes Server", get_channel=lambda cid: category)
        self.topic = None
        # discord.py: TextChannel exposes category_id, NOT parent_id.
        self.category_id = getattr(category, "id", None)
        self._category = category

    @property
    def category(self):
        return self._category


class FakeThread:
    """A thread under a text channel (discord.Thread exposes parent_id + category)."""

    def __init__(self, channel_id: int = 2, name: str = "thread", parent=None):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None)
        self.topic = None

    @property
    def category(self):
        # discord.py's Thread.category resolves through the parent channel.
        return getattr(self.parent, "category", None)


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(channel):
    return SimpleNamespace(
        id=1,
        content="hello",
        mentions=[],
        attachments=[],
        reference=None,
        channel=channel,
        author=SimpleNamespace(id=42, display_name="TestUser", name="TestUser"),
        created_at=None,
    )


# ── category resolution ─────────────────────────────────────────────


def test_text_channel_resolves_its_category(adapter):
    category = FakeCategory()
    channel = FakeTextChannel(channel_id=700, category=category)

    assert adapter._discord_channel_category(channel) is category


def test_thread_resolves_parent_channel_category(adapter):
    category = FakeCategory()
    parent = FakeTextChannel(channel_id=700, category=category)
    thread = FakeThread(channel_id=800, parent=parent)

    assert adapter._discord_channel_category(thread) is category


def test_channel_without_category_returns_none(adapter):
    assert adapter._discord_channel_category(FakeTextChannel(channel_id=700, category=None)) is None


def test_raising_category_property_is_swallowed(adapter):
    """discord.py raises ClientException when the parent channel is not cached."""

    class ExplodingChannel:
        id = 900
        name = "orphan"

        @property
        def category(self):
            raise RuntimeError("Parent channel not found")

    assert adapter._discord_channel_category(ExplodingChannel()) is None


# ── channel keys include the category ───────────────────────────────


def test_channel_keys_include_category_id_and_name(adapter):
    category = FakeCategory()
    channel = FakeTextChannel(channel_id=700, category=category)

    keys = adapter._discord_channel_keys_from_channel(channel)

    assert str(CATEGORY_ID) in keys
    assert CATEGORY_NAME in keys
    assert f"#{CATEGORY_NAME}" in keys
    # the channel's own keys are still present
    assert "700" in keys
    assert "labs-workspace" in keys


def test_thread_keys_include_category_id(adapter):
    category = FakeCategory()
    parent = FakeTextChannel(channel_id=700, category=category)
    thread = FakeThread(channel_id=800, parent=parent)

    keys = adapter._discord_channel_keys_from_channel(thread)

    assert str(CATEGORY_ID) in keys
    assert "800" in keys
    assert "700" in keys


def test_channel_keys_unchanged_without_category(adapter):
    """A channel outside any category keeps exactly the pre-change key set."""
    channel = FakeTextChannel(channel_id=700, category=None)

    keys = adapter._discord_channel_keys_from_channel(channel)

    assert keys == {"700", "labs-workspace", "#labs-workspace"}


# ── the gate: a category ID in allowed_channels admits the channel ──


@pytest.mark.asyncio
async def test_category_id_in_allowed_channels_admits_channel(adapter, monkeypatch):
    """A channel inside the gated category is admitted by the category ID alone."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", str(CATEGORY_ID))
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    channel = FakeTextChannel(channel_id=700, category=FakeCategory())

    await adapter._handle_message(make_message(channel))

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_category_id_in_allowed_channels_rejects_other_category(adapter, monkeypatch):
    """A channel in a different category stays blocked."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", str(CATEGORY_ID))
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    other = FakeCategory(category_id=999999, name="OTHER")
    channel = FakeTextChannel(channel_id=700, name="other-lane", category=other)

    await adapter._handle_message(make_message(channel))

    adapter.handle_message.assert_not_awaited()
