"""The receiving bot's own @mention is stripped only where it is routing (leading), not mid-text.

A body that names the whole crew — ``<@a> <@b> <@self> <@c>`` — must reach the model with every
name in it. Deleting the self-token anywhere in the text silently shortens the content: the bot
counts one name fewer than the channel shows and reports a list it cannot see itself in.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    from unittest.mock import MagicMock

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.Message = type("Message", (), {})
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

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

SELF_ID = 999
OTHER_ID = 111
THIRD_ID = 222


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


class FakeThread(FakeTextChannel):
    def __init__(self, channel_id: int = 1, name: str = "thread", parent=None, guild_name: str = "Hermes Server"):
        super().__init__(channel_id=channel_id, name=name, guild_name=guild_name)
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


FakeDMChannel = type("FakeDMChannel", (), {})
FakeForumChannel = type("FakeForumChannel", (), {})


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel, raising=False)
    for _var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "DISCORD_HISTORY_BACKFILL",
        "DISCORD_ALLOW_BOTS",
    ):
        monkeypatch.delenv(_var, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=SELF_ID, bot=True))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(*, channel, content: str, mentions=None, author_bot=False):
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Ari", name="Ari", bot=author_bot),
        type=discord_platform.discord.MessageType.default,
    )


@pytest.mark.asyncio
async def test_own_mention_inside_a_list_stays_visible(adapter, monkeypatch):
    """Roster regression: seven others plus the bot must arrive as seven others plus the bot."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    content = (
        f"Probe: ihr drei seid neu im Server. <@{OTHER_ID}> <@!{SELF_ID}> <@{THIRD_ID}>\n\n"
        "Meldet euch jeder mit einem Satz."
    )
    message = make_message(
        channel=FakeTextChannel(channel_id=321),
        content=content,
        mentions=[SimpleNamespace(id=SELF_ID, bot=True)],
    )

    await adapter._handle_message(message)

    event = adapter.handle_message.await_args.args[0]
    assert event.text.count("<@") == 3
    assert f"<@!{SELF_ID}>" in event.text
    assert f"<@{OTHER_ID}>" in event.text


@pytest.mark.asyncio
async def test_leading_own_mention_is_still_stripped(adapter, monkeypatch):
    """Routing stays intact: an address in front (also doubled, also <@!id>) never reaches the model."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    message = make_message(
        channel=FakeTextChannel(channel_id=321),
        content=f"  <@{SELF_ID}> <@!{SELF_ID}>   /status bitte",
        mentions=[SimpleNamespace(id=SELF_ID, bot=True)],
    )

    await adapter._handle_message(message)

    event = adapter.handle_message.await_args.args[0]
    assert event.text == "/status bitte"


@pytest.mark.asyncio
async def test_foreign_mention_at_the_start_is_left_alone(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    message = make_message(
        channel=FakeTextChannel(channel_id=321),
        content=f"<@{OTHER_ID}> <@{SELF_ID}> bitte prüfen",
        mentions=[SimpleNamespace(id=SELF_ID, bot=True), SimpleNamespace(id=OTHER_ID, bot=False)],
    )

    await adapter._handle_message(message)

    event = adapter.handle_message.await_args.args[0]
    assert event.text == f"<@{OTHER_ID}> <@{SELF_ID}> bitte prüfen"


@pytest.mark.asyncio
async def test_bare_own_mention_stays_a_bare_ping(adapter, monkeypatch):
    """A ping with nothing else is still an empty body, so it can't spawn a content-free turn."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    message = make_message(
        channel=FakeTextChannel(channel_id=321),
        content=f"<@{SELF_ID}>",
        mentions=[SimpleNamespace(id=SELF_ID, bot=True)],
    )

    dispatched = await adapter._handle_message(message)

    assert dispatched is False
    adapter.handle_message.assert_not_awaited()
    assert message.content == ""
