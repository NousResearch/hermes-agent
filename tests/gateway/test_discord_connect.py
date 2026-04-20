import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


class _FakeAllowedMentions:
    """Stand-in for ``discord.AllowedMentions`` — exposes the same four
    boolean flags as real attributes so tests can assert on safe defaults.
    """

    def __init__(self, *, everyone=True, roles=True, users=True, replied_user=True):
        self.everyone = everyone
        self.roles = roles
        self.users = users
        self.replied_user = replied_user


def _ensure_discord_mock():
    """Install (or augment) a mock ``discord`` module.

    Always force ``AllowedMentions`` onto whatever is in ``sys.modules`` —
    other test files also stub the module via ``setdefault``, and we need
    ``_build_allowed_mentions()``'s return value to have real attribute
    access regardless of which file loaded first.
    """
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        sys.modules["discord"].AllowedMentions = _FakeAllowedMentions
        return

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.Client = MagicMock
        discord_mod.File = MagicMock
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
        discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3, grey=4, secondary=5)
        discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
        discord_mod.Interaction = object
        discord_mod.Embed = MagicMock
        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        discord_mod.opus = SimpleNamespace(is_loaded=lambda: True)

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)

    sys.modules["discord"].AllowedMentions = _FakeAllowedMentions


_ensure_discord_mock()

import gateway.platforms.discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class FakeTree:
    def __init__(self):
        self.sync = AsyncMock(return_value=[])

    def command(self, *args, **kwargs):
        return lambda fn: fn


class FakeBot:
    def __init__(self, *, intents, proxy=None, allowed_mentions=None, **_):
        self.intents = intents
        self.allowed_mentions = allowed_mentions
        self.user = SimpleNamespace(id=999, name="Hermes")
        self._events = {}
        self.tree = FakeTree()

    def event(self, fn):
        self._events[fn.__name__] = fn
        return fn

    async def start(self, token):
        if "on_ready" in self._events:
            await self._events["on_ready"]()

    async def close(self):
        return None


class SlowSyncTree(FakeTree):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.allow_finish = asyncio.Event()

        async def _slow_sync():
            self.started.set()
            await self.allow_finish.wait()
            return []

        self.sync = AsyncMock(side_effect=_slow_sync)


class SlowSyncBot(FakeBot):
    def __init__(self, *, intents, proxy=None):
        super().__init__(intents=intents, proxy=proxy)
        self.tree = SlowSyncTree()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed_users", "expected_members_intent"),
    [
        ("769524422783664158", False),
        ("abhey-gupta", True),
        ("769524422783664158,abhey-gupta", True),
    ],
)
async def test_connect_only_requests_members_intent_when_needed(monkeypatch, allowed_users, expected_members_intent):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", allowed_users)
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    created = {}

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        created["bot"] = FakeBot(intents=intents, allowed_mentions=allowed_mentions)
        return created["bot"]

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    ok = await adapter.connect()

    assert ok is True
    assert created["bot"].intents.members is expected_members_intent
    # Safe-default AllowedMentions must be applied on every connect so the
    # bot cannot @everyone from LLM output.  Granular overrides live in the
    # dedicated test_discord_allowed_mentions.py module.
    am = created["bot"].allowed_mentions
    assert am is not None, "connect() must pass an AllowedMentions to commands.Bot"
    assert am.everyone is False
    assert am.roles is False

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_releases_token_lock_on_timeout(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    released = []
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: released.append((scope, identity)))

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    monkeypatch.setattr(
        discord_platform.commands,
        "Bot",
        lambda **kwargs: FakeBot(
            intents=kwargs["intents"],
            proxy=kwargs.get("proxy"),
            allowed_mentions=kwargs.get("allowed_mentions"),
        ),
    )

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(discord_platform.asyncio, "wait_for", fake_wait_for)

    ok = await adapter.connect()

    assert ok is False
    assert released == [("discord-bot-token", "test-token")]
    assert adapter._platform_lock_identity is None


@pytest.mark.asyncio
async def test_connect_does_not_wait_for_slash_sync(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    created = {}

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        bot = SlowSyncBot(intents=intents, proxy=proxy)
        created["bot"] = bot
        return bot

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    ok = await asyncio.wait_for(adapter.connect(), timeout=1.0)

    assert ok is True
    assert adapter._ready_event.is_set()

    await asyncio.wait_for(created["bot"].tree.started.wait(), timeout=1.0)
    assert created["bot"].tree.sync.await_count == 1

    created["bot"].tree.allow_finish.set()
    await asyncio.sleep(0)
    await adapter.disconnect()
