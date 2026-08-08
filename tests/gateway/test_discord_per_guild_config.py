"""Tests for per-guild/per-server Discord configuration overrides (issue #23051).

Guild-scoped config under ``discord.guilds:<guild_id>:`` allows different
mention rules, channel lists, and threading behavior per Discord server.

Resolution priority: per-guild → global (config extra / env) → default.
Single-guild deployments without a ``guilds`` block see zero behavior change.
"""

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
    discord_mod.MessageType = SimpleNamespace(default=1, reply=2)
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
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

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

_GATE_ENV_VARS = [
    "DISCORD_REQUIRE_MENTION",
    "DISCORD_FREE_RESPONSE_CHANNELS",
    "DISCORD_ALLOWED_CHANNELS",
    "DISCORD_IGNORED_CHANNELS",
    "DISCORD_NO_THREAD_CHANNELS",
    "DISCORD_AUTO_THREAD",
    "DISCORD_ALLOW_ALL_USERS",
    "DISCORD_ALLOWED_USERS",
    "DISCORD_ALLOWED_ROLES",
    "DISCORD_IGNORE_NO_MENTION",
    "GATEWAY_ALLOW_ALL_USERS",
]


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeGuild:
    def __init__(self, guild_id: int = 100, name: str = "Guild 100"):
        self.id = guild_id
        self.name = name


class FakeTextChannel:
    def __init__(
        self,
        channel_id: int = 1,
        name: str = "general",
        guild_name: str = "Hermes Server",
        guild_id: int = 100,
    ):
        self.id = channel_id
        self.name = name
        self.guild = FakeGuild(guild_id=guild_id, name=guild_name)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


class FakeForumChannel:
    def __init__(
        self,
        channel_id: int = 1,
        name: str = "support-forum",
        guild_name: str = "Hermes Server",
        guild_id: int = 100,
    ):
        self.id = channel_id
        self.name = name
        self.guild = FakeGuild(guild_id=guild_id, name=guild_name)
        self.type = 15
        self.topic = None


class FakeThread:
    def __init__(
        self,
        channel_id: int = 1,
        name: str = "thread",
        parent=None,
        guild_name: str = "Hermes Server",
        guild_id: int = 100,
    ):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or FakeGuild(guild_id=guild_id, name=guild_name)
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    for var in _GATE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))
    return adapter


def make_message(*, channel, content: str, mentions=None, msg_type=None):
    author = SimpleNamespace(id=42, display_name="Jezza", name="Jezza")
    guild = getattr(channel, "guild", None)
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        guild=guild,
        type=msg_type if msg_type is not None else discord_platform.discord.MessageType.default,
    )


# ---------------------------------------------------------------------------
# _discord_guild_config
# ---------------------------------------------------------------------------


def test_guild_config_returns_entry(adapter):
    adapter.config.extra["guilds"] = {
        "100": {"require_mention": False},
    }
    assert adapter._discord_guild_config("100") == {"require_mention": False}


def test_guild_config_accepts_numeric_lookup(adapter):
    """Guild IDs are stored as strings; int lookups must normalize too."""
    adapter.config.extra["guilds"] = {
        "100": {"require_mention": False},
    }
    assert adapter._discord_guild_config(100) == {"require_mention": False}


def test_guild_config_returns_empty_for_unknown(adapter):
    adapter.config.extra["guilds"] = {"100": {"require_mention": False}}
    assert adapter._discord_guild_config("999") == {}


def test_guild_config_returns_empty_for_none(adapter):
    adapter.config.extra["guilds"] = {"100": {"require_mention": False}}
    assert adapter._discord_guild_config(None) == {}


def test_guild_config_returns_empty_for_missing_key(adapter):
    assert adapter._discord_guild_config("100") == {}


def test_guild_config_returns_empty_for_non_dict_entry(adapter):
    adapter.config.extra["guilds"] = {"100": "oops"}
    assert adapter._discord_guild_config("100") == {}


# ---------------------------------------------------------------------------
# _apply_yaml_config guild seeding
# ---------------------------------------------------------------------------


def test_apply_yaml_config_seeds_string_normalized_guilds(monkeypatch):
    """Bare snowflake int keys must be string-normalized (64-bit overflow)."""
    for var in _GATE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    seeded = discord_platform._apply_yaml_config(
        {},
        {
            "require_mention": True,
            "guilds": {
                100: {"require_mention": False},
                "101": {"free_response_channels": ["11", "12"]},
                "not-a-dict": "ignored",
            },
        },
    )
    assert seeded["guilds"] == {
        "100": {"require_mention": False},
        "101": {"free_response_channels": ["11", "12"]},
    }


def test_apply_yaml_config_skips_missing_guilds(monkeypatch):
    for var in _GATE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    seeded = discord_platform._apply_yaml_config({}, {"require_mention": False})
    # No seeded extras (require_mention is env-bridged only) → None return.
    assert not seeded or "guilds" not in seeded


# ---------------------------------------------------------------------------
# _discord_require_mention
# ---------------------------------------------------------------------------


def test_require_mention_guild_override_wins_over_global(adapter):
    adapter.config.extra["require_mention"] = True
    adapter.config.extra["guilds"] = {"100": {"require_mention": False}}
    assert adapter._discord_require_mention("100") is False
    assert adapter._discord_require_mention("999") is True


def test_require_mention_guild_string_override(adapter):
    adapter.config.extra["guilds"] = {"100": {"require_mention": "false"}}
    assert adapter._discord_require_mention("100") is False


def test_require_mention_falls_back_to_env_without_guild(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    assert adapter._discord_require_mention(None) is False
    assert adapter._discord_require_mention("100") is False


def test_require_mention_default_true(adapter):
    assert adapter._discord_require_mention() is True
    assert adapter._discord_require_mention("100") is True


# ---------------------------------------------------------------------------
# Channel-set accessors
# ---------------------------------------------------------------------------


def test_allowed_channels_guild_override(adapter):
    adapter.config.extra["allowed_channels"] = "1000"
    adapter.config.extra["guilds"] = {"100": {"allowed_channels": ["2000"]}}
    assert adapter._get_allowed_channels("100") == {"2000"}
    assert adapter._get_allowed_channels("999") == {"1000"}
    assert adapter._get_allowed_channels() == {"1000"}


def test_allowed_channels_guild_csv_override(adapter):
    adapter.config.extra["guilds"] = {"100": {"allowed_channels": "2000,2001"}}
    assert adapter._get_allowed_channels("100") == {"2000", "2001"}


def test_allowed_channels_guild_wildcard(adapter):
    adapter.config.extra["guilds"] = {"100": {"allowed_channels": "*"}}
    assert adapter._get_allowed_channels("100") == {"*"}


def test_allowed_channels_guild_override_beats_env(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "3000")
    adapter.config.extra["guilds"] = {"100": {"allowed_channels": ["2000"]}}
    assert adapter._get_allowed_channels("100") == {"2000"}
    assert adapter._get_allowed_channels("999") == {"3000"}


def test_ignored_channels_guild_override(adapter):
    adapter.config.extra["ignored_channels"] = "1000"
    adapter.config.extra["guilds"] = {"100": {"ignored_channels": ["2000"]}}
    assert adapter._get_ignored_channels("100") == {"2000"}
    assert adapter._get_ignored_channels("999") == {"1000"}


def test_no_thread_channels_guild_override(adapter):
    adapter.config.extra["no_thread_channels"] = "1000"
    adapter.config.extra["guilds"] = {"100": {"no_thread_channels": ["2000"]}}
    assert adapter._get_no_thread_channels("100") == {"2000"}
    assert adapter._get_no_thread_channels("999") == {"1000"}


def test_free_response_channels_guild_override(adapter):
    adapter.config.extra["free_response_channels"] = "1000"
    adapter.config.extra["guilds"] = {"100": {"free_response_channels": ["2000"]}}
    assert adapter._discord_free_response_channels("100") == {"2000"}
    assert adapter._discord_free_response_channels("999") == {"1000"}


def test_free_response_channels_guild_wildcard(adapter):
    adapter.config.extra["guilds"] = {"100": {"free_response_channels": ["*"]}}
    assert adapter._discord_free_response_channels("100") == {"*"}


def test_free_response_channels_guild_numeric_scalar(adapter):
    adapter.config.extra["guilds"] = {"100": {"free_response_channels": 1491973769726791812}}
    assert adapter._discord_free_response_channels("100") == {"1491973769726791812"}


# ---------------------------------------------------------------------------
# _discord_auto_thread
# ---------------------------------------------------------------------------


def test_auto_thread_guild_override(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    adapter.config.extra["guilds"] = {"100": {"auto_thread": False}}
    assert adapter._discord_auto_thread("100") is False
    assert adapter._discord_auto_thread("999") is True


def test_auto_thread_guild_string_override(adapter):
    adapter.config.extra["guilds"] = {"100": {"auto_thread": "no"}}
    assert adapter._discord_auto_thread("100") is False


def test_auto_thread_default_true(adapter):
    assert adapter._discord_auto_thread() is True


# ---------------------------------------------------------------------------
# _discord_channel_ids_allowed (fail-closed admission gate)
# ---------------------------------------------------------------------------


def test_channel_ids_allowed_scoped_to_guild(adapter):
    adapter.config.extra["allowed_channels"] = "1000"
    adapter.config.extra["guilds"] = {"100": {"allowed_channels": ["2000"]}}
    # Guild 100's own allowlist admits its channel...
    assert adapter._discord_channel_ids_allowed({"2000"}, guild_id="100") is True
    # ...but not a channel that only the global list admits.
    assert adapter._discord_channel_ids_allowed({"1000"}, guild_id="100") is False
    # No guild context → global list.
    assert adapter._discord_channel_ids_allowed({"1000"}) is True
    assert adapter._discord_channel_ids_allowed({"1000"}, guild_id="999") is True


def test_channel_ids_allowed_guild_wildcard(adapter):
    adapter.config.extra["guilds"] = {"100": {"allowed_channels": ["*"]}}
    assert adapter._discord_channel_ids_allowed({"anything"}, guild_id="100") is True


# ---------------------------------------------------------------------------
# _handle_message end-to-end paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_guild_free_response_without_mention(adapter):
    """Guild-scoped free_response_channels admit unmentioned messages."""
    adapter.config.extra["guilds"] = {
        "100": {"free_response_channels": ["2000"]},
    }
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=100),
        content="hello",
    )
    await adapter._handle_message(message)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_guild_require_mention_true_blocks(adapter):
    """Guild-scoped require_mention: true overrides a global false."""
    adapter.config.extra["require_mention"] = False
    adapter.config.extra["guilds"] = {
        "100": {"require_mention": True},
    }
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=100),
        content="hello",
    )
    await adapter._handle_message(message)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_message_guild_require_mention_false_admits(adapter):
    """Guild-scoped require_mention: false admits unmentioned messages."""
    adapter.config.extra["require_mention"] = True
    adapter.config.extra["guilds"] = {
        "100": {"require_mention": False},
    }
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=100),
        content="hello",
    )
    await adapter._handle_message(message)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_guild_allowed_channels_scoped(adapter):
    """A guild-scoped allowlist replaces the global list for that guild only."""
    adapter.config.extra["allowed_channels"] = ["3000"]
    adapter.config.extra["guilds"] = {
        "100": {"allowed_channels": ["2000"]},
    }
    bot_user = adapter._client.user

    # Guild 100: the per-guild override admits only channel 2000 (the
    # global 3000 entry does NOT apply to this guild).
    in_guild = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=100),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(in_guild)
    adapter.handle_message.assert_awaited_once()

    adapter.handle_message.reset_mock()
    global_only = make_message(
        channel=FakeTextChannel(channel_id=3000, guild_id=100),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(global_only)
    adapter.handle_message.assert_not_awaited()

    # Guild 200 has no override block: the global allowlist still applies.
    adapter.handle_message.reset_mock()
    other_guild_global = make_message(
        channel=FakeTextChannel(channel_id=3000, guild_id=200),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(other_guild_global)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_guild_ignored_channel_blocks_even_with_mention(adapter):
    """Guild-scoped ignored_channels drop even @mentioned messages."""
    adapter.config.extra["guilds"] = {
        "100": {"ignored_channels": ["2000"]},
    }
    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=100),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_message_guild_ignored_does_not_leak_to_other_guild(adapter):
    """ignored_channels for one guild must not silence the same channel elsewhere."""
    adapter.config.extra["require_mention"] = False
    adapter.config.extra["guilds"] = {
        "100": {"ignored_channels": ["2000"]},
    }
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=200),
        content="hello",
    )
    await adapter._handle_message(message)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_guild_no_thread_skips_auto_thread(adapter):
    """Guild-scoped no_thread_channels skip auto-threading for that guild."""
    adapter.config.extra["require_mention"] = False
    adapter.config.extra["guilds"] = {
        "100": {"no_thread_channels": ["2000"]},
    }
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=100),
        content="hello",
    )
    await adapter._handle_message(message)
    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_guild_auto_thread_false_skips(adapter):
    """Guild-scoped auto_thread: false disables auto-threading for that guild."""
    adapter.config.extra["require_mention"] = False
    adapter.config.extra["guilds"] = {
        "100": {"auto_thread": False},
    }
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=100),
        content="hello",
    )
    await adapter._handle_message(message)
    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_other_guild_still_auto_threads(adapter):
    """Guilds without an auto_thread override keep the global default."""
    adapter.config.extra["require_mention"] = False
    adapter.config.extra["guilds"] = {
        "100": {"auto_thread": False},
    }
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=200),
        content="hello",
    )
    await adapter._handle_message(message)
    adapter._auto_create_thread.assert_awaited_once()


# ---------------------------------------------------------------------------
# _discord_message_admission pre-dispatch gate
# ---------------------------------------------------------------------------


def _admission_ready_adapter(adapter):
    """Bypass user/role allowlist for admission-focused tests.

    The fixture author is user id ``42`` (see ``make_message``); granting it
    here keeps ``_is_allowed_user`` from fail-closing before the
    free-response mention gate under test is reached.
    """
    adapter._allowed_user_ids = {"42"}
    adapter._allowed_role_ids = set()
    adapter._is_pairing_approved_user = lambda user_id: False
    return adapter


@pytest.mark.asyncio
async def test_admission_guild_free_response_accepts_mention_of_other_user(adapter):
    """Mentioning another user in a guild free-response channel is admitted."""
    _admission_ready_adapter(adapter)
    adapter.config.extra["guilds"] = {
        "100": {"free_response_channels": ["2000"]},
    }
    other = SimpleNamespace(id=1, bot=False)
    message = make_message(
        channel=FakeTextChannel(channel_id=2000, guild_id=100),
        content="<@1> hello",
        mentions=[other],
    )
    admitted, role_authorized = await asyncio_ctx(adapter, message)
    assert admitted is True
    assert role_authorized is False


@pytest.mark.asyncio
async def test_admission_guild_free_response_rejects_other_guild(adapter):
    """Mentioning another user outside the guild's free-response list is dropped."""
    _admission_ready_adapter(adapter)
    adapter.config.extra["guilds"] = {
        "100": {"free_response_channels": ["2000"]},
    }
    other = SimpleNamespace(id=1, bot=False)
    message = make_message(
        channel=FakeTextChannel(channel_id=3000, guild_id=100),
        content="<@1> hello",
        mentions=[other],
    )
    admitted, _ = await asyncio_ctx(adapter, message)
    assert admitted is False


# ---------------------------------------------------------------------------
# _missed_message_backfill_channels union
# ---------------------------------------------------------------------------


def test_backfill_channels_union_includes_guild_scoped_lists(adapter):
    adapter.config.extra["allowed_channels"] = "1000"
    adapter.config.extra["free_response_channels"] = "1001"
    adapter.config.extra["guilds"] = {
        "100": {"allowed_channels": ["2000"], "free_response_channels": ["2001"]},
    }
    assert adapter._missed_message_backfill_channels() == {
        "1000", "1001", "2000", "2001",
    }


async def asyncio_ctx(adapter, message):
    """Thin wrapper so admission tests read as plain async asserts."""
    return adapter._discord_message_admission(message, claim=False)
