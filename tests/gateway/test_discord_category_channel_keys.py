"""Regression tests for Discord category-level channel gates."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock
import os
import sys

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
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: lambda fn: fn,
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1,
        primary=2,
        secondary=2,
        danger=3,
        green=1,
        grey=2,
        blurple=2,
        red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
        purple=lambda: 5,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: lambda fn: fn,
        choices=lambda **kwargs: lambda fn: fn,
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

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter, _apply_yaml_config  # noqa: E402


class FakeCategory:
    id = 500
    name = "Team Ops"


class FakeTextChannel:
    id = 123
    name = "adops"
    category_id = 500
    category = FakeCategory()


class FakeThread:
    """Thread whose category lives on its parent channel (discord.py semantics)."""

    def __init__(self, thread_id=456, name="topic", parent=None):
        self.id = thread_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)


class UncachedParentThread:
    """Thread-like object whose category properties raise, as discord.py's
    ``Thread.category_id`` / ``Thread.category`` do when the parent isn't
    cached (they raise ``ClientException``, which ``getattr`` does not swallow).
    """

    id = 789
    name = "orphan-thread"
    parent = None

    @property
    def category_id(self):
        raise RuntimeError("Parent channel not found")

    @property
    def category(self):
        raise RuntimeError("Parent channel not found")


class DiscordCategoryChannelGateTests(TestCase):
    def setUp(self):
        self.adapter = object.__new__(DiscordAdapter)
        self.adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})

    def tearDown(self):
        os.environ.pop("DISCORD_ALLOWED_CATEGORIES", None)
        os.environ.pop("DISCORD_FREE_RESPONSE_CATEGORIES", None)

    def test_yaml_config_maps_category_gates_to_env(self):
        _apply_yaml_config(
            {},
            {
                "allowed_categories": [500, 600],
                "free_response_categories": [700, 800],
            },
        )

        self.assertEqual("500,600", os.environ["DISCORD_ALLOWED_CATEGORIES"])
        self.assertEqual("700,800", os.environ["DISCORD_FREE_RESPONSE_CATEGORIES"])

    def test_channel_keys_do_not_overload_category_ids(self):
        keys = self.adapter._discord_channel_keys_from_channel(FakeTextChannel())

        self.assertIn("123", keys)
        self.assertNotIn("500", keys)
        self.assertNotIn("Team Ops", keys)
        self.assertNotIn("#Team Ops", keys)

    def test_category_keys_include_category_id_and_name(self):
        keys = self.adapter._discord_category_keys_from_channel(FakeTextChannel())

        self.assertEqual({"500", "Team Ops", "#Team Ops"}, keys)

    def test_allowed_categories_uses_explicit_env_var(self):
        os.environ["DISCORD_ALLOWED_CATEGORIES"] = "500,600"

        self.assertTrue(
            self.adapter._discord_category_ids_allowed(
                self.adapter._discord_category_keys_from_channel(FakeTextChannel())
            )
        )

    def test_free_response_categories_uses_explicit_yaml_extra(self):
        self.adapter.config.extra["free_response_categories"] = [500, 600]

        self.assertEqual(
            {"500", "600"}, self.adapter._discord_free_response_categories()
        )

    def test_category_keys_follow_thread_parent(self):
        """A thread inherits its parent channel's category id/name."""
        thread = FakeThread(parent=FakeTextChannel())

        keys = self.adapter._discord_category_keys_from_channel(thread)

        self.assertEqual({"500", "Team Ops", "#Team Ops"}, keys)

    def test_category_keys_uncached_parent_degrades_without_raising(self):
        """A thread with an uncached parent (properties raise) yields no keys
        instead of crashing the caller."""
        keys = self.adapter._discord_category_keys_from_channel(UncachedParentThread())

        self.assertEqual(set(), keys)

    def test_no_mention_gate_admits_free_response_category(self):
        """The early no-mention gate admits a message whose category is a
        configured free-response category."""
        self.adapter._client = None
        os.environ["DISCORD_FREE_RESPONSE_CATEGORIES"] = "500"
        message = SimpleNamespace(channel=FakeTextChannel(), content="hi")

        self.assertTrue(self.adapter._no_mention_free_gate_admits(message))

    def test_no_mention_gate_rejects_non_free_category(self):
        self.adapter._client = None
        os.environ["DISCORD_FREE_RESPONSE_CATEGORIES"] = "999"
        message = SimpleNamespace(channel=FakeTextChannel(), content="hi")

        self.assertFalse(self.adapter._no_mention_free_gate_admits(message))


# ---------------------------------------------------------------------------
# Behavioural coverage: message path (_handle_message) and slash authorization
# for category-scoped gates. These need a fully-initialized adapter with the
# discord type checks pointed at the fakes below (real discord.py is installed
# in CI, so isinstance() must resolve against our doubles).
# ---------------------------------------------------------------------------

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402


class BehaviourCategory:
    def __init__(self, category_id=500, name="Team Ops"):
        self.id = category_id
        self.name = name


def _empty_history(*, limit, before, after=None, oldest_first=None):
    async def _iter():
        return
        yield

    return _iter()


class BehaviourTextChannel:
    def __init__(
        self, channel_id=123, name="general", category_id=500, category_name="Team Ops"
    ):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name="Hermes Server")
        self.topic = None
        self.category_id = category_id
        self.category = (
            BehaviourCategory(category_id, category_name) if category_name else None
        )

    def history(self, *, limit, before, after=None, oldest_first=None):
        return _empty_history(
            limit=limit, before=before, after=after, oldest_first=oldest_first
        )


class BehaviourThread:
    def __init__(self, channel_id=456, name="topic", parent=None):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(
            name="Hermes Server"
        )
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        return _empty_history(
            limit=limit, before=before, after=after, oldest_first=oldest_first
        )


class BehaviourDMChannel:
    def __init__(self, channel_id=1, name="dm"):
        self.id = channel_id
        self.name = name


@pytest.fixture
def badapter(monkeypatch):
    monkeypatch.setattr(
        discord_platform.discord, "DMChannel", BehaviourDMChannel, raising=False
    )
    monkeypatch.setattr(
        discord_platform.discord, "Thread", BehaviourThread, raising=False
    )
    for _var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_THREAD_REQUIRE_MENTION",
        "DISCORD_AUTO_THREAD",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_ALLOWED_CATEGORIES",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_FREE_RESPONSE_CATEGORIES",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_HISTORY_BACKFILL",
    ):
        monkeypatch.delenv(_var, raising=False)

    from gateway.config import PlatformConfig

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


def _make_message(*, channel, content, mentions=None):
    author = SimpleNamespace(id=42, display_name="Jezza", name="Jezza")
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        type=discord_platform.discord.MessageType.default,
    )


@pytest.mark.asyncio
async def test_handle_message_allows_message_in_allowed_category(badapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_ALLOWED_CATEGORIES", "500")

    bot = badapter._client.user
    message = _make_message(
        channel=BehaviourTextChannel(channel_id=123, category_id=500),
        content=f"<@{bot.id}> hi",
        mentions=[bot],
    )

    await badapter._handle_message(message)

    badapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_rejects_message_outside_allowed_category(
    badapter, monkeypatch
):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_ALLOWED_CATEGORIES", "500")

    bot = badapter._client.user
    message = _make_message(
        channel=BehaviourTextChannel(
            channel_id=123, category_id=999, category_name="Other"
        ),
        content=f"<@{bot.id}> hi",
        mentions=[bot],
    )

    await badapter._handle_message(message)

    badapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_message_allows_thread_under_allowed_category(
    badapter, monkeypatch
):
    """A thread inherits its parent channel's category for the allow gate."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_ALLOWED_CATEGORIES", "500")

    bot = badapter._client.user
    parent = BehaviourTextChannel(channel_id=123, category_id=500)
    thread = BehaviourThread(channel_id=456, parent=parent)
    badapter._threads.mark("456")  # bot has participated → mention not required
    message = _make_message(channel=thread, content="follow-up", mentions=[])

    await badapter._handle_message(message)

    badapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_message_free_response_category_without_mention(
    badapter, monkeypatch
):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "500")

    message = _make_message(
        channel=BehaviourTextChannel(channel_id=123, category_id=500),
        content="no mention but free category",
    )

    await badapter._handle_message(message)

    badapter.handle_message.assert_awaited_once()
    event = badapter.handle_message.await_args.args[0]
    assert event.text == "no mention but free category"


def test_slash_authorization_admits_thread_under_allowed_category(
    badapter, monkeypatch
):
    monkeypatch.setenv("DISCORD_ALLOWED_CATEGORIES", "500")

    parent = BehaviourTextChannel(channel_id=123, category_id=500)
    thread = BehaviourThread(channel_id=456, parent=parent)
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=42, name="Jezza"),
        guild=SimpleNamespace(owner_id=1, id=7, get_member=lambda uid: None),
        guild_id=7,
        channel_id=456,
        channel=thread,
    )

    allowed, reason = badapter._evaluate_slash_authorization(interaction)

    assert allowed is True, reason


def test_slash_authorization_rejects_thread_outside_allowed_category(
    badapter, monkeypatch
):
    monkeypatch.setenv("DISCORD_ALLOWED_CATEGORIES", "500")

    parent = BehaviourTextChannel(
        channel_id=123, category_id=999, category_name="Other"
    )
    thread = BehaviourThread(channel_id=456, parent=parent)
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=42, name="Jezza"),
        guild=SimpleNamespace(owner_id=1, id=7, get_member=lambda uid: None),
        guild_id=7,
        channel_id=456,
        channel=thread,
    )

    allowed, reason = badapter._evaluate_slash_authorization(interaction)

    assert allowed is False
    assert reason == "category not in DISCORD_ALLOWED_CATEGORIES"
