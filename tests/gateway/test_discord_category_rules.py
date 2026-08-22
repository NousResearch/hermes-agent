"""Tests for Discord per-category mention and ignore rules.

Covers the three category-level config keys added alongside the existing
channel-ID controls:

- ``free_response_categories``: bot responds without @mention in every
  channel/thread inside the listed categories
- ``ignored_categories``: bot never responds inside the listed categories,
  even when @mentioned
- ``require_mention_categories``: bot requires @mention inside the listed
  categories even when the global ``require_mention`` default is false

Also covers thread inheritance (a thread whose parent channel sits in a
listed category picks up the category rule), precedence (ignore beats
free; require-mention beats free), the startup mutual-exclusion warning,
CSV/YAML parsing parity, the ingress admission gate, and the config.yaml
→ env bridge.
"""

from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

discord = discord_platform.discord


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeCategory:
    def __init__(self, category_id: int = 1, name: str = "Category"):
        self.id = category_id
        self.name = name


class FakeTextChannel:
    def __init__(
        self,
        channel_id: int = 1,
        name: str = "general",
        category: FakeCategory = None,
        guild_name: str = "Hermes Server",
    ):
        self.id = channel_id
        self.name = name
        self.category = category
        self.guild = SimpleNamespace(name=guild_name)
        self.topic = None


class FakeThread:
    def __init__(
        self,
        channel_id: int = 1,
        name: str = "thread",
        parent: FakeTextChannel = None,
        guild_name: str = "Hermes Server",
    ):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        # Mirror discord.py: Thread.category resolves through the parent
        # channel, so a thread in a channel inside a category inherits it.
        self.category = getattr(parent, "category", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name=guild_name)
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(*, channel, content: str, mentions=None, bot_author: bool = False):
    author = SimpleNamespace(
        id=42, display_name="TestUser", name="TestUser", bot=bot_author
    )
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
    )


def _message_type_default():
    """MessageType.default against either the real discord.py or the mock."""
    message_type = getattr(discord, "MessageType", None)
    if message_type is None:
        return 0
    return message_type.default


# ── free_response_categories ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_free_response_category_waives_mention_in_category_channels(adapter, monkeypatch):
    """Channels inside a free-response category respond without @mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "1")

    message = make_message(
        channel=FakeTextChannel(channel_id=500, category=FakeCategory(category_id=1)),
        content="hello",
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_free_response_category_still_gates_other_channels(adapter, monkeypatch):
    """Channels outside a free-response category still require @mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "1")

    message = make_message(
        channel=FakeTextChannel(channel_id=500, name="uncategorised"),
        content="hello",
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_free_response_category_applies_to_threads_in_category(adapter, monkeypatch):
    """A thread whose parent channel is in a free category inherits the rule."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "1")
    # Thread messages always trigger history backfill; the fake channel has no
    # history API, so disable it to keep this test scoped to the category rule.
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")

    parent = FakeTextChannel(
        channel_id=500, name="support", category=FakeCategory(category_id=1)
    )
    thread = FakeThread(channel_id=900, name="ticket-1", parent=parent)
    message = make_message(channel=thread, content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_free_response_category_wildcard_waives_mention_everywhere(adapter, monkeypatch):
    """A ``*`` free-response category entry is a global free-response scope."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "*")

    message = make_message(channel=FakeTextChannel(channel_id=500), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


# ── ignored_categories ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ignored_category_silences_even_when_mentioned(adapter, monkeypatch):
    """Ignored categories take priority — even @mentions are dropped."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_IGNORED_CATEGORIES", "1")

    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=500, category=FakeCategory(category_id=1)),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_ignored_category_applies_to_threads(adapter, monkeypatch):
    """Threads inside an ignored category are silent even when mentioned."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_IGNORED_CATEGORIES", "1")

    bot_user = adapter._client.user
    parent = FakeTextChannel(
        channel_id=500, name="support", category=FakeCategory(category_id=1)
    )
    thread = FakeThread(channel_id=900, name="ticket-1", parent=parent)
    message = make_message(
        channel=thread, content=f"<@{bot_user.id}> hello", mentions=[bot_user]
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_ignored_category_overrides_free_response_category(adapter, monkeypatch):
    """A category in both lists is ignored: ignore wins over free (precedence)."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "1")
    monkeypatch.setenv("DISCORD_IGNORED_CATEGORIES", "1")

    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=500, category=FakeCategory(category_id=1)),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_ignored_category_wildcard_silences_everywhere(adapter, monkeypatch):
    """A ``*`` ignored-category entry silences every channel, even when mentioned."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_IGNORED_CATEGORIES", "*")

    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=500),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


# ── require_mention_categories ───────────────────────────────────────


@pytest.mark.asyncio
async def test_require_mention_category_gates_category_when_globally_free(adapter, monkeypatch):
    """A listed category flips to gated even when require_mention is false."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION_CATEGORIES", "1")

    message = make_message(
        channel=FakeTextChannel(channel_id=500, category=FakeCategory(category_id=1)),
        content="hello",
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_require_mention_category_mentions_still_work(adapter, monkeypatch):
    """@mention still triggers the bot inside a require-mention category."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION_CATEGORIES", "1")

    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))
    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=500, category=FakeCategory(category_id=1)),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_category_leaves_other_channels_free(adapter, monkeypatch):
    """Channels outside the listed category stay free under require_mention=false."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION_CATEGORIES", "1")

    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))
    message = make_message(channel=FakeTextChannel(channel_id=500), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_require_mention_category_overrides_free_response_channel(adapter, monkeypatch):
    """A require-mention category beats a channel-level free-response exemption."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "500")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION_CATEGORIES", "1")

    message = make_message(
        channel=FakeTextChannel(channel_id=500, category=FakeCategory(category_id=1)),
        content="hello",
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


# ── accessor parsing parity ──────────────────────────────────────────

_CATEGORY_ACCESSORS = [
    ("free_response_categories", "DISCORD_FREE_RESPONSE_CATEGORIES", "_discord_free_response_categories"),
    ("ignored_categories", "DISCORD_IGNORED_CATEGORIES", "_discord_ignored_categories"),
    ("require_mention_categories", "DISCORD_REQUIRE_MENTION_CATEGORIES", "_discord_require_mention_categories"),
]


@pytest.mark.parametrize("extra_key,env_key,attr", _CATEGORY_ACCESSORS)
def test_category_accessors_parse_yaml_list(monkeypatch, extra_key, env_key, attr):
    """A YAML list in PlatformConfig.extra parses to the ID set."""
    config = PlatformConfig(enabled=True, token="fake-token", extra={extra_key: ["11", "22"]})
    adapter = DiscordAdapter(config)
    monkeypatch.delenv(env_key, raising=False)
    assert getattr(adapter, attr)() == {"11", "22"}


@pytest.mark.parametrize("extra_key,env_key,attr", _CATEGORY_ACCESSORS)
def test_category_accessors_parse_env_csv(monkeypatch, extra_key, env_key, attr):
    """A CSV env var parses to the same ID set as the YAML list."""
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    monkeypatch.setenv(env_key, "33, 44")
    assert getattr(adapter, attr)() == {"33", "44"}


@pytest.mark.parametrize("extra_key,env_key,attr", _CATEGORY_ACCESSORS)
def test_category_accessors_coerce_numeric_scalar(monkeypatch, extra_key, env_key, attr):
    """A bare numeric scalar in YAML is coerced to str, matching the channel keys."""
    config = PlatformConfig(
        enabled=True, token="fake-token", extra={extra_key: 1491973769726791812}
    )
    adapter = DiscordAdapter(config)
    monkeypatch.delenv(env_key, raising=False)
    assert getattr(adapter, attr)() == {"1491973769726791812"}


# ── mutual-exclusion startup warning ─────────────────────────────────


def test_mutual_exclusion_warning_logged(adapter, monkeypatch, caplog):
    """A category in both free_response_categories and ignored_categories warns once."""
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "1,2")
    monkeypatch.setenv("DISCORD_IGNORED_CATEGORIES", "2")

    with caplog.at_level("WARNING", logger="plugins.platforms.discord.adapter"):
        adapter._warn_category_rule_conflicts()

    messages = [record.message for record in caplog.records]
    assert any("2" in m and "free_response_categories" in m and "ignored_categories" in m for m in messages)
    # Category 1 is only in the free list — must not be flagged.
    assert not any("1" in m and "free_response_categories" in m and "ignored_categories" in m for m in messages)


def test_mutual_exclusion_no_warning_when_disjoint(adapter, monkeypatch, caplog):
    """Disjoint category lists produce no startup warning."""
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "1")
    monkeypatch.setenv("DISCORD_IGNORED_CATEGORIES", "2")

    with caplog.at_level("WARNING", logger="plugins.platforms.discord.adapter"):
        adapter._warn_category_rule_conflicts()

    assert not caplog.records


# ── ingress admission gate ───────────────────────────────────────────


def test_admission_gate_honors_free_response_categories(adapter, monkeypatch):
    """The pre-dispatch admission gate admits unmentioned messages in free categories."""
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("DISCORD_IGNORE_NO_MENTION", "true")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "1")
    monkeypatch.delenv("DISCORD_ALLOWED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)

    channel = FakeTextChannel(channel_id=500, category=FakeCategory(category_id=1))
    message = make_message(
        channel=channel, content="hello", mentions=[SimpleNamespace(id=7, bot=False)]
    )
    message.type = _message_type_default()
    message.guild = channel.guild

    admitted, _ = adapter._discord_message_admission(message, claim=False)
    assert admitted is True


def test_admission_gate_blocks_unmentioned_outside_free_categories(adapter, monkeypatch):
    """The admission gate still drops unmentioned messages outside free categories."""
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("DISCORD_IGNORE_NO_MENTION", "true")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "1")
    monkeypatch.delenv("DISCORD_ALLOWED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)

    channel = FakeTextChannel(channel_id=500, name="uncategorised")
    message = make_message(
        channel=channel, content="hello", mentions=[SimpleNamespace(id=7, bot=False)]
    )
    message.type = _message_type_default()
    message.guild = channel.guild

    admitted, _ = adapter._discord_message_admission(message, claim=False)
    assert admitted is False


# ── config.yaml bridging ─────────────────────────────────────────────


def test_config_bridges_category_rules(monkeypatch, tmp_path):
    """gateway/config.py bridges the three discord.*_categories keys to env vars."""
    import yaml
    import os

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump(
            {
                "discord": {
                    "free_response_categories": ["111", "222"],
                    "ignored_categories": ["333"],
                    "require_mention_categories": ["444"],
                },
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", "")
    monkeypatch.setenv("DISCORD_IGNORED_CATEGORIES", "")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION_CATEGORIES", "")

    from gateway.config import load_gateway_config

    load_gateway_config()

    assert os.getenv("DISCORD_FREE_RESPONSE_CATEGORIES") == "111,222"
    assert os.getenv("DISCORD_IGNORED_CATEGORIES") == "333"
    assert os.getenv("DISCORD_REQUIRE_MENTION_CATEGORIES") == "444"
