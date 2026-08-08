"""Tests for Discord per-channel message controls."""

import os
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config


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
from plugins.platforms.discord.adapter import DiscordAdapter, _apply_yaml_config  # noqa: E402


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)
        self.topic = None


class FakeThread:
    def __init__(self, channel_id: int = 1, name: str = "thread", parent=None, guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
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


def make_message(*, channel, content: str, mentions=None):
    author = SimpleNamespace(id=42, display_name="TestUser", name="TestUser")
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


# ── ignored_channels ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ignored_channel_blocks_even_with_mention(adapter, monkeypatch):
    """Ignored channels take priority — even @mentions are dropped."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500")

    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=500),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_ignored_channel_processes_normally(adapter, monkeypatch):
    """Channels not in the ignored list process normally."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500,600")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    # Stub auto-thread creation so this test focuses on ignored-channel
    # routing only — auto-thread failures now correctly skip agent invocation
    # (#20243), which would otherwise mask the assertion below.
    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))

    message = make_message(channel=FakeTextChannel(channel_id=700), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_ignored_channels_empty_string_ignores_nothing(adapter, monkeypatch):
    """Empty DISCORD_IGNORED_CHANNELS means nothing is ignored."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    # Stub auto-thread creation so this test focuses on ignored-channel
    # routing only — auto-thread failures now correctly skip agent invocation
    # (#20243), which would otherwise mask the assertion below.
    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))

    message = make_message(channel=FakeTextChannel(channel_id=500), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


# ── require_mention_channels ─────────────────────────────────────────


def test_require_mention_channels_default_empty(adapter, monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION_CHANNELS", raising=False)
    assert adapter._discord_require_mention_channels() == set()


def test_require_mention_channels_parses_csv_and_list(adapter, monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION_CHANNELS", raising=False)
    adapter.config.extra["require_mention_channels"] = "500, 600"
    assert adapter._discord_require_mention_channels() == {"500", "600"}

    adapter.config.extra["require_mention_channels"] = ["700", "800"]
    assert adapter._discord_require_mention_channels() == {"700", "800"}


def test_require_mention_channels_yaml_bridge_seeds_profile_extra(monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION_CHANNELS", raising=False)

    seeded = _apply_yaml_config({}, {"require_mention_channels": ["500", "600"]})

    assert seeded is not None
    assert seeded["require_mention_channels"] == ["500", "600"]
    assert "DISCORD_REQUIRE_MENTION_CHANNELS" not in os.environ


def test_require_mention_channels_do_not_leak_between_profiles(monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION_CHANNELS", raising=False)

    first_extra = _apply_yaml_config({}, {"require_mention_channels": ["500"]})
    second_extra = _apply_yaml_config({}, {})
    first = DiscordAdapter(
        PlatformConfig(enabled=True, token="first-token", extra=first_extra or {})
    )
    second = DiscordAdapter(
        PlatformConfig(enabled=True, token="second-token", extra=second_extra or {})
    )

    assert first._discord_require_mention_channels() == {"500"}
    assert second._discord_require_mention_channels() == set()
    assert "DISCORD_REQUIRE_MENTION_CHANNELS" not in os.environ


def test_require_mention_channels_loads_from_config_yaml(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "discord:\n  require_mention_channels:\n    - '500'\n    - '600'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION_CHANNELS", raising=False)

    config = load_gateway_config()

    assert config.platforms[Platform.DISCORD].extra["require_mention_channels"] == [
        "500",
        "600",
    ]


@pytest.mark.asyncio
async def test_targeted_channel_requires_mention_when_global_policy_is_free(adapter, monkeypatch):
    """A profile can require mentions in one channel without changing others."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["require_mention_channels"] = ["500"]

    message = make_message(channel=FakeTextChannel(channel_id=500), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_targeted_channel_overrides_free_response(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["free_response_channels"] = "500"
    adapter.config.extra["require_mention_channels"] = "500"

    message = make_message(channel=FakeTextChannel(channel_id=500), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_targeted_channel_accepts_explicit_self_mention(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["require_mention_channels"] = "500"
    adapter.config.extra["history_backfill_limit"] = 0
    bot_user = adapter._client.user

    message = make_message(
        channel=FakeTextChannel(channel_id=500),
        content=f"<@{bot_user.id}> hello",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_targeted_parent_channel_requires_mentions_in_bot_threads(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["require_mention_channels"] = "500"
    parent = FakeTextChannel(channel_id=500)
    thread = FakeThread(channel_id=501, parent=parent)
    adapter._threads.mark("501")

    message = make_message(channel=thread, content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_targeted_channel_policy_does_not_change_other_channels(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["require_mention_channels"] = ["500"]

    message = make_message(channel=FakeTextChannel(channel_id=600), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


# ── no_thread_channels ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_thread_channel_skips_auto_thread(adapter, monkeypatch):
    """Channels in no_thread_channels should not auto-create threads."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_NO_THREAD_CHANNELS", "800")
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))

    message = make_message(channel=FakeTextChannel(channel_id=800), content="hello")
    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "group"


# ── auto-thread failure must not silently fall back to inline (#20243) ──


@pytest.mark.asyncio
async def test_auto_thread_failure_skips_agent_and_notifies_user(adapter, monkeypatch):
    """Auto-thread creation failure must not trigger an inline parent-channel reply.

    Before #20243, ``effective_channel = auto_threaded_channel or message.channel``
    silently routed the response back to the parent channel when thread creation
    failed, breaking thread-first Discord workflows. The fix surfaces a short
    visible error to the parent channel and skips agent invocation entirely so
    the user can retry.
    """
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.delenv("DISCORD_NO_THREAD_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    adapter._auto_create_thread = AsyncMock(return_value=None)

    channel = FakeTextChannel(channel_id=800)
    channel.send = AsyncMock()
    message = make_message(channel=channel, content="hello")
    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_awaited_once()
    # Agent must NOT be invoked when the routing target failed.
    adapter.handle_message.assert_not_awaited()
    # User gets a visible explanation in the parent channel instead of a silent
    # inline reply.
    channel.send.assert_awaited_once()
    sent_text = channel.send.await_args.args[0]
    assert "could not create" in sent_text.lower()
    assert "thread" in sent_text.lower()


# ── config.py bridging ───────────────────────────────────────────────


def test_config_bridges_ignored_channels(monkeypatch, tmp_path):
    """gateway/config.py bridges discord.ignored_channels to env var."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "discord": {
            "ignored_channels": ["111", "222"],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("DISCORD_IGNORED_CHANNELS") == "111,222"


