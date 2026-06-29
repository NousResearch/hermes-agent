"""Tests for Discord ignored_channels and no_thread_channels config."""

from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import asyncio
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


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(id=12345, name=guild_name)
        self.topic = None
        self.sent_messages = []
        self.created_threads = []

    async def send(self, content, **kwargs):
        self.sent_messages.append(content)
        return SimpleNamespace(id=9000 + len(self.sent_messages), content=content, create_thread=AsyncMock())

    async def create_thread(self, name: str, **kwargs):
        thread = FakeThread(channel_id=8000 + len(self.created_threads), name=name, parent=self)
        self.created_threads.append(thread)
        return thread


class FakeThread:
    def __init__(self, channel_id: int = 1, name: str = "thread", parent=None, guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(id=12345, name=guild_name)
        self.topic = None
        self.sent_messages = []
        self.added_users = []
        self._history_messages = []

    async def send(self, content, **kwargs):
        self.sent_messages.append(content)
        author = SimpleNamespace(display_name="Pixoid", name="Pixoid", bot=True)
        msg = SimpleNamespace(id=9100 + len(self.sent_messages), content=content, clean_content=content, author=author)
        self._history_messages.append(msg)
        return msg

    async def add_user(self, user):
        self.added_users.append(user)

    def history(self, *, limit=100, oldest_first=False):
        messages = list(self._history_messages)[:limit]
        if not oldest_first:
            messages.reverse()

        async def _iter():
            for msg in messages:
                yield msg

        return _iter()


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    monkeypatch.delenv("DISCORD_ALLOWED_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(*, channel, content: str, mentions=None, role_mentions=None):
    author = SimpleNamespace(id=42, display_name="TestUser", name="TestUser")
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        role_mentions=list(role_mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        guild=getattr(channel, "guild", None),
    )


# ── ignored_channels ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ignored_channel_blocks_message(adapter, monkeypatch):
    """Messages in ignored channels are silently dropped."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(channel=FakeTextChannel(channel_id=500), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


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
async def test_managed_bot_role_mention_counts_as_bot_mention(adapter, monkeypatch):
    """Selecting a bot's managed @role should route like a direct bot mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    bot_role = SimpleNamespace(id=1234, managed=True)
    other_role = SimpleNamespace(id=5678, managed=True)
    channel = FakeTextChannel(channel_id=700)
    channel.guild.me = SimpleNamespace(roles=[bot_role])
    message = make_message(
        channel=channel,
        content=f"<@&{bot_role.id}> hello",
        role_mentions=[bot_role],
        mentions=[],
    )

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"

    adapter.handle_message.reset_mock()
    message = make_message(
        channel=channel,
        content=f"<@&{other_role.id}> hello",
        role_mentions=[other_role],
        mentions=[],
    )
    await adapter._handle_message(message)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_role_mention_with_human_mention_counts_as_bot_call(adapter, monkeypatch):
    """@Crew plus @Human should still call every bot that has the Crew role."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    crew_role = SimpleNamespace(id=2468, managed=False)
    human = SimpleNamespace(id=42, bot=False)
    channel = FakeTextChannel(channel_id=700)
    channel.guild.me = SimpleNamespace(roles=[crew_role])
    message = make_message(
        channel=channel,
        content=f"<@&{crew_role.id}> say hi to <@{human.id}>",
        role_mentions=[crew_role],
        mentions=[human],
    )

    assert adapter._message_mentions_own_role(message) == [crew_role]

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == f"say hi to <@{human.id}>"


@pytest.mark.asyncio
async def test_configured_crew_alias_counts_as_bot_call(adapter, monkeypatch):
    """Configured crew phrases wake each participating bot without a direct tag."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_CREW_ALIASES", "crew:,get the crew,calling the crew")
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(
        channel=FakeTextChannel(channel_id=700),
        content="crew: discuss this together",
        mentions=[],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "crew: discuss this together"


@pytest.mark.asyncio
async def test_council_mode_opens_huddle_invites_workers_closes_and_synthesizes(adapter, monkeypatch):
    """Coordinator council mode turns a crew call into a bounded huddle, not a dogpile."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_CREW_ALIASES", "crew:")
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    adapter.config.extra["council_mode"] = {
        "enabled": True,
        "wait_seconds": 0,
        "workers": [
            {"name": "Boba", "id": "111", "role": "reality check"},
            {"name": "Quill", "id": "222", "role": "docs"},
            {"name": "Tinker", "id": "333", "role": "implementation"},
        ],
    }
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_user=lambda user_id: SimpleNamespace(id=user_id),
    )

    channel = FakeTextChannel(channel_id=700)
    message = make_message(channel=channel, content="crew: decide the multiplayer flow", mentions=[])
    await adapter._handle_message(message)

    assert len(channel.created_threads) == 1
    huddle = channel.created_threads[0]
    assert huddle.id in {8000, "8000"} or str(huddle.id) == "8000"
    assert len(huddle.added_users) == 3
    assert "<@111>" in huddle.sent_messages[0]
    assert not huddle.sent_messages[0].startswith("Huddle closed")
    assert "@Crew is coordinator-led" in channel.sent_messages[0]
    assert "bounded crew huddle" in channel.sent_messages[0]
    assert "@Boba" in channel.sent_messages[0]

    for _ in range(10):
        if adapter.handle_message.await_count:
            break
        await asyncio.sleep(0)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.text.startswith("[COUNCIL MODE SYNTHESIS]")
    assert "Huddle transcript" in event.text
    assert "Huddle closed" in huddle.sent_messages[-1]
    assert adapter._council_huddle_threads[str(huddle.id)] == "closed"


@pytest.mark.asyncio
async def test_council_mode_no_tools_smoke_test_explains_no_huddle(adapter, monkeypatch):
    """@Crew + no-tools should be legible and should not wake workers."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_CREW_ALIASES", "crew:")
    adapter.config.extra["council_mode"] = {
        "enabled": True,
        "wait_seconds": 0,
        "workers": [{"name": "Boba", "id": "111"}],
    }

    channel = FakeTextChannel(channel_id=700)
    message = make_message(
        channel=channel,
        content="crew: can we test the multiplayer experience? just reply yes; dont run any tools yet",
        mentions=[],
    )
    await adapter._handle_message(message)

    assert channel.created_threads == []
    adapter.handle_message.assert_not_awaited()
    assert len(channel.sent_messages) == 1
    assert "Yes — I hear you" in channel.sent_messages[0]
    assert "asked me not to run tools" in channel.sent_messages[0]
    assert "won't open a huddle" in channel.sent_messages[0]
    assert "@Boba" in channel.sent_messages[0]


@pytest.mark.asyncio
async def test_council_mode_missing_workers_reports_route_gap_not_bare_answer(adapter, monkeypatch):
    """Enabled @Crew route with no workers should not collapse to the LLM."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_CREW_ALIASES", "crew:")
    adapter.config.extra["council_mode"] = {"enabled": True, "wait_seconds": 0, "workers": []}

    channel = FakeTextChannel(channel_id=700)
    message = make_message(
        channel=channel,
        content="crew: can we test the multiplayer experience? just reply yes if you hear it from me",
        mentions=[],
    )
    await adapter._handle_message(message)

    assert channel.created_threads == []
    adapter.handle_message.assert_not_awaited()
    assert len(channel.sent_messages) == 1
    assert "@Crew is coordinator-led" in channel.sent_messages[0]
    assert "couldn't start a crew huddle" in channel.sent_messages[0]
    assert "no council workers are configured" in channel.sent_messages[0]


@pytest.mark.asyncio
async def test_council_huddle_suppresses_ambient_worker_chatter(adapter, monkeypatch):
    """Open/closed huddle threads do not wake Pixoid on unmentioned worker messages."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    parent = FakeTextChannel(channel_id=700)
    thread = FakeThread(channel_id=8000, name="crew-huddle", parent=parent)
    adapter._council_huddle_threads[str(thread.id)] = "open"
    worker_author = SimpleNamespace(id=111, display_name="Boba", name="Boba", bot=True)
    message = make_message(channel=thread, content="Boba view: risk here", mentions=[])
    message.author = worker_author

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_council_mode_reuses_current_workbench_thread(adapter, monkeypatch):
    """@Crew inside an agent-workbench thread invites workers into that thread."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_CREW_ALIASES", "crew:")
    adapter.config.extra["council_mode"] = {
        "enabled": True,
        "wait_seconds": 0,
        "workbench_channel_id": "700",
        "workers": [{"name": "Tinker", "id": "333"}],
    }
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_channel=lambda channel_id: None,
        fetch_channel=AsyncMock(),
        get_user=lambda user_id: SimpleNamespace(id=user_id),
    )
    parent = FakeTextChannel(channel_id=700, name="agent-workbench")
    thread = FakeThread(channel_id=8000, name="existing", parent=parent)
    message = make_message(channel=thread, content="crew: discuss in here", mentions=[])

    await adapter._handle_message(message)

    assert parent.created_threads == []
    assert len(thread.added_users) == 1
    assert "<@333>" in thread.sent_messages[0]


@pytest.mark.asyncio
async def test_unconfigured_crew_word_does_not_wake_bot(adapter, monkeypatch):
    """The word crew is not ambient wakeup unless an explicit alias is configured."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_CREW_ALIASES", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(
        channel=FakeTextChannel(channel_id=700),
        content="the crew should discuss things",
        mentions=[],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_ignored_channel_processes_normally(adapter, monkeypatch):
    """Channels not in the ignored list process normally."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500,600")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(channel=FakeTextChannel(channel_id=700), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_ignored_channels_csv_parsing(adapter, monkeypatch):
    """Multiple channel IDs are parsed correctly from CSV."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500, 600 , 700")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    for ch_id in (500, 600, 700):
        adapter.handle_message.reset_mock()
        message = make_message(channel=FakeTextChannel(channel_id=ch_id), content="hello")
        await adapter._handle_message(message)
        adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_ignored_channels_empty_string_ignores_nothing(adapter, monkeypatch):
    """Empty DISCORD_IGNORED_CHANNELS means nothing is ignored."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(channel=FakeTextChannel(channel_id=500), content="hello")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_ignored_channel_thread_parent_match(adapter, monkeypatch):
    """Thread whose parent channel is ignored should also be ignored."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    parent = FakeTextChannel(channel_id=500, name="ignored-channel")
    thread = FakeThread(channel_id=501, name="thread-in-ignored", parent=parent)
    message = make_message(channel=thread, content="hello from thread")
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dms_unaffected_by_ignored_channels(adapter, monkeypatch):
    """DMs should never be affected by ignored_channels."""
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "500")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(channel=FakeDMChannel(channel_id=500), content="dm hello")
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


@pytest.mark.asyncio
async def test_normal_channel_still_auto_threads(adapter, monkeypatch):
    """Channels NOT in no_thread_channels still get auto-threading."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_NO_THREAD_CHANNELS", "800")
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    fake_thread = FakeThread(channel_id=999, name="auto-thread")
    adapter._auto_create_thread = AsyncMock(return_value=fake_thread)

    message = make_message(channel=FakeTextChannel(channel_id=900), content="hello")
    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "thread"


@pytest.mark.asyncio
async def test_no_thread_channels_csv_parsing(adapter, monkeypatch):
    """Multiple no_thread channel IDs parsed from CSV."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_NO_THREAD_CHANNELS", "800, 900")
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(channel_id=999))

    for ch_id in (800, 900):
        adapter._auto_create_thread.reset_mock()
        adapter.handle_message.reset_mock()
        message = make_message(channel=FakeTextChannel(channel_id=ch_id), content="hello")
        await adapter._handle_message(message)
        adapter._auto_create_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_thread_with_auto_thread_disabled_is_noop(adapter, monkeypatch):
    """no_thread_channels is a no-op when auto_thread is globally disabled."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_NO_THREAD_CHANNELS", "800")
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    adapter._auto_create_thread = AsyncMock()

    message = make_message(channel=FakeTextChannel(channel_id=800), content="hello")
    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_create_thread_reuses_existing_starter_thread(adapter):
    """Multi-bot races should reuse the starter's thread, not spawn fallback threads."""
    existing_thread = FakeThread(channel_id=999, name="crew-council")
    refreshed_message = SimpleNamespace(thread=existing_thread)
    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=refreshed_message),
        send=AsyncMock(),
    )
    message = SimpleNamespace(
        id=123,
        content="<@999> <@888> discuss this together",
        author=SimpleNamespace(display_name="Pixiedust"),
        channel=channel,
        thread=None,
        create_thread=AsyncMock(side_effect=RuntimeError("message already has a thread")),
    )

    thread = await adapter._auto_create_thread(message)

    assert thread is existing_thread
    channel.fetch_message.assert_awaited_once_with(123)
    channel.send.assert_not_awaited()


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


def test_config_bridges_no_thread_channels(monkeypatch, tmp_path):
    """gateway/config.py bridges discord.no_thread_channels to env var."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "discord": {
            "no_thread_channels": ["333"],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_NO_THREAD_CHANNELS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("DISCORD_NO_THREAD_CHANNELS") == "333"


def test_config_bridges_allow_bots(monkeypatch, tmp_path):
    """gateway/config.py bridges discord.allow_bots to DISCORD_ALLOW_BOTS."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "discord": {
            "allow_bots": "mentions",
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("DISCORD_ALLOW_BOTS") == "mentions"


def test_config_bridges_crew_aliases(monkeypatch, tmp_path):
    """gateway/config.py bridges discord.crew_aliases to DISCORD_CREW_ALIASES."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "discord": {
            "crew_aliases": ["crew:", "get the crew"],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_CREW_ALIASES", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("DISCORD_CREW_ALIASES") == "crew:,get the crew"


def test_config_env_var_takes_precedence(monkeypatch, tmp_path):
    """Env vars should take precedence over config.yaml values."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "discord": {
            "ignored_channels": ["111"],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "999")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    # Env var should NOT be overwritten
    assert os.getenv("DISCORD_IGNORED_CHANNELS") == "999"
