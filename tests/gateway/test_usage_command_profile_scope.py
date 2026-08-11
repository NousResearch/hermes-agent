"""Regression coverage for Discord /usage profile routing.

Issue #69178: native Discord slash events must retain guild context so /usage
looks up persisted agent usage in the routed profile's session namespace.
"""

import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.Interaction = object

        class _FakeGroup:
            def __init__(self, *, name, description, parent=None):
                self.name = name
                self.description = description
                self.parent = parent
                self._children = {}
                if parent is not None:
                    parent.add_command(self)

            def add_command(self, command):
                self._children[command.name] = command

        class _FakeCommand:
            def __init__(self, *, name, description, callback, parent=None):
                self.name = name
                self.description = description
                self.callback = callback
                self.parent = parent

        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            autocomplete=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
            Group=_FakeGroup,
            Command=_FakeCommand,
        )

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)

    app_commands = getattr(sys.modules["discord"], "app_commands", None)
    if app_commands is not None and not hasattr(app_commands, "autocomplete"):
        app_commands.autocomplete = lambda **kwargs: (lambda fn: fn)


_ensure_discord_mock()

from gateway.profile_routing import ProfileRoute  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _StubbableRunner(GatewayRunner):
    """GatewayRunner.async_session_store is a read-only property; shadow it
    with a plain class attribute so the bare test instance can install a fake."""

    async_session_store = None


CHAT_ID = "123"
USER_ID = "42"
GUILD_ID = "456"
OTHER_GUILD_ID = "789"
WORK_PROFILE = "workprof"
NO_USAGE_DATA = "No usage data available for this session."


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator

    def add_command(self, command):
        self.commands[command.name] = command

    def get_commands(self):
        return [SimpleNamespace(name=name) for name in self.commands]


class _FakeTextChannel:
    """A channel that is neither a Discord thread nor a DM channel."""

    def __init__(self, channel_id=int(CHAT_ID), guild_id=int(GUILD_ID)):
        self.id = channel_id
        self.name = "general"
        self.guild = SimpleNamespace(name="TestGuild", id=guild_id)
        self.topic = None

    def history(self, *args, **kwargs):
        async def _empty():
            return
            yield

        return _empty()


class _EmptyAsyncSessionStore:
    async def get_or_create_session(self, source):
        return SimpleNamespace(session_id="empty-session")

    async def load_transcript(self, session_id):
        return []


class _UsageAgent:
    provider = None
    base_url = None
    api_key = None
    model = "usage-test-model"
    session_input_tokens = 1_234
    session_output_tokens = 567
    session_total_tokens = 1_801
    session_api_calls = 2

    def __init__(self):
        self.context_compressor = SimpleNamespace(
            last_prompt_tokens=0,
            context_length=128_000,
            compression_count=0,
        )

    def get_rate_limit_state(self):
        return None


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    instance = DiscordAdapter(config)
    instance._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _channel_id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    instance._text_batch_delay_seconds = 0
    return instance


@pytest.fixture(autouse=True)
def _isolate_usage_dependencies(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name",
        lambda: "default",
    )
    monkeypatch.setattr(
        "agent.account_usage.nous_credits_lines",
        lambda **kwargs: [],
    )


def _make_runner(route_guild_id):
    runner = object.__new__(_StubbableRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=True,
        profile_routes=[
            ProfileRoute(
                name="work-channel",
                platform="discord",
                profile=WORK_PROFILE,
                guild_id=route_guild_id,
                chat_id=CHAT_ID,
            )
        ],
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_db = None
    runner.async_session_store = _EmptyAsyncSessionStore()
    runner._context_breakdown_lines = lambda agent, source: []
    return runner


def _make_interaction(guild_id=GUILD_ID):
    return SimpleNamespace(
        channel=_FakeTextChannel(guild_id=int(guild_id)),
        channel_id=int(CHAT_ID),
        guild_id=int(guild_id),
        user=SimpleNamespace(id=int(USER_ID), display_name="Jezza"),
    )


def _build_routed_source(adapter, guild_id):
    return adapter.build_source(
        chat_id=CHAT_ID,
        chat_name="TestGuild / #general",
        chat_type="group",
        user_id=USER_ID,
        user_name="Jezza",
        guild_id=guild_id,
    )


@pytest.mark.asyncio
async def test_usage_reads_agent_from_discord_guild_routed_profile(adapter):
    runner = _make_runner(GUILD_ID)
    adapter.gateway_runner = runner

    routed_source = _build_routed_source(adapter, GUILD_ID)
    routed_key = runner._session_key_for_source(routed_source)
    assert routed_source.profile == WORK_PROFILE
    assert routed_key == "agent:workprof:discord:group:123:42"

    runner._running_agents[routed_key] = _UsageAgent()
    event = adapter._build_slash_event(_make_interaction(), "/usage")

    reply = await runner._handle_usage_command(event)

    assert reply != NO_USAGE_DATA
    assert "usage-test-model" in reply
    assert "1,234" in reply
    assert "567" in reply
    assert "1,801" in reply
    assert event.source.profile == WORK_PROFILE
    assert runner._session_key_for_source(event.source) == routed_key


@pytest.mark.asyncio
async def test_usage_without_matching_guild_route_uses_default_namespace(adapter):
    runner = _make_runner(OTHER_GUILD_ID)
    adapter.gateway_runner = runner

    routed_source = _build_routed_source(adapter, OTHER_GUILD_ID)
    routed_key = runner._session_key_for_source(routed_source)
    assert routed_key == "agent:workprof:discord:group:123:42"
    runner._running_agents[routed_key] = _UsageAgent()

    event = adapter._build_slash_event(_make_interaction(GUILD_ID), "/usage")
    default_key = runner._session_key_for_source(event.source)

    reply = await runner._handle_usage_command(event)

    assert event.source.profile is None
    assert default_key == "agent:main:discord:group:123:42"
    assert reply == NO_USAGE_DATA
