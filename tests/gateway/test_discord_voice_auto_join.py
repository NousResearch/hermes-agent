"""Tests for opt-in "auto-join voice channel on user join" (discord.voice.*).

Registers the real ``connect()`` event handlers against a FakeBot (same
pattern as test_discord_connect.py), then invokes the captured
``on_voice_state_update`` handler directly with a simulated voice-state
event to exercise the auto-join branch without a live discord.py gateway.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

# Reuse the same discord.py mock bootstrap as test_discord_connect.py so this
# file works standalone regardless of test collection order.
if "discord" not in sys.modules or not hasattr(sys.modules.get("discord"), "__file__"):
    from tests.gateway.test_discord_connect import _ensure_discord_mock
    _ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from tests.gateway.test_discord_connect import FakeBot  # noqa: E402


def _make_voice_state(channel):
    return SimpleNamespace(channel=channel)


def _make_member(user_id, guild, *, name="testuser", bot=False):
    return SimpleNamespace(
        id=user_id,
        name=name,
        display_name=name,
        guild=guild,
        bot=bot,
    )


async def _connect_adapter(monkeypatch, adapter):
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(
        message_content=False, dm_messages=False, guild_messages=False,
        members=False, voice_states=False,
    )
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    created = {}

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        created["bot"] = FakeBot(intents=intents, allowed_mentions=allowed_mentions)
        return created["bot"]

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    ok = await adapter.connect()
    assert ok is True
    return created["bot"]


@pytest.fixture
def adapter(monkeypatch):
    a = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    yield a


def test_auto_join_config_normalizes_cli_string_lists(monkeypatch, adapter):
    """`hermes config set ... '[...]'` must not turn allowlists into characters."""
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "discord": {
                "voice": {
                    "auto_join_users": '["552613509909971005"]',
                    "auto_join_voice_channels": '["Development"]',
                }
            }
        },
    )

    config = adapter._load_voice_auto_join_config()

    assert config["auto_join_users"] == ["552613509909971005"]
    assert config["auto_join_voice_channels"] == ["Development"]


@pytest.mark.asyncio
async def test_auto_join_called_when_enabled_and_user_allowed(monkeypatch, adapter):
    adapter._voice_auto_join_cfg = {"auto_join_on_user_join": True, "auto_join_users": []}
    adapter._allowed_user_ids = {"42"}
    bot = await _connect_adapter(monkeypatch, adapter)

    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    member = _make_member(42, guild)
    channel = SimpleNamespace(id=99, name="General")
    before = _make_voice_state(None)
    after = _make_voice_state(channel)

    await bot._events["on_voice_state_update"](member, before, after)

    adapter.join_voice_channel.assert_awaited_once_with(channel)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_binds_configured_text_channel_for_transcript_routing(monkeypatch, adapter):
    """Auto-join must bind a text/session anchor.

    Without this, the adapter can join the VC and transcribe speech, but
    GatewayRunner._handle_voice_channel_input has no chat to route the
    synthetic VOICE MessageEvent through.
    """
    adapter._voice_auto_join_cfg = {
        "auto_join_on_user_join": True,
        "auto_join_users": [],
        "auto_join_text_channel_id": "123456789",
    }
    adapter._allowed_user_ids = {"42"}
    bot = await _connect_adapter(monkeypatch, adapter)

    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    member = _make_member(42, guild, name="mxu")
    channel = SimpleNamespace(id=99, name="General")

    await bot._events["on_voice_state_update"](
        member,
        _make_voice_state(None),
        _make_voice_state(channel),
    )

    adapter.join_voice_channel.assert_awaited_once_with(channel)
    assert adapter._voice_text_channels[guild.id] == 123456789
    assert "123456789" in adapter._auto_tts_enabled_chats
    assert "123456789" not in adapter._auto_tts_disabled_chats
    assert adapter._voice_sources[guild.id]["platform"] == "discord"
    assert adapter._voice_sources[guild.id]["chat_id"] == "123456789"
    assert adapter._voice_sources[guild.id]["user_id"] == "42"
    assert adapter._voice_sources[guild.id]["scope_id"] == "1"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_not_called_when_flag_off(monkeypatch, adapter):
    adapter._voice_auto_join_cfg = {"auto_join_on_user_join": False, "auto_join_users": []}
    adapter._allowed_user_ids = {"42"}
    bot = await _connect_adapter(monkeypatch, adapter)

    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    member = _make_member(42, guild)
    channel = SimpleNamespace(id=99, name="General")
    before = _make_voice_state(None)
    after = _make_voice_state(channel)

    await bot._events["on_voice_state_update"](member, before, after)

    adapter.join_voice_channel.assert_not_awaited()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_not_called_for_bot_member(monkeypatch, adapter):
    adapter._voice_auto_join_cfg = {"auto_join_on_user_join": True, "auto_join_users": []}
    adapter._allowed_user_ids = {"*"}
    bot = await _connect_adapter(monkeypatch, adapter)

    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    member = _make_member(7, guild, bot=True)
    channel = SimpleNamespace(id=99, name="General")
    before = _make_voice_state(None)
    after = _make_voice_state(channel)

    await bot._events["on_voice_state_update"](member, before, after)

    adapter.join_voice_channel.assert_not_awaited()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_not_called_when_user_not_allowed(monkeypatch, adapter):
    adapter._voice_auto_join_cfg = {"auto_join_on_user_join": True, "auto_join_users": []}
    adapter._allowed_user_ids = {"999"}  # 42 is not in the allowlist
    bot = await _connect_adapter(monkeypatch, adapter)

    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    member = _make_member(42, guild)
    channel = SimpleNamespace(id=99, name="General")
    before = _make_voice_state(None)
    after = _make_voice_state(channel)

    await bot._events["on_voice_state_update"](member, before, after)

    adapter.join_voice_channel.assert_not_awaited()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_respects_trigger_user_list(monkeypatch, adapter):
    """When auto_join_users is non-empty, only listed users trigger auto-join
    even if other users pass the general allowlist."""
    adapter._voice_auto_join_cfg = {
        "auto_join_on_user_join": True,
        "auto_join_users": ["42"],
    }
    adapter._allowed_user_ids = {"42", "43"}
    bot = await _connect_adapter(monkeypatch, adapter)

    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    channel = SimpleNamespace(id=99, name="General")

    # User 43 is allowed generally but not in the trigger list -> no auto-join.
    member_43 = _make_member(43, guild)
    await bot._events["on_voice_state_update"](member_43, _make_voice_state(None), _make_voice_state(channel))
    adapter.join_voice_channel.assert_not_awaited()

    # User 42 is in the trigger list -> auto-join fires.
    member_42 = _make_member(42, guild)
    await bot._events["on_voice_state_update"](member_42, _make_voice_state(None), _make_voice_state(channel))
    adapter.join_voice_channel.assert_awaited_once_with(channel)

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_skipped_when_already_connected_to_that_channel(monkeypatch, adapter):
    adapter._voice_auto_join_cfg = {"auto_join_on_user_join": True, "auto_join_users": []}
    adapter._allowed_user_ids = {"42"}
    bot = await _connect_adapter(monkeypatch, adapter)

    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    channel = SimpleNamespace(id=99, name="General")
    existing_vc = MagicMock()
    existing_vc.is_connected.return_value = True
    existing_vc.channel = channel
    adapter._voice_clients[guild.id] = existing_vc

    member = _make_member(42, guild)
    await bot._events["on_voice_state_update"](member, _make_voice_state(None), _make_voice_state(channel))

    adapter.join_voice_channel.assert_not_awaited()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_manual_voice_join_path_unaffected(monkeypatch, adapter):
    """Sanity: with auto-join disabled (the default), on_voice_state_update
    must still run its existing join/leave/switch tracking without raising,
    keeping /voice join as the only trigger.
    """
    adapter._voice_auto_join_cfg = {"auto_join_on_user_join": False, "auto_join_users": []}
    bot = await _connect_adapter(monkeypatch, adapter)

    guild = SimpleNamespace(id=1)
    channel_a = SimpleNamespace(id=1, name="A")
    channel_b = SimpleNamespace(id=2, name="B")
    member = _make_member(1, guild)

    # join, switch, leave — should not raise even though the bot has no
    # voice client tracked in this guild.
    await bot._events["on_voice_state_update"](member, _make_voice_state(None), _make_voice_state(channel_a))
    await bot._events["on_voice_state_update"](member, _make_voice_state(channel_a), _make_voice_state(channel_b))
    await bot._events["on_voice_state_update"](member, _make_voice_state(channel_b), _make_voice_state(None))

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_join_voice_channel_restarts_missing_receiver_when_already_connected(monkeypatch, adapter):
    """A retry of /voice channel must repair a connected-but-deaf VC state."""
    started = []

    class FakeReceiver:
        def __init__(self, vc, allowed_user_ids=None):
            self._running = False
            self.vc = vc
            self.allowed_user_ids = allowed_user_ids

        def start(self):
            self._running = True
            started.append(self.vc)

        def stop(self):
            self._running = False

    async def fake_loop(_guild_id):
        await asyncio.sleep(0)

    monkeypatch.setattr(discord_platform, "VoiceReceiver", FakeReceiver)
    monkeypatch.setattr(adapter, "_voice_listen_loop", fake_loop)
    adapter._client = object()
    monkeypatch.setattr(discord_platform, "DISCORD_AVAILABLE", True)

    guild = SimpleNamespace(id=1)
    channel = SimpleNamespace(id=99, name="General", guild=guild)
    existing_vc = MagicMock()
    existing_vc.is_connected.return_value = True
    existing_vc.channel = channel
    adapter._voice_clients[guild.id] = existing_vc

    assert await adapter.join_voice_channel(channel) is True

    assert started == [existing_vc]
    assert adapter._voice_receivers[guild.id]._running is True
    assert guild.id in adapter._voice_listen_tasks
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_join_voice_channel_keeps_running_receiver_when_already_connected(monkeypatch, adapter):
    """The existing fast path remains cheap when the receiver/listen loop is healthy."""
    created = []

    class FakeReceiver:
        def __init__(self, vc, allowed_user_ids=None):
            created.append(vc)
            self._running = False

        def start(self):
            self._running = True

        def stop(self):
            self._running = False

    monkeypatch.setattr(discord_platform, "VoiceReceiver", FakeReceiver)
    adapter._client = object()
    monkeypatch.setattr(discord_platform, "DISCORD_AVAILABLE", True)

    guild = SimpleNamespace(id=1)
    channel = SimpleNamespace(id=99, name="General", guild=guild)
    existing_vc = MagicMock()
    existing_vc.is_connected.return_value = True
    existing_vc.channel = channel
    adapter._voice_clients[guild.id] = existing_vc
    healthy_receiver = SimpleNamespace(_running=True)
    healthy_task = MagicMock()
    healthy_task.done.return_value = False
    adapter._voice_receivers[guild.id] = healthy_receiver
    adapter._voice_listen_tasks[guild.id] = healthy_task

    assert await adapter.join_voice_channel(channel) is True

    assert created == []
    assert adapter._voice_receivers[guild.id] is healthy_receiver
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_respects_voice_channel_name_filter(monkeypatch, adapter):
    adapter._voice_auto_join_cfg = {
        "auto_join_on_user_join": True,
        "auto_join_users": ["42"],
        "auto_join_voice_channels": ["Development"],
        "auto_join_text_channel_id": "123456789",
    }
    adapter._allowed_user_ids = {"42"}
    bot = await _connect_adapter(monkeypatch, adapter)
    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    member = _make_member(42, guild, name="mxu")
    general = SimpleNamespace(id=98, name="General")
    development = SimpleNamespace(id=99, name="Development")

    await bot._events["on_voice_state_update"](
        member,
        _make_voice_state(None),
        _make_voice_state(general),
    )
    adapter.join_voice_channel.assert_not_awaited()

    await bot._events["on_voice_state_update"](
        member,
        _make_voice_state(None),
        _make_voice_state(development),
    )
    adapter.join_voice_channel.assert_awaited_once_with(development)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_respects_voice_channel_id_filter(monkeypatch, adapter):
    adapter._voice_auto_join_cfg = {
        "auto_join_on_user_join": True,
        "auto_join_users": ["42"],
        "auto_join_voice_channels": ["99"],
        "auto_join_text_channel_id": "123456789",
    }
    adapter._allowed_user_ids = {"42"}
    bot = await _connect_adapter(monkeypatch, adapter)
    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    member = _make_member(42, guild, name="mxu")
    channel = SimpleNamespace(id=99, name="Renamed Later")

    await bot._events["on_voice_state_update"](
        member,
        _make_voice_state(None),
        _make_voice_state(channel),
    )
    adapter.join_voice_channel.assert_awaited_once_with(channel)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_follows_allowed_user_when_switching_voice_channels(monkeypatch, adapter):
    """Moving between VCs should be treated like a fresh join for follow-me bots."""
    adapter._voice_auto_join_cfg = {
        "auto_join_on_user_join": True,
        "auto_join_users": ["42"],
        "auto_join_voice_channels": ["Development"],
        "auto_join_text_channel_id": "123456789",
    }
    adapter._allowed_user_ids = {"42"}
    bot = await _connect_adapter(monkeypatch, adapter)
    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    member = _make_member(42, guild, name="mxu")
    general = SimpleNamespace(id=98, name="General")
    development = SimpleNamespace(id=99, name="Development")

    await bot._events["on_voice_state_update"](
        member,
        _make_voice_state(general),
        _make_voice_state(development),
    )

    adapter.join_voice_channel.assert_awaited_once_with(development)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_auto_join_handles_existing_voice_client_with_no_channel(monkeypatch, adapter):
    """discord.py can expose a transient connected client with channel=None during moves."""
    adapter._voice_auto_join_cfg = {"auto_join_on_user_join": True, "auto_join_users": ["42"]}
    adapter._allowed_user_ids = {"42"}
    bot = await _connect_adapter(monkeypatch, adapter)
    adapter.join_voice_channel = AsyncMock(return_value=True)

    guild = SimpleNamespace(id=1)
    existing_vc = MagicMock()
    existing_vc.is_connected.return_value = True
    existing_vc.channel = None
    adapter._voice_clients[guild.id] = existing_vc

    member = _make_member(42, guild, name="mxu")
    channel = SimpleNamespace(id=99, name="Development")

    await bot._events["on_voice_state_update"](
        member,
        _make_voice_state(None),
        _make_voice_state(channel),
    )

    adapter.join_voice_channel.assert_awaited_once_with(channel)
    await adapter.disconnect()
