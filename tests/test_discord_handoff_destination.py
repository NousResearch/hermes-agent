"""Real discord.py + SQLite handoff routing (outside gateway's SDK-mocking conftest)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

discord = pytest.importorskip("discord")

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig, load_gateway_config
from gateway.run_startup import GatewayStartupMixin
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True])
async def test_handoff_delivery_and_followup_share_destination(tmp_path, inline):
    home_id, thread_id, owner_id = "100", "200", "300"
    cfg = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(
        enabled=True, token="test", extra={"no_thread_channels": [home_id] if inline else []},
        home_channel=HomeChannel(Platform.DISCORD, home_id, "home", user_id=owner_id),
    )})
    adapter = DiscordAdapter(cfg.platforms[Platform.DISCORD])
    state = MagicMock()
    guild = MagicMock(id=400)
    home = discord.TextChannel(state=state, guild=guild, data={
        "id": home_id, "name": "home", "type": 0, "position": 0, "permission_overwrites": [],
    })
    channels = {int(home_id): home}

    async def create_thread(channel_id, **kwargs):
        data = {
            "id": thread_id, "parent_id": home_id, "owner_id": "999", "name": kwargs["name"],
            "type": kwargs["type"], "message_count": 0, "member_count": 1,
            "thread_metadata": {"archived": False, "auto_archive_duration": 1440,
                                "archive_timestamp": "2026-09-01T00:00:00+00:00"},
        }
        channels[int(thread_id)] = discord.Thread(state=state, guild=guild, data=data)
        return data

    state.http.start_thread_without_message = AsyncMock(side_effect=create_thread)
    adapter._client = SimpleNamespace(get_channel=lambda cid: channels.get(cid))
    # Keep the real adapter's send/format/split path; only the remote HTTP I/O is fake.
    state.http.send_message = AsyncMock(return_value={"id": "500"})
    state.create_message.return_value = SimpleNamespace(id=500)

    class Runner(GatewayStartupMixin):
        pass

    runner = Runner()
    runner.config, runner.adapters = cfg, {Platform.DISCORD: adapter}
    store = SessionStore(tmp_path / "sessions", cfg)
    runner.async_session_store = AsyncSessionStore(store)
    original = store.get_or_create_session(SessionSource(
        platform=Platform.LOCAL, chat_id="desktop", chat_type="dm", user_id=owner_id,
    ))
    runner._evict_cached_agent = lambda key: None
    runner._release_running_agent_state = lambda key: None
    runner._handle_message = AsyncMock(return_value="Handoff confirmed")

    await runner._process_handoff({"id": original.session_id, "handoff_platform": "discord"})

    target = home_id if inline else thread_id
    assert state.http.send_message.call_args.args[0] == int(target)
    if inline:
        state.http.start_thread_without_message.assert_not_awaited()
    else:
        assert channels[int(target)].type == discord.ChannelType.public_thread
    organic = adapter.build_source(
        chat_id=target, chat_type="group" if inline else "thread", user_id=owner_id,
        thread_id=None if inline else thread_id, guild_id=str(guild.id),
    )
    followup = store.get_or_create_session(organic)
    assert followup.session_id == original.session_id
    assert runner._handle_message.call_args.args[0].source.user_id == owner_id


@pytest.mark.parametrize("env_target", ["100", "200"])
def test_restart_retains_home_owner_only_for_the_same_destination(tmp_path, monkeypatch, env_target):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-not-a-real-token")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", env_target)
    (tmp_path / "config.yaml").write_text(
        "platforms:\n  discord:\n    enabled: true\n    home_channel:\n"
        "      platform: discord\n      chat_id: '100'\n      user_id: '300'\n",
        encoding="utf-8",
    )
    home = load_gateway_config().get_home_channel(Platform.DISCORD)
    assert home.chat_id == env_target
    assert home.user_id == ("300" if env_target == "100" else None)
