"""A configured Discord username keeps exactly its current holder's access across reconnects.

connect() resolves ``allow_from`` usernames to IDs on its first ``on_ready``; a fresh connection must
resolve them again (the account may have renamed), and in a single-profile gateway, where that
resolution rewrites ``DISCORD_ALLOWED_USERS`` to numeric IDs, a later unrelated config.yaml save must
not lock the resolved account out (#154562).
"""
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import hermes_yaml as yaml
import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from tests.gateway.test_discord_connect import FakeBot, _member, discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture
def connectable(tmp_path, monkeypatch):
    """A real connect() against a fake bot whose guild has one member holding ``holder["alice"]``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("DISCORD_ALLOWED_USERS", "HERMES_BRIDGED_DISCORD_ALLOWED_USERS"):
        monkeypatch.setenv(var, "")  # setenv first so teardown also removes what the loader bridges in
        monkeypatch.delenv(var)
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status_scoped_locks.release_scoped_lock", lambda scope, identity: None)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: SimpleNamespace(
        message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False))
    holder = {"alice": 111}

    class GuildBot(FakeBot):
        @property
        def guilds(self):
            return [SimpleNamespace(name="g", members=[_member(holder["alice"], "alice")], member_count=1)]

    monkeypatch.setattr(discord_platform.commands, "Bot", lambda **kw: GuildBot(intents=kw["intents"]))
    monkeypatch.setattr(DiscordAdapter, "_run_post_connect_initialization", AsyncMock())
    return holder


def _write(home, discord_cfg, stamp_offset):
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump({"platforms": {"discord": {"enabled": True, "token": "test-token"}},
                                    "discord": discord_cfg}), encoding="utf-8")
    stamp = time.time_ns() - 10_000_000_000 + stamp_offset * 1_000_000  # distinct, settled mtimes
    os.utime(path, ns=(stamp, stamp))


@pytest.mark.asyncio
async def test_a_fresh_connect_resolves_a_configured_username_again(connectable):
    """A username resolved on one connection must be resolved again on the next connect(): the account
    that held it may have renamed, and the name's new holder is the one the operator allowlisted."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={"allow_from": ["alice"]}))
    token = set_multiplex_context(True)
    try:
        assert await adapter.connect() is True
        assert adapter._is_allowed_user("111") is True
        connectable["alice"] = 222  # 111 renamed; another account now holds "alice"
        assert await adapter.connect() is True
        verdicts = (adapter._is_allowed_user("222"), adapter._is_allowed_user("111"))
    finally:
        reset_multiplex_context(token)
        await adapter.disconnect()
    assert verdicts == (True, False)


@pytest.mark.asyncio
async def test_a_single_profile_reconnect_then_an_unrelated_save_keeps_the_resolved_username(
        connectable, tmp_path):
    """Single profile: the first resolution rewrites DISCORD_ALLOWED_USERS to ``111``, so a second
    connect() snapshots only numeric IDs. The configured ``alice`` must still map after a later save."""
    _write(tmp_path, {"allow_from": ["alice"]}, 0)
    adapter = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    try:
        assert await adapter.connect() is True
        assert adapter._is_allowed_user("111") is True
        assert await adapter.connect() is True
        _write(tmp_path, {"allow_from": ["alice"], "allowed_channels": ["555"]}, 1)  # an unrelated edit
        verdict = adapter._is_allowed_user("111")
    finally:
        await adapter.disconnect()
    assert verdict is True


@pytest.mark.asyncio
async def test_an_operators_own_list_still_wins_over_the_yaml_on_a_reconnect(connectable, tmp_path, monkeypatch):
    """Only the bridge's own value is swapped for the YAML names on connect(): an operator's
    DISCORD_ALLOWED_USERS keeps beating config.yaml (#109032), on every connection."""
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "333")
    _write(tmp_path, {"allow_from": ["alice"]}, 0)
    adapter = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    try:
        assert await adapter.connect() is True
        assert await adapter.connect() is True
        verdicts = (adapter._is_allowed_user("333"), adapter._is_allowed_user("111"))
    finally:
        await adapter.disconnect()
    assert verdicts == (True, False)


@pytest.mark.asyncio
async def test_a_multiplexed_profiles_own_env_list_wins_over_its_yaml(connectable, tmp_path, monkeypatch):
    """Under multiplex the process env is the launch profile's: its bridge marker says nothing about a
    served profile, whose own .env list must keep beating that profile's config.yaml."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.platforms._shared import bridged_env_digest
    from gateway.run import _profile_runtime_scope

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "alice")  # the launch profile's bridged YAML value
    monkeypatch.setenv("HERMES_BRIDGED_DISCORD_ALLOWED_USERS", bridged_env_digest("alice"))
    home = tmp_path / "b"
    home.mkdir()
    _write(home, {"allow_from": ["alice"]}, 0)
    (home / ".env").write_text("DISCORD_ALLOWED_USERS=333\n", encoding="utf-8")
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(home):
            adapter = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
            assert await adapter.connect() is True
            verdicts = (adapter._is_allowed_user("333"), adapter._is_allowed_user("111"))
            await adapter.disconnect()
    finally:
        reset_multiplex_context(token)
    assert verdicts == (True, False)
