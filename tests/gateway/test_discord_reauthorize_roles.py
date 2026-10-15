"""Durable role admission must agree with the real, profile-scoped ingress loader."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_yaml as yaml
from agent.secret_scope import reset_multiplex_context, set_multiplex_context
from gateway.config import Platform, load_gateway_config
from gateway.run import _profile_runtime_scope
from plugins.platforms.discord.adapter import DiscordAdapter


def _role_config(location, roles):
    policy = {"allowed_roles": roles}
    if location == "top":
        return {"discord": policy}
    if location == "root-extra":
        return {"discord": {"extra": policy}}
    if location == "gateway":
        return {"gateway": {"discord": policy}}
    if location == "gateway-platforms":
        return {"gateway": {"platforms": {"discord": policy}}}
    if location == "conflict":
        return {"discord": {"allowed_roles": [3]}, **_role_config("extra", roles)}
    return {"platforms": {"discord": {"extra": policy} if location == "extra" else policy}}


def _write_config(home, config):
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _source(chat_type="group", scope_id="111"):
    return SimpleNamespace(user_id="42", chat_type=chat_type, scope_id=scope_id)


def _ingress():
    ingress = object.__new__(DiscordAdapter)
    ingress.config = load_gateway_config().platforms[Platform.DISCORD]
    return ingress


@pytest.fixture
def scoped_discord(tmp_path, monkeypatch):
    home = tmp_path / "transport"
    home.mkdir()
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    # Scoped misses must borrow neither process env nor the launch profile's YAML.
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "9999")
    _write_config(launch, _role_config("top", [9999]))
    _write_config(home, {})
    adapter = object.__new__(DiscordAdapter)
    guild = MagicMock(id=111)
    member = SimpleNamespace(id=42, guild=guild, roles=[SimpleNamespace(id=2)])
    guild.fetch_member = AsyncMock(return_value=member)
    adapter._client = MagicMock()
    adapter._client.get_guild.return_value = guild
    adapter._allowed_role_ids = {9999}
    adapter._gate_env_snapshot = {"DISCORD_ALLOWED_ROLES": "9999"}
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(home):
            yield adapter, home, guild, member
    finally:
        reset_multiplex_context(token)


_POLICY_CASES = [
    pytest.param({"discord": {"allowed_roles": [3]}, **_role_config("extra", [2])}, [2], id="top-vs-extra"),
    pytest.param({"discord": {"allowed_roles": []}, **_role_config("extra", [2])}, [2], id="empty-top"),
    pytest.param({"discord": {"allowed_roles": [3]}, **_role_config("extra", [])}, [], id="empty-extra"),
    pytest.param({"discord": {"allowed_roles": [2], "extra": {"allowed_roles": [3]}}},
                 [2], id="block-own-extra"),
    *(pytest.param(_role_config(location, [2]), [2], id=location)
      for location in ("flat", "extra", "top", "gateway-platforms", "gateway", "root-extra")),
    *(pytest.param(_role_config(location, []), [], id=f"{location}-deny")
      for location in ("flat", "extra", "top")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("config,expected_roles", _POLICY_CASES)
@pytest.mark.parametrize("role", [2, 3])
async def test_reauthorize_matches_ingress(scoped_discord, config, expected_roles, role, caplog):
    adapter, home, _, member = scoped_discord
    _write_config(home, config)
    member.roles = [SimpleNamespace(id=role)]
    ingress = _ingress()
    assert ingress._get_allowed_roles() == set(expected_roles)
    caplog.clear()

    assert await adapter.reauthorize_native_roles(_source()) is (role in ingress._get_allowed_roles())
    assert not [r for r in caplog.records if "took precedence" in r.getMessage()]


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["flat", "extra", "top", "conflict"])
@pytest.mark.parametrize("updated_roles", [[], [2, 3]], ids=["revoked", "expanded"])
async def test_reauthorize_denies_policy_edits_during_fetch(scoped_discord, location, updated_roles):
    adapter, home, guild, member = scoped_discord
    _write_config(home, _role_config(location, [2]))
    assert 2 in _ingress()._get_allowed_roles()

    async def fetch_member(user_id):
        _write_config(home, _role_config(location, updated_roles))
        return member

    guild.fetch_member.side_effect = fetch_member
    assert await adapter.reauthorize_native_roles(_source()) is False
    guild.fetch_member.assert_awaited_once_with(42)


@pytest.mark.asyncio
@pytest.mark.parametrize("env_roles", ["2", "3"])
async def test_scoped_env_wins(scoped_discord, env_roles):
    adapter, home, _, _ = scoped_discord
    _write_config(home, {"discord": {"allowed_roles": [3]}, **_role_config("extra", [2])})
    (home / ".env").write_text(f"DISCORD_ALLOWED_ROLES={env_roles}\n", encoding="utf-8")
    with _profile_runtime_scope(home):
        ingress = _ingress()
        assert ingress._get_allowed_roles() == {int(env_roles)}
        assert await adapter.reauthorize_native_roles(_source()) is (2 in ingress._get_allowed_roles())


@pytest.mark.asyncio
@pytest.mark.parametrize("dm_guild,expected", [(None, False), (111, True), (222, False)])
async def test_reauthorize_dm_requires_the_opted_in_guild(scoped_discord, dm_guild, expected):
    adapter, home, _, _ = scoped_discord
    config = _role_config("flat", [2])
    config["discord"] = {"dm_role_auth_guild": dm_guild}
    _write_config(home, config)
    assert 2 in _ingress()._get_allowed_roles()
    assert await adapter.reauthorize_native_roles(_source("dm", None)) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex", [True, False])
async def test_reauthorize_does_not_bridge_env(scoped_discord, monkeypatch, multiplex):
    adapter, home, _, _ = scoped_discord
    _write_config(home, {"discord": {"require_mention": False, "allowed_roles": [2]},
                         "gateway": {"allow_all_users": True}, "require_mention": False})
    ingress = _ingress()
    expected = 2 in ingress._get_allowed_roles()
    for key in ("DISCORD_ALLOWED_ROLES", "DISCORD_REQUIRE_MENTION", "GATEWAY_ALLOW_ALL_USERS",
                "TELEGRAM_REQUIRE_MENTION"):
        monkeypatch.delenv(key, raising=False)
    token = set_multiplex_context(multiplex)
    try:
        before = dict(os.environ)
        assert await adapter.reauthorize_native_roles(_source()) is expected
        assert dict(os.environ) == before
    finally:
        reset_multiplex_context(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["top", "root-extra", "gateway-platforms", "extra"])
@pytest.mark.parametrize("role", [2, 3])
async def test_managed_pins_match_ingress(scoped_discord, tmp_path, monkeypatch, location, role):
    adapter, home, _, member = scoped_discord
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _write_config(managed, _role_config(location, [3]))
    _write_config(home, {"discord": {"allow_bots": True}, **_role_config("extra", [2])})
    member.roles = [SimpleNamespace(id=role)]
    with _profile_runtime_scope(home):
        ingress = _ingress()
        assert await adapter.reauthorize_native_roles(_source()) is (role in ingress._get_allowed_roles())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["client", "member", "guild"])
async def test_reauthorize_fails_closed_after_discord_io(scoped_discord, failure):
    adapter, home, guild, member = scoped_discord
    _write_config(home, _role_config("top", [2]))
    assert 2 in _ingress()._get_allowed_roles()

    async def fetch_member(user_id):
        if failure == "client":
            adapter._client = MagicMock()
        elif failure == "member":
            member.id = 99
        elif failure == "guild":
            member.guild = SimpleNamespace(id=222)
        return member

    guild.fetch_member.side_effect = fetch_member
    assert await adapter.reauthorize_native_roles(_source()) is False
    guild.fetch_member.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_reauthorize_reloads_scoped_env_after_fetch(scoped_discord):
    adapter, home, guild, member = scoped_discord
    _write_config(home, _role_config("extra", [3]))
    (home / ".env").write_text("DISCORD_ALLOWED_ROLES=2\n", encoding="utf-8")

    async def fetch_member(user_id):
        (home / ".env").write_text("DISCORD_ALLOWED_ROLES=3\n", encoding="utf-8")
        return member

    guild.fetch_member.side_effect = fetch_member
    with _profile_runtime_scope(home):
        assert 2 in _ingress()._get_allowed_roles()
        assert await adapter.reauthorize_native_roles(_source()) is False


@pytest.mark.asyncio
async def test_reauthorize_never_borrows_launch_profile_roles(scoped_discord):
    adapter, home, guild, member = scoped_discord
    _write_config(home, _role_config("flat", []))
    member.roles = [SimpleNamespace(id=9999)]
    assert 9999 not in _ingress()._get_allowed_roles()
    assert await adapter.reauthorize_native_roles(_source()) is False
    guild.fetch_member.assert_not_awaited()
