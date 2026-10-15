"""Durable Discord re-authorization resolves ``allowed_roles`` like ingress, from current config (#139148).

Ingress reads the gate through ``DiscordAdapter._gate_raw``: scoped env first, then the Discord
``PlatformConfig.extra`` that ``load_gateway_config()`` builds from every YAML spelling. Recovery
must apply that same resolution to the config as it stands now, so a revocation still takes effect.
"""

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import hermes_yaml as yaml
import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

GUILD_ID = 222222
USER_ID = "42"
SOURCE = SimpleNamespace(chat_type="group", scope_id=str(GUILD_ID), user_id=USER_ID)


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The default-profile YAML bridge seeds DISCORD_ALLOWED_ROLES first-writer-wins; undo it per test.
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    if getattr(sys.modules.get("discord"), "HTTPException", None) is None:
        try:
            importlib.import_module("discord")
        except ImportError:
            stub = ModuleType("discord")
            stub.HTTPException = type("HTTPException", (Exception,), {})
            monkeypatch.setitem(sys.modules, "discord", stub)
    return tmp_path


def _write_config(home, config):
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _adapter(role_id, connect_extra=None):
    """An adapter whose guild member holds *role_id*; *connect_extra* is its connect-time extra."""
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, extra=dict(connect_extra or {}))
    adapter._gate_env_snapshot = None
    member = SimpleNamespace(id=int(USER_ID), roles=[SimpleNamespace(id=role_id)])

    async def fetch_member(_uid):
        return member

    guild = SimpleNamespace(id=GUILD_ID, fetch_member=fetch_member)
    member.guild = guild
    adapter._client = SimpleNamespace(get_guild=lambda gid: guild if gid == GUILD_ID else None)
    return adapter


def _reauthorize(adapter):
    return asyncio.run(adapter.reauthorize_native_roles(SOURCE))


def test_flat_platforms_allowed_roles_authorize(_home):
    _write_config(_home, {"platforms": {"discord": {"enabled": True, "allowed_roles": ["2"]}}})
    assert _reauthorize(_adapter(2)) is True


def test_platforms_extra_allowed_roles_authorize(_home):
    _write_config(_home, {"platforms": {"discord": {"enabled": True, "extra": {"allowed_roles": ["2"]}}}})
    assert _reauthorize(_adapter(2)) is True


@pytest.mark.parametrize("role_id, expected", [(2, True), (3, False)])
def test_platforms_extra_beats_top_level_block_like_ingress(_home, role_id, expected):
    _write_config(_home, {
        "discord": {"allowed_roles": ["3"]},
        "platforms": {"discord": {"enabled": True, "extra": {"allowed_roles": ["2"]}}},
    })
    assert _reauthorize(_adapter(role_id)) is expected


def test_legacy_top_level_allowed_roles_still_authorize(_home):
    _write_config(_home, {"discord": {"allowed_roles": ["2"]}, "platforms": {"discord": {"enabled": True}}})
    assert _reauthorize(_adapter(2)) is True


def test_revocation_in_current_config_beats_connect_time_extra(_home):
    _write_config(_home, {"platforms": {"discord": {"enabled": True, "allowed_roles": ["9"]}}})
    assert _reauthorize(_adapter(2, connect_extra={"allowed_roles": "2"})) is False


def test_scoped_env_allowlist_wins_over_config(_home, monkeypatch):
    _write_config(_home, {"platforms": {"discord": {"enabled": True, "allowed_roles": ["2"]}}})
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "9")
    assert _reauthorize(_adapter(2)) is False
