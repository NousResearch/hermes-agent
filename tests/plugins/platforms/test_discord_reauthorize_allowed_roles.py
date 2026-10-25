"""Durable Discord re-authorization resolves ``allowed_roles`` like ingress, from current config (#139148).

Ingress reads the gate through ``DiscordAdapter._gate_raw``: scoped env first, then the Discord
``PlatformConfig.extra`` that ``load_gateway_config()`` builds from every YAML spelling. Recovery
applies that same resolution to the config as it stands now. The loader's YAML->env copy of
``allowed_roles`` (first writer wins) is part of that resolution: outside multiplex a role removed
from YAML stays granted until restart, exactly as for ingress; a multiplexed profile skips the copy.
"""

import asyncio
import contextlib
import importlib
import os
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
    # The default-profile YAML bridge seeds DISCORD_ALLOWED_ROLES first-writer-wins; undo it per test
    # (setenv first so teardown also removes a value the loader copies in during the test).
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "")
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES")
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


@contextlib.contextmanager
def _multiplexed_profile(home):
    """The scope the gateway loads and serves a multiplexed profile in: its own secret scope."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from gateway.run import _profile_runtime_scope
    token = set_multiplex_context(True)
    try:
        with _profile_runtime_scope(home):
            yield
    finally:
        reset_multiplex_context(token)


@pytest.mark.parametrize("multiplexed, granted_after_removal", [(True, False), (False, True)])
def test_role_removed_from_yaml_after_the_first_reauthorization(_home, multiplexed, granted_after_removal):
    """Both loads go through the real ``load_gateway_config()``. A multiplexed profile's load never
    copies YAML into process env, so the current YAML (not the connect-time extra) revokes at once.
    A single-profile load copies ``allowed_roles`` into ``DISCORD_ALLOWED_ROLES`` the first time and
    that env value keeps precedence, so the removal takes effect at restart (unchanged from main)."""
    connect_extra = {"allowed_roles": "2"}
    _write_config(_home, {"platforms": {"discord": {"enabled": True, "allowed_roles": ["2"]}}})
    with _multiplexed_profile(_home) if multiplexed else contextlib.nullcontext():
        assert _reauthorize(_adapter(2, connect_extra)) is True
        _write_config(_home, {"platforms": {"discord": {"enabled": True, "allowed_roles": ["9"]}}})
        assert _reauthorize(_adapter(2, connect_extra)) is granted_after_removal
    assert os.environ.get("DISCORD_ALLOWED_ROLES") == (None if multiplexed else "2")


def test_scoped_env_allowlist_wins_over_config(_home, monkeypatch):
    _write_config(_home, {"platforms": {"discord": {"enabled": True, "allowed_roles": ["2"]}}})
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "9")
    assert _reauthorize(_adapter(2)) is False
