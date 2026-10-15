"""Regression guard: durable native re-authorization must resolve ``allowed_roles``
from the same per-profile config spellings ingress honors (#139148).

Ingress resolves the role gate through ``_gate_raw("allowed_roles", ...)``: scoped
env, then ``config.extra``, where ``PlatformConfig.from_dict`` promotes both the
flat ``platforms.discord.allowed_roles`` spelling and ``platforms.discord.extra.
allowed_roles`` (#10206). Durable re-authorization read only the scoped env and
the top-level ``discord.allowed_roles`` spelling, so a profile configuring roles
under ``platforms:`` admitted the user at ingress and then dropped every message
at re-authorization with ``permission_denied``.
"""

from types import SimpleNamespace

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter

GUILD_ID = 111111
ROLE_ID = 5555
USER_ID = 42


def _patch_config(monkeypatch, cfg):
    """Point every ``read_raw_config`` caller (policy(), _read_dm_role_auth_guild) at cfg."""
    import hermes_cli.config as _cfg_mod
    monkeypatch.setattr(_cfg_mod, "read_raw_config", lambda: cfg, raising=True)


def _make_adapter(guild):
    """Build a minimal DiscordAdapter without running __init__."""
    adapter = object.__new__(DiscordAdapter)
    adapter._client = SimpleNamespace(get_guild=lambda gid: guild if gid == guild.id else None)
    return adapter


def _make_guild(member_role_ids):
    member = SimpleNamespace(
        id=USER_ID,
        roles=[SimpleNamespace(id=rid) for rid in member_role_ids],
        guild=None,
    )
    guild = SimpleNamespace(id=GUILD_ID, fetch_member=None)
    guild.fetch_member = _async_return(member)
    member.guild = guild
    return guild


def _async_return(value):
    async def _fetch(_uid):
        return value
    return _fetch


def _source():
    return SimpleNamespace(chat_type="group", scope_id=GUILD_ID, user_id=str(USER_ID))


@pytest.mark.asyncio
async def test_platform_block_roles_authorize_after_refusal_of_top_level(monkeypatch):
    """Roles configured ONLY under ``platforms.discord.allowed_roles`` (the spelling
    ingress admits through) must also pass durable re-authorization."""
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    _patch_config(monkeypatch, {
        "platforms": {"discord": {"enabled": True, "allowed_roles": [str(ROLE_ID)]}},
    })
    import contextlib
    import gateway.run as _run_mod
    monkeypatch.setattr(_run_mod, "_profile_runtime_scope", contextlib.nullcontext)

    adapter = _make_adapter(_make_guild([ROLE_ID]))
    assert await adapter.reauthorize_native_roles(_source()) is True


@pytest.mark.asyncio
async def test_platform_extra_roles_authorize(monkeypatch):
    """The explicit ``platforms.discord.extra.allowed_roles`` spelling works too."""
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    _patch_config(monkeypatch, {
        "platforms": {"discord": {"enabled": True, "extra": {"allowed_roles": [str(ROLE_ID)]}}},
    })
    import contextlib
    import gateway.run as _run_mod
    monkeypatch.setattr(_run_mod, "_profile_runtime_scope", contextlib.nullcontext)

    adapter = _make_adapter(_make_guild([ROLE_ID]))
    assert await adapter.reauthorize_native_roles(_source()) is True


@pytest.mark.asyncio
async def test_top_level_roles_still_authorize(monkeypatch):
    """Backwards compatibility: the pre-existing top-level spelling keeps working."""
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    _patch_config(monkeypatch, {"discord": {"allowed_roles": [str(ROLE_ID)]}})
    import contextlib
    import gateway.run as _run_mod
    monkeypatch.setattr(_run_mod, "_profile_runtime_scope", contextlib.nullcontext)

    adapter = _make_adapter(_make_guild([ROLE_ID]))
    assert await adapter.reauthorize_native_roles(_source()) is True


@pytest.mark.asyncio
async def test_empty_top_level_list_does_not_fall_through_to_platform_block(monkeypatch):
    """Revocation keeps working: an explicitly empty top-level list must not be
    rescued by a stale ``platforms.discord`` spelling (fail-closed)."""
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    _patch_config(monkeypatch, {
        "discord": {"allowed_roles": []},
        "platforms": {"discord": {"allowed_roles": [str(ROLE_ID)]}},
    })

    adapter = _make_adapter(_make_guild([ROLE_ID]))
    assert await adapter.reauthorize_native_roles(_source()) is False


@pytest.mark.asyncio
async def test_explicit_empty_extra_roles_beats_flat_spelling(monkeypatch):
    """Match ``PlatformConfig.from_dict`` merge semantics: an explicit (empty)
    ``extra.allowed_roles`` wins over the flat spelling, closing role auth."""
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    _patch_config(monkeypatch, {
        "platforms": {"discord": {"allowed_roles": [str(ROLE_ID)],
                                  "extra": {"allowed_roles": []}}},
    })

    adapter = _make_adapter(_make_guild([ROLE_ID]))
    assert await adapter.reauthorize_native_roles(_source()) is False


@pytest.mark.asyncio
async def test_member_without_the_role_is_denied(monkeypatch):
    """Resolving the platform block must not weaken the membership check itself."""
    monkeypatch.delenv("DISCORD_ALLOWED_ROLES", raising=False)
    _patch_config(monkeypatch, {
        "platforms": {"discord": {"allowed_roles": [str(ROLE_ID)]}},
    })

    adapter = _make_adapter(_make_guild([999999]))
    assert await adapter.reauthorize_native_roles(_source()) is False
