"""Regression guard: DISCORD_ALLOWED_ROLES must be guild-scoped, not global.

Prior to this fix, ``_is_allowed_user`` iterated ``self._client.guilds`` and
returned True if the user held any allowed role in ANY mutual guild. This
allowed a cross-guild DM bypass:

1. Bot is in both a large public server A and a private trusted server B.
2. User has role ``R`` in public server A. ``DISCORD_ALLOWED_ROLES`` is
   configured with ``R`` intending it to authorize server B members.
3. User DMs the bot. The role check scans every mutual guild, finds ``R``
   in public server A, and authorizes the DM.

The fix scopes role checks to the originating guild and disables role-based
auth on DMs unless ``DISCORD_DM_ROLE_AUTH_GUILD`` explicitly opts into a
single trusted guild.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.platforms.discord import DiscordAdapter


def _make_adapter(allowed_users=None, allowed_roles=None, guilds=None):
    """Build a minimal DiscordAdapter without running __init__."""
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set(allowed_roles or [])

    client = MagicMock()
    client.guilds = guilds or []
    client.get_guild = lambda gid: next(
        (g for g in (guilds or []) if getattr(g, "id", None) == gid),
        None,
    )
    adapter._client = client
    return adapter


def _role(role_id):
    return SimpleNamespace(id=role_id)


def _guild_with_member(guild_id, member_id, role_ids):
    """Build a fake guild that holds one member with the given roles."""
    member = SimpleNamespace(
        id=member_id,
        roles=[_role(rid) for rid in role_ids],
        guild=None,  # filled below
    )
    guild = SimpleNamespace(
        id=guild_id,
        get_member=lambda uid: member if uid == member_id else None,
    )
    member.guild = guild
    return guild, member


# ---------------------------------------------------------------------------
# Cross-guild DM bypass — MUST be rejected
# ---------------------------------------------------------------------------


def test_dm_rejects_role_held_in_other_guild(monkeypatch):
    """A user with an allowed role in a DIFFERENT guild must NOT pass a DM.

    Regression guard for the cross-guild DM bypass in the initial
    DISCORD_ALLOWED_ROLES implementation.
    """
    monkeypatch.delenv("DISCORD_DM_ROLE_AUTH_GUILD", raising=False)

    public_guild, _ = _guild_with_member(
        guild_id=111111,
        member_id=42,
        role_ids=[5555],  # allowed role, but in the wrong guild
    )
    trusted_guild = SimpleNamespace(id=222222, get_member=lambda uid: None)

    adapter = _make_adapter(
        allowed_roles=[5555],
        guilds=[public_guild, trusted_guild],
    )

    # DM from user 42: role check must NOT scan other guilds.
    assert (
        adapter._is_allowed_user("42", author=None, guild=None, is_dm=True)
        is False
    )


def test_dm_role_auth_requires_explicit_guild_optin(monkeypatch):
    """With DISCORD_DM_ROLE_AUTH_GUILD set, only that specific guild counts.

    The user has the role in the opted-in guild — allowed.
    """
    trusted_guild, _ = _guild_with_member(
        guild_id=222222,
        member_id=42,
        role_ids=[5555],
    )
    other_guild = SimpleNamespace(id=333333, get_member=lambda uid: None)

    adapter = _make_adapter(
        allowed_roles=[5555],
        guilds=[other_guild, trusted_guild],
    )
    monkeypatch.setenv("DISCORD_DM_ROLE_AUTH_GUILD", "222222")

    assert (
        adapter._is_allowed_user("42", author=None, guild=None, is_dm=True)
        is True
    )


def test_dm_role_auth_optin_rejects_when_not_member(monkeypatch):
    """DISCORD_DM_ROLE_AUTH_GUILD set but user isn't a member → reject."""
    trusted_guild = SimpleNamespace(
        id=222222,
        get_member=lambda uid: None,  # user not in trusted guild
    )
    public_guild, _ = _guild_with_member(
        guild_id=111111,
        member_id=42,
        role_ids=[5555],
    )
    adapter = _make_adapter(
        allowed_roles=[5555],
        guilds=[public_guild, trusted_guild],
    )
    monkeypatch.setenv("DISCORD_DM_ROLE_AUTH_GUILD", "222222")

    assert (
        adapter._is_allowed_user("42", author=None, guild=None, is_dm=True)
        is False
    )


# ---------------------------------------------------------------------------
# Guild messages — role check must be scoped to THIS guild only
# ---------------------------------------------------------------------------


def test_guild_message_role_check_scoped_to_originating_guild(monkeypatch):
    """A user with the role in a DIFFERENT guild than the message origin
    must NOT be authorized, even when both guilds are mutual.
    """
    monkeypatch.delenv("DISCORD_DM_ROLE_AUTH_GUILD", raising=False)

    public_guild, _ = _guild_with_member(
        guild_id=111111,
        member_id=42,
        role_ids=[5555],  # allowed role in public guild only
    )
    # Message arrives in trusted_guild where user 42 has NO role
    trusted_guild = SimpleNamespace(id=222222, get_member=lambda uid: None)

    adapter = _make_adapter(
        allowed_roles=[5555],
        guilds=[public_guild, trusted_guild],
    )

    # No author object passed → falls through to guild.get_member path
    assert (
        adapter._is_allowed_user(
            "42", author=None, guild=trusted_guild, is_dm=False
        )
        is False
    )


def test_guild_message_role_check_allows_when_role_in_same_guild(monkeypatch):
    """Positive path: user has the role IN the message's guild → allowed."""
    monkeypatch.delenv("DISCORD_DM_ROLE_AUTH_GUILD", raising=False)

    trusted_guild, _ = _guild_with_member(
        guild_id=222222,
        member_id=42,
        role_ids=[5555],
    )
    adapter = _make_adapter(
        allowed_roles=[5555],
        guilds=[trusted_guild],
    )

    assert (
        adapter._is_allowed_user(
            "42", author=None, guild=trusted_guild, is_dm=False
        )
        is True
    )


def test_guild_message_rejects_author_roles_from_different_guild(monkeypatch):
    """If an author Member object comes from a different guild than the
    message, the cached .roles on it must NOT be trusted — rely on the
    current guild's Member lookup instead.
    """
    monkeypatch.delenv("DISCORD_DM_ROLE_AUTH_GUILD", raising=False)

    # Author is a Member of a DIFFERENT guild with the allowed role
    foreign_guild = SimpleNamespace(id=999, get_member=lambda uid: None)
    foreign_author = SimpleNamespace(
        id=42,
        roles=[_role(5555)],
        guild=foreign_guild,
    )
    # Message arrives in this_guild where user 42 has NO role
    this_guild = SimpleNamespace(id=222222, get_member=lambda uid: None)

    adapter = _make_adapter(
        allowed_roles=[5555],
        guilds=[foreign_guild, this_guild],
    )

    assert (
        adapter._is_allowed_user(
            "42", author=foreign_author, guild=this_guild, is_dm=False
        )
        is False
    )


# ---------------------------------------------------------------------------
# Backwards-compatibility — user-ID allowlist still works in both contexts
# ---------------------------------------------------------------------------


def test_user_id_allowlist_works_in_dm():
    adapter = _make_adapter(allowed_users=["42"])
    assert (
        adapter._is_allowed_user("42", author=None, guild=None, is_dm=True)
        is True
    )


def test_user_id_allowlist_works_in_guild():
    adapter = _make_adapter(allowed_users=["42"])
    some_guild = SimpleNamespace(id=111, get_member=lambda uid: None)
    assert (
        adapter._is_allowed_user(
            "42", author=None, guild=some_guild, is_dm=False
        )
        is True
    )


def test_empty_allowlists_allow_everyone():
    adapter = _make_adapter()
    assert (
        adapter._is_allowed_user("42", author=None, guild=None, is_dm=True)
        is True
    )
