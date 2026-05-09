"""Regression tests for #22334.

OpenClaw stored Discord allowlists as ``allowFrom: ["<id>", "*"]`` where
``"*"`` meant "any user".  ``hermes claw migrate`` preserves the literal
list as ``DISCORD_ALLOWED_USERS=<id>,*``, but
``DiscordAdapter._is_allowed_user`` did set-membership against the literal
``"*"`` instead of treating it as an open-mode wildcard, silently rejecting
every user except ``<id>`` after migration.

The fix mirrors the convention already used by
``DISCORD_ALLOWED_CHANNELS``, ``DISCORD_IGNORED_CHANNELS``, and
``SIGNAL_ALLOWED_USERS``: ``"*"`` in the allowlist short-circuits to True
for any user.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from gateway.platforms.discord import DiscordAdapter


def _make_adapter(allowed_users=None, allowed_roles=None):
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set(allowed_roles or [])
    adapter._client = MagicMock()
    return adapter


class TestDiscordAllowedUsersWildcard:
    def test_wildcard_alone_allows_any_user(self):
        adapter = _make_adapter(allowed_users={"*"})
        assert adapter._is_allowed_user("42") is True
        assert adapter._is_allowed_user("999999999999999999") is True

    def test_wildcard_mixed_with_id_allows_other_users(self):
        """The bug from #22334: ``{"<id>", "*"}`` must allow any user, not
        only ``<id>``."""
        adapter = _make_adapter(allowed_users={"123456789012345678", "*"})
        assert adapter._is_allowed_user("42") is True
        assert adapter._is_allowed_user("123456789012345678") is True

    def test_wildcard_in_dm_allows_any_user(self):
        """DMs (``is_dm=True``, no guild context) must also honor the user
        wildcard — DMs have no role-fallback path."""
        adapter = _make_adapter(allowed_users={"*"})
        assert adapter._is_allowed_user("42", is_dm=True, guild=None) is True

    def test_no_wildcard_keeps_strict_membership(self):
        """No wildcard → must still reject non-listed users."""
        adapter = _make_adapter(allowed_users={"123456789012345678"})
        assert adapter._is_allowed_user("42") is False
        assert adapter._is_allowed_user("123456789012345678") is True

    def test_empty_allowlist_still_open(self):
        """Empty allowlist preserves the existing 'no restriction' default;
        the wildcard short-circuit must not regress that path."""
        adapter = _make_adapter(allowed_users=set(), allowed_roles=set())
        assert adapter._is_allowed_user("42") is True
