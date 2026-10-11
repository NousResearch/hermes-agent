"""Guild-scoped Discord user and role access policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from plugins.platforms.discord.adapter import DiscordAdapter


class DiscordAccessPolicyMixin:
    def _is_allowed_user(
        self: DiscordAdapter,
        user_id: str,
        author=None,
        *,
        guild=None,
        is_dm: bool = False,
        channel_ids: Optional[set[str]] = None,
    ) -> bool:
        """Allow via DISCORD_ALLOWED_USERS/ROLES (OR); with no allowlists, validated channel
        context may pass on DISCORD_ALLOWED_CHANNELS (never voice). Role checks are guild-scoped:
        DMs use user IDs only unless ``discord.dm_role_auth_guild`` names one guild (no escalation).
        """
        from plugins.platforms.discord.adapter import _read_dm_role_auth_guild

        # getattr fallbacks: test fixtures build the adapter via object.__new__ and skip __init__.
        allowed_users = getattr(self, "_allowed_user_ids", set())
        allowed_roles = getattr(self, "_allowed_role_ids", set())
        has_users = bool(allowed_users)
        has_roles = bool(allowed_roles)
        # Pairing is a first-class grant in the gateway auth union; honor it here too.
        if self._is_pairing_approved_user(user_id):
            return True
        if not has_users and not has_roles:
            if self._discord_allow_all_users():
                return True
            if self._gateway_allow_all_users():
                return True
            # Channel-scoped access needs validated channel context; not a user-wide bypass.
            # In shared channels, respond only when addressed — unless require_mention is disabled, in which
            # case respond to every message. A NIP-10 thread reply whose direct parent is one of our
            # messages is treated as addressed (parity with Signal/WhatsApp; fixes #75826 — e.g. Desktop
            # "/approve session" replies that never type @name). Explicit addressing is a text @mention OR a
            # signed recipient p-tag (#92781). DMs always dispatch.
            if (
                not is_dm
                and channel_ids is not None
                and self._discord_channel_ids_allowed(channel_ids)
            ):
                return True
            return False
        # "*" is the open-mode wildcard (mirrors other DISCORD_* lists; ``claw migrate`` emits it).
        if has_users and ("*" in allowed_users or user_id in allowed_users):
            return True
        if not has_roles:
            return False
        # DM path: roles need explicit opt-in via ``discord.dm_role_auth_guild`` (else cross-guild leakage).
        if is_dm or guild is None:
            dm_guild_id = _read_dm_role_auth_guild()
            if dm_guild_id is None:
                return False
            if self._client is None:
                return False
            dm_guild = self._client.get_guild(dm_guild_id)
            if dm_guild is None:
                return False
            return self._guild_member_has_role(dm_guild, user_id, allowed_roles)
        direct_roles = getattr(author, "roles", None) if author is not None else None
        author_guild = getattr(author, "guild", None)
        if (
            direct_roles is not None
            and author_guild is not None
            and author_guild.id == guild.id
        ):
            return any(getattr(r, "id", None) in allowed_roles for r in direct_roles)
        if direct_roles and author_guild is None:
            if any(getattr(r, "id", None) in allowed_roles for r in direct_roles):
                return True
        return self._guild_member_has_role(guild, user_id, allowed_roles)

    @staticmethod
    def _guild_member_has_role(guild, user_id: str, allowed_roles: set) -> bool:
        """Look ``user_id`` up as a member of ``guild`` only and test its roles."""
        try:
            uid_int = int(user_id)
        except (TypeError, ValueError):
            return False
        m = guild.get_member(uid_int)
        if m is None:
            return False
        m_roles = getattr(m, "roles", None) or []
        return any(getattr(r, "id", None) in allowed_roles for r in m_roles)
