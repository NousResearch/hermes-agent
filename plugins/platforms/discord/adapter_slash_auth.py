"""Slash-command authorization verdict for the Discord adapter: the on_message gates, side-effect free."""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

try:
    import discord
except ImportError:  # the adapter does not start without discord.py
    discord = None


class DiscordSlashAuthMixin:
    _get_parent_channel_id: Callable[[Any], Optional[str]]
    _discord_channel_keys_from_channel: Callable[..., set]
    _get_allowed_channels: Callable[[], set]
    _get_ignored_channels: Callable[[], set]
    _is_allowed_user: Callable[..., bool]

    # ── Slash command authorization ─────────────────────────────────────
    # ``_check_slash_authorization`` mirrors the on_message gates one-for-one. No allowlist =>
    # fail closed unless allow-all; DISCORD_ALLOWED_CHANNELS alone authorizes per validated channel.

    def _evaluate_slash_authorization(
        self, interaction: "discord.Interaction",
    ) -> tuple[bool, Optional[str]]:
        """Evaluate slash authorization without responding; returns ``(allowed, reason)``.
        Shared with side-effect-free callers (``/skill`` autocomplete returns [] per keystroke).
        Fail closed on malformed payloads: with an allowlist, a missing channel id/user REJECTS.
        """
        chan_obj = getattr(interaction, "channel", None)
        in_dm = isinstance(chan_obj, discord.DMChannel) if chan_obj is not None else False
        channel_ids: set = set()
        channel_keys: set = set()
        # Channel scope mirrors on_message; DMs use on_message's DM lockdown path instead.
        if not in_dm:
            chan_id_raw = getattr(interaction, "channel_id", None) or getattr(chan_obj, "id", None)
            if chan_id_raw is not None:
                channel_ids.add(str(chan_id_raw))
                # Threads: also test the parent channel, as on_message does.
                if isinstance(chan_obj, discord.Thread):
                    parent_id = self._get_parent_channel_id(chan_obj)
                    if parent_id:
                        channel_ids.add(str(parent_id))
            # Name-form keys (ID, name, #name, parent) so name-based lists work for slash too.
            channel_keys = self._discord_channel_keys_from_channel(
                chan_obj,
                self._get_parent_channel_id(chan_obj)
                if isinstance(chan_obj, discord.Thread)
                else None,
            )
            allowed = self._get_allowed_channels()
            if allowed:
                if "*" not in allowed:
                    if not channel_ids:
                        # Channel policy configured but no resolvable channel id: fail closed.
                        return (
                            False, "channel id missing with DISCORD_ALLOWED_CHANNELS configured",
                        )
                    if not (channel_keys & allowed):
                        return (False, "channel not in DISCORD_ALLOWED_CHANNELS")
            # Ignored beats allowed, including via a thread's parent.
            ignored = self._get_ignored_channels()
            if ignored and channel_ids:
                if "*" in ignored or (channel_keys & ignored):
                    return (False, "channel in DISCORD_IGNORED_CHANNELS")
        # ── User / role allowlist (mirrors on_message line 681) ──
        user = getattr(interaction, "user", None)
        allowed_users = getattr(self, "_allowed_user_ids", set()) or set()
        allowed_roles = getattr(self, "_allowed_role_ids", set()) or set()
        if user is None or getattr(user, "id", None) is None:
            # No identifiable user: fail closed even with allow-all; downstream handlers need interaction.user.id.
            if allowed_users or allowed_roles:
                return (False, "missing interaction.user with allowlist configured")
            return (False, "missing interaction.user")
        user_id = str(user.id)
        # guild + is_dm scope the role check so the cross-guild DM bypass can't land via slash.
        # See #12136.
        interaction_guild = getattr(interaction, "guild", None)
        if not self._is_allowed_user(
            user_id, author=user, guild=interaction_guild, is_dm=in_dm,
            channel_ids=channel_keys if not in_dm else None,
        ):
            return (False, "user not in DISCORD_ALLOWED_USERS / DISCORD_ALLOWED_ROLES")
        return (True, None)

    def _slash_role_grant(self, interaction: "discord.Interaction") -> bool:
        """``role_authorized`` for an event built from ``interaction``, with the message path's
        meaning (_discord_message_admission): a role allowlist exists and THIS actor passes the gate.
        Re-evaluated for the interaction instead of trusted from call order, so a builder can never
        stamp the grant on an actor the slash gate would refuse."""
        return bool(getattr(self, "_allowed_role_ids", set())) and self._evaluate_slash_authorization(
            interaction)[0]
