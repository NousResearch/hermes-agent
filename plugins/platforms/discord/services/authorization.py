"""Discord authorization and access-control service."""

from __future__ import annotations

import asyncio
from typing import Optional, Tuple

from .. import adapter as _adapter

discord = _adapter.discord
logger = _adapter.logger
Platform = _adapter.Platform


class AuthorizationMixin:
    """Authorization policy methods used by Discord ingress and interactions."""

    def _discord_channel_ids_allowed(self, channel_ids: set[str]) -> bool:
        """True when *channel_ids* intersect ``DISCORD_ALLOWED_CHANNELS``."""
        if not channel_ids:
            return False
        allowed = self._get_allowed_channels()
        if not allowed:
            return False
        if "*" in allowed:
            return True
        return bool(channel_ids & allowed)

    def _is_pairing_approved_user(self, user_id: str) -> bool:
        """True when the Discord user has an explicit Hermes pairing grant."""
        user_id = str(user_id or "").strip()
        if not user_id:
            return False
        try:
            from gateway.pairing import PairingStore
            return bool(PairingStore().is_approved("discord", user_id))
        except Exception:
            return False

    def _is_allowed_user(
        self, user_id: str, author=None, *, guild=None, is_dm: bool = False,
        channel_ids: Optional[set[str]] = None,
    ) -> bool:
        """Allow via DISCORD_ALLOWED_USERS/ROLES (OR); with no allowlists, validated channel
        context may pass on DISCORD_ALLOWED_CHANNELS (never voice). Role checks are guild-scoped:
        DMs use user IDs only unless ``discord.dm_role_auth_guild`` names one guild (no escalation).
        """
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
            dm_guild_id = _adapter._read_dm_role_auth_guild()
            if dm_guild_id is None:
                return False
            if self._client is None:
                return False
            dm_guild = self._client.get_guild(dm_guild_id)
            if dm_guild is None:
                return False
            return self._guild_member_has_role(dm_guild, user_id, allowed_roles)
        # Guild path: scoped to THIS guild. 1) Prefer the passed Member (correct guild by construction).
        direct_roles = getattr(author, "roles", None) if author is not None else None
        author_guild = getattr(author, "guild", None)
        if direct_roles and (author_guild is None or author_guild.id == guild.id):
            if any(getattr(r, "id", None) in allowed_roles for r in direct_roles):
                return True
        # 2) Fallback: resolve Member in this guild only — NEVER scan other mutual guilds.
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

    def _warn_if_fail_closed_default(self) -> None:
        """Log once when Discord is rejecting traffic with no allowlist set."""
        if getattr(self, "_warned_fail_closed_default", False):
            return
        allowed_users = getattr(self, "_allowed_user_ids", set()) or set()
        allowed_roles = getattr(self, "_allowed_role_ids", set()) or set()
        if allowed_users or allowed_roles:
            return
        if self._get_allowed_channels():
            return
        if self._discord_allow_all_users():
            return
        if self._gateway_allow_all_users():
            return
        self._warned_fail_closed_default = True
        logger.warning(
            "[%s] Discord messages are being denied because no allowlist is configured. "
            "Set DISCORD_ALLOWED_USERS, DISCORD_ALLOWED_ROLES, or "
            "DISCORD_ALLOWED_CHANNELS, or set DISCORD_ALLOW_ALL_USERS=true for open access.",
            self.name,
        )

    # ── Slash command authorization ─────────────────────────────────────
    # ``_check_slash_authorization`` mirrors the on_message gates one-for-one. No allowlist =>
    # fail closed unless allow-all; DISCORD_ALLOWED_CHANNELS alone authorizes per validated channel.

    def _evaluate_slash_authorization(
        self, interaction: "discord.Interaction",
    ) -> Tuple[bool, Optional[str]]:
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

    async def _check_slash_authorization(
        self, interaction: "discord.Interaction", command_text: str,
    ) -> bool:
        """Mirror on_message's gates onto a slash invocation.
        Returns False only *after* sending the ephemeral rejection, so the caller just stops."""
        allowed, reason = self._evaluate_slash_authorization(interaction)
        if allowed:
            return True
        return await self._reject_slash(interaction, command_text, reason=reason or "unauthorized")

    async def _reject_slash(
        self, interaction: "discord.Interaction", command_text: str, *, reason: str,
    ) -> bool:
        """Send ephemeral reject + log + schedule admin alert; returns False.
        Tolerates a missing ``interaction.user`` (fail-closed branch routes malformed payloads here)."""
        user = getattr(interaction, "user", None)
        if user is not None:
            user_id = str(getattr(user, "id", "?"))
            user_name = getattr(user, "name", "?")
        else:
            user_id = "?"
            user_name = "?"
        chan_id = getattr(interaction, "channel_id", None) or getattr(
            getattr(interaction, "channel", None), "id", None,
        )
        guild_id = getattr(interaction, "guild_id", None)
        logger.warning(
            "[Discord] Unauthorized slash attempt: user=%s id=%s channel=%s "
            "guild=%s cmd=%r reason=%r",
            user_name, user_id, chan_id, guild_id, command_text, reason,
        )
        try:
            await interaction.response.send_message(
                "You're not authorized to use this command.", ephemeral=True,
            )
        except Exception as e:
            # Interaction may already be responded to (caller deferred, Discord retry).
            logger.debug("[Discord] Could not send unauthorized ephemeral: %s", e)
        # Fire-and-forget: don't block the interaction handler on Telegram I/O.
        try:
            asyncio.create_task(self._notify_unauthorized_slash(
                user_name, user_id, chan_id, guild_id, command_text, reason,
            ))
        except Exception as e:
            logger.debug("[Discord] Could not schedule admin notify task: %s", e)
        return False

    @staticmethod
    async def _alert_adapters_and_config(runner, profile):
        """``(adapter_map, gateway_config)`` of the profile owning this adapter. Default/primary: the
        runner's own. Secondary: ``_profile_adapters[profile]`` and the config loaded under that
        profile's runtime scope (its ``home_channel`` entries live in ITS config.yaml)."""
        if not profile:
            return runner.adapters, runner.config
        adapters = runner._adapters_for_profile(profile)
        if adapters is runner.adapters:  # profile IS the primary
            return adapters, runner.config
        from gateway.config import load_gateway_config
        from gateway.run import _async_profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir
        async with _async_profile_runtime_scope(get_profile_dir(profile)):
            return adapters, load_gateway_config()

    async def _notify_unauthorized_slash(
        self, user_name: str, user_id: str, chan_id, guild_id, command_text: str, reason: str,
    ) -> None:
        """Best-effort operator alert: TELEGRAM first, then SLACK; no-op without a home channel.
        A soft failure (``SendResult(success=False)``, e.g. rate-limit) continues the fallback chain.
        Under multiplex the alert stays inside THIS adapter's profile: its own adapter map (fail closed
        when the profile has no Telegram/Slack bot) and its own home channels — never the default
        profile's bot or channel, which is what a bare ``runner.adapters`` lookup resolves."""
        runner = getattr(self, "gateway_runner", None)
        if not runner:
            return
        profile = getattr(self, "_owner_profile", None)
        try:
            adapters, config = await self._alert_adapters_and_config(runner, profile)
        except Exception as e:
            logger.debug("[Discord] Admin notify: profile %r resolution failed: %s", profile, e)
            return
        for target in (Platform.TELEGRAM, Platform.SLACK):
            try:
                adapter = adapters.get(target)
                if not adapter:
                    continue
                home = config.get_home_channel(target)
                if not home or not getattr(home, "chat_id", None):
                    continue
                msg = (
                    "⚠️ Unauthorized Discord slash attempt\n"
                    f"User: {user_name} ({user_id})\n"
                    f"Channel: {chan_id} (guild {guild_id})\n"
                    f"Command: {command_text}\n"
                    f"Reason: {reason}"
                )
                result = await adapter.send(str(home.chat_id), msg)
                # Only return on confirmed delivery.
                if getattr(result, "success", None) is False:
                    logger.debug(
                        "[Discord] Admin notify via %s returned success=False"
                        " (error=%r); falling through",
                        target, getattr(result, "error", None),
                    )
                    continue
                return
            except Exception as e:
                logger.debug("[Discord] Admin notify via %s failed: %s", target, e)
