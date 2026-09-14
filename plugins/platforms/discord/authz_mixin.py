"""User-authorization methods for ``DiscordAdapter``.

Extracted from ``plugins/platforms/discord/adapter.py`` as part of the god-file
decomposition campaign, following the same mechanical mixin lift that produced
``gateway/authz_mixin.py``. This mixin holds the Discord authorization cluster:
who is allowed to talk to the agent (users, roles, channels, pairing grants),
the parallel gate for the slash-command interaction surface, the two rejection
paths, and the startup resolution of usernames to numeric IDs.

Behavior-neutral: every method is lifted verbatim from ``DiscordAdapter``.
``self.*`` calls resolve unchanged via the MRO, and ``DiscordAuthorizationMixin``
precedes ``DiscordMediaMixin`` and ``BasePlatformAdapter`` in the bases so
resolution order is what it was when these methods were defined on the class
itself.

Three module-level names need care so the lift stays observationally identical:

* ``logger`` is derived from ``__package__`` rather than ``__name__``, so
  records emitted from these methods keep the adapter's logger identity. The
  name cannot be hard-coded: the plugin manager loads directory plugins under
  ``hermes_plugins.<slug>``, so the adapter's own ``getLogger(__name__)`` is
  ``hermes_plugins.discord.adapter`` there and
  ``plugins.platforms.discord.adapter`` under the canonical path. Deriving
  from ``__package__`` tracks whichever namespace is live, and ``getLogger``
  returns the same singleton object the adapter module holds.
* ``discord`` is imported under the same ``ImportError`` guard the adapter uses,
  and resolves to the same module object, so tests that patch attributes on it
  (for example ``discord.DMChannel``) still take effect here.
* ``_read_dm_role_auth_guild`` moves with the mixin: it is a module-level helper
  read by ``_is_allowed_user`` and by nothing else in the tree.

``_multiplex_active`` stays in the adapter, which owns the rest of the
profile-scoping helpers; the wrapper below delegates to it at call time rather
than copying the body, so the two modules cannot drift apart and a patch of
``adapter._multiplex_active`` is still observed here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Tuple

from gateway.config import Platform

try:
    import discord
except ImportError:  # pragma: no cover - mirrors the adapter's import guard
    discord = None

# Bind the adapter's logger relative to whichever package this module was
# imported under, so log records lifted with these methods are emitted under
# exactly the name they were before, on both the canonical and the
# PluginManager-namespaced import paths.
logger = logging.getLogger(f"{__package__}.adapter")


def _multiplex_active() -> bool:
    """True when the gateway is running in multiplex_profiles mode.

    ``_resolve_allowed_usernames`` guards its legacy ``os.environ`` rewrite on
    this. The check stays defined in the adapter, which owns the rest of the
    profile-scoping helpers; delegating rather than copying keeps the two
    modules from drifting apart. The import is deferred to call time because
    the adapter imports this module while it is still executing.
    """
    from .adapter import _multiplex_active as _adapter_multiplex_active

    return _adapter_multiplex_active()

def _read_dm_role_auth_guild() -> Optional[int]:
    """Return the guild ID opted-in for DM role-based auth, or None (secure default). Read from
    config.yaml ``discord.dm_role_auth_guild`` only (behavioral, not a secret); int or numeric string."""
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config() or {}
        discord_cfg = cfg.get("discord", {}) or {}
        raw = discord_cfg.get("dm_role_auth_guild")
    except Exception:
        return None
    if raw is None or raw == "":
        return None
    try:
        guild_id = int(raw)
    except (TypeError, ValueError):
        return None
    return guild_id if guild_id > 0 else None


class DiscordAuthorizationMixin:
    """Authorization cluster lifted verbatim from ``DiscordAdapter``."""

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
            dm_guild_id = _read_dm_role_auth_guild()
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

    async def _resolve_allowed_usernames(self) -> None:
        """Resolve username/display-name entries in DISCORD_ALLOWED_USERS to numeric IDs."""
        if not self._allowed_user_ids or not self._client:
            return
        numeric_ids = set()
        to_resolve = set()
        for entry in self._allowed_user_ids:
            if entry.isdigit():
                numeric_ids.add(entry)
            elif entry == "*":
                # Keep the "*" wildcard verbatim; it can't resolve and would be silently dropped.
                numeric_ids.add(entry)
            else:
                to_resolve.add(entry.lower())
        if not to_resolve:
            return
        print(f"[{self.name}] Resolving {len(to_resolve)} username(s): {', '.join(to_resolve)}")
        resolved_count = 0
        for guild in self._client.guilds:
            # Fetch full member list (requires members intent)
            try:
                members = guild.members
                if len(members) < guild.member_count:
                    members = [m async for m in guild.fetch_members(limit=None)]
            except Exception as e:
                logger.warning("Failed to fetch members for guild %s: %s", guild.name, e)
                continue
            for member in members:
                name_lower = member.name.lower()
                display_lower = member.display_name.lower()
                global_lower = (member.global_name or "").lower()
                matched = name_lower in to_resolve or display_lower in to_resolve or global_lower in to_resolve
                if matched:
                    uid = str(member.id)
                    numeric_ids.add(uid)
                    resolved_count += 1
                    matched_name = name_lower if name_lower in to_resolve else (
                        display_lower if display_lower in to_resolve else global_lower
                    )
                    to_resolve.discard(matched_name)
                    print(f"[{self.name}] Resolved '{matched_name}' -> {uid} ({member.name}#{member.discriminator})")
            if not to_resolve:
                break
        if to_resolve:
            print(f"[{self.name}] Could not resolve usernames: {', '.join(to_resolve)}")
        # Adapter-local: under multiplex_profiles os.environ writes would clobber other profiles.
        # Update the internal set. Keep the resolved IDs adapter-local first: under multiplex_profiles,
        # writing os.environ here would clobber every OTHER profile's DISCORD_ALLOWED_USERS after this
        # adapter's on_ready — an unguarded runtime mutation of process-global state (issue #72348). Refresh
        # this adapter's own snapshot instead.
        self._allowed_user_ids = numeric_ids
        snap = getattr(self, "_gate_env_snapshot", None)
        if snap is not None:
            snap["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric_ids))
        if not _multiplex_active():
            # Single-profile: legacy env rewrite so gateway env-based auth sees numeric IDs.
            os.environ["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric_ids))
        if resolved_count:
            print(f"[{self.name}] Updated DISCORD_ALLOWED_USERS with {resolved_count} resolved ID(s)")
