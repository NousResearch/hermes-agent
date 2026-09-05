"""Discord commands methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from gateway.platforms.base import MessageEvent
try:
    import discord
    from discord import Message as DiscordMessage
except ImportError:
    discord = None
    DiscordMessage = Any


class DiscordCommandsMixin:
    def _get_discord_command_sync_policy(self) -> str:
        from . import adapter as _adapter

        raw = str(_adapter.os.getenv("DISCORD_COMMAND_SYNC_POLICY", "safe") or "").strip().lower()
        if raw in _adapter._DISCORD_COMMAND_SYNC_POLICIES:
            return raw
        if raw:
            _adapter.logger.warning(
                "[%s] Invalid DISCORD_COMMAND_SYNC_POLICY=%r; falling back to 'safe'", self.name,
                raw,
            )
        return "safe"

    def _canonicalize_app_command_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce command payloads to the semantic fields Hermes manages."""
        from . import adapter as _adapter

        contexts = payload.get("contexts")
        integration_types = payload.get("integration_types")
        return {
            "type": int(payload.get("type", 1) or 1),
            "name": str(payload.get("name", "") or ""),
            "description": str(payload.get("description", "") or ""),
            "default_member_permissions": self._normalize_permissions(
                payload.get("default_member_permissions")
            ),
            "dm_permission": bool(payload.get("dm_permission", True)),
            "nsfw": bool(payload.get("nsfw", False)),
            "contexts": sorted(int(c) for c in contexts) if contexts else None,
            "integration_types": (
                sorted(int(i) for i in integration_types) if integration_types else None
            ),
            "options": [
                self._canonicalize_app_command_option(item)
                for item in payload.get("options", []) or []
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def _normalize_permissions(value: Any) -> Optional[str]:
        """Normalize default_member_permissions to str-or-None (Discord returns str, discord.py sets int)."""
        from . import adapter as _adapter

        if value is None:
            return None
        return str(value)

    def _existing_command_to_payload(self, command: Any) -> Dict[str, Any]:
        """Build a canonical-ready dict from an AppCommand; ``to_dict()`` omits nsfw/dm_permission/
        default_member_permissions, so pull them from attributes or every startup diffs."""
        payload = dict(command.to_dict())
        nsfw = getattr(command, "nsfw", None)
        if nsfw is not None:
            payload["nsfw"] = bool(nsfw)
        guild_only = getattr(command, "guild_only", None)
        if guild_only is not None:
            payload["dm_permission"] = not bool(guild_only)
        default_permissions = getattr(command, "default_member_permissions", None)
        if default_permissions is not None:
            payload["default_member_permissions"] = getattr(
                default_permissions, "value", default_permissions
            )
        return payload

    def _canonicalize_app_command_option(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from . import adapter as _adapter

        return {
            "type": int(payload.get("type", 0) or 0),
            "name": str(payload.get("name", "") or ""),
            "description": str(payload.get("description", "") or ""),
            "required": bool(payload.get("required", False)),
            "autocomplete": bool(payload.get("autocomplete", False)),
            "choices": [
                {
                    "name": str(choice.get("name", "") or ""), "value": choice.get("value"),
                }
                for choice in payload.get("choices", []) or []
                if isinstance(choice, dict)
            ],
            "channel_types": list(payload.get("channel_types", []) or []),
            "min_value": payload.get("min_value"),
            "max_value": payload.get("max_value"),
            "min_length": payload.get("min_length"),
            "max_length": payload.get("max_length"),
            "options": [
                self._canonicalize_app_command_option(item)
                for item in payload.get("options", []) or []
                if isinstance(item, dict)
            ],
        }

    def _patchable_app_command_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fields supported by discord.py's edit_global_command route."""
        canonical = self._canonicalize_app_command_payload(payload)
        return {
            "name": canonical["name"], "description": canonical["description"],
            "options": canonical["options"],
        }

    async def _safe_sync_slash_commands(self) -> Dict[str, int]:
        """Diff existing global commands and only mutate the commands that changed."""
        from . import adapter as _adapter

        summary = {"total": 0, "unchanged": 0, "updated": 0, "recreated": 0, "created": 0, "deleted": 0}
        if not self._client:
            return summary
        tree = self._client.tree
        app_id = getattr(self._client, "application_id", None) or getattr(getattr(self._client, "user", None), "id", None)
        if not app_id:
            raise RuntimeError("Discord application ID is unavailable for slash command sync")
        desired_payloads = [command.to_dict(tree) for command in tree.get_commands()]
        desired_by_key = {
            (int(payload.get("type", 1) or 1), str(payload.get("name", "") or "").lower()): payload
            for payload in desired_payloads
        }
        existing_commands = await tree.fetch_commands()
        existing_by_key = {
            (
                int(getattr(getattr(command, "type", None), "value", getattr(command, "type", 1)) or 1),
                str(command.name or "").lower(),
            ): command
            for command in existing_commands
        }
        http = self._client.http
        mutation_count = 0

        async def mutate(call, *args):
            nonlocal mutation_count
            if mutation_count:
                await self._sleep_between_command_sync_mutations()
            result = await call(*args)
            mutation_count += 1
            return result
        # Delete obsolete commands FIRST: an upsert pushing the live total over 100 fails with
        # 30032 (breaks ALL slash commands), so an app at the cap must shrink before creating.
        obsolete_keys = set(existing_by_key.keys()) - set(desired_by_key.keys())
        for key in obsolete_keys:
            current = existing_by_key.pop(key)
            await mutate(http.delete_global_command, app_id, current.id)
            summary["deleted"] += 1
        for key, desired in desired_by_key.items():
            current = existing_by_key.pop(key, None)
            if current is None:
                await mutate(http.upsert_global_command, app_id, desired)
                summary["created"] += 1
                continue
            current_existing_payload = self._existing_command_to_payload(current)
            current_payload = self._canonicalize_app_command_payload(current_existing_payload)
            desired_payload = self._canonicalize_app_command_payload(desired)
            if current_payload == desired_payload:
                summary["unchanged"] += 1
                continue
            if self._patchable_app_command_payload(current_existing_payload) == self._patchable_app_command_payload(desired):
                await mutate(http.delete_global_command, app_id, current.id)
                await mutate(http.upsert_global_command, app_id, desired)
                summary["recreated"] += 1
                continue
            await mutate(http.edit_global_command, app_id, current.id, desired)
            summary["updated"] += 1
        summary["total"] = len(desired_payloads)
        return summary

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
        from . import adapter as _adapter

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
        from . import adapter as _adapter

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
        from . import adapter as _adapter

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
        from . import adapter as _adapter

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
        _adapter.logger.warning(
            "[%s] Discord messages are being denied because no allowlist is configured. "
            "Set DISCORD_ALLOWED_USERS, DISCORD_ALLOWED_ROLES, or "
            "DISCORD_ALLOWED_CHANNELS, or set DISCORD_ALLOW_ALL_USERS=true for open access.",
            self.name,
        )

    def _evaluate_slash_authorization(
        self, interaction: "discord.Interaction",
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate slash authorization without responding; returns ``(allowed, reason)``.
        Shared with side-effect-free callers (``/skill`` autocomplete returns [] per keystroke).
        Fail closed on malformed payloads: with an allowlist, a missing channel id/user REJECTS.
        """
        from . import adapter as _adapter

        chan_obj = getattr(interaction, "channel", None)
        in_dm = isinstance(chan_obj, _adapter.discord.DMChannel) if chan_obj is not None else False
        channel_ids: set = set()
        channel_keys: set = set()
        # Channel scope mirrors on_message; DMs use on_message's DM lockdown path instead.
        if not in_dm:
            chan_id_raw = getattr(interaction, "channel_id", None) or getattr(chan_obj, "id", None)
            if chan_id_raw is not None:
                channel_ids.add(str(chan_id_raw))
                # Threads: also test the parent channel, as on_message does.
                if isinstance(chan_obj, _adapter.discord.Thread):
                    parent_id = self._get_parent_channel_id(chan_obj)
                    if parent_id:
                        channel_ids.add(str(parent_id))
            # Name-form keys (ID, name, #name, parent) so name-based lists work for slash too.
            channel_keys = self._discord_channel_keys_from_channel(
                chan_obj,
                self._get_parent_channel_id(chan_obj)
                if isinstance(chan_obj, _adapter.discord.Thread)
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
        from . import adapter as _adapter

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
        _adapter.logger.warning(
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
            _adapter.logger.debug("[Discord] Could not send unauthorized ephemeral: %s", e)
        # Fire-and-forget: don't block the interaction handler on Telegram I/O.
        try:
            _adapter.asyncio.create_task(self._notify_unauthorized_slash(
                user_name, user_id, chan_id, guild_id, command_text, reason,
            ))
        except Exception as e:
            _adapter.logger.debug("[Discord] Could not schedule admin notify task: %s", e)
        return False

    async def _notify_unauthorized_slash(
        self, user_name: str, user_id: str, chan_id, guild_id, command_text: str, reason: str,
    ) -> None:
        """Best-effort operator alert: TELEGRAM first, then SLACK; no-op without a home channel.
        A soft failure (``SendResult(success=False)``, e.g. rate-limit) continues the fallback chain."""
        from . import adapter as _adapter

        runner = getattr(self, "gateway_runner", None)
        if not runner:
            return
        for target in (_adapter.Platform.TELEGRAM, _adapter.Platform.SLACK):
            try:
                adapter = runner.adapters.get(target)
                if not adapter:
                    continue
                home = runner.config.get_home_channel(target)
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
                    _adapter.logger.debug(
                        "[Discord] Admin notify via %s returned success=False"
                        " (error=%r); falling through",
                        target, getattr(result, "error", None),
                    )
                    continue
                return
            except Exception as e:
                _adapter.logger.debug("[Discord] Admin notify via %s failed: %s", target, e)

    async def _defer_unless_expired(self, interaction: discord.Interaction, warn_fmt: str, *warn_args) -> bool:
        """Ephemeral defer(); False (after a warning) when the interaction token already expired
        so the caller still runs the command but skips followups. Other errors propagate."""
        from . import adapter as _adapter

        try:
            await interaction.response.defer(ephemeral=True)
            return True
        except Exception as e:
            if not self._is_discord_unknown_interaction(e):
                raise
            _adapter.logger.warning(warn_fmt, *warn_args)
            return False

    async def _run_simple_slash(
        self, interaction: discord.Interaction, command_text: str, followup_msg: str | None = None,
    ) -> None:
        """Defer, dispatch the command string, then replace/delete the "thinking..." indicator."""
        # Log the invoker so ghost-command reports can be triaged post-mortem.
        from . import adapter as _adapter

        try:
            _user = interaction.user
            _chan_id = getattr(interaction.channel, "id", None) or getattr(interaction, "channel_id", None)
            _adapter.logger.info(
                "[Discord] slash '%s' invoked by user=%s id=%s channel=%s guild=%s", command_text,
                getattr(_user, "name", "?"), getattr(_user, "id", "?"), _chan_id,
                getattr(interaction, "guild_id", None),
            )
        except Exception:
            pass  # logging must never block command dispatch
        # Auth gate must precede defer() so the ephemeral rejection can still be sent.
        if not await self._check_slash_authorization(interaction, command_text):
            return
        deferred_response = await self._defer_unless_expired(
            interaction,
            "[Discord] slash %s: interaction expired before defer. "
            "Executing command anyway, skipping interaction followup.", command_text,
        )
        event = self._build_slash_event(interaction, command_text)
        await self.handle_message(event)
        if not deferred_response:
            return
        try:
            if followup_msg:
                await interaction.edit_original_response(content=followup_msg)
            else:
                await interaction.delete_original_response()
        except Exception as e:
            _adapter.logger.debug("Discord interaction cleanup failed: %s", e)

    def _slash_proxy(self, name: str, args: tuple, template: str, followup: Optional[str], *,
                     strip: bool = True, prefix: str = "slash_"):
        """Build a slash callback rendering ``template`` from its args via ``_run_simple_slash``;
        the introspected signature is synthesised from ``args`` (see ``_NATIVE_SLASH_COMMANDS``)."""
        from . import adapter as _adapter

        async def _handler(interaction: discord.Interaction, **kwargs):
            text = template.format(**kwargs)
            call_args = (text.strip() if strip else text,) + (() if followup is None else (followup,))
            await self._run_simple_slash(interaction, *call_args)
        _handler.__name__ = prefix + {"bg": "background"}.get(name, name).replace("-", "_")
        params = [_adapter.inspect.Parameter("interaction", _adapter.inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=_adapter.discord.Interaction)]
        for arg_name, arg_type, default, _desc, _choices in args:
            params.append(_adapter.inspect.Parameter(
                arg_name, _adapter.inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=arg_type,
                default=_adapter.inspect.Parameter.empty if default is _adapter._REQUIRED else default,
            ))
        _handler.__signature__ = _adapter.inspect.Signature(params)
        if args:
            _handler = _adapter.discord.app_commands.describe(**{a[0]: a[3] for a in args})(_handler)
            choices = {a[0]: [_adapter.discord.app_commands.Choice(name=lbl, value=val) for lbl, val in a[4]] for a in args if a[4]}
            if choices:
                _handler = _adapter.discord.app_commands.choices(**choices)(_handler)
        return _handler

    def _register_thread_slash(self, tree, name: str, description: str) -> None:
        from . import adapter as _adapter

        @tree.command(name=name, description=description)
        @_adapter.discord.app_commands.describe(
            name="Thread name", message="Optional first message to send to Hermes in the thread",
            auto_archive_duration="Auto-archive in minutes (60, 1440, 4320, 10080)",
        )
        async def slash_thread(
            interaction: discord.Interaction, name: str, message: str = "",
            auto_archive_duration: int = 1440,
        ):
            # defer() happens inside the handler *after* the auth gate.
            await self._handle_thread_create_slash(interaction, name, message, auto_archive_duration)

    def _register_slash_commands(self) -> None:
        """Register Discord slash commands on the command tree."""
        from . import adapter as _adapter

        if not self._client:
            return
        tree = self._client.tree
        for name, description, args, template, followup in _adapter._NATIVE_SLASH_COMMANDS:
            if template is None:
                self._register_thread_slash(tree, name, description)
                continue
            tree.command(name=name, description=description)(
                self._slash_proxy(name, args, template, followup, strip=name != "insights")
            )
        # Auto-register COMMAND_REGISTRY + plugin commands not yet on the tree. Native
        # commands above always survive the 100-command cap; reserve one slot for /skill.
        already_registered: set[str] = set()
        slot_cap = _adapter._DISCORD_MAX_APP_COMMANDS - 1
        dropped_over_cap = 0

        def _auto_register(name: str, description: str, args_hint: str) -> None:
            nonlocal dropped_over_cap
            # Discord command names: lowercase, hyphens OK, max 32 chars.
            discord_name = name.lower()[:32]
            if discord_name in already_registered:
                return
            if len(already_registered) >= slot_cap:
                dropped_over_cap += 1
                return
            args = (("args", str, "", f"Arguments: {args_hint}"[:100], None),) if args_hint else ()
            template = f"/{name} {{args}}" if args_hint else f"/{name}"
            auto_cmd = _adapter.discord.app_commands.Command(
                name=discord_name, description=(description or f"Run /{name}")[:100],
                callback=self._slash_proxy(name, args, template, None, strip=bool(args_hint), prefix="auto_slash_"),
            )
            try:
                tree.add_command(auto_cmd)
                already_registered.add(discord_name)
            except Exception:
                # e.g. name conflict with a subcommand group.
                pass
        try:
            from hermes_cli.commands import COMMAND_REGISTRY, _is_gateway_available, _resolve_config_gates
            try:
                already_registered = {cmd.name for cmd in tree.get_commands()}
            except Exception:
                pass
            config_overrides = _resolve_config_gates()
            for cmd_def in COMMAND_REGISTRY:
                if _is_gateway_available(cmd_def, config_overrides):
                    _auto_register(cmd_def.name, cmd_def.description, cmd_def.args_hint)
            _adapter.logger.debug("Discord auto-registered %d commands from COMMAND_REGISTRY", len(already_registered))
        except Exception as e:
            _adapter.logger.warning("Discord auto-register from COMMAND_REGISTRY failed: %s", e)
        # Mirror PluginContext.register_command() commands into the native slash picker.
        try:
            from hermes_cli.commands import _iter_plugin_command_entries
            for plugin_name, plugin_desc, plugin_args_hint in _iter_plugin_command_entries():
                _auto_register(plugin_name, plugin_desc, plugin_args_hint)
        except Exception as e:
            _adapter.logger.warning("Discord auto-register from plugin commands failed: %s", e)
        self._register_skill_group(tree)
        if dropped_over_cap:
            # One over-limit command makes Discord reject the entire sync (error 30032).
            _adapter.logger.warning(
                "[%s] Reached Discord's limit of %d slash commands; skipped %d "
                "lower-priority command(s) to keep the command sync working. "
                "Disable slash commands you don't need or trim installed plugins "
                "to surface them all.",
                self.name,
                _adapter._DISCORD_MAX_APP_COMMANDS,
                dropped_over_cap,
            )
        # Opt-in UX only: hide slash commands from non-admins; real gate is _check_slash_authorization.
        if _adapter.os.getenv("DISCORD_HIDE_SLASH_COMMANDS", "false").strip().lower() in {
            "true", "1", "yes", "on",
        }:
            self._apply_owner_only_visibility(tree)

    def _apply_owner_only_visibility(self, tree) -> None:
        """Set default_member_permissions=0 on every registered slash command.
        Discord hides ``Permissions(0)`` commands from all but Administrators (re-grantable via
        Integrations); ``_check_slash_authorization`` remains the authoritative gate."""
        from . import adapter as _adapter

        try:
            no_perms = _adapter.discord.Permissions(0)
        except Exception as e:
            _adapter.logger.warning(
                "[Discord] _apply_owner_only_visibility: cannot build Permissions(0): %s", e,
            )
            return
        applied = 0
        for cmd in tree.get_commands():
            try:
                cmd.default_permissions = no_perms
                applied += 1
            except Exception as e:
                _adapter.logger.debug(
                    "[Discord] Could not set default_permissions on %r: %s",
                    getattr(cmd, "name", "?"), e,
                )
        _adapter.logger.info(
            "[Discord] Hid %d slash command(s) from non-admin guild members "
            "(opt-in defense in depth via DISCORD_HIDE_SLASH_COMMANDS).",
            applied,
        )

    def _register_skill_group(self, tree) -> None:
        """Register one flat ``/skill`` command with autocomplete on ``name``.
        A nested ``/skill <category> <name>`` layout blew Discord's ~8000-byte payload cap and broke
        ``tree.sync()``; autocomplete options are fetched dynamically. Entries live on ``self``.

        The older nested layout (``/skill <category> <name>``) registered one giant command whose serialized
        payload grew linearly with the skill catalog — with the default ~75 skills the payload was ~14 KB
        and ``tree.sync()`` rejected the entire slash-command batch (issues 11321, #10259, #11385, #10261,
        #10214).
        """
        from . import adapter as _adapter

        try:
            existing_names = set()
            try:
                existing_names = {cmd.name for cmd in tree.get_commands()}
            except Exception:
                pass
            # Instance-level state so the callbacks always read the freshest entries.
            self._skill_entries: list[tuple[str, str, str]] = []
            self._skill_lookup: dict[str, tuple[str, str]] = {}
            self._skill_group_reserved_names: set[str] = set(existing_names)
            self._refresh_skill_catalog_state()
            if not self._skill_entries:
                return

            async def _autocomplete_name(interaction: "discord.Interaction", current: str) -> list:
                """Filter skills by typed prefix against name and description (Discord caps at 25).
                Unauthorized users get ``[]``: no catalog leak, no per-keystroke ephemeral rejections."""
                try:
                    allowed, _reason = self._evaluate_slash_authorization(interaction)
                except Exception:
                    # Never raise from autocomplete; fail closed.
                    return []
                if not allowed:
                    return []
                q = (current or "").strip().lower()
                choices: list = []
                for name, desc, _key in self._skill_entries:
                    if not q or q in name.lower() or (desc and q in desc.lower()):
                        label = f"{name} — {desc}" if desc else name
                        # Discord's Choice.name is capped at 100 chars.
                        if len(label) > 100:
                            label = label[:97] + "..."
                        choices.append(_adapter.discord.app_commands.Choice(name=label, value=name))
                        if len(choices) >= 25:
                            break
                return choices

            @_adapter.discord.app_commands.describe(
                name="Which skill to run", args="Optional arguments for the skill",
            )
            @_adapter.discord.app_commands.autocomplete(name=_autocomplete_name)
            async def _skill_handler(interaction: "discord.Interaction", name: str, args: str = ""):
                # Authorize BEFORE lookup so unknown/known names reject identically (no catalog probing).
                if not await self._check_slash_authorization(interaction, "/skill"):
                    return
                entry = self._skill_lookup.get(name)
                if not entry:
                    await interaction.response.send_message(
                        f"Unknown skill: `{name}`. Start typing for "
                        f"autocomplete suggestions.",
                        ephemeral=True,
                    )
                    return
                _desc, cmd_key = entry
                await self._run_simple_slash(interaction, f"{cmd_key} {args}".strip())
            cmd = _adapter.discord.app_commands.Command(
                name="skill", description="Run a Hermes skill", callback=_skill_handler,
            )
            tree.add_command(cmd)
            _adapter.logger.info(
                "[%s] Registered /skill command with %d skill(s) via autocomplete",
                self.name, len(self._skill_entries),
            )
            if self._skill_group_hidden_count:
                _adapter.logger.info(
                    "[%s] %d skill(s) filtered out of /skill (name clamp / reserved)",
                    self.name, self._skill_group_hidden_count,
                )
        except Exception as exc:
            _adapter.logger.warning("[%s] Failed to register /skill command: %s", self.name, exc)

    def _refresh_skill_catalog_state(self) -> None:
        """Re-scan disk and repopulate ``self._skill_entries``/``_skill_lookup`` in place.
        No Discord API calls: autocomplete and handler read these attributes directly."""
        from . import adapter as _adapter

        from hermes_cli.commands_platforms import discord_skill_commands_by_category
        reserved = getattr(self, "_skill_group_reserved_names", set())
        categories, uncategorized, hidden = discord_skill_commands_by_category(
            reserved_names=set(reserved),
        )
        entries: list[tuple[str, str, str]] = list(uncategorized)
        for cat_skills in categories.values():
            entries.extend(cat_skills)
        # Stable alphabetical order so autocomplete is predictable across restarts.
        entries.sort(key=lambda t: t[0])
        self._skill_entries = entries
        self._skill_lookup = {n: (d, k) for n, d, k in entries}
        self._skill_group_hidden_count = hidden

    def refresh_skill_group(self) -> tuple[int, int]:
        """Rescan skills and refresh live ``/skill`` autocomplete; returns ``(new_count, hidden_count)``.
        Called after ``reload_skills``; no ``tree.sync()`` since autocomplete options are dynamic."""
        from . import adapter as _adapter

        try:
            self._refresh_skill_catalog_state()
        except Exception as exc:
            _adapter.logger.warning(
                "[%s] Failed to refresh /skill autocomplete after reload: %s", self.name, exc,
            )
            return (len(getattr(self, "_skill_entries", [])), 0)
        _adapter.logger.info(
            "[%s] Refreshed /skill autocomplete: %d skill(s) available (%d filtered)", self.name,
            len(self._skill_entries), self._skill_group_hidden_count,
        )
        return (len(self._skill_entries), self._skill_group_hidden_count)

    def _interaction_guild_id(self, interaction: discord.Interaction) -> Optional[str]:
        """Resolve the guild id of a slash interaction (mirrors the message path)."""
        from . import adapter as _adapter

        guild_id = getattr(interaction, "guild_id", None)
        if guild_id is None:
            guild = getattr(getattr(interaction, "channel", None), "guild", None)
            guild_id = getattr(guild, "id", None)
        return str(guild_id) if guild_id else None

    def _build_slash_event(self, interaction: discord.Interaction, text: str) -> MessageEvent:
        """Build a MessageEvent from a Discord slash command interaction."""
        from . import adapter as _adapter

        is_dm = isinstance(interaction.channel, _adapter.discord.DMChannel)
        is_thread = isinstance(interaction.channel, _adapter.discord.Thread)
        thread_id = None
        if is_dm:
            chat_type = "dm"
        elif is_thread:
            chat_type = "thread"
            thread_id = str(interaction.channel_id)
        else:
            chat_type = "group"
        chat_name = ""
        if not is_dm and hasattr(interaction.channel, "name"):
            chat_name = interaction.channel.name
            if hasattr(interaction.channel, "guild") and interaction.channel.guild:
                chat_name = f"{interaction.channel.guild.name} / #{chat_name}"
        # Forum threads inherit the parent forum's topic.
        chat_topic = self._get_effective_topic(interaction.channel, is_thread=is_thread)
        # guild_id/parent_chat_id feed profile_routes matching, as on_message does.
        # guild_id/parent_chat_id feed profile_routes matching in build_source, exactly as on_message passes
        # them — without them a guild- or channel-routed profile never matches a native slash command
        # (#69178).
        parent_id = (self._get_parent_channel_id(interaction.channel) if is_thread else None) or ""
        source = self.build_source(
            chat_id=str(interaction.channel_id), chat_name=chat_name, chat_type=chat_type,
            user_id=str(interaction.user.id), user_name=interaction.user.display_name,
            thread_id=thread_id, chat_topic=chat_topic,
            guild_id=self._interaction_guild_id(interaction), parent_chat_id=parent_id or None,
        )
        source.is_one_to_one = is_dm
        source.message_is_edit = False
        msg_type = _adapter.MessageType.COMMAND if text.startswith("/") else _adapter.MessageType.TEXT
        channel_id = str(interaction.channel_id)
        return _adapter.MessageEvent(
            text=text, message_type=msg_type, source=source, raw_message=interaction,
            channel_prompt=self._resolve_channel_prompt(channel_id, parent_id or None),
        )

    async def _handle_thread_create_slash(
        self, interaction: discord.Interaction, name: str, message: str = "",
        auto_archive_duration: int = 1440,
    ) -> None:
        """Create a Discord thread from a slash command and start a session in it."""
        if not await self._check_slash_authorization(interaction, "/thread"):
            return
        deferred_response = await self._defer_unless_expired(
            interaction,
            "[Discord] /thread: interaction expired before defer. "
            "Creating the thread anyway, skipping interaction followups.",
        )
        result = await self._create_thread(
            interaction, name=name, message=message, auto_archive_duration=auto_archive_duration,
        )
        if not result.get("success"):
            error = result.get("error", "unknown error")
            if deferred_response:
                await interaction.followup.send(f"Failed to create thread: {error}", ephemeral=True)
            return
        thread_id = result.get("thread_id")
        thread_name = result.get("thread_name") or name
        link = f"<#{thread_id}>" if thread_id else f"**{thread_name}**"
        if deferred_response:
            await interaction.followup.send(f"Created thread {link}", ephemeral=True)
        # Track thread participation so follow-ups don't require @mention
        if thread_id:
            self._threads.mark(thread_id)
        starter = (message or "").strip()
        if starter and thread_id:
            await self._dispatch_thread_session(interaction, thread_id, thread_name, starter)

    async def _dispatch_thread_session(
        self, interaction: discord.Interaction, thread_id: str, thread_name: str, text: str,
    ) -> None:
        """Build a MessageEvent pointing at a thread and send it through handle_message."""
        from . import adapter as _adapter

        guild_name = ""
        if hasattr(interaction, "guild") and interaction.guild:
            guild_name = interaction.guild.name
        chat_name = f"{guild_name} / {thread_name}" if guild_name else thread_name
        # Inherit forum topic when the thread was created inside a forum channel.
        _chan = getattr(interaction, "channel", None)
        chat_topic = self._get_effective_topic(_chan, is_thread=True) if _chan else None
        _parent_channel = self._thread_parent_channel(getattr(interaction, "channel", None))
        _parent_id = str(getattr(_parent_channel, "id", "") or "")
        source = self.build_source(
            chat_id=thread_id, chat_name=chat_name, chat_type="thread",
            user_id=str(interaction.user.id), user_name=interaction.user.display_name,
            thread_id=thread_id, chat_topic=chat_topic,
            guild_id=self._interaction_guild_id(interaction), parent_chat_id=_parent_id or None,
        )
        _skills = self._resolve_channel_skills(thread_id, _parent_id or None)
        _channel_prompt = self._resolve_channel_prompt(thread_id, _parent_id or None)
        event = _adapter.MessageEvent(
            text=text, message_type=_adapter.MessageType.TEXT, source=source, raw_message=interaction,
            auto_skill=_skills, channel_prompt=_channel_prompt,
        )
        await self.handle_message(event)
