"""/profile, /whoami, /platform, /version, /commands, /help slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import MessageEvent

class InfoCommandsMixin:
    """/profile, /whoami, /platform, /version, /commands, /help handlers."""

    async def _handle_profile_command(self, event: MessageEvent) -> str:
        """Handle /profile — show the profile serving this source and its home.

        On a multiplexed gateway the process-level active profile is always
        the multiplexer's own (usually ``default``), so reporting it would
        answer "default" in every chat regardless of which profile actually
        serves the room/channel (``source.profile`` — stamped by the
        ``/p/<profile>/`` URL prefix, a per-credential adapter, or a room→
        profile map). When ``multiplex_profiles`` is on, report the stamped
        profile and, like the scoped /reset banner (#59003), resolve the
        displayed home under that profile's runtime scope. When multiplexing
        is off (the default) the stamp is ignored — mirroring the gating in
        ``_run_agent`` and ``_reset_notice_session_info`` — and the command
        reports the active profile and default home, byte-identical to before.
        """
        from hermes_constants import display_hermes_home
        from hermes_cli.slash_exec import CommandContext, execute_command

        multiplexed = getattr(
            getattr(self, "config", None), "multiplex_profiles", False
        )
        source = getattr(event, "source", None)

        profile_name = ""
        display = ""
        if multiplexed:
            profile_name = (getattr(source, "profile", "") or "").strip()
            try:
                from gateway.run import _profile_runtime_scope

                profile_home = self._resolve_profile_home_for_source(source)
                with _profile_runtime_scope(profile_home):
                    display = display_hermes_home()
            except Exception:
                display = display_hermes_home()

        # Shared executor resolves process-level fallbacks; the multiplexed
        # per-source overrides (when any) ride in via options.
        reply = execute_command(
            "profile",
            CommandContext(
                surface="gateway",
                options={"profile_name": profile_name, "home_display": display},
            ),
        )

        lines = [
            t("gateway.profile.header", profile=reply.data["profile"]),
            t("gateway.profile.home", home=reply.data["home"]),
        ]

        return "\n".join(lines)

    async def _handle_whoami_command(self, event: MessageEvent) -> str:
        """Handle /whoami — show the user's slash command access on this scope.

        Always works (it's in the always-allowed floor of slash_access).
        Reports: platform, scope (DM vs group), the user's tier
        (admin / user / unrestricted), and the slash commands they can
        actually run on this scope.
        """
        from gateway.slash_access import policy_for_source as _policy_for_source

        source = event.source
        policy = _policy_for_source(self.config, source)
        platform = source.platform.value if source and source.platform else "?"
        chat_type = (source.chat_type if source else "") or "dm"
        scope = "DM" if chat_type.lower() in {"dm", "direct", "private", ""} else "group/channel"
        user_id = (source.user_id if source else None) or "?"

        if not policy.enabled:
            return (
                f"**You** — {platform} ({scope})\n"
                f"User ID: `{user_id}`\n"
                f"Tier: unrestricted (no admin list configured for this scope)\n"
                f"Slash commands: all available"
            )

        if policy.is_admin(user_id):
            return (
                f"**You** — {platform} ({scope})\n"
                f"User ID: `{user_id}`\n"
                f"Tier: **admin**\n"
                f"Slash commands: all available"
            )

        # Non-admin user. Show what's actually reachable.
        floor = ["help", "whoami"]  # mirrors slash_access._ALWAYS_ALLOWED_FOR_USERS
        configured = sorted(policy.user_allowed_commands)
        # Combine + dedupe, preserve order: floor first, then operator additions.
        seen: set[str] = set()
        runnable: list[str] = []
        for c in floor + configured:
            if c not in seen:
                seen.add(c)
                runnable.append(c)
        runnable_str = ", ".join(f"/{c}" for c in runnable) if runnable else "(none)"
        return (
            f"**You** — {platform} ({scope})\n"
            f"User ID: `{user_id}`\n"
            f"Tier: user\n"
            f"Slash commands you can run: {runnable_str}"
        )

    async def _handle_platform_command(self, event: MessageEvent) -> str:
        """Handle ``/platform list|pause|resume [name]`` — surface and
        manually control failed/paused gateway adapters.

        Examples:
            ``/platform list``           — show connected + failed/paused platforms
            ``/platform pause whatsapp`` — stop the reconnect watcher hammering whatsapp
            ``/platform resume whatsapp`` — re-queue a paused platform for retry
        """
        text = (getattr(event, "content", "") or "").strip()
        # Strip the leading "/platform" (or "/PLATFORM") token if present
        parts = text.split(maxsplit=2)
        if parts and parts[0].lower().lstrip("/").startswith("platform"):
            parts = parts[1:]
        action = (parts[0] if parts else "list").lower()
        target = parts[1].lower() if len(parts) > 1 else ""

        # Resolve platform name (case-insensitive, value match)
        def _resolve_platform(name: str):
            if not name:
                return None
            for p in Platform.__members__.values():
                if p.value.lower() == name:
                    return p
            return None

        if action == "list":
            lines = ["**Gateway platforms**"]
            connected = sorted(p.value for p in self.adapters.keys())
            if connected:
                lines.append("Connected: " + ", ".join(connected))
            else:
                lines.append("Connected: (none)")
            failed = getattr(self, "_failed_platforms", {}) or {}
            if failed:
                for p, info in failed.items():
                    if info.get("paused"):
                        reason = info.get("pause_reason") or "paused"
                        lines.append(
                            f"  · {p.value} — PAUSED ({reason}). "
                            f"Resume with `/platform resume {p.value}`."
                        )
                    else:
                        attempts = info.get("attempts", 0)
                        lines.append(
                            f"  · {p.value} — retrying (attempt {attempts})"
                        )
            else:
                lines.append("Failed/paused: (none)")
            return "\n".join(lines)

        if action in {"pause", "resume"}:
            if not target:
                return f"Usage: /platform {action} <name>"
            platform = _resolve_platform(target)
            if platform is None:
                return f"Unknown platform: {target}"
            failed = getattr(self, "_failed_platforms", {}) or {}
            if action == "pause":
                if platform not in failed:
                    return (
                        f"{platform.value} is not in the retry queue "
                        f"(it's either connected or not enabled)."
                    )
                if failed[platform].get("paused"):
                    return f"{platform.value} is already paused."
                self._pause_failed_platform(platform, reason="paused via /platform pause")
                return (
                    f"✓ {platform.value} paused. "
                    f"Resume with `/platform resume {platform.value}` or "
                    f"`hermes gateway restart` to reset."
                )
            # action == "resume"
            if platform not in failed:
                return (
                    f"{platform.value} is not in the retry queue — "
                    f"nothing to resume."
                )
            if not failed[platform].get("paused"):
                return (
                    f"{platform.value} is already retrying — "
                    f"no resume needed."
                )
            self._resume_paused_platform(platform)
            return f"✓ {platform.value} resumed — retrying on next watcher tick."

        return (
            "Usage: /platform <list|pause|resume> [name]\n"
            "  /platform list — show platform status\n"
            "  /platform pause <name> — stop retrying a failing platform\n"
            "  /platform resume <name> — re-queue a paused platform"
        )

    async def _handle_version_command(self, event: MessageEvent) -> str:
        """Handle /version — show the running Hermes Agent version."""
        from hermes_cli.slash_exec import CommandContext, execute_command

        return execute_command("version", CommandContext(surface="gateway")).text

    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        from gateway.run import _telegramize_command_mentions
        from hermes_cli.slash_exec import CommandContext, execute_command

        reply = execute_command("help", CommandContext(surface="gateway"))
        return _telegramize_command_mentions(
            reply.text,
            getattr(getattr(event, "source", None), "platform", None),
        )

    async def _handle_commands_command(self, event: MessageEvent) -> str:
        from gateway.run import _telegramize_command_mentions
        from hermes_cli.slash_exec import CommandContext, execute_command
        from gateway.config import Platform

        # Page size is a surface parameter (Telegram messages are shorter).
        page_size = 15 if event.source.platform == Platform.TELEGRAM else 20
        reply = execute_command(
            "commands",
            CommandContext(
                surface="gateway",
                args=event.get_command_args(),
                options={"page_size": page_size},
            ),
        )
        return _telegramize_command_mentions(
            reply.text,
            getattr(getattr(event, "source", None), "platform", None),
        )
