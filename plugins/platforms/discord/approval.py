"""Native approval prompt rendering and request callbacks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from gateway.platforms.base import BasePlatformAdapter, ExecApprovalPrompt, SendResult


class DiscordApprovalMixin(BasePlatformAdapter):
    if TYPE_CHECKING:
        _allowed_role_ids: set

        _allowed_user_ids: set

        def _approval_mention_content(self) -> Optional[str]:
            ...

        async def _send_prompt(self, chat_id: str, metadata: Optional[dict], build, *, fail_log: Optional[str]=...) -> SendResult:
            ...

        config: Any

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Send an approval with content as its canonical payload and an embed for state."""
        from .adapter import ExecApprovalView, _DISCORD_EMBED_TITLE_LIMIT, _resolve_exec_approval_admin_gate, _truncate_discord_component_text, discord, t

        def _build(_channel):
            content = prompt.text
            mention_content = self._approval_mention_content()
            if mention_content:
                content = f"{mention_content}\n{content}"
            embed = discord.Embed(
                title=_truncate_discord_component_text(f"⚠️ {t('gateway.exec_approval.header')}", _DISCORD_EMBED_TITLE_LIMIT),
                color=discord.Color.orange(),
            )
            require_admin, admin_user_ids = _resolve_exec_approval_admin_gate(getattr(self.config, "extra", None))
            choices = set(prompt.choices)
            view = ExecApprovalView(
                session_key=prompt.session_key, request_id=prompt.request_id, allowed_user_ids=self._allowed_user_ids,
                allowed_role_ids=self._allowed_role_ids, require_admin=require_admin,
                admin_user_ids=admin_user_ids, allow_permanent="always" in choices,
                allow_session="session" in choices, smart_denied=prompt.smart_denied,
            )
            send_kwargs: dict[str, Any] = {"content": content, "embed": embed, "view": view}
            if mention_content:
                allowed_mentions_cls = getattr(discord, "AllowedMentions", None)
                if allowed_mentions_cls is not None:
                    send_kwargs["allowed_mentions"] = allowed_mentions_cls(
                        users=True, roles=False, everyone=False, replied_user=False,
                    )
            return send_kwargs, view
        return await self._send_prompt(prompt.chat_id, prompt.metadata, _build)



def create_exec_approval_view(_HermesView):
    from .adapter import discord

    class ExecApprovalView(_HermesView):
        """Allow Once / Allow Session / Always Allow / Deny buttons for a dangerous command.
        Clicks call ``resolve_gateway_approval()`` — the same mechanism as the text ``/approve`` flow."""

        def __init__(
            self, session_key: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None,
            require_admin: bool = False, admin_user_ids: Optional[set] = None,
            allow_permanent: bool = True, allow_session: bool = True, smart_denied: bool = False,
            request_id: Optional[str] = None,
        ):
            from .adapter import _read_discord_prompt_timeout

            super().__init__(allowed_user_ids, allowed_role_ids, timeout=_read_discord_prompt_timeout())
            self.session_key = session_key
            self.request_id = request_id
            self.require_admin = require_admin
            self.admin_user_ids = {str(a).strip() for a in (admin_user_ids or set()) if str(a).strip()}
            self._localize_buttons(
                allow_once="gateway.exec_approval.action_once", allow_session="gateway.exec_approval.action_session",
                allow_always="gateway.exec_approval.action_always", deny="gateway.exec_approval.action_deny")
            if smart_denied or not allow_session:
                self.remove_item(self.allow_session)
                self.remove_item(self.allow_always)
            elif not allow_permanent:
                self.remove_item(self.allow_always)

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            """Base admission always required; with ``require_admin`` the clicker must
            also be an admin. Fails closed (logged once) when no admins are configured."""
            from .adapter import logger

            if not super()._check_auth(interaction):
                return False
            if not self.require_admin:
                return True
            user = getattr(interaction, "user", None)
            try:
                uid = str(getattr(user, "id", "") or "")
            except Exception:
                uid = ""
            if uid and uid in self.admin_user_ids:
                return True
            if not self.admin_user_ids:
                logger.warning(
                    "[Discord] require_admin_for_exec_approval is enabled but "
                    "no admins are configured (allow_admin_from is empty) — "
                    "exec approval buttons are disabled for everyone. Add "
                    "admin user IDs under the discord platform's "
                    "allow_admin_from, or disable the toggle."
                )
            return False

        async def _resolve(self, interaction: discord.Interaction, choice: str, color: discord.Color, label_key: str):
            """Resolve the approval via the gateway approval queue and update the embed."""
            from .adapter import _unauthorized, discord, logger, t

            if not await self._gate(
                interaction, resolved_msg=t("platform.discord.approval.already_resolved"),
                unauth_msg=_unauthorized(),
            ):
                return
            label = t(label_key)
            self.resolved = True
            # Unblock the waiting agent thread FIRST. A click after the approval
            # wait timed out (count == 0) must not claim "Approved".
            try:
                from tools.approval import resolve_gateway_approval
                count = (resolve_gateway_approval(self.session_key, choice, request_id=self.request_id)
                         if self.request_id else 0)
                logger.info(
                    "Discord button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, self.session_key, choice, interaction.user.display_name,
                )
            except Exception as exc:
                logger.error("Failed to resolve gateway approval from button: %s", exc)
                count = 0
            if not count:
                color = discord.Color.dark_grey()
                label = t("platform.discord.approval.expired")
            await self._finalize_embed(
                interaction, color,
                t("platform.discord.approval.by_user", label=label, user=interaction.user.display_name) if count else label)

        # Decorator labels are placeholders; ``_localize_buttons`` in __init__ sets the real text.
        @discord.ui.button(label="Allow Once", style=discord.ButtonStyle.green)
        async def allow_once(self, interaction: discord.Interaction, button: discord.ui.Button):
            from .adapter import discord

            await self._resolve(interaction, "once", discord.Color.green(), "platform.discord.approval.resolved_once")

        @discord.ui.button(label="Allow Session", style=discord.ButtonStyle.grey)
        async def allow_session(self, interaction: discord.Interaction, button: discord.ui.Button):
            from .adapter import discord

            await self._resolve(interaction, "session", discord.Color.blue(), "platform.discord.approval.resolved_session")

        @discord.ui.button(label="Always Allow", style=discord.ButtonStyle.blurple)
        async def allow_always(self, interaction: discord.Interaction, button: discord.ui.Button):
            from .adapter import discord

            await self._resolve(interaction, "always", discord.Color.purple(), "platform.discord.approval.resolved_always")

        @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
        async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
            from .adapter import discord

            await self._resolve(interaction, "deny", discord.Color.red(), "platform.discord.approval.resolved_deny")

    return ExecApprovalView
