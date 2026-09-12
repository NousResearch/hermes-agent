"""Dangerous-command approval buttons."""

from __future__ import annotations

from typing import Optional

from .. import adapter as _adapter
from .base import _HermesView

discord = _adapter.discord
logger = _adapter.logger
_read_discord_prompt_timeout = _adapter._read_discord_prompt_timeout

class ExecApprovalView(_HermesView):
    """Allow Once / Allow Session / Always Allow / Deny buttons for a dangerous command.
    Clicks call ``resolve_gateway_approval()`` — the same mechanism as the text ``/approve`` flow."""

    def __init__(
        self, session_key: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None,
        require_admin: bool = False, admin_user_ids: Optional[set] = None,
        allow_permanent: bool = True, allow_session: bool = True, smart_denied: bool = False,
    ):
        super().__init__(allowed_user_ids, allowed_role_ids, timeout=_read_discord_prompt_timeout())
        self.session_key = session_key
        self.require_admin = require_admin
        self.admin_user_ids = {str(a).strip() for a in (admin_user_ids or set()) if str(a).strip()}
        if smart_denied or not allow_session:
            self.remove_item(self.allow_session)
            self.remove_item(self.allow_always)
        elif not allow_permanent:
            self.remove_item(self.allow_always)

    def _check_auth(self, interaction: discord.Interaction) -> bool:
        """Base admission always required; with ``require_admin`` the clicker must
        also be an admin. Fails closed (logged once) when no admins are configured."""
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

    async def _resolve(self, interaction: discord.Interaction, choice: str, color: discord.Color, label: str):
        """Resolve the approval via the gateway approval queue and update the embed."""
        if not await self._gate(
            interaction, resolved_msg="This approval has already been resolved~",
            unauth_msg="You're not authorized to approve commands~",
        ):
            return
        self.resolved = True
        # Unblock the waiting agent thread FIRST. A click after the approval
        # wait timed out (count == 0) must not claim "Approved".
        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(self.session_key, choice)
            logger.info(
                "Discord button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                count, self.session_key, choice, interaction.user.display_name,
            )
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from button: %s", exc)
            count = 0
        if not count:
            color = discord.Color.dark_grey()
            label = "⌛ Approval expired — command was not run (already timed out or resolved elsewhere)"
        await self._finalize_embed(
            interaction, color, f"{label} by {interaction.user.display_name}" if count else label)

    @discord.ui.button(label="Allow Once", style=discord.ButtonStyle.green)
    async def allow_once(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._resolve(interaction, "once", discord.Color.green(), "Approved once")

    @discord.ui.button(label="Allow Session", style=discord.ButtonStyle.grey)
    async def allow_session(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._resolve(interaction, "session", discord.Color.blue(), "Approved for session")

    @discord.ui.button(label="Always Allow", style=discord.ButtonStyle.blurple)
    async def allow_always(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._resolve(interaction, "always", discord.Color.purple(), "Approved permanently")

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
    async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._resolve(interaction, "deny", discord.Color.red(), "Denied")
