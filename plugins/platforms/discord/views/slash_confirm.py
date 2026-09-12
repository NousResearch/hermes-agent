"""Slash-command confirmation view."""

from __future__ import annotations

from typing import Optional

from .. import adapter as _adapter
from .base import _HermesView

discord = _adapter.discord
logger = _adapter.logger
_read_discord_prompt_timeout = _adapter._read_discord_prompt_timeout
class SlashConfirmView(_HermesView):
    """Approve Once / Always Approve / Cancel for slash-command confirmations (``/reload-mcp``,
    ``GatewayRunner._request_slash_confirm``); clicks call ``tools.slash_confirm.resolve(...)``."""

    def __init__(self, session_key: str, confirm_id: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None):
        super().__init__(allowed_user_ids, allowed_role_ids, timeout=_read_discord_prompt_timeout())
        self.session_key = session_key
        self.confirm_id = confirm_id

    async def _resolve(self, interaction: discord.Interaction, choice: str, color: discord.Color, label: str):
        if not await self._gate(
            interaction, resolved_msg="This prompt has already been resolved~",
            unauth_msg="You're not authorized to answer this prompt~",
        ):
            return
        await self._finalize_embed(interaction, color, f"{label} by {interaction.user.display_name}")
        # A returned follow-up message is posted in the same channel.
        try:
            from tools import slash_confirm as _slash_confirm_mod
            result_text = await _slash_confirm_mod.resolve(self.session_key, self.confirm_id, choice)
            if result_text:
                await interaction.followup.send(result_text)
            logger.info(
                "Discord button resolved slash-confirm for session %s "
                "(choice=%s, user=%s)",
                self.session_key, choice, interaction.user.display_name,
            )
        except Exception as exc:
            logger.error("Discord slash-confirm resolve failed: %s", exc, exc_info=True)

    @discord.ui.button(label="Approve Once", style=discord.ButtonStyle.green)
    async def approve_once(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._resolve(interaction, "once", discord.Color.green(), "Approved once")

    @discord.ui.button(label="Always Approve", style=discord.ButtonStyle.blurple)
    async def approve_always(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._resolve(interaction, "always", discord.Color.purple(), "Always approved")

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._resolve(interaction, "cancel", discord.Color.greyple(), "Cancelled")
