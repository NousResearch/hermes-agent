"""Gateway update confirmation view."""

from __future__ import annotations

from typing import Optional

from .. import adapter as _adapter
from .base import _HermesView

discord = _adapter.discord
logger = _adapter.logger
_read_discord_prompt_timeout = _adapter._read_discord_prompt_timeout
class UpdatePromptView(_HermesView):
    """Yes/No buttons for ``hermes update`` prompts; the answer is written to
    ``.update_response`` for the detached update process to pick up."""

    def __init__(self, session_key: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None):
        super().__init__(allowed_user_ids, allowed_role_ids, timeout=_read_discord_prompt_timeout())
        self.session_key = session_key

    async def _respond(self, interaction: discord.Interaction, answer: str, color: discord.Color, label: str):
        if not await self._gate(interaction, resolved_msg="Already answered~", unauth_msg="You're not authorized~"):
            return
        await self._finalize_embed(interaction, color, f"{label} by {interaction.user.display_name}")
        try:
            from hermes_constants import get_hermes_home
            response_path = get_hermes_home() / ".update_response"
            tmp = response_path.with_suffix(".tmp")
            tmp.write_text(answer, encoding="utf-8")
            tmp.replace(response_path)
            logger.info("Discord update prompt answered '%s' by %s", answer, interaction.user.display_name)
        except Exception as exc:
            logger.error("Failed to write update response: %s", exc)

    @discord.ui.button(label="Yes", style=discord.ButtonStyle.green, emoji="✓")
    async def yes_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._respond(interaction, "y", discord.Color.green(), "Yes")

    @discord.ui.button(label="No", style=discord.ButtonStyle.red, emoji="✗")
    async def no_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._respond(interaction, "n", discord.Color.red(), "No")
