"""Shared plumbing for Discord component views."""

from __future__ import annotations

from typing import Optional

import discord


class _HermesView(discord.ui.View):
    """Shared plumbing for Hermes component views: allowlist auth, single-use
    ``resolved`` flag, ``_message`` handle for timeout edits."""

    def __init__(self, allowed_user_ids: set, allowed_role_ids: Optional[set], *, timeout):
        super().__init__(timeout=timeout)
        self.allowed_user_ids = allowed_user_ids
        self.allowed_role_ids = allowed_role_ids or set()
        self.resolved = False
        self._message = None

    def _check_auth(self, interaction: discord.Interaction) -> bool:
        from ..adapter import _component_check_auth
        return _component_check_auth(interaction, self.allowed_user_ids, self.allowed_role_ids)

    async def _gate(self, interaction: discord.Interaction, *, resolved_msg: Optional[str], unauth_msg: str) -> bool:
        """Reject (ephemerally) an already-resolved or unauthorized click; True when it may proceed."""
        if resolved_msg is not None and self.resolved:
            await interaction.response.send_message(resolved_msg, ephemeral=True)
            return False
        if not self._check_auth(interaction):
            await interaction.response.send_message(unauth_msg, ephemeral=True)
            return False
        return True

    def _disable_all(self) -> None:
        for child in self.children:
            child.disabled = True

    @staticmethod
    def _first_embed(message):
        return message.embeds[0] if message.embeds else None

    async def _expire_embed(self, footer: str) -> None:
        """Grey out the original message's embed after a timeout (best effort)."""
        msg = self._message
        if msg:
            try:
                embed = self._first_embed(msg)
                if embed:
                    embed.color = discord.Color.greyple()
                    embed.set_footer(text=footer)
                await msg.edit(embed=embed, view=self)
            except Exception:
                pass  # message deleted or too old to edit

    async def _finalize_embed(self, interaction: discord.Interaction, color, footer: str) -> None:
        """Mark resolved, stamp the embed (color + footer), disable buttons, edit in place."""
        self.resolved = True
        embed = self._first_embed(interaction.message)
        if embed:
            embed.color = color
            embed.set_footer(text=footer)
        self._disable_all()
        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self):
        self.resolved = True
        self._disable_all()
        await self._expire_embed("⏱ Prompt expired — no action taken")
