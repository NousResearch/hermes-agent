"""Finite-choice Discord select view."""

from __future__ import annotations

from typing import Optional

from .. import adapter as _adapter
from .base import _HermesView

discord = _adapter.discord
logger = _adapter.logger
_DISCORD_SELECT_MAX_OPTIONS = _adapter._DISCORD_SELECT_MAX_OPTIONS
_DISCORD_SELECT_FIELD_LIMIT = _adapter._DISCORD_SELECT_FIELD_LIMIT
_truncate_discord_component_text = _adapter._truncate_discord_component_text
class ChoicePickerView(_HermesView):
    """Flat single-select picker for finite-choice commands (/reasoning, /fast); 2-minute timeout."""

    def __init__(self, choices: list, on_choice_selected, allowed_user_ids: set, allowed_role_ids: Optional[set] = None):
        super().__init__(allowed_user_ids, allowed_role_ids, timeout=120)
        self.choices = list(choices)[:_DISCORD_SELECT_MAX_OPTIONS]
        self.on_choice_selected = on_choice_selected
        options = []
        for choice in self.choices:
            label = str(choice.get("label") or choice.get("value") or "")
            options.append(
                discord.SelectOption(
                    label=_truncate_discord_component_text(label, _DISCORD_SELECT_FIELD_LIMIT),
                    value=str(choice.get("value") or ""),
                    description="current" if choice.get("is_current") else None,
                )
            )
        select = discord.ui.Select(placeholder="Choose an option...", options=options)
        select.callback = self._on_select
        self.add_item(select)

    async def _on_select(self, interaction: discord.Interaction):
        if not self._check_auth(interaction):
            await interaction.response.send_message("⛔ You are not authorized to change this setting.", ephemeral=True)
            return
        if self.resolved:
            await interaction.response.defer()
            return
        self.resolved = True
        value = interaction.data.get("values", [""])[0]
        try:
            result_text = await self.on_choice_selected(str(interaction.channel_id), value)
        except Exception as exc:
            logger.error("Choice picker selection failed: %s", exc)
            result_text = f"Error applying selection: {exc}"
        embed = discord.Embed(description=result_text, color=discord.Color.green())
        self.clear_items()
        self.stop()
        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self):
        if self.resolved:
            return
        msg = self._message
        if msg is not None:
            try:
                embed = discord.Embed(description="⏱ Selection expired — no change made.", color=discord.Color.greyple())
                self.clear_items()
                await msg.edit(embed=embed, view=self)
            except Exception:
                pass
