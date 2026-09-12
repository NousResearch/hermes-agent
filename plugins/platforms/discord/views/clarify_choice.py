"""Clarification choice buttons for Discord."""

from __future__ import annotations

from typing import List, Optional

from .. import adapter as _adapter
from .base import _HermesView

discord = _adapter.discord
logger = _adapter.logger
_read_discord_prompt_timeout = _adapter._read_discord_prompt_timeout
_DISCORD_BUTTON_LABEL_LIMIT = _adapter._DISCORD_BUTTON_LABEL_LIMIT
_DISCORD_ELLIPSIS = _adapter._DISCORD_ELLIPSIS
_prefix_within_utf16_limit = _adapter._prefix_within_utf16_limit
utf16_len = _adapter.utf16_len
_truncate_discord_component_text = _adapter._truncate_discord_component_text

class ClarifyChoiceView(_HermesView):
    """One button per clarify choice (max 24) plus ``✏️ Other``. A numeric click resolves the
    gateway clarify entry immediately; ``Other`` flips to text-capture (next message answers).
    Single-use: after the first valid click all buttons disable."""

    def __init__(self, choices: List[str], clarify_id: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None):
        super().__init__(allowed_user_ids, allowed_role_ids, timeout=_read_discord_prompt_timeout())
        self.choices = list(choices)[:24]
        self.clarify_id = clarify_id
        for index, choice in enumerate(self.choices):
            button = discord.ui.Button(
                label=self._button_label(index, choice), style=discord.ButtonStyle.primary,
                custom_id=f"clarify:{clarify_id}:{index}",
            )
            button.callback = self._make_choice_callback(index, choice)
            self.add_item(button)
        other_btn = discord.ui.Button(
            label="✏️ Other (type answer)", style=discord.ButtonStyle.secondary,
            custom_id=f"clarify:{clarify_id}:other",
        )
        other_btn.callback = self._on_other
        self.add_item(other_btn)

    @staticmethod
    def _button_label(index: int, choice: str) -> str:
        """``"N. <choice>"`` within Discord's 80-char (UTF-16) label cap.
        Mobile wraps early, so long choices cut at a word boundary in the trailing half, else a
        soft boundary (``- , . )``, inclusive), else hard."""
        prefix = f"{index + 1}. "
        budget = _DISCORD_BUTTON_LABEL_LIMIT - utf16_len(prefix)
        if utf16_len(choice) <= budget:
            return f"{prefix}{choice}"
        truncated = _prefix_within_utf16_limit(choice, max(0, budget - utf16_len(_DISCORD_ELLIPSIS))).rstrip()
        cut_at = -1
        space = truncated.rfind(" ")
        if space >= len(truncated) // 2:
            cut_at = space
        if cut_at < 0:
            latest_soft = max((truncated.rfind(s) for s in ("-", ",", ".", ")")), default=-1)
            if latest_soft >= len(truncated) // 2:
                cut_at = latest_soft + 1
        if cut_at > 0:
            truncated = truncated[:cut_at]
        return f"{prefix}{truncated.rstrip() + _DISCORD_ELLIPSIS}"

    def _make_choice_callback(self, index: int, choice: str):
        async def _callback(interaction: "discord.Interaction"):
            await self._resolve_choice(interaction, index, choice)
        return _callback

    async def _finish(self, interaction: "discord.Interaction", color, footer: str, *, log_edit_failure: bool) -> None:
        """Disable the buttons and stamp the embed; fall back to a bare defer."""
        self.resolved = True
        self._disable_all()
        embed = self._first_embed(interaction.message) if interaction.message else None
        if embed:
            embed.color = color
            embed.set_footer(text=footer)
        try:
            await interaction.response.edit_message(embed=embed, view=self)
        except Exception:
            if log_edit_failure:
                logger.debug("Discord clarify edit_message failed for %s", self.clarify_id, exc_info=True)
            try:
                await interaction.response.defer()
            except Exception:
                pass

    async def _resolve_choice(self, interaction: "discord.Interaction", index: int, choice: str) -> None:
        """Resolve the clarify with a chosen option."""
        if not await self._gate(
            interaction, resolved_msg="This prompt has already been answered~",
            unauth_msg="You're not authorized to answer this prompt~",
        ):
            return
        display_name = getattr(getattr(interaction, "user", None), "display_name", "user")
        await self._finish(interaction, discord.Color.green(), f"Answered by {display_name}: {choice}", log_edit_failure=True)
        # Round-trip the canonical choice text from the entry, not the button label.
        resolved_text: Optional[str] = None
        try:
            from tools.clarify_gateway import _entries as _clarify_entries  # type: ignore
            entry = _clarify_entries.get(self.clarify_id)
            if entry and entry.choices and 0 <= index < len(entry.choices):
                resolved_text = entry.choices[index]
        except Exception:
            resolved_text = None
        if resolved_text is None:
            resolved_text = choice
        try:
            from tools.clarify_gateway import resolve_gateway_clarify
            resolved = resolve_gateway_clarify(self.clarify_id, resolved_text)
            logger.info(
                "Discord clarify button resolved (id=%s, choice=%r, user=%s, ok=%s)",
                self.clarify_id, resolved_text,
                getattr(getattr(interaction, "user", None), "display_name", "?"), resolved,
            )
        except Exception as exc:
            logger.error("Discord clarify resolve_gateway_clarify failed (id=%s): %s", self.clarify_id, exc)

    async def _on_other(self, interaction: "discord.Interaction") -> None:
        """Flip the clarify entry into text-capture mode."""
        if not await self._gate(
            interaction, resolved_msg="This prompt has already been answered~",
            unauth_msg="You're not authorized to answer this prompt~",
        ):
            return
        # Don't pop: the gateway text-intercept needs the entry until the user types.
        try:
            from tools.clarify_gateway import mark_awaiting_text
            mark_awaiting_text(self.clarify_id)
        except Exception as exc:
            logger.warning("Discord clarify mark_awaiting_text failed (id=%s): %s", self.clarify_id, exc)
        display_name = getattr(getattr(interaction, "user", None), "display_name", "user")
        await self._finish(interaction, discord.Color.blue(), f"Awaiting typed response from {display_name}…", log_edit_failure=False)
