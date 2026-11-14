"""The Discord card for the browser vault's one-time-code prompt: a button that opens a modal.

The gateway asks through ``send_vault_code_prompt`` (``gateway/session_vault_code_card.py``). Only the
turn's author may open the modal, and the code goes from the modal submit to the gateway's ``submit``
callback: it is never posted, echoed in a reply or logged. ``VaultCodeView`` is built from
``adapter._define_discord_view_classes`` because discord.py can be installed lazily after the adapter
module is imported.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from gateway.platforms.base import SendResult

# Discord caps a modal title and a modal label at 45 characters.
_MODAL_TEXT_LIMIT = 45
_HINT_LIMIT = 300


def build_vault_code_view(discord, base):
    """Return the ``VaultCodeView`` class on ``base`` (the adapter's shared ``_HermesView``)."""
    from plugins.platforms.discord.adapter import _t_discord

    class VaultCodeView(base):
        """One "Enter code" button. The base admission (allowlist, roles, pairing) still applies, and
        on top of it only ``user_id``, whose turn asked for the code, may open the modal."""

        def __init__(self, *, site: str, user_id: str, submit: Callable[[str], bool], allowed_user_ids: set,
                     allowed_role_ids: Optional[set] = None, timeout: float):
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=timeout)
            self.site = site
            self.user_id = str(user_id)
            self._submit = submit
            self._localize_buttons(enter_code="platform.shared.vault_code.button")

        def _check_auth(self, interaction) -> bool:
            if not super()._check_auth(interaction):
                return False
            return str(getattr(getattr(interaction, "user", None), "id", "")) == self.user_id

        def build_modal(self):
            view = self

            class VaultCodeModal(discord.ui.Modal):
                def __init__(self):
                    super().__init__(title=_t_discord("platform.shared.vault_code.modal_title", _MODAL_TEXT_LIMIT,
                                                      site=view.site), timeout=view.timeout)
                    self.code = discord.ui.TextInput(min_length=1, max_length=32)
                    self.add_item(discord.ui.Label(
                        text=_t_discord("platform.shared.vault_code.input_label", _MODAL_TEXT_LIMIT),
                        component=self.code))

                async def on_submit(self, interaction):
                    await view.submit_code(interaction, str(self.code.value or ""))

            return VaultCodeModal()

        async def submit_code(self, interaction, code: str) -> None:
            if not await self._gate(interaction, resolved_msg=_t_discord("platform.shared.vault_code.ended", 2000),
                                    unauth_msg=_t_discord("platform.shared.vault_code.not_author", 2000)):
                return
            code = code.strip()
            if not code:
                # Blank: nothing to enter and nothing ended; the button stays live for another try.
                await interaction.response.defer()
                return
            accepted = self._submit(code)
            self.resolved = True
            self._disable_all()
            self.stop()
            notice = "platform.shared.vault_code.submitted" if accepted else "platform.shared.vault_code.ended"
            await interaction.response.send_message(_t_discord(notice, 2000), ephemeral=True)
            footer = ("platform.shared.vault_code.submitted_footer" if accepted
                      else "platform.discord.prompt.expired_footer")
            await self._expire_embed(_t_discord(footer, 2048))

        @discord.ui.button(label="Enter code", style=discord.ButtonStyle.blurple)
        async def enter_code(self, interaction, button):
            if not await self._gate(interaction, resolved_msg=_t_discord("platform.shared.vault_code.ended", 2000),
                                    unauth_msg=_t_discord("platform.shared.vault_code.not_author", 2000)):
                return
            await interaction.response.send_modal(self.build_modal())

    return VaultCodeView


class DiscordVaultCodeMixin:
    """``send_vault_code_prompt``: the capability the gateway detects to offer the code prompt here."""

    _allowed_user_ids: set
    _allowed_role_ids: set
    _send_prompt: Any

    async def send_vault_code_prompt(self, chat_id: str, site: str, hint: str, *, user_id: str,
                                     submit: Callable[[str], bool], timeout_s: float,
                                     metadata: Optional[dict] = None) -> SendResult:
        from plugins.platforms.discord.adapter import _DISCORD_EMBED_TITLE_LIMIT, VaultCodeView, _t_discord, discord
        view = VaultCodeView(site=site, user_id=user_id, submit=submit, allowed_user_ids=self._allowed_user_ids,
                             allowed_role_ids=self._allowed_role_ids, timeout=timeout_s)

        def _build(_channel):
            content = _t_discord("platform.shared.vault_code.body", 1500, site=site)
            if hint:
                content += "\n" + hint[:_HINT_LIMIT]
            embed = discord.Embed(
                title=_t_discord("platform.shared.vault_code.title", _DISCORD_EMBED_TITLE_LIMIT, site=site),
                color=discord.Color.gold())
            return {"content": content, "embed": embed, "view": view}, view
        return await self._send_prompt(chat_id, metadata, _build, fail_log="vault code prompt")
