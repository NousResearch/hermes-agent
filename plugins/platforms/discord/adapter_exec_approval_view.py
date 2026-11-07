"""The Discord exec-approval card's button view.

Built by ``build_exec_approval_view`` from ``adapter._define_discord_view_classes``, because discord.py can
be installed lazily after the adapter module is imported.
"""
import asyncio
import logging
from typing import Optional

from agent.i18n import t
from plugins.platforms.discord.adapter_config_readers import _read_discord_prompt_timeout

logger = logging.getLogger(__name__)


def build_exec_approval_view(discord, base, *, unauthorized):
    """Return the ``ExecApprovalView`` class on ``base`` (the adapter's shared ``_HermesView``)."""
    class ExecApprovalView(base):
        """Allow Once / Allow Session / Always Allow / Deny buttons for a dangerous command.
        Clicks call ``resolve_gateway_approval()`` — the same mechanism as the text ``/approve`` flow."""

        def __init__(
            self, session_key: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None,
            require_admin: bool = False, admin_user_ids: Optional[set] = None,
            allow_permanent: bool = True, allow_session: bool = True, smart_denied: bool = False,
            request_id: Optional[str] = None,
        ):
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=_read_discord_prompt_timeout())
            self.session_key = session_key
            self.request_id = request_id
            self.pending_retire: Optional[str] = None  # settle reason that arrived before the card was sent
            self._backend_settled = False
            self._retire_lock = asyncio.Lock()
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

        async def on_timeout(self):
            await self.retire("timeout", backend_settled=False)

        async def retire(self, reason="resolved", *, backend_settled=True):
            """Disable a card when its particular backend request is no longer pending."""
            self.stop()
            # A UI timeout can precede a text answer. Serialize their edits, and
            # never let a queued UI timeout overwrite a backend settlement.
            async with self._retire_lock:
                if self._backend_settled:
                    return
                self._backend_settled = backend_settled
                self.resolved = True
                self._disable_all()
                footer = t({
                    "resolved": "platform.discord.approval.already_resolved",
                    "timeout": "platform.discord.prompt.expired_footer",
                    "gone": "platform.discord.approval.no_longer_pending",
                }.get(reason, "platform.discord.slash_confirm.resolved_cancel"))
                # The timeout notice edits only the content, so it never restores the controls.
                await self._expire_embed(footer, keep_controls=False)

        async def _resolve(self, interaction: discord.Interaction, choice: str, color: discord.Color, label_key: str):
            """Resolve the approval via the gateway approval queue and update the embed."""
            if not await self._gate(
                interaction, resolved_msg=t("platform.discord.approval.already_resolved"),
                unauth_msg=unauthorized(),
            ):
                return
            label = t(label_key)
            self.resolved = True
            self._backend_settled = True
            # Unblock the waiting agent thread FIRST. A click after the approval
            # wait timed out (count == 0) must not claim "Approved".
            try:
                from tools.approval import resolve_gateway_approval
                count = resolve_gateway_approval(self.session_key, choice, request_id=self.request_id)
                logger.info(
                    "Discord button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, self.session_key, choice, interaction.user.display_name,
                )
            except Exception as exc:
                logger.error("Failed to resolve gateway approval from button: %s", exc, exc_info=True)
                # Nothing was recorded and the request may still be pending: tell only the clicker
                # and keep the card live for a retry. A later click, the request's observer, or
                # (with no request id) the view timeout does the final edit.
                self.resolved = False
                self._backend_settled = False
                await interaction.response.send_message(
                    t("platform.discord.approval.click_failed"), ephemeral=True)
                return
            if not count:
                self._backend_settled = False
                if self.request_id:
                    # The request already settled through another path, whose observer owns this
                    # card; editing it here too could land last and hide the real outcome.
                    await interaction.response.send_message(
                        t("platform.discord.approval.no_longer_pending"), ephemeral=True)
                    return
                color = discord.Color.dark_grey()
                label = t("platform.discord.approval.expired")
            try:
                await self._finalize_embed(
                    interaction, color,
                    t("platform.discord.approval.by_user", label=label, user=interaction.user.display_name) if count else label)
            except Exception:
                if count:
                    # The answer is recorded but this click's edit failed, and the observer skips cards
                    # whose click owns the edit: stamp the card through its message instead.
                    self._backend_settled = False
                    await self.retire("resolved")
                raise

        # Decorator labels are placeholders; ``_localize_buttons`` in __init__ sets the real text.
        @discord.ui.button(label="Allow Once", style=discord.ButtonStyle.green)
        async def allow_once(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "once", discord.Color.green(), "platform.discord.approval.resolved_once")

        @discord.ui.button(label="Allow Session", style=discord.ButtonStyle.grey)
        async def allow_session(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "session", discord.Color.blue(), "platform.discord.approval.resolved_session")

        @discord.ui.button(label="Always Allow", style=discord.ButtonStyle.blurple)
        async def allow_always(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "always", discord.Color.purple(), "platform.discord.approval.resolved_always")

        @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
        async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "deny", discord.Color.red(), "platform.discord.approval.resolved_deny")

    return ExecApprovalView
