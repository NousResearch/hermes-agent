"""Request-bound Discord exec-approval cards (#104915).

Dedicated owner for the dangerous-command approval-card authorization surface:
admin-gate resolution, the card payload, and the request-bound view whose
buttons settle only the approval generation the card was issued for. The
transport adapter keeps a thin wiring method (``send_exec_approval``) and
re-exports the view class for its lazy discord runtime.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_exec_approval_admin_gate(config_extra: Optional[dict]) -> Tuple[bool, set]:
    """Resolve the exec-approval admin gate from ``extra``; returns ``(require_admin, admin_user_ids)``.
    Default OFF (user-scope buttons). When ``require_admin_for_exec_approval`` is true only
    ``allow_admin_from`` ids may click; on with no admins -> ``(True, set())`` (fail closed, log once).
    """
    extra = config_extra if isinstance(config_extra, dict) else {}
    raw_toggle = extra.get("require_admin_for_exec_approval", False)
    require_admin = str(raw_toggle).strip().lower() in {"true", "1", "yes"}
    if not require_admin:
        return (False, set())
    try:
        from gateway.slash_access import _coerce_id_list
        admin_ids = set(_coerce_id_list(extra.get("allow_admin_from")))
    except Exception:
        admin_ids = set()
    return (True, admin_ids)


def build_exec_approval_card(
    command: str, description: str, *, smart_denied: bool, mention_content: Optional[str],
    max_message_length: int, embed_body: Any,
) -> tuple:
    """Build the ``(content, embed)`` card payload for ``send_exec_approval``.

    Payload in plain content: embeds can be invisible/detached on web/mobile.
    """
    import discord  # runtime discord (real or the shared test mock); cards are only built on a live adapter

    reason_budget = 300
    reason_display = str(description or "dangerous command")
    if len(reason_display) > reason_budget:
        reason_display = reason_display[: reason_budget - 15] + "... [truncated]"
    prompt_prefix = (
        "⚠️ **Command Approval Required**\n\n"
        "Do you want Hermes to run this command?\n\n"
        "**Requested command:**\n```bash\n"
    )
    if smart_denied:
        prompt_prefix += "**Smart DENY:** owner override applies to this one operation only.\n\n"
    if mention_content:
        prompt_prefix = f"{mention_content}\n{prompt_prefix}"
    prompt_tail = f"\n```\n**Reason:** {reason_display}"
    truncated_suffix = "\n... [truncated]"
    command_budget = max(0, max_message_length - len(prompt_prefix) - len(prompt_tail))
    content_cmd_display = str(command or "")
    if len(content_cmd_display) > command_budget:
        content_cmd_display = content_cmd_display[: max(0, command_budget - len(truncated_suffix))] + truncated_suffix
    content = f"{prompt_prefix}{content_cmd_display}{prompt_tail}"
    embed = discord.Embed(
        title="⚠️ Command Approval Required",
        description=f"```\n{embed_body(str(command or ''))}\n```",
        color=discord.Color.orange(),
    )
    embed.add_field(name="Reason", value=reason_display, inline=False)
    return content, embed


def build_exec_approval_view(
    *, config_extra: Optional[dict], session_key: str, allowed_user_ids: set,
    allowed_role_ids: Optional[set] = None, allow_permanent: bool = True,
    allow_session: bool = True, smart_denied: bool = False, request_id: Optional[str] = None,
):
    """Instantiate the adapter-registered ``ExecApprovalView`` with the resolved admin gate."""
    from plugins.platforms.discord.adapter import ExecApprovalView  # registered by _define_discord_view_classes

    require_admin, admin_user_ids = resolve_exec_approval_admin_gate(config_extra)
    return ExecApprovalView(
        session_key=session_key, allowed_user_ids=allowed_user_ids,
        allowed_role_ids=allowed_role_ids, require_admin=require_admin,
        admin_user_ids=admin_user_ids, allow_permanent=allow_permanent,
        allow_session=allow_session, smart_denied=smart_denied, request_id=request_id,
    )


def define_exec_approval_view(base_view) -> type:
    """Build ``ExecApprovalView`` on the adapter's shared component-view base.

    Called from ``_define_discord_view_classes()`` at module load and after a
    lazy install, so the class exists whenever DISCORD_AVAILABLE and always
    shares the base's auth/gate/embed plumbing with the sibling views.
    """
    import discord  # only defined once the discord runtime is importable

    class ExecApprovalView(base_view):
        """Allow Once / Allow Session / Always Allow / Deny buttons for a dangerous command.
        Clicks call ``resolve_gateway_approval()`` — the same mechanism as the text ``/approve`` flow."""

        def __init__(
            self, session_key: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None,
            require_admin: bool = False, admin_user_ids: Optional[set] = None,
            allow_permanent: bool = True, allow_session: bool = True, smart_denied: bool = False,
            request_id: Optional[str] = None,
        ):
            from plugins.platforms.discord.adapter import _read_discord_prompt_timeout
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=_read_discord_prompt_timeout())
            self.session_key = session_key
            self.request_id = request_id
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
                # A card resolves only the request it was issued for; an unbound card or one
                # whose request is gone must never settle a different queued command (#104915).
                # Text /approve keeps its FIFO semantics inside resolve_gateway_approval.
                count = (
                    resolve_gateway_approval(self.session_key, choice, request_id=self.request_id)
                    if self.request_id
                    else 0
                )
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

    return ExecApprovalView
