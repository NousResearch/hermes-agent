"""/approve, /deny, /approvals slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from typing import Optional

from agent.i18n import t
from gateway.platforms.base import MessageEvent

from gateway.slash_commands._shared import logger

class ApprovalsCommandsMixin:
    """/approve, /deny, /approvals handlers."""

    async def _handle_approvals_command(self, event: MessageEvent) -> str:
        """Show or persist the profile-wide dangerous-command approval mode."""
        from gateway.slash_access import policy_for_source
        from hermes_cli.approval_mode import run_approval_mode_command

        requested = event.get_command_args().strip() or None
        # This mutates profile-wide security policy. The central slash gate can
        # allow selected commands to non-admin users, so enforce admin again at
        # this side-effect boundary. Unconfigured policies remain unrestricted.
        policy = policy_for_source(self.config, event.source)
        if requested and not policy.is_admin(event.source.user_id):
            return "Only gateway admins can change the persistent approval mode."
        result = run_approval_mode_command(requested)
        # Approval checks load config dynamically; do not evict the cached agent
        # or alter its system prompt/tool schema (prompt-cache prefix is sacred).
        return result.message

    async def _handle_approve_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /approve command — unblock waiting agent thread(s).

        The agent thread(s) are blocked inside tools/approval.py waiting for
        the user to respond.  This handler signals the event so the agent
        resumes and the terminal_tool executes the command inline — the same
        flow as the CLI's synchronous input() approval.

        Supports multiple concurrent approvals (parallel subagents,
        execute_code).  ``/approve`` resolves the oldest pending command;
        ``/approve all`` resolves every pending command at once.

        Usage:
            /approve              — approve oldest pending command once
            /approve all          — approve ALL pending commands at once
            /approve session      — approve oldest + remember for session
            /approve all session  — approve all + remember for session
            /approve always       — approve oldest + remember permanently
            /approve all always   — approve all + remember permanently
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )

        if not has_blocking_approval(session_key):
            if session_key in self._pending_approvals:
                self._pending_approvals.pop(session_key)
                return t("gateway.approval_expired")
            return t("gateway.approve.no_pending")

        # Parse args: support "all", "all session", "all always", "session", "always"
        args = event.get_command_args().strip().lower().split()
        resolve_all = "all" in args
        remaining = [a for a in args if a != "all"]

        if any(a in {"always", "permanent", "permanently"} for a in remaining):
            choice = "always"
        elif any(a in {"session", "ses"} for a in remaining):
            choice = "session"
        else:
            choice = "once"

        count = resolve_gateway_approval(session_key, choice, resolve_all=resolve_all)
        if not count:
            return t("gateway.approve.no_pending")

        # Resume typing indicator — agent is about to continue processing.
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        logger.info("User approved %d dangerous command(s) via /approve (%s)", count, choice)
        plural = "plural" if count > 1 else "singular"
        return t(f"gateway.approve.{choice}_{plural}", count=count)

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — reject pending dangerous command(s).

        Signals blocked agent thread(s) with a 'deny' result so they receive
        a definitive BLOCKED message, same as the CLI deny flow.

        ``/deny`` denies the oldest; ``/deny all`` denies everything.
        ``/deny <reason>`` (or ``/deny all <reason>``) attaches a one-line
        reason that is relayed back to the agent so it can adapt instead of
        only hearing "denied". Ported from qwibitai/nanoclaw#2832.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )

        if not has_blocking_approval(session_key):
            if session_key in self._pending_approvals:
                self._pending_approvals.pop(session_key)
                return t("gateway.deny.stale")
            return t("gateway.deny.no_pending")

        # Parse args: a leading "all" token denies every pending command;
        # anything after it (or the whole arg string when "all" is absent) is
        # captured verbatim as the optional deny reason relayed to the agent.
        raw_args = event.get_command_args().strip()
        tokens = raw_args.split()
        resolve_all = bool(tokens) and tokens[0].lower() == "all"
        if resolve_all:
            reason = raw_args[len(tokens[0]):].strip()
        else:
            reason = raw_args
        # Cap to a sane one-liner; the agent only needs a short hint.
        if reason:
            reason = reason[:280].strip()

        count = resolve_gateway_approval(
            session_key, "deny", resolve_all=resolve_all,
            reason=reason or None,
        )
        if not count:
            return t("gateway.deny.no_pending")

        # Resume typing indicator — agent continues (with BLOCKED result).
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        logger.info(
            "User denied %d dangerous command(s) via /deny%s",
            count, " (with reason)" if reason else "",
        )
        if reason:
            if count > 1:
                return t("gateway.deny.denied_reason_plural", count=count, reason=reason)
            return t("gateway.deny.denied_reason_singular", reason=reason)
        if count > 1:
            return t("gateway.deny.denied_plural", count=count)
        return t("gateway.deny.denied_singular")
