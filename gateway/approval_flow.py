"""
Approval flow — dangerous-command approval handling for the gateway.

Extracted from gateway/run.py so the approval machinery lives in its own
module.  The public interface is re-exported from ``gateway.run`` so
downstream imports remain valid.

Core pieces:
- ``_handle_approve_command`` — handles ``/approve`` to unblock agent threads
- ``_handle_deny_command`` — handles ``/deny`` to reject pending commands
- ``_APPROVAL_TIMEOUT_SECONDS`` — 5-minute timeout for pending approvals
- ``_build_approval_notify_sync`` — factory for the per-run approval callback
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from gateway.message_router import MessageEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Command handlers — these methods live on the HermesRunner class but are
# defined here for readability.  They reference ``self`` and expect the
# standard HermesRunner attributes (_pending_approvals, adapters, etc.).
# ---------------------------------------------------------------------------


async def _handle_approve_command(self, event: "MessageEvent") -> Optional[str]:
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
    from agent.i18n import t

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


async def _handle_deny_command(self, event: "MessageEvent") -> str:
    """Handle /deny command — reject pending dangerous command(s).

    Signals blocked agent thread(s) with a 'deny' result so they receive
    a definitive BLOCKED message, same as the CLI deny flow.

    ``/deny`` denies the oldest; ``/deny all`` denies everything.
    """
    from agent.i18n import t

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

    args = event.get_command_args().strip().lower()
    resolve_all = "all" in args

    count = resolve_gateway_approval(session_key, "deny", resolve_all=resolve_all)
    if not count:
        return t("gateway.deny.no_pending")

    # Resume typing indicator — agent continues (with BLOCKED result).
    _adapter = self.adapters.get(source.platform)
    if _adapter:
        _adapter.resume_typing_for_chat(source.chat_id)

    logger.info("User denied %d dangerous command(s) via /deny", count)
    if count > 1:
        return t("gateway.deny.denied_plural", count=count)
    return t("gateway.deny.denied_singular")

# ---------------------------------------------------------------------------
# Approval notification callback factory
#
# Called inside _process_message_with_agent to build the per-run callback
# that bridges sync -> async for sending approval requests to the user.
# ---------------------------------------------------------------------------


def _build_approval_notify_sync(
    self,
    status_adapter: Any,
    status_chat_id: Any,
    status_thread_metadata: Any,
    approval_session_key: str,
    loop_for_step: Any,
) -> Callable[[dict], None]:
    """Return a sync callback that sends a dangerous-command approval request.

    The callback is registered with ``tools.approval.register_gateway_notify``
    so the agent thread can block waiting for user input and this function
    delivers the prompt to the user via the messaging adapter.

    If the adapter supports interactive button-based approvals
    (e.g. Discord's ``send_exec_approval``), use that for a richer UX.
    Otherwise fall back to a plain text message with ``/approve`` instructions.
    """
    from agent.async_utils import safe_schedule_threadsafe

    def _approval_notify_sync(approval_data: dict) -> None:
        """Send the approval request to the user from the agent thread.

        If the adapter supports interactive button-based approvals
        (e.g. Discord's ``send_exec_approval``), use that for a richer
        UX.  Otherwise fall back to a plain text message with
        ``/approve`` instructions.
        """
        # Pause the typing indicator while the agent waits for
        # user approval.  Critical for Slack's Assistant API where
        # assistant_threads_setStatus disables the compose box — the
        # user literally cannot type /approve while "is thinking..."
        # is active.  The approval message send auto-clears the Slack
        # status; pausing prevents _keep_typing from re-setting it.
        # Typing resumes in _handle_approve_command/_handle_deny_command.
        status_adapter.pause_typing_for_chat(status_chat_id)

        cmd = approval_data.get("command", "")
        desc = approval_data.get("description", "dangerous command")

        # Prefer button-based approval when the adapter supports it.
        # Check the *class* for the method, not the instance — avoids
        # false positives from MagicMock auto-attribute creation in tests.
        if getattr(type(status_adapter), "send_exec_approval", None) is not None:
            try:
                _approval_fut = safe_schedule_threadsafe(
                    status_adapter.send_exec_approval(
                        chat_id=status_chat_id,
                        command=cmd,
                        session_key=approval_session_key,
                        description=desc,
                        metadata=status_thread_metadata,
                    ),
                    loop_for_step,
                    logger=logger,
                    log_message="send_exec_approval scheduling error",
                )
                if _approval_fut is None:
                    raise RuntimeError("send_exec_approval: loop unavailable")
                _approval_result = _approval_fut.result(timeout=15)
                if _approval_result.success:
                    return
                logger.warning(
                    "Button-based approval failed (send returned error), falling back to text: %s",
                    _approval_result.error,
                )
            except Exception as _e:
                logger.warning(
                    "Button-based approval failed, falling back to text: %s", _e
                )

        # Fallback: plain text approval prompt
        cmd_preview = cmd[:200] + "..." if len(cmd) > 200 else cmd
        msg = (
            f"⚠️ **Dangerous command requires approval:**\n"
            f"```\n{cmd_preview}\n```\n"
            f"Reason: {desc}\n\n"
            f"Reply `/approve` to execute, `/approve session` to approve this pattern "
            f"for the session, `/approve always` to approve permanently, or `/deny` to cancel."
        )
        try:
            _approval_send_fut = safe_schedule_threadsafe(
                status_adapter.send(
                    status_chat_id,
                    msg,
                    metadata=status_thread_metadata,
                ),
                loop_for_step,
                logger=logger,
                log_message="Approval text-send scheduling error",
            )
            if _approval_send_fut is not None:
                _approval_send_fut.result(timeout=15)
        except Exception as _e:
            logger.error("Failed to send approval request: %s", _e)

    return _approval_notify_sync
