"""``can_use_tool`` wiring of ``ClaudeAgentSdkSession``: the audited approval bypass,
the bounded auto-allowed MCP readers, the immutable Bash floors and the
canonical-request path into the approval callback. Extracted from
``claude_agent_sdk_session.py``; every method resolves through
``ClaudeAgentSdkSession``'s MRO unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from agent.transports.claude_agent_sdk_session_sanitize import (
    _SDK_CALLBACK_CHOICE_MAX_UTF8_BYTES,
    _SDK_CALLBACK_REASON_MAX_UTF8_BYTES,
    _SDK_FIXED_LOG_TOOL_IDENTITIES,
    _canonical_sdk_tool_request,
    _is_bounded_sdk_callback_string,
    _safe_sdk_deny_log_reason,
    _safe_sdk_tool_use_id,
    safe_sdk_tool_presentation_from_canonical,
    validate_canonical_sdk_request_serialization,
)

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


# Exact names only: these are the SDK profile's two bounded readers. Never
# widen this to an MCP/server wildcard or an unexposed mutation identity.
_SDK_AUTO_ALLOWED_MCP_TOOLS = frozenset({
    "mcp__hermes-tools__read_file",
    "mcp__hermes-tools__search_files",
})


class ClaudeSdkPermissionsMixin:
    """Approval bypass, can_use_tool factory and resolver (see module docstring)."""

    def _sdk_approval_bypass_active(self) -> bool:
        """Resolve only trusted session/process/config bypass intent."""
        if self._sdk_approval_bypass_requested:
            return True
        provider = self._approval_bypass_provider
        if provider is not None:
            try:
                return provider() is True
            except Exception:
                return False
        try:
            from tools.approval import is_approval_bypass_active_for_session

            return is_approval_bypass_active_for_session(
                self._hermes_session_id or "",
            )
        except Exception:
            return False

    def _make_can_use_tool(self) -> Any:
        """Bridge SDK permission requests onto Hermes' approval callback.
        Fail-closed: any callback failure denies.

        Silent-deny observability (P2.d): every deny that was NOT an
        operator's choice is logged at INFO here — this is the choke point
        every SDK-lane deny transits with tool name and honest reason in
        hand (the 2026-08-06 incident's silent denies had no log line at
        all). "denied by user" (with or without ": <text>") appears IFF a
        human chose deny — W8/W11 reserved that wording — so the prefix is
        the discriminator; operator denies are not silent and not logged
        here. Honesty boundary: settings deny-rule hits on the SDK side
        (the CLI consulting ~/.claude/settings.json deny rules, e.g. an
        installer-only skills-dir rule) never invoke can_use_tool and never
        transit hermes code — they are UNLOGGABLE here by construction;
        operator-facing relief for that class is a separate deny-notice
        feature decision."""
        approval_callback = self._approval_callback
        hermes_session_id = self._hermes_session_id

        async def _can_use_tool(tool_name: str, tool_input: dict, context: Any):
            from claude_agent_sdk import (
                PermissionResultAllow,
                PermissionResultDeny,
            )

            # Capture the watch OBJECT at entry: an approval that outlives
            # its turn (orphaned wait, self-expiring at approvals.timeout)
            # must decrement the watch it suspended, never a later turn's.
            # None = no Hermes turn in flight (unsolicited CLI-side turn) —
            # nothing to suspend.
            watch = self._turn_watch
            if watch is not None:
                # A human is being asked — their think time is not the
                # turn's silence. Both watchdog rules stand down until the
                # tap (or the approval machinery's own timeout) resolves.
                watch.approval_begin()
            try:
                return await self._resolve_can_use_tool(
                    tool_name, tool_input, context,
                    approval_callback, hermes_session_id,
                    PermissionResultAllow, PermissionResultDeny,
                )
            finally:
                if watch is not None:
                    watch.approval_end()

        return _can_use_tool

    async def _resolve_can_use_tool(
        self,
        tool_name: str,
        tool_input: dict,
        context: Any,
        approval_callback: Any,
        hermes_session_id: Any,
        PermissionResultAllow: Any,
        PermissionResultDeny: Any,
    ) -> Any:
        try:
            canonical_tool_input = _canonical_sdk_tool_request(tool_name, tool_input)
            checked_request = validate_canonical_sdk_request_serialization(
                canonical_tool_input,
            )
            presentation = safe_sdk_tool_presentation_from_canonical(
                canonical_tool_input,
            )
        except Exception:
            checked_request = None
            presentation = None
        if checked_request is None or presentation is None:
            return PermissionResultDeny(message="canonical request is unassessable")
        frozen_tool_input = checked_request[1]["tool_input"]
        # Validate correlation metadata before every policy decision. Legacy
        # callbacks retain their exact ABI, but no floor or bypass decision can
        # be made from callback markers or callback-controlled text.
        safe_tool_use_id = _safe_sdk_tool_use_id(context)
        callback_command, callback_description = presentation
        log_tool_identity = (
            tool_name if tool_name in _SDK_FIXED_LOG_TOOL_IDENTITIES else "sdk-tool"
        )
        # Native Read remains unavailable even if this callback is invoked
        # directly despite the SDK disallowed_tools option.
        if tool_name == "Read":
            return PermissionResultDeny(message="native SDK Read is disallowed")
        if tool_name == "Bash":
            try:
                from tools.approval_sdk_gateway import sdk_bash_immutable_floor_reason

                floor_reason = sdk_bash_immutable_floor_reason(
                    frozen_tool_input.get("command"),
                )
            except Exception:
                floor_reason = "canonical request is unassessable"
            if floor_reason is not None:
                return PermissionResultDeny(message="approval denied by callback")
        if tool_name in _SDK_AUTO_ALLOWED_MCP_TOOLS:
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if self._sdk_approval_bypass_active():
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if approval_callback is None:
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback unavailable",
            )
            return PermissionResultDeny(message="approval callback unavailable")
        try:
            kwargs: dict = {"allow_permanent": False}
            try:
                from tools.approval_sdk_gateway import (
                    is_trusted_sdk_gateway_approval_callback,
                )

                trusted_gateway_callback = (
                    is_trusted_sdk_gateway_approval_callback(approval_callback)
                )
            except Exception:
                trusted_gateway_callback = False
            # Correlation and canonical JSON are additive only for callbacks
            # that explicitly advertise the corresponding SDK ABI extension.
            if getattr(approval_callback, "_accepts_tool_use_id", False):
                kwargs["tool_use_id"] = safe_tool_use_id
            if getattr(approval_callback, "_accepts_canonical_tool_input", False):
                kwargs["canonical_tool_input"] = canonical_tool_input
            result = await asyncio.to_thread(
                approval_callback,
                callback_command,
                callback_description,
                **kwargs,
            )
        except Exception:
            logger.warning("SDK approval callback failed at protected boundary")
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback failed",
            )
            return PermissionResultDeny(message="approval callback failed")
        # Widened callback contract: a plain choice string, or a dict
        # {"choice": str, "reason": str} carrying an honest deny reason
        # (no-approver / timeout / notify-failure / teardown-expiry).
        # "denied by user" is reserved for a real human deny — a
        # reason-bearing deny must never be attributed to the user.
        try:
            reason = None
            operator_denied = False
            reason_shape = False
            operator_denial_shape = False
            choice = result
            if type(result) is dict:
                # The callback result is untrusted. Protocol dicts have exactly
                # one of three tiny widths; reject all others before collecting,
                # copying, hashing, or otherwise traversing callback keys:
                # {choice}, {choice, reason}, or the trusted gateway-only
                # {choice, operator_denial, reason} structural denial.
                width = len(result)
                if width == 1 and "choice" in result:
                    choice = result["choice"]
                elif width == 2 and "choice" in result and "reason" in result:
                    choice = result["choice"]
                    reason = result["reason"]
                    reason_shape = True
                elif (
                    width == 3
                    and "choice" in result
                    and "operator_denial" in result
                    and "reason" in result
                ):
                    choice = result["choice"]
                    reason = result["reason"]
                    reason_shape = True
                    operator_denial_shape = True
                else:
                    raise ValueError("malformed callback result shape")
            elif type(result) is not str:
                raise TypeError("unsupported callback result")
            if not _is_bounded_sdk_callback_string(
                choice,
                _SDK_CALLBACK_CHOICE_MAX_UTF8_BYTES,
                allow_space=False,
            ) or (
                reason is not None
                and not _is_bounded_sdk_callback_string(
                    reason,
                    _SDK_CALLBACK_REASON_MAX_UTF8_BYTES,
                    allow_space=True,
                )
            ):
                raise TypeError("malformed callback result")
            if reason_shape and choice != "deny":
                raise ValueError("reason is valid only for deny")
            if operator_denial_shape:
                operator_denied = (
                    choice == "deny"
                    and result["operator_denial"] is True
                    and trusted_gateway_callback
                )
                if not operator_denied:
                    raise ValueError("untrusted operator-denial result")
            if choice not in {"once", "session", "always", "deny", "timeout"}:
                raise ValueError("unknown callback choice")
        except Exception:
            logger.warning("SDK approval callback failed at protected boundary")
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, "approval callback failed",
            )
            return PermissionResultDeny(message="approval callback failed")
        if choice in ("once", "session", "always"):
            return PermissionResultAllow(updated_input=frozen_tool_input)
        if operator_denied:
            message = "denied by user"
            if reason:
                message = f"{message}: {reason}"
        elif choice == "timeout":
            message = "approval timed out — no operator response"
        else:
            safe_reason = _safe_sdk_deny_log_reason(reason)
            message = (
                safe_reason
                if safe_reason != "non-operator denial"
                else "approval denied by callback"
            )
        if not operator_denied:
            logger.info(
                "claude-agent-sdk: silent deny (no operator choice): "
                "tool=%s reason=%s",
                log_tool_identity, _safe_sdk_deny_log_reason(message),
            )
        return PermissionResultDeny(message=message)
