"""Resolve plugin approval directives using the host-owned approval gate."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_cli.plugins import _PreToolCallDirective


def _resolve_block_from_details(
    details: _PreToolCallDirective, tool_name: str, *, turn_id: str = "", tool_call_id: str = "",
    session_id: str = "", effective_args: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """The ONE place for the fail-closed approval logic: ``block`` blocks with its message; an
    ``approve`` whose gate errors, denies, or times out is blocked; anything else proceeds."""
    if details.action == "block":
        return details.message
    if details.action != "approve":
        return None
    try:
        from tools.approval import request_tool_approval
        from tools.approval_context import reset_current_observability_context, set_current_observability_context
        approval_tokens = set_current_observability_context(
            turn_id=turn_id, tool_call_id=tool_call_id, session_id=session_id)
        try:
            approval_kwargs = {"rule_key": details.rule_key or tool_name}
            if details.approval_policy:
                approval_kwargs.update(approval_policy=details.approval_policy, effective_args=effective_args)
            result = request_tool_approval(tool_name, details.message or "", **approval_kwargs)
        finally:
            reset_current_observability_context(approval_tokens)
    except Exception:  # health: allow BLE001 -- gate failures must stop execution without leaking error bodies
        # Fail-closed: if the gate itself errors, block rather than silently execute an action a
        # plugin flagged for approval.
        return f"BLOCKED: plugin approval gate failed for {tool_name}"
    if not result.get("approved"):
        denied_message = str(result.get("message") or f"BLOCKED: plugin approval required for {tool_name}")
        if details.approval_policy == "mandatory" and details.message:
            return details.message + "\n" + denied_message
        return denied_message
    return None
