"""``pre_tool_call`` directive seam: parse hook returns into a directive and resolve it.

Owns the per-thread tool whitelist (side questions, background review), the
``block`` / ``approve`` / ``modify`` directive parser, and the ONE fail-closed resolver
that runs an ``approve`` through the human-approval gate. Callers (``model_tools``,
``agent/tool_executor``, ``agent/agent_runtime_helpers``) import from here.
"""

from __future__ import annotations

import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


_thread_tool_whitelist = threading.local()


@dataclass(frozen=True)
class _PreToolCallDirective:
    action: Optional[str] = None
    message: Optional[str] = None
    rule_key: Optional[str] = None
    review: str = "human"
    modified_args: Optional[Dict[str, Any]] = None


def set_thread_tool_whitelist(
    allowed: Optional[Set[str]],
    deny_msg_fmt: str = "Tool '{tool_name}' denied: not in this thread's tool whitelist",
) -> None:
    _thread_tool_whitelist.allowed = allowed
    _thread_tool_whitelist.fmt = deny_msg_fmt


def clear_thread_tool_whitelist() -> None:
    _thread_tool_whitelist.allowed = None


def _get_pre_tool_call_directive_details(
    tool_name: str, args: Optional[Dict[str, Any]], task_id: str = "", session_id: str = "",
    tool_call_id: str = "", turn_id: str = "", api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> _PreToolCallDirective:
    """Check ``pre_tool_call`` hooks for ``{"action": "block", "message"}`` (veto; message becomes
    the tool result) or ``{"action": "approve", "message", "rule_key"?, "review"?}`` (escalate ANY
    tool to the human-approval gate; ``rule_key`` picks the ``[a]lways`` allowlist grain;
    ``review: "smart"`` lets the smart-approval guardian answer first, default ``"human"``). First
    valid directive wins; irrelevant returns are ignored."""
    allowed = getattr(_thread_tool_whitelist, "allowed", None)
    if allowed is not None and tool_name not in allowed:
        fmt = getattr(_thread_tool_whitelist, "fmt", "Tool '{tool_name}' denied")
        return _PreToolCallDirective(action="block", message=fmt.format(tool_name=tool_name))
    from hermes_cli.lifecycle import invoke_hook as invoke_lifecycle_hook
    hook_results = invoke_lifecycle_hook(
        "pre_tool_call", tool_name=tool_name, args=args if isinstance(args, dict) else {},
        task_id=task_id, session_id=session_id, tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=list(middleware_trace or []),
    )
    modified_args: Optional[Dict[str, Any]] = None
    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        # "modify" — transform tool_input before dispatch. Processed before the block/approve gate
        # so modify directives are visible even when a later hook blocks. Each modify directive
        # shallow-merges its keys into one accumulated dict built from the original args.
        if action == "modify":
            partial = result.get("args")
            if isinstance(partial, dict) and partial:
                modified_args = {**(modified_args if modified_args is not None else
                                    (args if isinstance(args, dict) else {})), **partial}
            continue
        if action not in ("block", "approve"):
            continue
        message = result.get("message")
        message = message if isinstance(message, str) and message else None
        # A block directive requires a message (it becomes the tool result); approve's is optional.
        if action == "block" and not message:
            continue
        rule_key = result.get("rule_key") if action == "approve" else None
        rule_key = (rule_key.strip() or None) if isinstance(rule_key, str) else None
        review = result.get("review") if action == "approve" else None
        review = review.strip().lower() if isinstance(review, str) else ""
        return _PreToolCallDirective(action=action, message=message, rule_key=rule_key,
                                     review="smart" if review == "smart" else "human",
                                     modified_args=modified_args)
    return _PreToolCallDirective(modified_args=modified_args)


def get_pre_tool_call_directive(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> tuple[Optional[str], Optional[str]]:
    """Back-compat: ``(directive, message)`` with directive ``"block"`` / ``"approve"`` / ``None``.
    ``hook_kwargs`` are the observability ids of :func:`_get_pre_tool_call_directive_details`."""
    details = _get_pre_tool_call_directive_details(tool_name, args, **hook_kwargs)
    return (details.action, details.message)


def get_pre_tool_call_block_message(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Optional[str]:
    """Deprecated shim: only the ``block`` message (or ``None``); ``approve`` is invisible here."""
    directive, message = get_pre_tool_call_directive(tool_name, args, **hook_kwargs)
    return message if directive == "block" else None


def resolve_pre_tool_block(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Optional[str]:
    """Resolve the pre_tool_call directive to a final block message (or ``None`` to proceed),
    running the human-approval gate for ``approve``. See :func:`_resolve_block_from_details`."""
    return _dispatch_pre_tool_call_hooks(tool_name, args, **hook_kwargs)[0]


def _resolve_block_from_details(
    details: "_PreToolCallDirective", tool_name: str, *, turn_id: str = "", tool_call_id: str = "",
    session_id: str = "",
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
        approval_tokens = None
        with suppress(Exception):
            approval_tokens = set_current_observability_context(
                turn_id=turn_id, tool_call_id=tool_call_id, session_id=session_id)
        try:
            result = request_tool_approval(tool_name, details.message or "", rule_key=details.rule_key or tool_name,
                                           review=details.review)
        finally:
            if approval_tokens is not None:
                with suppress(Exception):
                    reset_current_observability_context(approval_tokens)
    except Exception:
        # Fail-closed: if the gate itself errors, block rather than silently execute an action a
        # plugin flagged for approval.
        return f"BLOCKED: plugin approval gate failed for {tool_name}"
    if not result.get("approved"):
        return str(result.get("message") or f"BLOCKED: plugin approval required for {tool_name}")
    return None


def _dispatch_pre_tool_call_hooks(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Invoke ``pre_tool_call`` hooks once; return ``(block_message, modified_args)`` — the resolved
    block/approve message (``None`` to proceed) and merged ``modify`` args (``None`` if none)."""
    details = _get_pre_tool_call_directive_details(tool_name, args, **hook_kwargs)
    block_msg = _resolve_block_from_details(
        details, tool_name, **{k: hook_kwargs.get(k, "") for k in ("turn_id", "tool_call_id", "session_id")})
    return (block_msg, details.modified_args)
