"""Host policy layered over the plugin runtime.

This module owns CLI/application decisions made from plugin hook results. It depends downward on
plugin_runtime; plugin_runtime must never depend back on this module.
"""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from plugin_runtime.api import invoke_hook
from plugin_runtime.lifecycle import get_plugin_manager

logger = logging.getLogger("hermes_cli.plugins")
_thread_tool_whitelist = threading.local()


def fire_pre_command_hook(
    *, surface: str, command: str, alias_used: str, args_raw: str,
    session_key: Optional[str] = None, platform: Optional[str] = None,
) -> None:
    """Fire the observer-only ``pre_command`` hook; never raises. Directive-shaped returns are
    logged at debug so future block/rewrite adopters are discoverable."""
    try:
        manager = get_plugin_manager()
        if not manager.has_hook("pre_command"):
            return
        results = manager.invoke_hook(
            "pre_command", surface=surface, command=command, alias_used=alias_used,
            args_raw=args_raw, session_key=session_key, platform=platform,
        )
        for result in results:
            if isinstance(result, dict) and ("action" in result or "decision" in result):
                logger.debug("pre_command is observer-only in v1: ignoring directive %r for /%s (surface=%s). "
                             "Block/rewrite will arrive with the command middleware variant (#64204/#64231).",
                             result, command, surface)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("pre_command hook dispatch failed (non-fatal): %s", exc)

@dataclass(frozen=True)
class _PreToolCallDirective:
    action: Optional[str] = None
    message: Optional[str] = None
    rule_key: Optional[str] = None
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
    the tool result) or ``{"action": "approve", "message", "rule_key"?}`` (escalate ANY tool to the
    human-approval gate; ``rule_key`` picks the ``[a]lways`` allowlist grain). Precedence is
    ``block`` > ``approve`` > none, not registration order: any plugin's valid veto wins over an
    earlier plugin's request for human confirmation (#87420); among approves the first valid one
    wins. Irrelevant returns are ignored."""
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
    first_approve: Optional[Tuple[Optional[str], Optional[str]]] = None  # (message, rule_key)
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
        if action == "block":
            return _PreToolCallDirective(action="block", message=message, modified_args=modified_args)
        # approve is held back until the whole list has been scanned for a veto.
        if first_approve is None:
            rule_key = result.get("rule_key")
            first_approve = (message, (rule_key.strip() or None) if isinstance(rule_key, str) else None)
    if first_approve is not None:
        return _PreToolCallDirective(action="approve", message=first_approve[0], rule_key=first_approve[1],
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
            result = request_tool_approval(tool_name, details.message or "", rule_key=details.rule_key or tool_name)
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

def get_pre_verify_continue_message(
    *, session_id: str = "", platform: str = "", model: str = "", coding: bool = False,
    attempt: int = 0, final_response: str = "", changed_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Check ``pre_verify`` hooks for ``{"action": "continue", "message"}`` (or Claude-Code Stop
    ``{"decision": "block", "reason"}``) to keep the turn going; first non-empty message wins, any
    other return lets the turn finish. ``coding``/``attempt`` let hooks scope and self-throttle."""
    hook_results = invoke_hook(
        "pre_verify", session_id=session_id, platform=platform, model=model, coding=coding,
        attempt=attempt, final_response=final_response, changed_paths=list(changed_paths or []),
    )
    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or result.get("decision") or "").strip().lower()
        message = result.get("message") or result.get("reason")
        if action in ("continue", "block") and isinstance(message, str) and message.strip():
            return message.strip()
    return None

def get_plugin_error_classification(
    *, provider: str = "", model: str = "", status_code: Optional[int] = None, error_type: str = "",
    error_code: str = "", error_message: str = "", error_body: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None, approx_tokens: int = 0, context_length: int = 0,
    num_messages: int = 0,
) -> Optional[Dict[str, Any]]:
    """Consult ``transform_api_error_classification`` hooks BEFORE the built-in classifier.
    Run-all-then-pick-first: the first valid result in registration order wins, losing valid results
    warn (conflicts visible, not shadowed). Returns a sanitized dict (``reason`` -> ``FailoverReason``,
    hint flags -> bool, ``message`` capped at 500) or ``None``. Privacy: inputs may be unredacted.

    A callback returns ``None`` to decline, or a dict with a required ``"reason"`` (a
    :class:`agent.error_classifier.FailoverReason` member or its string name) plus optional recovery-hint
    overrides. Dispatch is run-all-then-pick-first: ``invoke_hook`` runs every registered callback with
    failures isolated, then the first result carrying a valid reason wins in registration order — mirroring
    :func:`get_pre_tool_call_block_message`, invalid or irrelevant returns are silently ignored so a
    misbehaving plugin degrades to a no-op. When more than one callback returns a valid classification, the
    losing results are skipped with a runtime warning (the #64714 skipped-transform rule) so conflicting
    provider plugins are visible in logs instead of silently shadowed.
    """
    from agent.error_classifier import FailoverReason
    hook_results = invoke_hook(
        "transform_api_error_classification", provider=provider, model=model,
        status_code=status_code, error_type=error_type, error_code=error_code,
        error_message=error_message, error_body=error_body if isinstance(error_body, dict) else {},
        error=error, approx_tokens=approx_tokens, context_length=context_length,
        num_messages=num_messages,
    )

    def _reason(result: Any) -> Any:
        reason = result.get("reason") if isinstance(result, dict) else None
        if isinstance(reason, str):
            with suppress(ValueError):
                return FailoverReason(reason.strip().lower())
            return None
        return reason if isinstance(reason, FailoverReason) else None

    valid = [(result, reason) for result in hook_results if (reason := _reason(result)) is not None]
    if not valid:
        return None
    result, reason = valid[0]
    winner: Dict[str, Any] = {"reason": reason}
    for key in ("retryable", "should_compress", "should_rotate_credential", "should_fallback"):
        if key in result:
            winner[key] = bool(result[key])
    message = result.get("message")
    if isinstance(message, str) and message.strip():
        winner["message"] = message.strip()[:500]
    if isinstance(result.get("error_context"), dict):
        winner["error_context"] = result["error_context"]
    if len(valid) > 1:
        logger.warning("transform_api_error_classification: skipped %d valid classification(s) after the "
                       "first result in registration order won (run-all-then-pick-first)", len(valid) - 1)
    return winner
