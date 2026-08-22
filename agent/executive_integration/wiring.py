"""Conversation-loop wiring for Executive Integration Layer.

This module is a **read-only** hook that consults the EIL
before the LLM call in `conversation_loop.run_conversation()`.

Cardinal rules (HARD, never overridden):
  * NO LLM invocation (we never call the LLM).
  * NO provider call (no openai, anthropic, litellm, etc.).
  * NO network access (no requests, urllib, httpx, aiohttp).
  * NO worker invocation (no delegate_task, kanban.create).
  * NO subprocess (no subprocess.run, os.system, os.popen).
  * NO Kanban DB mutation.
  * NO GBrain / Obsidian / NotebookLM calls.
  * NO gateway restart.
  * NO R7 / Hermes artifact modification.
  * NO self-improvement activation.
  * NO DB writes (stateless).
  * NO new tables.
  * NO commit / push / PR.

Default-off: all 3 flags (HERMES_EXECUTIVE_INTEGRATION_ENABLED,
HERMES_OBJECTIVE_GATEWAY_ENABLED, HERMES_EXECUTIVE_AUTOLAUNCH_ENABLED)
must be set to 1 by the operator to activate this hook.

Fail-open: if the hook raises an exception, the caller
(``run_conversation()``) catches it and continues normally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .objective_gateway import _flags_enabled
from .result_adapter import ExecutiveResultAdapter
from .types import (
    ExecutiveLaunchRequest,
    ExecutiveUserSummary,
    ObjectiveGatewayDecision,
    LaunchStatus,
    RouteKind,
    _now_iso8601,
)
from .launcher import ExecutiveLauncher


# Public API (re-exported from the package).
__all__ = [
    "maybe_route_with_executive_integration",
    "_build_eil_response",
    "_get_available_tool_names",
    "EIL_BYPASS_REASON_DEFAULT_OFF",
    "EIL_BYPASS_REASON_INTEGRATION_DISABLED",
    "EIL_BYPASS_REASON_GATEWAY_DISABLED",
]


# ──────────────────────────────────────────────────────────────────────
# Bypass reasons (informational; no telemetry)
# ──────────────────────────────────────────────────────────────────────


EIL_BYPASS_REASON_DEFAULT_OFF = "EIL default-off (HERMES_EXECUTIVE_INTEGRATION_ENABLED=0)"
EIL_BYPASS_REASON_INTEGRATION_DISABLED = "EIL disabled (HERMES_EXECUTIVE_INTEGRATION_ENABLED=0)"
EIL_BYPASS_REASON_GATEWAY_DISABLED = (
    "EIL gateway disabled (HERMES_OBJECTIVE_GATEWAY_ENABLED=0); "
    "EXECUTIVE route unavailable"
)


# ──────────────────────────────────────────────────────────────────────
# Public hook
# ──────────────────────────────────────────────────────────────────────


def maybe_route_with_executive_integration(
    agent: Any,
    user_message: str,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[ObjectiveGatewayDecision]:
    """Read-only hook that consults the EIL before the LLM call.

    Returns:
        None if the EIL is disabled or the route is CHAT/TOOL.
        ObjectiveGatewayDecision if the route is EXECUTIVE/CLARIFY/REJECT.
    """
    flags = _flags_enabled()
    if not flags["integration_enabled"]:
        return None

    # Lazy import (avoids circular imports and minimizes import cost).
    from .router import ExecutiveIntegrationRouter

    # Build the router on-demand (the agent's intent_router and
    # policy_engine are passed in read-only).
    intent_router = getattr(agent, "intent_router", None)
    policy_engine = getattr(agent, "policy_engine", None)
    router = ExecutiveIntegrationRouter(
        intent_router=intent_router,
        policy_engine=policy_engine,
    )
    decision = router.route(user_message, context=context)

    # CHAT and TOOL routes are passthrough; let the normal flow continue.
    if decision.route_kind in (RouteKind.CHAT, RouteKind.TOOL):
        return None

    # EXECUTIVE, CLARIFY, REJECT routes are short-circuits.
    # Note: the EIL only produces EXECUTIVE if HERMES_OBJECTIVE_GATEWAY_ENABLED=1;
    # the ObjectiveGateway already enforces this internally.
    return decision


# ──────────────────────────────────────────────────────────────────────
# Response builder
# ──────────────────────────────────────────────────────────────────────


def _build_eil_response(
    decision: ObjectiveGatewayDecision,
    agent: Any,
    *,
    persist_user_message: Optional[str] = None,
    user_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a deterministic response from an EIL decision.

    Does NOT call the LLM. Does NOT modify the messages list.
    Does NOT write to state.db or state_meta.

    Returns a dict in the same shape as
    ``run_conversation()``'s return value, but with
    ``ei_routed=True`` and ``api_call_count=0``.
    """
    route_kind = decision.route_kind
    raw_user_message = user_message if user_message is not None else persist_user_message
    user_facing_text, next_steps, warnings, launch_edge = _text_for_route(
        decision,
        agent,
        user_message=raw_user_message,
    )

    response = {
        "content": user_facing_text,
        "role": "assistant",
        "finish_reason": "stop",
        "ei_routed": True,
        "ei_route_kind": route_kind.value,
        "ei_decision_fingerprint": decision.fingerprint,
        "ei_rationale": decision.rationale,
        "ei_next_steps": list(next_steps),
        "ei_warnings": list(warnings),
        "api_call_count": 0,
        # The caller (run_conversation) is expected to NOT mutate
        # the messages list. We do not include a `messages` key.
    }
    if launch_edge:
        response["ei_launch_edge"] = launch_edge
    return response


def _text_for_route(
    decision: ObjectiveGatewayDecision,
    agent: Any,
    *,
    user_message: Optional[str] = None,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...], Optional[Dict[str, Any]]]:
    """Return (text, next_steps, warnings) for a given EIL decision."""
    adapter = ExecutiveResultAdapter()
    if decision.route_kind == RouteKind.EXECUTIVE:
        # Prepare an ExecutiveLaunchRequest. Raw message must remain payload;
        # the gateway fingerprint is only an identifier.
        launcher = ExecutiveLauncher()
        raw_user_message = user_message if user_message is not None else decision.fingerprint or ""
        request = launcher.prepare(
            user_message=raw_user_message,
            gateway_decision=decision,
        )
        if launcher.launch_edge_enabled() and not request.requires_human_approval:
            launcher.approve(request.request_id, approver_id="launch-edge-canary")
            request = launcher.launch(request.request_id)
            preview = launcher.get_launch_edge_result(request.request_id)
            if preview and request.status == LaunchStatus.EXECUTION_PREVIEW_READY:
                summary = adapter.adapt_preview_ready(request, preview=preview)
                return (
                    summary.body,
                    summary.next_steps,
                    summary.warnings,
                    preview,
                )
            summary = adapter.adapt_blocked(
                request,
                reason=request.risk_rationale,
            )
            return (
                summary.body,
                summary.next_steps,
                summary.warnings,
                None,
            )
        summary = adapter.adapt_pending(request)
        return (
            summary.body,
            summary.next_steps,
            summary.warnings,
            None,
        )
    elif decision.route_kind == RouteKind.CLARIFY:
        text = (
            "Your request needs more detail. "
            f"Rationale: {decision.rationale} "
            "Please provide a more specific question or request."
        )
        return (
            text,
            ("Provide more detail",),
            (),
            None,
        )
    elif decision.route_kind == RouteKind.REJECT:
        text = (
            "Your request was rejected. "
            f"Rationale: {decision.rationale}"
        )
        return (
            text,
            (),
            (),
            None,
        )
    else:  # CHAT or TOOL (should not happen; passthrough)
        text = (
            f"[EIL passthrough] route_kind={decision.route_kind.value}"
        )
        return (
            text,
            (),
            (),
            None,
        )


# ──────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────


def _get_available_tool_names(agent: Any) -> Tuple[str, ...]:
    """Read-only helper. Returns the list of available tool names.

    Used by the EIL for routing context (informational; not used
    for gating in the canary).
    """
    if agent is None:
        return ()
    tool_names = getattr(agent, "_available_tool_names", None)
    if tool_names is None:
        return ()
    return tuple(str(n) for n in tool_names)
