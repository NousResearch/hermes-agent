"""
example_policy_guardrail.py

Reference implementation of a tool-call policy guardrail that integrates with
the ToolCallGuardrailController.register_policy_hook() API.

Demonstrates a 3-tier risk classification (READ / WRITE / PRIVILEGED) that
matches the well-established principle of least authority for browser and
side-effecting tool calls. Project-specific deployments can adapt the
risk table to their own threat model.

Hook signature
--------------
    def policy_hook(tool_name: str, args: dict | None) -> ToolGuardrailDecision | None

Returning None means "no opinion, fall through to the built-in guardrail logic."
Returning a ToolGuardrailDecision short-circuits the chain.

Registration
------------
    from agent.tool_guardrails import ToolCallGuardrailController
    from agent.policy_hooks.example_policy_guardrail import risk_policy_guardrail

    controller.register_policy_hook(risk_policy_guardrail)

Or, via config (see agent/policy_hooks/__init__.py for the import-path format):
    tool_loop_guardrails:
      policy_hooks:
        - "agent.policy_hooks.example_policy_guardrail:risk_policy_guardrail"
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from agent.tool_guardrails import ToolCallSignature, ToolGuardrailDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk classification — adapt to your deployment's threat model
# ---------------------------------------------------------------------------

RISK_TABLE: dict[str, str] = {
    # READ — no side effects, no approval needed by default
    "browser_navigate": "R",
    "browser_snapshot": "R",
    "browser_scroll": "R",
    "browser_back": "R",
    "browser_get_images": "R",
    "web_search": "R",
    # WRITE — mutates remote state; needs explicit allow-list or policy check
    "browser_click": "W",
    "browser_type": "W",
    "browser_press": "W",
    "browser_vision": "W",
    # PRIVILEGED — high-blast-radius; require operator approval (e.g. 888_HOLD)
    "browser_cdp": "P",
    "browser_dialog": "P",
    "browser_console": "P",
}


def _operator_approval_required(risk: str, tool_name: str, args: Mapping[str, Any]) -> ToolGuardrailDecision:
    """Block the call and surface a verdict that downstream code can route to a human."""
    return ToolGuardrailDecision(
        action="block",
        code="POLICY_OPERATOR_APPROVAL_REQUIRED",
        message=(
            f"Blocked by policy: {tool_name} is risk class {risk}. "
            "Operator approval required before execution."
        ),
        tool_name=tool_name,
    )


def risk_policy_guardrail(
    tool_name: str,
    args: dict[str, Any] | None,
) -> ToolGuardrailDecision | None:
    """Policy hook implementing the 3-tier risk table above.

    Returns None for tool calls that are not in the risk table so the built-in
    guardrail logic can take over. Returns a ToolGuardrailDecision to block.
    """
    args = args or {}
    risk = RISK_TABLE.get(tool_name)
    if risk is None:
        # Not a classified tool — let the chain continue.
        return None

    if risk == "R":
        # Read-only — pass through, do not consume the policy slot.
        return None

    if risk in ("W", "P"):
        # Default-deny for write/privileged actions. Projects should replace
        # this with their own approval workflow (allow-list, signed token,
        # operator confirmation, etc.).
        logger.info(
            "risk_policy_guardrail: %s class %s blocked — awaiting operator approval",
            tool_name, risk,
        )
        return _operator_approval_required(risk, tool_name, args)

    return None