"""Loop escalation — port of pinning/loop-escalation.ts.

Detects bounded repeated tool failures and signals session escalation to a
frontier-capable tier. Escalation fires once per session (no tier
oscillation). Economical pins count consecutive failures of ANY signature
(a session failing with varied errors is just as stuck); other pins count
identical failure signatures only (FR-014).

Pure evaluation: the caller persists pin state via the state store.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional

from .types import (
    ModelProfile,
    PIN_REASON_LOOP_ESCALATION,
    RoutingRequest,
    SessionPin,
    TIER_ECONOMICAL,
    TIER_FRONTIER,
    TIER_LOCAL,
    TURN_TOOL_RESULT,
)

_FAILURE_PATTERNS = (
    "error", "fail", "exception", "timed out", "timeout",
    "econnrefused", "enotfound", "econnreset", "epipe",
)
_RATE_LIMIT_PATTERNS = ("rate limit", "429", "too many requests", "quota exceeded")
_AUTH_DENIED_PATTERNS = ("permission denied", "access denied")
_UNSUPPORTED_TOOL_PATTERNS = (
    "unknown tool", "unsupported tool", "tool not found", "no such tool",
    "unrecognized tool", "is not a valid tool", "tool does not exist",
    "not a known tool",
)

ZERO_TIER_TOOL_CHURN_SIGNATURE = "zt:tool_churn"


@dataclass(frozen=True)
class LoopEscalationResult:
    should_escalate: bool
    updated_pin: Optional[SessionPin]
    escalation_target: Optional[ModelProfile]
    reason: str


def _last_tool_message(request: RoutingRequest):
    messages = request.messages
    if not messages:
        return None
    last = messages[-1]
    return last if getattr(last, "role", None) == "tool" else None


def _looks_like_failure(content: str) -> bool:
    lower = content.lower()
    if "no error" in lower or "no errors" in lower or "no failure" in lower or "no failures" in lower:
        return False
    if "without error" in lower or "without an error" in lower or "without failure" in lower or "without a failure" in lower:
        return False
    return any(p in lower for p in _FAILURE_PATTERNS + _RATE_LIMIT_PATTERNS + _AUTH_DENIED_PATTERNS)


def _is_tool_failure(msg) -> bool:
    if getattr(msg, "is_error", None) is True:
        return True
    if getattr(msg, "is_error", None) is False:
        return False
    status = getattr(msg, "status", None)
    if status is not None:
        return status >= 400
    return _looks_like_failure(getattr(msg, "content", "") or "")


def _compute_signature(content: str) -> str:
    """Deterministic djb2 hash of normalised error content."""
    normalized = content.strip()[:256].lower()
    h = 5381
    for ch in normalized:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return f"tf:{h:x}"


def extract_tool_failure_signature(request: RoutingRequest) -> Optional[str]:
    """Extract a tool-failure signature, or None when no failure is present."""
    msg = _last_tool_message(request)
    if msg is None:
        return None
    if _is_tool_failure(msg):
        return _compute_signature(getattr(msg, "content", "") or "")
    return None


def is_unsupported_or_unknown_tool_result(request: RoutingRequest) -> bool:
    msg = _last_tool_message(request)
    if msg is None:
        return False
    lower = (getattr(msg, "content", "") or "").lower()
    return any(p in lower for p in _UNSUPPORTED_TOOL_PATTERNS)


def _select_escalation_target(fleet, current_model_id: str) -> Optional[ModelProfile]:
    """Best healthy frontier model that differs from the current pin."""
    frontier = [
        m for m in fleet
        if m.tier == TIER_FRONTIER and m.id != current_model_id and m.healthy
    ]
    if not frontier:
        return None
    return sorted(frontier, key=lambda m: (-m.quality, m.id))[0]


def _pinned_tier(pin: SessionPin, fleet) -> Optional[str]:
    for m in fleet:
        if m.id == pin.pinned_model_id:
            return m.tier
    return None


def _no_escalation(reason: str) -> LoopEscalationResult:
    return LoopEscalationResult(False, None, None, reason)


def _try_escalate(pin: SessionPin, fleet, updated: SessionPin, reason: str) -> LoopEscalationResult:
    target = _select_escalation_target(fleet, pin.pinned_model_id)
    if target:
        return LoopEscalationResult(True, updated, target, reason)
    return LoopEscalationResult(False, updated, None, "no_frontier_available")


def evaluate_loop_escalation(
    pin: Optional[SessionPin],
    request: RoutingRequest,
    fleet,
    threshold: int = 3,
) -> LoopEscalationResult:
    """Evaluate whether the session should escalate to a higher tier."""
    if pin is None:
        return _no_escalation("no_pin")
    if pin.pin_reason == PIN_REASON_LOOP_ESCALATION:
        return _no_escalation("already_escalated")
    if request.turn_type != TURN_TOOL_RESULT:
        return _no_escalation("not_tool_result")

    now = time.time()

    # Local (zero-tier) pin: unsupported tool → immediate escalate; else count
    # every tool_result turn as observational churn.
    if _pinned_tier(pin, fleet) == TIER_LOCAL:
        if is_unsupported_or_unknown_tool_result(request):
            updated = replace(
                pin,
                consecutive_tool_failures=max(pin.consecutive_tool_failures, 1),
                last_tool_failure_signature="zt:unsupported_tool",
                updated_at=now,
            )
            return _try_escalate(pin, fleet, updated, "zero_tier_unsupported_tool")
        is_churn = pin.last_tool_failure_signature == ZERO_TIER_TOOL_CHURN_SIGNATURE
        new_count = pin.consecutive_tool_failures + 1 if is_churn else 1
        updated = replace(
            pin,
            consecutive_tool_failures=new_count,
            last_tool_failure_signature=ZERO_TIER_TOOL_CHURN_SIGNATURE,
            updated_at=now,
        )
        if new_count >= threshold:
            return _try_escalate(pin, fleet, updated, "zero_tier_tool_churn")
        return LoopEscalationResult(False, updated, None, "zero_tier_below_threshold")

    signature = extract_tool_failure_signature(request)
    if not signature:
        if pin.consecutive_tool_failures > 0:
            return LoopEscalationResult(
                False,
                replace(pin, consecutive_tool_failures=0, last_tool_failure_signature=None, updated_at=now),
                None,
                "success_reset",
            )
        return _no_escalation("no_failure")

    is_economical = _pinned_tier(pin, fleet) == TIER_ECONOMICAL
    is_identical = pin.last_tool_failure_signature == signature
    new_count = pin.consecutive_tool_failures + 1 if (is_economical or is_identical) else 1
    updated = replace(
        pin,
        consecutive_tool_failures=new_count,
        last_tool_failure_signature=signature,
        updated_at=now,
    )
    if new_count >= threshold:
        return _try_escalate(pin, fleet, updated, "threshold_exceeded")
    return LoopEscalationResult(False, updated, None, "below_threshold")
