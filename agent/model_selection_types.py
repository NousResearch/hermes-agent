"""Typed contracts for guided model routing (design doc §3-4, §12).

Pure data shapes shared by the selector and its adapters. No I/O, no provider
clients, no credentials. See /Users/jhaynes/.hermes/plans/2026-09-15_141016-guided-model-routing.md
for the binding design this implements.
"""
from __future__ import annotations

REQUIRED_ROUTE_FIELDS = (
    "route_id", "route_revision", "provider", "model", "endpoint", "maker",
    "model_family", "status", "allowed_roles", "capabilities",
    "verified_input_budget", "allowed_reasoning", "qualifications",
    "assessment", "evidence",
)

REQUIRED_REQUIREMENT_FIELDS = (
    "schema_version", "role", "execution_kind", "execution_id", "attempt_id",
    "task_class", "required_capabilities", "input_tokens", "reserve_tokens",
    "reasoning", "provenance",
)

REQUIRED_PROVENANCE_FIELDS = ("frozen_sha", "verified_by", "complete", "contributors")

TASK_CLASSES = ("established-pattern", "investigative", "cross-component", "high-consequence")

# High-consequence task classes force the "deep" quality floor regardless of
# any model-proposed classification (design §3.B, §4 step 2).
DEEP_QUALITY_TASK_CLASSES = ("cross-component", "high-consequence")

ROUTE_STATUSES = ("candidate", "approved", "suspended", "retired")


class RoutingBlocked(Exception):
    """Raised when no candidate route can satisfy the requirements.

    `reason` is one of the typed reason codes from design §4:
    schema_invalid, provenance_incomplete, no_role_match, independence_unavailable,
    input_too_large, reasoning_unsupported, provider_unavailable,
    unsupported_executor, stale_or_revoked_decision, no_qualified_route.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)
