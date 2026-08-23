"""Pure compute-class routing, downstream of the existing role router."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from hermes_constants import VALID_REASONING_EFFORTS

COMPUTE_CLASSES = ("quick", "standard", "deep", "architect", "specialist")
REASONING_EFFORTS = ("none", *VALID_REASONING_EFFORTS)
ROLE_BOUNDARY_FIELDS = (
    "role",
    "assignee",
    "write_scope",
    "workspace",
    "approval",
    "toolsets",
    "master_forbidden",
)
WORK_FAILURES = {"test_failed", "contract_failed", "invalid_output"}


def _decision_id(
    task_context, role_decision, capability_snapshot, policy, compute_class
):
    identity = {
        "task_id": task_context.get("task_id"),
        "role_ref": role_decision.get("role_ref"),
        "snapshot_id": capability_snapshot.get("snapshot_id"),
        "policy_version": policy.get("policy_version"),
        "compute_class": compute_class,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"route-{digest}"


def _task_contract(role_decision: Mapping[str, Any]) -> dict[str, Any]:
    return {field: deepcopy(role_decision.get(field)) for field in ROLE_BOUNDARY_FIELDS}


def _boundary_changed(proposed: object, baseline: Mapping[str, Any]) -> bool:
    if not isinstance(proposed, Mapping):
        return False
    return any(
        field in proposed and proposed[field] != baseline.get(field)
        for field in ROLE_BOUNDARY_FIELDS
    )


def _classify(task_context: Mapping[str, Any], policy: Mapping[str, Any]) -> str | None:
    signals = {str(item).strip().lower() for item in task_context.get("signals", ())}
    precedence = policy.get("precedence") or (
        "specialist",
        "architect",
        "deep",
        "standard",
        "quick",
    )
    return next(
        (
            str(item).lower()
            for item in precedence
            if str(item).lower() in signals and str(item).lower() in COMPUTE_CLASSES
        ),
        None,
    )


def _actual_route(execution: Mapping[str, Any]) -> dict[str, Any] | None:
    actual = execution.get("actual")
    keys = ("provider", "model", "reasoning_effort")
    if not isinstance(actual, Mapping) or not all(actual.get(key) for key in keys):
        return None
    return {key: actual[key] for key in keys}


def route_task(
    task_context: Mapping[str, Any],
    role_decision: Mapping[str, Any],
    capability_snapshot: Mapping[str, Any],
    role_constraints: Mapping[str, Any],
    policy: Mapping[str, Any],
    execution: Mapping[str, Any],
    unattended: bool,
) -> dict[str, Any]:
    """Resolve a route without mutating inputs or dispatching work."""
    task_contract = _task_contract(role_decision)

    def result(status: str, **fields: Any) -> dict[str, Any]:
        return {"status": status, "task_contract": task_contract, **fields}

    if _boundary_changed(task_context.get("proposed_boundary"), role_decision):
        return result("rejected", spawn=False, reason_code="role_boundary_mutation")

    requested = task_context.get("requested_route")
    if requested is not None and not isinstance(requested, Mapping):
        return result("rejected", spawn=False, reason_code="invalid_route")
    requested = requested or {}
    if _boundary_changed(requested, role_decision) or _boundary_changed(
        requested, role_constraints
    ):
        return result("rejected", spawn=False, reason_code="safety_boundary_mutation")
    if requested.get("provider") and not requested.get("model"):
        return result("rejected", spawn=False, reason_code="provider_requires_model")

    requested_effort = requested.get("reasoning_effort")
    if requested_effort is not None:
        requested_effort = str(requested_effort).strip().lower()
        if requested_effort not in REASONING_EFFORTS:
            return result(
                "rejected", spawn=False, reason_code="invalid_reasoning_effort"
            )

    compute_class = _classify(task_context, policy)
    if compute_class is None:
        return result("rejected", spawn=False, reason_code="compute_class_ambiguous")
    policy_classes = policy.get("classes")
    class_policy = (
        policy_classes.get(compute_class)
        if isinstance(policy_classes, Mapping)
        else None
    )
    if not isinstance(class_policy, Mapping):
        return result("rejected", spawn=False, reason_code="route_policy_incomplete")

    persisted_input = task_context.get("persisted_route")
    if persisted_input is not None:
        required = ("compute_class", "route_decision_id", "policy_version")
        if not isinstance(persisted_input, Mapping) or not all(
            persisted_input.get(key) for key in required
        ):
            fields: dict[str, Any] = {
                "spawn": False,
                "reason_code": "compute_class_not_persisted",
            }
            actual = _actual_route(execution)
            if actual is not None:
                fields["actual_route"] = actual
            if execution.get("reported_done"):
                fields["outcome"] = "reported_done"
            return result("rejected", **fields)

    route = {
        "provider": requested.get("provider", class_policy.get("provider")),
        "model": requested.get("model", class_policy.get("model")),
        "reasoning_effort": requested_effort or class_policy.get("reasoning_effort"),
        "max_fallbacks": min(max(int(class_policy.get("max_fallbacks", 0)), 0), 1),
    }
    if route["provider"] and not route["model"]:
        return result("rejected", spawn=False, reason_code="provider_requires_model")
    if route["reasoning_effort"] not in REASONING_EFFORTS:
        return result("rejected", spawn=False, reason_code="invalid_reasoning_effort")

    allowed_providers = role_constraints.get("allowed_providers")
    allowed_models = role_constraints.get("allowed_models")
    if (
        allowed_providers is not None and route["provider"] not in allowed_providers
    ) or (allowed_models is not None and route["model"] not in allowed_models):
        return result("rejected", spawn=False, reason_code="no_eligible_candidate")

    route_decision_id = _decision_id(
        task_context, role_decision, capability_snapshot, policy, compute_class
    )
    persisted_route = {
        "compute_class": compute_class,
        "route_decision_id": route_decision_id,
        "policy_version": policy.get("policy_version"),
    }
    common: dict[str, Any] = {
        "compute_class": compute_class,
        "route": route,
        "persisted_route": persisted_route,
        "route_decision_id": route_decision_id,
        "policy_version": policy.get("policy_version"),
        "verification_required": True,
    }

    if not (
        capability_snapshot.get("catalog_observed")
        and capability_snapshot.get("auth_observed")
        and capability_snapshot.get("runtime_observed")
    ):
        return result(
            "prepared_not_observed",
            **common,
            spawn=False,
            reason_code="capability_unverified",
        )

    candidates = capability_snapshot.get("candidates") or ()
    candidate = next(
        (
            item
            for item in candidates
            if isinstance(item, Mapping)
            and item.get("provider") == route["provider"]
            and item.get("model") == route["model"]
        ),
        None,
    )
    if candidate is None or route["reasoning_effort"] not in candidate.get(
        "reasoning_efforts", ()
    ):
        return result(
            "prepared_not_observed",
            **common,
            spawn=False,
            reason_code="no_eligible_candidate",
        )

    if unattended and class_policy.get("automation") == "attended":
        return result(
            "approval_required",
            **common,
            spawn=False,
            reason_code="attended_routing_required",
        )

    fallback_index = min(max(int(execution.get("fallback_index", 0)), 0), 1)
    error = execution.get("error")
    eligible_errors = set(policy.get("eligible_fallback_reasons") or ())
    ineligible_errors = set(policy.get("ineligible_fallback_reasons") or ())

    if error in WORK_FAILURES:
        return result(
            "rework_required",
            **common,
            spawn=False,
            outcome="work_failed",
            event_kind="verification_failed",
            fallback_index=fallback_index,
            retry_policy="same_owner",
        )
    if error in eligible_errors:
        if fallback_index < route["max_fallbacks"]:
            return result(
                "fallback_pending",
                **common,
                spawn=True,
                outcome="routing_unavailable",
                event_kind="fallback_started",
                fallback_index=fallback_index + 1,
            )
        return result(
            "blocked",
            **common,
            spawn=False,
            outcome="routing_unavailable",
            event_kind="fallback_exhausted",
            fallback_index=fallback_index,
        )
    if error in ineligible_errors:
        return result(
            "blocked",
            **common,
            spawn=False,
            outcome="blocked",
            event_kind="route_blocked",
            fallback_index=fallback_index,
            retry_policy="none",
        )
    if error:
        return result(
            "blocked",
            **common,
            spawn=False,
            outcome="work_failed",
            event_kind="route_blocked",
            fallback_index=fallback_index,
            reason_code="unclassified_execution_error",
        )

    actual = _actual_route(execution)
    if execution.get("reported_done"):
        if actual is None:
            return result(
                "observation_incomplete",
                **common,
                spawn=False,
                outcome="reported_done",
                reason_code="actual_route_missing",
            )
        if any(actual[key] != route[key] for key in actual):
            return result(
                "observation_mismatch",
                **common,
                actual_route=actual,
                spawn=False,
                outcome="reported_done",
                reason_code="actual_route_mismatch",
            )
        verification = execution.get("verification") or {}
        if not (verification.get("performed") and verification.get("passed")):
            return result(
                "verification_required",
                **common,
                actual_route=actual,
                spawn=False,
                outcome="reported_done",
            )
        return result(
            "verified",
            **common,
            actual_route=actual,
            spawn=False,
            outcome="verified",
            event_kind="verification_passed",
        )

    if execution.get("attempted"):
        fields = {**common, "spawn": False, "fallback_index": fallback_index}
        if actual is not None:
            fields["actual_route"] = actual
        return result("running", **fields)

    if str(policy.get("mode", "shadow")).strip().lower() == "shadow":
        return result(
            "shadow_recorded",
            **common,
            spawn=True,
            apply_route_override=False,
            dispatch_overrides={},
            event_kind="route_resolved",
        )
    return result(
        "dispatchable",
        **common,
        spawn=True,
        apply_route_override=True,
        dispatch_overrides={
            "model_override": route["model"],
            "provider_override": route["provider"],
            "reasoning_effort": route["reasoning_effort"],
        },
        event_kind="route_resolved",
    )


__all__ = ["COMPUTE_CLASSES", "route_task"]
