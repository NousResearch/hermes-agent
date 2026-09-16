"""Pure deterministic model-selection library (design §4, §12).

`select()` is intentionally pure: no I/O, no provider clients, no credentials,
no clock reads (caller passes `now`). It filters an approved-route policy
snapshot against structured task requirements and returns an immutable
decision dict, or raises RoutingBlocked with a typed reason code.

See /Users/jhaynes/.hermes/plans/2026-09-15_141016-guided-model-routing.md
sections 3-4 and 12 for the binding design. This module implements ONLY the
pure selection contract (implementation step 2); the common managed-route
guard, delegation/Kanban/MoA adapters and receipt persistence are separate,
not-yet-implemented steps (3-6) tracked in .hermes/implementation-status.md.
"""
from __future__ import annotations

from .model_selection_types import (
    DEEP_QUALITY_TASK_CLASSES,
    REQUIRED_PROVENANCE_FIELDS,
    REQUIRED_REQUIREMENT_FIELDS,
    REQUIRED_ROUTE_FIELDS,
    RoutingBlocked,
)

__all__ = ["select", "RoutingBlocked"]

_FRESHNESS_HEALTHY_SECONDS = 300  # design §12: 5 minutes for a successful route health check
_FRESHNESS_COOLDOWN_SECONDS = 60  # unclassified transient failure cooldown


def _quality_floor(task_class: str) -> str:
    """Deterministic quality floor: deep for cross-component/high-consequence.

    Unknown/unclassified scope also defaults to deep per design §3.B
    ("unclassified or incomplete scope defaults to deep").
    """
    if task_class in DEEP_QUALITY_TASK_CLASSES:
        return "deep"
    if task_class == "established-pattern":
        return "shallow"
    # investigative and any unrecognized/incomplete class: conservative deep default
    return "deep"


def _validate_requirements(requirements: dict) -> None:
    missing = [f for f in REQUIRED_REQUIREMENT_FIELDS if f not in requirements]
    if missing:
        raise RoutingBlocked("schema_invalid", f"requirements missing fields: {missing}")
    provenance = requirements.get("provenance") or {}
    missing_prov = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in provenance]
    if missing_prov:
        raise RoutingBlocked("schema_invalid", f"provenance missing fields: {missing_prov}")


def _validate_policy(policy: dict) -> None:
    for field in ("schema_version", "policy_id", "revision", "approval_ref", "routes"):
        if field not in policy:
            raise RoutingBlocked("schema_invalid", f"policy missing field: {field}")
    for route in policy["routes"]:
        missing = [f for f in REQUIRED_ROUTE_FIELDS if f not in route]
        if missing:
            raise RoutingBlocked("schema_invalid", f"route {route.get('route_id')} missing: {missing}")


def _contributing_makers(requirements: dict) -> set:
    provenance = requirements["provenance"]
    if not provenance.get("complete", False):
        raise RoutingBlocked("provenance_incomplete", "implementation provenance not verified complete")
    contributors = provenance.get("contributors") or []
    makers = set()
    for contributor in contributors:
        maker = contributor.get("maker")
        if not maker:
            raise RoutingBlocked("provenance_incomplete", "contributor with unknown maker")
        makers.add(maker)
    return makers


def _route_rejection(route: dict, requirements: dict, quality: str, excluded_makers: set) -> str | None:
    role = requirements["role"]
    if route["status"] != "approved":
        return "not_approved"
    if role not in route["allowed_roles"]:
        return "role_not_allowed"
    if route["maker"] in excluded_makers:
        return "contributing_maker"
    missing_caps = set(requirements["required_capabilities"]) - set(route["capabilities"])
    if missing_caps:
        return "missing_capabilities"
    needed_tokens = requirements["input_tokens"] + requirements["reserve_tokens"]
    budget = route["verified_input_budget"]
    # A null/non-numeric budget is an UNVERIFIED route -- a roster-discovery
    # candidate that has never been curated, not a route with "no limit".
    # Treat it identically to "does not fit": fail closed, never let a raw
    # comparison against None escape as a TypeError and never silently
    # admit an unverified route as fitting any request.
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        return "input_too_large"
    if needed_tokens > budget:
        return "input_too_large"
    if requirements["reasoning"] not in route["allowed_reasoning"]:
        return "reasoning_unsupported"
    qualifications = route["qualifications"]
    # A roster-discovery/candidate artifact stores `qualifications` as an
    # assessment OBJECT (e.g. {"disposition": "qualified", ...}), not the
    # approved schema's list of satisfied task-class strings. `in` against a
    # dict would silently check its KEYS, which happens to look plausible
    # but is not the approved contract -- fail closed instead of accepting
    # publication shape as if it were curated qualification.
    if not isinstance(qualifications, (list, tuple, set)):
        return "qualification_unmet"
    if quality not in qualifications:
        return "qualification_unmet"
    return None


def select(requirements: dict, policy: dict, availability: dict, now: int) -> dict:
    """Deterministically select a route for `requirements` under `policy`.

    Returns an immutable-shaped decision dict (never mutated by callers).
    Raises RoutingBlocked with a typed reason code when no candidate qualifies.
    `availability` is a bounded freshness snapshot keyed by route_id (design §12);
    an empty dict means "no availability evidence gathered", which this pure
    selector does not itself treat as unavailable — the common guard (step 3)
    owns bounded startup attempts and cooldowns.
    """
    _validate_requirements(requirements)
    _validate_policy(policy)

    quality = _quality_floor(requirements["task_class"])
    excluded_makers = _contributing_makers(requirements)

    role = requirements["role"]
    ranking = policy.get("rankings", {}).get(role, {}).get(quality, [])
    routes_by_id = {r["route_id"]: r for r in policy["routes"]}

    rejections: dict = {}
    ordered_candidates = [rid for rid in ranking if rid in routes_by_id]
    # Include any approved routes not present in the curated ranking so their
    # rejection reasons are still visible, but never let them win a selection.
    unranked = [rid for rid in routes_by_id if rid not in ordered_candidates]

    selected = None
    for route_id in ordered_candidates:
        route = routes_by_id[route_id]
        reason = _route_rejection(route, requirements, quality, excluded_makers)
        if reason is not None:
            rejections[route_id] = [reason]
            continue
        if selected is None:
            selected = route
        # Keep evaluating every remaining ranked candidate even after a winner
        # is found: an ineligible later route must be recorded as a rejection,
        # never surface as a viable "alternate" just because we stopped early.

    for route_id in unranked:
        if route_id in rejections:
            continue
        route = routes_by_id[route_id]
        reason = _route_rejection(route, requirements, quality, excluded_makers)
        rejections[route_id] = [reason] if reason else ["not_in_curated_ranking"]

    if selected is None:
        if excluded_makers and all(
            rejections.get(rid) == ["contributing_maker"] for rid in ordered_candidates
        ) and ordered_candidates:
            raise RoutingBlocked("independence_unavailable", "no non-contributing-maker route qualifies")
        raise RoutingBlocked("no_qualified_route", f"no route satisfies role={role} quality={quality}")

    alternates = [
        rid for rid in ordered_candidates
        if rid != selected["route_id"] and rid not in rejections
    ]

    decision = {
        "policy_id": policy["policy_id"],
        "policy_revision": policy["revision"],
        "requirements": {
            "role": role,
            "execution_kind": requirements["execution_kind"],
            "execution_id": requirements["execution_id"],
            "attempt_id": requirements["attempt_id"],
            "slot_id": requirements.get("slot_id", ""),
            "task_class": requirements["task_class"],
            "quality": quality,
            "reasoning": requirements["reasoning"],
        },
        "selected": {
            "route_id": selected["route_id"],
            "route_revision": selected["route_revision"],
            "provider": selected["provider"],
            "model": selected["model"],
            "endpoint": selected["endpoint"],
            "maker": selected["maker"],
        },
        "rejections": rejections,
        "alternates": alternates,
        "selection_timestamp": now,
    }
    return decision
