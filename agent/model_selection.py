"""Pure deterministic model-selection library (design §4, §12).

`select()` is intentionally pure: no I/O, no provider clients, no credentials,
no clock reads (caller passes `now`). It filters an approved-route policy
snapshot against structured task requirements and returns an immutable
decision dict, or raises RoutingBlocked with a typed reason code.

Runtime guards, receipt persistence and the Kanban/delegation/MoA adapters
are separate consumers; selection itself never launches work.
"""
from __future__ import annotations

from .model_selection_integrity import content_hash
from .model_selection_types import (
    DEEP_QUALITY_TASK_CLASSES,
    REQUIRED_PROVENANCE_FIELDS,
    REQUIRED_REQUIREMENT_FIELDS,
    REQUIRED_ROUTE_FIELDS,
    RoutingBlocked,
)

__all__ = ["select", "RoutingBlocked"]

_OPTIONAL_REQUIREMENT_FIELDS = frozenset({
    "allowed_route_ids", "cohort_excluded_makers", "input_empty", "slot_id", "target_profile",
    "risk_flags", "inherited_quality", "inherited_risk_flags",
})


def _quality_floor(task_class: str, classification: dict | None = None) -> str:
    """Deterministic quality floor: deep for cross-component/high-consequence.

    Unknown/unclassified scope also defaults to deep per design §3.B
    ("unclassified or incomplete scope defaults to deep").
    """
    if task_class in DEEP_QUALITY_TASK_CLASSES:
        return "deep"
    if (task_class == "established-pattern" and classification
            and classification["complete"] and not classification["risk_flags"]):
        return "shallow"
    # A model-proposed class alone does not attest that the scope is complete
    # or free of mandatory risk. Missing independent classification is deep.
    return "deep"


def _validate_requirements(requirements: dict) -> None:
    if not isinstance(requirements, dict):
        raise RoutingBlocked("schema_invalid", "requirements must be an object")
    missing = [f for f in REQUIRED_REQUIREMENT_FIELDS if f not in requirements]
    if missing:
        raise RoutingBlocked("schema_invalid", f"requirements missing fields: {missing}")
    unknown = set(requirements) - set(REQUIRED_REQUIREMENT_FIELDS) - _OPTIONAL_REQUIREMENT_FIELDS
    if unknown:
        raise RoutingBlocked("schema_invalid", f"requirements has unknown fields: {sorted(unknown)}")
    if requirements["schema_version"] != 1 or type(requirements["schema_version"]) is not int:
        raise RoutingBlocked("schema_invalid", "unsupported requirements schema_version")
    for field in ("role", "execution_kind", "execution_id", "attempt_id", "task_class", "reasoning"):
        if not isinstance(requirements[field], str):
            raise RoutingBlocked("schema_invalid", f"{field} must be text")
    _string_list(requirements["required_capabilities"], "required_capabilities")
    from agent.model_selection_classification import RISK_FLAGS

    for field in ("risk_flags", "inherited_risk_flags"):
        _string_list(requirements.get(field, []), field)
        if set(requirements.get(field, [])) - RISK_FLAGS:
            raise RoutingBlocked("schema_invalid", "unknown risk flags")
    if requirements.get("inherited_quality") not in (None, "shallow", "deep"):
        raise RoutingBlocked("schema_invalid", "invalid inherited quality")
    if "allowed_route_ids" in requirements:
        _string_list(requirements["allowed_route_ids"], "allowed_route_ids")
    _string_list(requirements.get("cohort_excluded_makers", []), "cohort_excluded_makers")
    provenance = requirements.get("provenance")
    if not isinstance(provenance, dict):
        raise RoutingBlocked("schema_invalid", "provenance must be an object")
    missing_prov = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in provenance]
    if missing_prov:
        raise RoutingBlocked("schema_invalid", f"provenance missing fields: {missing_prov}")
    if type(provenance["complete"]) is not bool or not isinstance(provenance["contributors"], list):
        raise RoutingBlocked("schema_invalid", "provenance needs a boolean complete and contributor list")
    if any(not isinstance(c, dict) or not isinstance(c.get("maker", ""), str) for c in provenance["contributors"]):
        raise RoutingBlocked("schema_invalid", "contributors must be objects with textual maker identities")
    for field in ("input_tokens", "reserve_tokens"):
        value = requirements[field]
        if value is not None and (type(value) is not int or value < 0):
            raise RoutingBlocked("schema_invalid", f"{field} must be a non-negative integer")
        if field == "input_tokens" and value == 0 and requirements.get("input_empty") is True:
            continue
        if value is None or value == 0:
            raise RoutingBlocked(
                "missing_input_estimate",
                f"{field} is unknown; supply a positive estimate covering assembled prompt, "
                "context and tools plus an output/tool-growth reserve before dispatch",
            )



def _string_list(value, field: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RoutingBlocked("schema_invalid", f"{field} must be a list of strings")


def _validate_policy(policy: dict) -> None:
    if not isinstance(policy, dict):
        raise RoutingBlocked("schema_invalid", "policy must be an object")
    for field in ("schema_version", "policy_id", "revision", "approval_ref", "routes"):
        if field not in policy:
            raise RoutingBlocked("schema_invalid", f"policy missing field: {field}")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise RoutingBlocked("schema_invalid", "unsupported policy schema_version")
    if type(policy["revision"]) is not int or policy["revision"] < 1:
        raise RoutingBlocked("schema_invalid", "policy revision must be a positive integer")
    if not isinstance(policy["routes"], list) or not isinstance(policy.get("rankings", {}), dict):
        raise RoutingBlocked("schema_invalid", "policy needs a routes list and rankings object")
    for ranking in policy.get("rankings", {}).values():
        if not isinstance(ranking, dict):
            raise RoutingBlocked("schema_invalid", "role rankings must be an object")
        for routes in ranking.values():
            _string_list(routes, "ranked route ids")
    seen = set()
    for route in policy["routes"]:
        if not isinstance(route, dict):
            raise RoutingBlocked("schema_invalid", "each route must be an object")
        missing = [f for f in REQUIRED_ROUTE_FIELDS if f not in route]
        if missing:
            raise RoutingBlocked("schema_invalid", f"route {route.get('route_id')} missing: {missing}")
        if not isinstance(route["route_id"], str) or route["route_id"] in seen:
            raise RoutingBlocked("schema_invalid", "route ids must be unique strings")
        seen.add(route["route_id"])
        if route["status"] not in ("candidate", "approved", "suspended", "retired"):
            raise RoutingBlocked("schema_invalid", f"route {route['route_id']} has invalid status")
        for field in ("allowed_roles", "capabilities", "allowed_reasoning"):
            _string_list(route[field], field)
    ranked_ids = {
        route_id
        for role_ranking in policy.get("rankings", {}).values()
        for ranked in role_ranking.values()
        for route_id in ranked
    }
    unknown_ranked_ids = ranked_ids - seen
    if unknown_ranked_ids:
        raise RoutingBlocked(
            "schema_invalid", f"rankings reference unknown route ids: {sorted(unknown_ranked_ids)}",
        )


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
    if "allowed_route_ids" in requirements and route["route_id"] not in requirements["allowed_route_ids"]:
        return "outside_parent_authority"
    if route["status"] != "approved":
        return "not_approved"
    if not isinstance(route["maker"], str) or not route["maker"]:
        return "unknown_maker"
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


def select(requirements: dict, policy: dict, availability: dict, now: int, *,
           classification: dict | None = None) -> dict:
    """Deterministically select a route for `requirements` under `policy`.

    Returns an immutable-shaped decision dict (never mutated by callers).
    Raises RoutingBlocked with a typed reason code when no candidate qualifies.
    Availability is scoped to the target profile, route revision and endpoint.
    Unknown health permits a receipted startup attempt; known unavailable
    candidates remain excluded through their cooldown, without lowering quality.
    """
    _validate_requirements(requirements)
    _validate_policy(policy)
    from agent.model_selection_availability import UNAVAILABLE, availability_snapshot

    health = availability_snapshot(requirements, policy["routes"], availability, now)

    if classification is not None:
        from agent.model_selection_classification import execution_identity

        if classification["execution"] != execution_identity(requirements):
            raise RoutingBlocked("stale_or_revoked_decision", "classification execution mismatch")
    quality = _quality_floor(requirements["task_class"], classification)
    if (requirements.get("risk_flags") or requirements.get("inherited_risk_flags")
            or requirements.get("inherited_quality") == "deep"):
        quality = "deep"
    excluded_makers = _contributing_makers(requirements)
    excluded_makers.update(requirements.get("cohort_excluded_makers", []))

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
        if reason is None and health[route_id]["status"] in UNAVAILABLE:
            reason = "provider_unavailable"
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
        if any(reasons == ["provider_unavailable"] for reasons in rejections.values()):
            raise RoutingBlocked("provider_unavailable", "qualified routes are in cooldown", rejections=rejections)
        if excluded_makers and all(
            rejections.get(rid) == ["contributing_maker"] for rid in ordered_candidates
        ) and ordered_candidates:
            raise RoutingBlocked("independence_unavailable", "no non-contributing-maker route qualifies", rejections=rejections)
        raise RoutingBlocked("no_qualified_route", f"no route satisfies role={role} quality={quality}", rejections=rejections)

    alternates = [
        rid for rid in ordered_candidates
        if rid != selected["route_id"] and rid not in rejections
    ]

    decision = {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "policy_revision": policy["revision"],
        "policy_hash": content_hash(policy),
        "requirements": {
            "role": role,
            "execution_kind": requirements["execution_kind"],
            "execution_id": requirements["execution_id"],
            "attempt_id": requirements["attempt_id"],
            "slot_id": requirements.get("slot_id", ""),
            "task_class": requirements["task_class"],
            "quality": quality,
            "reasoning": requirements["reasoning"],
            "input_tokens": requirements["input_tokens"],
            "input_empty": requirements.get("input_empty", False),
            "reserve_tokens": requirements["reserve_tokens"],
            "required_capabilities": requirements["required_capabilities"],
            "provenance": requirements["provenance"],
            "cohort_excluded_makers": requirements.get("cohort_excluded_makers", []),
            "target_profile": requirements.get("target_profile"),
            "classification": classification,
            "risk_flags": requirements.get("risk_flags", []),
            "inherited_quality": requirements.get("inherited_quality"),
            "inherited_risk_flags": requirements.get("inherited_risk_flags", []),
        },
        "selected": {
            "route_id": selected["route_id"],
            "route_revision": selected["route_revision"],
            "provider": selected["provider"],
            "model": selected["model"],
            "endpoint": selected["endpoint"],
            "maker": selected["maker"],
            "verified_input_budget": selected["verified_input_budget"],
        },
        "rejections": rejections,
        "alternates": alternates,
        "selection_timestamp": now,
        "availability": health,
    }
    return decision
