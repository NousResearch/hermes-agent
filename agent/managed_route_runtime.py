"""Neutral (non-Kanban-owned) managed-routing runtime seam (design §5, §6, §12).

All adapters import the shared guard here without depending on the Kanban CLI layer.

Pure orchestration: no provider clients, no credentials, no network. Delegates to
``agent.model_selection_store`` (receipt/policy storage) and ``agent.model_selection_guard``
(route/reasoning equality) -- the same primitives every adapter shares, so this module does not
re-implement selection or persistence.
"""
from __future__ import annotations

from typing import Optional

from agent.model_selection import select
from agent.model_selection_guard import managed_child_kwargs, validate_actual_route
from agent.model_selection_store import (
    append_outcome, is_route_revoked, get_active_policy, get_policy_revision, get_receipt, persist_receipt,
)
from agent.model_selection_types import RoutingBlocked
from agent.model_selection_integrity import content_hash

__all__ = ["resolve_route", "enforce_worker_route"]


def resolve_route(hermes_home, policy_id: str, requirements: dict, *, now: int,
                  policy_revision: Optional[int] = None) -> dict:
    """Select, receipt and persist a route for ``requirements`` under the active policy
    ``policy_id``. Returns ``managed_child_kwargs``-shaped dict plus ``receipt_id``.

    Raises ``RoutingBlocked`` (never swallowed -- the caller's normal failure path must record
    it and refuse to fall back to an unmanaged/default route, design §5: no silent escape).
    This function does NOT itself know about Kanban claims/task rows or delegation batches --
    callers own their own execution-kind-specific linking (e.g. Kanban's claim/run CAS).
    """
    policy = (get_active_policy(hermes_home, policy_id) if policy_revision is None
              else get_policy_revision(hermes_home, policy_id, policy_revision))
    if policy is None:
        raise RoutingBlocked("schema_invalid", f"no active policy published for {policy_id!r}")
    from agent.managed_route_health import load_availability
    from hermes_constants import get_hermes_home

    requirements = dict(requirements)
    requirements.setdefault("target_profile", str(get_hermes_home().resolve()))
    availability = load_availability(hermes_home, policy_id, requirements["target_profile"])
    from agent.model_selection_classification import get_classification

    decision = select(requirements, policy, availability, now,
                      classification=get_classification(hermes_home, requirements))
    receipt_id = persist_receipt(hermes_home, decision)
    append_outcome(hermes_home, receipt_id, "routing_selected", {
        "execution_kind": requirements["execution_kind"], "execution_id": requirements["execution_id"],
        "attempt_id": requirements["attempt_id"], "route_id": decision["selected"]["route_id"],
    })
    kwargs = managed_child_kwargs(decision)
    kwargs["receipt_id"] = receipt_id
    kwargs["role"] = requirements["role"]
    return kwargs


def enforce_worker_route(
    hermes_home,
    receipt_id: str,
    *,
    actual_provider: str,
    actual_model: str,
    actual_endpoint: Optional[str],
    actual_reasoning: Optional[str],
    record_outcome: bool = True,
    outcome_kind: str = "routing_started",
) -> None:
    """The worker/child-side half of the guard (design §4 step 7, §12 "Claim/start/crash
    sequence" step 5): called immediately before the actual constructed route's first real
    inference, with the actually-constructed provider/model/endpoint/reasoning. Loads the SAME
    receipt persisted at resolution time and raises ``RoutingBlocked`` on any divergence
    (revoked/stale decision, a code path that silently substituted a different route, etc.) --
    the caller must stop before sending task content, never silently fall back to whatever it
    was actually constructed with.

    ``record_outcome=False`` skips the append-only outcome write. Callers that re-run this same
    check before EVERY subsequent request in a managed turn (design §12: "best-effort revocation
    generation check before each subsequent managed request") pass this so the receipt's outcome
    log grows once per turn's first call, not once per iteration.
    """
    decision = get_receipt(hermes_home, receipt_id)
    if decision is None:
        raise RoutingBlocked(
            "stale_or_revoked_decision",
            f"no routing receipt found for id={receipt_id!r}; the decision this run "
            "was authorized under is missing or was never persisted",
        )
    # Emergency revocation (design §12 "Availability, budget, reasoning and revocation":
    # "best-effort revocation generation check before each subsequent managed request; it never
    # substitutes another model"). This is deliberately NOT "does the active policy revision
    # still equal the receipted revision" -- routine policy edits (publish + activate a new
    # revision) are ordinary admin actions that must only affect NEW attempts, never an
    # already-receipted, in-flight one (§12: "Routine policy edits affect new attempts, not
    # active conversations"). Only an EXPLICIT emergency revocation record
    # (``model_selection_store.revoke_route``) blocks a request under this route -- as DURABLE
    # authorization state about the route itself, not scoped to any one receipt's creation
    # time: it blocks every receipt referencing this route, including one persisted AFTER the
    # revocation (a routine `select()`+`persist_receipt()` cannot silently readmit a revoked
    # route -- that is the parent-reproduced bypass this closes). Only an explicit
    # ``model_selection_store.readmit_route`` call clears it. A missing active policy at all
    # (e.g. the whole policy_id was never published/activated in this store) is still treated
    # as blocked -- that is not a routine edit, it means there is no admission for this policy
    # at all.
    retained = get_policy_revision(hermes_home, decision["policy_id"], decision["policy_revision"])
    if retained is None or content_hash(retained) != decision.get("policy_hash"):
        raise RoutingBlocked("stale_or_revoked_decision", "receipt policy revision missing or mismatched")
    classification = decision["requirements"].get("classification")
    if classification is not None:
        from agent.model_selection_classification import get_classification

        if get_classification(hermes_home, decision["requirements"], version=classification["version"]) != classification:
            raise RoutingBlocked("stale_or_revoked_decision", "retained classification missing or mismatched")
    if get_active_policy(hermes_home, decision["policy_id"]) is None:
        raise RoutingBlocked(
            "stale_or_revoked_decision",
            f"policy {decision['policy_id']!r} has no active revision in this store; "
            "refusing to launch under a route with no current admission",
        )
    revocation = is_route_revoked(
        hermes_home, decision["policy_id"], decision["selected"]["route_id"],
    )
    if revocation is not None:
        raise RoutingBlocked(
            "stale_or_revoked_decision",
            f"policy {decision['policy_id']!r} route {decision['selected']['route_id']!r} was "
            f"explicitly revoked (reason={revocation['reason']!r}, "
            f"approval_ref={revocation['approval_ref']!r}); refusing to launch under a "
            "revoked route",
        )
    validate_actual_route(
        decision,
        actual_provider=actual_provider,
        actual_model=actual_model,
        actual_endpoint=actual_endpoint,
        actual_reasoning=actual_reasoning,
    )
    if record_outcome:
        append_outcome(hermes_home, receipt_id, outcome_kind, {
            "actual_provider": actual_provider, "actual_model": actual_model,
        })
