"""Neutral (non-Kanban-owned) managed-routing runtime seam (design §5, §6, §12).

Moved out of ``hermes_cli.kanban_model_routing`` so a non-Kanban adapter (delegation, MoA) does
not have to import through the Kanban CLI layer to reach the common guard -- the design calls
this out explicitly: "shared managed_route_guard currently imports
hermes_cli.kanban_model_routing.enforce_worker_route -> factor neutral implementation if needed,
preserve Kanban wrapper paths/tests." ``hermes_cli.kanban_model_routing.enforce_worker_route``
re-exports this function unchanged so every existing Kanban call site, patch target and test
keeps working byte-for-byte.

Pure orchestration: no provider clients, no credentials, no network. Delegates to
``agent.model_selection_store`` (receipt/policy storage) and ``agent.model_selection_guard``
(route/reasoning equality) -- the same primitives every adapter shares, so this module does not
re-implement selection or persistence.
"""
from __future__ import annotations

from typing import Optional

from agent.model_selection import select
from agent.model_selection_guard import managed_child_kwargs, validate_actual_route
from agent.model_selection_store import append_outcome, get_active_policy, persist_receipt, get_receipt
from agent.model_selection_types import RoutingBlocked

__all__ = ["resolve_route", "enforce_worker_route"]


def resolve_route(hermes_home, policy_id: str, requirements: dict, *, now: int) -> dict:
    """Select, receipt and persist a route for ``requirements`` under the active policy
    ``policy_id``. Returns ``managed_child_kwargs``-shaped dict plus ``receipt_id``.

    Raises ``RoutingBlocked`` (never swallowed -- the caller's normal failure path must record
    it and refuse to fall back to an unmanaged/default route, design §5: no silent escape).
    This function does NOT itself know about Kanban claims/task rows or delegation batches --
    callers own their own execution-kind-specific linking (e.g. Kanban's claim/run CAS).
    """
    policy = get_active_policy(hermes_home, policy_id)
    if policy is None:
        raise RoutingBlocked("schema_invalid", f"no active policy published for {policy_id!r}")
    decision = select(requirements, policy, {}, now)
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
    # substitutes another model"). Already-in-flight requests cannot be recalled -- this
    # checkpoint runs before the FIRST request AND before every later request in the same
    # managed turn, so a policy edited/suspended between requests must still stop the NEXT one,
    # never silently launch/continue under a routing that is no longer current.
    active_policy = get_active_policy(hermes_home, decision["policy_id"])
    if active_policy is None or active_policy.get("revision") != decision["policy_revision"]:
        raise RoutingBlocked(
            "stale_or_revoked_decision",
            f"policy {decision['policy_id']!r} revision {decision['policy_revision']} "
            "is no longer the active revision (revoked/superseded since this decision "
            "was receipted); refusing to launch under a stale route",
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
