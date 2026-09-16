"""Kanban integration seam for guided model routing (design §4 step 4, §12).

Resolves a `routing_role` task's decision at CLAIM time (not creation), using
the origin profile's active policy in `model_routing.db`, persists the
receipt, records the receipt id on the task row, and returns the real
provider/model/reasoning kwargs the dispatcher must pass to the worker
argv — the ONLY place a Kanban worker's actual route is decided for a
managed task. An unmanaged task (`routing_role` is None) is untouched.
"""
from __future__ import annotations

from typing import Optional

from agent.model_selection import select
from agent.model_selection_guard import managed_child_kwargs, validate_actual_route
from agent.model_selection_store import append_outcome, get_active_policy, get_receipt, persist_receipt
from agent.model_selection_types import RoutingBlocked

_POLICY_ID = "kanban-default"


def resolve_task_route(
    hermes_home,
    conn,
    task,
    *,
    now: int,
    frozen_sha: str,
    verified_by: str,
) -> Optional[dict]:
    """Resolve, receipt and persist the route for a claimed managed task.

    Returns the ``managed_child_kwargs``-shaped dict to merge into the worker
    argv, or ``None`` when the task is unmanaged (``routing_role`` unset).
    Raises ``RoutingBlocked`` (never swallowed here — the dispatcher's normal
    spawn-failure path records it and re-queues/blocks per the existing
    breaker, same as any other spawn-time exception) when policy/eligibility
    fails; the caller must NOT fall back to an unmanaged/default route on
    failure (design §5: no silent escape).
    """
    role = getattr(task, "routing_role", None)
    if not role:
        return None

    policy = get_active_policy(hermes_home, _POLICY_ID)
    if policy is None:
        raise RoutingBlocked("schema_invalid", f"no active policy published for {_POLICY_ID!r}")

    requirements = {
        "schema_version": 1,
        "role": role,
        "execution_kind": "kanban",
        "execution_id": task.id,
        "attempt_id": str(task.current_run_id or 0),
        "slot_id": "",
        "task_class": "cross-component",
        "required_capabilities": [],
        "input_tokens": 0,
        "reserve_tokens": 0,
        "reasoning": task.reasoning_effort or "medium",
        "provenance": {
            "frozen_sha": frozen_sha,
            "verified_by": verified_by,
            "complete": True,
            "contributors": [],
        },
    }
    decision = select(requirements, policy, {}, now)
    receipt_id = persist_receipt(hermes_home, decision)

    from hermes_cli.kanban_db import set_routing_receipt

    # Claim/run CAS (design §12 "Claim/start/crash sequence" step 3): link
    # this receipt to the task ONLY if the run this decision was resolved
    # under (task.current_run_id, captured by the caller's claim a moment
    # earlier) is STILL the task's current run. If a concurrent
    # reclaim/replace has already moved current_run_id on, this decision is
    # stale — persisting it left an inert receipt (harmless audit residue,
    # design §12), but it must never be linked/authorized against the new
    # run. Never silently proceed as if this attempt still owns the task.
    linked = set_routing_receipt(
        conn, task.id, receipt_id, expected_run_id=task.current_run_id,
    )
    if not linked:
        raise RoutingBlocked(
            "stale_or_revoked_decision",
            f"task {task.id}: claim/run changed before the routing receipt "
            f"could be linked (resolved under run_id={task.current_run_id!r}); "
            "refusing to authorize a launch under a superseded claim",
        )
    from agent.model_selection_store import append_outcome

    append_outcome(hermes_home, receipt_id, "routing_selected", {
        "task_id": task.id, "run_id": task.current_run_id,
        "route_id": decision["selected"]["route_id"],
    })
    kwargs = managed_child_kwargs(decision)
    # The worker process cannot re-run select(); it validates its own
    # actually-constructed route against this SAME receipted decision right
    # before its first inference (see ``enforce_worker_route`` below). Carried
    # through the dispatcher's env, never re-derived.
    kwargs["receipt_id"] = receipt_id
    return kwargs


def enforce_worker_route(
    hermes_home,
    receipt_id: str,
    *,
    actual_provider: str,
    actual_model: str,
    actual_endpoint: Optional[str],
    actual_reasoning: Optional[str],
) -> None:
    """The worker-side half of the guard (design §4 step 7, §12 "Claim/start/crash
    sequence" step 5): called from the actual Kanban worker process immediately
    before its first real inference, with the actually-constructed
    provider/model/endpoint/reasoning. Loads the SAME receipt the dispatcher
    persisted at claim time and raises ``RoutingBlocked`` on any divergence
    (revoked/stale decision, a code path that silently substituted a different
    route, etc.) — the worker must stop before sending task content, never
    silently fall back to whatever it was actually constructed with.
    """
    decision = get_receipt(hermes_home, receipt_id)
    if decision is None:
        raise RoutingBlocked(
            "stale_or_revoked_decision",
            f"no routing receipt found for id={receipt_id!r}; the decision this worker "
            "was claimed under is missing or was never persisted",
        )
    # Emergency revocation (design §12 "Availability, budget, reasoning and
    # revocation": "best-effort revocation generation check before each
    # subsequent managed request; it never substitutes another model").
    # Already-in-flight requests cannot be recalled -- this is the one
    # checkpoint before the FIRST request, so a policy edited/suspended after
    # the claim but before this worker's first inference must still stop it,
    # never silently launch under a routing that is no longer current.
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
    append_outcome(hermes_home, receipt_id, "routing_started", {
        "actual_provider": actual_provider, "actual_model": actual_model,
    })
