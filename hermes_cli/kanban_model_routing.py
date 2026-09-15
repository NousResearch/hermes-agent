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
from agent.model_selection_guard import managed_child_kwargs
from agent.model_selection_store import get_active_policy, persist_receipt
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

    set_routing_receipt(conn, task.id, receipt_id)
    return managed_child_kwargs(decision)
