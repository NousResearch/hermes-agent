"""MoA adapter for guided model routing (design §4 step 4, §6 "MoA").

Wires MoA reference and aggregator slots to the SAME neutral primitives Kanban and delegation
already use (`agent.managed_route_runtime`, `agent.model_selection`, `agent.model_selection_store`)
-- no duplicate selection/persistence/guard logic here, only the MoA-specific requirements shape.

A slot (reference or aggregator dict from a preset) is UNMANAGED unless it carries an explicit
`routing_role` -- the existing fixed-provider/model preset shape is untouched byte-for-byte when
no `routing_role` is present (design §5, "preserve unmanaged behavior").

Public entries:
  - `resolve_moa_slot_route(slot, *, execution_id, attempt_id, slot_id)` -> `None` (unmanaged) or a
    `managed_child_kwargs`-shaped dict (+ `receipt_id`/`routing_home`) from the SAME selector/store
    used by Kanban/delegation.
  - `moa_runtime_overrides(resolution)` -> real provider/model/base_url/api_key/api_mode/reasoning
    kwargs for the actual client construction (`call_llm`), resolved through the SAME
    `hermes_cli.runtime_provider.resolve_runtime_provider` every other adapter uses -- never a
    fabricated/parent-inherited endpoint.
  - `enforce_moa_slot_route(resolution, *, actual_provider, actual_model, actual_endpoint,
    actual_reasoning)` -> the guard call site, invoked at the ACTUAL MoA call boundary
    (`agent.moa_loop._run_reference` / `_call_prepared_aggregator`) immediately before the real
    `call_llm` for that slot -- never merely stamped as an attribute that nothing consumes.

Nested/managed-parent authority: an MoA slot's `routing_role` is genuine per-slot intake
(mirroring the Kanban `--routing-role`/delegation `routing_role` contract); this module does not
itself widen a policy_id/role beyond what the slot explicitly asked for. A denied/mismatched slot
raises `RoutingBlocked` and the caller MUST stop before any content is sent to that slot -- no
silent fallback to the slot's plain provider/model (§5 no silent escape).
"""
from __future__ import annotations

import time
from typing import Any, Optional

from agent.managed_route_runtime import enforce_worker_route, resolve_route
from agent.model_selection_types import RoutingBlocked

__all__ = ["resolve_moa_slot_route", "moa_runtime_overrides", "enforce_moa_slot_route", "MoARoutingBlocked"]

DEFAULT_MOA_POLICY_ID = "kanban-default"


class MoARoutingBlocked(RoutingBlocked):
    """Re-raised with the MoA call site's context; same reason codes as RoutingBlocked."""


def _slot_intake(slot: dict) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """``(routing_role, routing_requirements, routing_policy_id)`` as supplied on the slot dict --
    genuine per-slot intake, never a model-invented default. Validated with the SAME validator
    Kanban's ``--routing-requirements``/delegation's ``routing_requirements`` share."""
    role = slot.get("routing_role")
    role = role.strip() if isinstance(role, str) and role.strip() else None
    raw_requirements = slot.get("routing_requirements")
    requirements = None
    if raw_requirements is not None:
        from hermes_cli.kanban_model_routing import validate_routing_requirements

        requirements = validate_routing_requirements(raw_requirements)  # raises ValueError on malformed intake
    policy_id = slot.get("routing_policy_id")
    policy_id = policy_id.strip() if isinstance(policy_id, str) and policy_id.strip() else None
    return role, requirements, policy_id


def _build_requirements(
    slot: dict, intake: Optional[dict], *, role: str, execution_id: str, attempt_id: str, slot_id: str,
) -> dict:
    """The real `TaskRequirements` shape for one MoA slot, mirroring the Kanban/delegation adapters
    (design §3.B): genuine per-slot intake or the honest, non-fabricated conservative defaults."""
    intake = intake or {}
    requirements = {
        "schema_version": 1,
        "role": role,
        "execution_kind": "moa",
        "execution_id": execution_id,
        "attempt_id": attempt_id,
        "slot_id": slot_id,
        "task_class": intake.get("task_class", ""),
        "required_capabilities": list(intake.get("required_capabilities", [])),
        "input_tokens": int(intake.get("input_tokens", 0)),
        "reserve_tokens": int(intake.get("reserve_tokens", 0)),
        "reasoning": slot.get("reasoning_effort") or "medium",
    }
    provenance = intake.get("provenance")
    from hermes_cli.kanban_model_routing import _is_review_role

    if _is_review_role(role):
        if provenance is None:
            raise MoARoutingBlocked(
                "provenance_incomplete",
                f"MoA slot {slot_id!r}: role {role!r} is an independent-review role and requires "
                "genuine contributor-maker provenance (slot['routing_requirements']['provenance']); "
                "none was supplied -- refusing to manufacture completeness",
            )
        requirements["provenance"] = provenance
    elif provenance is not None:
        requirements["provenance"] = provenance
    else:
        requirements["provenance"] = {
            "frozen_sha": "", "verified_by": "moa-adapter", "complete": True, "contributors": [],
        }
    return requirements


def resolve_moa_slot_route(
    slot: dict, *, execution_id: str, attempt_id: str = "0", slot_id: str,
) -> Optional[dict]:
    """Resolve a managed route for one MoA reference/aggregator slot, or ``None`` when the slot
    carries no ``routing_role`` (the existing unmanaged path, byte-for-byte).

    Raises ``RoutingBlocked``/``MoARoutingBlocked`` on any policy/eligibility failure; the caller
    (``agent.moa_loop``) must propagate it as a real slot failure (a labelled ``[failed: ...]``
    reference note, or an aborted aggregator call) and must NEVER fall back to constructing an
    unmanaged/default-route slot instead (§5 no silent escape).
    """
    role, requirements_intake, policy_id_override = _slot_intake(slot)
    if role is None:
        return None
    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    policy_id = policy_id_override or DEFAULT_MOA_POLICY_ID
    requirements = _build_requirements(
        slot, requirements_intake, role=role, execution_id=execution_id, attempt_id=attempt_id, slot_id=slot_id,
    )
    decision_kwargs = resolve_route(hermes_home, policy_id, requirements, now=int(time.time()))
    decision_kwargs["routing_home"] = hermes_home
    return decision_kwargs


def moa_runtime_overrides(resolution: dict) -> dict[str, Any]:
    """The real ``call_llm`` kwargs (provider/model/base_url/api_key/api_mode/reasoning_effort) for
    a receipted MoA slot resolution -- resolved through the SAME
    ``hermes_cli.runtime_provider.resolve_runtime_provider`` every other adapter uses, so the
    endpoint/credentials are never fabricated or silently inherited from the preset's plain slot.
    Disables the plain-slot runtime cache entirely: these are authoritative overrides (§5).
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider

    provider = resolution["provider"]
    model = resolution["model"]
    out: dict[str, Any] = {"provider": provider, "model": model}
    rt = resolve_runtime_provider(requested=provider, target_model=model)
    out.update({k: rt[k] for k in ("base_url", "api_key", "api_mode") if rt.get(k)})
    overrides = rt.get("request_overrides")
    extra_body = overrides.get("extra_body") if isinstance(overrides, dict) else None
    if isinstance(extra_body, dict) and extra_body:
        out["extra_body"] = dict(extra_body)
    # The receipted endpoint (when the route specifies one) is authoritative over whatever the
    # generic runtime-provider resolution returned for `base_url` -- the guard below re-checks this.
    if resolution.get("endpoint"):
        out["base_url"] = resolution["endpoint"]
    reasoning_effort = resolution.get("reasoning_effort")
    if reasoning_effort:
        out["reasoning_effort"] = reasoning_effort
    return out


def enforce_moa_slot_route(
    resolution: dict, *, actual_provider: str, actual_model: str, actual_endpoint: Optional[str],
    actual_reasoning: Optional[str],
) -> None:
    """The MoA-side guard call: invoked at the ACTUAL call boundary
    (``call_llm`` in ``_run_reference``/``_call_prepared_aggregator``) immediately before that
    slot's real inference, with the ACTUALLY constructed provider/model/endpoint/reasoning.
    Raises ``RoutingBlocked`` on any divergence (revoked/stale decision, a fallback that silently
    swapped the runtime mid-call, etc.) -- the caller must stop before sending slot content.
    """
    enforce_worker_route(
        resolution["routing_home"], resolution["receipt_id"],
        actual_provider=actual_provider, actual_model=actual_model,
        actual_endpoint=actual_endpoint, actual_reasoning=actual_reasoning,
        outcome_kind="routing_started",
    )
