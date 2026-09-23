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

__all__ = [
    "resolve_moa_slot_route", "moa_runtime_overrides", "enforce_moa_slot_route",
    "MoARoutingBlocked", "MoARequiredSlotDenied", "is_moa_slot_required", "resolve_moa_cohort",
    "resolve_moa_cohort_pinned", "resolve_moa_slot_route_pinned", "execution_id_for_agent",
    "cohort_has_managed_slot",
]


def execution_id_for_agent(agent: Any, fallback_prefix: str) -> str:
    """The genuine per-attempt identity a live MoA run pins its cohort to.

    A real turn (agent.conversation_loop._bind_turn_identity) stamps
    agent._current_turn_id with a value that is unique per user turn/attempt and STABLE
    across every iteration of that same turn (repeated fan-out cadence calls, the aggregator
    call after fan-out, a rebased prepared request) -- exactly the boundary design section 6
    "MoA" means by "resolve the complete cohort once per MoA run and bind an immutable
    effective preset to that client". Using it (instead of a constant string like the former
    "moa-oneshot"/"moa-preset-<name>") is what makes the pinning cache in
    resolve_moa_cohort_pinned/resolve_moa_slot_route_pinned actually run-scoped: a NEW
    turn/attempt gets a NEW execution_id and therefore a fresh cohort resolution, while
    concurrent/successive calls WITHIN one turn share the identical pinned cohort.

    When no live turn context is available (agent is None, or a standalone call with no
    bound turn -- e.g. a bare unit test), a fresh random id is minted so nothing is ever pinned
    across genuinely unrelated calls; it is never safe to fall back to a shared constant.
    """
    turn_id = getattr(agent, "_current_turn_id", None) if agent is not None else None
    if isinstance(turn_id, str) and turn_id.strip():
        return turn_id
    import uuid as _uuid

    return f"{fallback_prefix}-{_uuid.uuid4().hex}"


def cohort_has_managed_slot(reference_slots: list, aggregator: dict) -> bool:
    """Whether ANY slot in the cohort (reference or aggregator) carries a routing_role --
    the cheap pre-check that lets an entirely unmanaged preset skip cohort resolution
    altogether (byte-for-byte unmanaged behavior, design section 5)."""
    for slot in (*reference_slots, aggregator):
        if isinstance(slot, dict) and isinstance(slot.get("routing_role"), str) and slot.get("routing_role").strip():
            return True
    return False

DEFAULT_MOA_POLICY_ID = "kanban-default"


class MoARoutingBlocked(RoutingBlocked):
    """Re-raised with the MoA call site's context; same reason codes as RoutingBlocked."""


class MoARequiredSlotDenied(MoARoutingBlocked):
    """A managed slot's route could not be resolved, or was denied at the call-boundary guard.

    Per the BINDING design (plan §6 "MoA": "Missing/partial required output fails that review
    attempt... No silent optional-reference or aggregator-only mode"), a managed slot
    (``routing_role`` set) is REQUIRED BY DEFAULT -- the caller (``agent.moa_loop``) MUST
    propagate this as a hard failure of the whole MoA attempt: never swallow it into a labelled
    ``[failed: ...]`` note that quietly degrades into aggregator-only/empty synthesis, and never
    let it fall through to an unmanaged continuation. Enforced slots cannot opt out."""

    def __init__(self, slot_id: str, role: str, reason: str, detail: str = ""):
        self.slot_id = slot_id
        self.role = role
        super().__init__(reason, detail)


def is_moa_slot_required(slot: dict) -> bool:
    """Whether a managed slot (``routing_role`` set) gates the whole MoA attempt on its success.

    BINDING default (plan §6 "MoA"): every managed slot is required. "Missing/partial required
    output fails that review attempt" and "No silent optional-reference or aggregator-only
    mode" are unconditional for the managed cohort -- a denied managed reference or aggregator
    must raise, not degrade to a labelled ``[failed: ...]`` note that lets the turn quietly
    complete as if the gate had passed. Fixed and shadow slots remain advisory.
    """
    if not isinstance(slot, dict):
        return False
    role = slot.get("routing_role")
    if not (isinstance(role, str) and role.strip()):
        return False
    if str(slot.get("routing_mode") or "enforced").strip().lower() == "shadow":
        return False
    return True


def _resolved_maker(resolution: dict) -> str:
    """The curated maker identity behind a receipted resolution -- read from the PERSISTED
    receipt (``agent.model_selection_store.get_receipt``), never from slot-supplied text. A slot
    cannot spoof independence by claiming its own maker: the roster/policy is the sole source of
    truth for ``selected.maker`` (design §12 "Identity, provenance and classification")."""
    from agent.model_selection_store import get_receipt

    decision = get_receipt(resolution["routing_home"], resolution["receipt_id"])
    if decision is None:
        raise MoARoutingBlocked(
            "stale_or_revoked_decision",
            f"no routing receipt found for id={resolution['receipt_id']!r} while checking cohort maker diversity",
        )
    maker = decision["selected"].get("maker")
    if not maker or str(maker).strip().lower() == "moa":
        # Design §6: "the virtual maker 'moa' cannot satisfy an independence constraint."
        raise MoARoutingBlocked(
            "independence_unavailable",
            f"resolved route has no real maker identity (maker={maker!r}); the virtual 'moa' "
            "maker or an unknown maker cannot satisfy cohort diversity",
        )
    return str(maker)


def resolve_moa_cohort(
    reference_slots: list, aggregator: dict, *, execution_id: str, hermes_home: Optional[str] = None,
) -> dict:
    """Resolve the COMPLETE MoA cohort (every reference slot plus the aggregator) ONCE, and
    validate joint cohort diversity/independence before any content is sent to any slot (design
    §6 "MoA": "Resolve the complete cohort once per MoA run and bind an immutable effective
    preset to that client"; "Joint selection validates required cohort diversity/independence
    and actual maker identities").

    Returns ``{slot_id: resolution_or_None}`` -- ``None`` for an unmanaged slot (no
    ``routing_role``), a ``resolve_moa_slot_route``-shaped dict for a managed one. Callers
    (``agent.moa_loop``) MUST reuse these resolutions verbatim for the actual calls rather than
    re-resolving per-slot -- re-resolving independently is exactly the bug this function fixes
    (a joint cohort check cannot be expressed as N independent per-slot resolutions).

    Raises ``MoARoutingBlocked``/``MoARequiredSlotDenied`` before ANY slot is called when:
      * any REQUIRED slot's individual route resolution fails (missing policy, no qualified
        route, independence unavailable for that slot alone), or
      * the cohort of REQUIRED reference slots that resolve successfully has fewer than 2
        distinct real maker identities among 2+ required references (design §6: "two required
        references from different approved makers"). The aggregator MAY share a reference's
        maker (design §6 explicitly allows this) so it is resolved but not counted in the
        reference-diversity check.

    A cohort with fewer than 2 managed required references (e.g. a single-reference preset, or a
    fully unmanaged preset) has nothing to diversify and is not rejected on diversity grounds --
    this function only enforces the plan's specific "two distinct-maker references" shape when
    that shape is actually present (2+ required managed references), never invents a floor a
    smaller/unmanaged preset never opted into.
    """
    resolutions: dict = {}
    required_reference_makers: dict = {}
    excluded_makers: set = set()

    for idx, slot in enumerate(reference_slots):
        slot_id = f"reference-{idx}"
        try:
            resolution = resolve_moa_slot_route(
                slot, execution_id=execution_id, slot_id=slot_id, hermes_home=hermes_home,
                cohort_excluded_makers=sorted(excluded_makers) if is_moa_slot_required(slot) else (),
            )
        except RoutingBlocked as exc:
            if is_moa_slot_required(slot):
                raise MoARequiredSlotDenied(slot_id, slot.get("routing_role", ""), exc.reason, exc.detail) from exc
            resolutions[slot_id] = None
            continue
        resolutions[slot_id] = resolution
        if resolution is not None and is_moa_slot_required(slot):
            maker = _resolved_maker(resolution)
            required_reference_makers[slot_id] = maker
            excluded_makers.add(maker)

    try:
        agg_resolution = resolve_moa_slot_route(
            aggregator, execution_id=execution_id, slot_id="aggregator", hermes_home=hermes_home,
        )
    except RoutingBlocked as exc:
        if is_moa_slot_required(aggregator):
            raise MoARequiredSlotDenied("aggregator", aggregator.get("routing_role", ""), exc.reason, exc.detail) from exc
        agg_resolution = None
    resolutions["aggregator"] = agg_resolution

    if len(required_reference_makers) >= 2 and len(set(required_reference_makers.values())) < 2:
        raise MoARoutingBlocked(
            "independence_unavailable",
            "managed required references resolved to fewer than 2 distinct makers "
            f"({sorted(set(required_reference_makers.values()))}); design §6 requires two "
            "required references from different approved makers",
        )
    return resolutions


import threading as _threading

# Pinned-cohort cache (design §6 "MoA": "Resolve the complete cohort once per MoA run and bind
# an immutable effective preset to that client... Cohort resumes only within the original live
# pinned run; restarting a failed cohort is a new attempt"). Keyed by (execution_id, origin_home)
# -- NOT execution_id alone -- so repeated fanout iterations and aggregator calls within the SAME
# run reuse the identical resolutions while two DIFFERENT origin profiles (Hermes homes) that
# happen to mint the same execution_id (e.g. concurrent multiplexed profiles in one process)
# never share, evict, or leak into each other's pinned cohort/receipt/endpoint. A config edit
# mid-run cannot reroute an already-pinned cohort, and a slot is never re-resolved (no repeated
# reselection per iteration).
_cohort_cache_lock = _threading.Lock()
_cohort_cache: dict = {}


def _origin_home_key(hermes_home: Optional[str] = None) -> str:
    """The canonical origin-profile identity a cohort/slot cache entry is scoped to.

    Host-owned: derived from the real Hermes home path this process/profile is bound to
    (``hermes_constants.get_hermes_home()`` when not explicitly supplied), never an
    arbitrary caller-supplied claim -- a caller cannot widen or narrow its own cache
    isolation by passing a different string.
    """
    import os

    home = hermes_home
    if home is None:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    return os.path.realpath(str(home))


def resolve_moa_cohort_pinned(
    reference_slots: list, aggregator: dict, *, execution_id: str, hermes_home: Optional[str] = None,
) -> dict:
    """Cached wrapper over ``resolve_moa_cohort``: the FIRST call for a given
    ``(execution_id, origin_home)`` performs the real joint resolution (and may raise); every
    subsequent call for the SAME execution_id WITHIN THE SAME origin profile, within the same
    process, returns the identical pinned resolutions dict without re-resolving or
    re-validating diversity, even if the live policy changed in between -- config/policy edits
    take effect on the NEXT run's execution_id, never retroactively reroute an in-flight one.

    Scoped by origin Hermes home (design/root AGENTS.md named bug class: "module globals...
    hold the launch profile's state... a silent default-profile leak") so two profiles that
    independently mint the SAME execution_id in one process (e.g. concurrent multiplexed
    profiles) never share, evict, or leak into each other's pinned cohort.

    Also seeds the per-slot cache (``resolve_moa_slot_route_pinned``'s backing store) with every
    resolution from this cohort, keyed by the SAME ``(execution_id, origin_home)`` -- so a later
    per-slot lookup for one of these exact slot_ids (``reference-0``, ``reference-1``, ...,
    ``aggregator``) within the same run/profile reuses the joint resolution verbatim instead of
    re-resolving independently (a joint cohort decision cannot be safely re-derived as N separate
    per-slot calls; this is what makes the fan-out/aggregator call sites in ``agent.moa_loop``
    share ONE cohort decision).
    """
    home = _origin_home_key(hermes_home)
    key = (execution_id, home)
    with _cohort_cache_lock:
        cached = _cohort_cache.get(key)
    if cached is not None:
        return cached
    resolved = resolve_moa_cohort(reference_slots, aggregator, execution_id=execution_id, hermes_home=home)
    with _cohort_cache_lock:
        _cohort_cache.setdefault(key, resolved)
        per_slot = _cohort_cache.setdefault("__slots__", {})
        for slot_id, resolution in resolved.items():
            per_slot.setdefault((execution_id, slot_id, home), resolution)
        return _cohort_cache[key]


def resolve_moa_slot_route_pinned(
    slot: dict, *, execution_id: str, attempt_id: str = "0", slot_id: str,
    hermes_home: Optional[str] = None,
) -> "Optional[dict]":
    """Per-slot pinned wrapper over ``resolve_moa_slot_route`` (design §6 "MoA": "Resolve the
    complete cohort once per MoA run and bind an immutable effective preset to that client").

    The FIRST resolution for a given ``(execution_id, slot_id, origin_home)`` is cached; every
    later call within the same run/profile (repeated fan-out iterations, the aggregator call
    after fan-out, a live config/policy edit mid-run) returns the SAME resolution rather than
    re-resolving -- a config change takes effect on the NEXT run's execution_id only, never
    reroutes an in-flight one, and a slot is never reselected per iteration. Scoped by origin
    Hermes home so a same-execution_id collision across two concurrently-running profiles in one
    process never shares or evicts either profile's pinned entry.
    """
    home = _origin_home_key(hermes_home)
    key = (execution_id, slot_id, home)
    with _cohort_cache_lock:
        per_slot = _cohort_cache.setdefault("__slots__", {})
        if key in per_slot:
            return per_slot[key]
    resolved = resolve_moa_slot_route(
        slot, execution_id=execution_id, attempt_id=attempt_id, slot_id=slot_id, hermes_home=home,
    )
    with _cohort_cache_lock:
        per_slot = _cohort_cache.setdefault("__slots__", {})
        per_slot.setdefault(key, resolved)
        return per_slot[key]


def _forget_cohort(execution_id: str, *, hermes_home: Optional[str] = None) -> None:
    """Turn-end cleanup: drop a pinned cohort AND its per-slot cache entries so a NEW
    attempt (never the same execution_id) can resolve fresh. Production code never calls
    this for a live execution_id -- restarting a failed cohort means a new attempt with a
    new execution_id, per design §6. Scoped to the SAME origin home a pinning call would
    have used, so forgetting one profile's entry never touches another profile's pinned
    cohort under a colliding execution_id.

    Also evicts every ``__slots__`` entry keyed to this ``(execution_id, home)`` -- the
    cohort cache and the per-slot cache (``resolve_moa_slot_route_pinned``'s backing store)
    are two separate dicts under the same lock, and the real turn-end hook
    (``agent.agent_runtime_helpers.note_turn_persisted``) only ever calls this one function.
    Popping only the cohort key left every slot resolution
    (``(execution_id, slot_id, home)`` for ``reference-0``, ``reference-1``, ...,
    ``aggregator``) permanently cached in ``__slots__`` with nothing to ever remove them --
    a real per-turn memory leak across every finished MoA run in a long-lived process, since
    a turn's execution_id is never reused (design §6: a new attempt always mints a new one).
    """
    home = _origin_home_key(hermes_home)
    key = (execution_id, home)
    with _cohort_cache_lock:
        _cohort_cache.pop(key, None)
        per_slot = _cohort_cache.get("__slots__")
        if per_slot:
            for slot_key in [k for k in per_slot if k[0] == execution_id and k[2] == home]:
                per_slot.pop(slot_key, None)


def _slot_intake(slot: dict) -> tuple[Optional[str], Optional[dict], Optional[str], Optional[str]]:
    """``(routing_role, routing_requirements, routing_policy_id, routing_mode)`` as supplied on the slot dict --
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
    from hermes_cli.kanban_model_routing import normalize_routing_mode

    mode = normalize_routing_mode(slot.get("routing_mode"), has_role=bool(role))
    return role, requirements, policy_id, mode


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
        "risk_flags": list(intake.get("risk_flags", [])),
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
    hermes_home: Optional[str] = None,
    cohort_excluded_makers=(),
) -> Optional[dict]:
    """Resolve a managed route for one MoA reference/aggregator slot, or ``None`` when the slot
    carries no ``routing_role`` (the existing unmanaged path, byte-for-byte).

    Raises ``RoutingBlocked``/``MoARoutingBlocked`` on any policy/eligibility failure; the caller
    (``agent.moa_loop``) must propagate it as a real slot failure (a labelled ``[failed: ...]``
    reference note, or an aborted aggregator call) and must NEVER fall back to constructing an
    unmanaged/default-route slot instead (§5 no silent escape).

    ``hermes_home``: the origin profile to resolve against. Defaults to
    ``hermes_constants.get_hermes_home()`` (the existing byte-for-byte behavior for every
    caller that does not explicitly pin a profile) -- explicitly threaded through by the
    ``*_pinned`` wrappers below so a cache lookup keyed on one profile's home actually resolves
    against THAT profile's store, not whatever profile this process happens to be launched as.
    """
    role, requirements_intake, policy_id_override, mode = _slot_intake(slot)
    if role is None:
        return None
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    policy_id = policy_id_override or DEFAULT_MOA_POLICY_ID
    try:
        requirements = _build_requirements(
            slot, requirements_intake, role=role, execution_id=execution_id, attempt_id=attempt_id, slot_id=slot_id,
        )
        requirements["cohort_excluded_makers"] = list(cohort_excluded_makers)
        decision_kwargs = resolve_route(hermes_home, policy_id, requirements, now=int(time.time()))
        if mode == "shadow":
            from agent.model_selection_store import append_outcome, get_receipt

            shadow_decision = get_receipt(hermes_home, decision_kwargs["receipt_id"])
            append_outcome(hermes_home, decision_kwargs["receipt_id"], "routing_shadow", {
                "execution_kind": "moa", "execution_id": execution_id, "slot_id": slot_id,
                "recommended_route_id": shadow_decision["selected"]["route_id"] if shadow_decision else None,
            })
            return {
                "routing_mode": "shadow", "routing_home": hermes_home,
                "receipt_id": decision_kwargs["receipt_id"],
            }
    except Exception as exc:
        if mode != "shadow":
            raise
        return {
            "routing_mode": "shadow", "routing_home": hermes_home,
            "shadow_error": exc.reason if isinstance(exc, RoutingBlocked) else "observation_failed",
        }
    decision_kwargs["routing_mode"] = "enforced"
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
