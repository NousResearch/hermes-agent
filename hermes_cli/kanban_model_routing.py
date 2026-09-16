"""Kanban integration seam for guided model routing (design §4 step 4, §12).

Resolves a `routing_role` task's decision at CLAIM time (not creation), using
the origin profile's active policy in `model_routing.db`, persists the
receipt, records the receipt id on the task row, and returns the real
provider/model/reasoning kwargs the dispatcher must pass to the worker
argv — the ONLY place a Kanban worker's actual route is decided for a
managed task. An unmanaged task (`routing_role` is None) is untouched.

Genuine per-task requirements intake (design §3.B): a task's actual
task_class/required_capabilities/input_tokens/reserve_tokens/provenance are
supplied by the operator at `hermes kanban create --routing-role ROLE
--routing-requirements '{...}'` (validated at creation time by
`validate_routing_requirements`/`parse_routing_requirements_json`) and stored
verbatim on `tasks.routing_requirements`. `resolve_task_route` reads that
stored intake back at claim time and builds the real `TaskRequirements`
object from it — never a fixed placeholder shape. Missing/omitted fields
fall back to conservative, DOCUMENTED defaults (unclassified scope defaults
to the deep quality floor per design §3.B/§4 step 2; ordinary non-review
roles are not blocked for lacking review-only contributor evidence), never
a fabricated "complete"/populated shape.
"""
from __future__ import annotations

import json
from typing import Optional

from agent.model_selection import select
from agent.model_selection_guard import managed_child_kwargs, validate_actual_route
from agent.model_selection_store import append_outcome, get_active_policy, get_receipt, persist_receipt
from agent.model_selection_types import REQUIRED_PROVENANCE_FIELDS, TASK_CLASSES, RoutingBlocked

_POLICY_ID = "kanban-default"

# Roles that perform INDEPENDENT REVIEW of another attempt's output (design
# §3.B/§4 step 3: "For independent reviews exclude every maker that
# materially contributed to the implementation under review"). Only these
# roles require genuine contributor-provenance evidence; an ordinary builder
# role has nothing to independently review and must not be blocked for
# lacking it (explicit task requirement: "missing mandatory review
# provenance must fail closed... not block ordinary builders solely for
# lacking review-only evidence"). This is a naming convention consistent
# with the roles used throughout the design doc and existing tests
# (reviewquality, reviewarchitecture, reviewsystem); a role outside this
# convention that still needs independence checking is a policy addition,
# not a per-call guess.
_REVIEW_ROLE_PREFIX = "review"

# `input_tokens`/`reserve_tokens` are the only requirement fields where "0"
# is a meaningful, storable value (an omitted estimate), so validation must
# accept 0 while still rejecting negative/non-integer input.
_MAX_REASONABLE_TOKENS = 50_000_000  # sanity ceiling against unit confusion, not a product limit


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _validate_capabilities(value) -> list:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        _fail("required_capabilities must be a list of non-empty strings")
    # Dedup while preserving order — a duplicated capability is not a
    # distinct requirement and must not silently inflate matching logic.
    seen: set = set()
    out = []
    for v in value:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _validate_token_count(value, field: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    if value < 0:
        _fail(f"{field} must be >= 0")
    if value > _MAX_REASONABLE_TOKENS:
        _fail(f"{field} exceeds sanity ceiling ({_MAX_REASONABLE_TOKENS}); check units")
    return value


def _validate_provenance(value) -> Optional[dict]:
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail("provenance must be an object")
    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in value]
    if missing:
        _fail(f"provenance missing fields: {missing}")
    if not isinstance(value.get("frozen_sha"), str) or not value["frozen_sha"]:
        _fail("provenance.frozen_sha must be a non-empty string")
    if not isinstance(value.get("verified_by"), str) or not value["verified_by"]:
        _fail("provenance.verified_by must be a non-empty string")
    if not isinstance(value.get("complete"), bool):
        _fail("provenance.complete must be a boolean")
    contributors = value.get("contributors")
    if not isinstance(contributors, list):
        _fail("provenance.contributors must be a list")
    for c in contributors:
        if not isinstance(c, dict) or not c.get("maker"):
            _fail("each provenance.contributors entry needs a non-empty 'maker'")
    return {
        "frozen_sha": value["frozen_sha"],
        "verified_by": value["verified_by"],
        "complete": value["complete"],
        "contributors": [dict(c) for c in contributors],
    }


def validate_routing_requirements(value: Optional[dict]) -> Optional[dict]:
    """Validate a user-supplied per-task requirements intake dict.

    Raises ``ValueError`` on any malformed field (unknown task_class,
    negative/non-int token counts, malformed provenance, unexpected keys) so
    a bad intake fails LOUDLY at task-creation time, never silently at claim
    time under a hardcoded fallback. Returns ``None`` unchanged for "no
    explicit intake" (the documented conservative claim-time defaults apply).
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail("routing_requirements must be a JSON object")
    known_fields = {
        "task_class", "required_capabilities", "input_tokens", "reserve_tokens", "provenance",
        # "required" (bool): whether a managed slot's failure/denial must hard-fail the whole
        # attempt (the MoA adapter's default for any slot carrying routing_role) or may degrade
        # to the existing partial/quorum behavior. Optional across every adapter; Kanban/
        # delegation callers simply never set it.
        "required",
    }
    unknown = set(value) - known_fields
    if unknown:
        _fail(f"routing_requirements has unknown fields: {sorted(unknown)}")

    task_class = value.get("task_class")
    if task_class is not None and task_class not in TASK_CLASSES:
        _fail(f"task_class must be one of {TASK_CLASSES} or omitted, got {task_class!r}")

    required = value.get("required")
    if required is not None and not isinstance(required, bool):
        _fail("required must be a boolean")

    out: dict = {}
    if task_class is not None:
        out["task_class"] = task_class
    caps = _validate_capabilities(value.get("required_capabilities"))
    if caps:
        out["required_capabilities"] = caps
    input_tokens = _validate_token_count(value.get("input_tokens"), "input_tokens")
    if input_tokens:
        out["input_tokens"] = input_tokens
    reserve_tokens = _validate_token_count(value.get("reserve_tokens"), "reserve_tokens")
    if reserve_tokens:
        out["reserve_tokens"] = reserve_tokens
    provenance = _validate_provenance(value.get("provenance"))
    if provenance is not None:
        out["provenance"] = provenance
    if required is not None:
        out["required"] = required
    return out or None


def parse_routing_requirements_json(raw: Optional[str]) -> Optional[dict]:
    """CLI boundary: parse+validate a ``--routing-requirements`` JSON string.

    Raises ``ValueError`` (with a JSON-decode detail wrapped in, so the CLI
    error message is actionable) on malformed JSON or a schema violation.
    """
    if raw is None or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON: {exc}")
    return validate_routing_requirements(parsed)


def _is_review_role(role: str) -> bool:
    return role.startswith(_REVIEW_ROLE_PREFIX)


def _build_requirements(task, *, frozen_sha: str, verified_by: str) -> dict:
    """Build the real `TaskRequirements` shape for `task` from its genuinely
    stored per-task intake (`task.routing_requirements`), never a fixed
    fabricated placeholder.

    - `task_class`: the operator-supplied class, or unset. Unset/unknown
      scope deterministically defaults to the "deep" quality floor inside
      `agent.model_selection._quality_floor` (design §3.B: "unclassified or
      incomplete scope defaults to deep") — this function does not itself
      guess a class, it passes through what is actually known.
    - `required_capabilities`/`input_tokens`/`reserve_tokens`: the operator's
      real intake, or the honest "nothing specified" default of `[]`/`0`;
      an unspecified budget is not silently inflated or shrunk.
    - `provenance`: for an independent-review role, MUST come from the
      task's genuine stored intake (real contributor makers + frozen SHA) —
      a review task with no provenance intake fails closed
      (`provenance_incomplete`), it is never granted a fabricated
      `complete=True`. For an ordinary (non-review) role there is nothing to
      independently review, so the honest, non-fabricated shape is
      `contributors=[]`/`complete=True`: no exclusion applies, and this is
      NOT "manufactured completeness" about implementation review evidence
      that was never claimed to exist for a builder task in the first place.
    """
    stored = task.routing_requirements or {}
    role = task.routing_role

    requirements = {
        "schema_version": 1,
        "role": role,
        "execution_kind": "kanban",
        "execution_id": task.id,
        "attempt_id": str(task.current_run_id or 0),
        "slot_id": "",
        # Unknown/omitted task_class is passed through as "" (not in
        # TASK_CLASSES) rather than guessed — select()'s quality-floor logic
        # already treats anything outside DEEP_QUALITY_TASK_CLASSES /
        # "established-pattern" as the conservative deep default.
        "task_class": stored.get("task_class", ""),
        "required_capabilities": list(stored.get("required_capabilities", [])),
        "input_tokens": int(stored.get("input_tokens", 0)),
        "reserve_tokens": int(stored.get("reserve_tokens", 0)),
        "reasoning": task.reasoning_effort or "medium",
    }

    provenance = stored.get("provenance")
    if _is_review_role(role):
        if provenance is None:
            raise RoutingBlocked(
                "provenance_incomplete",
                f"task {task.id}: role {role!r} is an independent-review role and "
                "requires genuine contributor-maker provenance (--routing-requirements "
                "provenance={frozen_sha, verified_by, complete, contributors}); none was "
                "supplied at task creation — refusing to manufacture completeness",
            )
        requirements["provenance"] = provenance
    elif provenance is not None:
        # An operator supplied provenance for a non-review role: honor it as
        # given rather than discarding real evidence.
        requirements["provenance"] = provenance
    else:
        # Ordinary builder/non-review role with no provenance intake: no
        # independence exclusion applies to this task, so the honest shape
        # is "no contributors to exclude", not a claim about review evidence
        # that was never required in the first place.
        requirements["provenance"] = {
            "frozen_sha": frozen_sha,
            "verified_by": verified_by,
            "complete": True,
            "contributors": [],
        }
    return requirements


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

    requirements = _build_requirements(task, frozen_sha=frozen_sha, verified_by=verified_by)
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


# The worker-side half of the guard now lives in `agent.managed_route_runtime` (design §6: "shared
# managed_route_guard currently imports hermes_cli.kanban_model_routing.enforce_worker_route ->
# factor neutral implementation if needed"). Re-exported here UNCHANGED so every existing Kanban
# call site, `unittest.mock.patch("hermes_cli.kanban_model_routing.enforce_worker_route", ...)`
# target and test keeps passing byte-for-byte -- this is a wrapper, not a reimplementation.
from agent.managed_route_runtime import enforce_worker_route  # noqa: F401,E402
