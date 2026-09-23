"""Bounded engineering handoffs and host-owned verification contracts.

The router consumes only these validated objects. Model text is never a
verification receipt and cannot choose a provider or an executable command.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


MAX_HANDOFF_BYTES = 16_384
MAX_FIELD_CHARS = 2_048
MAX_LIST_ITEMS = 32
_PLAN_KEYS = frozenset({"objective", "constraints", "steps", "acceptance_criteria"})


class HandoffError(ValueError):
    """The model's structured handoff cannot be admitted."""


class ReceiptError(ValueError):
    """A host verification receipt is missing, stale, or malformed."""


@dataclass(frozen=True)
class EngineeringPlan:
    objective: str
    constraints: tuple[str, ...]
    steps: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]


@dataclass(frozen=True)
class VerificationContext:
    run_id: str
    workspace_id: str
    attempt_id: str
    revision: int
    snapshot_digest: str
    check_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerificationReceipt:
    run_id: str
    workspace_id: str
    attempt_id: str
    revision: int
    snapshot_digest: str
    check_id: str
    exit_code: int
    complete: bool
    timed_out: bool


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise HandoffError("duplicate JSON key")
        result[key] = value
    return result


def _bounded_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_FIELD_CHARS:
        raise HandoffError(f"invalid {field}")
    if any(marker in value.casefold() for marker in ("bearer ", "api_key=", "password=", "secret=")):
        raise HandoffError(f"{field} contains credential material")
    return value.strip()


def _bounded_list(value: object, *, field: str, required: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS or (required and not value):
        raise HandoffError(f"invalid {field}")
    return tuple(_bounded_text(item, field=field) for item in value)


def parse_handoff(payload: str) -> EngineeringPlan:
    """Admit one strict planner/reviewer handoff; reject ambiguous JSON and extra fields."""
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_HANDOFF_BYTES:
        raise HandoffError("handoff exceeds size limit")
    try:
        document = json.loads(payload, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError, TypeError, RecursionError) as exc:
        raise HandoffError("invalid handoff JSON") from exc
    if not isinstance(document, dict) or document.keys() != _PLAN_KEYS:
        raise HandoffError("handoff schema mismatch")
    return EngineeringPlan(
        objective=_bounded_text(document["objective"], field="objective"),
        constraints=_bounded_list(document["constraints"], field="constraints", required=False),
        steps=_bounded_list(document["steps"], field="steps", required=True),
        acceptance_criteria=_bounded_list(
            document["acceptance_criteria"], field="acceptance_criteria", required=True,
        ),
    )


def verify_receipts(
    expected: VerificationContext, receipts: Iterable[VerificationReceipt],
) -> bool:
    """Validate exact host provenance; return pass/fail only for complete real exits.

    Any provenance or process-state error is distinct from a finished check
    returning a nonzero exit status. An empty check set cannot certify a run.
    """
    if (
        not expected.run_id or not expected.workspace_id or not expected.attempt_id
        or not expected.snapshot_digest or type(expected.revision) is not int
        or not expected.check_ids or len(set(expected.check_ids)) != len(expected.check_ids)
    ):
        raise ReceiptError("invalid verification context")
    supplied = list(receipts)
    if len(supplied) != len(expected.check_ids):
        raise ReceiptError("missing or duplicate verification receipt")
    seen = set()
    passed = True
    for receipt in supplied:
        if not isinstance(receipt, VerificationReceipt):
            raise ReceiptError("untrusted verification receipt")
        if receipt.check_id not in expected.check_ids or receipt.check_id in seen:
            raise ReceiptError("unexpected or duplicate check")
        seen.add(receipt.check_id)
        if (
            receipt.run_id != expected.run_id
            or receipt.workspace_id != expected.workspace_id
            or receipt.attempt_id != expected.attempt_id
            or type(receipt.revision) is not int
            or receipt.revision != expected.revision
            or receipt.snapshot_digest != expected.snapshot_digest
        ):
            raise ReceiptError("foreign or stale verification receipt")
        if (
            type(receipt.exit_code) is not int
            or type(receipt.complete) is not bool
            or type(receipt.timed_out) is not bool
            or not receipt.complete or receipt.timed_out
        ):
            raise ReceiptError("incomplete or invalid verification process")
        passed &= receipt.exit_code == 0
    return passed


class ModelRouteError(ValueError):
    """An operator-selected stage route is no longer in the live picker."""


@dataclass(frozen=True)
class StageRoute:
    stage: str
    provider: str
    model: str


_STAGES = frozenset({"planner", "worker", "reviewer"})


def revalidate_route(route: StageRoute, catalogue: dict) -> None:
    """Check a selected pair against the current shared Hermes model inventory."""
    if route.provider in {"", "auto", "moa"} or not route.model:
        raise ModelRouteError("stage requires a concrete picker model")
    rows = catalogue.get("providers") if isinstance(catalogue, dict) else None
    if not isinstance(rows, list):
        raise ModelRouteError("model catalogue unavailable")
    matches = [row for row in rows if isinstance(row, dict) and row.get("slug") == route.provider]
    if len(matches) != 1:
        raise ModelRouteError("selected provider is absent or ambiguous")
    row = matches[0]
    if (
        row.get("authenticated") is not True
        or route.model not in (row.get("models") or [])
        or route.model in (row.get("unavailable_models") or [])
    ):
        raise ModelRouteError("selected model is unavailable")


def admit_stage_routes(assignments: dict, catalogue: dict) -> dict[str, StageRoute]:
    """Freeze three operator picks; callers revalidate before every stage call."""
    if not isinstance(assignments, dict) or assignments.keys() != _STAGES:
        raise ModelRouteError("planner, worker, and reviewer routes are required")
    routes = {}
    for stage in ("planner", "worker", "reviewer"):
        row = assignments[stage]
        if not isinstance(row, dict) or row.keys() != {"provider", "model"}:
            raise ModelRouteError("stage route schema mismatch")
        provider, model = row["provider"], row["model"]
        if not isinstance(provider, str) or not isinstance(model, str):
            raise ModelRouteError("stage route must use picker strings")
        route = StageRoute(stage=stage, provider=provider, model=model)
        revalidate_route(route, catalogue)
        routes[stage] = route
    return routes


@dataclass(frozen=True)
class WorkflowLimits:
    max_attempts: int = 3
    max_replans: int = 2
    max_stage_calls: int = 7


@dataclass(frozen=True)
class WorkerOutcome:
    status: str
    summary: str
    decision_required: str = ""


@dataclass(frozen=True)
class WorkflowResult:
    status: str
    reason: str
    run_id: str
    workspace_id: str
    attempts: int
    revision: int
    stage_calls: int
    decision_required: str = ""


def _plan_payload(plan: EngineeringPlan) -> str:
    return json.dumps(
        {
            "objective": plan.objective,
            "constraints": plan.constraints,
            "steps": plan.steps,
            "acceptance_criteria": plan.acceptance_criteria,
        },
        ensure_ascii=False,
    )


def run_engineering_workflow(
    *, objective: str, assignments: dict, catalogue_reader, infer, execute_worker,
    verify, snapshot_digest, workspace_id: str, check_ids: tuple[str, ...],
    limits: WorkflowLimits = WorkflowLimits(), stop_requested=None,
) -> WorkflowResult:
    """Run a finite planner, worker, host verifier, and reviewer sequence.

    Inference stays in the parent process. Worker execution and verification
    are host callbacks; their result cannot be supplied by a model.
    The catalogue is read at admission and before each model call.
    """
    import uuid

    if (
        type(limits.max_attempts) is not int or limits.max_attempts < 1
        or type(limits.max_replans) is not int or limits.max_replans < 0
        or type(limits.max_stage_calls) is not int or limits.max_stage_calls < 1
    ):
        raise ValueError("workflow limits must be finite nonnegative integers")
    if not workspace_id or not check_ids:
        raise ValueError("workspace and checks are required")

    run_id = uuid.uuid4().hex
    calls = attempts = replans = 0
    revision = 1

    def finish(status: str, reason: str, decision_required: str = "") -> WorkflowResult:
        return WorkflowResult(
            status=status, reason=reason, run_id=run_id, workspace_id=workspace_id,
            attempts=attempts, revision=revision, stage_calls=calls,
            decision_required=decision_required,
        )

    def stopped() -> bool:
        return bool(stop_requested and stop_requested())

    class _Stopped(Exception):
        pass

    def call_stage(stage: str, payload: str) -> str:
        nonlocal calls
        if stopped() or calls >= limits.max_stage_calls:
            raise _Stopped
        revalidate_route(routes[stage], catalogue_reader())
        calls += 1
        return infer(stage, routes[stage], payload)

    try:
        routes = admit_stage_routes(assignments, catalogue_reader())
        plan = parse_handoff(call_stage("planner", objective))
    except _Stopped:
        return finish("STOP", "stage_budget_or_interrupt")
    except (HandoffError, ModelRouteError):
        return finish("BLOCKED", "invalid_plan_or_model")
    except Exception:
        return finish("BLOCKED", "stage_unavailable")

    while attempts < limits.max_attempts:
        if stopped():
            return finish("STOP", "interrupt")
        attempt_id = uuid.uuid4().hex
        try:
            worker_text = call_stage("worker", _plan_payload(plan))
            attempts += 1
            precheck = VerificationContext(
                run_id=run_id, workspace_id=workspace_id, attempt_id=attempt_id,
                revision=revision, snapshot_digest="", check_ids=check_ids,
            )
            outcome = execute_worker(worker_text, precheck)
            if not isinstance(outcome, WorkerOutcome) or outcome.status not in {"READY", "BLOCKED"}:
                return finish("BLOCKED", "invalid_worker_status")
            if outcome.status == "BLOCKED":
                if not outcome.decision_required.strip():
                    return finish("BLOCKED", "missing_worker_decision")
                return finish("BLOCKED", "worker_blocked", outcome.decision_required)
            digest = snapshot_digest()
            if not isinstance(digest, str) or not digest:
                return finish("BLOCKED", "workspace_snapshot_unavailable")
            context = VerificationContext(
                run_id=run_id, workspace_id=workspace_id, attempt_id=attempt_id,
                revision=revision, snapshot_digest=digest, check_ids=check_ids,
            )
            passed = verify_receipts(context, verify(context))
            if passed:
                return finish("DONE", "verified")
            if attempts >= limits.max_attempts or replans >= limits.max_replans:
                return finish("STOP", "retry_or_replan_limit")
            reviewer_input = json.dumps(
                {
                    "objective": objective,
                    "plan": json.loads(_plan_payload(plan)),
                    "verification": "failed",
                    "attempt_id": attempt_id,
                    "revision": revision,
                },
                ensure_ascii=False,
            )
            plan = parse_handoff(call_stage("reviewer", reviewer_input))
            replans += 1
            revision += 1
        except _Stopped:
            return finish("STOP", "stage_budget_or_interrupt")
        except (HandoffError, ModelRouteError, ReceiptError):
            return finish("BLOCKED", "invalid_handoff_model_or_receipt")
        except Exception:
            return finish("BLOCKED", "stage_unavailable")
    return finish("STOP", "retry_limit")
