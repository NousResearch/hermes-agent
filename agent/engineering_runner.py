"""Parent-owned inference adapter for the sequential engineering workflow."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from agent.auxiliary_client import call_llm, extract_content_or_reasoning
from agent.engineering_execution import (
    CheckSpec,
    execute_checks,
    run_host_command,
    workspace_digest,
)
from agent.engineering_workflow import (
    EngineeringPlan,
    HandoffError,
    MAX_HANDOFF_BYTES,
    ModelRouteError,
    VerificationContext,
    WorkerOutcome,
    WorkflowLimits,
    _unique_object,
    run_engineering_workflow,
)


class WorkerActionError(ValueError):
    """The worker response is not one bounded action or terminal status."""


@dataclass(frozen=True)
class WorkerAction:
    status: str
    summary: str
    argv: tuple[str, ...] = ()
    decision_required: str = ""


_ACTION_KEYS = {
    "RUN": frozenset({"status", "summary", "argv"}),
    "READY": frozenset({"status", "summary"}),
    "BLOCKED": frozenset({"status", "summary", "decision_required"}),
}


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2_048:
        raise WorkerActionError(f"invalid {field}")
    if any(
        mark in value.casefold()
        for mark in ("bearer ", "api_key=", "password=", "secret=")
    ):
        raise WorkerActionError(f"{field} contains credential material")
    return value.strip()


def parse_worker_action(payload: str) -> WorkerAction:
    """Reject extra keys, duplicate keys, invented statuses, and oversized argv."""
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_HANDOFF_BYTES:
        raise WorkerActionError("worker action exceeds size limit")
    try:
        document = json.loads(payload, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise WorkerActionError("invalid worker JSON") from exc
    if not isinstance(document, dict):
        raise WorkerActionError("worker action must be an object")
    status = document.get("status")
    if (
        not isinstance(status, str)
        or status not in _ACTION_KEYS
        or document.keys() != _ACTION_KEYS[status]
    ):
        raise WorkerActionError("unexpected worker status or schema")
    summary = _text(document["summary"], "summary")
    if status == "RUN":
        args = document["argv"]
        if (
            not isinstance(args, list)
            or not 1 <= len(args) <= 32
            or any(
                not isinstance(arg, str) or not arg or len(arg) > 2_048 for arg in args
            )
            or sum(len(arg) for arg in args) > 16_384
        ):
            raise WorkerActionError("invalid worker argv")
        markers = (
            "bearer ",
            "api_key",
            "apikey",
            "password",
            "secret=",
            "--token",
            "authorization:",
        )
        if any(any(marker in arg.casefold() for marker in markers) for arg in args):
            raise WorkerActionError("worker argv contains credential syntax")
        known_secrets = (
            value
            for name, value in os.environ.items()
            if len(value) >= 8
            and any(
                word in name.upper()
                for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
            )
        )
        if any(value in arg for value in known_secrets for arg in args):
            raise WorkerActionError("worker argv contains parent credential material")
        return WorkerAction(status=status, summary=summary, argv=tuple(args))
    if status == "BLOCKED":
        return WorkerAction(
            status=status,
            summary=summary,
            decision_required=_text(document["decision_required"], "decision_required"),
        )
    return WorkerAction(status=status, summary=summary)


_STAGE_INSTRUCTIONS = {
    "planner": (
        "Plan the engineering task. Copy the operator objective exactly. Return one JSON object with exactly objective, constraints, "
        "steps, and acceptance_criteria. Each list contains short strings. Do not include "
        "credentials, auth configuration, model names, or verification claims."
    ),
    "worker": (
        "Work through the plan one step at a time. Return one JSON object: "
        "RUN with status, summary, argv (an executable and arguments, never a shell string); "
        "READY with status and summary when work is complete; or BLOCKED with status, summary, "
        "decision_required when the plan conflicts with a real constraint. "
        "A RUN action is executed without provider credentials. You receive its bounded result "
        "on the next call. Do not claim tests passed; the host will run its own checks."
    ),
    "reviewer": (
        "Review the failed host checks and revise the plan. Keep the operator objective exactly. Return exactly objective, constraints, "
        "steps, and acceptance_criteria as bounded JSON. Do not claim a check passed."
    ),
}


def run_project_workflow(
    *,
    objective: str,
    assignments: dict,
    workspace: Path,
    checks: tuple[CheckSpec, ...],
    backend: str = "docker",
    image: str = "",
    limits: WorkflowLimits = WorkflowLimits(),
    stop_requested=None,
):
    """Use Hermes's live picker and auxiliary provider path for all model stages."""
    objective = _text(objective, "objective")
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace is not a directory")
    if not checks or len({check.check_id for check in checks}) != len(checks):
        raise ValueError("at least one distinct host check is required")
    workspace_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()

    def catalogue_reader():
        from hermes_cli.inventory import (
            build_model_options_payload,
            load_picker_context,
        )

        return build_model_options_payload(
            load_picker_context(),
            explicit_only=True,
            refresh=True,
        )

    def infer(stage, route, payload):
        route_info = {}
        response = call_llm(
            task="engineering_workflow",
            provider=route.provider,
            model=route.model,
            messages=[
                {"role": "system", "content": _STAGE_INSTRUCTIONS[stage]},
                {"role": "user", "content": payload},
            ],
            max_tokens=2_048,
            timeout=120.0,
            strict_route=True,
            route_info=route_info,
        )
        if route_info != {"provider": route.provider, "model": route.model}:
            raise ModelRouteError("provider changed the operator-selected model")
        return extract_content_or_reasoning(response)

    def execute_worker(
        text: str,
        context: VerificationContext,
        next_worker,
        plan: EngineeringPlan,
    ) -> WorkerOutcome:
        while True:
            action = parse_worker_action(text)
            if action.status == "BLOCKED":
                return WorkerOutcome(
                    status="BLOCKED",
                    summary=action.summary,
                    decision_required=action.decision_required,
                )
            if action.status == "READY":
                return WorkerOutcome(status="READY", summary=action.summary)
            result = run_host_command(
                action.argv,
                root,
                backend=backend,
                image=image,
                timeout=120,
                stop_requested=stop_requested,
            )
            if not result.complete or result.timed_out:
                return WorkerOutcome(
                    status="BLOCKED",
                    summary="worker process did not complete",
                    decision_required="Inspect the failed execution backend before continuing.",
                )
            feedback = json.dumps(
                {
                    "plan": {
                        "objective": plan.objective,
                        "constraints": plan.constraints,
                        "steps": plan.steps,
                        "acceptance_criteria": plan.acceptance_criteria,
                    },
                    "last_command": action.argv,
                    "exit_code": result.exit_code,
                    "output": result.output,
                    "attempt_id": context.attempt_id,
                    "revision": context.revision,
                },
                ensure_ascii=False,
            )
            text = next_worker(feedback)

    def verify(context: VerificationContext):
        return execute_checks(
            context,
            checks,
            root,
            backend=backend,
            image=image,
            snapshot_digest=lambda: workspace_digest(root),
            stop_requested=stop_requested,
        )

    return run_engineering_workflow(
        objective=objective,
        assignments=assignments,
        catalogue_reader=catalogue_reader,
        infer=infer,
        execute_worker=execute_worker,
        verify=verify,
        snapshot_digest=lambda: workspace_digest(root),
        workspace_id=workspace_id,
        check_ids=tuple(check.check_id for check in checks),
        limits=limits,
        stop_requested=stop_requested,
    )
