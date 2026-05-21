"""Verification planning helpers for the Jeeves swarm operator.

The planner is deliberately deterministic and side-effect free. It records what
should be checked after direct/script/swarm work, but it does not execute tests or
call external services; callers provide verification outcomes explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

from agent.swarm_state import AuditEvent, EvalResult, JobStatus, RoutingPlan, SwarmJob


@dataclass
class VerificationPlan:
    """A compact, serializable checklist for parent-owned verification."""

    required: bool
    steps: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "steps": [dict(step) for step in self.steps],
            "reason": self.reason,
        }

    def summary(self, *, results: Mapping[str, Any] | None = None) -> dict[str, Any]:
        results = results or {}
        total = len(self.steps)
        passed = 0
        failed = 0
        for step in self.steps:
            step_id = str(step.get("id") or "")
            if step_id not in results:
                continue
            result = _coerce_result(step_id, results[step_id])
            if result.passed:
                passed += 1
            else:
                failed += 1
        if failed:
            status = "failed"
        elif total and passed == total:
            status = "passed"
        elif not self.required and not total:
            status = "skipped"
        else:
            status = "pending"
        return {"status": status, "total": total, "passed": passed, "failed": failed}


def _step(step_id: str, title: str, *, required: bool = True, source: str = "parent", **metadata: Any) -> dict[str, Any]:
    return {"id": step_id, "title": title, "required": required, "source": source, **metadata}


def _job_text(job: SwarmJob, plan: RoutingPlan | None) -> str:
    chunks: list[str] = [job.original_request]
    if plan:
        chunks.extend([plan.mode, plan.reason])
        chunks.extend(str(item) for item in plan.suggested_tasks)
        chunks.extend(grant.description for grant in plan.permission_requests)
    for task in job.tasks:
        chunks.extend([task.title, task.description, task.mode, str(task.metadata)])
    return " ".join(chunks).lower()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def build_verification_plan(job: SwarmJob, routing_plan: RoutingPlan | None = None) -> VerificationPlan:
    """Build a conservative verification checklist for a swarm job."""

    plan = routing_plan or job.routing_plan
    required = bool(getattr(plan, "verification_required", False)) if plan else False
    mode = str(getattr(plan, "mode", "direct") or "direct") if plan else "direct"
    text = _job_text(job, plan)
    steps: list[dict[str, Any]] = []

    if mode in {"swarm", "script", "pipe"} or required:
        steps.append(_step("final_consistency_check", "Verify final synthesis matches the original request"))
    else:
        steps.append(_step("direct_sanity_check", "Sanity-check the direct answer before final response", required=False))

    if job.tasks:
        steps.append(_step("child_result_review", "Review child task results and evidence", source="children"))
    if any(task.status in {"blocked", "awaiting_permission"} or task.permission_required for task in job.tasks):
        steps.append(_step("permission_blocker_review", "Confirm blocked tasks are disclosed and not executed"))
    if any(task.result and (task.result.get("error") or task.result.get("ok") is False) for task in job.tasks):
        steps.append(_step("failure_review", "Account for failed child results in the final outcome"))
    if job.permissions or (plan and getattr(plan, "permission_requests", None)):
        steps.append(_step("permission_request_review", "Review outstanding permission requests"))
    if _contains_any(text, ("code", "config", "file edit", "patch", "implement", "test", "lint", "typecheck")):
        steps.append(_step("code_config_verification", "Require test/lint/typecheck evidence or an explicit skipped-verification reason"))
    if mode in {"script", "pipe"} or _contains_any(text, ("n8n", "docker", "script", "workflow", "webhook", "pipe")):
        steps.append(_step("dry_run_payload_contract", "Require dry-run output or payload-contract evidence before live execution"))
    if _contains_any(text, ("send", "email", "slack message", "client-facing", "external", "publish", "deploy")):
        steps.append(_step("human_approval_review", "Require reviewer or human approval before external/client-facing action"))

    for requirement in getattr(plan, "evidence_requirements", []) or []:
        kind = str(getattr(requirement, "kind", "") or "artifact")
        description = str(getattr(requirement, "description", "") or f"Provide {kind} evidence")
        required_flag = bool(getattr(requirement, "required", True))
        source = str(getattr(requirement, "source", "router") or "router")
        steps.append(
            _step(
                f"evidence_{kind}",
                description,
                required=required_flag,
                source=source,
                evidence_kind=kind,
                evidence_metadata=dict(getattr(requirement, "metadata", {}) or {}),
            )
        )

    # Stable de-duplication if several signals produced the same step.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in steps:
        step_id = str(item.get("id"))
        if step_id not in seen:
            seen.add(step_id)
            deduped.append(item)

    return VerificationPlan(required=required or any(step.get("required") for step in deduped), steps=deduped, reason=(plan.reason if plan else ""))


def _coerce_result(name: str, raw: Any) -> EvalResult:
    if isinstance(raw, EvalResult):
        return raw
    if isinstance(raw, tuple):
        passed = bool(raw[0]) if raw else False
        details = str(raw[1]) if len(raw) > 1 else ""
        return EvalResult(name=name, passed=passed, details=details)
    if isinstance(raw, Mapping):
        return EvalResult(
            name=str(raw.get("name") or name),
            passed=bool(raw.get("passed", False)),
            details=str(raw.get("details") or raw.get("summary") or ""),
            metadata=dict(raw.get("metadata") or {}),
        )
    return EvalResult(name=name, passed=bool(raw), details="")


def apply_verification_results(job: SwarmJob, verification: VerificationPlan, results: Mapping[str, Any]) -> dict[str, Any]:
    """Persist verification results on ``job`` without executing checks."""

    coerced: dict[str, EvalResult] = {}
    for step in verification.steps:
        step_id = str(step.get("id") or "")
        if not step_id or step_id not in results:
            continue
        eval_result = _coerce_result(step_id, results[step_id])
        coerced[eval_result.name] = eval_result
        job.evals.append(eval_result)

    summary = verification.summary(results=coerced)
    job.metadata["swarm_verification"] = {**verification.to_dict(), **summary}
    job.audit.append(AuditEvent("verification_recorded", "Swarm verification results recorded.", metadata=summary))

    if summary["status"] == "failed" and job.status not in {JobStatus.FAILED.value, JobStatus.CANCELLED.value}:
        job.transition(JobStatus.PARTIALLY_COMPLETED.value, "Verification failed", metadata=summary)
    elif summary["status"] == "passed" and job.status in {JobStatus.RUNNING.value, JobStatus.VERIFYING.value}:
        job.transition(JobStatus.COMPLETED.value, "Verification passed", metadata=summary)
    return summary


__all__ = ["VerificationPlan", "apply_verification_results", "build_verification_plan"]
