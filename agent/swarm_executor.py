"""Bounded synchronous fan-out wrapper for Jeeves swarm operator.

This is a thin adapter around an injected delegate function. It does not import
or invoke the live delegate gateway path directly; production wiring can pass the
existing ``delegate_task`` function later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent.swarm_evidence import EvidencePacket, validate_evidence_packet
from agent.swarm_learning import detect_weak_output
from agent.swarm_state import AuditEvent, JobStatus, RoutingPlan, SwarmJob, SwarmTask, TaskStatus
from agent.swarm_synthesis import synthesize_swarm_result


@dataclass
class SwarmExecutionResult:
    dispatched: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    blocked: List[Dict[str, Any]] = field(default_factory=list)
    status: str = JobStatus.COMPLETED.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispatched": [dict(item) for item in self.dispatched],
            "results": [dict(item) for item in self.results],
            "blocked": [dict(item) for item in self.blocked],
            "status": self.status,
        }


def _task_requires_permission(item: Dict[str, Any]) -> bool:
    return bool(
        item.get("permission_required")
        or item.get("requires_permission")
        or item.get("permission")
        or item.get("permission_request")
    )


def _coerce_toolsets(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


def _format_evidence_contract(routing_plan: RoutingPlan) -> str:
    requirements = getattr(routing_plan, "evidence_requirements", []) or []
    lines = [
        "",
        "Required evidence contract:",
        "Return a JSON-compatible result with keys: summary, claims, evidence, assumptions, unresolved, risks, confidence, side_effects_requested, side_effects_performed.",
        "Each evidence item must include kind plus kind-specific fields such as command/output, url, path, approved_by, or approval_id.",
        "If you cannot provide required evidence, say so explicitly in unresolved; do not claim completion without proof.",
        "Required evidence:",
    ]
    if not requirements:
        lines.append("- none explicitly required; still include evidence for material claims when available.")
    for item in requirements:
        if hasattr(item, "to_dict"):
            data = item.to_dict()
        elif isinstance(item, dict):
            data = item
        else:
            continue
        required = "required" if data.get("required", True) else "optional"
        lines.append(f"- {data.get('kind', 'artifact')} ({required}): {data.get('description', '')}")
    return "\n".join(lines)


def build_delegate_tasks(
    routing_plan: RoutingPlan,
    job: SwarmJob,
    *,
    max_children: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Translate routing suggestions into delegate_task batch entries.

    Permission-required suggestions are excluded here and marked blocked by
    ``execute_swarm``. Returned entries preserve per-task toolsets.
    """
    limit = max(0, int(max_children if max_children is not None else 3))
    delegates: List[Dict[str, Any]] = []
    for index, item in enumerate(routing_plan.suggested_tasks or [], start=1):
        if not isinstance(item, dict) or _task_requires_permission(item):
            continue
        title = str(item.get("title") or item.get("goal") or f"Task {index}")
        description = str(item.get("description") or item.get("context") or title)
        context = description + _format_evidence_contract(routing_plan)
        delegates.append(
            {
                "goal": title,
                "context": context,
                "toolsets": _coerce_toolsets(item.get("toolsets")),
                "swarm_task_id": str(item.get("task_id") or f"{job.job_id}_task_{index}"),
            }
        )
        if len(delegates) >= limit:
            break
    return delegates


def _ensure_job_tasks(job: SwarmJob, routing_plan: RoutingPlan) -> None:
    if job.tasks:
        return
    for index, item in enumerate(routing_plan.suggested_tasks or [], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("goal") or f"Task {index}")
        description = str(item.get("description") or item.get("context") or title)
        job.add_task(
            title,
            description,
            task_id=str(item.get("task_id") or f"{job.job_id}_task_{index}"),
            mode=str(item.get("mode") or routing_plan.mode),
            toolsets=_coerce_toolsets(item.get("toolsets")),
            permission_required=_task_requires_permission(item),
        )


def _parse_delegate_response(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, dict):
        if isinstance(response.get("results"), list):
            return [dict(item) for item in response["results"] if isinstance(item, dict)]
        return [dict(response)]
    if isinstance(response, list):
        return [dict(item) for item in response if isinstance(item, dict)]
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return [{"ok": True, "content": response}]
        return _parse_delegate_response(parsed)
    return [{"ok": True, "content": repr(response)}]


def _result_failed(result: Dict[str, Any]) -> bool:
    status = str(result.get("status") or "").lower()
    if status in {"failed", "error", "cancelled", "timeout"}:
        return True
    if result.get("error"):
        return True
    if result.get("ok") is False:
        return True
    return False


def execute_swarm(
    job: SwarmJob,
    routing_plan: RoutingPlan,
    delegate_fn: Callable[..., Any],
    *,
    max_children: int = 3,
) -> SwarmExecutionResult:
    """Run bounded independent tasks through an injected delegate function.

    Updates ``job`` in-place and returns a structured summary. Critical task
    failures (``metadata.critical`` or suggested task ``critical``) fail the job;
    non-critical partial failures mark it ``partially_completed``.
    """
    job.routing_plan = routing_plan
    _ensure_job_tasks(job, routing_plan)

    blocked: List[Dict[str, Any]] = []
    for task in job.tasks:
        if task.permission_required:
            task.status = TaskStatus.BLOCKED.value
            task.updated_at = job.updated_at
            blocked.append({"task_id": task.task_id, "title": task.title, "reason": "permission_required"})

    delegates = build_delegate_tasks(routing_plan, job, max_children=max_children)
    if blocked:
        job.audit.append(AuditEvent("tasks_blocked", "Permission-required tasks were not dispatched", metadata={"blocked": blocked}))

    if not delegates:
        status = JobStatus.AWAITING_PERMISSION.value if blocked else JobStatus.COMPLETED.value
        job.transition(status, metadata={"blocked_count": len(blocked), "dispatched_count": 0})
        result = SwarmExecutionResult(dispatched=[], results=[], blocked=blocked, status=job.status)
        job.metadata["swarm_execution"] = result.to_dict()
        synthesize_swarm_result(job, persist=True)
        return result

    job.transition(JobStatus.RUNNING.value, metadata={"dispatched_count": len(delegates), "max_children": max_children})
    by_id: Dict[str, SwarmTask] = {task.task_id: task for task in job.tasks}
    for delegate in delegates:
        task = by_id.get(str(delegate.get("swarm_task_id") or ""))
        if task:
            task.status = TaskStatus.RUNNING.value

    try:
        raw = delegate_fn(tasks=[{k: v for k, v in item.items() if k != "swarm_task_id"} for item in delegates])
        results = _parse_delegate_response(raw)
    except Exception as exc:
        results = [{"error": str(exc), "status": "failed", "critical": True}]

    failures = 0
    weak_count = 0
    critical_failure = False
    for index, delegate in enumerate(delegates):
        task = by_id.get(str(delegate.get("swarm_task_id") or ""))
        result = results[index] if index < len(results) else {"error": "missing child result", "status": "failed"}
        failed = _result_failed(result)
        requirements = getattr(routing_plan, "evidence_requirements", []) or []
        if requirements:
            evidence_validation = validate_evidence_packet(EvidencePacket.from_result(result), requirements)
            if isinstance(result, dict):
                result = {**result, "evidence_validation": evidence_validation.to_dict()}
                if index < len(results):
                    results[index] = result
        weak_output = detect_weak_output(result, requirements)
        if weak_output["weak"]:
            result = {**result, "weak_output": weak_output}
            if index < len(results):
                results[index] = result
        weak = bool(weak_output["weak"])
        weak_count += 1 if weak else 0
        failures += 1 if failed else 0
        suggested = (routing_plan.suggested_tasks or [])[index] if index < len(routing_plan.suggested_tasks or []) else {}
        is_critical = bool(result.get("critical") or (isinstance(suggested, dict) and suggested.get("critical")))
        critical_failure = critical_failure or (failed and is_critical)
        if task:
            task.result = dict(result)
            task.status = TaskStatus.FAILED.value if failed else (TaskStatus.NEEDS_REVIEW.value if weak else TaskStatus.COMPLETED.value)
            if weak:
                job.audit.append(AuditEvent("weak_child_output", "Child result lacked required evidence", metadata={"task_id": task.task_id, **weak_output}))

    if critical_failure:
        final_status = JobStatus.FAILED.value
    elif failures or blocked or weak_count:
        final_status = JobStatus.PARTIALLY_COMPLETED.value
    else:
        final_status = JobStatus.COMPLETED.value
    job.transition(final_status, metadata={"failure_count": failures, "blocked_count": len(blocked), "weak_output_count": weak_count})

    execution = SwarmExecutionResult(dispatched=delegates, results=results, blocked=blocked, status=job.status)
    job.metadata["swarm_execution"] = execution.to_dict()
    synthesize_swarm_result(job, persist=True)
    return execution


__all__ = ["SwarmExecutionResult", "build_delegate_tasks", "execute_swarm"]
