"""Human-readable status formatting for Jeeves swarm operator jobs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from agent.swarm_state import SwarmJob
from agent.swarm_store import SwarmStore
from agent.swarm_synthesis import synthesize_swarm_result

_SECRET_MARKERS = ("secret", "token", "key", "password", "credential", "api_key", "auth", "bearer")


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def redact_secrets(value: Any) -> Any:
    """Recursively redact secret-looking metadata values."""
    if isinstance(value, Mapping):
        return {str(k): ("[REDACTED]" if _is_secret_key(str(k)) else redact_secrets(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return value


def _short(text: str, limit: int = 96) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _format_metadata(metadata: Dict[str, Any]) -> str:
    redacted = redact_secrets(metadata or {})
    if not redacted:
        return ""
    parts = []
    for key in sorted(redacted):
        value = redacted[key]
        parts.append(f"{key}={_short(repr(value), 80)}")
    return ", ".join(parts)


def format_swarm_status(jobs: Iterable[SwarmJob], *, include_completed: bool = False) -> str:
    """Format active swarm jobs with tasks, blockers, permissions and last event."""
    selected: List[SwarmJob] = []
    terminal = {"completed", "failed", "cancelled", "partially_completed"}
    for job in jobs:
        if include_completed or job.status not in terminal:
            selected.append(job)

    if not selected:
        return "No active swarm operator jobs."

    lines: List[str] = ["Swarm operator status:"]
    for job in selected:
        lines.append(f"- {job.job_id}: {job.status} — {_short(job.original_request)}")
        if job.routing_plan:
            lines.append(f"  route: {job.routing_plan.mode} ({_short(job.routing_plan.reason)})")
        if job.tasks:
            lines.append("  tasks:")
            for task in job.tasks:
                marker = " blocked" if task.status in {"blocked", "awaiting_permission"} or task.permission_required else ""
                result = ""
                if isinstance(task.result, dict):
                    if task.result.get("summary"):
                        result = f" — {_short(str(task.result.get('summary')))}"
                    elif task.result.get("error"):
                        result = f" — error: {_short(str(task.result.get('error')))}"
                lines.append(f"    - {task.task_id}: {task.status}{marker} — {_short(task.title)}{result}")
        blockers = [task for task in job.tasks if task.status in {"blocked", "awaiting_permission"} or task.permission_required]
        if blockers:
            lines.append("  blockers:")
            for task in blockers:
                lines.append(f"    - {task.task_id}: permission required for {_short(task.title)}")
        permissions = list(job.permissions)
        if job.routing_plan:
            permissions.extend(job.routing_plan.permission_requests)
        if permissions:
            lines.append("  permission requests:")
            for grant in permissions:
                scope = _format_metadata(grant.scope)
                suffix = f" ({scope})" if scope else ""
                lines.append(f"    - {grant.permission_id}: {grant.status} — {_short(grant.description)}{suffix}")
        if job.audit:
            event = job.audit[-1]
            metadata = _format_metadata(event.metadata)
            suffix = f" [{metadata}]" if metadata else ""
            lines.append(f"  last event: {event.event_type} — {_short(event.message)}{suffix}")
        if job.routing_plan and job.routing_plan.evidence_requirements:
            synthesis = synthesize_swarm_result(job)
            lines.append("  evidence:")
            for kind, count in sorted(synthesis.missing_evidence.items()):
                lines.append(f"    - {kind}: missing ({count})")
            if not synthesis.missing_evidence:
                lines.append("    - required evidence satisfied")
            lines.append(f"  safe to present complete: {'yes' if synthesis.safe_to_present_complete else 'no'}")
    return "\n".join(lines)


def load_swarm_status_text(*, session_id: str = "", include_completed: bool = False) -> str:
    """Load persisted swarm jobs and format them for status surfaces.

    Fail closed: callers such as gateway /status should never error because the
    optional swarm status file is absent or unreadable.
    """
    try:
        jobs = list(SwarmStore().load_jobs().values())
    except Exception:
        return ""
    if session_id:
        scoped = [job for job in jobs if job.session_id == session_id]
        if not scoped:
            return ""
        jobs = scoped
    text = format_swarm_status(jobs, include_completed=include_completed)
    return "" if text == "No active swarm operator jobs." else text


__all__ = ["format_swarm_status", "load_swarm_status_text", "redact_secrets"]
