"""Inert Honcho summary adapter for Jeeves swarm operator.

This module never imports or initializes Honcho by default. Production wiring may
pass an injected writer; without one, persistence is a no-op unless a compatible
memory plugin is present and explicitly enabled by the caller.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from agent.swarm_state import SwarmJob
from agent.swarm_status import redact_secrets

_SCRATCH_KEYS = {"scratchpad", "scratch", "chain_of_thought", "reasoning", "messages", "transcript"}


def _short(text: Any, limit: int = 240) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _safe_result_summary(result: Mapping[str, Any] | None) -> str:
    if not isinstance(result, Mapping):
        return ""
    cleaned = {k: v for k, v in result.items() if str(k).lower() not in _SCRATCH_KEYS}
    redacted = redact_secrets(cleaned)
    if redacted.get("summary"):
        return _short(redacted["summary"])
    if redacted.get("error"):
        return "error: " + _short(redacted["error"])
    if redacted.get("status"):
        return "status: " + _short(redacted["status"])
    return ""


def build_swarm_honcho_summary(job: SwarmJob) -> Dict[str, Any]:
    """Return a compact, redacted payload suitable for parent-owned memory."""

    lines = [
        f"Jeeves swarm job {job.job_id}: {job.status}",
        f"Request: {_short(job.original_request, 500)}",
    ]
    if job.routing_plan:
        lines.append(f"Route: {job.routing_plan.mode} — {_short(job.routing_plan.reason)}")
    if job.tasks:
        lines.append("Tasks:")
        for task in job.tasks:
            result = _safe_result_summary(task.result)
            suffix = f" — {result}" if result else ""
            blocker = " (blocked)" if task.permission_required or task.status in {"blocked", "awaiting_permission"} else ""
            lines.append(f"- {task.task_id}: {task.status}{blocker} — {_short(task.title)}{suffix}")
    if job.evals:
        passed = sum(1 for item in job.evals if item.passed)
        lines.append(f"Verification: {passed}/{len(job.evals)} checks passed")
    if job.permissions or (job.routing_plan and job.routing_plan.permission_requests):
        lines.append("Permissions: outstanding or requested actions were parent-gated.")

    metadata = redact_secrets(
        {
            "job_id": job.job_id,
            "status": job.status,
            "platform": job.platform,
            "session_id": job.session_id,
            "task_count": len(job.tasks),
        }
    )
    return {"content": "\n".join(lines), "metadata": metadata}


def _default_writer() -> Optional[Callable[[Dict[str, Any]], Any]]:
    """Best-effort optional plugin hook; intentionally returns None if absent."""

    try:
        from plugins.memory.honcho.swarm import write_swarm_summary  # type: ignore
    except Exception:
        return None
    return write_swarm_summary


def persist_swarm_honcho_summary(
    job: SwarmJob,
    *,
    enabled: bool = False,
    writer: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """Persist a compact swarm summary when explicitly enabled.

    The function is safe to call unconditionally: disabled, missing dependency,
    or writer failures are reported as structured no-op/error dictionaries and do
    not raise into gateway flow.
    """

    if not enabled:
        return {"persisted": False, "reason": "disabled"}
    resolved_writer = writer or _default_writer()
    if resolved_writer is None:
        return {"persisted": False, "reason": "honcho_unavailable"}
    payload = build_swarm_honcho_summary(job)
    try:
        resolved_writer(payload)
    except Exception as exc:
        return {"persisted": False, "reason": "writer_failed", "error": str(exc)}
    return {"persisted": True, "reason": "ok", "metadata": payload["metadata"]}


__all__ = ["build_swarm_honcho_summary", "persist_swarm_honcho_summary"]
