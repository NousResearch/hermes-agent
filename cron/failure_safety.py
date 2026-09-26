"""Closed-label boundary for cron failures exposed outside the local run log.

Arbitrary exception and script text is private diagnostic data.  It may contain
credentials, customer data, host paths, or stack traces, so public cron surfaces
must describe only a bounded failure class.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_SCRIPT_EXIT = re.compile(r"^Script exited with code\s+(-?\d+)\b", re.IGNORECASE)
_ALREADY_SAFE = re.compile(
    r"^(?:script_failed(?: \(exit -?\d+\))?|script_timeout|provider_[a-z_]+|"
    r"blocked_config|job_stalled|empty_response|run_interrupted|run_discarded|"
    r"schedule_failed|job_failed)(?: \(incident [A-Za-z0-9_-]+\))?$"
)
_INCIDENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def cron_failure_code(error: Any, *, no_agent: bool = False) -> str:
    """Return a closed failure label without copying any attacker-controlled text."""
    text = str(error or "").strip()
    if not text:
        return "job_failed"
    if _ALREADY_SAFE.fullmatch(text):
        return text.split(" (incident ", 1)[0]

    lower = text.lower()
    match = _SCRIPT_EXIT.match(text)
    if match:
        return f"script_failed (exit {match.group(1)})"
    if lower.startswith("script timed out"):
        return "script_timeout"
    if "[blocked_config" in lower:
        return "blocked_config"
    if re.search(r"idle for \d+s\s*\(limit \d+s\)", lower):
        return "job_stalled"
    if lower.startswith("agent completed but produced empty response"):
        return "empty_response"
    if lower.startswith("interrupted by gateway shutdown"):
        return "run_interrupted"
    if lower.startswith("fire claim") or "ownership lost" in lower:
        return "run_discarded"
    if lower.startswith("failed to compute next run"):
        return "schedule_failed"
    if no_agent or lower.startswith("script execution failed"):
        return "script_failed"

    try:
        from cron.scheduler_failure_copy import classify_cron_failure_reason

        reason = classify_cron_failure_reason(text)
    except Exception:
        reason = "unknown"
    return f"provider_{reason}" if reason and reason != "unknown" else "job_failed"


def public_cron_failure(
    error: Any, *, no_agent: bool = False, incident_id: Optional[str] = None,
) -> str:
    """Safe text for persisted records, APIs, CLI/tool output, and chat notices."""
    text = str(error or "").strip()
    if _ALREADY_SAFE.fullmatch(text) and incident_id is None:
        return text
    label = cron_failure_code(text, no_agent=no_agent)
    incident = str(incident_id or "").strip()
    if incident and _INCIDENT_ID.fullmatch(incident):
        return f"{label} (incident {incident})"
    return label


def sanitize_job_failure_fields(job: dict[str, Any]) -> dict[str, Any]:
    """Replace legacy arbitrary error strings on the read boundary.

    The normalized job may later be persisted by an unrelated edit, gradually
    removing unsafe historical strings from jobs.json as well.
    """
    no_agent = bool(job.get("no_agent"))
    if job.get("last_error"):
        job["last_error"] = public_cron_failure(job["last_error"], no_agent=no_agent)
    if job.get("last_delivery_error"):
        job["last_delivery_error"] = public_cron_failure(job["last_delivery_error"])
    fire_error = job.get("last_fire_error")
    if isinstance(fire_error, dict):
        sanitized = dict(fire_error)
        sanitized["detail"] = public_cron_failure(fire_error.get("detail"))
        job["last_fire_error"] = sanitized
    elif fire_error:
        job["last_fire_error"] = public_cron_failure(fire_error)
    return job


def sanitize_execution_failure_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Return an execution row whose public error field contains only a closed label."""
    sanitized = dict(record)
    if sanitized.get("error"):
        sanitized["error"] = public_cron_failure(sanitized["error"])
    return sanitized


def private_failure_output(output: Any, error: Any) -> str:
    """Build the owner-only run log body for a failed run."""
    chunks = []
    output_text = str(output or "").strip()
    error_text = str(error or "").strip()
    if output_text:
        chunks.append(output_text)
    if error_text and error_text not in output_text:
        chunks.append("## Failure diagnostics\n\n" + error_text)
    return "\n\n".join(chunks)
