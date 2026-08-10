"""Read-only intake preflight for durable workflow-configuration tasks."""

from __future__ import annotations

from typing import Any, Optional

WORKFLOW_CONFIGURATION = "workflow_configuration"
IMPLEMENTATION_PROFILES = ("forge", "dev", "chip", "quill")
CONTROL_PROFILES = frozenset({"default", "halo"})


class WorkflowConfigurationIntakeError(ValueError):
    """Actionable rejection raised before a workflow task is written."""


def _profile_exists(name: str) -> bool:
    try:
        from hermes_cli.profiles import profile_exists
        return bool(profile_exists(name))
    except Exception:
        return False


def _implementation_assignee(requested: Optional[str]) -> str:
    value = (str(requested).strip().lower() if requested is not None else "")
    if value and value not in CONTROL_PROFILES:
        return value
    for candidate in IMPLEMENTATION_PROFILES:
        if _profile_exists(candidate):
            return candidate
    raise WorkflowConfigurationIntakeError(
        "workflow_configuration requires an implementation profile; Forge is unavailable "
        "and DEV is not installed. Create/restore the dev profile before dispatch "
        "(default/Halo cannot own mutable config work)."
    )


def preflight_workflow_configuration(*, task_type: Optional[str], assignee: Optional[str],
                                     metadata: Optional[dict[str, Any]] = None,
                                     requested_agent: Optional[str] = None
                                     ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Validate and normalize a workflow task without any Kanban writes."""
    normalized_type = str(task_type or "").strip().lower()
    if normalized_type != WORKFLOW_CONFIGURATION:
        return assignee, metadata
    raw = dict(metadata or {})
    coding_agent = raw.get("coding_agent")
    if coding_agent not in (None, "", "codex", "cursor"):
        raise WorkflowConfigurationIntakeError(
            f"invalid workflow_configuration metadata: coding_agent={coding_agent!r} is not "
            "allowed; use coding_agent=codex (or explicitly cursor), or declare route=dev_direct "
            "with use_coding_router=false and omit coding_agent"
        )
    if requested_agent is not None and requested_agent not in {"codex", "cursor"}:
        raise WorkflowConfigurationIntakeError(
            f"invalid coding_agent {requested_agent!r}; choose codex or cursor"
        )
    if raw.get("route") == "dev_direct" and raw.get("use_coding_router") is not False:
        raise WorkflowConfigurationIntakeError(
            "invalid workflow_configuration route: dev_direct requires use_coding_router=false"
        )
    if raw.get("use_coding_router") is False and coding_agent is not None:
        raise WorkflowConfigurationIntakeError(
            "invalid workflow_configuration metadata: direct execution must omit coding_agent "
            "(coding_agent=direct is never valid)"
        )
    selected = _implementation_assignee(assignee)
    if selected not in IMPLEMENTATION_PROFILES or not _profile_exists(selected):
        raise WorkflowConfigurationIntakeError(
            f"workflow_configuration assignee {selected!r} is not a validated implementation "
            "specialist; use Forge, DEV, Chip, or Quill"
        )
    if selected in CONTROL_PROFILES:
        raise WorkflowConfigurationIntakeError(
            f"workflow_configuration cannot be assigned to {selected}; assign Forge or DEV instead"
        )
    direct = raw.get("route") == "dev_direct"
    canonical = {
        **raw,
        "canonical": True,
        "task_type": WORKFLOW_CONFIGURATION,
        "lane": selected.upper(),
        "implementation_lane": True,
        "route": "dev_direct" if direct else "coding_cli_router",
        "use_coding_router": False if direct else True,
        "coding_agent": None if direct else (requested_agent or raw.get("coding_agent") or "codex"),
        "coding_agent_resolution": "explicit" if requested_agent else ("dev_direct" if direct else "default"),
    }
    if direct:
        canonical["execution_fallback"] = "dev_direct"
    return selected, canonical


def preflight_output(*, task_type: Optional[str], assignee: Optional[str],
                     metadata: Optional[dict[str, Any]] = None,
                     requested_agent: Optional[str] = None) -> dict[str, Any]:
    """Return a JSON-friendly routing decision for a dry-run caller."""
    try:
        resolved_assignee, resolved_metadata = preflight_workflow_configuration(
            task_type=task_type, assignee=assignee, metadata=metadata,
            requested_agent=requested_agent,
        )
    except WorkflowConfigurationIntakeError as exc:
        return {"accepted": False, "task_type": str(task_type or "").strip().lower(),
                "requested_assignee": assignee, "error": str(exc),
                "writes": 0, "duplicates_created": 0}
    return {"accepted": True, "task_type": str(task_type or "").strip().lower(),
            "requested_assignee": assignee, "resolved_assignee": resolved_assignee,
            "metadata": resolved_metadata, "writes": 0, "duplicates_created": 0}