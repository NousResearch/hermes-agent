"""Quality-lane gates for high-rigor Hermes work.

The gateway prompt tells the model what to do; this module provides the
structured template and checks used by prompts, recovery reports, and
diagnostics so the policy is observable instead of only advisory prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


HIGH_RIGOR_TRIGGERS = (
    "code",
    "commit",
    "test",
    "validation",
    "restart",
    "deploy",
    "production",
    "state.db",
    "sessions.json",
    "memory",
    "storage",
    "cloud",
    "signal room",
    "video production",
    "waha",
    "app",
    "hermes reliability",
    "repo",
    "worktree",
    "cleanup",
    "recovery",
)

QUALITY_LANE_REQUIRED_FIELDS = (
    "Task classification:",
    "Required lanes:",
    "Implementation lane result:",
    "Review lane result:",
    "Verification lane result:",
    "Safety lane result:",
    "Remaining risks:",
)

QUALITY_LANE_GOAL_REQUIREMENT = (
    "\n\nQuality lanes requirement: if this goal turn includes high-rigor work "
    "(code changes, validation, restart/deployment, production state, "
    "storage/cloud, Signal Room/video, WAHA, app-building, Hermes reliability, "
    "repo cleanup, or recovery), the final response must include a "
    "'## Quality lanes' section with task classification, required lanes, "
    "implementation, review, verification, safety, deployment/runtime when "
    "applicable, and remaining risks. Subagent unavailable/not invoked; "
    "checklist fallback used. Do not claim delegate_task actually ran unless "
    "it did."
    " Required fields include Task classification."
)


def classify_task(text: str | None) -> dict[str, Any]:
    folded = (text or "").lower()
    matched = [trigger for trigger in HIGH_RIGOR_TRIGGERS if trigger in folded]
    return {
        "classification": "high-rigor" if matched else "standard",
        "matched_triggers": matched,
        "deployment_applicable": any(
            trigger in folded for trigger in ("restart", "deploy", "runtime", "production")
        ),
    }


def detect_delegate_task_capability(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    delegate_tool = root / "tools" / "delegate_tool.py"
    toolsets = root / "toolsets.py"
    try:
        delegate_source = delegate_tool.read_text(encoding="utf-8", errors="replace")
        toolsets_source = toolsets.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"available": None, "reason": "delegate_task source unavailable"}
    available = "def delegate_task" in delegate_source and "delegate_task" in toolsets_source
    return {
        "available": available,
        "reason": "delegate_task source and toolset entry detected" if available else "delegate_task not fully wired",
    }


def validate_quality_lane_section(report: str | None) -> dict[str, Any]:
    text = report or ""
    missing = [field for field in QUALITY_LANE_REQUIRED_FIELDS if field not in text]
    return {
        "valid": "## Quality lanes" in text and not missing,
        "missing_fields": missing,
    }


def require_quality_lane_section(
    task_text: str | None,
    *,
    verification_summary: str = "Not yet run.",
    safety_summary: str = "Constraints must be reported explicitly.",
    remaining_risks: str = "Report any unverified paths or residual risk.",
    subagent_available: bool | None = None,
    subagent_invoked: bool = False,
) -> str:
    classification = classify_task(task_text)
    is_high = classification["classification"] == "high-rigor"
    required_lanes = ["implementation", "review", "verification", "safety"]
    if classification["deployment_applicable"]:
        required_lanes.append("deployment/runtime")

    real_subagent = bool(subagent_invoked)
    if real_subagent:
        review = "real subagent used: yes; independent review evidence must be summarized."
    else:
        if subagent_available is False:
            reason = "subagent unavailable"
        elif subagent_available is True:
            reason = "subagent not invoked"
        else:
            reason = "subagent availability unknown"
        review = (
            f"real subagent used: no; {reason}. "
            "Subagent unavailable/not invoked; checklist fallback used."
        )

    lines = [
        "## Quality lanes",
        f"Task classification: {classification['classification']}",
        f"Required lanes: {', '.join(required_lanes)}",
        "Implementation lane result: Summarize the implementation or inspection outcome.",
        f"Review lane result: {review}",
        f"Verification lane result: {verification_summary}",
        f"Safety lane result: {safety_summary}",
    ]
    if classification["deployment_applicable"]:
        lines.append(
            "Deployment/runtime lane result: restart required/performed/not performed; rollback available if applicable."
        )
    if not is_high:
        lines.append("Standard task note: quality lanes are optional unless high-rigor work appears.")
    lines.append(f"Remaining risks: {remaining_risks}")
    return "\n".join(lines)


def ensure_quality_lane_section(
    report: str,
    task_text: str | None,
    *,
    verification_summary: str = "Final report generated by Hermes.",
    safety_summary: str = "No forbidden operations should be claimed without evidence.",
    subagent_available: bool | None = None,
    subagent_invoked: bool = False,
) -> str:
    if validate_quality_lane_section(report).get("valid"):
        return report
    if classify_task(task_text).get("classification") != "high-rigor":
        return report
    section = require_quality_lane_section(
        task_text,
        verification_summary=verification_summary,
        safety_summary=safety_summary,
        subagent_available=subagent_available,
        subagent_invoked=subagent_invoked,
    )
    return f"{report.rstrip()}\n\n{section}" if report.strip() else section
