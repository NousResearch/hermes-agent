"""Hermes-native operator mission and triage surfaces."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gateway.status import read_runtime_status, validate_runtime_artifacts
from hermes_constants import get_hermes_home


MISSION_SCHEMA = "hermes.operator_mission_surface.v1"
TRIAGE_SCHEMA = "hermes.operator_triage_surface.v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evidence_refs() -> dict[str, str]:
    home = get_hermes_home()
    return {
        "runtime_status": str(home / "gateway_state.json"),
        "runtime_events": str(home / "gateway_runtime_events.jsonl"),
        "gateway_pid": str(home / "gateway.pid"),
    }


def build_operator_mission_surface() -> dict[str, Any]:
    runtime = read_runtime_status() or {}
    validation = validate_runtime_artifacts()
    gateway_state = runtime.get("gateway_state") or "unknown"
    runtime_valid = bool(validation.get("runtime_status", {}).get("valid", False))
    pid_valid = bool(validation.get("pid", {}).get("valid", False))

    recommended_actions: list[str] = []
    if not runtime_valid:
        recommended_actions.append("Inspect gateway runtime artifact validity")
    if not pid_valid:
        recommended_actions.append("Check gateway PID ownership and restart posture")
    if runtime.get("restart_requested"):
        recommended_actions.append("Review pending restart before resuming normal operations")
    if gateway_state in {"startup_failed", "stopped", "crashed", "error"}:
        recommended_actions.append("Inspect failing platforms and recent runtime evidence")

    headline = "Gateway healthy"
    if recommended_actions:
        headline = "Gateway needs operator attention"

    platforms = []
    for name, payload in sorted((runtime.get("platforms") or {}).items()):
        if not isinstance(payload, dict):
            continue
        platforms.append(
            {
                "name": name,
                "state": payload.get("state") or "unknown",
                "error_code": payload.get("error_code"),
                "error_message": payload.get("error_message"),
                "updated_at": payload.get("updated_at"),
            }
        )

    return {
        "schema": MISSION_SCHEMA,
        "generated_at": _utc_now_iso(),
        "headline": headline,
        "gateway": {
            "state": gateway_state,
            "exit_reason": runtime.get("exit_reason"),
            "restart_requested": bool(runtime.get("restart_requested")),
            "active_agents": int(runtime.get("active_agents") or 0),
            "updated_at": runtime.get("updated_at"),
        },
        "platforms": platforms,
        "validation": {
            "pid_valid": pid_valid,
            "runtime_valid": runtime_valid,
            "pid_errors": list(validation.get("pid", {}).get("errors") or []),
            "runtime_errors": list(validation.get("runtime_status", {}).get("errors") or []),
            "evidence_exists": bool(validation.get("evidence", {}).get("exists", False)),
            "evidence_line_count": int(validation.get("evidence", {}).get("line_count") or 0),
            "last_evidence_event": validation.get("evidence", {}).get("last_event"),
        },
        "evidence_refs": _evidence_refs(),
        "recommended_actions": recommended_actions,
    }


def build_operator_triage_surface() -> dict[str, Any]:
    mission = build_operator_mission_surface()
    issues: list[dict[str, Any]] = []

    for platform in mission.get("platforms", []):
        if platform.get("state") in {"fatal", "degraded", "error"}:
            severity = "critical" if platform.get("state") == "fatal" else "warning"
            issues.append(
                {
                    "kind": "platform_failure",
                    "severity": severity,
                    "summary": f"{platform['name']} is {platform['state']}",
                    "rationale": platform.get("error_message") or platform.get("error_code") or "platform reported unhealthy state",
                    "evidence_ref": mission["evidence_refs"]["runtime_status"],
                    "suggested_command": "hermes gateway operator-status --json",
                }
            )

    if not mission["validation"].get("runtime_valid", False):
        issues.append(
            {
                "kind": "runtime_artifact_invalid",
                "severity": "critical",
                "summary": "Gateway runtime artifact is invalid",
                "rationale": "; ".join(mission["validation"].get("runtime_errors", [])) or "runtime artifact failed validation",
                "evidence_ref": mission["evidence_refs"]["runtime_status"],
                "suggested_command": "python scripts/validate_gateway_runtime.py",
            }
        )

    if mission["gateway"].get("restart_requested"):
        issues.append(
            {
                "kind": "restart_requested",
                "severity": "warning",
                "summary": "Gateway restart has been requested",
                "rationale": "A restart request is pending; confirm drain and replacement posture.",
                "evidence_ref": mission["evidence_refs"]["runtime_events"],
                "suggested_command": "hermes gateway operator-status --json",
            }
        )

    severity_rank = {"info": 0, "warning": 1, "critical": 2}
    severity = "info"
    if issues:
        severity = max((issue["severity"] for issue in issues), key=lambda item: severity_rank[item])

    summary = "No operator action required" if not issues else "Operator action required"
    return {
        "schema": TRIAGE_SCHEMA,
        "generated_at": _utc_now_iso(),
        "severity": severity,
        "issue_count": len(issues),
        "summary": summary,
        "issues": issues,
        "mission_headline": mission.get("headline"),
    }
