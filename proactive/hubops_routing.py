from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
DEFAULT_HUB_OPS_DIR = Path(__file__).resolve().parents[2] / "docs" / "projects" / "hub-ops"


def route_clawops_objective(
    objective: str,
    *,
    project: str = "hub_ops",
    task_type: str = "ops",
    risk_level: str = "low",
    approved: bool = False,
    hub_ops_dir: str | Path | None = None,
) -> dict[str, Any]:
    clean_objective = " ".join((objective or "").split())
    if not clean_objective:
        return _blocked("objective is required", objective="", project=project, task_type=task_type, risk_level=risk_level)

    docs_dir = Path(hub_ops_dir) if hub_ops_dir else DEFAULT_HUB_OPS_DIR
    try:
        registry = _read_yaml(docs_dir / "agent-registry.yaml")
        rules = _read_yaml(docs_dir / "routing-rules.yaml")
    except (OSError, ValueError) as exc:
        return _blocked(str(exc), objective=clean_objective, project=project, task_type=task_type, risk_level=risk_level)

    worker_routes = rules.get("worker_routes")
    worker_profiles = registry.get("worker_profiles")
    if not isinstance(worker_routes, list) or not isinstance(worker_profiles, Mapping):
        return _blocked(
            "HubOps worker routing is not configured.",
            objective=clean_objective,
            project=project,
            task_type=task_type,
            risk_level=risk_level,
        )

    route = _match_worker_route(worker_routes, project=project, task_type=task_type, risk_level=risk_level)
    if route is None:
        return _blocked(
            "No HubOps worker route matched this task.",
            objective=clean_objective,
            project=project,
            task_type=task_type,
            risk_level=risk_level,
        )

    assign = route.get("assign") if isinstance(route, Mapping) else None
    worker_id = str((assign or {}).get("worker") or "").strip() if isinstance(assign, Mapping) else ""
    worker = worker_profiles.get(worker_id) if worker_id else None
    if not worker_id or not isinstance(worker, Mapping):
        return _blocked(
            f"HubOps worker profile is missing: {worker_id or '<empty>'}",
            objective=clean_objective,
            project=project,
            task_type=task_type,
            risk_level=risk_level,
        )

    risk = _normalize_risk(risk_level)
    risk_limit = _normalize_risk(str(worker.get("risk_level_limit") or "low"))
    approval_required = bool((assign or {}).get("approval_required", worker.get("approval_required", False)))
    if risk in {"high", "critical"} and not approved:
        return _blocked(
            f"Human approval is required before routing risk_level={risk} ClawOps work.",
            objective=clean_objective,
            project=project,
            task_type=task_type,
            risk_level=risk,
            worker_id=worker_id,
            worker=worker,
            approval_required=True,
        )
    if RISK_ORDER[risk] > RISK_ORDER[risk_limit]:
        return _blocked(
            f"Task risk_level={risk} exceeds worker risk_level_limit={risk_limit}.",
            objective=clean_objective,
            project=project,
            task_type=task_type,
            risk_level=risk,
            worker_id=worker_id,
            worker=worker,
            approval_required=True,
        )

    return {
        "status": "routed",
        "objective": clean_objective,
        "project": project,
        "task_type": task_type,
        "risk_level": risk,
        "assignment": _assignment(worker_id, worker, approval_required=approval_required),
        "approval_checklist": str((assign or {}).get("approval_checklist") or worker.get("approval_checklist") or ""),
        "output_schema": worker.get("output_schema") or {},
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"HubOps YAML must be an object: {path}")
    return raw


def _match_worker_route(
    routes: list[Any],
    *,
    project: str,
    task_type: str,
    risk_level: str,
) -> Mapping[str, Any] | None:
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        match = route.get("match")
        if isinstance(match, Mapping) and _matches(match, project=project, task_type=task_type, risk_level=risk_level):
            return route
    return None


def _matches(match: Mapping[str, Any], *, project: str, task_type: str, risk_level: str) -> bool:
    expected = {
        "project": project,
        "task_type": task_type,
        "risk_level": _normalize_risk(risk_level),
    }
    for key, value in match.items():
        if str(value) != expected.get(str(key), value):
            return False
    return True


def _assignment(
    worker_id: str,
    worker: Mapping[str, Any],
    *,
    approval_required: bool,
) -> dict[str, Any]:
    return {
        "assigned_worker": worker_id,
        "display_name": str(worker.get("display_name") or worker_id),
        "allowed_tools": list(worker.get("allowed_tools") or []),
        "risk_level_limit": _normalize_risk(str(worker.get("risk_level_limit") or "low")),
        "approval_required": approval_required,
        "approval_required_actions": list(worker.get("approval_required_actions") or []),
        "timeout_seconds": int(worker.get("timeout_seconds") or 900),
        "retry_policy": worker.get("retry_policy") or {"max_attempts": 1, "backoff_seconds": 0},
    }


def _blocked(
    reason: str,
    *,
    objective: str,
    project: str,
    task_type: str,
    risk_level: str,
    worker_id: str = "",
    worker: Mapping[str, Any] | None = None,
    approval_required: bool = True,
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "objective": objective,
        "project": project,
        "task_type": task_type,
        "risk_level": _normalize_risk(risk_level),
        "blocked_reason": reason,
        "assignment": _assignment(worker_id, worker or {}, approval_required=approval_required),
        "approval_checklist": str((worker or {}).get("approval_checklist") or ""),
        "output_schema": (worker or {}).get("output_schema") or {},
    }


def _normalize_risk(value: str) -> str:
    risk = (value or "low").strip().lower()
    return risk if risk in RISK_ORDER else "low"
