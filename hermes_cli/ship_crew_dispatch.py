"""Pre-dispatch validation and quarantine for routed Ship's Crew tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.ship_crew_routes import RouteRegistry, RouteRequest, RoutePolicyError, select_route


@dataclass(frozen=True)
class DispatchValidation:
    valid: bool
    issues: tuple[str, ...] = ()


def _metadata(task: kb.Task) -> dict[str, Any]:
    value = task.routing_metadata
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {"_invalid_json": True}
        return parsed if isinstance(parsed, dict) else {"_invalid_json": True}
    return {}


def validate_task_for_dispatch(
    task: kb.Task,
    *,
    registry: Optional[RouteRegistry] = None,
    require_contracts: bool = False,
) -> DispatchValidation:
    """Validate immutable routing and governance metadata without spawning.

    Legacy profile-routed tasks are allowed unless ``require_contracts`` is
    requested. Once any route field is set, all fields needed to reproduce the
    route are mandatory and cannot silently fall back to a profile default.
    """
    issues: list[str] = []
    meta = _metadata(task)
    if meta.get("_invalid_json"):
        issues.append("routing_metadata_not_json_object")

    routed = any(
        getattr(task, name, None)
        for name in ("route_id", "executor", "executor_mode", "quota_domain", "complexity_tier")
    )
    governed = bool(meta.get("contract_version") or meta.get("governance_class"))
    if require_contracts and not governed:
        issues.append("contract_version_missing")
    if routed or governed:
        for name in ("complexity_tier", "route_id", "reasoning_effort", "executor", "executor_mode", "quota_domain"):
            if not getattr(task, name, None):
                issues.append(f"routing_{name}_missing")
        if governed and not meta.get("contract_version"):
            issues.append("contract_version_missing")
        if governed and meta.get("governance_class") not in {"lite", "standard", "constitutional"}:
            issues.append("governance_class_invalid")

    if registry is not None and task.route_id:
        request = RouteRequest(
            profile=task.assignee or "",
            complexity_tier=task.complexity_tier or "",
            governance_class=str(meta.get("governance_class", "standard")),
            write_scope=str(meta.get("write_scope", "none")),
            output_class=str(meta.get("output_class", "text")),
            required_capability=meta.get("required_capability"),
        )
        try:
            selected = select_route(request, registry)
        except RoutePolicyError as exc:
            issues.append(str(exc))
        else:
            if selected.route_id != task.route_id:
                issues.append("route_selection_mismatch")

    return DispatchValidation(not issues, tuple(issues))


def quarantine_task(
    conn,
    task_id: str,
    validation: DispatchValidation,
    *,
    actor: str = "dispatcher",
) -> bool:
    """Move a malformed task to a non-spawnable quarantine state."""
    if validation.valid:
        return False
    with kb.write_txn(conn):
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None or row["status"] in {"done", "archived"}:
            return False
        conn.execute(
            "UPDATE tasks SET status='blocked', block_kind='contract_rejected', claim_lock=NULL, claim_expires=NULL WHERE id=?",
            (task_id,),
        )
        kb._append_event(
            conn,
            task_id,
            "contract_rejected",
            {"actor": actor, "issues": list(validation.issues)},
        )
    return True


def should_validate_dispatch(task: kb.Task, *, require_contracts: bool = False) -> bool:
    """Return whether a dispatcher should apply the contract gate."""
    return require_contracts or any(
        getattr(task, name, None)
        for name in ("route_id", "executor", "executor_mode", "quota_domain", "complexity_tier")
    ) or bool(_metadata(task).get("contract_version"))
