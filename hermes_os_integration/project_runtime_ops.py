"""Extended Hermes OS project runtime operation contracts.

These helpers keep the next runtime phases inspectable and dry-run-first:
command surface audits, guarded live runtime execution, workspace restore
plans, dashboard modules, durable persistence projections, template packs, and
continuous workspace health checks.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import sqlite3
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .runtime_policies import RuntimePolicy, aggregate_cost_budget, approval_prompt_for_decision, evaluate_runtime_policy


HERMES_OS_COMMANDS = ["architect", "plan", "workspace", "projects", "switch", "start", "snapshot"]
PROJECT_RUNTIME_COLLECTIONS = [
    "project-definitions",
    "workspace-snapshots",
    "snapshot-restore-attempts",
    "runtime-service-status",
    "agent-messages",
    "agent-traces",
    "runtime-approvals",
    "runtime-costs",
    "infrastructure-registry",
    "vector-registry",
    "live-runtime-executions",
    "approval-requests",
    "automation-workflows",
    "cross-project-dependencies",
    "agent-fleet",
    "telemetry-events",
    "connectors",
    "evaluations",
    "project-memory-index",
    "release-checks",
]

LIVE_RUNTIME_STATES = ["queued", "running", "validating", "completed", "failed", "canceled", "rolled_back"]
APPROVAL_STATUSES = ["pending", "approved", "rejected", "expired"]
EVALUATION_STATUSES = ["pass", "fail", "warning", "waived"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CommandSurfaceAudit:
    command: str
    installed: bool
    supports_json: bool
    error_codes: List[str]
    smoke_test: str
    docs_ref: str = ""


@dataclass(frozen=True)
class LiveRuntimePlan:
    project_id: str
    action: str
    dry_run: bool
    allowed: bool
    approval: Dict[str, Any]
    audit: Dict[str, Any]
    command_allowed: bool
    timeout_seconds: int
    retry: Dict[str, Any]
    artifact_quarantine: Dict[str, Any]
    rollback: Dict[str, Any]


@dataclass(frozen=True)
class RestoreStep:
    kind: str
    target: str
    status: str
    requires_approval: bool = False
    command: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class RestorePlan:
    project_id: str
    dry_run: bool
    platform: str
    steps: List[RestoreStep]
    conflicts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TemplatePackManifest:
    pack_id: str
    name: str
    version: str = "1"
    min_hermes_os_version: str = "1"
    templates: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    source_path: str = ""


@dataclass(frozen=True)
class LiveRuntimeExecution:
    execution_id: str
    project_id: str
    command: str
    state: str = "queued"
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    stdout_ref: str = ""
    stderr_ref: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_ms: int = 0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    rollback_ref: str = ""


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    project_id: str
    requester: str
    scope: str
    risk: str
    action: str
    status: str = "pending"
    reviewer: str = ""
    reason: str = ""
    expires_at: str = ""
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class AutomationWorkflow:
    workflow_id: str
    project_id: str
    steps: List[Dict[str, Any]]
    dry_run: bool = True


@dataclass(frozen=True)
class CrossProjectDependency:
    source_project: str
    target_project: str
    reason: str
    status: str = "open"


@dataclass(frozen=True)
class AgentFleetMember:
    agent_id: str
    capabilities: List[str]
    model: str = ""
    tools: List[str] = field(default_factory=list)
    available: bool = True
    quarantined: bool = False
    cost_score: float = 1.0
    latency_score: float = 1.0
    success_rate: float = 1.0


@dataclass(frozen=True)
class TelemetryEvent:
    event_id: str
    project_id: str
    phase: str
    severity: str
    source: str
    correlation_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)


@dataclass(frozen=True)
class ConnectorManifest:
    connector_id: str
    name: str
    permissions: List[str]
    resources: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    risk_profile: str = "medium"
    min_hermes_os_version: str = "1"


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_id: str
    project_id: str
    target_ref: str
    status: str
    evidence: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    waiver: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryIndexRecord:
    record_id: str
    project_id: str
    source_path: str
    summary: str
    topic: str = ""
    confidence: float = 1.0
    timestamp: str = field(default_factory=_now)


def command_completion_metadata() -> Dict[str, Any]:
    return {
        "commands": {
            "architect": ["review", "--json", "--persist", "--db", "--generate-docs", "--generate-tasks"],
            "plan": ["--json", "--write", "--template", "--generate-tasks", "--persist", "--db"],
            "workspace": ["--projects-root", "--json"],
            "projects": ["--workspace-root"],
            "switch": ["<project>", "--workspace-root", "--live"],
            "start": ["<project>", "--workspace-root", "--live"],
            "snapshot": ["save", "restore", "<project>", "--workspace-root", "--live"],
        }
    }


def audit_command_surface(commands: Optional[Sequence[str]] = None) -> List[CommandSurfaceAudit]:
    selected = list(commands or HERMES_OS_COMMANDS)
    json_commands = {"architect", "plan", "workspace", "projects", "switch", "start", "snapshot"}
    return [
        CommandSurfaceAudit(
            command=command,
            installed=True,
            supports_json=command in json_commands,
            error_codes=["missing_project", "invalid_registry", "unsafe_action", "persistence_failure"],
            smoke_test=f"hermes {command} --help" if command not in {"switch", "start"} else f"hermes {command} <project> --help",
            docs_ref="docs/hermes-os-system-explainer.md",
        )
        for command in selected
    ]


def normalize_command_envelope(command: str, payload: Dict[str, Any], *, ok: bool = True, error_code: str = "") -> Dict[str, Any]:
    return {
        "ok": ok,
        "command": command,
        "error": {"code": error_code, "message": payload.get("message", "")} if not ok else None,
        "data": payload if ok else {},
    }


def module_cli_redirects() -> Dict[str, str]:
    return {
        "python -m hermes_os_integration.architect_cli": "hermes architect",
        "python -m hermes_os_integration.plan_cli": "hermes plan",
        "python -m hermes_os_integration.workspace_control": "hermes workspace",
        "python -m hermes_os_integration.project_runtime": "hermes projects|switch|start|snapshot",
    }


def validate_command_allowlist(command: str, allowlist: Sequence[str]) -> bool:
    if not command.strip():
        return False
    executable = shlex.split(command)[0]
    return executable in set(allowlist)


def build_live_runtime_plan(
    *,
    project_id: str,
    action: str,
    command: str,
    allowlist: Sequence[str],
    approved: bool = False,
    estimated_cost_usd: float = 0.0,
    retry_count: int = 0,
    timeout_seconds: int = 300,
    dry_run: bool = True,
    policy: Optional[RuntimePolicy] = None,
) -> LiveRuntimePlan:
    command_allowed = validate_command_allowlist(command, allowlist)
    decision = evaluate_runtime_policy(
        action=action,
        estimated_cost_usd=estimated_cost_usd,
        retry_count=retry_count,
        approved=approved,
        policy=policy,
    )
    allowed = bool(command_allowed and decision.allowed and (dry_run or approved))
    reasons = list(decision.reasons)
    if not command_allowed:
        reasons.append("command is not allowlisted")
    audit = {
        **decision.audit,
        "project_id": project_id,
        "command": command,
        "command_allowed": command_allowed,
        "allowed": allowed,
        "dry_run": dry_run,
        "reasons": reasons,
    }
    return LiveRuntimePlan(
        project_id=project_id,
        action=action,
        dry_run=dry_run,
        allowed=allowed,
        approval=approval_prompt_for_decision(decision),
        audit=audit,
        command_allowed=command_allowed,
        timeout_seconds=timeout_seconds,
        retry={
            "retry_count": retry_count,
            "retry_allowed": decision.retry_allowed,
            "exhausted": not decision.retry_allowed,
        },
        artifact_quarantine={
            "enabled": True,
            "path": f".hermes/quarantine/{project_id}",
            "ingest_after_validation": True,
        },
        rollback={
            "required_on_failure": not dry_run,
            "report_ref": f"hermes-os://rollback/{project_id}/{_now()}",
        },
    )


def detect_restore_platform() -> str:
    system = platform.system().lower() or "unknown"
    if system == "darwin":
        return "macos"
    if system.startswith("win"):
        return "windows"
    if system == "linux":
        return "linux"
    return system


def build_restore_plan(snapshot: Dict[str, Any], *, dry_run: bool = True, dirty_worktree: bool = False, running_services: Optional[Iterable[str]] = None) -> RestorePlan:
    project_id = str(snapshot.get("project_id") or "unknown")
    running = set(running_services or [])
    steps: List[RestoreStep] = []
    conflicts: List[Dict[str, Any]] = []
    for file_name in snapshot.get("open_files", []):
        steps.append(RestoreStep(kind="editor", target=str(file_name), status="planned", command=["code", "-g", str(file_name)]))
    if not snapshot.get("open_files"):
        steps.append(RestoreStep(kind="editor", target="default", status="skipped", reason="no open files captured"))
    for url in snapshot.get("browser_urls", []):
        steps.append(RestoreStep(kind="browser", target=str(url), status="planned", command=["open", str(url)]))
    for terminal in snapshot.get("active_terminals", []):
        steps.append(RestoreStep(kind="terminal", target=str(terminal), status="planned", command=["sh", "-lc", str(terminal)]))
    for service in snapshot.get("running_services", []):
        status = "skipped" if service in running else "planned"
        if service in running:
            conflicts.append({"type": "running-service", "service": service})
        steps.append(RestoreStep(kind="service", target=str(service), status=status, requires_approval=True))
    branch = str(snapshot.get("current_branch") or "")
    if branch:
        if dirty_worktree:
            conflicts.append({"type": "dirty-worktree", "branch": branch})
        steps.append(RestoreStep(kind="git-branch", target=branch, status="blocked" if dirty_worktree else "planned", requires_approval=True, command=["git", "checkout", branch]))
    for task in snapshot.get("open_tasks", []):
        steps.append(RestoreStep(kind="active-task", target=str(task), status="planned"))
    return RestorePlan(project_id=project_id, dry_run=dry_run, platform=detect_restore_platform(), steps=steps, conflicts=conflicts)


def partial_restore_result(plan: RestorePlan) -> Dict[str, Any]:
    statuses: Dict[str, int] = {}
    for step in plan.steps:
        statuses[step.status] = statuses.get(step.status, 0) + 1
    return {
        "project_id": plan.project_id,
        "dry_run": plan.dry_run,
        "statuses": statuses,
        "conflicts": plan.conflicts,
        "steps": [asdict(step) for step in plan.steps],
    }


def runtime_dashboard_modules(project_id: str, status: Dict[str, Any], *, snapshots: Sequence[Dict[str, Any]] = (), traces: Sequence[Dict[str, Any]] = (), costs: Sequence[Dict[str, Any]] = (), approvals: Sequence[Dict[str, Any]] = (), template_packs: Sequence[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    service_statuses = status.get("runtime", {}).get("services", []) if isinstance(status.get("runtime"), dict) else []
    return [
        {"panel_id": "project-runtime-services", "title": "Runtime Services", "data": {"project_id": project_id, "services": service_statuses}},
        {"panel_id": "workspace-snapshots", "title": "Workspace Snapshots", "data": {"project_id": project_id, "snapshots": list(snapshots), "count": len(snapshots)}},
        {"panel_id": "snapshot-restore-preview", "title": "Snapshot Restore Preview", "data": {"project_id": project_id, "latest": snapshots[-1] if snapshots else None}},
        {"panel_id": "agent-trace-timeline", "title": "Agent Trace Timeline", "data": {"project_id": project_id, "timeline": list(traces), "count": len(traces)}},
        {"panel_id": "agent-message-detail", "title": "Agent Messages", "data": {"project_id": project_id, "messages": list(traces)}},
        {"panel_id": "runtime-cost-budget", "title": "Runtime Cost And Budget", "data": aggregate_cost_budget(list(costs), project_id=project_id)},
        {"panel_id": "runtime-approval-queue", "title": "Runtime Approval Queue", "data": {"project_id": project_id, "approvals": list(approvals), "pending_count": len([item for item in approvals if item.get("status") == "pending"])}},
        {"panel_id": "infrastructure-registry", "title": "Infrastructure", "data": {"project_id": project_id, "infrastructure": status.get("infrastructure", {})}},
        {"panel_id": "vector-registry", "title": "Vector Databases", "data": {"project_id": project_id, "vector_databases": status.get("vector_databases", {})}},
        {"panel_id": "runtime-dashboard-state", "title": "Dashboard State", "data": {"empty": False, "loading": False, "error": ""}},
        {"panel_id": "template-packs", "title": "Template Packs", "data": {"project_id": project_id, "packs": list(template_packs), "count": len(template_packs)}},
        {"panel_id": "activity-feed", "title": "Activity Feed", "data": {"project_id": project_id, "items": []}},
    ]


def ensure_project_runtime_schema(repository: Any) -> Dict[str, Any]:
    db_path = getattr(repository, "db_path", "")
    if not db_path:
        return {"status": "skipped", "reason": "repository is not sqlite-backed"}
    with sqlite3.connect(db_path) as conn:
        for collection in PROJECT_RUNTIME_COLLECTIONS:
            table = "hermes_os_" + collection.replace("-", "_")
            conn.execute(
                f"""
                create table if not exists {table} (
                    project_id text not null,
                    record_id text not null,
                    stored_at text not null,
                    payload text not null,
                    primary key (project_id, record_id)
                )
                """
            )
            conn.execute(f"create index if not exists idx_{table}_project on {table}(project_id, stored_at)")
    return {"status": "ready", "collections": PROJECT_RUNTIME_COLLECTIONS, "db_path": db_path}


def persist_project_runtime_record(repository: Any, collection: str, project_id: str, record_id: str, record: Dict[str, Any]) -> str:
    ensure_project_runtime_schema(repository)
    payload = {**record, "project_id": project_id, "id": record_id, "stored_at": record.get("stored_at", _now())}
    if hasattr(repository, "save"):
        repository.save(collection, f"{project_id}:{record_id}", payload)
    return str(getattr(repository, "db_path", ""))


def project_runtime_integrity_check(repository: Any) -> Dict[str, Any]:
    base = repository.integrity_check() if hasattr(repository, "integrity_check") else {"ok": True, "issues": [], "checked": 0}
    records = []
    for collection in PROJECT_RUNTIME_COLLECTIONS:
        if hasattr(repository, "list"):
            records.extend((collection, item) for item in repository.list(collection))
    issues = list(base.get("issues", []))
    known_snapshots = {item.get("id") for collection, item in records if collection == "workspace-snapshots"}
    for collection, item in records:
        if collection in {"snapshot-restore-attempts"} and item.get("snapshot_id") and item.get("snapshot_id") not in known_snapshots:
            issues.append({"type": "orphaned-snapshot-reference", "collection": collection, "record_id": item.get("id")})
        if collection in {"agent-traces"} and not item.get("correlation_id"):
            issues.append({"type": "missing-correlation-id", "collection": collection, "record_id": item.get("id")})
        if collection in {"runtime-approvals"} and item.get("status") not in {"pending", "approved", "rejected", "expired"}:
            issues.append({"type": "invalid-approval-status", "collection": collection, "record_id": item.get("id")})
    return {"ok": not issues, "issues": issues, "checked": base.get("checked", 0) + len(records)}


def load_template_pack_manifest(path: str) -> TemplatePackManifest:
    source = Path(path)
    manifest_path = source / "template-pack.json" if source.is_dir() else source
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return TemplatePackManifest(
        pack_id=str(payload.get("pack_id") or payload.get("id") or manifest_path.parent.name),
        name=str(payload.get("name") or payload.get("pack_id") or manifest_path.parent.name),
        version=str(payload.get("version") or "1"),
        min_hermes_os_version=str(payload.get("min_hermes_os_version") or "1"),
        templates=[str(item) for item in payload.get("templates", [])],
        dependencies=[str(item) for item in payload.get("dependencies", [])],
        source_path=str(manifest_path),
    )


def discover_template_packs(paths: Sequence[str]) -> Dict[str, Any]:
    packs = []
    diagnostics = []
    for root in paths:
        path = Path(root).expanduser()
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else sorted(item for item in path.iterdir() if item.is_dir() or item.name == "template-pack.json")
        for candidate in candidates:
            manifest = candidate / "template-pack.json" if candidate.is_dir() else candidate
            if not manifest.exists():
                continue
            try:
                packs.append(load_template_pack_manifest(str(candidate)))
            except Exception as exc:
                diagnostics.append({"path": str(candidate), "error": str(exc)})
    return {"packs": [asdict(pack) for pack in packs], "diagnostics": diagnostics}


def validate_template_pack(pack: TemplatePackManifest, *, installed_pack_ids: Sequence[str] = (), hermes_os_version: str = "1") -> Dict[str, Any]:
    diagnostics = []
    try:
        if int(pack.min_hermes_os_version.split(".", 1)[0]) > int(hermes_os_version.split(".", 1)[0]):
            diagnostics.append({"type": "incompatible-version", "required": pack.min_hermes_os_version})
    except ValueError:
        diagnostics.append({"type": "invalid-version", "required": pack.min_hermes_os_version})
    for dependency in pack.dependencies:
        if dependency not in installed_pack_ids:
            diagnostics.append({"type": "missing-dependency", "dependency": dependency})
    if not pack.templates:
        diagnostics.append({"type": "missing-templates"})
    return {"valid": not diagnostics, "diagnostics": diagnostics, "pack": asdict(pack)}


def template_pack_install_plan(pack: TemplatePackManifest, *, live: bool = False) -> Dict[str, Any]:
    return {
        "pack": asdict(pack),
        "dry_run": not live,
        "requires_approval": live,
        "actions": [
            {"type": "copy-template", "source": template, "status": "planned"}
            for template in pack.templates
        ],
    }


def template_pack_update_diff(current: TemplatePackManifest, candidate: TemplatePackManifest) -> Dict[str, Any]:
    return {
        "pack_id": current.pack_id,
        "from_version": current.version,
        "to_version": candidate.version,
        "changed_templates": sorted(set(current.templates).symmetric_difference(candidate.templates)),
        "dependency_changes": sorted(set(current.dependencies).symmetric_difference(candidate.dependencies)),
    }


def template_pack_uninstall_safety(pack: TemplatePackManifest, *, in_use_templates: Sequence[str] = ()) -> Dict[str, Any]:
    in_use = sorted(set(pack.templates).intersection(in_use_templates))
    return {"allowed": not in_use, "blocked_by": in_use, "requires_approval": bool(in_use)}


def continuous_workspace_health(
    projects: Sequence[Dict[str, Any]],
    *,
    runtime_records: Sequence[Dict[str, Any]] = (),
    approvals: Sequence[Dict[str, Any]] = (),
    traces: Sequence[Dict[str, Any]] = (),
    cost_records: Sequence[Dict[str, Any]] = (),
    dry_run: bool = True,
) -> Dict[str, Any]:
    reports = []
    for project in projects:
        project_id = str(project.get("project_id") or project.get("name") or "unknown")
        blockers = []
        if project.get("architecture_score", 100) < 80:
            blockers.append("architecture-drift")
        if project.get("stale_tasks", 0):
            blockers.append("stale-tasks")
        if project.get("stale_snapshot"):
            blockers.append("stale-snapshot")
        service_drift = [record for record in runtime_records if record.get("project_id") == project_id and record.get("status") in {"failed", "missing"}]
        aged_approvals = [item for item in approvals if item.get("project_id") == project_id and item.get("status") == "pending"]
        failed_traces = [item for item in traces if item.get("project_id") == project_id and item.get("status") == "failed"]
        budget = aggregate_cost_budget(list(cost_records), project_id=project_id)
        if service_drift:
            blockers.append("runtime-service-drift")
        if aged_approvals:
            blockers.append("approval-aging")
        if float(budget.get("actual_cost_usd", 0)) > float(project.get("cost_budget_usd", 999999)):
            blockers.append("cost-budget-drift")
        score = max(0, 100 - (len(set(blockers)) * 10) - len(failed_traces))
        reports.append({
            "project_id": project_id,
            "score": score,
            "blockers": sorted(set(blockers)),
            "service_drift_count": len(service_drift),
            "pending_approval_count": len(aged_approvals),
            "agent_failure_count": len(failed_traces),
            "cost_budget": budget,
            "requires_approval": bool(blockers),
        })
    return {
        "dry_run": dry_run,
        "generated_at": _now(),
        "project_count": len(reports),
        "reports": reports,
        "activity_feed": [
            {"type": "workspace-health", "project_id": report["project_id"], "score": report["score"]}
            for report in reports
        ],
    }


def transition_live_runtime(execution: LiveRuntimeExecution, new_state: str, **updates: Any) -> LiveRuntimeExecution:
    if new_state not in LIVE_RUNTIME_STATES:
        raise ValueError("unknown live runtime state: " + new_state)
    allowed = {
        "queued": {"running", "canceled"},
        "running": {"validating", "failed", "canceled"},
        "validating": {"completed", "failed", "rolled_back"},
        "completed": set(),
        "failed": {"rolled_back"},
        "canceled": set(),
        "rolled_back": set(),
    }
    if new_state != execution.state and new_state not in allowed.get(execution.state, set()):
        raise ValueError(f"invalid live runtime transition: {execution.state} -> {new_state}")
    data = asdict(execution)
    data.update(updates)
    data["state"] = new_state
    if new_state == "running" and not data.get("started_at"):
        data["started_at"] = _now()
    if new_state in {"completed", "failed", "canceled", "rolled_back"} and not data.get("ended_at"):
        data["ended_at"] = _now()
    return LiveRuntimeExecution(**data)


def live_runtime_artifact_manifest(project_id: str, artifacts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = []
    for index, artifact in enumerate(artifacts):
        ref = str(artifact.get("ref") or artifact.get("path") or f"artifact-{index}")
        checksum = str(artifact.get("checksum") or f"unchecked:{index}")
        normalized.append({"ref": ref, "checksum": checksum, "validation_status": artifact.get("validation_status", "pending")})
    return {"project_id": project_id, "artifacts": normalized, "valid": all(item["validation_status"] == "passed" for item in normalized)}


def live_runtime_history(executions: Sequence[LiveRuntimeExecution], *, project_id: str = "") -> List[Dict[str, Any]]:
    rows = [asdict(item) for item in executions if not project_id or item.project_id == project_id]
    return sorted(rows, key=lambda item: item.get("started_at") or item.get("execution_id") or "")


def runtime_log_tail_contract(execution_id: str, *, stream: str = "stdout", lines: int = 100) -> Dict[str, Any]:
    return {"execution_id": execution_id, "stream": stream, "lines": int(lines), "path": f".hermes/runtime/logs/{execution_id}.{stream}.log"}


def redact_environment(env: Dict[str, str], allowlist: Sequence[str]) -> Dict[str, str]:
    allowed = set(allowlist)
    return {key: (value if key in allowed else "<redacted>") for key, value in env.items()}


def create_approval_request(
    approval_id: str,
    project_id: str,
    *,
    requester: str,
    scope: str,
    risk: str,
    action: str,
    expires_at: str = "",
) -> ApprovalRequest:
    return ApprovalRequest(approval_id=approval_id, project_id=project_id, requester=requester, scope=scope, risk=risk, action=action, expires_at=expires_at)


def decide_approval(request: ApprovalRequest, *, reviewer: str, approved: bool, reason: str) -> ApprovalRequest:
    if not reason.strip():
        raise ValueError("approval decision requires a reason")
    data = asdict(request)
    data.update({"reviewer": reviewer, "reason": reason, "status": "approved" if approved else "rejected"})
    return ApprovalRequest(**data)


def expire_approvals(requests: Sequence[ApprovalRequest], *, now: str) -> List[ApprovalRequest]:
    expired = []
    for request in requests:
        if request.status == "pending" and request.expires_at and request.expires_at <= now:
            data = asdict(request)
            data["status"] = "expired"
            expired.append(ApprovalRequest(**data))
        else:
            expired.append(request)
    return expired


def approval_risk_score(*, action: str, risk: str = "medium", estimated_cost_usd: float = 0.0, resources: Sequence[str] = ()) -> int:
    score = {"low": 10, "medium": 30, "high": 60}.get(risk, 30)
    if action in {"write", "deploy", "purchase", "delete"}:
        score += 20
    if estimated_cost_usd > 1:
        score += 10
    score += min(10, len(resources) * 2)
    return min(100, score)


def approval_audit_export(requests: Sequence[ApprovalRequest]) -> Dict[str, Any]:
    return {"exported_at": _now(), "count": len(requests), "approvals": [asdict(item) for item in requests]}


def build_automation_workflow(project_id: str, steps: Sequence[str], *, dry_run: bool = True) -> AutomationWorkflow:
    plan = [{"id": f"step-{index + 1:03d}", "type": step, "status": "planned", "depends_on": [] if index == 0 else [f"step-{index:03d}"]} for index, step in enumerate(steps)]
    return AutomationWorkflow(workflow_id=f"{project_id}:workflow:{len(plan)}", project_id=project_id, steps=plan, dry_run=dry_run)


def automation_preflight(*, dirty_worktree: bool = False, unavailable_tools: Sequence[str] = (), blocked_approvals: Sequence[str] = (), missing_config: Sequence[str] = ()) -> Dict[str, Any]:
    issues = []
    if dirty_worktree:
        issues.append({"type": "dirty-worktree"})
    issues.extend({"type": "unavailable-tool", "tool": tool} for tool in unavailable_tools)
    issues.extend({"type": "blocked-approval", "approval_id": approval} for approval in blocked_approvals)
    issues.extend({"type": "missing-config", "key": key} for key in missing_config)
    return {"ok": not issues, "issues": issues}


def automation_dry_run_diff(current: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    changes = []
    for key in sorted(set(current) | set(target)):
        if current.get(key) != target.get(key):
            changes.append({"field": key, "from": current.get(key), "to": target.get(key)})
    return {"change_count": len(changes), "changes": changes}


def automation_failure_report(workflow: AutomationWorkflow, failed_step: str, reason: str) -> Dict[str, Any]:
    return {"workflow_id": workflow.workflow_id, "project_id": workflow.project_id, "failed_step": failed_step, "reason": reason, "next_actions": ["inspect logs", "resolve preflight issues", "rerun dry-run"]}


def resolve_cross_project_dependencies(dependencies: Sequence[CrossProjectDependency]) -> Dict[str, Any]:
    blocked = [asdict(dep) for dep in dependencies if dep.status != "completed"]
    ordered = []
    for dep in dependencies:
        if dep.source_project not in ordered:
            ordered.append(dep.source_project)
        if dep.target_project not in ordered:
            ordered.append(dep.target_project)
    return {"ordered_projects": ordered, "blocked": blocked, "blocked_count": len(blocked)}


def balance_project_queue(projects: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(project: Dict[str, Any]) -> float:
        return float(project.get("priority", 0)) + float(project.get("readiness", 0)) - float(project.get("risk", 0))

    return sorted(({**project, "queue_score": score(project)} for project in projects), key=lambda item: item["queue_score"], reverse=True)


def detect_shared_resource_conflicts(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    conflicts = []
    for action in actions:
        resource = str(action.get("resource") or "")
        if not resource:
            continue
        if resource in seen and seen[resource].get("project_id") != action.get("project_id"):
            conflicts.append({"resource": resource, "projects": sorted({str(seen[resource].get("project_id")), str(action.get("project_id"))})})
        seen[resource] = action
    return conflicts


def score_agent(member: AgentFleetMember, *, required_capabilities: Sequence[str] = ()) -> Dict[str, Any]:
    required = set(required_capabilities)
    matched = required.intersection(member.capabilities)
    capability_score = 1.0 if not required else len(matched) / len(required)
    health_score = 0.0 if member.quarantined or not member.available else round((member.success_rate + member.cost_score + member.latency_score + capability_score) / 4, 3)
    return {"agent_id": member.agent_id, "capability_score": capability_score, "health_score": health_score, "available": member.available, "quarantined": member.quarantined}


def route_agent(members: Sequence[AgentFleetMember], required_capabilities: Sequence[str]) -> Dict[str, Any]:
    scored = [score_agent(member, required_capabilities=required_capabilities) for member in members]
    scored.sort(key=lambda item: item["health_score"], reverse=True)
    selected = scored[0] if scored else None
    return {"selected": selected, "candidates": scored, "fallback": bool(selected and selected["capability_score"] < 1.0)}


def quarantine_agent(member: AgentFleetMember, *, failures: int, policy_limit: int = 3) -> AgentFleetMember:
    if failures < policy_limit:
        return member
    data = asdict(member)
    data["quarantined"] = True
    data["available"] = False
    return AgentFleetMember(**data)


def redact_telemetry_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sensitive = {"prompt", "api_key", "token", "password", "secret", "env"}
    return {key: ("<redacted>" if key.lower() in sensitive else value) for key, value in payload.items()}


def create_telemetry_event(event_id: str, project_id: str, phase: str, severity: str, source: str, correlation_id: str, payload: Dict[str, Any]) -> TelemetryEvent:
    return TelemetryEvent(event_id=event_id, project_id=project_id, phase=phase, severity=severity, source=source, correlation_id=correlation_id, payload=redact_telemetry_payload(payload))


def telemetry_rollup(events: Sequence[TelemetryEvent]) -> Dict[str, Any]:
    by_project: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    missing_correlation = []
    for event in events:
        by_project[event.project_id] = by_project.get(event.project_id, 0) + 1
        by_severity[event.severity] = by_severity.get(event.severity, 0) + 1
        if not event.correlation_id:
            missing_correlation.append(event.event_id)
    return {"event_count": len(events), "by_project": by_project, "by_severity": by_severity, "missing_correlation": missing_correlation}


def diagnostics_bundle(project_id: str, *, events: Sequence[TelemetryEvent] = (), executions: Sequence[LiveRuntimeExecution] = (), approvals: Sequence[ApprovalRequest] = ()) -> Dict[str, Any]:
    return {"project_id": project_id, "generated_at": _now(), "events": [asdict(item) for item in events], "executions": [asdict(item) for item in executions], "approvals": [asdict(item) for item in approvals]}


def validate_connector_permission(manifest: ConnectorManifest, action: str) -> Dict[str, Any]:
    allowed = action in manifest.permissions
    high_risk = action in {"write", "deploy", "purchase", "delete"} or manifest.risk_profile == "high"
    return {"allowed": allowed, "requires_approval": allowed and high_risk, "connector_id": manifest.connector_id, "action": action}


def connector_dry_run(manifest: ConnectorManifest, action: str, target: str) -> Dict[str, Any]:
    permission = validate_connector_permission(manifest, action)
    return {"dry_run": True, "permission": permission, "audit": {"connector_id": manifest.connector_id, "target": target, "action": action, "timestamp": _now()}}


def normalize_connector_output(manifest: ConnectorManifest, output: Dict[str, Any]) -> Dict[str, Any]:
    return {"connector_id": manifest.connector_id, "source": "connector", "payload": output, "normalized_at": _now()}


def connector_secret_policy(secret_refs: Sequence[str]) -> Dict[str, Any]:
    return {"raw_secret_storage_allowed": False, "secret_refs": list(secret_refs), "status": "reference-only"}


def run_evaluations(project_id: str, target_ref: str, checks: Sequence[Dict[str, Any]]) -> List[EvaluationResult]:
    results = []
    for index, check in enumerate(checks):
        required = list(check.get("required_fields", []))
        payload = check.get("payload", {})
        failures = [field for field in required if field not in payload or payload.get(field) in ("", None, [])]
        status = "fail" if failures else str(check.get("status", "pass"))
        results.append(EvaluationResult(evaluation_id=f"eval-{index + 1:03d}", project_id=project_id, target_ref=target_ref, status=status, evidence=list(check.get("evidence", [])), failures=failures))
    return results


def apply_quality_gate(evaluations: Sequence[EvaluationResult], *, allow_warnings: bool = True) -> Dict[str, Any]:
    blocking = [item for item in evaluations if item.status == "fail" and not item.waiver]
    warnings = [item for item in evaluations if item.status == "warning"]
    allowed = not blocking and (allow_warnings or not warnings)
    return {"allowed": allowed, "blocked_count": len(blocking), "warning_count": len(warnings), "evaluations": [asdict(item) for item in evaluations]}


def waive_evaluation(evaluation: EvaluationResult, *, reviewer: str, reason: str, expires_at: str) -> EvaluationResult:
    if not reason.strip():
        raise ValueError("waiver requires a reason")
    data = asdict(evaluation)
    data["status"] = "waived"
    data["waiver"] = {"reviewer": reviewer, "reason": reason, "expires_at": expires_at, "timestamp": _now()}
    return EvaluationResult(**data)


def cost_aware_evaluation_plan(checks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    deterministic = [check for check in checks if check.get("type", "deterministic") == "deterministic"]
    model = [check for check in checks if check.get("type") == "model"]
    return {"stages": [{"name": "deterministic", "checks": deterministic}, {"name": "model", "checks": model}], "model_checks_deferred": bool(deterministic)}


def index_project_memory(records: Sequence[MemoryIndexRecord]) -> Dict[str, Any]:
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_topic.setdefault(record.topic or "general", []).append(asdict(record))
    return {"record_count": len(records), "by_topic": by_topic}


def retrieve_project_decisions(records: Sequence[MemoryIndexRecord], *, topic: str = "", min_confidence: float = 0.0) -> List[Dict[str, Any]]:
    matches = [record for record in records if (not topic or record.topic == topic) and record.confidence >= min_confidence]
    return [asdict(record) for record in sorted(matches, key=lambda item: item.timestamp, reverse=True)]


def summarize_project_memory(records: Sequence[MemoryIndexRecord]) -> Dict[str, Any]:
    return {"summary": [record.summary for record in records], "traceability": [{"summary": record.summary, "source_path": record.source_path, "record_id": record.record_id} for record in records]}


def detect_memory_drift(records: Sequence[MemoryIndexRecord]) -> Dict[str, Any]:
    by_topic: Dict[str, set[str]] = {}
    for record in records:
        by_topic.setdefault(record.topic, set()).add(record.summary)
    conflicts = [{"topic": topic, "summaries": sorted(values)} for topic, values in by_topic.items() if topic and len(values) > 1]
    return {"conflict_count": len(conflicts), "conflicts": conflicts}


def memory_compaction_plan(records: Sequence[MemoryIndexRecord], *, keep_latest: int = 5) -> Dict[str, Any]:
    sorted_records = sorted(records, key=lambda item: item.timestamp, reverse=True)
    return {"keep": [item.record_id for item in sorted_records[:keep_latest]], "compact": [item.record_id for item in sorted_records[keep_latest:]]}


def release_checklist() -> List[Dict[str, Any]]:
    items = ["cli", "dashboard", "runtime", "persistence", "templates", "connectors", "docs", "migrations", "packaging", "failure-drills"]
    return [{"id": f"release-{index + 1:03d}", "area": item, "status": "pending"} for index, item in enumerate(items)]


def migration_compatibility_matrix(versions: Sequence[int], current: int) -> Dict[str, Any]:
    return {"current": current, "versions": [{"from": version, "to": current, "supported": version <= current} for version in versions]}


def failure_drill(name: str, *, recovery_steps: Sequence[str]) -> Dict[str, Any]:
    return {"name": name, "status": "ready", "recovery_steps": list(recovery_steps), "verified": False}


def release_notes_from_tasks(tasks: Sequence[Dict[str, Any]]) -> str:
    completed = [task for task in tasks if task.get("status") == "completed"]
    lines = ["# Hermes OS Release Notes", ""]
    phases: Dict[str, int] = {}
    for task in completed:
        phases[str(task.get("phase", "Unknown"))] = phases.get(str(task.get("phase", "Unknown")), 0) + 1
    for phase, count in sorted(phases.items()):
        lines.append(f"- {phase}: {count} completed task(s)")
    return "\n".join(lines) + "\n"


def integration_suite_target() -> Dict[str, Any]:
    return {
        "name": "hermes-os-integration",
        "commands": [
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/hermes_os_integration -q",
            "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile hermes_os_integration/project_runtime_ops.py",
        ],
    }


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value
