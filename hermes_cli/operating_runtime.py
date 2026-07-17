"""Durable Hermes operating-runtime store.

This is the server-side bridge for the V1-V30 dashboard consolidation work:
project snapshots, durable memory, permission/audit checks, workbench records,
quality gates, and autonomy readiness evidence are stored in SQLite under
``HERMES_HOME``.  It intentionally stays small and dependency-free so the web
dashboard can use it before a larger production service split exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import socket
import ssl
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from hermes_constants import get_hermes_home

RuntimeKind = Literal[
    "snapshot",
    "memory",
    "permission",
    "model",
    "loop",
    "business",
    "workbench",
    "quality",
    "autonomy",
    "registry",
    "telemetry",
    "incident",
    "deployment",
    "secrets",
    "catalog",
    "finance",
    "learning",
    "eval",
    "executive",
]
RuntimeState = Literal["ready", "stored", "allowed", "blocked", "gated", "warning", "failed"]
Approval = Literal["none", "confirm", "explicit"]


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    approval: Approval
    reason: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path() -> Path:
    return get_hermes_home() / "operating_runtime.db"


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    seed_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runtime_evidence (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            subject TEXT NOT NULL,
            state TEXT NOT NULL,
            owner TEXT NOT NULL,
            detail TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_evidence_kind ON runtime_evidence(kind);
        CREATE INDEX IF NOT EXISTS idx_runtime_evidence_state ON runtime_evidence(state);

        CREATE TABLE IF NOT EXISTS runtime_audit (
            id TEXT PRIMARY KEY,
            action TEXT NOT NULL,
            actor TEXT NOT NULL,
            allowed INTEGER NOT NULL,
            approval TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_audit_action ON runtime_audit(action);

        CREATE TABLE IF NOT EXISTS runtime_workbench (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            owner TEXT NOT NULL,
            approval TEXT NOT NULL,
            artifacts TEXT NOT NULL DEFAULT '[]',
            report TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runtime_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()


def seed_db(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) AS count FROM runtime_evidence").fetchone()["count"]
    if count:
        return
    rows = [
        ("snapshot-hermes", "snapshot", "Hermes dashboard", "ready", "Hermes", "Reference health, actions, and dashboard metadata are available."),
        ("snapshot-media-engine", "snapshot", "Media Engine", "gated", "Media Engine", "Project-owned /dashboard-snapshot endpoint still needs implementation."),
        ("snapshot-khashi-vc", "snapshot", "Khashi VC", "gated", "Khashi VC", "Project-owned /dashboard-snapshot endpoint still needs implementation."),
        ("memory-decisions", "memory", "Decision records", "stored", "Hermes", "Decisions are stored in the operating runtime database."),
        ("memory-tasks", "memory", "Task records", "stored", "Hermes", "Tasks can be represented as durable runtime evidence."),
        ("permission-readonly", "permission", "Read-only checks", "allowed", "viewer", "Read-only readiness checks are allowed."),
        ("permission-deploy", "permission", "Deploy production", "gated", "admin", "Production deploys require explicit approval and audit persistence."),
        ("model-local", "model", "Local Codex routing", "ready", "Hermes", "Local-first routing is available when the local worker is reachable."),
        ("model-premium", "model", "Premium fallback", "gated", "Hermes", "Premium fallback requires explicit approval and outcome logging."),
        ("loop-dry-run", "loop", "Loop dry run", "ready", "Hermes", "Dry-run loop execution can be recorded safely."),
        ("loop-scheduler", "loop", "Production scheduler", "gated", "Operations", "Scheduling waits for permission runtime and kill switch."),
        ("business-command", "business", "TLC business command", "warning", "TLC Capital Group OS", "Business scorecards need live revenue/cost feeds."),
        ("workbench-plan", "workbench", "Plan to approval workflow", "ready", "Hermes", "Workbench plans can be recorded with approval and artifacts."),
        ("workbench-execute", "workbench", "Live execution bridge", "gated", "Operations", "Live execution requires permission and audit middleware."),
        ("quality-local", "quality", "Local visual checks", "ready", "Hermes", "Local Playwright checks cover governed dashboard routes."),
        ("quality-production", "quality", "Production URL checks", "gated", "Operations", "Production screenshot and health checks need deployed URLs."),
        ("autonomy-kill-switch", "autonomy", "Kill switch", "gated", "Operations", "Kill switch must be wired before scheduled autonomy."),
        ("autonomy-budget-breaker", "autonomy", "Budget breaker", "gated", "Operations", "Budget breaker must stop runaway loops and provider spend."),
        ("registry-root", "registry", "Production project registry", "ready", "Hermes", "Root and sibling manifests can be merged into one dashboard catalog."),
        ("telemetry-fabric", "telemetry", "Telemetry fabric", "warning", "Hermes", "Telemetry families are defined; project adapters still need richer signals."),
        ("incident-command", "incident", "Incident command queue", "ready", "Operations", "Incidents can be recorded with severity, owner, next step, and rollback path."),
        ("deployment-promotion", "deployment", "Deployment promotion rail", "gated", "Operations", "Promotion evidence is modeled; live deploy runner remains gated."),
        ("secrets-posture", "secrets", "Secrets and access posture", "gated", "Operations", "Secret presence can be tracked without exposing values."),
        ("catalog-source-schema", "catalog", "Data source catalog", "ready", "Hermes", "Data sources can track owner, cadence, freshness, retention, and consumers."),
        ("finance-attribution", "finance", "Finance and cost attribution", "warning", "TLC Capital Group OS", "Cost buckets exist; exact invoice reconciliation remains gated."),
        ("learning-engine", "learning", "Learning engine", "ready", "Hermes", "Learning evidence can be recorded and promoted through governed states."),
        ("agent-eval-lab", "eval", "Agent evaluation lab", "warning", "Hermes", "Provider eval records can be stored; outcome history is still shallow."),
        ("executive-cockpit", "executive", "Executive cockpit", "ready", "TLC Capital Group OS", "Executive rollups can reference registry, telemetry, incidents, finance, learning, and autonomy."),
        ("quality-v71-screenshot-runner", "quality", "Production screenshot runner", "gated", "Operations", "Browser screenshot capture is ready to attach to approved production sweeps once artifact storage is configured."),
        ("deployment-v72-hetzner-transport", "deployment", "Hetzner promotion transport", "gated", "Operations", "SSH promotion transport records command plans, receipts, rollback evidence, and post-deploy checks."),
        ("secrets-v73-server-posture", "secrets", "Server secret posture scanner", "gated", "Operations", "Server env-name presence checks can run without exposing raw values."),
        ("incident-v74-notification-fanout", "incident", "Incident notification fanout", "warning", "Operations", "Incident notification targets, dedupe, acknowledgement, and resolution states can be tracked."),
        ("catalog-v75-artifact-backend", "catalog", "Durable artifact backend", "warning", "Hermes", "Artifact backends can track storage mode, retention class, and cleanup posture."),
        ("telemetry-v76-outcome-adapters", "telemetry", "Remaining project outcome adapters", "warning", "Project teams", "Project dashboard outcome adapter adoption can be tracked across the portfolio."),
        ("autonomy-v77-breaker-rollout", "autonomy", "Breaker middleware rollout", "gated", "Operations", "Scheduler, provider, deploy, and autopilot live path enforcement can be audited."),
        ("eval-v78-provider-execution", "eval", "Provider eval execution", "gated", "Hermes", "Provider eval runs can be scored only after budget breakers and approval pass."),
        ("finance-v79-billing-integrations", "finance", "Billing provider integrations", "warning", "Finance", "Provider billing imports can reconcile usage against manual actuals."),
        ("deployment-v80-release-execution", "deployment", "Release train execution", "gated", "Operations", "Release train execution requires green sweeps, secrets, breakers, artifacts, incidents, rollback, and final summaries."),
    ]
    ts = now_iso()
    conn.executemany(
        """
        INSERT OR IGNORE INTO runtime_evidence
        (id, kind, subject, state, owner, detail, payload, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, '{}', ?)
        """,
        [(id_, kind, subject, state, owner, detail, ts) for id_, kind, subject, state, owner, detail in rows],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO runtime_audit
        (id, action, actor, allowed, approval, reason, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?, '{}', ?)
        """,
        ("audit-seed-readonly", "run-readiness-check", "Hermes", 1, "none", "Read-only readiness checks can run without explicit approval.", ts),
    )
    conn.commit()


def list_evidence(conn: sqlite3.Connection, kind: str | None = None) -> list[dict[str, Any]]:
    if kind:
        rows = conn.execute(
            "SELECT * FROM runtime_evidence WHERE kind = ? ORDER BY updated_at DESC, subject ASC",
            (kind,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM runtime_evidence ORDER BY updated_at DESC, subject ASC").fetchall()
    return [_evidence_row(row) for row in rows]


def upsert_evidence(
    conn: sqlite3.Connection,
    *,
    kind: RuntimeKind,
    subject: str,
    state: RuntimeState,
    owner: str,
    detail: str,
    payload: dict[str, Any] | None = None,
    id: str | None = None,
) -> dict[str, Any]:
    record_id = id or f"{kind}-{uuid4().hex}"
    ts = now_iso()
    payload_json = json.dumps(payload or {}, sort_keys=True)
    conn.execute(
        """
        INSERT INTO runtime_evidence (id, kind, subject, state, owner, detail, payload, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            kind = excluded.kind,
            subject = excluded.subject,
            state = excluded.state,
            owner = excluded.owner,
            detail = excluded.detail,
            payload = excluded.payload,
            updated_at = excluded.updated_at
        """,
        (record_id, kind, subject, state, owner, detail, payload_json, ts),
    )
    conn.commit()
    return get_evidence(conn, record_id)


def get_evidence(conn: sqlite3.Connection, id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM runtime_evidence WHERE id = ?", (id,)).fetchone()
    if row is None:
        raise KeyError(id)
    return _evidence_row(row)


def list_audit(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))
    rows = conn.execute(
        "SELECT * FROM runtime_audit ORDER BY created_at DESC LIMIT ?",
        (safe_limit,),
    ).fetchall()
    return [_audit_row(row) for row in rows]


def audit(
    conn: sqlite3.Connection,
    *,
    action: str,
    actor: str,
    decision: PermissionDecision,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = now_iso()
    record_id = f"audit-{uuid4().hex}"
    conn.execute(
        """
        INSERT INTO runtime_audit (id, action, actor, allowed, approval, reason, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            action,
            actor,
            1 if decision.allowed else 0,
            decision.approval,
            decision.reason,
            json.dumps(payload or {}, sort_keys=True),
            ts,
        ),
    )
    conn.commit()
    return _audit_row(conn.execute("SELECT * FROM runtime_audit WHERE id = ?", (record_id,)).fetchone())


def decide_permission(action: str, actor_role: str = "operator", explicit_approval: bool = False) -> PermissionDecision:
    action_lc = action.lower()
    high_risk_terms = ("deploy", "secret", "autonomy", "execute", "scheduler", "kill-switch", "budget-breaker")
    if any(term in action_lc for term in high_risk_terms):
        if actor_role != "admin":
            return PermissionDecision(False, "explicit", "Admin role is required for production-affecting actions.")
        if not explicit_approval:
            return PermissionDecision(False, "explicit", "Explicit approval is required before this high-risk action can run.")
        return PermissionDecision(True, "explicit", "High-risk action allowed with admin role and explicit approval.")
    if "refresh" in action_lc or "evaluate" in action_lc or "dry-run" in action_lc:
        return PermissionDecision(True, "confirm", "Readiness or dry-run action allowed with confirm-level approval.")
    return PermissionDecision(True, "none", "Read-only operating-runtime action allowed.")


def run_readiness_check(
    conn: sqlite3.Connection,
    *,
    stage: str,
    action: str,
    actor: str = "Hermes operator",
    actor_role: str = "operator",
    explicit_approval: bool = False,
) -> dict[str, Any]:
    decision = decide_permission(action, actor_role, explicit_approval)
    audit_record = audit(
        conn,
        action=action,
        actor=actor,
        decision=decision,
        payload={"stage": stage, "actor_role": actor_role},
    )
    evidence = upsert_evidence(
        conn,
        kind=_kind_for_stage(stage),
        subject=f"{stage} readiness check",
        state="ready" if decision.allowed else "gated",
        owner=actor_role,
        detail=decision.reason,
        payload={"action": action, "audit_id": audit_record["id"]},
    )
    return {"decision": decision.__dict__, "audit": audit_record, "evidence": evidence}


def require_permission(
    conn: sqlite3.Connection,
    *,
    action: str,
    actor: str = "Hermes operator",
    actor_role: str = "operator",
    explicit_approval: bool = False,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate and audit a runtime action before a caller executes it.

    This is the middleware-shaped primitive V24/V30 need: endpoints can call it
    before deploys, secret changes, scheduler changes, or autonomous work. It
    does not execute the action; it creates the durable permission decision.
    """
    decision = decide_permission(action, actor_role, explicit_approval)
    audit_record = audit(
        conn,
        action=action,
        actor=actor,
        decision=decision,
        payload={"actor_role": actor_role, **(payload or {})},
    )
    return {"decision": decision.__dict__, "audit": audit_record}


def record_production_check(
    conn: sqlite3.Connection,
    *,
    project: str,
    url: str,
    health_url: str = "",
    state: RuntimeState = "gated",
    detail: str = "Production route check recorded; live check still pending.",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return upsert_evidence(
        conn,
        id=f"registry-production-{_slug(project)}",
        kind="registry",
        subject=f"{project} production route",
        state=state,
        owner=project,
        detail=detail,
        payload={"url": url, "health_url": health_url, **(payload or {})},
    )


def run_production_sweep(
    conn: sqlite3.Connection,
    *,
    targets: Iterable[dict[str, Any]],
    actor: str = "Hermes operator",
    actor_role: str = "operator",
    explicit_approval: bool = False,
    live: bool = False,
) -> dict[str, Any]:
    action = "execute-production-sweep" if live else "dry-run-production-sweep"
    decision = decide_permission(action, actor_role, explicit_approval)
    audit_record = audit(
        conn,
        action=action,
        actor=actor,
        decision=decision,
        payload={"actor_role": actor_role, "live": live},
    )
    target_list = list(targets)
    records: list[dict[str, Any]] = []
    breaker = None
    if live and decision.allowed:
        breaker = check_circuit_breakers(
            conn,
            project="*",
            action=action,
            context={"target_count": len(target_list), "actor_role": actor_role},
        )
        if not breaker["allowed"]:
            record_incident(
                conn,
                title="Production sweep blocked by circuit breaker",
                severity="critical",
                owner="Operations",
                next_step="Review active breaker controls before running live production sweeps.",
                source="production-sweep",
                status="open",
            )

    for target in target_list:
        project = str(target.get("project") or "Unknown project")
        url = str(target.get("url") or "")
        health_url = str(target.get("health_url") or target.get("healthUrl") or "")
        probe = _probe_production_target(url=url, health_url=health_url) if live and decision.allowed and (not breaker or breaker["allowed"]) else None
        state: RuntimeState = (
            "ready"
            if probe and probe["ok"]
            else "failed"
            if probe
            else "blocked"
            if live and decision.allowed and breaker and not breaker["allowed"]
            else "gated"
        )
        detail = (
            "Live production sweep passed DNS/TLS/HTTP checks."
            if probe and probe["ok"]
            else "Live production sweep failed; incident recorded."
            if probe
            else "Production sweep blocked by an active circuit breaker."
            if live and decision.allowed and breaker and not breaker["allowed"]
            else "Production sweep planned; live network execution still gated."
        )
        payload = {
            "live": live,
            "audit_id": audit_record["id"],
            "checks": ["dns", "tls", "health", "snapshot"],
            "breaker": breaker["record"] if breaker else None,
            "probe": probe,
        }
        artifact_uri = str(target.get("artifact_uri") or target.get("artifactUri") or "")
        if artifact_uri:
            artifact = index_evidence_artifact(
                conn,
                title=f"{project} production sweep artifact",
                artifact_type=str(target.get("artifact_type") or target.get("artifactType") or "production-sweep"),
                uri=artifact_uri,
                source="production-sweep",
                retention=str(target.get("retention") or "standard"),
                evidence_id=f"registry-production-{_slug(project)}",
            )
            payload["artifact_id"] = artifact["id"]
        records.append(
            record_production_check(
                conn,
                project=project,
                url=url,
                health_url=health_url,
                state=state,
                detail=detail,
                payload=payload,
            )
        )
        if probe and not probe["ok"]:
            record_incident(
                conn,
                title=f"Production sweep failed: {project}",
                severity="critical",
                owner=project,
                next_step=f"Review failed checks for {url or health_url or project} before promotion.",
                rollback="Pause release train or restore last healthy route if this followed a deploy.",
                source="production-sweep",
                status="open",
            )
    return {"decision": decision.__dict__, "audit": audit_record, "records": records}


def record_incident(
    conn: sqlite3.Connection,
    *,
    title: str,
    severity: str,
    owner: str,
    next_step: str,
    rollback: str = "",
    source: str = "",
    status: str = "open",
) -> dict[str, Any]:
    state: RuntimeState = "failed" if severity == "critical" else "warning" if severity in {"high", "warning"} else "ready"
    return upsert_evidence(
        conn,
        id=f"incident-{_slug(title)}",
        kind="incident",
        subject=title,
        state=state,
        owner=owner,
        detail=next_step,
        payload={"severity": severity, "rollback": rollback, "source": source, "status": status},
    )


def record_gate_coverage(
    conn: sqlite3.Connection,
    *,
    handler: str,
    risk: str,
    covered: bool,
    owner: str = "Operations",
    source: str = "",
) -> dict[str, Any]:
    state: RuntimeState = "ready" if covered else "warning"
    evidence = upsert_evidence(
        conn,
        id=f"permission-gate-coverage-{_slug(handler)}",
        kind="permission",
        subject=f"{handler} gate coverage",
        state=state,
        owner=owner,
        detail=f"{risk} handler is {'covered by' if covered else 'missing'} command gate enforcement.",
        payload={"handler": handler, "risk": risk, "covered": covered, "source": source},
    )
    if not covered:
        record_incident(
            conn,
            title=f"Missing command gate: {handler}",
            severity="warning",
            owner=owner,
            next_step="Route this handler through the V42 permission decision primitive before live use.",
            source=source or handler,
            status="open",
        )
    return evidence


def record_adapter_rollout(
    conn: sqlite3.Connection,
    *,
    project: str,
    manifest_url: str = "",
    snapshot_url: str = "",
    missing_fields: Iterable[str] = (),
    owner: str = "Hermes",
) -> dict[str, Any]:
    missing = list(missing_fields)
    state: RuntimeState = "ready" if not missing else "warning"
    return upsert_evidence(
        conn,
        id=f"telemetry-adapter-rollout-{_slug(project)}",
        kind="telemetry",
        subject=f"{project} telemetry adapter",
        state=state,
        owner=owner,
        detail="Telemetry adapter adopted." if not missing else f"Telemetry adapter missing: {', '.join(missing)}.",
        payload={"project": project, "manifest_url": manifest_url, "snapshot_url": snapshot_url, "missing_fields": missing},
    )


def run_incident_automation(
    conn: sqlite3.Connection,
    *,
    sources: Iterable[dict[str, Any]],
    owner: str = "Operations",
    auto_remediate: bool = False,
) -> dict[str, Any]:
    incidents: list[dict[str, Any]] = []
    for source in sources:
        status = str(source.get("status") or source.get("state") or "warning")
        if status in {"ready", "healthy", "verified"}:
            continue
        title = str(source.get("title") or source.get("subject") or source.get("source") or "Operational signal")
        incidents.append(
            record_incident(
                conn,
                title=f"Automated incident: {title}",
                severity=str(source.get("severity") or "warning"),
                owner=str(source.get("owner") or owner),
                next_step=str(source.get("next_step") or source.get("nextStep") or "Review source evidence and assign remediation."),
                rollback=str(source.get("rollback") or ""),
                source=str(source.get("source") or title),
                status="open",
            )
        )
    policy = upsert_evidence(
        conn,
        id="incident-automation-policy",
        kind="incident",
        subject="Incident automation policy",
        state="gated" if auto_remediate else "ready",
        owner=owner,
        detail="Auto-remediation requested; approval still required." if auto_remediate else "Incident creation enabled; remediation remains manual.",
        payload={"auto_remediate": auto_remediate, "created": len(incidents)},
    )
    return {"policy": policy, "incidents": incidents}


def record_deployment(
    conn: sqlite3.Connection,
    *,
    project: str,
    version: str,
    environment: str,
    status: str,
    migration_required: bool = False,
    rollback: str = "",
    evidence: Iterable[str] = (),
) -> dict[str, Any]:
    state: RuntimeState = "ready" if status in {"current", "healthy", "deployed"} else "failed" if status in {"failed", "rolled-back"} else "gated"
    return upsert_evidence(
        conn,
        id=f"deployment-{_slug(project)}-{_slug(environment)}",
        kind="deployment",
        subject=f"{project} {environment} deployment",
        state=state,
        owner=project,
        detail=f"Deployment {status}; version {version}.",
        payload={
            "version": version,
            "environment": environment,
            "status": status,
            "migration_required": migration_required,
            "rollback": rollback,
            "evidence": list(evidence),
        },
    )


def plan_promotion_execution(
    conn: sqlite3.Connection,
    *,
    project: str,
    version: str,
    environment: str = "production",
    app_dir: str = "",
    url: str = "",
    migration_required: bool = False,
    live: bool = False,
    actor: str = "Hermes operator",
    actor_role: str = "operator",
    explicit_approval: bool = False,
    execute_commands: bool = False,
    command_runner: Callable[[list[str], Path | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    action = "execute-hetzner-promotion" if live else "dry-run-hetzner-promotion"
    decision = decide_permission(action, actor_role, explicit_approval)
    command_plan = _hetzner_promotion_command_plan(app_dir=app_dir, migration_required=migration_required, url=url)
    audit_record = audit(
        conn,
        action=action,
        actor=actor,
        decision=decision,
        payload={
            "project": project,
            "environment": environment,
            "app_dir": app_dir,
            "url": url,
            "migration_required": migration_required,
            "execute_commands": execute_commands,
            "command_plan": command_plan,
        },
    )
    breaker = None
    execution: dict[str, Any] | None = None
    if live and decision.allowed:
        breaker = check_circuit_breakers(
            conn,
            project=project,
            action=action,
            context={"environment": environment, "app_dir": app_dir, "migration_required": migration_required},
        )
        if not breaker["allowed"]:
            record_incident(
                conn,
                title=f"Promotion blocked by circuit breaker: {project}",
                severity="critical",
                owner="Operations",
                next_step="Disable or resolve the active breaker before promoting this project.",
                rollback=f"Keep previous {project} deployment active.",
                source="hetzner-promotion",
                status="open",
            )
        elif execute_commands:
            execution = _execute_promotion_commands(
                command_plan,
                cwd=Path(app_dir) if app_dir else None,
                command_runner=command_runner,
            )
            if not execution["ok"]:
                record_incident(
                    conn,
                    title=f"Promotion execution failed: {project}",
                    severity="critical",
                    owner=project,
                    next_step="Review promotion command output and run rollback if production health is degraded.",
                    rollback=f"Restore previous {project} deployment from {app_dir or 'production app directory'}.",
                    source="hetzner-promotion",
                    status="open",
                )
    status = (
        "deployed"
        if live and decision.allowed and (not breaker or breaker["allowed"]) and (not execute_commands or (execution and execution["ok"]))
        else "failed"
        if execution and not execution["ok"]
        else "blocked"
        if live and decision.allowed and breaker and not breaker["allowed"]
        else "planned"
    )
    deployment = record_deployment(
        conn,
        project=project,
        version=version,
        environment=environment,
        status=status,
        migration_required=migration_required,
        rollback=f"Restore previous {project} deployment from {app_dir or 'production app directory'}.",
        evidence=[audit_record["id"], "validate", "build", "test", "migrate", "sync", "restart", "verify"],
    )
    deployment["payload"]["command_plan"] = command_plan
    deployment["payload"]["breaker"] = breaker["record"] if breaker else None
    deployment["payload"]["execution"] = execution
    return {"decision": decision.__dict__, "audit": audit_record, "deployment": deployment, "command_plan": command_plan, "breaker": breaker, "execution": execution}


def record_cost(
    conn: sqlite3.Connection,
    *,
    project: str,
    business_unit: str,
    bucket: str,
    amount: float | None = None,
    unit: str = "estimated",
    period: str = "monthly",
    source: str = "manual",
) -> dict[str, Any]:
    state: RuntimeState = "ready" if amount is not None else "warning"
    detail = f"{bucket} cost attributed to {business_unit}; {amount if amount is not None else 'rate missing'} {unit} per {period}."
    return upsert_evidence(
        conn,
        id=f"finance-{_slug(project)}-{_slug(bucket)}",
        kind="finance",
        subject=f"{project} {bucket}",
        state=state,
        owner=business_unit,
        detail=detail,
        payload={"project": project, "business_unit": business_unit, "bucket": bucket, "amount": amount, "unit": unit, "period": period, "source": source},
    )


def scan_secret_presence(
    conn: sqlite3.Connection,
    *,
    project: str,
    required: Iterable[str],
    present: Iterable[str] = (),
    scope: str = "unknown",
    live: bool = False,
    actor: str = "Hermes operator",
    actor_role: str = "operator",
    explicit_approval: bool = False,
) -> dict[str, Any]:
    action = "scan-live-secret-presence" if live else "dry-run-secret-presence"
    decision = decide_permission(action, actor_role, explicit_approval)
    audit_record = audit(conn, action=action, actor=actor, decision=decision, payload={"project": project, "scope": scope, "live": live})
    required_names = list(required)
    present_names = set(present)
    missing = [name for name in required_names if name not in present_names]
    state: RuntimeState = "ready" if not missing else "warning"
    record = upsert_evidence(
        conn,
        id=f"secrets-presence-scan-{_slug(project)}",
        kind="secrets",
        subject=f"{project} secret presence",
        state=state,
        owner="Operations",
        detail="Required secret names are present." if not missing else f"Missing secret names: {', '.join(missing)}.",
        payload={"project": project, "required": required_names, "present_count": len(present_names), "missing": missing, "scope": scope, "live": live, "audit_id": audit_record["id"]},
    )
    if missing:
        record_incident(
            conn,
            title=f"Missing secrets: {project}",
            severity="high",
            owner="Operations",
            next_step=f"Add missing secret names in {scope}: {', '.join(missing)}.",
            source="secret-presence-scan",
            status="open",
        )
    return {"decision": decision.__dict__, "audit": audit_record, "record": record}


def scan_github_secret_presence(
    conn: sqlite3.Connection,
    *,
    project: str,
    repo: str,
    required: Iterable[str],
    include_variables: bool = True,
    live: bool = False,
    actor: str = "Hermes operator",
    actor_role: str = "operator",
    explicit_approval: bool = False,
    command_runner: Callable[[list[str], Path | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    action = "scan-live-github-secrets" if live else "dry-run-github-secrets"
    decision = decide_permission(action, actor_role, explicit_approval)
    audit_record = audit(
        conn,
        action=action,
        actor=actor,
        decision=decision,
        payload={"project": project, "repo": repo, "include_variables": include_variables, "live": live},
    )
    required_names = list(required)
    present_names: set[str] = set()
    cli_results: list[dict[str, Any]] = []
    if live and decision.allowed:
        commands = [["gh", "secret", "list", "--repo", repo]]
        if include_variables:
            commands.append(["gh", "variable", "list", "--repo", repo])
        for command in commands:
            result = (command_runner or _run_command)(command, None)
            cli_results.append(_redact_command_result(result))
            if result.get("ok"):
                present_names.update(_parse_gh_name_list(str(result.get("stdout") or "")))
    missing = [name for name in required_names if name not in present_names]
    state: RuntimeState = "ready" if live and decision.allowed and not missing else "warning" if live and decision.allowed else "gated"
    record = upsert_evidence(
        conn,
        id=f"secrets-github-scan-{_slug(project)}",
        kind="secrets",
        subject=f"{project} GitHub secret scan",
        state=state,
        owner="Operations",
        detail="GitHub secret and variable names are present." if live and decision.allowed and not missing else "GitHub secret scan planned or missing required names.",
        payload={
            "project": project,
            "repo": repo,
            "required": required_names,
            "present_count": len(present_names),
            "missing": missing if live and decision.allowed else [],
            "include_variables": include_variables,
            "live": live,
            "audit_id": audit_record["id"],
            "cli_results": cli_results,
        },
    )
    if live and decision.allowed and missing:
        record_incident(
            conn,
            title=f"Missing GitHub secrets: {project}",
            severity="high",
            owner="Operations",
            next_step=f"Add missing GitHub secret or variable names for {repo}: {', '.join(missing)}.",
            source="github-secret-scan",
            status="open",
        )
    return {"decision": decision.__dict__, "audit": audit_record, "record": record}


def import_cost_reconciliation(
    conn: sqlite3.Connection,
    *,
    records: Iterable[dict[str, Any]],
    source: str = "manual-rate-sheet",
) -> dict[str, Any]:
    costs: list[dict[str, Any]] = []
    for record in records:
        costs.append(
            record_cost(
                conn,
                project=str(record.get("project") or "Unknown project"),
                business_unit=str(record.get("business_unit") or record.get("businessUnit") or "TLC Capital Group OS"),
                bucket=str(record.get("bucket") or "operations"),
                amount=record.get("amount"),
                unit=str(record.get("unit") or "actual"),
                period=str(record.get("period") or "monthly"),
                source=str(record.get("source") or source),
            )
        )
    summary_record = upsert_evidence(
        conn,
        id=f"finance-reconciliation-{_slug(source)}",
        kind="finance",
        subject=f"{source} reconciliation import",
        state="ready" if costs else "warning",
        owner="Finance",
        detail=f"Imported {len(costs)} cost reconciliation records.",
        payload={"source": source, "count": len(costs)},
    )
    return {"summary": summary_record, "costs": costs}


def import_manual_billing(
    conn: sqlite3.Connection,
    *,
    records: Iterable[dict[str, Any]],
    source: str = "manual-billing-import",
    imported_by: str = "Finance",
    period: str = "",
) -> dict[str, Any]:
    enriched = []
    for record in records:
        next_record = dict(record)
        if period and not next_record.get("period"):
            next_record["period"] = period
        next_record["source"] = next_record.get("source") or source
        enriched.append(next_record)
    result = import_cost_reconciliation(conn, records=enriched, source=source)
    result["summary"]["payload"]["imported_by"] = imported_by
    result["summary"]["payload"]["period"] = period
    return result


def record_data_source(
    conn: sqlite3.Connection,
    *,
    name: str,
    owner: str,
    cadence: str,
    freshness: str,
    retention: str = "",
    consumers: Iterable[str] = (),
) -> dict[str, Any]:
    stale_terms = {"stale", "missing", "unknown", "overdue"}
    state: RuntimeState = "warning" if freshness.lower() in stale_terms else "ready"
    return upsert_evidence(
        conn,
        id=f"catalog-{_slug(name)}",
        kind="catalog",
        subject=name,
        state=state,
        owner=owner,
        detail=f"{cadence} source; freshness={freshness}; retention={retention or 'not declared'}.",
        payload={"name": name, "cadence": cadence, "freshness": freshness, "retention": retention, "consumers": list(consumers)},
    )


def record_learning(
    conn: sqlite3.Connection,
    *,
    title: str,
    source: str,
    evidence_count: int = 1,
    confidence: float = 0.0,
    recommendation: str = "",
    status: str = "candidate",
) -> dict[str, Any]:
    state: RuntimeState = "ready" if status in {"finding", "policy-proposed"} else "warning" if status == "candidate" else "gated"
    return upsert_evidence(
        conn,
        id=f"learning-{_slug(title)}",
        kind="learning",
        subject=title,
        state=state,
        owner="Hermes",
        detail=recommendation or f"{status} learning from {source}.",
        payload={"source": source, "evidence_count": evidence_count, "confidence": confidence, "status": status},
    )


def ingest_learning_batch(
    conn: sqlite3.Connection,
    *,
    events: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for event in events:
        records.append(
            record_learning(
                conn,
                title=str(event.get("title") or event.get("subject") or "Learning event"),
                source=str(event.get("source") or "unknown"),
                evidence_count=int(event.get("evidence_count") or event.get("evidenceCount") or 1),
                confidence=float(event.get("confidence") or 0.0),
                recommendation=str(event.get("recommendation") or ""),
                status=str(event.get("status") or "candidate"),
            )
        )
    return {"learning": records}


def ingest_project_outcomes(
    conn: sqlite3.Connection,
    *,
    project: str,
    outcomes: Iterable[dict[str, Any]] = (),
    url: str = "",
    live: bool = False,
    actor: str = "Hermes operator",
    actor_role: str = "operator",
    explicit_approval: bool = False,
) -> dict[str, Any]:
    action = "execute-project-outcome-ingest" if live and url else "ingest-project-outcomes"
    decision = decide_permission(action, actor_role, explicit_approval)
    audit_record = audit(
        conn,
        action=action,
        actor=actor,
        decision=decision,
        payload={"project": project, "url": url, "live": live},
    )
    outcome_list = list(outcomes)
    fetch_result: dict[str, Any] | None = None
    if live and url and decision.allowed:
        fetch_result = _fetch_json(url)
        if fetch_result["ok"]:
            body = fetch_result.get("body") or {}
            if isinstance(body, dict):
                outcome_list.extend(list(body.get("outcomes") or []))
        else:
            record_incident(
                conn,
                title=f"Project outcome ingest failed: {project}",
                severity="high",
                owner=project,
                next_step=f"Check the Hermes outcome adapter at {url}.",
                source="project-outcome-ingest",
                status="open",
            )
    records: list[dict[str, Any]] = []
    for outcome in outcome_list:
        records.append(
            record_learning(
                conn,
                title=str(outcome.get("id") or outcome.get("title") or f"{project} outcome"),
                source=str(outcome.get("source") or project),
                evidence_count=int(outcome.get("evidence_count") or outcome.get("evidenceCount") or 1),
                confidence=float(outcome.get("confidence") or 0.0),
                recommendation=str(outcome.get("evidence") or outcome.get("recommendation") or ""),
                status="finding" if str(outcome.get("state") or "") == "passed" else "candidate",
            )
        )
    adapter_record = record_adapter_run(
        conn,
        adapter="project-outcome-emitter",
        project=project,
        kind="learning",
        status="emitted" if records else "failed" if fetch_result and not fetch_result["ok"] else "planned",
        live=live,
        actor=actor,
        actor_role=actor_role,
        explicit_approval=explicit_approval,
        payload={"url": url, "outcome_count": len(records), "audit_id": audit_record["id"], "fetch": _redact_fetch_result(fetch_result)},
    )
    return {"decision": decision.__dict__, "audit": audit_record, "adapter": adapter_record, "learning": records, "fetch": fetch_result}


def record_eval(
    conn: sqlite3.Connection,
    *,
    provider: str,
    task_family: str,
    correctness: float,
    cost_score: float,
    latency_score: float,
    verdict: str = "insufficient-data",
) -> dict[str, Any]:
    state: RuntimeState = "ready" if verdict == "passed" else "failed" if verdict == "failed" else "warning"
    return upsert_evidence(
        conn,
        id=f"eval-{_slug(provider)}-{_slug(task_family)}",
        kind="eval",
        subject=f"{provider} on {task_family}",
        state=state,
        owner="Hermes",
        detail=f"{verdict}: correctness={correctness}, cost={cost_score}, latency={latency_score}.",
        payload={"provider": provider, "task_family": task_family, "correctness": correctness, "cost_score": cost_score, "latency_score": latency_score, "verdict": verdict},
    )


def run_golden_eval_batch(
    conn: sqlite3.Connection,
    *,
    runs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for run in runs:
        records.append(
            record_eval(
                conn,
                provider=str(run.get("provider") or "unknown-provider"),
                task_family=str(run.get("task_family") or run.get("taskFamily") or "general"),
                correctness=float(run.get("correctness") or 0.0),
                cost_score=float(run.get("cost_score") or run.get("costScore") or 0.0),
                latency_score=float(run.get("latency_score") or run.get("latencyScore") or 0.0),
                verdict=str(run.get("verdict") or "insufficient-data"),
            )
        )
    return {"evals": records}


def set_autonomy_control(
    conn: sqlite3.Connection,
    *,
    project: str,
    control: str,
    enabled: bool,
    limit: str = "",
    reason: str = "",
) -> dict[str, Any]:
    state: RuntimeState = "ready" if enabled else "gated"
    return upsert_evidence(
        conn,
        id=f"autonomy-control-{_slug(project)}-{_slug(control)}",
        kind="autonomy",
        subject=f"{project} {control}",
        state=state,
        owner="Operations",
        detail=reason or f"{control} is {'enabled' if enabled else 'gated'} for {project}.",
        payload={"project": project, "control": control, "enabled": enabled, "limit": limit},
    )


def check_circuit_breakers(
    conn: sqlite3.Connection,
    *,
    project: str,
    action: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controls = [
        row for row in list_evidence(conn, "autonomy")
        if row["payload"].get("project") in {project, "*", "all"} and row["payload"].get("enabled")
    ]
    matched = [
        row for row in controls
        if row["payload"].get("control") in {"kill-switch", "budget-breaker", "provider-spend-cap", "scheduler-stop", "project-autonomy-limit"}
    ]
    allowed = not matched
    record = upsert_evidence(
        conn,
        id=f"autonomy-breaker-check-{_slug(project)}-{_slug(action)}",
        kind="autonomy",
        subject=f"{project} {action} breaker check",
        state="ready" if allowed else "blocked",
        owner="Operations",
        detail="No matching breaker is active." if allowed else f"Blocked by {len(matched)} active breaker control(s).",
        payload={"project": project, "action": action, "allowed": allowed, "matched": [row["id"] for row in matched], "context": context or {}},
    )
    if not allowed:
        record_incident(
            conn,
            title=f"Circuit breaker blocked action: {project} {action}",
            severity="critical",
            owner="Operations",
            next_step="Resolve the matching breaker control before re-running this action.",
            source="breaker-check",
            status="open",
        )
    return {"allowed": allowed, "matched": matched, "record": record}


def record_adapter_run(
    conn: sqlite3.Connection,
    *,
    adapter: str,
    project: str,
    kind: RuntimeKind,
    status: str = "planned",
    live: bool = False,
    actor: str = "Hermes operator",
    actor_role: str = "operator",
    explicit_approval: bool = False,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = f"execute-{_slug(adapter)}" if live else f"plan-{_slug(adapter)}"
    decision = decide_permission(action, actor_role, explicit_approval)
    audit_record = audit(
        conn,
        action=action,
        actor=actor,
        decision=decision,
        payload={"adapter": adapter, "project": project, "actor_role": actor_role, "live": live, **(payload or {})},
    )
    state: RuntimeState = "ready" if decision.allowed and status in {"passed", "executed", "imported", "emitted", "indexed"} else "failed" if status == "failed" else "gated" if live else "warning"
    evidence = upsert_evidence(
        conn,
        id=f"{kind}-adapter-{_slug(adapter)}-{_slug(project)}",
        kind=kind,
        subject=f"{project} {adapter}",
        state=state,
        owner=project,
        detail=f"{adapter} {status}; live={live}; approval={decision.approval}.",
        payload={"adapter": adapter, "project": project, "status": status, "live": live, "audit_id": audit_record["id"], **(payload or {})},
    )
    return {"decision": decision.__dict__, "audit": audit_record, "evidence": evidence}


def record_incident_subscription(
    conn: sqlite3.Connection,
    *,
    source: str,
    owner: str,
    severity: str = "warning",
    dedupe_key: str = "",
    enabled: bool = True,
) -> dict[str, Any]:
    return upsert_evidence(
        conn,
        id=f"incident-subscription-{_slug(source)}-{_slug(dedupe_key or owner)}",
        kind="incident",
        subject=f"{source} incident subscription",
        state="ready" if enabled else "gated",
        owner=owner,
        detail=f"{source} will create {severity} incidents with dedupe key {dedupe_key or 'source'}." if enabled else f"{source} incident subscription disabled.",
        payload={"source": source, "owner": owner, "severity": severity, "dedupe_key": dedupe_key or source, "enabled": enabled},
    )


def index_evidence_artifact(
    conn: sqlite3.Connection,
    *,
    title: str,
    artifact_type: str,
    uri: str,
    source: str,
    retention: str = "standard",
    content_hash: str = "",
    evidence_id: str = "",
) -> dict[str, Any]:
    content_fingerprint = content_hash or hashlib.sha256(f"{title}|{artifact_type}|{uri}|{source}".encode("utf-8")).hexdigest()
    return upsert_evidence(
        conn,
        id=f"catalog-artifact-{_slug(title)}",
        kind="catalog",
        subject=title,
        state="ready" if uri else "warning",
        owner="Hermes",
        detail=f"{artifact_type} artifact indexed from {source}; retention={retention}.",
        payload={"artifact_type": artifact_type, "uri": uri, "source": source, "retention": retention, "content_hash": content_fingerprint, "evidence_id": evidence_id},
    )


def plan_release_train(
    conn: sqlite3.Connection,
    *,
    train: str,
    projects: Iterable[str],
    version: str,
    approved: bool = False,
    rollback: str = "",
) -> dict[str, Any]:
    project_list = list(projects)
    state: RuntimeState = "gated" if not approved else "ready"
    return upsert_evidence(
        conn,
        id=f"deployment-release-train-{_slug(train)}",
        kind="deployment",
        subject=f"{train} release train",
        state=state,
        owner="Operations",
        detail=f"Release train {train} targets {len(project_list)} project(s); version {version}.",
        payload={"train": train, "projects": project_list, "version": version, "approved": approved, "rollback": rollback},
    )


def record_production_screenshot_run(
    conn: sqlite3.Connection,
    *,
    project: str,
    url: str,
    viewport: str = "desktop",
    artifact_uri: str = "",
    status: str = "planned",
    blank_detected: bool = False,
    live: bool = False,
) -> dict[str, Any]:
    state: RuntimeState = "ready" if status in {"captured", "passed"} and not blank_detected else "failed" if blank_detected or status == "failed" else "gated" if live else "warning"
    artifact_id = ""
    if artifact_uri:
        artifact_id = index_evidence_artifact(
            conn,
            title=f"{project} {viewport} production screenshot",
            artifact_type="screenshot",
            uri=artifact_uri,
            source="production-screenshot-runner",
            retention="short",
        )["id"]
    record = upsert_evidence(
        conn,
        id=f"quality-screenshot-{_slug(project)}-{_slug(viewport)}",
        kind="quality",
        subject=f"{project} {viewport} screenshot",
        state=state,
        owner=project,
        detail="Production screenshot captured." if state == "ready" else "Production screenshot requires review or approval.",
        payload={"project": project, "url": url, "viewport": viewport, "artifact_uri": artifact_uri, "artifact_id": artifact_id, "status": status, "blank_detected": blank_detected, "live": live},
    )
    if state == "failed":
        record_incident(conn, title=f"Production screenshot failed: {project}", severity="high", owner=project, next_step="Review screenshot artifact and route rendering.", source="production-screenshot-runner")
    return record


def record_hetzner_transport_run(
    conn: sqlite3.Connection,
    *,
    project: str,
    service_key: str,
    version: str,
    script: str = "/root/apps/deploy/scripts/promote-service.sh",
    status: str = "planned",
    receipt_uri: str = "",
    rollback: str = "",
) -> dict[str, Any]:
    receipt_id = ""
    if receipt_uri:
        receipt_id = index_evidence_artifact(conn, title=f"{project} promotion receipt", artifact_type="deploy-receipt", uri=receipt_uri, source="hetzner-promotion-transport", retention="standard")["id"]
    deployment = record_deployment(
        conn,
        project=project,
        version=version,
        environment="production",
        status="deployed" if status in {"executed", "promoted"} else "failed" if status == "failed" else "pending",
        rollback=rollback,
        evidence=[receipt_id] if receipt_id else [],
    )
    deployment["payload"]["transport"] = {"service_key": service_key, "script": script, "status": status, "receipt_uri": receipt_uri}
    return deployment


def record_server_secret_posture(
    conn: sqlite3.Connection,
    *,
    project: str,
    required: Iterable[str],
    present: Iterable[str] = (),
    source: str = "hetzner-env",
) -> dict[str, Any]:
    return scan_secret_presence(conn, project=project, required=required, present=present, scope=source, live=False)["record"]


def record_incident_notification_target(
    conn: sqlite3.Connection,
    *,
    channel: str,
    target: str,
    severity: str = "high",
    enabled: bool = True,
    cooldown_minutes: int = 30,
) -> dict[str, Any]:
    return upsert_evidence(
        conn,
        id=f"incident-fanout-{_slug(channel)}-{_slug(target)}",
        kind="incident",
        subject=f"{channel} incident fanout",
        state="ready" if enabled else "gated",
        owner="Operations",
        detail=f"{channel} fanout {'enabled' if enabled else 'disabled'} for {severity}+ incidents.",
        payload={"channel": channel, "target": target, "severity": severity, "enabled": enabled, "cooldown_minutes": cooldown_minutes},
    )


def record_artifact_backend(
    conn: sqlite3.Connection,
    *,
    name: str,
    mode: str,
    base_uri: str = "",
    retention_days: int = 30,
    cleanup_enabled: bool = False,
) -> dict[str, Any]:
    return upsert_evidence(
        conn,
        id=f"catalog-artifact-backend-{_slug(name)}",
        kind="catalog",
        subject=f"{name} artifact backend",
        state="ready" if base_uri else "warning",
        owner="Hermes",
        detail=f"{mode} artifact backend; retention={retention_days} days; cleanup={'enabled' if cleanup_enabled else 'manual'}.",
        payload={"name": name, "mode": mode, "base_uri": base_uri, "retention_days": retention_days, "cleanup_enabled": cleanup_enabled},
    )


def record_project_outcome_adapter_adoption(
    conn: sqlite3.Connection,
    *,
    project: str,
    endpoint: str = "/api/hermes/outcomes",
    adopted: bool = False,
    last_checked: str = "",
) -> dict[str, Any]:
    return upsert_evidence(
        conn,
        id=f"telemetry-outcome-adapter-{_slug(project)}",
        kind="telemetry",
        subject=f"{project} outcome adapter",
        state="ready" if adopted else "warning",
        owner=project,
        detail="Project outcome adapter adopted." if adopted else "Project outcome adapter still missing or unverified.",
        payload={"project": project, "endpoint": endpoint, "adopted": adopted, "last_checked": last_checked},
    )


def record_breaker_middleware_rollout(
    conn: sqlite3.Connection,
    *,
    project: str,
    path_class: str,
    wrapped: bool = False,
    test_status: str = "planned",
) -> dict[str, Any]:
    return upsert_evidence(
        conn,
        id=f"autonomy-breaker-rollout-{_slug(project)}-{_slug(path_class)}",
        kind="autonomy",
        subject=f"{project} {path_class} breaker rollout",
        state="ready" if wrapped and test_status == "passed" else "warning" if wrapped else "gated",
        owner=project,
        detail=f"{path_class} path {'is' if wrapped else 'is not'} breaker-wrapped; test={test_status}.",
        payload={"project": project, "path_class": path_class, "wrapped": wrapped, "test_status": test_status},
    )


def record_provider_eval_execution(
    conn: sqlite3.Connection,
    *,
    provider: str,
    task_family: str,
    artifact_uri: str = "",
    correctness: float = 0.0,
    cost_score: float = 0.0,
    latency_score: float = 0.0,
    verdict: str = "insufficient-data",
) -> dict[str, Any]:
    eval_record = record_eval(conn, provider=provider, task_family=task_family, correctness=correctness, cost_score=cost_score, latency_score=latency_score, verdict=verdict)
    if artifact_uri:
        artifact = index_evidence_artifact(conn, title=f"{provider} {task_family} eval artifact", artifact_type="eval", uri=artifact_uri, source="provider-eval-execution", retention="standard", evidence_id=eval_record["id"])
        eval_record["payload"]["artifact_id"] = artifact["id"]
    return eval_record


def record_billing_provider_integration(
    conn: sqlite3.Connection,
    *,
    provider: str,
    project: str,
    amount: float | None = None,
    period: str = "monthly",
    source: str = "provider-api",
    variance: float | None = None,
) -> dict[str, Any]:
    cost = record_cost(conn, project=project, business_unit="TLC Capital Group OS", bucket=f"{provider} usage", amount=amount, unit="usd", period=period, source=source)
    cost["payload"]["variance"] = variance
    return cost


def execute_release_train_record(
    conn: sqlite3.Connection,
    *,
    train: str,
    projects: Iterable[str],
    version: str,
    gates_passed: bool = False,
    approved: bool = False,
    rollback: str = "",
) -> dict[str, Any]:
    release = plan_release_train(conn, train=train, projects=projects, version=version, approved=approved and gates_passed, rollback=rollback)
    release["payload"]["gates_passed"] = gates_passed
    release["payload"]["execution_state"] = "ready" if approved and gates_passed else "blocked"
    if approved and not gates_passed:
        record_incident(conn, title=f"Release train gates failed: {train}", severity="high", owner="Operations", next_step="Resolve failed hard gates before executing the release train.", source="release-train-execution")
    return release


def create_workbench_item(
    conn: sqlite3.Connection,
    *,
    title: str,
    owner: str = "Hermes",
    approval: Approval = "explicit",
    status: str = "planned",
    artifacts: Iterable[str] = (),
    report: str = "",
) -> dict[str, Any]:
    ts = now_iso()
    record_id = f"workbench-{uuid4().hex}"
    conn.execute(
        """
        INSERT INTO runtime_workbench (id, title, status, owner, approval, artifacts, report, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (record_id, title, status, owner, approval, json.dumps(list(artifacts)), report, ts),
    )
    upsert_evidence(
        conn,
        id=f"workbench-evidence-{record_id}",
        kind="workbench",
        subject=title,
        state="gated" if approval == "explicit" else "ready",
        owner=owner,
        detail=f"Workbench item {status}; approval={approval}",
        payload={"workbench_id": record_id},
    )
    conn.commit()
    return get_workbench_item(conn, record_id)


def list_workbench(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM runtime_workbench ORDER BY updated_at DESC").fetchall()
    return [_workbench_row(row) for row in rows]


def get_workbench_item(conn: sqlite3.Connection, id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM runtime_workbench WHERE id = ?", (id,)).fetchone()
    if row is None:
        raise KeyError(id)
    return _workbench_row(row)


def summary(conn: sqlite3.Connection) -> dict[str, Any]:
    evidence = list_evidence(conn)
    audit_rows = list_audit(conn)
    return {
        "db_path": str(db_path()),
        "evidence_count": len(evidence),
        "audit_count": conn.execute("SELECT COUNT(*) AS count FROM runtime_audit").fetchone()["count"],
        "workbench_count": conn.execute("SELECT COUNT(*) AS count FROM runtime_workbench").fetchone()["count"],
        "ready_count": sum(1 for row in evidence if row["state"] in {"ready", "stored", "allowed"}),
        "gated_count": sum(1 for row in evidence if row["state"] in {"gated", "blocked", "warning", "failed"}),
        "latest_audit": audit_rows[0] if audit_rows else None,
    }


def _kind_for_stage(stage: str) -> RuntimeKind:
    return {
        "V22": "snapshot",
        "V23": "memory",
        "V24": "permission",
        "V25": "model",
        "V26": "loop",
        "V27": "business",
        "V28": "workbench",
        "V29": "quality",
        "V30": "autonomy",
        "V31": "registry",
        "V32": "telemetry",
        "V33": "incident",
        "V34": "deployment",
        "V35": "secrets",
        "V36": "catalog",
        "V37": "finance",
        "V38": "learning",
        "V39": "eval",
        "V40": "executive",
        "V41": "registry",
        "V42": "permission",
        "V43": "telemetry",
        "V44": "incident",
        "V45": "deployment",
        "V46": "secrets",
        "V47": "finance",
        "V48": "learning",
        "V49": "eval",
        "V50": "autonomy",
        "V51": "registry",
        "V52": "deployment",
        "V53": "permission",
        "V54": "telemetry",
        "V55": "incident",
        "V56": "secrets",
        "V57": "finance",
        "V58": "learning",
        "V59": "eval",
        "V60": "autonomy",
        "V61": "registry",
        "V62": "deployment",
        "V63": "secrets",
        "V64": "finance",
        "V65": "learning",
        "V66": "eval",
        "V67": "autonomy",
        "V68": "incident",
        "V69": "catalog",
        "V70": "deployment",
        "V71": "quality",
        "V72": "deployment",
        "V73": "secrets",
        "V74": "incident",
        "V75": "catalog",
        "V76": "telemetry",
        "V77": "autonomy",
        "V78": "eval",
        "V79": "finance",
        "V80": "deployment",
    }.get(stage.upper(), "quality")  # type: ignore[return-value]


def _slug(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "item"


def _probe_production_target(*, url: str, health_url: str = "", timeout: float = 4.0) -> dict[str, Any]:
    target_url = health_url or url
    parsed = urlparse(target_url)
    checks: list[dict[str, Any]] = []
    if not parsed.scheme or not parsed.hostname:
        return {"ok": False, "checks": [{"name": "url", "ok": False, "detail": "Target URL is missing or invalid."}]}

    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        checks.append({"name": "dns", "ok": True, "detail": f"{len(addresses)} address record(s) resolved."})
    except OSError as exc:
        checks.append({"name": "dns", "ok": False, "detail": str(exc)})

    if parsed.scheme == "https":
        try:
            context = ssl.create_default_context()
            with socket.create_connection((parsed.hostname, parsed.port or 443), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=parsed.hostname) as tls:
                    cert = tls.getpeercert()
            checks.append({"name": "tls", "ok": True, "detail": f"TLS certificate issued to {cert.get('subject', 'unknown subject')}."})
        except Exception as exc:
            checks.append({"name": "tls", "ok": False, "detail": str(exc)})

    try:
        request = Request(target_url, method="GET", headers={"user-agent": "HermesProductionSweep/1.0"})
        with urlopen(request, timeout=timeout) as response:
            status_code = int(response.status)
        checks.append({"name": "http", "ok": 200 <= status_code < 500, "detail": f"HTTP {status_code}", "status_code": status_code})
    except HTTPError as exc:
        checks.append({"name": "http", "ok": 200 <= int(exc.code) < 500, "detail": f"HTTP {exc.code}", "status_code": int(exc.code)})
    except (URLError, TimeoutError, OSError) as exc:
        checks.append({"name": "http", "ok": False, "detail": str(exc)})

    if url and health_url and health_url != url:
        checks.append({"name": "snapshot", "ok": True, "detail": "Primary URL and health URL are both declared for follow-up screenshot/snapshot checks."})
    return {"ok": bool(checks) and all(check["ok"] for check in checks), "checks": checks, "url": target_url}


def _hetzner_promotion_command_plan(*, app_dir: str, migration_required: bool, url: str) -> list[list[str]]:
    commands: list[list[str]] = [
        ["git", "status", "--short"],
        ["npm", "run", "build"],
        ["npm", "test"],
    ]
    if migration_required:
        commands.append(["npx", "prisma", "migrate", "deploy"])
    commands.append(["docker", "compose", "up", "-d", "--build"])
    if url:
        commands.append(["curl", "-fsS", url])
    return commands


def _execute_promotion_commands(
    command_plan: Iterable[list[str]],
    *,
    cwd: Path | None,
    command_runner: Callable[[list[str], Path | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    runner = command_runner or _run_command
    for command in command_plan:
        result = runner(command, cwd)
        results.append(_redact_command_result(result))
        if not result.get("ok"):
            return {"ok": False, "results": results}
    return {"ok": True, "results": results}


def _run_command(command: list[str], cwd: Path | None) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=120, check=False)
        return {
            "ok": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"ok": False, "command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def _redact_command_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(result.get("ok")),
        "command": list(result.get("command") or []),
        "returncode": result.get("returncode"),
        "stdout": _redact_lines(str(result.get("stdout") or "")),
        "stderr": _redact_lines(str(result.get("stderr") or "")),
    }


def _redact_lines(value: str) -> str:
    lines = value.splitlines()
    safe_lines = []
    for line in lines[:50]:
        if "=" in line and any(term in line.lower() for term in ("key", "token", "secret", "password")):
            safe_lines.append(line.split("=", 1)[0] + "=***")
        else:
            safe_lines.append(line[:500])
    return "\n".join(safe_lines)


def _parse_gh_name_list(output: str) -> set[str]:
    names: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("name"):
            continue
        names.add(stripped.split()[0])
    return names


def _fetch_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        request = Request(url, method="GET", headers={"user-agent": "HermesOutcomeIngest/1.0"})
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(2_000_000).decode("utf-8")
            body = json.loads(raw)
        return {"ok": True, "url": url, "status_code": int(response.status), "body": body}
    except HTTPError as exc:
        return {"ok": False, "url": url, "status_code": int(exc.code), "error": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"ok": False, "url": url, "status_code": None, "error": str(exc)}


def _redact_fetch_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "ok": result.get("ok"),
        "url": result.get("url"),
        "status_code": result.get("status_code"),
        "error": result.get("error"),
    }


def _evidence_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "subject": row["subject"],
        "state": row["state"],
        "owner": row["owner"],
        "detail": row["detail"],
        "payload": _json(row["payload"], {}),
        "updated_at": row["updated_at"],
    }


def _audit_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "action": row["action"],
        "actor": row["actor"],
        "allowed": bool(row["allowed"]),
        "approval": row["approval"],
        "reason": row["reason"],
        "payload": _json(row["payload"], {}),
        "created_at": row["created_at"],
    }


def _workbench_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "owner": row["owner"],
        "approval": row["approval"],
        "artifacts": _json(row["artifacts"], []),
        "report": row["report"],
        "updated_at": row["updated_at"],
    }


def _json(raw: str, fallback: Any) -> Any:
    try:
        return json.loads(raw or "")
    except Exception:
        return fallback
