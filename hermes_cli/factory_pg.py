"""PostgreSQL-backed Factory progress API for Agent Core DB.

This is the canonical runtime backend for the SitioUno Software Factory. Do not
route production Factory work to SQLite.
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_cli import agent_core_sql as sql
from hermes_cli.factory_catalog import DEFAULT_LANES, FACTORY_AGENTS, VALID_METHODS, slugify

FACTORY_REQUIRED_DOCS = (
    "PRD.md",
    "ADRS.md",
    "METHODOLOGY_PLAN.md",
    "TECHNICAL_BLUEPRINT.md",
    "SPRINT_PLAN.md",
    "TASK_GRAPH.md",
    "TRACKER.md",
    "DOCUMENTATION_INDEX.md",
    "QA_GATES.md",
    "SECURITY_GATES.md",
    "QA_REPORT.md",
    "SECURITY_REVIEW.md",
    "DELIVERY_REPORT.md",
)
CRITICAL_RISK_LEVELS = {"critical", "high"}
RECONCILIATION_TASK_SPECS: dict[str, dict[str, Any]] = {
    "missing_notion_project": {
        "title": "R0 — Reconciliation: create or link Factory Notion PM tracker",
        "phase": "documentation",
        "owner": "factory-reporter",
        "reviewer": "factory-orchestrator",
        "engine": "zeus",
        "priority": 10,
        "acceptance": [
            "Project-specific Notion PM tracker exists, unless Jean explicitly authorized a no-Notion exception for this project.",
            "Factory DB project metadata includes notion_tracker_page_id or notion_tracker_url when Notion is required.",
            "Notion remains human reporting only; Factory DB stays the source of truth.",
        ],
    },
    "missing_project_artifact_dir": {
        "title": "R1 — Reconciliation: restore project-local Factory artifact directory",
        "phase": "documentation",
        "owner": "factory-reporter",
        "reviewer": "factory-orchestrator",
        "engine": "zeus",
        "priority": 20,
        "acceptance": [
            "Project-local artifact directory exists under the project repo and matches project metadata.",
            "Directory ownership and source-of-truth notes are recorded in DOCUMENTATION_INDEX.md or equivalent.",
        ],
    },
    "missing_required_docs": {
        "title": "R2 — Reconciliation: complete required Factory methodology documentation",
        "phase": "documentation",
        "owner": "factory-reporter",
        "reviewer": "factory-orchestrator",
        "engine": "zeus",
        "priority": 30,
        "acceptance": [
            "Required project-local Factory docs exist, unless Jean explicitly authorized a document exception for this project.",
            "PRD, ADRs, sprint plan, task graph, QA/security gates, tracker, and delivery report are reconciled against Factory DB.",
        ],
    },
    "docs_not_indexed": {
        "title": "R2b — Reconciliation: update DOCUMENTATION_INDEX.md",
        "phase": "documentation",
        "owner": "factory-reporter",
        "reviewer": "factory-orchestrator",
        "engine": "zeus",
        "priority": 35,
        "acceptance": [
            "DOCUMENTATION_INDEX.md lists every required project-local Factory artifact path.",
            "The documentation index is generated from real artifact files, not only filename guesses.",
            "Builder context points at DOCUMENTATION_INDEX.md before implementation starts.",
        ],
    },
    "uncommitted_project_artifacts": {
        "title": "R2c — Reconciliation: commit project-local Factory artifacts",
        "phase": "documentation",
        "owner": "factory-reporter",
        "reviewer": "factory-orchestrator",
        "engine": "zeus",
        "priority": 37,
        "acceptance": [
            "Project-local Factory artifacts have a git commit checkpoint, unless Jean explicitly authorized a project-specific exception.",
            "Factory DB metadata or gate evidence records the commit SHA used as the source-of-truth checkpoint.",
            "Untracked or modified methodology docs are not treated as completed delivery evidence.",
        ],
    },
    "missing_task_graph": {
        "title": "R3 — Reconciliation: rebuild canonical Factory task graph",
        "phase": "planning",
        "owner": "implementation-planner",
        "reviewer": "factory-orchestrator",
        "engine": "zeus",
        "priority": 40,
        "acceptance": [
            "Factory DB contains pending implementation/QA/security/delivery tasks that cover the remaining objective.",
            "Task graph is derived from project artifacts and deliverable evidence, not from UI state alone.",
            "Each generated task has owner, reviewer, phase, acceptance criteria, and evidence requirements.",
        ],
    },
    "pending_effective_gates": {
        "title": "R4 — Reconciliation: resolve stale or pending effective Factory gates",
        "phase": "review",
        "owner": "factory-orchestrator",
        "reviewer": "quality-reviewer",
        "engine": "zeus",
        "priority": 50,
        "acceptance": [
            "Every effective pending gate is either passed, failed with rework, or waived with a recorded reason.",
            "Old superseded gate rows do not block the current project status.",
        ],
    },
    "deliverable_unverified": {
        "title": "R5 — Reconciliation: verify deliverable completion and close delivery gate",
        "phase": "delivery",
        "owner": "qa-verifier",
        "reviewer": "factory-orchestrator",
        "engine": "zeus",
        "priority": 60,
        "acceptance": [
            "Deliverable evidence is checked against PRD, task graph, tests/builds, and local artifacts.",
            "Delivery gate is recorded as passed, failed with rework, or waived with a human/admin reason.",
            "Project is not marked completed from prose alone.",
        ],
    },
}
TERMINAL_TASK_STATUSES = {"done", "verified", "cancelled", "superseded"}
IN_FLIGHT_TASK_STATUSES = {"claimed", "running", "in_progress", "review_ready", "review_running", "qa_ready"}
ACTIVE_TASK_STATUSES = IN_FLIGHT_TASK_STATUSES | {"rework"}
DISPATCHABLE_PROJECT_STATUSES = {"active", "planned", "intake", "blocked"}
TERMINAL_GATE_STATUSES = {"passed", "failed", "waived"}
BLOCKER_ACTION_CATEGORIES = {"auto_resolvable", "technical_rework", "human_question_required", "stale_orphan_state"}
DELIVERY_HOLD_STATUS = "delivery_hold"
BLOCKED_ALERT_DEFAULT_MINUTES = 60
CLAIMED_NULL_ALERT_ROUNDS = 3
_BLOCKER_TECHNICAL_KEYWORDS = (
    "error", "exception", "traceback", "crash", "timeout", "fail", "failed", "failing", "import error",
    "syntax error", "typeerror", "attributeerror", "test failed", "pytest", "bug", "regression", "rework",
)
_BLOCKER_HUMAN_KEYWORDS = (
    "human decision", "decisión humana", "jean debe", "jean needs", "owner decision", "business decision",
    "legal approval", "customer approval", "api key", "token", "password", "credential", "secret", "2fa", "mfa",
    "access denied", "permission denied", "403", "401", "billing", "payment method",
)
_BLOCKER_AUTO_KEYWORDS = (
    "run pytest", "run tests", "correr pytest", "correr tests", "aprobación para correr", "approval to run",
    "implementación o cierre administrativo", "administrative closure", "gate passed", "already passed",
)
_SCHEMA_READY = False


def _user() -> str:
    return sql.runtime_env().get("FACTORY_DB_RUNTIME_USER", "factory_runtime")


def _admin_user() -> str:
    return sql.runtime_env().get("AGENT_DB_ADMIN_USER", "agent_admin")


def _q(v: Any) -> str:
    return sql.quote_literal(v)


def _j(v: Any) -> str:
    return sql.quote_jsonb(v)


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    metadata = data.get("metadata")
    if metadata is None:
        data["metadata"] = {}
    if "owner_profile" in data and "owner_agent_id" not in data:
        data["owner_agent_id"] = data.get("owner_profile")
    if "reviewer_profile" in data and "reviewer_agent_id" not in data:
        data["reviewer_agent_id"] = data.get("reviewer_profile")
    if "timestamp" in data and "created_at" not in data:
        data["created_at"] = data.get("timestamp")
    if "profile_name" in data and not data.get("display_name"):
        data["display_name"] = data.get("profile_name")
    if data.get("status") == "active" and "active" not in data:
        data["active"] = 1
    return data


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize(row) for row in rows]


def available() -> bool:
    if not sql.enabled():
        return False
    try:
        sql.psql("SELECT 1;", user=_user())
        return True
    except Exception:
        return False


def ensure_runtime_schema() -> None:
    """Ensure orchestration runtime tables/columns exist.

    Migrations normally create these objects. This guard keeps the autonomous
    cron path safe after deploy/restart when a runtime DB is slightly behind.
    """

    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    ddl = Path(__file__).resolve().parents[1] / "db" / "modules" / "factory" / "000003_orchestration_runtime.sql"
    if ddl.exists():
        sql.psql(ddl.read_text(encoding="utf-8"), user=_admin_user())
    _SCHEMA_READY = True


def seed_agents() -> None:
    ensure_runtime_schema()
    values = []
    for agent_id, display_name, role, engine, toolsets, skills, greenlights in FACTORY_AGENTS:
        values.append(
            f"({_q(agent_id)}, {_q(agent_id)}, {_q(display_name)}, {_q(role)}, {_q(engine)}, {_j({'toolsets': toolsets, 'skills': skills, 'greenlight_required': greenlights})})"
        )
    active_ids = ", ".join(_q(agent[0]) for agent in FACTORY_AGENTS)
    sql.psql(
        """
        INSERT INTO factory.agents (agent_id, profile_name, display_name, role, preferred_engine, metadata)
        VALUES """ + ",\n".join(values) + """
        ON CONFLICT (agent_id) DO UPDATE SET
          profile_name = EXCLUDED.profile_name,
          display_name = EXCLUDED.display_name,
          role = EXCLUDED.role,
          preferred_engine = EXCLUDED.preferred_engine,
          metadata = EXCLUDED.metadata,
          status = 'active',
          updated_at = now();
        UPDATE factory.agents
        SET status = 'retired', updated_at = now()
        WHERE agent_id IN ('olga-openhands')
          AND agent_id NOT IN (""" + active_ids + """)
          AND status = 'active';
        """,
        user=_user(),
    )


def list_agents() -> list[dict[str, Any]]:
    seed_agents()
    return _normalize_rows(
        sql.rows(
            "SELECT agent_id, profile_name, display_name, role, status, preferred_engine, metadata, created_at, updated_at FROM factory.agents WHERE status = 'active' ORDER BY agent_id",
            user=_user(),
        )
    )


def create_project(name: str, *, project_id: Optional[str] = None, repo_path: Optional[str] = None, repo_remote: Optional[str] = None, base_branch: Optional[str] = None, human_owner: Optional[str] = None, summary: Optional[str] = None, risk_level: str = "medium", autonomy_level: int = 3, methodology: str = "hybrid", create_default_lanes: bool = True, metadata: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
    seed_agents()
    pid = project_id or slugify(name)
    meta = {"source_of_truth": "agent_core_postgres", "artifact_dir": f"factory/projects/{pid}", **(metadata or {})}
    sql.psql(f"""
        INSERT INTO factory.projects (project_id, name, repo_path, repo_remote, base_branch, status, autonomy_level, methodology, risk_level, human_owner, summary, metadata, started_at, updated_at)
        VALUES ({_q(pid)}, {_q(name)}, {_q(repo_path)}, {_q(repo_remote)}, {_q(base_branch)}, 'intake', {int(autonomy_level)}, {_q(methodology)}, {_q(risk_level)}, {_q(human_owner)}, {_q(summary)}, {_j(meta)}, now(), now())
        ON CONFLICT (project_id) DO UPDATE SET
          name=EXCLUDED.name, repo_path=EXCLUDED.repo_path, repo_remote=EXCLUDED.repo_remote, base_branch=EXCLUDED.base_branch,
          autonomy_level=EXCLUDED.autonomy_level, methodology=EXCLUDED.methodology, risk_level=EXCLUDED.risk_level,
          human_owner=EXCLUDED.human_owner, summary=EXCLUDED.summary,
          metadata=factory.projects.metadata || EXCLUDED.metadata,
          updated_at=now();
        INSERT INTO factory.events(project_id, actor, event_type, message, metadata)
        VALUES ({_q(pid)}, 'factory-orchestrator', 'project_created', {_q(f'Factory project {pid} initialized')}, {_j({'methodology': methodology, 'source_of_truth': 'agent_core_postgres'})});
    """, user=_user())
    lanes = []
    if create_default_lanes:
        for _suffix, lane_name, method in DEFAULT_LANES:
            lanes.append(create_lane(pid, lane_name, method))
    return {"project_id": pid, "lanes": lanes}


def create_lane(project_id: str, name: str, methodology: str, *, lane_id: Optional[str] = None, kanban_board: Optional[str] = None, branch: Optional[str] = None, worktree_path: Optional[str] = None, metadata: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
    ensure_runtime_schema()
    if methodology not in VALID_METHODS:
        raise ValueError(f"unknown methodology {methodology!r}; expected one of {sorted(VALID_METHODS)}")
    suffix = "zeus" if methodology == "zeus_native" else "bmad" if methodology == "bmad_hybrid" else slugify(methodology)
    lid = lane_id or f"{project_id}-{suffix}"
    branch_value = branch or f"factory/{project_id}/{suffix}"
    meta = {"execution_surface": "factory", **(metadata or {})}
    if kanban_board:
        # Explicitly recorded as an exception; Factory DB remains canonical.
        meta["execution_surface"] = "kanban_bridge"
        meta["kanban_board"] = kanban_board
    sql.psql(f"""
        INSERT INTO factory.lanes (lane_id, project_id, name, methodology, branch, worktree_path, status, metadata, created_at, updated_at)
        VALUES ({_q(lid)}, {_q(project_id)}, {_q(name)}, {_q(methodology)}, {_q(branch_value)}, {_q(worktree_path)}, 'planned', {_j(meta)}, now(), now())
        ON CONFLICT (lane_id) DO UPDATE SET name=EXCLUDED.name, methodology=EXCLUDED.methodology, branch=EXCLUDED.branch, worktree_path=EXCLUDED.worktree_path, metadata=factory.lanes.metadata || EXCLUDED.metadata, updated_at=now();
        INSERT INTO factory.events(project_id, lane_id, actor, event_type, message, metadata)
        VALUES ({_q(project_id)}, {_q(lid)}, 'factory-orchestrator', 'lane_created', {_q(f'Lane {lid} initialized')}, {_j({'methodology': methodology, 'execution_surface': meta['execution_surface'], 'branch': branch_value})});
    """, user=_user())
    result = {"lane_id": lid, "project_id": project_id, "methodology": methodology, "execution_surface": meta["execution_surface"], "branch": branch_value}
    if kanban_board:
        result["kanban_board"] = kanban_board
    return result


def _next_increment_key(project_id: str, priority: int, title: str, metadata: Optional[dict[str, Any]]) -> str:
    if metadata and metadata.get("increment_key"):
        return str(metadata["increment_key"])
    return f"inc-{int(priority):03d}-{slugify(title)[:32]}"


def create_task(project_id: str, title: str, *, lane_id: Optional[str] = None, kanban_id: Optional[str] = None, description: Optional[str] = None, phase: str = "planning", status: str = "todo", owner_agent_id: Optional[str] = None, reviewer_agent_id: Optional[str] = None, engine: str = "zeus", priority: int = 100, acceptance_criteria: Optional[list[str]] = None, dependencies: Optional[list[str]] = None, metadata: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
    ensure_runtime_schema()
    base = f"{project_id}-{slugify(title)[:40]}"
    existing = sql.rows(f"SELECT task_id FROM factory.tasks WHERE task_id LIKE {_q(base + '%')} ORDER BY task_id", user=_user())
    taken = {r['task_id'] for r in existing}
    task_id = base
    n = 2
    while task_id in taken:
        task_id = f"{base}-{n}"
        n += 1
    increment_key = _next_increment_key(project_id, priority, title, metadata)
    meta = {"source": "factory_task_create", **(metadata or {})}
    sql.psql(f"""
        INSERT INTO factory.tasks (task_id, project_id, lane_id, kanban_id, title, description, phase, status, owner_profile, reviewer_profile, engine, priority, dependencies, acceptance_criteria, evidence_required, evidence_status, risk_level, metadata, increment_key, increment_order, created_at, updated_at)
        VALUES ({_q(task_id)}, {_q(project_id)}, {_q(lane_id)}, {_q(kanban_id)}, {_q(title)}, {_q(description)}, {_q(phase)}, {_q(status)}, {_q(owner_agent_id)}, {_q(reviewer_agent_id)}, {_q(engine)}, {int(priority)}, {_j(dependencies or [])}, {_j(acceptance_criteria or [])}, true, 'missing', 'medium', {_j(meta)}, {_q(increment_key)}, {int(priority)}, now(), now());
        INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
        VALUES ({_q(project_id)}, {_q(lane_id)}, {_q(task_id)}, 'factory-orchestrator', 'task_created', {_q(f'Task {task_id} created')}, {_j({'engine': engine, 'owner': owner_agent_id, 'kanban_id': kanban_id, 'increment_key': increment_key})});
    """, user=_user())
    return {"task_id": task_id, "project_id": project_id, "lane_id": lane_id}


def _project(project_id: str) -> dict[str, Any] | None:
    row = sql.one(f"SELECT * FROM factory.projects WHERE project_id={_q(project_id)}", user=_user())
    return _normalize(row) if row else None


def _tasks(project_id: str) -> list[dict[str, Any]]:
    return _normalize_rows(sql.rows(f"SELECT * FROM factory.tasks WHERE project_id={_q(project_id)} ORDER BY priority, created_at", user=_user()))


def _latest_gate_rows(project_id: str) -> list[dict[str, Any]]:
    """Latest effective stage gate per gate_type for a project.

    Gate rows are an audit log. Older ``pending`` or ``failed`` rows must not
    keep a project blocked after a later authoritative stage decision exists.
    """

    return _normalize_rows(sql.rows(
        f"""
        SELECT DISTINCT ON (gate_type) *
        FROM factory.gates
        WHERE project_id={_q(project_id)}
        ORDER BY gate_type, timestamp DESC, gate_id DESC
        """,
        user=_user(),
    ))


def _active_pending_gates(project_id: str) -> list[dict[str, Any]]:
    return [gate for gate in _latest_gate_rows(project_id) if str(gate.get("status") or "") == "pending"]


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _waiver_explicitly_authorized(metadata: dict[str, Any], waiver_key: str) -> bool:
    """Return True only for a project-specific waiver Jean authorized.

    Suppressor booleans such as ``notion_waived`` and
    ``required_docs_waived`` explain historical drift, but they must not mute
    canonical Factory readiness unless an explicit human authorization is
    attached to that project.
    """

    if not metadata.get(waiver_key):
        return False
    authorizer = (
        metadata.get(f"{waiver_key}_authorized_by")
        or metadata.get("waiver_authorized_by")
        or metadata.get("waivers_authorized_by")
    )
    reason = (
        metadata.get(f"{waiver_key}_reason")
        or metadata.get("waiver_reason")
        or metadata.get("waiver_authorization_reason")
    )
    return bool(authorizer and reason)


def _any_explicit_waiver(metadata: dict[str, Any], *waiver_keys: str) -> bool:
    return any(_waiver_explicitly_authorized(metadata, key) for key in waiver_keys)


def _has_unauthorized_completion_waivers(metadata: dict[str, Any]) -> bool:
    waiver_keys = ("notion_waived", "tracker_waived", "required_docs_waived")
    return any(metadata.get(key) and not _waiver_explicitly_authorized(metadata, key) for key in waiver_keys)


def _required_docs_explicitly_waived(metadata: dict[str, Any]) -> bool:
    if _waiver_explicitly_authorized(metadata, "required_docs_waived"):
        return True
    waivers = metadata.get("required_doc_waivers")
    if not waivers:
        return False
    authorizer = metadata.get("required_doc_waivers_authorized_by") or metadata.get("waiver_authorized_by")
    reason = metadata.get("required_doc_waivers_reason") or metadata.get("waiver_reason")
    return bool(authorizer and reason)


def _all_open_work_is_reconciliation(tasks: list[dict[str, Any]]) -> bool:
    open_tasks = [task for task in tasks if str(task.get("status") or "") not in TERMINAL_TASK_STATUSES]
    return bool(open_tasks) and all(_is_reconciliation_task(task) for task in open_tasks)


def _all_blocked_work_is_documentation_hold(tasks: list[dict[str, Any]]) -> bool:
    blocked_statuses = {"blocked", "review_pending_human"}
    blocking_tasks = [task for task in tasks if str(task.get("status") or "") in blocked_statuses]
    if not blocking_tasks:
        return False
    documentation_phases = {"documentation", "docs", "pm", "reporting"}
    for task in blocking_tasks:
        phase = str(task.get("phase") or "").lower()
        if phase not in documentation_phases and not _is_reconciliation_task(task):
            return False
    return True


def _project_artifact_dir(project: dict[str, Any]) -> tuple[Path | None, str]:
    project_id = str(project.get("project_id") or "")
    metadata = _metadata(project)
    artifact_dir = str(metadata.get("artifact_dir") or f"factory/projects/{project_id}")
    repo_path = str(project.get("repo_path") or "").strip()
    if not repo_path:
        return None, artifact_dir
    return Path(repo_path).expanduser() / artifact_dir, artifact_dir


def _docs_missing_from_documentation_index(factory_dir: Path, required_docs: tuple[str, ...] = FACTORY_REQUIRED_DOCS) -> list[str]:
    """Return required docs not referenced by the project documentation index.

    The repo-first Factory contract treats ``DOCUMENTATION_INDEX.md`` as the
    builder entry point. A file existing on disk is not enough; builders and
    reviewers need the index to enumerate every canonical artifact path.
    """

    index_path = factory_dir / "DOCUMENTATION_INDEX.md"
    try:
        index_text = index_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return list(required_docs)
    return [name for name in required_docs if name not in index_text]


def _repo_commit_explicitly_waived(metadata: dict[str, Any]) -> bool:
    return _any_explicit_waiver(
        metadata,
        "repo_commit_waived",
        "commit_checkpoint_waived",
        "uncommitted_artifacts_waived",
    )


def _uncommitted_project_artifacts(repo_path: Path, artifact_dir: str) -> list[str]:
    """Return git porcelain rows for uncommitted project-local artifacts.

    If the repo is unavailable or not a git worktree, return an empty list so the
    reconciler does not turn filesystem errors into false blockers. Git-backed
    Factory repos, however, must leave no modified/untracked project artifacts
    before critical readiness/delivery closure.
    """

    repo = Path(repo_path).expanduser()
    if not repo.exists():
        return []
    try:
        probe = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if probe.returncode != 0 or probe.stdout.strip() != "true":
            return []
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", artifact_dir],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return []
    if status.returncode != 0:
        return []
    return [line for line in status.stdout.splitlines() if line.strip()]


def _is_reconciliation_task(task: dict[str, Any]) -> bool:
    metadata = _metadata(task)
    if metadata.get("factory_reconciliation_task") or metadata.get("reconciliation_anomaly"):
        return True
    text = "\n".join(str(task.get(key) or "") for key in ("task_id", "title", "description", "phase")).lower()
    return "reconciliation" in text or "reconciliación" in text


def _task_covers_reconciliation_anomaly(task: dict[str, Any], code: str) -> bool:
    if str(task.get("status") or "") in TERMINAL_TASK_STATUSES:
        return False
    metadata = _metadata(task)
    if metadata.get("reconciliation_anomaly") == code:
        return True
    text = "\n".join(str(task.get(key) or "") for key in ("task_id", "title", "description", "phase")).lower()
    if code == "missing_notion_project":
        return "notion" in text or "pm tracker" in text or "tracker" in text
    if code == "missing_project_artifact_dir":
        return "artifact" in text or "documentation" in text or "documentación" in text
    if code == "missing_required_docs":
        return any(term in text for term in ("documentation", "documentación", "docs", "tracker", "delivery report"))
    if code == "docs_not_indexed":
        return "documentation_index" in text or "documentation index" in text or "índice" in text or "indexed" in text
    if code == "uncommitted_project_artifacts":
        return "commit" in text or "uncommitted" in text or "git" in text or "checkpoint" in text
    if code == "missing_task_graph":
        return "task graph" in text or "task-graph" in text or "canonical task" in text or "task graph recovery" in text
    if code == "pending_effective_gates":
        return str(task.get("phase") or "").lower() in {"review", "qa", "security"} or "gate" in text
    if code == "deliverable_unverified":
        return str(task.get("phase") or "").lower() in {"delivery", "qa", "security"} or any(term in text for term in ("delivery", "deliverable", "qa", "security review"))
    return False


def _latest_gate_statuses(gates: list[dict[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for gate in gates:
        gate_type = str(gate.get("gate_type") or "").strip()
        if gate_type and gate_type not in statuses:
            statuses[gate_type] = str(gate.get("status") or "").strip()
    return statuses


def reconciliation_findings(project: dict[str, Any], tasks: list[dict[str, Any]], pending_gates: list[dict[str, Any]], gates: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    """Return deterministic project-completeness anomalies for Factory reconciliation.

    This is intentionally objective: it inspects Factory DB state, project-local
    artifact paths, Notion tracker metadata, gates, and task graph shape. It does
    not ask a worker to decide whether an incomplete project is implementation or
    administrative closure; it creates recovery work that can be executed if the
    project is resumed.
    """

    metadata = _metadata(project)
    if (
        str(project.get("status") or "") in {"completed", "accepted"}
        and not metadata.get("force_reconcile_completed")
        and not _has_unauthorized_completion_waivers(metadata)
    ):
        return []

    findings: list[dict[str, Any]] = []

    def add(code: str, message: str, **metadata: Any) -> None:
        if code not in RECONCILIATION_TASK_SPECS:
            raise ValueError(f"unknown reconciliation anomaly code: {code}")
        findings.append({"code": code, "message": message, "metadata": metadata})

    factory_dir, artifact_dir = _project_artifact_dir(project)
    if not (
        metadata.get("notion_tracker_url")
        or metadata.get("notion_tracker_page_id")
        or _any_explicit_waiver(metadata, "notion_waived", "tracker_waived")
    ):
        add("missing_notion_project", "Missing project-specific Notion PM tracker metadata")

    if factory_dir is None or not factory_dir.is_dir():
        add("missing_project_artifact_dir", f"Missing project-local Factory artifact directory: {artifact_dir}", artifact_dir=artifact_dir)
        missing_docs = list(FACTORY_REQUIRED_DOCS)
    else:
        missing_docs = [name for name in FACTORY_REQUIRED_DOCS if not (factory_dir / name).is_file()]
    if missing_docs and not _required_docs_explicitly_waived(metadata):
        add("missing_required_docs", "Missing required Factory methodology documents", missing_docs=missing_docs, artifact_dir=artifact_dir)
    elif factory_dir is not None and factory_dir.is_dir():
        missing_from_index = _docs_missing_from_documentation_index(factory_dir)
        if missing_from_index and not _required_docs_explicitly_waived(metadata):
            add(
                "docs_not_indexed",
                "Required Factory methodology documents are missing from DOCUMENTATION_INDEX.md",
                missing_from_index=missing_from_index,
                artifact_dir=artifact_dir,
            )

    repo_path = str(project.get("repo_path") or "").strip()
    if repo_path and factory_dir is not None and factory_dir.is_dir() and not _repo_commit_explicitly_waived(metadata):
        uncommitted_paths = _uncommitted_project_artifacts(Path(repo_path), artifact_dir)
        if uncommitted_paths:
            add(
                "uncommitted_project_artifacts",
                "Project-local Factory artifacts have uncommitted git changes",
                uncommitted_paths=uncommitted_paths,
                artifact_dir=artifact_dir,
            )

    non_reconciliation_tasks = [task for task in tasks if not _is_reconciliation_task(task)]
    if not non_reconciliation_tasks:
        add("missing_task_graph", "Factory DB has no canonical non-reconciliation task graph for this project")

    if pending_gates:
        add(
            "pending_effective_gates",
            "Project has effective pending gates that must be resolved",
            pending_gates=[gate.get("gate_type") for gate in pending_gates],
        )

    latest_gate_status = _latest_gate_statuses(gates or [])
    delivery_passed = latest_gate_status.get("delivery") == "passed" or latest_gate_status.get("critical_readiness") == "passed"
    has_delivery_coverage = any(_task_covers_reconciliation_anomaly(task, "deliverable_unverified") for task in tasks)
    if not delivery_passed and not has_delivery_coverage:
        add("deliverable_unverified", "No passed delivery/critical readiness gate or pending delivery verification task exists")

    return findings


def ensure_reconciliation_tasks(project: dict[str, Any], findings: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ensure_runtime_schema()
    project_id = str(project.get("project_id") or "")
    created: list[dict[str, Any]] = []
    statements: list[str] = []
    terminal = ",".join(_q(status) for status in TERMINAL_TASK_STATUSES)
    for finding in findings:
        code = str(finding.get("code") or "")
        if not code or any(_task_covers_reconciliation_anomaly(task, code) for task in tasks):
            continue
        spec = RECONCILIATION_TASK_SPECS[code]
        task_id = f"{project_id}-reconcile-{code.replace('_', '-')}"
        metadata = {
            "source": "factory_reconciler",
            "factory_reconciliation_task": True,
            "reconciliation_anomaly": code,
            "finding": finding,
        }
        description = (
            f"Deterministic reconciliation task generated because: {finding.get('message')}.\n\n"
            "Do not close the project manually from UI state. Inspect Factory DB, project-local artifacts, Notion metadata, gates, and deliverable evidence; then record gates/evidence canonically."
        )
        reopen_note = f"\n\n[factory-reconciler] Reopened because the canonical anomaly recurred: {code}."
        statements.append(
            f"""
            INSERT INTO factory.tasks (
              task_id, project_id, lane_id, title, description, phase, status,
              owner_profile, reviewer_profile, engine, priority, dependencies,
              acceptance_criteria, evidence_required, evidence_status, risk_level,
              metadata, increment_key, increment_order, created_at, updated_at
            )
            VALUES (
              {_q(task_id)}, {_q(project_id)}, (SELECT lane_id FROM factory.lanes WHERE project_id={_q(project_id)} ORDER BY created_at, lane_id LIMIT 1),
              {_q(spec['title'])}, {_q(description)}, {_q(spec['phase'])}, 'todo',
              {_q(spec['owner'])}, {_q(spec['reviewer'])}, {_q(spec['engine'])}, {int(spec['priority'])}, '[]'::jsonb,
              {_j(spec['acceptance'])}, true, 'missing', {_q(str(project.get('risk_level') or 'medium'))},
              {_j(metadata)}, {_q('reconcile-' + code)}, {int(spec['priority'])}, now(), now()
            )
            ON CONFLICT (task_id) DO UPDATE SET
              status=CASE WHEN factory.tasks.status IN ({terminal}) THEN 'todo' ELSE factory.tasks.status END,
              evidence_status=CASE WHEN factory.tasks.status IN ({terminal}) THEN 'missing' ELSE factory.tasks.evidence_status END,
              result_summary=CASE
                WHEN factory.tasks.status IN ({terminal}) THEN COALESCE(factory.tasks.result_summary, '') || {_q(reopen_note)}
                ELSE factory.tasks.result_summary
              END,
              description=EXCLUDED.description,
              acceptance_criteria=EXCLUDED.acceptance_criteria,
              metadata=factory.tasks.metadata || EXCLUDED.metadata || {_j({'reopened_by': 'factory_reconciler', 'reopen_reason': 'canonical_anomaly_recurred'})},
              updated_at=now();
            INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
            VALUES ({_q(project_id)}, (SELECT lane_id FROM factory.lanes WHERE project_id={_q(project_id)} ORDER BY created_at, lane_id LIMIT 1), {_q(task_id)}, 'factory-reconciler', 'reconciliation_task_ensured', {_q(f'Reconciliation task ensured for anomaly {code}')}, {_j({'task_id': task_id, 'reconciliation_anomaly': code})});
            """
        )
        created.append({"task_id": task_id, "code": code})
    if statements:
        sql.psql("\n".join(statements), user=_user())
    return created


def critical_readiness_findings(project_id: str) -> list[str]:
    project = _project(project_id)
    if not project:
        return [f"project {project_id} is not visible in Agent Core Postgres"]
    risk = str(project.get("risk_level") or "").lower()
    if risk not in CRITICAL_RISK_LEVELS:
        return []

    findings: list[str] = []
    raw_metadata = project.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    repo_path = str(project.get("repo_path") or "").strip()
    artifact_dir = metadata.get("artifact_dir") or f"factory/projects/{project_id}"
    factory_dir = Path(repo_path).expanduser() / str(artifact_dir) if repo_path else None

    tracker_ok = bool(
        metadata.get("notion_tracker_url")
        or metadata.get("notion_tracker_page_id")
        or _any_explicit_waiver(metadata, "notion_waived", "tracker_waived")
    )
    if not tracker_ok:
        findings.append("missing project-specific Notion PM tracker metadata")

    if not factory_dir or not factory_dir.is_dir():
        findings.append(f"missing project-local factory documentation directory: {artifact_dir}")
    else:
        missing_docs = [name for name in FACTORY_REQUIRED_DOCS if not (factory_dir / name).is_file()]
        if missing_docs and not _required_docs_explicitly_waived(metadata):
            findings.append("missing minimum docs: " + ", ".join(missing_docs))
        elif not _required_docs_explicitly_waived(metadata):
            missing_from_index = _docs_missing_from_documentation_index(factory_dir)
            if missing_from_index:
                findings.append("documentation index missing required docs: " + ", ".join(missing_from_index))
        if repo_path and not _repo_commit_explicitly_waived(metadata):
            uncommitted_paths = _uncommitted_project_artifacts(Path(repo_path), str(artifact_dir))
            if uncommitted_paths:
                findings.append("uncommitted project-local factory artifacts: " + ", ".join(uncommitted_paths))

    self_approved = sql.rows(
        f"""
        SELECT task_id, title, owner_profile, reviewer_profile
        FROM factory.tasks
        WHERE project_id={_q(project_id)}
          AND owner_profile IS NOT NULL
          AND owner_profile = reviewer_profile
        ORDER BY task_id
        """,
        user=_user(),
    )
    if self_approved:
        findings.append(f"{len(self_approved)} task(s) have no independent reviewer")
    if metadata.get("dashboard_visible") is False:
        findings.append("project is explicitly hidden from dashboard")
    return findings


def record_gate(project_id: str, gate_type: str, status: str, *, lane_id: Optional[str] = None, task_id: Optional[str] = None, reviewer: Optional[str] = None, evidence: Optional[dict[str, Any]] = None, notes: Optional[str] = None, **_: Any) -> dict[str, Any]:
    ensure_runtime_schema()
    gate = str(gate_type or "").strip()
    state = str(status or "").strip()
    if gate in {"delivery", "critical_readiness"} and state == "passed":
        blockers = critical_readiness_findings(project_id)
        if blockers:
            raise ValueError("critical readiness gate blocked: " + "; ".join(blockers))

    row = sql.statement_one(f"""
      INSERT INTO factory.gates (project_id, lane_id, task_id, gate_type, status, reviewer, evidence, notes, timestamp)
      VALUES ({_q(project_id)}, {_q(lane_id)}, {_q(task_id)}, {_q(gate)}, {_q(state)}, {_q(reviewer)}, {_j(evidence or {})}, {_q(notes)}, now())
      RETURNING gate_id, project_id, status, timestamp
    """, user=_user())
    gate_id = row["gate_id"] if row else None
    if gate_id and state in TERMINAL_GATE_STATUSES:
        sql.psql(
            f"""
            UPDATE factory.gates
            SET status='superseded',
                evidence = evidence || {_j({'superseded_by_gate_id': gate_id, 'superseded_by_status': state})}
            WHERE project_id={_q(project_id)}
              AND gate_type={_q(gate)}
              AND status='pending'
              AND gate_id <> {int(gate_id)}
              AND timestamp <= (SELECT timestamp FROM factory.gates WHERE gate_id={int(gate_id)});
            """,
            user=_user(),
        )
    sql.psql(f"INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata) VALUES ({_q(project_id)}, {_q(lane_id)}, {_q(task_id)}, {_q(reviewer or 'factory-orchestrator')}, {_q('gate_' + state)}, {_q(f'{gate} gate {state}')}, {_j({'gate_id': gate_id})});", user=_user())
    reconcile_project(project_id)
    return {"gate_id": gate_id, "project_id": project_id, "status": state}


def update_project_metadata(project_id: str, metadata: dict[str, Any]) -> None:
    ensure_runtime_schema()
    sql.psql(
        f"UPDATE factory.projects SET metadata = metadata || {_j(metadata)}, updated_at = now() WHERE project_id={_q(project_id)};",
        user=_user(),
    )


def _status_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _has_active_increment(tasks: list[dict[str, Any]]) -> bool:
    return any(str(t.get("status") or "") in ACTIVE_TASK_STATUSES for t in tasks)


def _has_in_flight_increment(tasks: list[dict[str, Any]]) -> bool:
    """Return True when a task has a worker/reviewer currently in flight.

    A task in ``rework`` is an active increment for project status and for
    blocking later increments, but it is also runnable: the orchestrator must be
    able to hand it back to the owner. Treating rework as in-flight makes it an
    absorbing state with no autonomous recovery path.
    """

    return any(str(t.get("status") or "") in IN_FLIGHT_TASK_STATUSES for t in tasks)


def _next_runnable_task(project_id: str) -> dict[str, Any] | None:
    tasks = _tasks(project_id)
    if _has_active_increment(tasks):
        return None
    terminal = ",".join(_q(s) for s in TERMINAL_TASK_STATUSES)
    candidates = sql.rows(
        f"""
        SELECT * FROM factory.tasks
        WHERE project_id={_q(project_id)}
          AND status IN ('todo', 'ready')
          AND NOT EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(factory.tasks.dependencies) dep
            WHERE dep NOT IN (SELECT task_id FROM factory.tasks WHERE project_id={_q(project_id)} AND status IN ({terminal}))
          )
        ORDER BY priority, created_at
        LIMIT 1
        """,
        user=_user(),
    )
    return _normalize(candidates[0]) if candidates else None


def reconcile_project(project_id: str) -> dict[str, Any]:
    ensure_runtime_schema()
    tasks = _tasks(project_id)
    project = _project(project_id)
    if not project:
        raise ValueError(f"Factory project not found: {project_id}")
    metadata = _metadata(project)
    pending_gates = _active_pending_gates(project_id)
    latest_gates = _latest_gate_rows(project_id)
    findings = reconciliation_findings(project, tasks, pending_gates, latest_gates)
    created_reconciliation_tasks = ensure_reconciliation_tasks(project, findings, tasks) if findings else []
    if created_reconciliation_tasks:
        tasks = _tasks(project_id)
    counts = _status_counts(tasks)
    active_runs = sql.rows(f"SELECT run_id FROM factory.task_runs WHERE project_id={_q(project_id)} AND status IN ('queued','running')", user=_user())
    if str(project.get("status")) == "paused" or metadata.get("autonomous_enabled") is False and project.get("paused_at"):
        new_status = "paused"
    elif any(status in counts for status in ("blocked", "review_pending_human")):
        new_status = DELIVERY_HOLD_STATUS if _all_blocked_work_is_documentation_hold(tasks) else "blocked"
    elif active_runs or any(status in counts for status in ACTIVE_TASK_STATUSES):
        new_status = "active"
    elif tasks and not any(str(t.get("status") or "") not in TERMINAL_TASK_STATUSES for t in tasks) and not pending_gates and not findings:
        new_status = DELIVERY_HOLD_STATUS if metadata.get("closure_pending_human_ui") else "completed"
    elif findings and _all_open_work_is_reconciliation(tasks) and not active_runs:
        # Documentation/PM reconciliation must keep the project visibly open,
        # but it is not a technical blocker and should not present as a blocked
        # engineering increment. Jean can close it explicitly from the UI once
        # the document side-effect is generated/reconciled.
        new_status = DELIVERY_HOLD_STATUS
    elif findings:
        # Incomplete projects should not remain "active" solely because the
        # autonomous flag is true. Planned + autonomous_enabled=true is still
        # claimable by the orchestrator tick, but the UI no longer shows an
        # active project with no canonical recovery work.
        new_status = "planned"
    elif metadata.get("autonomous_enabled"):
        new_status = "active"
    elif tasks:
        new_status = "planned" if str(project.get("status") or "") not in {"active", "blocked"} else str(project.get("status"))
    else:
        new_status = "intake"
    finding_codes = [str(finding.get("code")) for finding in findings]
    sql.psql(
        f"""
        UPDATE factory.projects
        SET status={_q(new_status)},
            metadata = metadata || {_j({'reconciliation_required': bool(findings), 'reconciliation_anomalies': finding_codes})},
            last_reconciled_at=now(), updated_at=now()
        WHERE project_id={_q(project_id)};
        UPDATE factory.lanes
        SET status = CASE WHEN { _q(new_status) } IN ('active','completed','blocked','paused','planned','intake','delivery_hold') THEN { _q(new_status) } ELSE status END,
            updated_at=now()
        WHERE project_id={_q(project_id)};
        INSERT INTO factory.events(project_id, actor, event_type, message, metadata)
        VALUES ({_q(project_id)}, 'factory-reconciler', 'project_reconciled', {_q(f'Project reconciled as {new_status}')}, {_j({'task_counts': counts, 'pending_gates': len(pending_gates), 'active_runs': len(active_runs), 'anomalies': finding_codes, 'reconciliation_tasks_created': created_reconciliation_tasks})});
        """,
        user=_user(),
    )
    return {
        "project_id": project_id,
        "status": new_status,
        "task_counts": counts,
        "pending_gates": len(pending_gates),
        "active_runs": len(active_runs),
        "anomalies": finding_codes,
        "reconciliation_tasks_created": len(created_reconciliation_tasks),
    }


def resume_project(project_id: str) -> dict[str, Any]:
    ensure_runtime_schema()
    sql.psql(
        f"""
        UPDATE factory.projects
        SET status='active', autonomous_enabled=true, paused_at=NULL,
            metadata = metadata || {_j({'autonomous_enabled': True, 'autonomy_mode': 'incremental_single_active'})},
            updated_at=now()
        WHERE project_id={_q(project_id)};
        UPDATE factory.lanes SET status='active', updated_at=now() WHERE project_id={_q(project_id)} AND status IN ('planned','paused','intake');
        INSERT INTO factory.events(project_id, actor, event_type, message, metadata)
        VALUES ({_q(project_id)}, 'factory-orchestrator', 'autonomous_resume', 'Autonomous Factory execution resumed', {_j({'single_active_increment': True})});
        """,
        user=_user(),
    )
    next_task = _next_runnable_task(project_id)
    if next_task:
        sql.psql(
            f"UPDATE factory.tasks SET status='ready', updated_at=now() WHERE task_id={_q(next_task['task_id'])} AND status='todo';",
            user=_user(),
        )
    result = reconcile_project(project_id)
    result["next_task_id"] = next_task.get("task_id") if next_task else None
    return result


def pause_project(project_id: str) -> dict[str, Any]:
    ensure_runtime_schema()
    sql.psql(
        f"""
        UPDATE factory.projects
        SET status='paused', autonomous_enabled=false, paused_at=now(),
            metadata = metadata || {_j({'autonomous_enabled': False})},
            updated_at=now()
        WHERE project_id={_q(project_id)};
        UPDATE factory.lanes SET status='paused', updated_at=now() WHERE project_id={_q(project_id)} AND status='active';
        INSERT INTO factory.events(project_id, actor, event_type, message, metadata)
        VALUES ({_q(project_id)}, 'factory-orchestrator', 'autonomous_pause', 'Autonomous Factory execution paused', {_j({})});
        """,
        user=_user(),
    )
    return {"project_id": project_id, "status": "paused"}


def _task_blocker_text(task: dict[str, Any]) -> str:
    return "\n".join(str(task.get(key) or "") for key in ("title", "description", "result_summary")).lower()


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _age_minutes(value: Any, *, now: datetime | None = None) -> float | None:
    parsed = _parse_datetime(value)
    if not parsed:
        return None
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - parsed).total_seconds() / 60.0)


def _active_run_task_ids(payload: dict[str, Any]) -> set[str]:
    return {
        str(run.get("task_id"))
        for run in payload.get("task_runs", [])
        if str(run.get("status") or "") in {"queued", "running"} and run.get("task_id")
    }


def _latest_gate_status_map_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(gate_type).lower(): str(status).lower()
        for gate_type, status in _latest_gate_statuses(payload.get("gates", [])).items()
    }


def _mentions_resolved_gate(task: dict[str, Any], gate_statuses: dict[str, str]) -> bool:
    text = _task_blocker_text(task)
    blocker_terms = ("pending", "pendiente", "blocked", "blocker", "bloque", "espera", "requiere")
    if not any(term in text for term in blocker_terms):
        return False
    mentioned = [gate_type for gate_type in gate_statuses if gate_type and gate_type in text]
    return bool(mentioned) and all(gate_statuses.get(gate_type) == "passed" for gate_type in mentioned)


def classify_factory_blocker(task: dict[str, Any], *, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Classify a blocked/stuck Factory task into the runtime action taxonomy."""

    payload = payload or {}
    text = _task_blocker_text(task)
    status_value = str(task.get("status") or "")
    task_id = str(task.get("task_id") or "")
    active_task_ids = _active_run_task_ids(payload)
    gate_statuses = _latest_gate_status_map_from_payload(payload)

    if status_value in {"claimed", "running", "in_progress", "review_running"} and task_id not in active_task_ids:
        action_category = "stale_orphan_state"
        blocker_category = "runtime_state"
        recommended_action = "repair_orphan_inflight_state"
        requires_human = False
    elif _mentions_resolved_gate(task, gate_statuses) or any(keyword in text for keyword in _BLOCKER_AUTO_KEYWORDS):
        action_category = "auto_resolvable"
        blocker_category = "resolved_gate_or_routine_decision"
        recommended_action = "run_orchestrator_unblock_or_record_gate"
        requires_human = False
    elif any(keyword in text for keyword in _BLOCKER_HUMAN_KEYWORDS):
        action_category = "human_question_required"
        blocker_category = "external_or_owner_decision"
        recommended_action = "create_human_question_and_notify_owner"
        requires_human = True
    elif any(keyword in text for keyword in _BLOCKER_TECHNICAL_KEYWORDS):
        action_category = "technical_rework"
        blocker_category = "technical"
        recommended_action = "delegate_to_programming_worker_for_rework"
        requires_human = False
    else:
        action_category = "technical_rework" if status_value == "blocked" else "auto_resolvable"
        blocker_category = "unclassified"
        recommended_action = "delegate_to_programming_worker_for_rework" if status_value == "blocked" else "run_orchestrator_tick"
        requires_human = False

    alert_key = f"factory:{task.get('project_id')}:{task_id}:{action_category}"
    return {
        "task_id": task_id,
        "project_id": task.get("project_id"),
        "lane_id": task.get("lane_id"),
        "title": task.get("title"),
        "status": status_value,
        "blocker_category": blocker_category,
        "action_category": action_category,
        "recommended_action": recommended_action,
        "requires_human": requires_human,
        "alert_key": alert_key,
        "updated_at": task.get("updated_at"),
        "result_summary": task.get("result_summary"),
    }


def classify_factory_blockers(payload: Optional[dict[str, Any]] = None, *, project_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Return action classifications for blocked tasks and orphan in-flight rows."""

    payload = payload or status(project_id)
    active_task_ids = _active_run_task_ids(payload)
    classified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in payload.get("tasks", []):
        if project_id and task.get("project_id") != project_id:
            continue
        task_id = str(task.get("task_id") or "")
        status_value = str(task.get("status") or "")
        is_blocked = status_value == "blocked"
        is_orphan = status_value in {"claimed", "running", "in_progress", "review_running"} and task_id not in active_task_ids
        if not (is_blocked or is_orphan) or task_id in seen:
            continue
        seen.add(task_id)
        classified.append(classify_factory_blocker(task, payload=payload))
    return classified


def record_factory_blocker_actions(classified: Optional[list[dict[str, Any]]] = None, *, payload: Optional[dict[str, Any]] = None, create_questions: bool = True) -> dict[str, Any]:
    """Persist blocker classifications and create human questions when unavoidable."""

    ensure_runtime_schema()
    payload = payload or status()
    classified = classified if classified is not None else classify_factory_blockers(payload)
    events_recorded = 0
    questions_created = 0
    for item in classified:
        task_id = str(item.get("task_id") or "")
        project_id = str(item.get("project_id") or "")
        if not task_id or not project_id:
            continue
        compact = {
            "action_category": item.get("action_category"),
            "blocker_category": item.get("blocker_category"),
            "recommended_action": item.get("recommended_action"),
            "requires_human": item.get("requires_human"),
            "alert_key": item.get("alert_key"),
        }
        sql.psql(
            f"""
            UPDATE factory.tasks
            SET metadata = metadata || {_j({'last_blocker_classification': compact})}, updated_at=updated_at
            WHERE task_id={_q(task_id)};
            INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
            SELECT {_q(project_id)}, {_q(item.get('lane_id'))}, {_q(task_id)}, 'factory-blocker-detector', 'blocker_classified',
                   {_q('Factory blocker classified as ' + str(item.get('action_category')))}, {_j(compact)}
            WHERE NOT EXISTS (
              SELECT 1 FROM factory.events
              WHERE task_id={_q(task_id)}
                AND event_type='blocker_classified'
                AND metadata->>'alert_key'={_q(item.get('alert_key'))}
                AND timestamp > now() - interval '60 minutes'
            );
            """,
            user=_user(),
        )
        events_recorded += 1
        if create_questions and item.get("requires_human"):
            question_id = "hq-" + uuid.uuid5(uuid.NAMESPACE_URL, f"factory:{task_id}:human-question").hex[:16]
            question = (
                f"Factory project {project_id} requires a human decision for task {task_id}: "
                f"{item.get('title') or 'blocked task'}. Recommended action: {item.get('recommended_action')}."
            )
            before = sql.one(
                f"SELECT question_id FROM factory.human_questions WHERE question_id={_q(question_id)} OR (task_id={_q(task_id)} AND status='pending') LIMIT 1",
                user=_user(),
            )
            sql.psql(
                f"""
                INSERT INTO factory.human_questions(question_id, project_id, task_id, severity, question, options, asked_via, status, metadata)
                VALUES ({_q(question_id)}, {_q(project_id)}, {_q(task_id)}, 'high', {_q(question)}, '[]'::jsonb, 'factory_watchdog', 'pending', {_j({'alert_key': item.get('alert_key'), 'classification': compact})})
                ON CONFLICT (question_id) DO NOTHING;
                INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
                SELECT {_q(project_id)}, {_q(item.get('lane_id'))}, {_q(task_id)}, 'factory-blocker-detector', 'human_question_created', {_q('Created human question for indispensable Factory blocker')}, {_j({'question_id': question_id, 'alert_key': item.get('alert_key')})}
                WHERE NOT EXISTS (
                  SELECT 1 FROM factory.events
                  WHERE task_id={_q(task_id)} AND event_type='human_question_created' AND metadata->>'question_id'={_q(question_id)}
                );
                """,
                user=_user(),
            )
            if not before:
                questions_created += 1
    return {"classified": len(classified), "events_recorded": events_recorded, "questions_created": questions_created}


def factory_watchdog_alerts(payload: Optional[dict[str, Any]] = None, *, blocked_minutes: int = BLOCKED_ALERT_DEFAULT_MINUTES, claimed_null_rounds: int = 0, project_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Compute deterministic Factory watchdog alerts for dashboard/cron."""

    payload = payload or status(project_id)
    now = datetime.now(timezone.utc)
    projects = {str(project.get("project_id")): project for project in payload.get("projects", [])}
    tasks_by_project: dict[str, list[dict[str, Any]]] = {}
    for task in payload.get("tasks", []):
        if project_id and task.get("project_id") != project_id:
            continue
        tasks_by_project.setdefault(str(task.get("project_id")), []).append(task)
    pending_questions_by_project: dict[str, int] = {}
    for question in payload.get("human_questions", []):
        if str(question.get("status") or "") == "pending":
            pid = str(question.get("project_id") or "")
            pending_questions_by_project[pid] = pending_questions_by_project.get(pid, 0) + 1

    alerts: list[dict[str, Any]] = []
    for pid, project in projects.items():
        if project_id and pid != project_id:
            continue
        if not project.get("autonomous_enabled"):
            continue
        ptasks = tasks_by_project.get(pid, [])
        blocked_tasks = [task for task in ptasks if str(task.get("status") or "") == "blocked"]
        if str(project.get("status") or "") == "blocked" and blocked_tasks:
            ages = [age for age in (_age_minutes(task.get("updated_at") or task.get("created_at"), now=now) for task in blocked_tasks) if age is not None]
            max_age = max(ages) if ages else None
            if max_age is not None and max_age >= blocked_minutes:
                alerts.append({
                    "alert_key": f"factory:{pid}:blocked>{blocked_minutes}m",
                    "alert_type": "autonomous_project_blocked_too_long",
                    "severity": "high",
                    "project_id": pid,
                    "message": f"Autonomous Factory project {pid} has been blocked for {int(max_age)} minutes.",
                    "age_minutes": max_age,
                    "blocked_tasks": [task.get("task_id") for task in blocked_tasks],
                })
            if pending_questions_by_project.get(pid, 0) == 0:
                alerts.append({
                    "alert_key": f"factory:{pid}:blocked-without-human-question",
                    "alert_type": "blocked_without_human_question",
                    "severity": "medium",
                    "project_id": pid,
                    "message": f"Factory project {pid} has blocked tasks but no pending human question; Zeus must auto-resolve or create one.",
                    "blocked_tasks": [task.get("task_id") for task in blocked_tasks],
                })

    for item in classify_factory_blockers(payload, project_id=project_id):
        if item.get("action_category") == "stale_orphan_state":
            alerts.append({
                "alert_key": item.get("alert_key"),
                "alert_type": "orphan_inflight_without_active_run",
                "severity": "high",
                "project_id": item.get("project_id"),
                "task_id": item.get("task_id"),
                "message": f"Task {item.get('task_id')} is {item.get('status')} without an active task_run.",
                "recommended_action": item.get("recommended_action"),
            })
    if claimed_null_rounds >= CLAIMED_NULL_ALERT_ROUNDS:
        alerts.append({
            "alert_key": "factory:cron:claimed-null-repeated",
            "alert_type": "cron_claimed_null_repeated",
            "severity": "medium",
            "project_id": project_id,
            "message": f"Factory cron returned claimed=null for {claimed_null_rounds} consecutive rounds while cron status is OK.",
            "claimed_null_rounds": claimed_null_rounds,
        })
    return alerts


def clear_resolved_blockers(project_id: str) -> dict[str, Any]:
    """Reopen blocked tasks whose recorded blocker is already resolved.

    This is intentionally conservative. It only clears a blocker when the task
    text says it is waiting on a named gate and the latest effective gate row is
    already ``passed``. The task is moved to ``review_ready`` rather than
    self-approved; the assigned reviewer still has to close the gate.
    """

    ensure_runtime_schema()
    effective = {
        str(gate.get("gate_type") or "").lower(): str(gate.get("status") or "").lower()
        for gate in _latest_gate_rows(project_id)
    }
    blocked = [task for task in _tasks(project_id) if str(task.get("status") or "") == "blocked"]
    reopened: list[dict[str, Any]] = []
    blocker_terms = ("pending", "pendiente", "blocked", "blocker", "bloque", "espera", "requiere")
    for task in blocked:
        text = "\n".join(
            str(task.get(key) or "") for key in ("title", "description", "result_summary")
        ).lower()
        if not any(term in text for term in blocker_terms):
            continue
        mentioned_gates = [
            gate_type
            for gate_type in effective
            if gate_type and gate_type in text
        ]
        if not mentioned_gates:
            continue
        unresolved_gates = [gate_type for gate_type in mentioned_gates if effective.get(gate_type) != "passed"]
        if unresolved_gates:
            continue
        resolved_gates = mentioned_gates
        summary = str(task.get("result_summary") or "").rstrip()
        unblock_note = (
            "\n\n[factory-orchestrator] Bloqueo de gate resuelto: "
            + ", ".join(sorted(resolved_gates))
            + " ya está passed en el gate efectivo. Tarea reabierta para review; no auto-aprobada."
        )
        sql.psql(
            f"""
            UPDATE factory.tasks
            SET status='review_ready',
                evidence_status=CASE WHEN evidence_status='missing' THEN 'present' ELSE evidence_status END,
                result_summary={_q((summary + unblock_note).strip())},
                metadata = metadata || {_j({'unblocked_by': 'factory-orchestrator', 'unblock_reason': 'resolved_gate_blocker', 'resolved_gates': sorted(resolved_gates)})},
                updated_at=now()
            WHERE task_id={_q(task['task_id'])};
            INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
            VALUES ({_q(project_id)}, {_q(task.get('lane_id'))}, {_q(task['task_id'])}, 'factory-orchestrator', 'blocker_resolved', {_q('Resolved stale gate blocker and reopened task for review')}, {_j({'resolved_gates': sorted(resolved_gates)})});
            """,
            user=_user(),
        )
        reopened.append({"task_id": task["task_id"], "resolved_gates": sorted(resolved_gates)})
    reconciled = reconcile_project(project_id)
    return {"project_id": project_id, "reopened": reopened, **reconciled}


def claim_next_review(project_id: Optional[str] = None, *, worker: str = "factory-reviewer-dispatch") -> dict[str, Any] | None:
    ensure_runtime_schema()
    project_filter = f"AND p.project_id={_q(project_id)}" if project_id else ""
    row = sql.statement_one(
        f"""
        UPDATE factory.tasks t
        SET status='review_running', claimed_by={_q(worker)}, claimed_at=now(), lease_until=now() + interval '30 minutes', updated_at=now()
        FROM factory.projects p
        WHERE t.project_id=p.project_id
          AND p.autonomous_enabled IS TRUE
          AND p.status IN ('active','planned','intake','blocked')
          AND t.status='review_ready'
          {project_filter}
          AND NOT EXISTS (
            SELECT 1 FROM factory.task_runs r
            WHERE r.project_id=t.project_id AND r.status IN ('queued','running')
          )
          AND t.task_id = (
            SELECT t2.task_id
            FROM factory.tasks t2
            WHERE t2.project_id=t.project_id AND t2.status='review_ready'
            ORDER BY t2.priority, t2.created_at
            LIMIT 1
          )
        RETURNING t.*
        """,
        user=_user(),
    )
    if not row:
        return None
    normalized = _normalize(row)
    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    reviewer_profile = normalized.get("reviewer_profile") or normalized.get("reviewer_agent_id") or "quality-reviewer"
    sql.psql(
        f"""
        INSERT INTO factory.task_runs(run_id, task_id, project_id, lane_id, worker_profile, reviewer_profile, engine, status, started_at, heartbeat_at, metadata)
        VALUES ({_q(run_id)}, {_q(normalized['task_id'])}, {_q(normalized['project_id'])}, {_q(normalized.get('lane_id'))}, {_q(reviewer_profile)}, {_q(reviewer_profile)}, {_q(normalized.get('engine'))}, 'queued', now(), now(), {_j({'claimed_by': worker, 'run_type': 'review'})});
        INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
        VALUES ({_q(normalized['project_id'])}, {_q(normalized.get('lane_id'))}, {_q(normalized['task_id'])}, {_q(worker)}, 'review_claimed', {_q(f"Task {normalized['task_id']} claimed for review by {reviewer_profile}")}, {_j({'run_id': run_id, 'worker_profile': reviewer_profile, 'run_type': 'review'})});
        """,
        user=_user(),
    )
    return {"run_id": run_id, "task": normalized, "worker_profile": reviewer_profile, "run_type": "review"}


def claim_next_rework(project_id: Optional[str] = None, *, worker: str = "factory-dispatcher") -> dict[str, Any] | None:
    """Claim the oldest rework task for its original owner.

    Review failure should not be terminal and should not require Jean/Zeus to
    manually flip the row back to ``todo``. ``rework`` blocks later increments,
    but it is itself runnable by the owner with the reviewer findings carried in
    ``result_summary``.
    """

    ensure_runtime_schema()
    project_filter = f"AND p.project_id={_q(project_id)}" if project_id else ""
    in_flight = ",".join(_q(status) for status in IN_FLIGHT_TASK_STATUSES)
    row = sql.statement_one(
        f"""
        UPDATE factory.tasks t
        SET status='claimed',
            claimed_by={_q(worker)},
            claimed_at=now(),
            lease_until=now() + interval '30 minutes',
            retry_count=COALESCE(retry_count, 0) + 1,
            updated_at=now()
        FROM factory.projects p
        WHERE t.project_id=p.project_id
          AND p.autonomous_enabled IS TRUE
          AND p.status IN ('active','planned','intake','blocked')
          AND t.status='rework'
          {project_filter}
          AND NOT EXISTS (
            SELECT 1 FROM factory.task_runs r
            WHERE r.project_id=t.project_id AND r.status IN ('queued','running')
          )
          AND NOT EXISTS (
            SELECT 1 FROM factory.tasks t_busy
            WHERE t_busy.project_id=t.project_id AND t_busy.status IN ({in_flight})
          )
          AND t.task_id = (
            SELECT t2.task_id
            FROM factory.tasks t2
            WHERE t2.project_id=t.project_id AND t2.status='rework'
            ORDER BY t2.priority, t2.updated_at, t2.created_at
            LIMIT 1
          )
        RETURNING t.*
        """,
        user=_user(),
    )
    if not row:
        return None
    normalized = _normalize(row)
    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    worker_profile = normalized.get("owner_profile") or normalized.get("owner_agent_id") or "factory-orchestrator"
    sql.psql(
        f"""
        INSERT INTO factory.task_runs(run_id, task_id, project_id, lane_id, worker_profile, reviewer_profile, engine, status, started_at, heartbeat_at, metadata)
        VALUES ({_q(run_id)}, {_q(normalized['task_id'])}, {_q(normalized['project_id'])}, {_q(normalized.get('lane_id'))}, {_q(worker_profile)}, {_q(normalized.get('reviewer_profile'))}, {_q(normalized.get('engine'))}, 'queued', now(), now(), {_j({'claimed_by': worker, 'run_type': 'rework', 'previous_status': 'rework'})});
        INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
        VALUES ({_q(normalized['project_id'])}, {_q(normalized.get('lane_id'))}, {_q(normalized['task_id'])}, {_q(worker)}, 'rework_claimed', {_q(f"Task {normalized['task_id']} claimed for rework by {worker_profile}")}, {_j({'run_id': run_id, 'worker_profile': worker_profile, 'run_type': 'rework'})});
        """,
        user=_user(),
    )
    return {"run_id": run_id, "task": normalized, "worker_profile": worker_profile, "run_type": "rework"}


def claim_next_task(project_id: Optional[str] = None, *, worker: str = "factory-dispatcher") -> dict[str, Any] | None:
    ensure_runtime_schema()
    project_filter = f"AND p.project_id={_q(project_id)}" if project_id else ""
    projects = sql.rows(
        f"""
        SELECT p.project_id
        FROM factory.projects p
        WHERE p.autonomous_enabled IS TRUE
          AND p.status IN ('active','planned','intake','blocked')
          {project_filter}
        ORDER BY p.updated_at, p.started_at
        """,
        user=_user(),
    )
    for project in projects:
        pid = project["project_id"]
        tasks = _tasks(pid)
        if _has_active_increment(tasks):
            continue
        task = _next_runnable_task(pid)
        if not task:
            reconcile_project(pid)
            continue
        run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        worker_profile = task.get("owner_profile") or task.get("owner_agent_id") or "factory-orchestrator"
        row = sql.statement_one(
            f"""
            UPDATE factory.tasks
            SET status='claimed', claimed_by={_q(worker)}, claimed_at=now(), lease_until=now() + interval '30 minutes', updated_at=now()
            WHERE task_id={_q(task['task_id'])} AND status IN ('todo','ready')
            RETURNING *
            """,
            user=_user(),
        )
        if not row:
            continue
        sql.psql(
            f"""
            INSERT INTO factory.task_runs(run_id, task_id, project_id, lane_id, worker_profile, reviewer_profile, engine, status, started_at, heartbeat_at, metadata)
            VALUES ({_q(run_id)}, {_q(row['task_id'])}, {_q(row['project_id'])}, {_q(row.get('lane_id'))}, {_q(worker_profile)}, {_q(row.get('reviewer_profile'))}, {_q(row.get('engine'))}, 'queued', now(), now(), {_j({'claimed_by': worker})});
            INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
            VALUES ({_q(row['project_id'])}, {_q(row.get('lane_id'))}, {_q(row['task_id'])}, {_q(worker)}, 'task_claimed', {_q(f"Task {row['task_id']} claimed for {worker_profile}")}, {_j({'run_id': run_id, 'worker_profile': worker_profile})});
            """,
            user=_user(),
        )
        return {"run_id": run_id, "task": _normalize(row), "worker_profile": worker_profile}
    return None


def mark_run_spawned(run_id: str, *, process_id: int, log_path: str, prompt_path: str) -> None:
    ensure_runtime_schema()
    sql.psql(
        f"""
        UPDATE factory.task_runs
        SET status='running', process_id={int(process_id)}, log_path={_q(log_path)}, prompt_path={_q(prompt_path)}, heartbeat_at=now()
        WHERE run_id={_q(run_id)};
        UPDATE factory.tasks
        SET status='running', last_heartbeat_at=now(), updated_at=now()
        WHERE task_id=(SELECT task_id FROM factory.task_runs WHERE run_id={_q(run_id)});
        """,
        user=_user(),
    )


_SEMANTIC_DONE_MARKERS = ("STATE: DONE", "STATE:DONE")
_SEMANTIC_BLOCKED_MARKERS = ("STATE: BLOCKED", "STATE:BLOCKED")
_SEMANTIC_MARKERS = _SEMANTIC_DONE_MARKERS + _SEMANTIC_BLOCKED_MARKERS


def _final_semantic_state(text: str) -> Optional[str]:
    # Returns 'done', 'blocked', or None for the LAST semantic marker in `text`.
    # Historical STATE markers embedded in the prompt or in a prior
    # result_summary must never override the final assistant marker.
    last_done = max(text.rfind(marker) for marker in _SEMANTIC_DONE_MARKERS)
    last_blocked = max(text.rfind(marker) for marker in _SEMANTIC_BLOCKED_MARKERS)
    if last_done == -1 and last_blocked == -1:
        return None
    return "blocked" if last_blocked > last_done else "done"


def _read_worker_output_summary(log_path: str | Path, *, tail_chars: int = 4000) -> str:
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "Worker log unreadable"
    tail = text[-tail_chars:]
    # Identify the FINAL semantic marker line. Earlier markers (e.g. the prompt
    # embedding a previous result_summary with STATE: BLOCKED) must not be
    # promoted to the summary as if they were the current outcome.
    final_marker_line: str | None = None
    for line in text.splitlines():
        if any(marker in line for marker in _SEMANTIC_MARKERS):
            final_marker_line = line.strip()[-500:]
    if final_marker_line and final_marker_line not in tail:
        return "Final semantic state marker:\n" + final_marker_line + "\n\nLog tail:\n" + tail
    return tail


def _effective_exit_code(exit_code: int, output_summary: str) -> int:
    # Agents may exit 0 after producing a final blocked/rework report, or exit
    # non-zero after producing a final DONE. The FINAL semantic marker wins
    # over the process exit, but earlier historical markers (prompt context,
    # previous result_summary echoes) must not be allowed to mask it.
    state = _final_semantic_state(output_summary)
    if state == "blocked":
        return exit_code if exit_code != 0 else 1
    if state == "done":
        return 0
    return exit_code


def mark_run_finished(run_id: str, *, exit_code: int, output_summary: str = "") -> None:
    ensure_runtime_schema()
    effective_exit_code = _effective_exit_code(exit_code, output_summary)
    run_row = _normalize(sql.one(f"SELECT metadata FROM factory.task_runs WHERE run_id={_q(run_id)}", user=_user()) or {})
    metadata_value = run_row.get("metadata")
    metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
    run_type = str(metadata.get("run_type") or "implementation")
    if run_type == "review":
        status_value = "done" if effective_exit_code == 0 else "rework"
        evidence_status = "present"
    else:
        status_value = "review_ready" if effective_exit_code == 0 else "blocked"
        evidence_status = "present" if effective_exit_code == 0 else "missing"
    run_status = "succeeded" if effective_exit_code == 0 else "failed"
    sql.psql(
        f"""
        UPDATE factory.task_runs
        SET status={_q(run_status)}, exit_code={int(effective_exit_code)}, finished_at=now(), output_summary={_q(output_summary)}, heartbeat_at=now()
        WHERE run_id={_q(run_id)};
        UPDATE factory.tasks
        SET status={_q(status_value)}, evidence_status={_q(evidence_status)}, result_summary={_q(output_summary)}, finished_at=CASE WHEN {int(effective_exit_code)} = 0 THEN now() ELSE finished_at END, updated_at=now()
        WHERE task_id=(SELECT task_id FROM factory.task_runs WHERE run_id={_q(run_id)});
        """,
        user=_user(),
    )
    row = sql.one(f"SELECT project_id FROM factory.task_runs WHERE run_id={_q(run_id)}", user=_user())
    if row:
        reconcile_project(row["project_id"])


def _repair_orphan_in_flight_tasks(project_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Reopen in-flight task rows that no longer have an active run.

    A task in ``review_running``/``running``/``claimed`` without a queued or
    running ``factory.task_runs`` row is an impossible state: no worker can
    finish it and the dispatcher treats it as active forever.  Keep the repair
    conservative: only rows with **no** active run are reset.  Live long-running
    workers remain untouched even if their lease is old.
    """

    ensure_runtime_schema()
    project_filter = "" if not project_id else f"AND t.project_id={_q(project_id)}"
    active_run_filter = """
        NOT EXISTS (
            SELECT 1 FROM factory.task_runs r
            WHERE r.task_id=t.task_id
              AND r.project_id=t.project_id
              AND r.status IN ('queued','running')
        )
    """
    review_rows = _normalize_rows(sql.json_query(
        f"""
        WITH updated AS (
            UPDATE factory.tasks t
            SET status='review_ready',
                metadata = metadata || {_j({'orphan_repaired_by': 'factory-monitor', 'orphan_repair_reason': 'review_running_without_active_run'})},
                updated_at=now()
            WHERE t.status='review_running'
              {project_filter}
              AND {active_run_filter}
            RETURNING t.project_id, t.lane_id, t.task_id, t.status
        )
        SELECT COALESCE(jsonb_agg(to_jsonb(updated)), '[]'::jsonb)::text FROM updated;
        """,
        user=_user(),
    ) or [])
    worker_rows = _normalize_rows(sql.json_query(
        f"""
        WITH updated AS (
            UPDATE factory.tasks t
            SET status='ready',
                metadata = metadata || {_j({'orphan_repaired_by': 'factory-monitor', 'orphan_repair_reason': 'worker_in_flight_without_active_run'})},
                updated_at=now()
            WHERE t.status IN ('claimed','running','in_progress')
              {project_filter}
              AND {active_run_filter}
            RETURNING t.project_id, t.lane_id, t.task_id, t.status
        )
        SELECT COALESCE(jsonb_agg(to_jsonb(updated)), '[]'::jsonb)::text FROM updated;
        """,
        user=_user(),
    ) or [])
    repaired = review_rows + worker_rows
    if repaired:
        values = ",\n".join(
            f"({_q(row.get('project_id'))}, {_q(row.get('lane_id'))}, {_q(row.get('task_id'))}, 'factory-monitor', 'orphan_inflight_repaired', {_q('Reopened orphan in-flight task with no active run')}, {_j({'new_status': row.get('status')})})"
            for row in repaired
        )
        sql.psql(
            "INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata) VALUES\n"
            + values
            + ";",
            user=_user(),
        )
    return repaired


def monitor_runs() -> dict[str, Any]:
    ensure_runtime_schema()
    running = _normalize_rows(sql.rows("SELECT * FROM factory.task_runs WHERE status='running' ORDER BY started_at", user=_user()))
    checked = 0
    finished = 0
    for run in running:
        checked += 1
        metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
        exit_path = metadata.get("exit_path")
        if not exit_path:
            continue
        path = Path(str(exit_path)).expanduser()
        if not path.exists():
            continue
        try:
            exit_code = int(path.read_text(encoding="utf-8").strip() or "1")
        except Exception:
            exit_code = 1
        log_path = str(run.get("log_path") or "")
        summary = ""
        if log_path and Path(log_path).exists():
            summary = _read_worker_output_summary(log_path)
        mark_run_finished(run["run_id"], exit_code=exit_code, output_summary=summary)
        finished += 1
    repaired = _repair_orphan_in_flight_tasks()
    return {"checked": checked, "finished": finished, "orphan_inflight_repaired": len(repaired)}


def update_run_metadata(run_id: str, metadata: dict[str, Any]) -> None:
    ensure_runtime_schema()
    sql.psql(
        f"UPDATE factory.task_runs SET metadata=metadata || {_j(metadata)}, heartbeat_at=now() WHERE run_id={_q(run_id)};",
        user=_user(),
    )


def force_tick(project_id: Optional[str] = None) -> dict[str, Any]:
    ensure_runtime_schema()
    monitor = monitor_runs()
    if project_id:
        unblock_targets = [project_id]
    else:
        unblock_targets = [
            row["project_id"]
            for row in sql.rows(
                "SELECT project_id FROM factory.projects WHERE status='blocked' OR autonomous_enabled IS TRUE ORDER BY updated_at",
                user=_user(),
            )
        ]
    unblocked = []
    for target in unblock_targets:
        result = clear_resolved_blockers(target)
        if result.get("reopened"):
            unblocked.append({"project_id": target, "reopened": result.get("reopened")})
    if project_id:
        reconciled = [reconcile_project(project_id)]
    else:
        reconciled = [reconcile_project(row["project_id"]) for row in sql.rows("SELECT project_id FROM factory.projects ORDER BY updated_at", user=_user())]
    claimed = claim_next_review(project_id, worker="factory-force-tick")
    if not claimed:
        claimed = claim_next_rework(project_id, worker="factory-force-tick")
    if not claimed:
        claimed = claim_next_task(project_id, worker="factory-force-tick")
    return {"monitor": monitor, "unblocked": unblocked, "reconciled": reconciled, "claimed": claimed}


def control_action(project_id: str, action: str) -> dict[str, Any]:
    if action == "resume":
        return {"action": action, **resume_project(project_id)}
    if action == "pause":
        return {"action": action, **pause_project(project_id)}
    if action == "unblock":
        return {"action": action, **clear_resolved_blockers(project_id)}
    if action == "tick":
        return {"action": action, **force_tick(project_id)}
    if action == "reconcile":
        return {"action": action, **reconcile_project(project_id)}
    raise ValueError(f"Unsupported factory action: {action}")


def status(project_id: Optional[str] = None, **_: Any) -> dict[str, Any]:
    ensure_runtime_schema()
    where = "" if not project_id else f"WHERE project_id={_q(project_id)}"
    filter_expr = "TRUE" if not project_id else f"project_id={_q(project_id)}"
    projects = _normalize_rows(sql.rows(f"SELECT * FROM factory.projects {where} ORDER BY updated_at DESC", user=_user()))
    lanes = _normalize_rows(sql.rows(f"SELECT *, COALESCE(metadata->>'execution_surface', 'factory') AS execution_surface FROM factory.lanes WHERE {filter_expr} ORDER BY project_id, lane_id", user=_user()))
    tasks = _normalize_rows(sql.rows(f"SELECT * FROM factory.tasks WHERE {filter_expr} ORDER BY project_id, priority, created_at", user=_user()))
    for task in tasks:
        task.pop("kanban_id", None)
    gates = _normalize_rows(sql.rows(f"SELECT * FROM factory.gates WHERE {filter_expr} ORDER BY timestamp DESC LIMIT 300", user=_user()))
    events = _normalize_rows(sql.rows(f"SELECT * FROM factory.events WHERE {filter_expr} ORDER BY timestamp DESC LIMIT 300", user=_user()))
    artifacts = _normalize_rows(sql.rows(f"SELECT * FROM factory.artifacts WHERE {filter_expr} ORDER BY created_at DESC LIMIT 300", user=_user()))
    task_runs = _normalize_rows(sql.rows(f"SELECT * FROM factory.task_runs WHERE {filter_expr} ORDER BY started_at DESC LIMIT 300", user=_user()))
    human_questions = _normalize_rows(sql.rows(f"SELECT * FROM factory.human_questions WHERE {filter_expr} ORDER BY created_at DESC LIMIT 100", user=_user()))
    agents = list_agents()
    payload = {
        "db_backend": "agent_core_postgres",
        "database": sql.runtime_env().get("AGENT_DB_NAME", "zeus_agent"),
        "db_path": f"agent_core_postgres:{sql.runtime_env().get('AGENT_DB_NAME', 'zeus_agent')}.factory",
        "projects": projects,
        "lanes": lanes,
        "tasks": tasks,
        "gates": gates,
        "events": events,
        "artifacts": artifacts,
        "task_runs": task_runs,
        "human_questions": human_questions,
        "agents": agents,
    }
    payload["alerts"] = factory_watchdog_alerts(payload, project_id=project_id)
    return payload
