"""SitioUno Software Factory progress database.

This module is intentionally lightweight and stdlib-only. It provides the local
SQLite fallback used by tests, dry-runs, and single-host operation. Production
Cloud SQL uses the SQL migration under ``migrations/software_factory``; the CLI
schema mirrors the same core entities so the factory can operate offline and
later sync/report against Postgres.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home

VALID_METHODS = {"zeus_native", "bmad_hybrid", "hybrid", "dual_lane"}
DEFAULT_LANES = (
    ("zeus", "Zeus Native", "zeus_native"),
    ("bmad", "BMAD Hybrid", "bmad_hybrid"),
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    text = (value or "").strip().lower()
    text = _SLUG_RE.sub("-", text).strip("-")
    return text or "factory-project"


def default_db_path() -> Path:
    override = os.environ.get("HERMES_FACTORY_DB", "").strip()
    if override:
        return Path(override).expanduser()
    return get_hermes_home() / "factory" / "factory.db"


def _json(value: Any) -> str:
    if value is None:
        value = {} if isinstance(value, dict) else value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS factory_projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    repo_path TEXT,
    repo_remote TEXT,
    base_branch TEXT,
    status TEXT NOT NULL DEFAULT 'intake',
    autonomy_level INTEGER NOT NULL DEFAULT 3,
    methodology TEXT NOT NULL DEFAULT 'dual_lane',
    risk_level TEXT NOT NULL DEFAULT 'medium',
    human_owner TEXT,
    summary TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factory_agents (
    agent_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    preferred_engine TEXT,
    toolsets_json TEXT NOT NULL DEFAULT '[]',
    skills_json TEXT NOT NULL DEFAULT '[]',
    greenlight_required_json TEXT NOT NULL DEFAULT '[]',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factory_lanes (
    lane_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES factory_projects(project_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    methodology TEXT NOT NULL,
    kanban_board TEXT,
    branch TEXT,
    worktree_path TEXT,
    status TEXT NOT NULL DEFAULT 'planned',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factory_tasks (
    task_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES factory_lanes(lane_id) ON DELETE SET NULL,
    kanban_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    phase TEXT NOT NULL DEFAULT 'intake',
    status TEXT NOT NULL DEFAULT 'todo',
    owner_agent_id TEXT REFERENCES factory_agents(agent_id) ON DELETE SET NULL,
    reviewer_agent_id TEXT REFERENCES factory_agents(agent_id) ON DELETE SET NULL,
    engine TEXT NOT NULL DEFAULT 'zeus',
    priority INTEGER NOT NULL DEFAULT 100,
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    acceptance_criteria_json TEXT NOT NULL DEFAULT '[]',
    evidence_required INTEGER NOT NULL DEFAULT 1,
    evidence_status TEXT NOT NULL DEFAULT 'missing',
    risk_level TEXT NOT NULL DEFAULT 'medium',
    branch TEXT,
    worktree_path TEXT,
    result_summary TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factory_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT REFERENCES factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES factory_tasks(task_id) ON DELETE SET NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factory_gates (
    gate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES factory_tasks(task_id) ON DELETE SET NULL,
    gate_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reviewer TEXT,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factory_artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES factory_tasks(task_id) ON DELETE SET NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    checksum TEXT,
    created_by TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_factory_lanes_project ON factory_lanes(project_id);
CREATE INDEX IF NOT EXISTS idx_factory_tasks_project_lane ON factory_tasks(project_id, lane_id);
CREATE INDEX IF NOT EXISTS idx_factory_tasks_status ON factory_tasks(status);
CREATE INDEX IF NOT EXISTS idx_factory_events_project_created ON factory_events(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_factory_gates_project_status ON factory_gates(project_id, status);
"""

FACTORY_AGENTS = [
    ("factory-orchestrator", "Factory Orchestrator", "Intake, routing, gates, metrics, reports", "zeus", ["kanban", "delegation", "terminal", "file", "cronjob", "skills", "web"], ["software-factory-orchestration", "kanban-orchestrator", "programming-delegation-engines"], ["merge", "deploy", "destructive", "credential-change"]),
    ("product-analyst", "Product Analyst", "Functional analysis, PRD, acceptance criteria", "zeus", ["file", "web", "session_search", "skills"], ["writing-plans"], ["publish"]),
    ("solution-architect", "Solution Architect", "Architecture, boundaries, integration design", "claude_code", ["terminal", "file", "web", "skills"], ["writing-plans", "codebase-inspection"], ["architecture-approval"]),
    ("implementation-planner", "Implementation Planner", "Epics, stories, dependencies, task graph", "zeus", ["kanban", "file", "skills"], ["writing-plans", "software-factory-orchestration"], []),
    ("claude-builder", "Claude Builder", "Complex implementation and refactors", "claude_code", ["terminal", "file", "web", "skills"], ["claude-code", "test-driven-development"], ["push", "merge"]),
    ("codex-builder", "Codex Builder", "Bounded fixes, tests, QA on diffs", "codex", ["terminal", "file", "web", "skills"], ["codex", "test-driven-development"], ["push", "merge"]),
    ("openhands-lab", "OpenHands Lab", "Sandbox experiments and independent validation", "openhands", ["terminal", "file", "web", "skills"], ["openhands-gcp"], ["external-write"]),
    ("quality-reviewer", "Quality Reviewer", "Independent spec and quality gate", "codex", ["terminal", "file", "web", "skills"], ["requesting-code-review", "github-code-review"], ["approve-merge"]),
    ("security-reviewer", "Security Reviewer", "Security and fintech/PII gates", "codex", ["terminal", "file", "web", "skills"], ["requesting-code-review", "systematic-debugging"], ["security-waiver"]),
    ("qa-verifier", "QA Verifier", "Smoke tests and evidence capture", "zeus", ["terminal", "file", "browser", "vision", "skills"], ["dogfood"], ["waive-tests"]),
    ("devops-release", "DevOps Release", "CI, environments, release readiness", "claude_code", ["terminal", "file", "web", "skills"], ["github-pr-workflow"], ["deploy", "credential-change"]),
    ("factory-reporter", "Factory Reporter", "Executive reports and benchmarks", "zeus", ["file", "session_search", "skills"], ["software-factory-orchestration"], []),
]


def connect(path: Optional[str | Path] = None) -> sqlite3.Connection:
    db_path = Path(path).expanduser() if path else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    seed_agents(conn)
    conn.commit()


def seed_agents(conn: sqlite3.Connection) -> None:
    now = utc_now()
    for agent_id, display_name, role, engine, toolsets, skills, greenlights in FACTORY_AGENTS:
        conn.execute(
            """
            INSERT INTO factory_agents (
                agent_id, display_name, role, preferred_engine,
                toolsets_json, skills_json, greenlight_required_json,
                active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                display_name=excluded.display_name,
                role=excluded.role,
                preferred_engine=excluded.preferred_engine,
                toolsets_json=excluded.toolsets_json,
                skills_json=excluded.skills_json,
                greenlight_required_json=excluded.greenlight_required_json,
                active=1,
                updated_at=excluded.updated_at
            """,
            (agent_id, display_name, role, engine, _json(toolsets), _json(skills), _json(greenlights), now, now),
        )


def ensure_db(path: Optional[str | Path] = None) -> sqlite3.Connection:
    conn = connect(path)
    init_db(conn)
    return conn


def create_project(
    name: str,
    *,
    project_id: Optional[str] = None,
    repo_path: Optional[str] = None,
    repo_remote: Optional[str] = None,
    base_branch: Optional[str] = None,
    human_owner: Optional[str] = None,
    summary: Optional[str] = None,
    risk_level: str = "medium",
    autonomy_level: int = 3,
    methodology: str = "dual_lane",
    create_default_lanes: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    own_conn = conn is None
    conn = conn or ensure_db()
    now = utc_now()
    pid = project_id or slugify(name)
    conn.execute(
        """
        INSERT INTO factory_projects (
            project_id, name, repo_path, repo_remote, base_branch, status,
            autonomy_level, methodology, risk_level, human_owner, summary,
            metadata_json, started_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'intake', ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id) DO UPDATE SET
            name=excluded.name,
            repo_path=excluded.repo_path,
            repo_remote=excluded.repo_remote,
            base_branch=excluded.base_branch,
            autonomy_level=excluded.autonomy_level,
            methodology=excluded.methodology,
            risk_level=excluded.risk_level,
            human_owner=excluded.human_owner,
            summary=excluded.summary,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        """,
        (pid, name, repo_path, repo_remote, base_branch, autonomy_level, methodology, risk_level, human_owner, summary, _json(metadata or {}), now, now),
    )
    log_event(conn, project_id=pid, actor="factory-orchestrator", event_type="project_created", message=f"Factory project {pid} initialized", metadata={"methodology": methodology})
    lanes: list[dict[str, Any]] = []
    if create_default_lanes:
        for suffix, lane_name, method in DEFAULT_LANES:
            lanes.append(create_lane(pid, lane_name, method, conn=conn))
    if own_conn:
        conn.commit()
        conn.close()
    return {"project_id": pid, "lanes": lanes}


def create_lane(
    project_id: str,
    name: str,
    methodology: str,
    *,
    lane_id: Optional[str] = None,
    kanban_board: Optional[str] = None,
    branch: Optional[str] = None,
    worktree_path: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    if methodology not in VALID_METHODS:
        raise ValueError(f"unknown methodology {methodology!r}; expected one of {sorted(VALID_METHODS)}")
    own_conn = conn is None
    conn = conn or ensure_db()
    now = utc_now()
    suffix = "zeus" if methodology == "zeus_native" else "bmad" if methodology == "bmad_hybrid" else slugify(methodology)
    lid = lane_id or f"{project_id}-{suffix}"
    board = kanban_board or lid
    branch_value = branch or f"factory/{project_id}/{suffix}"
    conn.execute(
        """
        INSERT INTO factory_lanes (
            lane_id, project_id, name, methodology, kanban_board, branch,
            worktree_path, status, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?)
        ON CONFLICT(lane_id) DO UPDATE SET
            name=excluded.name,
            methodology=excluded.methodology,
            kanban_board=excluded.kanban_board,
            branch=excluded.branch,
            worktree_path=excluded.worktree_path,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        """,
        (lid, project_id, name, methodology, board, branch_value, worktree_path, _json(metadata or {}), now, now),
    )
    log_event(conn, project_id=project_id, lane_id=lid, actor="factory-orchestrator", event_type="lane_created", message=f"Lane {lid} initialized", metadata={"methodology": methodology, "kanban_board": board, "branch": branch_value})
    if own_conn:
        conn.commit(); conn.close()
    return {"lane_id": lid, "project_id": project_id, "methodology": methodology, "kanban_board": board, "branch": branch_value}


def create_task(
    project_id: str,
    title: str,
    *,
    lane_id: Optional[str] = None,
    description: Optional[str] = None,
    phase: str = "planning",
    status: str = "todo",
    owner_agent_id: Optional[str] = None,
    reviewer_agent_id: Optional[str] = None,
    engine: str = "zeus",
    priority: int = 100,
    acceptance_criteria: Optional[list[str]] = None,
    dependencies: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    own_conn = conn is None
    conn = conn or ensure_db()
    now = utc_now()
    base = f"{project_id}-{slugify(title)[:40]}"
    task_id = base
    n = 2
    while conn.execute("SELECT 1 FROM factory_tasks WHERE task_id=?", (task_id,)).fetchone():
        task_id = f"{base}-{n}"
        n += 1
    conn.execute(
        """
        INSERT INTO factory_tasks (
            task_id, project_id, lane_id, title, description, phase, status,
            owner_agent_id, reviewer_agent_id, engine, priority,
            dependencies_json, acceptance_criteria_json, evidence_required,
            evidence_status, risk_level, metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'missing', 'medium', ?, ?, ?)
        """,
        (task_id, project_id, lane_id, title, description, phase, status, owner_agent_id, reviewer_agent_id, engine, priority, _json(dependencies or []), _json(acceptance_criteria or []), _json(metadata or {}), now, now),
    )
    log_event(conn, project_id=project_id, lane_id=lane_id, task_id=task_id, actor="factory-orchestrator", event_type="task_created", message=f"Task {task_id} created", metadata={"engine": engine, "owner": owner_agent_id})
    if own_conn:
        conn.commit(); conn.close()
    return {"task_id": task_id, "project_id": project_id, "lane_id": lane_id}


def record_gate(project_id: str, gate_type: str, status: str, *, lane_id: Optional[str] = None, task_id: Optional[str] = None, reviewer: Optional[str] = None, evidence: Optional[dict[str, Any]] = None, notes: Optional[str] = None, conn: Optional[sqlite3.Connection] = None) -> dict[str, Any]:
    own_conn = conn is None
    conn = conn or ensure_db()
    now = utc_now()
    cur = conn.execute(
        "INSERT INTO factory_gates (project_id, lane_id, task_id, gate_type, status, reviewer, evidence_json, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (project_id, lane_id, task_id, gate_type, status, reviewer, _json(evidence or {}), notes, now),
    )
    log_event(conn, project_id=project_id, lane_id=lane_id, task_id=task_id, actor=reviewer or "factory-orchestrator", event_type=f"gate_{status}", message=f"{gate_type} gate {status}", metadata={"gate_id": cur.lastrowid})
    if own_conn:
        conn.commit(); conn.close()
    return {"gate_id": cur.lastrowid, "project_id": project_id, "status": status}


def log_event(conn: sqlite3.Connection, *, project_id: Optional[str], actor: str, event_type: str, message: str, lane_id: Optional[str] = None, task_id: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> int:
    cur = conn.execute(
        "INSERT INTO factory_events (project_id, lane_id, task_id, actor, event_type, message, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (project_id, lane_id, task_id, actor, event_type, message, _json(metadata or {}), utc_now()),
    )
    return int(cur.lastrowid)


def list_agents(conn: Optional[sqlite3.Connection] = None) -> list[dict[str, Any]]:
    own_conn = conn is None
    conn = conn or ensure_db()
    rows = conn.execute("SELECT * FROM factory_agents ORDER BY agent_id").fetchall()
    result = [_row_to_dict(row) for row in rows]
    if own_conn:
        conn.close()
    return result


def status(project_id: Optional[str] = None, conn: Optional[sqlite3.Connection] = None) -> dict[str, Any]:
    own_conn = conn is None
    conn = conn or ensure_db()
    params: tuple[Any, ...] = ()
    where = ""
    if project_id:
        where = "WHERE project_id=?"
        params = (project_id,)
    projects = [_row_to_dict(row) for row in conn.execute(f"SELECT * FROM factory_projects {where} ORDER BY updated_at DESC", params).fetchall()]
    lanes = [_row_to_dict(row) for row in conn.execute("SELECT * FROM factory_lanes WHERE (? IS NULL OR project_id=?) ORDER BY project_id, lane_id", (project_id, project_id)).fetchall()]
    tasks = [_row_to_dict(row) for row in conn.execute("SELECT * FROM factory_tasks WHERE (? IS NULL OR project_id=?) ORDER BY project_id, priority, created_at", (project_id, project_id)).fetchall()]
    gates = [_row_to_dict(row) for row in conn.execute("SELECT * FROM factory_gates WHERE (? IS NULL OR project_id=?) ORDER BY created_at DESC LIMIT 50", (project_id, project_id)).fetchall()]
    if own_conn:
        conn.close()
    return {"db_path": str(default_db_path()), "projects": projects, "lanes": lanes, "tasks": tasks, "gates": gates}


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in list(data):
        if key.endswith("_json"):
            data[key] = _loads(data[key], [] if key.endswith("s_json") else {})
    return data


# ──────────────────────────────────────────────
# Deterministic job query helpers (Fase 4)
# ──────────────────────────────────────────────


def update_task_status(
    task_id: str,
    status: str,
    *,
    evidence_status: Optional[str] = None,
    result_summary: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Update a task's status and optional fields. Returns the task dict."""
    own_conn = conn is None
    conn = conn or ensure_db()
    now = utc_now()
    updates = ["status=?", "updated_at=?"]
    params: list[Any] = [status, now]
    if evidence_status is not None:
        updates.append("evidence_status=?")
        params.append(evidence_status)
    if result_summary is not None:
        updates.append("result_summary=?")
        params.append(result_summary)
    params.append(task_id)
    conn.execute(
        f"UPDATE factory_tasks SET {', '.join(updates)} WHERE task_id=?",
        params,
    )
    if status == "in_progress" and conn.execute(
        "SELECT 1 FROM factory_tasks WHERE task_id=? AND started_at IS NULL",
        (task_id,),
    ).fetchone():
        conn.execute(
            "UPDATE factory_tasks SET started_at=? WHERE task_id=?",
            (now, task_id),
        )
    if status in ("done", "cancelled", "blocked", "rework"):
        conn.execute(
            "UPDATE factory_tasks SET finished_at=? WHERE task_id=? AND finished_at IS NULL",
            (now, task_id),
        )
    if own_conn:
        conn.commit()
        conn.close()
    return {"task_id": task_id, "status": status}


def list_stale_tasks(
    *,
    max_age_minutes: int = 30,
    project_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Find tasks stuck 'in_progress' beyond max_age_minutes."""
    own_conn = conn is None
    conn = conn or ensure_db()
    where = "WHERE status='in_progress' AND started_at IS NOT NULL"
    params: list[Any] = []
    if project_id:
        where += " AND project_id=?"
        params.append(project_id)
    where += f" AND (julianday('now') - julianday(started_at)) * 1440 > ?"
    params.append(max_age_minutes)
    rows = conn.execute(
        f"SELECT * FROM factory_tasks {where} ORDER BY started_at",
        params,
    ).fetchall()
    result = [_row_to_dict(row) for row in rows]
    if own_conn:
        conn.close()
    return result


def list_tasks_ready_for_review(
    *,
    project_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Find tasks with status='in_progress' or 'review' that have passed
    the implementation phase and need review assignment."""
    own_conn = conn is None
    conn = conn or ensure_db()
    where = "WHERE status IN ('in_progress', 'review') AND phase IN ('implementation', 'qa', 'review')"
    params: list[Any] = []
    if project_id:
        where += " AND project_id=?"
        params.append(project_id)
    rows = conn.execute(
        f"SELECT * FROM factory_tasks {where} ORDER BY priority",
        params,
    ).fetchall()
    result = [_row_to_dict(row) for row in rows]
    if own_conn:
        conn.close()
    return result


def list_pending_gates(
    *,
    project_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Find tasks that are 'done' but lack passed gates for required gate types."""
    own_conn = conn is None
    conn = conn or ensure_db()
    where = "WHERE t.status='done' AND NOT EXISTS (SELECT 1 FROM factory_gates g WHERE g.task_id=t.task_id AND g.status='passed')"
    params: list[Any] = []
    if project_id:
        where += " AND t.project_id=?"
        params.append(project_id)
    rows = conn.execute(
        f"SELECT t.* FROM factory_tasks t {where} LIMIT 50",
        params,
    ).fetchall()
    result = [_row_to_dict(row) for row in rows]
    if own_conn:
        conn.close()
    return result


def list_blocked_tasks(
    *,
    project_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Find tasks explicitly blocked for more than 15 minutes."""
    own_conn = conn is None
    conn = conn or ensure_db()
    where = "WHERE status='blocked'"
    params: list[Any] = []
    if project_id:
        where += " AND project_id=?"
        params.append(project_id)
    rows = conn.execute(
        f"SELECT * FROM factory_tasks {where} ORDER BY updated_at",
        params,
    ).fetchall()
    result = [_row_to_dict(row) for row in rows]
    if own_conn:
        conn.close()
    return result


def list_tasks_without_evidence(
    *,
    project_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Find completed tasks that still lack evidence."""
    own_conn = conn is None
    conn = conn or ensure_db()
    where = "WHERE status='done' AND evidence_status='missing' AND evidence_required=1"
    params: list[Any] = []
    if project_id:
        where += " AND project_id=?"
        params.append(project_id)
    rows = conn.execute(
        f"SELECT * FROM factory_tasks {where} ORDER BY finished_at",
        params,
    ).fetchall()
    result = [_row_to_dict(row) for row in rows]
    if own_conn:
        conn.close()
    return result


def list_tasks_dependency_ready(
    *,
    project_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Find tasks with status='todo' whose dependencies are all done."""
    own_conn = conn is None
    conn = conn or ensure_db()
    where = "WHERE t.status='todo' AND t.dependencies_json != '[]'"
    params: list[Any] = []
    if project_id:
        where += " AND t.project_id=?"
        params.append(project_id)
    rows = conn.execute(
        f"SELECT t.* FROM factory_tasks t {where}",
        params,
    ).fetchall()
    unblocked: list[dict[str, Any]] = []
    for row in rows:
        task = _row_to_dict(row)
        deps: list[str] = task.get("dependencies") or []
        if not deps:
            continue
        placeholders = ",".join("?" for _ in deps)
        done = conn.execute(
            f"SELECT COUNT(*) FROM factory_tasks WHERE task_id IN ({placeholders}) AND status='done'",
            deps,
        ).fetchone()[0]
        if done == len(deps):
            unblocked.append(task)
    if own_conn:
        conn.close()
    return unblocked


def list_project_summary(
    project_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Aggregate summary for a single project: counts per status, engine, gate."""
    own_conn = conn is None
    conn = conn or ensure_db()
    project = conn.execute(
        "SELECT * FROM factory_projects WHERE project_id=?", (project_id,)
    ).fetchone()
    if not project:
        if own_conn:
            conn.close()
        return {"error": f"project {project_id} not found"}
    project_data = _row_to_dict(project)
    total_tasks = conn.execute(
        "SELECT COUNT(*) FROM factory_tasks WHERE project_id=?", (project_id,)
    ).fetchone()[0]
    by_status = {
        row["status"]: row["cnt"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM factory_tasks WHERE project_id=? GROUP BY status",
            (project_id,),
        ).fetchall()
    }
    by_engine = {
        row["engine"]: row["cnt"]
        for row in conn.execute(
            "SELECT engine, COUNT(*) AS cnt FROM factory_tasks WHERE project_id=? GROUP BY engine",
            (project_id,),
        ).fetchall()
    }
    gates_by_type = {
        row["gate_type"]: row["cnt"]
        for row in conn.execute(
            "SELECT gate_type, COUNT(*) AS cnt FROM factory_gates WHERE project_id=? GROUP BY gate_type",
            (project_id,),
        ).fetchall()
    }
    recent_events = [
        _row_to_dict(row)
        for row in conn.execute(
            "SELECT * FROM factory_events WHERE project_id=? ORDER BY created_at DESC LIMIT 10",
            (project_id,),
        ).fetchall()
    ]
    if own_conn:
        conn.close()
    return {
        **project_data,
        "total_tasks": total_tasks,
        "by_status": by_status,
        "by_engine": by_engine,
        "gates_by_type": gates_by_type,
        "recent_events": recent_events,
    }


def list_all_projects_summary(
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict[str, Any]]:
    """Return summary for every project in the DB."""
    own_conn = conn is None
    conn = conn or ensure_db()
    projects = conn.execute(
        "SELECT project_id FROM factory_projects ORDER BY updated_at DESC"
    ).fetchall()
    result = [list_project_summary(row["project_id"], conn=conn) for row in projects]
    if own_conn:
        conn.close()
    return result
