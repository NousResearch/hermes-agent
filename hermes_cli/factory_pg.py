"""PostgreSQL-backed Factory progress API for Agent Core DB."""
from __future__ import annotations

from typing import Any, Optional

from hermes_cli import agent_core_sql as sql
from hermes_cli.factory_db import DEFAULT_LANES, FACTORY_AGENTS, VALID_METHODS, slugify


def available() -> bool:
    # Pytest suites intentionally exercise the stdlib SQLite fallback with a
    # temp HERMES_HOME. The developer shell may still carry Agent Core env vars,
    # so keep tests isolated unless a test explicitly opts into Postgres.
    import os
    if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("HERMES_FACTORY_USE_POSTGRES"):
        return False
    if not sql.enabled():
        return False
    try:
        sql.psql("SELECT 1;", user=sql.runtime_env().get("FACTORY_DB_RUNTIME_USER", "factory_runtime"))
        return True
    except Exception:
        return False


def _user() -> str:
    return sql.runtime_env().get("FACTORY_DB_RUNTIME_USER", "factory_runtime")


def _q(v: Any) -> str:
    return sql.quote_literal(v)


def _j(v: Any) -> str:
    return sql.quote_jsonb(v)


def seed_agents() -> None:
    values = []
    for agent_id, display_name, role, engine, toolsets, skills, greenlights in FACTORY_AGENTS:
        values.append(
            f"({_q(agent_id)}, {_q(agent_id)}, {_q(display_name)}, {_q(role)}, {_q(engine)}, {_j({'toolsets': toolsets, 'skills': skills, 'greenlight_required': greenlights})})"
        )
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
        """,
        user=_user(),
    )


def create_project(name: str, *, project_id: Optional[str] = None, repo_path: Optional[str] = None, repo_remote: Optional[str] = None, base_branch: Optional[str] = None, human_owner: Optional[str] = None, summary: Optional[str] = None, risk_level: str = "medium", autonomy_level: int = 3, methodology: str = "dual_lane", create_default_lanes: bool = True, metadata: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
    seed_agents()
    pid = project_id or slugify(name)
    sql.psql(f"""
        INSERT INTO factory.projects (project_id, name, repo_path, repo_remote, base_branch, status, autonomy_level, methodology, risk_level, human_owner, summary, metadata, started_at, updated_at)
        VALUES ({_q(pid)}, {_q(name)}, {_q(repo_path)}, {_q(repo_remote)}, {_q(base_branch)}, 'intake', {int(autonomy_level)}, {_q(methodology)}, {_q(risk_level)}, {_q(human_owner)}, {_q(summary)}, {_j(metadata or {})}, now(), now())
        ON CONFLICT (project_id) DO UPDATE SET
          name=EXCLUDED.name, repo_path=EXCLUDED.repo_path, repo_remote=EXCLUDED.repo_remote, base_branch=EXCLUDED.base_branch,
          autonomy_level=EXCLUDED.autonomy_level, methodology=EXCLUDED.methodology, risk_level=EXCLUDED.risk_level,
          human_owner=EXCLUDED.human_owner, summary=EXCLUDED.summary, metadata=EXCLUDED.metadata, updated_at=now();
        INSERT INTO factory.events(project_id, actor, event_type, message, metadata)
        VALUES ({_q(pid)}, 'factory-orchestrator', 'project_created', {_q(f'Factory project {pid} initialized')}, {_j({'methodology': methodology})});
    """, user=_user())
    lanes = []
    if create_default_lanes:
        for _suffix, lane_name, method in DEFAULT_LANES:
            lanes.append(create_lane(pid, lane_name, method))
    return {"project_id": pid, "lanes": lanes}


def create_lane(project_id: str, name: str, methodology: str, *, lane_id: Optional[str] = None, kanban_board: Optional[str] = None, branch: Optional[str] = None, worktree_path: Optional[str] = None, metadata: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
    if methodology not in VALID_METHODS:
        raise ValueError(f"unknown methodology {methodology!r}; expected one of {sorted(VALID_METHODS)}")
    suffix = "zeus" if methodology == "zeus_native" else "bmad" if methodology == "bmad_hybrid" else slugify(methodology)
    lid = lane_id or f"{project_id}-{suffix}"
    board = kanban_board or lid
    branch_value = branch or f"factory/{project_id}/{suffix}"
    meta = {**(metadata or {}), "kanban_board": board}
    sql.psql(f"""
        INSERT INTO factory.lanes (lane_id, project_id, name, methodology, branch, worktree_path, status, metadata, created_at, updated_at)
        VALUES ({_q(lid)}, {_q(project_id)}, {_q(name)}, {_q(methodology)}, {_q(branch_value)}, {_q(worktree_path)}, 'planned', {_j(meta)}, now(), now())
        ON CONFLICT (lane_id) DO UPDATE SET name=EXCLUDED.name, methodology=EXCLUDED.methodology, branch=EXCLUDED.branch, worktree_path=EXCLUDED.worktree_path, metadata=EXCLUDED.metadata, updated_at=now();
        INSERT INTO factory.events(project_id, lane_id, actor, event_type, message, metadata)
        VALUES ({_q(project_id)}, {_q(lid)}, 'factory-orchestrator', 'lane_created', {_q(f'Lane {lid} initialized')}, {_j({'methodology': methodology, 'kanban_board': board, 'branch': branch_value})});
    """, user=_user())
    return {"lane_id": lid, "project_id": project_id, "methodology": methodology, "kanban_board": board, "branch": branch_value}


def create_task(project_id: str, title: str, *, lane_id: Optional[str] = None, description: Optional[str] = None, phase: str = "planning", status: str = "todo", owner_agent_id: Optional[str] = None, reviewer_agent_id: Optional[str] = None, engine: str = "zeus", priority: int = 100, acceptance_criteria: Optional[list[str]] = None, dependencies: Optional[list[str]] = None, metadata: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
    base = f"{project_id}-{slugify(title)[:40]}"
    existing = sql.rows(f"SELECT task_id FROM factory.tasks WHERE task_id LIKE {_q(base + '%')} ORDER BY task_id", user=_user())
    taken = {r['task_id'] for r in existing}
    task_id = base
    n = 2
    while task_id in taken:
        task_id = f"{base}-{n}"; n += 1
    sql.psql(f"""
        INSERT INTO factory.tasks (task_id, project_id, lane_id, title, description, phase, status, owner_profile, reviewer_profile, engine, priority, dependencies, acceptance_criteria, evidence_required, evidence_status, risk_level, metadata, created_at, updated_at)
        VALUES ({_q(task_id)}, {_q(project_id)}, {_q(lane_id)}, {_q(title)}, {_q(description)}, {_q(phase)}, {_q(status)}, {_q(owner_agent_id)}, {_q(reviewer_agent_id)}, {_q(engine)}, {int(priority)}, {_j(dependencies or [])}, {_j(acceptance_criteria or [])}, true, 'missing', 'medium', {_j(metadata or {})}, now(), now());
        INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata)
        VALUES ({_q(project_id)}, {_q(lane_id)}, {_q(task_id)}, 'factory-orchestrator', 'task_created', {_q(f'Task {task_id} created')}, {_j({'engine': engine, 'owner': owner_agent_id})});
    """, user=_user())
    return {"task_id": task_id, "project_id": project_id, "lane_id": lane_id}


def record_gate(project_id: str, gate_type: str, status: str, *, lane_id: Optional[str] = None, task_id: Optional[str] = None, reviewer: Optional[str] = None, evidence: Optional[dict[str, Any]] = None, notes: Optional[str] = None, **_: Any) -> dict[str, Any]:
    row = sql.statement_one(f"""
      INSERT INTO factory.gates (project_id, lane_id, task_id, gate_type, status, reviewer, evidence, notes, timestamp)
      VALUES ({_q(project_id)}, {_q(lane_id)}, {_q(task_id)}, {_q(gate_type)}, {_q(status)}, {_q(reviewer)}, {_j(evidence or {})}, {_q(notes)}, now())
      RETURNING gate_id, project_id, status
    """, user=_user())
    gate_id = row["gate_id"] if row else None
    sql.psql(f"INSERT INTO factory.events(project_id, lane_id, task_id, actor, event_type, message, metadata) VALUES ({_q(project_id)}, {_q(lane_id)}, {_q(task_id)}, {_q(reviewer or 'factory-orchestrator')}, {_q('gate_' + status)}, {_q(f'{gate_type} gate {status}')}, {_j({'gate_id': gate_id})});", user=_user())
    return {"gate_id": gate_id, "project_id": project_id, "status": status}


def status(project_id: Optional[str] = None, **_: Any) -> dict[str, Any]:
    where = "" if not project_id else f"WHERE project_id={_q(project_id)}"
    filter_expr = "TRUE" if not project_id else f"project_id={_q(project_id)}"
    projects = sql.rows(f"SELECT * FROM factory.projects {where} ORDER BY updated_at DESC", user=_user())
    lanes = sql.rows(f"SELECT *, metadata->>'kanban_board' AS kanban_board FROM factory.lanes WHERE {filter_expr} ORDER BY project_id, lane_id", user=_user())
    tasks = sql.rows(f"SELECT * FROM factory.tasks WHERE {filter_expr} ORDER BY project_id, priority, created_at", user=_user())
    gates = sql.rows(f"SELECT * FROM factory.gates WHERE {filter_expr} ORDER BY timestamp DESC LIMIT 50", user=_user())
    return {"db_backend": "agent_core_postgres", "database": sql.runtime_env().get("AGENT_DB_NAME", "zeus_agent"), "projects": projects, "lanes": lanes, "tasks": tasks, "gates": gates}
