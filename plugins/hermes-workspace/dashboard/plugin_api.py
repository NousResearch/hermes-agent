"""Hermes Workspace dashboard plugin API.

Mounted at /api/plugins/hermes-workspace/.
This is a local dashboard helper: it reads non-secret runtime metadata,
summarises profiles/plugins/Kanban, and can create supervised Kanban tasks.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hermes_constants import get_hermes_home
from hermes_cli import kanban_db
from hermes_cli.config import cfg_get, load_config
from hermes_cli.profiles import list_profiles, normalize_profile_name

router = APIRouter()

DEFAULT_ACTIVE_PROFILE_HINTS = [
    "business",
    "cyber",
    "docs",
    "finance",
    "freelance",
    "growth",
    "jobsearch",
    "legal",
    "location",
    "meeting",
    "ops",
    "personal",
    "pilot",
    "pricing",
    "privacy",
    "research",
    "trading",
    "veille",
]

WORKSPACE_PATHS = {
    "scratch": {"label": "Scratch", "kind": "scratch", "path": None},
    "hermes_runtime": {
        "label": "Hermes runtime",
        "kind": "dir",
        "path": str(get_hermes_home() / "hermes-agent"),
    },
    "workspace_v2": {
        "label": "Hermes Workspace V2",
        "kind": "dir",
        "path": str(Path.home() / "Documents" / "hermes-agent-workspace-v2"),
    },
}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _load_workspace_catalog(installed_profile_names: set[str]) -> dict[str, Any]:
    """Load optional local roster metadata without hardcoding private data.

    Users can create ``~/.hermes/workspace_agents.json`` with:
    {
      "active_profiles": ["ops", "pilot"],
      "catalog_label": "Personal Agents",
      "catalog_agents": ["orchestrator", "research"]
    }
    """
    default_active = [name for name in DEFAULT_ACTIVE_PROFILE_HINTS if name in installed_profile_names]
    if not default_active:
        default_active = sorted(name for name in installed_profile_names if name != "default")

    catalog = {
        "active_profiles": default_active,
        "catalog_label": "Personal Agents",
        "catalog_agents": [],
    }
    path = get_hermes_home() / "workspace_agents.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return catalog
    if not isinstance(raw, dict):
        return catalog

    active_profiles = _string_list(raw.get("active_profiles"))
    catalog_agents = _string_list(raw.get("catalog_agents"))
    catalog_label = str(raw.get("catalog_label") or "").strip()

    if active_profiles:
        catalog["active_profiles"] = active_profiles
    if catalog_agents:
        catalog["catalog_agents"] = catalog_agents
    if catalog_label:
        catalog["catalog_label"] = catalog_label
    return catalog


class QuickTask(BaseModel):
    title: str = Field(..., min_length=1, max_length=240)
    body: Optional[str] = None
    assignee: Optional[str] = None
    priority: int = Field(default=5, ge=-100, le=100)
    triage: bool = False
    workspace: str = "scratch"
    skills: list[str] = Field(default_factory=list)


class BlueprintLaunch(BaseModel):
    workspace: str = "workspace_v2"
    priority: int = Field(default=8, ge=-100, le=100)
    triage: bool = False


def _safe_count_dir(path: Path, pattern: str) -> int:
    try:
        if not path.is_dir():
            return 0
        return sum(1 for _ in path.rglob(pattern))
    except Exception:
        return 0


def _workspace_ref(key: str) -> tuple[str, Optional[str], str]:
    ref = WORKSPACE_PATHS.get(key) or WORKSPACE_PATHS["scratch"]
    kind = ref["kind"]
    path = ref["path"]
    if kind == "dir" and path and not Path(path).exists():
        return "scratch", None, "scratch"
    return kind, path, key


def _task_dict(task: kanban_db.Task) -> dict[str, Any]:
    data = asdict(task)
    for key in ("body", "result", "claim_lock", "last_failure_error"):
        value = data.get(key)
        if isinstance(value, str) and len(value) > 220:
            data[key] = value[:220] + "..."
    return data


def _board_counts(board: str) -> dict[str, int]:
    try:
        kanban_db.init_db(board=board)
        with kanban_db.connect(board=board) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
            ).fetchall()
        return {str(r["status"]): int(r["n"]) for r in rows}
    except Exception:
        return {}


def _recent_tasks(board: str, limit: int = 8) -> list[dict[str, Any]]:
    try:
        kanban_db.init_db(board=board)
        with kanban_db.connect(board=board) as conn:
            tasks = kanban_db.list_tasks(conn, include_archived=False, limit=limit)
        return [_task_dict(t) for t in tasks]
    except Exception:
        return []


def _discover_dashboard_plugins() -> list[dict[str, Any]]:
    try:
        from hermes_cli import web_server

        plugins = web_server._get_dashboard_plugins(force_rescan=True)
        return [
            {k: v for k, v in p.items() if not k.startswith("_")}
            for p in plugins
        ]
    except Exception:
        return []


def _discover_agent_plugins() -> dict[str, Any]:
    try:
        from hermes_cli.plugins_cmd import (
            _discover_all_plugins,
            _get_current_context_engine,
            _get_current_memory_provider,
            _get_disabled_set,
            _get_enabled_set,
        )
    except Exception:
        return {
            "plugins": [],
            "providers": {"memory_provider": "", "context_engine": ""},
        }

    try:
        enabled = _get_enabled_set()
        disabled = _get_disabled_set()
    except Exception:
        enabled = set()
        disabled = set()

    rows: list[dict[str, Any]] = []
    for name, version, description, source, path in _discover_all_plugins():
        if name in disabled:
            status = "disabled"
        elif name in enabled:
            status = "enabled"
        else:
            status = "inactive"
        rows.append(
            {
                "name": name,
                "version": version,
                "description": description,
                "source": source,
                "status": status,
                "path": str(path),
            }
        )

    try:
        memory_provider = _get_current_memory_provider() or ""
    except Exception:
        memory_provider = ""
    try:
        context_engine = _get_current_context_engine() or ""
    except Exception:
        context_engine = ""

    return {
        "plugins": rows,
        "providers": {
            "memory_provider": memory_provider,
            "context_engine": context_engine,
        },
    }


def _git_summary(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"exists": path.exists(), "path": str(path)}
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception as exc:
        return {"exists": path.exists(), "path": str(path), "error": str(exc)}
    return {"exists": True, "path": str(path), "status": status, "head": head}


def _profile_row(profile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "path": str(profile.path),
        "is_default": bool(profile.is_default),
        "gateway_running": bool(profile.gateway_running),
        "model": profile.model or "",
        "provider": profile.provider or "",
        "has_env": bool(profile.has_env),
        "skill_count": int(profile.skill_count or 0),
        "alias": str(profile.alias_path) if profile.alias_path else "",
        "distribution": profile.distribution_name or "",
    }


def _blueprints() -> list[dict[str, Any]]:
    return [
        {
            "key": "workspace-v2-full",
            "label": "Workspace V2 full build",
            "profile": "pilot",
            "description": "Decoupe GUI, agents, memory, skills, debug and multi-agent flow.",
            "tasks": [
                ["pilot", "Orchestrer Workspace V2", "Decouper l'interface en vues agents, memoire, skills, debug et kanban."],
                ["ops", "Brancher debug runtime", "Verifier gateway, logs, dashboard, services launchd et workflow de restart."],
                ["docs", "Documenter l'utilisation Workspace V2", "Produire un guide court: dashboard, Kanban, Electron, agents et commandes."],
                ["research", "Cartographier plugins et providers", "Lister plugins utiles, providers memoire/contexte et opportunites d'extension."],
                ["cyber", "Revue securite locale", "Verifier exposition API plugin, donnees sensibles, hosts, logs et secrets."],
            ],
        },
        {
            "key": "kanban-orchestration",
            "label": "Kanban multi-agent",
            "profile": "pilot",
            "description": "Installe une boucle orchestrateur + workers specialises sur le board actif.",
            "tasks": [
                ["pilot", "Design workflow orchestrateur", "Definir colonnes, criteres ready/done, dependances et conventions de handoff."],
                ["ops", "Verifier dispatcher Kanban", "Controler gateway, DB board actif, reclaim/reassign, logs et diagnostics."],
                ["docs", "Guide worker Kanban", "Documenter create/list/watch/show/log/reassign/specify avec exemples."],
                ["research", "Backlog plugins agentiques", "Identifier les plugins/outils a brancher par profil."],
            ],
        },
        {
            "key": "personal-agent-grid",
            "label": "Personal agent grid",
            "profile": "personal",
            "description": "Aligns catalogued personal agents, active profiles and supervised usage.",
            "tasks": [
                ["personal", "Map personal agent catalogue", "Relier les agents catalogues aux profils Hermes actifs et aux dossiers locaux."],
                ["growth", "Agent business opportunities", "Transformer les agents business/cyber/growth en workflows monetisables."],
                ["privacy", "Personal data guardrails", "Definir ce qui reste local, ce qui ne doit jamais etre pousse, et les validations."],
                ["ops", "Operations routines", "Preparer commandes de run, logs, scorecards, cron et reprise d'erreur."],
            ],
        },
        {
            "key": "debug-sweep",
            "label": "Debug sweep",
            "profile": "ops",
            "description": "Passe rapide pour incidents dashboard, chat, plugins, gateway, Kanban.",
            "tasks": [
                ["ops", "Audit dashboard/chat", "Verifier /chat, /workspace, /kanban, plugins charges, console et endpoints API."],
                ["cyber", "Audit exposition locale", "Verifier localhost, routes plugins, secrets non exposes et logs."],
                ["docs", "Runbook debug", "Ecrire les commandes de restart, status, logs, curl et diagnostics."],
            ],
        },
    ]


@router.get("/summary")
async def summary():
    config = load_config()
    hermes_home = get_hermes_home()
    active_board = kanban_db.get_current_board() or kanban_db.DEFAULT_BOARD

    try:
        boards = kanban_db.list_boards(include_archived=False)
    except Exception:
        boards = []

    board_rows = []
    for board in boards:
        slug = board.get("slug") or kanban_db.DEFAULT_BOARD
        counts = _board_counts(slug)
        board_rows.append({**board, "counts": counts, "total": sum(counts.values())})

    profiles = [_profile_row(p) for p in list_profiles()]
    profile_names = {p["name"] for p in profiles}
    roster = _load_workspace_catalog(profile_names)
    active_profiles = [
        {"name": name, "installed": name in profile_names}
        for name in roster["active_profiles"]
    ]
    catalog_agents = [
        {
            "name": name,
            "profile_match": name if name in profile_names else "",
            "installed_profile": name in profile_names,
        }
        for name in roster["catalog_agents"]
    ]

    dashboard_plugins = _discover_dashboard_plugins()
    agent_plugins = _discover_agent_plugins()

    workspace_v2 = Path(WORKSPACE_PATHS["workspace_v2"]["path"] or "")
    runtime_repo = Path(WORKSPACE_PATHS["hermes_runtime"]["path"] or "")

    gateway_state_path = hermes_home / "gateway_state.json"
    gateway_state = {}
    try:
        gateway_state = json.loads(gateway_state_path.read_text(encoding="utf-8"))
    except Exception:
        gateway_state = {}

    memory_provider = cfg_get(config, "memory", "provider", default="") or agent_plugins["providers"].get("memory_provider", "")
    context_engine = cfg_get(config, "context_engine", "provider", default="") or agent_plugins["providers"].get("context_engine", "")

    return {
        "generated_at": int(time.time()),
        "hermes_home": str(hermes_home),
        "runtime": _git_summary(runtime_repo),
        "workspace_v2": _git_summary(workspace_v2),
        "memory": {
            "provider": memory_provider,
            "context_engine": context_engine,
            "default_memory_files": {
                "memory_md": (hermes_home / "memories" / "MEMORY.md").exists(),
                "user_md": (hermes_home / "memories" / "USER.md").exists(),
            },
        },
        "gateway": {
            "pid_file": str(hermes_home / "gateway.pid"),
            "state": gateway_state,
        },
        "profiles": profiles,
        "active_profiles": active_profiles,
        "catalog_label": roster["catalog_label"],
        "catalog_agents": catalog_agents,
        "skills": {
            "total_profile_skills": sum(p["skill_count"] for p in profiles),
            "runtime_skills": _safe_count_dir(runtime_repo / "skills", "SKILL.md"),
        },
        "plugins": {
            "dashboard": dashboard_plugins,
            "agent": agent_plugins["plugins"],
            "providers": agent_plugins["providers"],
        },
        "kanban": {
            "active_board": active_board,
            "boards": board_rows,
            "recent_tasks": _recent_tasks(active_board),
        },
        "workspaces": WORKSPACE_PATHS,
        "blueprints": _blueprints(),
    }


@router.get("/blueprints")
async def blueprints():
    return {"blueprints": _blueprints()}


@router.post("/tasks")
async def create_task(body: QuickTask):
    kind, path, workspace_key = _workspace_ref(body.workspace)
    assignee = normalize_profile_name(body.assignee) if body.assignee else None
    try:
        kanban_db.init_db()
        with kanban_db.connect() as conn:
            task_id = kanban_db.create_task(
                conn,
                title=body.title,
                body=body.body,
                assignee=assignee,
                created_by="dashboard:hermes-workspace",
                workspace_kind=kind,
                workspace_path=path,
                priority=body.priority,
                triage=body.triage,
                skills=body.skills or None,
            )
            task = kanban_db.get_task(conn, task_id)
    except (ValueError, sqlite3.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "task_id": task_id, "workspace": workspace_key, "task": _task_dict(task) if task else None}


@router.post("/blueprints/{key}/launch")
async def launch_blueprint(key: str, body: BlueprintLaunch):
    blueprint = next((bp for bp in _blueprints() if bp["key"] == key), None)
    if not blueprint:
        raise HTTPException(status_code=404, detail="Unknown blueprint.")

    kind, path, workspace_key = _workspace_ref(body.workspace)
    created: list[dict[str, Any]] = []
    try:
        kanban_db.init_db()
        with kanban_db.connect() as conn:
            for index, (assignee, title, task_body) in enumerate(blueprint["tasks"]):
                task_id = kanban_db.create_task(
                    conn,
                    title=title,
                    body=f"{task_body}\n\nBlueprint: {blueprint['label']}",
                    assignee=assignee,
                    created_by="dashboard:hermes-workspace",
                    workspace_kind=kind,
                    workspace_path=path,
                    priority=body.priority - index,
                    triage=body.triage,
                    skills=["kanban-worker"],
                    idempotency_key=f"hermes-workspace:{key}:{workspace_key}:{assignee}:{title}",
                )
                task = kanban_db.get_task(conn, task_id)
                created.append({"task_id": task_id, "task": _task_dict(task) if task else None})
    except (ValueError, sqlite3.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "blueprint": blueprint,
        "workspace": workspace_key,
        "created": created,
    }
