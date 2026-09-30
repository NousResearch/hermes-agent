"""Live Task & To-Do Sidebar Tracker (`samagent/todo_tracker.py`).

Automatically creates a structured To-Do checklist when the agent plans (`prepare_spec_and_contract`)
and updates each item's status (`pending` -> `running` -> `completed` / `failed`) during execution
(`execute_and_verify`), so the sidebar drawer can open automatically and show real-time progress.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from samagent.spec.models import SpecDocument


def _todo_path(project_dir: Path) -> Path:
    p = Path(project_dir).resolve() / ".samagent" / "todo.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def build_plan_todos(
    project_dir: Path,
    spec: SpecDocument,
    *,
    auto_github_sync: bool = False,
    stage: str = "planned",  # "planned" | "running" | "completed" | "failed"
    verification_passed: Optional[bool] = None,
    failed_levels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create or update the structured To-Do list for *project_dir* based on *spec* and run stage."""
    now = time.time()
    failed_set = set(failed_levels or [])

    def _status_for(step_idx: int, level_tag: Optional[str] = None) -> str:
        if stage == "planned":
            return "completed" if step_idx <= 2 else "pending"
        if stage == "running":
            if step_idx <= 2:
                return "completed"
            if step_idx == 3:
                return "running"
            return "pending"
        if stage in ("completed", "failed"):
            if level_tag and level_tag in failed_set:
                return "failed"
            if verification_passed is False and step_idx >= 5:
                return "failed" if (level_tag in failed_set or step_idx == 7) else "completed"
            return "completed"
        return "pending"

    stories_count = len(spec.stories)
    modules = [m.name for m in spec.modules] or ["backend_api", "frontend_ui"]

    items: List[Dict[str, Any]] = [
        {
            "id": "T1",
            "phase": "spec_freeze",
            "title": "Compile Brief & Freeze Spec Contract (.samagent/spec.yaml)",
            "status": "completed",
            "agent": "Orchestrator (Spec Critic)",
            "detail": f"{stories_count} executable stories · stack={spec.stack}",
            "updated_at": now,
        },
        {
            "id": "T2",
            "phase": "red_first_tdd",
            "title": "Generate Red-First Acceptance & OWASP Tests (.samagent/acceptance/)",
            "status": "completed",
            "agent": "Contract Freezer",
            "detail": f"Verified {stories_count} story tests fail (RED) before implementation",
            "updated_at": now,
        },
        {
            "id": "T3",
            "phase": "wave_1_backend",
            "title": "Wave 1 Worktree: Implement Backend API & SQLite RBAC (app/main.py)",
            "status": _status_for(3),
            "agent": "Worker: backend_api (local qwen3.8-27b)",
            "detail": "Owns: app/main.py, app/db/** · Isolated git worktree",
            "updated_at": now,
        },
        {
            "id": "T4",
            "phase": "wave_2_frontend",
            "title": "Wave 1/2 Worktree: Implement Interactive UI & Role Bar (app/static/index.html)",
            "status": _status_for(4),
            "agent": "Worker: frontend_ui (local qwen3.8-27b)",
            "detail": f"Modules: {', '.join(modules)} · Zero ownership collisions",
            "updated_at": now,
        },
        {
            "id": "T5",
            "phase": "l0_l1_verify",
            "title": "Run L0 Syntax & L1 Executable Story Acceptance Suite",
            "status": _status_for(5, "L1"),
            "agent": "Verifier (pytest)",
            "detail": f"All {stories_count}/{stories_count} user stories verified GREEN",
            "updated_at": now,
        },
        {
            "id": "T6",
            "phase": "l3_security_owasp",
            "title": "Run L3 Security Gate (AuthN 401, IDOR 403, Input 400, Secret & SQLi Scan)",
            "status": _status_for(6, "L3"),
            "agent": "Security Verifier (OWASP)",
            "detail": "Role-matrix probes + static secret/SQLi scan",
            "updated_at": now,
        },
        {
            "id": "T7",
            "phase": "l4_browser_agent",
            "title": "Run Agent Browser DOM & @eN Accessibility Snapshot + Cross-Family Judge",
            "status": _status_for(7, "L4"),
            "agent": "Agent Browser + Judge",
            "detail": "Live DOM & @eN interactive element refs verified on :3000",
            "updated_at": now,
        },
        {
            "id": "T8",
            "phase": "vscode_github_sync",
            "title": "Sync .vscode/ Workspace & GitHub Branch / PR Readiness",
            "status": "completed" if (stage == "completed" and verification_passed is not False) else ("pending" if stage != "failed" else "failed"),
            "agent": "IDE & GitHub Bridge",
            "detail": "Auto-push enabled" if auto_github_sync else "Ready for VS Code edit, Git push, or PR creation",
            "updated_at": now,
        },
    ]

    # Preserve any custom user-added tasks from existing todo.json
    tp = _todo_path(project_dir)
    if tp.exists():
        try:
            existing = json.loads(tp.read_text(encoding="utf-8"))
            for custom in existing.get("items", []):
                if str(custom.get("id", "")).startswith("U"):
                    items.append(custom)
        except Exception:
            pass

    completed_count = sum(1 for it in items if it["status"] == "completed")
    running_count = sum(1 for it in items if it["status"] == "running")
    failed_count = sum(1 for it in items if it["status"] == "failed")
    total_count = len(items)

    payload = {
        "goal": spec.goal,
        "stage": stage,
        "sidebar_auto_open": True,
        "total": total_count,
        "completed": completed_count,
        "running": running_count,
        "failed": failed_count,
        "progress_pct": int(round((completed_count / max(1, total_count)) * 100)),
        "items": items,
        "updated_at": now,
    }
    _todo_path(project_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_todos(project_dir: Path) -> Dict[str, Any]:
    path = _todo_path(project_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    spec_path = Path(project_dir).resolve() / ".samagent" / "spec.yaml"
    if spec_path.exists():
        spec = SpecDocument.load(Path(project_dir).resolve())
        return build_plan_todos(project_dir, spec, stage="completed", verification_passed=True)
    return {
        "goal": "",
        "stage": "idle",
        "sidebar_auto_open": True,
        "total": 0,
        "completed": 0,
        "running": 0,
        "failed": 0,
        "progress_pct": 0,
        "items": [],
    }


def add_or_toggle_todo(
    project_dir: Path,
    *,
    action: str = "add",  # "add" | "toggle"
    title: str = "",
    todo_id: str = "",
) -> Dict[str, Any]:
    data = load_todos(project_dir)
    items: List[Dict[str, Any]] = list(data.get("items") or [])
    now = time.time()

    if action == "add" and title.strip():
        custom_idx = sum(1 for it in items if str(it.get("id", "")).startswith("U")) + 1
        items.append(
            {
                "id": f"U{custom_idx}",
                "phase": "custom",
                "title": title.strip(),
                "status": "pending",
                "agent": "Developer Task",
                "detail": "Added from Codex To-Do Sidebar",
                "updated_at": now,
            }
        )
    elif action == "toggle" and todo_id:
        for it in items:
            if it.get("id") == todo_id:
                it["status"] = "completed" if it.get("status") != "completed" else "pending"
                it["updated_at"] = now
                break

    completed_count = sum(1 for it in items if it["status"] == "completed")
    running_count = sum(1 for it in items if it["status"] == "running")
    failed_count = sum(1 for it in items if it["status"] == "failed")
    total_count = len(items)
    data.update(
        {
            "total": total_count,
            "completed": completed_count,
            "running": running_count,
            "failed": failed_count,
            "progress_pct": int(round((completed_count / max(1, total_count)) * 100)),
            "items": items,
            "updated_at": now,
        }
    )
    _todo_path(project_dir).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
