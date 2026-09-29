"""SamAgent Mission Control Dashboard Plugin Backend — mounted at /api/plugins/samagent/.

Provides REST endpoints for the Local Installed Platform + VS Code Pre-Production Studio:
1. /state            — Spec, plan card, deliverable, IDE file tree, git diff, Pre-Prod Gate, and platform status
2. /interview        — Generate <=5 adaptive interview questions with defaults for a brief
3. /plan             — Freeze spec, contract, red-first acceptance tests, .vscode/ config, and Plan Card
4. /build            — Execute scaffold + isolated worktree workers + L0–L4 verification & security gates
5. /reverify         — Re-run L0–L4 + OWASP security verification after editing code in VS Code
6. /workspace/switch — Switch or create a persistent local project folder in ~/SamAgentProjects
7. /ide/open         — Launch VS Code (or return vscode://file/... deep links) for the workspace or file
8. /ide/file         — Read or save a workspace file (bidirectional sync with VS Code on disk)
9. /dev-app/action   — Interactive multi-role local dev testing (Visitor / Member Alice / Member Bob / Admin)
10. /platform/install — One-click installer for local OS desktop launcher, daemon & VS Code extension
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from samagent.acp_bridge import run_acp_samagent_command
from samagent.conductor.engine import SamAgentConductor, build_plan_card
from samagent.conductor.verify import VerificationRunner
from samagent.dev_server_manager import (
    get_dev_server_status,
    start_or_restart_dev_server,
    stop_dev_server,
)
from samagent.ide_bridge import (
    evaluate_pre_production_gate,
    generate_vscode_workspace_config,
    get_workspace_git_status_and_diff,
    list_workspace_files,
    open_in_vscode,
    read_workspace_file,
    write_workspace_file,
)
from samagent.ide_watcher import check_and_sync_external_edits
from samagent.ledger.store import ProjectLedger
from samagent.platform_installer import (
    get_default_projects_root,
    get_platform_install_status,
    install_os_desktop_platform,
)
from samagent.prod_bundler import list_production_releases, promote_to_production
from samagent.router.policy import TaskBoundaryRouter
from samagent.spec.interview import generate_interview_questions
from samagent.spec.models import SpecDocument

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEMO_DIR = Path(
    os.environ.get("SAMAGENT_WORKSPACE_DIR")
    or (get_default_projects_root() / "yoga-studio-local")
)
_ACTIVE_WORKSPACE: Dict[str, Optional[Path]] = {"path": None}


def _workspace_dir() -> Path:
    ws = _ACTIVE_WORKSPACE["path"] or _DEFAULT_DEMO_DIR
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _list_available_workspaces() -> List[Dict[str, Any]]:
    root = get_default_projects_root()
    items: List[Dict[str, Any]] = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir():
                has_spec = (d / ".samagent" / "spec.yaml").exists()
                items.append(
                    {
                        "name": d.name,
                        "path": str(d),
                        "has_spec": has_spec,
                        "active": d.resolve() == _workspace_dir().resolve(),
                    }
                )
    return items


def _ensure_seeded_workspace() -> Path:
    ws = _workspace_dir()
    if not (ws / ".samagent" / "spec.yaml").exists():
        cond = SamAgentConductor(ws, cloud_available=True)
        cond.prepare_spec_and_contract(
            "Booking site for my yoga studio where visitors see the schedule, members book classes (no double booking, private per member), and admins add classes."
        )
        cond.execute_and_verify(run_id="run_initial_demo", use_worktrees=True)
    generate_vscode_workspace_config(ws)
    return ws


def _load_live_dev_service(ws: Path):
    main_py = ws / "app" / "main.py"
    if not main_py.exists():
        return None
    mod_spec = importlib.util.spec_from_file_location("live_app_instance", main_py)
    if mod_spec is None or mod_spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod


def _get_live_dev_snapshot(ws: Path) -> Dict[str, Any]:
    try:
        mod = _load_live_dev_service(ws)
        if mod is None or not hasattr(mod, "SecureAppService"):
            return {"items": [], "bookings": []}
        svc = mod.SecureAppService(persistent=True)
        try:
            items_res = svc.list_items()
            rows = svc.conn.execute(
                "SELECT id, item_id, owner_id, status, created_at FROM bookings ORDER BY created_at DESC"
            ).fetchall()
            bookings = [dict(r) for r in rows]
            return {
                "items": items_res.get("items", []),
                "bookings": bookings,
                "db_path": str(svc.db_path),
            }
        finally:
            svc.conn.close()
    except Exception as exc:
        return {"items": [], "bookings": [], "error": str(exc)}


class InterviewRequest(BaseModel):
    brief: str = Field(..., min_length=3)


class PlanRequest(BaseModel):
    brief: str = Field(..., min_length=3)
    answers: Dict[str, str] = Field(default_factory=dict)
    router_policy: str = "default"
    autonomy: str = "milestones"
    max_usd: float = 6.0
    max_minutes: int = 45


class BuildRequest(BaseModel):
    autonomy: Optional[str] = None
    router_policy: Optional[str] = None


class WorkspaceSwitchRequest(BaseModel):
    workspace_name_or_path: str = Field(..., min_length=1)
    brief: Optional[str] = None


class IdeOpenRequest(BaseModel):
    rel_path: Optional[str] = None
    line: int = 1


class IdeFileWriteRequest(BaseModel):
    rel_path: str = Field(..., min_length=1)
    content: str
    auto_reverify: bool = True


class DevAppActionRequest(BaseModel):
    action: str = Field(..., description="list_items | create_booking | get_booking | create_item | reset_db")
    role: str = "visitor"
    user_id: Optional[str] = None
    item_id: str = "item_1"
    booking_id: str = ""
    title: str = ""
    description: str = ""


class SteerRequest(BaseModel):
    action: str = "steer"
    note: str = ""


class FactRequest(BaseModel):
    scope: str = "architecture"
    kind: str = "decision"
    text: str = Field(..., min_length=2)
    sensitivity: str = "public"
    supersede_id: Optional[str] = None


_RUN_CONTROL_STATE: Dict[str, Any] = {
    "mode": "running",
    "steer_notes": [],
}


def _load_contracts_preview(ws: Path) -> Dict[str, str]:
    cdir = ws / ".samagent" / "contract"
    out: Dict[str, str] = {}
    for rel in ("openapi.yaml", "db/schema.sql", "types.ts", "ownership.yaml", "version.json"):
        p = cdir / rel
        if p.exists():
            out[rel] = p.read_text(encoding="utf-8")
    return out


def _load_measurements() -> Dict[str, Any]:
    meas_dir = REPO_ROOT / "docs" / "samagent" / "measurements"
    out: Dict[str, Any] = {}
    for name in ("tool_footprint.json", "spikes_s1_s9.json", "sambench_v0_report.json", "ablation_h1_h7.json"):
        p = meas_dir / name
        if p.exists():
            try:
                out[name.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


def _latest_deliverable(ws: Path) -> Optional[Dict[str, Any]]:
    runs_dir = ws / ".samagent" / "runs"
    if runs_dir.exists():
        run_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        for rf in run_folders:
            deliv = rf / "deliverable.json"
            if deliv.exists():
                return json.loads(deliv.read_text(encoding="utf-8"))
    return None


@router.get("/state")
def get_mission_state() -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    ide_watcher = check_and_sync_external_edits(ws)
    spec = SpecDocument.load(ws)
    ledger = ProjectLedger(ws)
    rt = TaskBoundaryRouter(policy=spec.router_policy, cloud_available=True, ledger=ledger)
    plan_card = build_plan_card(ws, spec, rt)
    latest_deliverable = _latest_deliverable(ws)

    preview_html = ""
    idx_path = ws / "app" / "static" / "index.html"
    if idx_path.exists():
        preview_html = idx_path.read_text(encoding="utf-8")

    ide_files = list_workspace_files(ws)
    git_info = get_workspace_git_status_and_diff(ws)
    pre_prod_gate = evaluate_pre_production_gate(ws, latest_deliverable)

    return {
        "workspace": str(ws),
        "available_workspaces": _list_available_workspaces(),
        "spec": spec.to_dict(),
        "brief_markdown": spec.render_brief_markdown(),
        "plan_card": plan_card.to_dict(),
        "deliverable": latest_deliverable,
        "pre_prod_gate": pre_prod_gate,
        "ide": {
            "files": ide_files,
            "git": git_info,
            "watcher": ide_watcher,
            "vscode_workspace_uri": f"vscode://file{ws.resolve()}",
            "cursor_workspace_uri": f"cursor://file{ws.resolve()}",
            "code_workspace_file": str((ws / "project.code-workspace").resolve()),
        },
        "dev_server": get_dev_server_status(ws, port=3000),
        "dev_app": _get_live_dev_snapshot(ws),
        "releases": list_production_releases(ws),
        "platform_install": get_platform_install_status(),
        "control": _RUN_CONTROL_STATE,
        "contracts": _load_contracts_preview(ws),
        "preview_html": preview_html,
        "ledger": {
            "active_facts": [f.to_dict() for f in ledger.list_facts(active_only=True, include_private=True)],
            "superseded_facts": [f.to_dict() for f in ledger.list_facts(active_only=False, include_private=True) if f.valid_to is not None],
            "attempts": [a.to_dict() for a in ledger.list_attempts(limit=10)],
            "mirror_markdown": ledger.mirror_path.read_text(encoding="utf-8") if ledger.mirror_path.exists() else "",
        },
        "measurements": _load_measurements(),
    }


@router.post("/workspace/switch")
def switch_workspace(req: WorkspaceSwitchRequest) -> Dict[str, Any]:
    raw = req.workspace_name_or_path.strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        safe_slug = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in raw).strip("-") or "samagent-app"
        candidate = get_default_projects_root() / safe_slug
    candidate.mkdir(parents=True, exist_ok=True)
    _ACTIVE_WORKSPACE["path"] = candidate
    if req.brief and req.brief.strip():
        cond = SamAgentConductor(candidate, cloud_available=True)
        cond.prepare_spec_and_contract(req.brief.strip())
        cond.execute_and_verify(run_id=f"run_{int(time.time())}", use_worktrees=True)
    return get_mission_state()


@router.post("/ide/open")
def launch_ide(req: IdeOpenRequest) -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    return open_in_vscode(ws, req.rel_path, line=req.line)


@router.get("/ide/file")
def get_ide_file(path: str = Query(..., min_length=1)) -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    try:
        return read_workspace_file(ws, path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/ide/file")
def save_ide_file(req: IdeFileWriteRequest) -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    try:
        save_res = write_workspace_file(ws, req.rel_path, req.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if req.auto_reverify:
        _run_workspace_reverify(ws)
    state = get_mission_state()
    state["saved_file"] = save_res
    return state


def _run_workspace_reverify(ws: Path) -> Dict[str, Any]:
    spec = SpecDocument.load(ws)
    verifier = VerificationRunner(ws, spec)
    report = verifier.run_all()
    runs_dir = ws / ".samagent" / "runs" / f"run_reverify_{int(time.time())}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    prev = _latest_deliverable(ws) or {}
    deliverable = {
        **prev,
        "run_id": runs_dir.name,
        "goal": spec.goal,
        "stack": spec.stack,
        "verification": report.to_dict(),
    }
    (runs_dir / "deliverable.json").write_text(json.dumps(deliverable, indent=2), encoding="utf-8")
    return deliverable


@router.post("/reverify")
def reverify_workspace() -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    _run_workspace_reverify(ws)
    return get_mission_state()


@router.post("/promote-prod")
def promote_workspace_to_production() -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    _run_workspace_reverify(ws)
    promo = promote_to_production(ws)
    state = get_mission_state()
    state["promotion_result"] = promo
    return state


class DevServerControlRequest(BaseModel):
    action: str = "restart"  # "start" | "restart" | "stop"
    port: int = 3000


@router.post("/dev-server/control")
def control_local_dev_server(req: DevServerControlRequest) -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    act = req.action.strip().lower()
    if act == "stop":
        stop_dev_server()
    else:
        start_or_restart_dev_server(ws, port=req.port)
    return get_mission_state()


class AcpCommandRequest(BaseModel):
    command: str = "samagent-verify"
    args_text: str = ""


@router.post("/acp/command")
def execute_acp_command(req: AcpCommandRequest) -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    return run_acp_samagent_command(req.command, req.args_text, cwd=str(ws))


@router.post("/dev-app/action")
def execute_dev_app_action(req: DevAppActionRequest) -> Dict[str, Any]:
    """Run an interactive multi-role local dev action against the generated SQLite-backed app."""
    ws = _ensure_seeded_workspace()
    mod = _load_live_dev_service(ws)
    if mod is None or not hasattr(mod, "SecureAppService"):
        raise HTTPException(status_code=404, detail="Local dev app not built yet")
    svc = mod.SecureAppService(persistent=True)
    act = req.action.strip().lower()
    role = req.role.strip().lower()
    user_id = req.user_id
    if not user_id:
        if role == "admin":
            user_id = "u_admin"
        elif role == "member":
            user_id = "u_member_a"
        else:
            user_id = None

    try:
        if act == "list_items":
            res = svc.list_items()
        elif act == "create_booking":
            res = svc.create_booking(role=role, user_id=user_id, item_id=req.item_id)
        elif act == "get_booking":
            res = svc.get_booking(role=role, user_id=user_id, booking_id=req.booking_id)
        elif act == "create_item":
            res = svc.create_item(
                role=role,
                user_id=user_id,
                title=req.title,
                description=req.description or "Added from Local Pre-Production Studio",
            )
        elif act == "reset_db":
            svc.conn.close()
            if svc.db_path.exists():
                svc.db_path.unlink()
            svc = mod.SecureAppService(persistent=True)
            res = {"status": 200, "message": "Local dev SQLite database reset to clean seed state"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {act}")
    finally:
        try:
            svc.conn.close()
        except Exception:
            pass

    return {
        "action": act,
        "role": role,
        "user_id": user_id,
        "response": res,
        "dev_app": _get_live_dev_snapshot(ws),
    }


@router.post("/platform/install")
def install_local_platform() -> Dict[str, Any]:
    install_res = install_os_desktop_platform()
    state = get_mission_state()
    state["install_result"] = install_res
    return state


@router.post("/interview")
def create_interview(req: InterviewRequest) -> Dict[str, Any]:
    questions = [q.to_dict() for q in generate_interview_questions(req.brief)]
    return {"brief": req.brief, "questions": questions}


@router.post("/plan")
def create_plan(req: PlanRequest) -> Dict[str, Any]:
    ws = _workspace_dir()
    answers = dict(req.answers or {})
    if req.router_policy == "local_strict":
        answers["Q4_PRIVACY_ROUTING"] = "local_strict"
    if req.autonomy in ("plan_only", "milestones", "hands_off"):
        answers["Q5_AUTONOMY"] = req.autonomy

    cond = SamAgentConductor(ws, cloud_available=(req.router_policy != "local_strict"))
    cond.prepare_spec_and_contract(
        req.brief,
        answers=answers,
        max_usd=req.max_usd,
        max_minutes=req.max_minutes,
    )
    generate_vscode_workspace_config(ws)
    return get_mission_state()


@router.post("/build")
def run_build(req: BuildRequest) -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    spec = SpecDocument.load(ws)
    if req.autonomy in ("plan_only", "milestones", "hands_off"):
        spec.autonomy = req.autonomy
    if req.router_policy in ("default", "local_strict"):
        spec.router_policy = req.router_policy
    spec.save(ws)

    cond = SamAgentConductor(ws, cloud_available=(spec.router_policy != "local_strict"))
    _RUN_CONTROL_STATE["mode"] = "running"
    cond.execute_and_verify(run_id=f"run_{int(time.time())}", use_worktrees=True)
    generate_vscode_workspace_config(ws)
    return get_mission_state()


class StoryProbeRequest(BaseModel):
    story_id: str = "S1"
    method: str = "GET"
    route: str = "/api/items"
    role: str = "visitor"


@router.post("/probe-story")
def probe_live_story(req: StoryProbeRequest) -> Dict[str, Any]:
    """Execute a live story or security probe against the built application in the workspace."""
    ws = _ensure_seeded_workspace()
    mod = _load_live_dev_service(ws)
    if mod is None:
        raise HTTPException(status_code=404, detail="app/main.py not built yet")
    if req.story_id == "SECURITY_PROBES":
        return {"probe": "L3_SECURITY_MATRIX", "result": mod.run_security_probes()}
    res = mod.handle_request(story_id=req.story_id, method=req.method, route=req.route, role=req.role)
    return {"probe": req.story_id, "result": res}


@router.post("/steer")
def steer_run(req: SteerRequest) -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    ledger = ProjectLedger(ws)
    act = req.action.lower().strip()
    if act in ("pause", "stopped", "stop", "running", "resume"):
        _RUN_CONTROL_STATE["mode"] = "paused" if act == "pause" else ("stopped" if act in ("stop", "stopped") else "running")
    if req.note.strip():
        try:
            fact = ledger.record_fact(
                scope="steer",
                kind="operator_note",
                text=req.note.strip(),
                source_ref="mission_control",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _RUN_CONTROL_STATE["steer_notes"].append(
            {"id": fact.id, "text": fact.text, "timestamp": time.time()}
        )
    return get_mission_state()


@router.post("/ledger/fact")
def upsert_ledger_fact(req: FactRequest) -> Dict[str, Any]:
    ws = _ensure_seeded_workspace()
    ledger = ProjectLedger(ws)
    try:
        if req.supersede_id:
            ledger.supersede_fact(
                req.supersede_id,
                new_text=req.text,
                source_ref="mission_control_ui",
                sensitivity=req.sensitivity,
            )
        else:
            ledger.record_fact(
                scope=req.scope,
                kind=req.kind,
                text=req.text,
                source_ref="mission_control_ui",
                sensitivity=req.sensitivity,
            )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return get_mission_state()
