"""
Office plugin backend API.

Exposes a FastAPI APIRouter with a single endpoint that the Office dashboard
tab polls to get a live snapshot of all agent activity:

  GET /api/plugins/office/snapshot

Returns profiles, active sessions, kanban workers, cron jobs, and gateway
status — all read-only. The Office is a visualization layer, not a control
surface.
"""

import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(tags=["office"])


def _get_hermes_home() -> Path:
    """Resolve the Hermes home directory."""
    return Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


def _is_gateway_running(profile_name: str) -> bool:
    """Check if a profile's gateway is running by looking for a PID file.

    The PID file is JSON: ``{"pid": 12345, "kind": "hermes-gateway", ...}``.
    """
    home = _get_hermes_home()
    if profile_name == "default":
        pid_file = home / "gateway.pid"
    else:
        pid_file = home / "profiles" / profile_name / "gateway.pid"
    if pid_file.exists():
        try:
            raw = pid_file.read_text().strip()
            # PID file is JSON with a "pid" key
            try:
                data = json.loads(raw)
                pid = int(data.get("pid", 0))
            except (json.JSONDecodeError, TypeError):
                # Fallback: bare integer
                pid = int(raw)
            if pid > 0:
                os.kill(pid, 0)  # Check if process exists
                return True
        except (ProcessLookupError, ValueError, OSError):
            return False
    return False


def _count_skills(profile_dir: Path) -> int:
    """Count skills in a profile directory."""
    skills_dir = profile_dir / "skills"
    if not skills_dir.is_dir():
        return 0
    try:
        return sum(1 for s in skills_dir.iterdir() if s.is_dir() and not s.name.startswith("."))
    except Exception:
        return 0


def _read_profile_config(profile_dir: Path) -> Dict[str, Any]:
    """Read profile config.yaml for model/provider info."""
    cfg_path = profile_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_profiles_snapshot() -> List[Dict[str, Any]]:
    """List all profiles with gateway status."""
    home = _get_hermes_home()
    result = []

    # Default profile
    default_cfg = _read_profile_config(home)
    model_cfg = default_cfg.get("model", {})
    result.append({
        "name": "default",
        "path": str(home),
        "is_default": True,
        "model": model_cfg.get("default", "unknown"),
        "provider": model_cfg.get("provider", "unknown"),
        "gateway_running": _is_gateway_running("default"),
        "description": "",
        "skill_count": _count_skills(home),
    })

    # Other profiles
    profiles_dir = home / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            prof_cfg = _read_profile_config(child)
            model_cfg = prof_cfg.get("model", {})
            result.append({
                "name": child.name,
                "path": str(child),
                "is_default": False,
                "model": model_cfg.get("default", "unknown"),
                "provider": model_cfg.get("provider", "unknown"),
                "gateway_running": _is_gateway_running(child.name),
                "description": prof_cfg.get("profile_description", ""),
                "skill_count": _count_skills(child),
            })

    return result


def _get_sessions_snapshot() -> List[Dict[str, Any]]:
    """Get recent active sessions from the SessionDB."""
    try:
        from hermes_state import SessionDB
        home = _get_hermes_home()
        db_path = home / "state.db"
        if not db_path.exists():
            return []
        # SessionDB expects a Path object (it calls .parent.mkdir)
        db = SessionDB(db_path)
        sessions = db.list_sessions_rich(limit=20, order_by_last_active=True)
        result = []
        for s in sessions:
            if isinstance(s, dict):
                result.append({
                    "id": s.get("id", ""),
                    "source": s.get("source", ""),
                    "model": s.get("model", ""),
                    "title": s.get("title", "Untitled"),
                    "is_active": s.get("ended_at") is None,
                    "last_active": s.get("last_active", 0),
                    "message_count": s.get("message_count", 0),
                    "tool_call_count": s.get("tool_call_count", 0),
                })
        return result
    except Exception as e:
        return [{"error": str(e)}]


def _get_kanban_snapshot() -> List[Dict[str, Any]]:
    """Get kanban tasks that are claimed or in-progress."""
    try:
        from hermes_cli import kanban_db
        home = _get_hermes_home()
        db_path = home / "kanban.db"
        if not db_path.exists():
            return []
        # kanban_db uses module-level functions with a board path, not a class
        conn = kanban_db.connect(db_path)
        tasks = kanban_db.list_tasks(conn)
        result = []
        for t in tasks:
            if isinstance(t, dict):
                status = t.get("status", "")
            elif hasattr(t, "__dict__"):
                status = t.__dict__.get("status", "")
            else:
                status = ""
            # Include any task that is assigned and not done/archived.
            # "blocked" tasks still mean the agent is assigned and should
            # be at their desk waiting on the dependency.
            if status not in ("done", "archived", ""):
                d = t if isinstance(t, dict) else t.__dict__
                result.append({
                    "id": d.get("id", ""),
                    "title": d.get("title", ""),
                    "status": status,
                    "assignee": d.get("assignee", ""),
                    "board": d.get("board", ""),
                })
        conn.close()
        return result
    except Exception as e:
        return [{"error": str(e)}]


def _get_cron_snapshot() -> List[Dict[str, Any]]:
    """Get cron jobs with recent activity."""
    try:
        home = _get_hermes_home()
        jobs_file = home / "cron" / "jobs.json"
        if not jobs_file.exists():
            return []
        raw = json.loads(jobs_file.read_text())
        # jobs.json is {"jobs": [...], "updated_at": "..."} — not a bare list
        if isinstance(raw, dict):
            jobs = raw.get("jobs", [])
        elif isinstance(raw, list):
            jobs = raw  # legacy format
        else:
            jobs = []
        result = []
        for job in jobs:
            if not isinstance(job, dict):
                continue
            if not job.get("enabled", True):
                continue
            result.append({
                "id": job.get("id", ""),
                "name": job.get("name", ""),
                "schedule": job.get("schedule", ""),
                "enabled": job.get("enabled", True),
                "last_run_at": job.get("last_run_at"),
                "last_status": job.get("last_status"),
                "next_run_at": job.get("next_run_at"),
                "prompt_preview": (job.get("prompt", "") or "")[:100],
            })
        return result
    except Exception as e:
        return [{"error": str(e)}]


def _get_gateway_status() -> Dict[str, Any]:
    """Get gateway platform status from config."""
    try:
        home = _get_hermes_home()
        cfg = _read_profile_config(home)
        platforms = cfg.get("platforms", {})
        result = {}
        for name, conf in platforms.items():
            if isinstance(conf, dict) and conf.get("enabled"):
                result[name] = {"enabled": True, "state": "configured"}
        return result
    except Exception:
        return {}


@router.get("/snapshot")
async def get_snapshot() -> Dict[str, Any]:
    """GET /api/plugins/office/snapshot — aggregated agent activity snapshot."""
    profiles = _get_profiles_snapshot()
    sessions = _get_sessions_snapshot()
    kanban = _get_kanban_snapshot()
    cron = _get_cron_snapshot()
    gateway = _get_gateway_status()

    active_count = sum(1 for p in profiles if p.get("gateway_running"))
    active_sessions = sum(1 for s in sessions if s.get("is_active"))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_profiles": len(profiles),
            "active_gateways": active_count,
            "active_sessions": active_sessions,
            "kanban_tasks_active": len(kanban),
            "cron_jobs_enabled": sum(1 for c in cron if c.get("enabled", True)),
        },
        "profiles": profiles,
        "sessions": sessions,
        "kanban": kanban,
        "cron": cron,
        "gateway_platforms": gateway,
    }


# ── Office layout config ──────────────────────────────────────────

def _get_layout_path() -> Path:
    """Path to the user-editable office layout JSON."""
    return _get_hermes_home() / "office-layout.json"


def _default_layout() -> Dict[str, Any]:
    """The default Claw3D office layout, written on first load."""
    return {
        "version": 1,
        "name": "Claw3D Default Office",
        "desk_positions": [
            {"x": 150, "y": 260, "facing": 0, "deskId": "desk_0"},
            {"x": 350, "y": 260, "facing": 0, "deskId": "desk_1"},
            {"x": 550, "y": 260, "facing": 0, "deskId": "desk_2"},
            {"x": 750, "y": 260, "facing": 0, "deskId": "desk_3"},
            {"x": 150, "y": 460, "facing": 0, "deskId": "desk_4"},
            {"x": 350, "y": 460, "facing": 0, "deskId": "desk_5"},
            {"x": 550, "y": 460, "facing": 0, "deskId": "desk_6"},
            {"x": 750, "y": 460, "facing": 0, "deskId": "desk_7"},
        ],
        "roam_points": [
            {"x": 800, "y": 200},
            {"x": 850, "y": 500},
            {"x": 820, "y": 580},
            {"x": 450, "y": 420},
            {"x": 250, "y": 420},
            {"x": 650, "y": 420},
            {"x": 150, "y": 620},
        ],
        "furniture": _default_furniture_seeds(),
    }


def _default_furniture_seeds() -> List[Dict[str, Any]]:
    """The full Claw3D furniture layout as plain JSON seeds."""
    return [
        {"type": "round_table", "x": 50, "y": 50, "r": 90},
        {"type": "chair", "x": 130, "y": 50, "facing": 0},
        {"type": "chair", "x": 200, "y": 90, "facing": 325},
        {"type": "chair", "x": 180, "y": 170, "facing": 240},
        {"type": "chair", "x": 120, "y": 480, "facing": 180},
        {"type": "chair", "x": 50, "y": 150, "facing": 105},
        {"type": "chair", "x": 60, "y": 80, "facing": 60},
        {"type": "chair", "x": 550, "y": 50, "facing": 0},
        {"type": "bookshelf", "x": 600, "y": 30, "w": 80, "h": 120},
        {"type": "couch", "x": 270, "y": 90, "w": 40, "h": 80, "vertical": True, "facing": 180},
        {"type": "fridge", "x": 1050, "y": 20, "w": 40, "h": 80},
        {"type": "stove", "x": 920, "y": 20},
        {"type": "cabinet", "x": 980, "y": 30, "w": 40, "h": 40},
        {"type": "microwave", "x": 1030, "y": 10, "facing": 0},
        {"type": "sink", "x": 970, "y": 20},
        {"type": "dishwasher", "x": 950, "y": 20, "w": 40, "h": 40},
        {"type": "cabinet", "x": 840, "y": 30, "w": 80, "h": 40, "elevation": 0},
        {"type": "coffee_machine", "x": 880, "y": 30, "elevation": 0.56},
        {"type": "wall_cabinet", "x": 960, "y": 10, "w": 80, "h": 20, "elevation": 0.9},
        {"type": "wall_cabinet", "x": 880, "y": 10, "w": 80, "h": 20, "elevation": 0.9},
        {"type": "round_table", "x": 890, "y": 100, "r": 50},
        {"type": "chair", "x": 930, "y": 100, "facing": 0},
        {"type": "chair", "x": 930, "y": 180, "facing": 180},
        {"type": "chair", "x": 880, "y": 130, "facing": 90},
        {"type": "chair", "x": 970, "y": 130, "facing": 270},
        {"type": "vending", "x": 790, "y": 10},
        {"type": "trash", "x": 210, "y": 20},
        {"type": "desk_cubicle", "x": 100, "y": 300, "id": "desk_0"},
        {"type": "chair", "x": 120, "y": 290, "facing": 180},
        {"type": "computer", "x": 120, "y": 287},
        {"type": "keyboard", "x": 130, "y": 295},
        {"type": "mouse", "x": 152, "y": 295},
        {"type": "trash", "x": 170, "y": 290},
        {"type": "desk_cubicle", "x": 300, "y": 300, "id": "desk_1"},
        {"type": "chair", "x": 320, "y": 290, "facing": 180},
        {"type": "computer", "x": 320, "y": 287},
        {"type": "keyboard", "x": 330, "y": 295},
        {"type": "mouse", "x": 352, "y": 295},
        {"type": "trash", "x": 370, "y": 290},
        {"type": "desk_cubicle", "x": 500, "y": 300, "id": "desk_2"},
        {"type": "chair", "x": 520, "y": 290, "facing": 180},
        {"type": "computer", "x": 520, "y": 287},
        {"type": "keyboard", "x": 530, "y": 295},
        {"type": "mouse", "x": 552, "y": 295},
        {"type": "trash", "x": 570, "y": 290},
        {"type": "desk_cubicle", "x": 700, "y": 300, "id": "desk_3"},
        {"type": "chair", "x": 720, "y": 290, "facing": 180},
        {"type": "computer", "x": 720, "y": 287},
        {"type": "keyboard", "x": 730, "y": 295},
        {"type": "mouse", "x": 752, "y": 295},
        {"type": "trash", "x": 770, "y": 290},
        {"type": "desk_cubicle", "x": 100, "y": 500, "id": "desk_4"},
        {"type": "computer", "x": 120, "y": 487},
        {"type": "keyboard", "x": 130, "y": 490},
        {"type": "mouse", "x": 152, "y": 495},
        {"type": "trash", "x": 170, "y": 490},
        {"type": "desk_cubicle", "x": 300, "y": 500, "id": "desk_5"},
        {"type": "chair", "x": 310, "y": 490, "facing": 180},
        {"type": "computer", "x": 320, "y": 487},
        {"type": "keyboard", "x": 330, "y": 495},
        {"type": "mouse", "x": 352, "y": 495},
        {"type": "trash", "x": 370, "y": 500},
        {"type": "desk_cubicle", "x": 500, "y": 500, "id": "desk_6"},
        {"type": "chair", "x": 520, "y": 490, "facing": 180},
        {"type": "computer", "x": 520, "y": 487},
        {"type": "keyboard", "x": 530, "y": 495},
        {"type": "mouse", "x": 552, "y": 495},
        {"type": "trash", "x": 570, "y": 490},
        {"type": "desk_cubicle", "x": 700, "y": 500, "id": "desk_7"},
        {"type": "chair", "x": 720, "y": 490, "facing": 180},
        {"type": "computer", "x": 720, "y": 487},
        {"type": "keyboard", "x": 730, "y": 495},
        {"type": "mouse", "x": 752, "y": 495},
        {"type": "trash", "x": 770, "y": 490},
        {"type": "couch", "x": 1000, "y": 380, "w": 100, "h": 40, "facing": 90},
        {"type": "couch", "x": 390, "y": 630, "w": 100, "h": 40},
        {"type": "table_rect", "x": 980, "y": 380, "w": 60, "h": 30, "facing": 270},
        {"type": "pingpong", "x": 950, "y": 600, "w": 100, "h": 60},
        {"type": "beanbag", "x": 1000, "y": 330, "color": "#e65100", "facing": 90},
        {"type": "beanbag", "x": 1000, "y": 410, "color": "#1565c0", "facing": 90},
        {"type": "atm", "x": 430, "y": 210, "facing": 90},
        {"type": "phone_booth", "x": 1050, "y": 190, "facing": 270},
        {"type": "kanban_board", "x": 460, "y": -60, "facing": 180},
        {"type": "sms_booth", "x": 700, "y": 10, "facing": 0},
        {"type": "whiteboard", "x": 40, "y": 200, "w": 10, "h": 60},
        {"type": "clock", "x": 550, "y": 5},
        {"type": "lamp", "x": 430, "y": 100},
        {"type": "lamp", "x": 980, "y": 390},
        {"type": "trash", "x": 830, "y": 20},
        {"type": "plant", "x": 40, "y": 40},
        {"type": "plant", "x": 660, "y": 30},
        {"type": "plant", "x": 340, "y": 700},
        {"type": "plant", "x": 450, "y": 450},
        {"type": "plant", "x": 1090, "y": 310},
        {"type": "plant", "x": 1100, "y": 490},
        {"type": "plant", "x": 530, "y": 700},
        {"type": "chair", "x": 100, "y": 200, "facing": 180},
    ]


@router.get("/layout")
async def get_layout() -> Dict[str, Any]:
    """GET /api/plugins/office/layout — user office layout.

    Returns the furniture layout from ~/.hermes/office-layout.json.
    Creates the default Claw3D layout on first call if the file doesn't exist.
    """
    layout_path = _get_layout_path()
    if not layout_path.exists():
        layout = _default_layout()
        try:
            layout_path.write_text(json.dumps(layout, indent=2, ensure_ascii=False))
        except Exception:
            pass  # read-only filesystem — just return the default
        return layout

    try:
        return json.loads(layout_path.read_text())
    except Exception as e:
        # Corrupt JSON — return default and log
        layout = _default_layout()
        layout["_error"] = f"office-layout.json parse error: {e}; using defaults"
        return layout


@router.put("/layout")
async def put_layout(body: Dict[str, Any]) -> Dict[str, Any]:
    """PUT /api/plugins/office/layout — save user office layout.

    Writes the furniture layout to ~/.hermes/office-layout.json.
    The next /snapshot poll will pick up the new layout automatically.
    """
    layout_path = _get_layout_path()
    try:
        layout_path.write_text(json.dumps(body, indent=2, ensure_ascii=False))
        return {"ok": True, "saved_to": str(layout_path)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
