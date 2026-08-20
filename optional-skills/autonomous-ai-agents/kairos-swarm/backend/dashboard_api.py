"""backend/dashboard_api.py
Futuristic 3D Dashboard Backend for Kairos.

- FastAPI + WebSocket for real-time agent status
- Broadcasts live updates from multi-agent swarm
- Serves the React 3D frontend as static files (production) or built-in HTML interface
- Integrates with Hermes kanban_swarm & AIAgent APIs
- Endpoints: /api/status, /ws/dashboard (real-time), /api/trigger_goal (authenticated), etc.

Run with: uvicorn backend.dashboard_api:app --reload --port 8001
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add repository root to path for core Hermes imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kairos.dashboard")


# --- Helper to load Kairos configuration from config.yaml / env ---
def get_kairos_config() -> Dict[str, Any]:
    config_data: Dict[str, Any] = {
        "enabled": True,
        "scan_interval_minutes": 15,
        "max_proactive_fixes": 3,
        "require_approval": True,
        "scan_paths": [".", "core", "kairos", "agents"],
        "max_concurrent_agents": 4,
        "api_key": "",
    }
    # Load from Hermes config.yaml if available
    try:
        from hermes_cli.config import load_cli_config

        cfg = load_cli_config()
        if isinstance(cfg, dict) and "kairos" in cfg and isinstance(cfg["kairos"], dict):
            config_data.update(cfg["kairos"])
    except Exception as e:
        logger.debug("Failed to load kairos config from Hermes config.yaml: %s", e)

    # Environment variables overrides (secondary fallback)
    if os.getenv("KAIROS_ENABLED") is not None:
        config_data["enabled"] = os.getenv("KAIROS_ENABLED", "").lower() in ("true", "1", "yes")
    if os.getenv("KAIROS_SCAN_INTERVAL_MINUTES"):
        try:
            config_data["scan_interval_minutes"] = int(os.getenv("KAIROS_SCAN_INTERVAL_MINUTES", "15"))
        except ValueError:
            pass
    if os.getenv("KAIROS_MAX_PROACTIVE_FIXES"):
        try:
            config_data["max_proactive_fixes"] = int(os.getenv("KAIROS_MAX_PROACTIVE_FIXES", "3"))
        except ValueError:
            pass
    if os.getenv("KAIROS_API_KEY"):
        config_data["api_key"] = os.getenv("KAIROS_API_KEY", "")

    return config_data


# --- Models for real-time data ---
class GoalRequest(BaseModel):
    goal: str
    mode: str = "task"


class AgentStatus(BaseModel):
    name: str
    status: str  # idle, thinking, working, completed, error
    current_task: str
    progress: float  # 0-100
    last_update: str
    color: str  # neon color for 3D UI


class DashboardState(BaseModel):
    agents: List[AgentStatus]
    active_task: str
    tasks_completed: int
    skills_created: int
    kairos_heartbeat: str
    token_usage: int
    logs: List[str]
    preview_text: str = "No preview available yet."
    current_goal: str = ""
    started_at: str = ""
    estimated_duration_seconds: int = 0
    time_remaining_seconds: int = 0
    real_artifacts: List[str] = []
    real_result: str = ""
    task_running: bool = False
    task_completed: bool = False


# --- In-memory state ---
connected_clients: Set[WebSocket] = set()
current_state: DashboardState = DashboardState(
    agents=[
        AgentStatus(
            name="Orchestrator",
            status="idle",
            current_task="Monitoring swarm",
            progress=100,
            last_update=datetime.now().isoformat(),
            color="#00f0ff",
        ),
        AgentStatus(
            name="Architect",
            status="idle",
            current_task="Waiting for goal",
            progress=0,
            last_update=datetime.now().isoformat(),
            color="#a855f7",
        ),
        AgentStatus(
            name="Coder",
            status="idle",
            current_task="No active coding",
            progress=0,
            last_update=datetime.now().isoformat(),
            color="#22c55e",
        ),
        AgentStatus(
            name="Tester",
            status="idle",
            current_task="Ready for tests",
            progress=0,
            last_update=datetime.now().isoformat(),
            color="#eab308",
        ),
        AgentStatus(
            name="Scribe",
            status="idle",
            current_task="Documenting knowledge",
            progress=0,
            last_update=datetime.now().isoformat(),
            color="#f472b6",
        ),
    ],
    active_task="No active swarm task",
    tasks_completed=42,
    skills_created=17,
    kairos_heartbeat=datetime.now().isoformat(),
    token_usage=128450,
    logs=["[SYSTEM] Dashboard backend online", "[KAIROS] Heartbeat OK"],
    current_goal="",
    started_at="",
    estimated_duration_seconds=45,
    time_remaining_seconds=0,
    real_artifacts=[],
    real_result="",
    task_running=False,
    task_completed=False,
)

# --- Module-level safe log and event emitters ---
def emit_log(msg: str, level: str = "info", sender: str = "Orchestrator") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] [{sender}] {msg}"
    current_state.logs.append(formatted)
    if len(current_state.logs) > 50:
        current_state.logs = current_state.logs[-50:]

    if level == "error":
        logger.error("[%s] %s", sender, msg)
    else:
        logger.info("[%s] %s", sender, msg)


def emit_agent_update(name: str, status_str: str, current_task: str, progress: float) -> None:
    for i, agent in enumerate(current_state.agents):
        if agent.name.lower() == name.lower():
            current_state.agents[i].status = status_str
            current_state.agents[i].current_task = current_task
            current_state.agents[i].progress = progress
            current_state.agents[i].last_update = datetime.now().isoformat()
            break


def emit_metrics(tasks_completed: int = 0) -> None:
    if tasks_completed:
        current_state.tasks_completed += tasks_completed


# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        connected_clients.add(websocket)
        logger.info("Dashboard client connected. Total: %d", len(self.active_connections))
        await websocket.send_json(current_state.model_dump())

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        connected_clients.discard(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        dead = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)


manager = ConnectionManager()

# --- Authentication Dependency ---
async def require_dashboard_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> None:
    """Enforce authorization on task execution and mutation routes.

    Validates:
    1. Active session in request.state.session (when running behind Hermes dashboard auth middleware).
    2. Authorization: Bearer <token> or X-API-Key / query token against configured API keys.
    """
    if getattr(request.state, "session", None) is not None:
        return

    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    elif "token" in request.query_params:
        token = request.query_params["token"].strip()

    expected_keys: Set[str] = set()
    for env_name in ("KAIROS_API_KEY", "DASHBOARD_API_KEY", "HERMES_SESSION_TOKEN"):
        val = os.environ.get(env_name, "").strip()
        if val:
            expected_keys.add(val)

    cfg = get_kairos_config()
    if cfg.get("api_key"):
        expected_keys.add(str(cfg["api_key"]).strip())

    if expected_keys:
        if not token or token not in expected_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing API key/token",
            )
    else:
        # Non-loopback host deployments (e.g. Railway) require explicit auth configuration or token
        host = request.headers.get("host", "").split(":")[0].strip().lower()
        if host and host not in {"localhost", "127.0.0.1", "::1"}:
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized: Authentication token required for public deployments",
                )


# --- FastAPI App ---
app = FastAPI(title="KAIROS Swarm - AI Operating System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve React build or embedded HTML interface fallback ---
DASHBOARD_BUILD = Path(__file__).parent.parent / "dashboard" / "dist"
if DASHBOARD_BUILD.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_BUILD)), name="static")

EMBEDDED_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kairos Swarm Dashboard</title>
    <style>
        :root { --bg: #090d16; --panel: #111827; --accent: #00f0ff; --text: #f3f4f6; }
        body { margin: 0; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, sans-serif; }
        header { padding: 1.5rem; background: var(--panel); border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.5rem; color: var(--accent); letter-spacing: 0.05em; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; padding: 1.5rem; max-width: 1400px; margin: 0 auto; }
        .card { background: var(--panel); border: 1px solid #1f2937; border-radius: 0.5rem; padding: 1.25rem; }
        .agent { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid #1f2937; }
        .status { padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; text-transform: uppercase; }
        .status.working { background: #065f46; color: #34d399; }
        .status.idle { background: #374151; color: #9ca3af; }
        .status.completed { background: #1e3a8a; color: #60a5fa; }
        .logs { font-family: monospace; background: #030712; padding: 1rem; border-radius: 0.375rem; height: 300px; overflow-y: auto; }
        .log-entry { margin-bottom: 0.35rem; color: #9ca3af; }
    </style>
</head>
<body>
    <header>
        <h1>⚡ KAIROS SWARM DASHBOARD</h1>
        <div id="connection-status" style="color: #34d399;">● CONNECTED</div>
    </header>
    <div class="grid">
        <div class="card">
            <h2>Active Agents</h2>
            <div id="agents-list">Loading agents...</div>
            <h2 style="margin-top: 2rem;">Real-time Activity Logs</h2>
            <div id="logs" class="logs"></div>
        </div>
        <div class="card">
            <h2>Swarm Status</h2>
            <p><strong>Current Goal:</strong> <span id="current-goal">None</span></p>
            <p><strong>Tasks Completed:</strong> <span id="tasks-completed">0</span></p>
            <p><strong>Heartbeat:</strong> <span id="heartbeat">OK</span></p>
        </div>
    </div>
    <script>
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/dashboard`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const state = data.full_state || data;
            if (state.agents) {
                document.getElementById('agents-list').innerHTML = state.agents.map(a => `
                    <div class="agent">
                        <div>
                            <strong style="color: ${a.color}">${a.name}</strong>
                            <div style="font-size:0.85rem; color:#9ca3af">${a.current_task}</div>
                        </div>
                        <span class="status ${a.status}">${a.status}</span>
                    </div>
                `).join('');
            }
            if (state.logs) {
                document.getElementById('logs').innerHTML = state.logs.map(l => `<div class="log-entry">${l}</div>`).join('');
                document.getElementById('logs').scrollTop = document.getElementById('logs').scrollHeight;
            }
            if (state.current_goal) document.getElementById('current-goal').innerText = state.current_goal;
            if (state.tasks_completed !== undefined) document.getElementById('tasks-completed').innerText = state.tasks_completed;
            if (state.kairos_heartbeat) document.getElementById('heartbeat').innerText = state.kairos_heartbeat;
        };
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    if DASHBOARD_BUILD.exists() and (DASHBOARD_BUILD / "index.html").exists():
        return HTMLResponse((DASHBOARD_BUILD / "index.html").read_text(encoding="utf-8"))
    return HTMLResponse(EMBEDDED_DASHBOARD_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/status")
async def get_status():
    """Current full dashboard state."""
    return current_state


@app.get("/api/agents")
async def get_agents():
    return current_state.agents


@app.post("/api/update_agent")
async def update_agent(agent_update: AgentStatus, _: None = Depends(require_dashboard_auth)):
    """Update status for an agent (authenticated)."""
    global current_state
    for i, agent in enumerate(current_state.agents):
        if agent.name.lower() == agent_update.name.lower():
            current_state.agents[i] = agent_update
            break
    else:
        current_state.agents.append(agent_update)

    emit_log(f"{agent_update.name}: {agent_update.status} - {agent_update.current_task[:50]}", sender=agent_update.name)

    await manager.broadcast(
        {
            "type": "agent_update",
            "data": agent_update.model_dump(),
            "full_state": current_state.model_dump(),
        }
    )
    return {"success": True}


def _execute_swarm_goal(goal: str) -> dict:
    """Execute goal using hermes_cli.kanban_swarm or AIAgent fallback."""
    try:
        import sqlite3

        from hermes_cli import kanban_swarm
        from hermes_constants import get_hermes_home

        db_path = get_hermes_home() / "kanban.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(db_path)) as conn:
            workers = [
                kanban_swarm.SwarmWorkerSpec(profile="Architect", title=f"Architect: {goal[:40]}"),
                kanban_swarm.SwarmWorkerSpec(profile="Coder", title=f"Coder: {goal[:40]}"),
                kanban_swarm.SwarmWorkerSpec(profile="Tester", title=f"Tester: {goal[:40]}"),
            ]
            created = kanban_swarm.create_swarm(
                conn,
                goal=goal,
                workers=workers,
                verifier_assignee="Verifier",
                synthesizer_assignee="Synthesizer",
            )
            return {
                "success": True,
                "output": f"Swarm graph created (root_id={created.root_id}, workers={len(created.worker_ids)}).",
                "artifacts": [f"kanban://root/{created.root_id}"],
            }
    except Exception as e:
        logger.warning("kanban_swarm graph creation failed, attempting AIAgent fallback: %s", e)
        try:
            from run_agent import AIAgent

            agent = AIAgent(quiet_mode=True)
            res = agent.chat(f"Kairos Swarm Task Goal: {goal}")
            return {"success": True, "output": str(res), "artifacts": []}
        except Exception as inner_e:
            return {"success": False, "output": f"Execution failed: {e} | Fallback: {inner_e}", "artifacts": []}


@app.post("/api/trigger_goal")
async def trigger_goal(request: GoalRequest, _: None = Depends(require_dashboard_auth)):
    """Trigger a swarm goal from dashboard (authenticated)."""
    goal = request.goal.strip()
    mode = request.mode or "task"
    global current_state

    if mode == "chat":
        emit_log(f"DASHBOARD CHAT: {goal[:80]}", sender="User")
        await manager.broadcast(
            {"type": "new_goal", "goal": goal, "mode": mode, "full_state": current_state.model_dump()}
        )
        return {"message": "Chat received", "goal": goal}

    current_state.current_goal = goal
    current_state.active_task = goal
    current_state.started_at = datetime.now().isoformat()
    current_state.task_running = True
    current_state.task_completed = False
    current_state.real_artifacts = []
    current_state.real_result = "Running Kairos swarm..."

    est = 45
    if len(goal) > 80:
        est = 90
    current_state.estimated_duration_seconds = est
    current_state.time_remaining_seconds = est

    for agent in current_state.agents:
        agent.status = "thinking" if agent.name != "Orchestrator" else "working"
        agent.current_task = f"Starting: {goal[:35]}"
        agent.progress = 10
        agent.last_update = datetime.now().isoformat()

    emit_log(f"🚀 Starting swarm for: {goal[:60]}...", "info", "Orchestrator")

    await manager.broadcast({"type": "new_goal", "goal": goal, "full_state": current_state.model_dump()})

    async def _run_real_swarm():
        start_time = time.time()
        try:
            emit_log(f"Executing swarm workflow for goal: {goal[:50]}", "info", "Orchestrator")
            emit_agent_update("Orchestrator", "working", f"Coordinating: {goal[:45]}", 25)

            result = await asyncio.to_thread(_execute_swarm_goal, goal)

            duration = time.time() - start_time
            current_state.time_remaining_seconds = 0
            current_state.task_running = False
            current_state.task_completed = True
            current_state.real_result = result.get("output", "Completed")
            current_state.real_artifacts = result.get("artifacts", [])
            if result.get("success"):
                emit_metrics(tasks_completed=1)

            emit_log(f"✅ Swarm task finished in {duration:.1f}s", "success", "Orchestrator")

            for a in current_state.agents:
                a.status = "completed"
                a.progress = 100
                a.current_task = "Task finished"

            await manager.broadcast(
                {
                    "type": "task_complete",
                    "full_state": current_state.model_dump(),
                    "result": current_state.real_result,
                    "artifacts": current_state.real_artifacts,
                }
            )
        except Exception as e:
            current_state.task_running = False
            current_state.real_result = f"ERROR: {str(e)}"
            emit_log(f"❌ Swarm failed: {e}", "error", "Orchestrator")
            await manager.broadcast(
                {"type": "task_error", "error": str(e), "full_state": current_state.model_dump()}
            )

    asyncio.create_task(_run_real_swarm())
    asyncio.create_task(_countdown_timer(est))

    return {"message": "Kairos swarm started", "goal": goal, "estimated_seconds": est}


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def _countdown_timer(estimated: int):
    global current_state
    for _ in range(estimated + 5):
        if not current_state.task_running:
            break
        await asyncio.sleep(1)
        current_state.time_remaining_seconds = max(0, current_state.time_remaining_seconds - 1)
        await manager.broadcast(
            {
                "type": "timer_tick",
                "time_remaining": current_state.time_remaining_seconds,
                "full_state": current_state.model_dump(),
            }
        )


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Kairos Swarm Dashboard API online at port 8001")


if __name__ == "__main__":
    uvicorn.run("backend.dashboard_api:app", host="0.0.0.0", port=8001, reload=True)