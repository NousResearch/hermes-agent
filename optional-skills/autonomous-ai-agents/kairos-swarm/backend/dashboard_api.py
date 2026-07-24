"""backend/dashboard_api.py
Futuristic 3D Dashboard Backend for Kairos.

- FastAPI + WebSocket for real-time agent status
- Broadcasts live updates from multi-agent swarm
- Serves the React 3D frontend as static files (production) or proxy in dev
- Integrates with existing core (orchestrator, kairos, etc.)
- Endpoints: /status, /ws/dashboard (real-time), /trigger_goal, etc.

Run with: uvicorn backend.dashboard_api:app --reload --port 8001
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

class GoalRequest(BaseModel):
    goal: str
    mode: str = 'task'

import uvicorn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from core.dashboard_events import get_broadcaster
except ImportError:
    # Fallback if not available
    def get_broadcaster(): return None
    class MockBroadcaster:
        def register_listener(self, cb): pass
        def enable(self): pass
    def get_broadcaster(): return MockBroadcaster()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard_api")

# --- Models for real-time data ---
class AgentStatus(BaseModel):
    name: str
    status: str  # idle, thinking, working, completed, error
    current_task: str
    progress: float  # 0-100
    last_update: str
    color: str  # neon color for 3D

class DashboardState(BaseModel):
    agents: List[AgentStatus]
    active_task: str
    tasks_completed: int
    skills_created: int
    kairos_heartbeat: str
    token_usage: int
    logs: List[str]
    preview_text: str = "No preview available yet."
    # Real task tracking + timer
    current_goal: str = ""
    started_at: str = ""
    estimated_duration_seconds: int = 0
    time_remaining_seconds: int = 0
    real_artifacts: List[str] = []
    real_result: str = ""
    task_running: bool = False
    task_completed: bool = False

# --- In-memory state (in production use Redis or proper pubsub) ---
connected_clients: Set[WebSocket] = set()
current_state: DashboardState = DashboardState(
    agents=[
        AgentStatus(name="Orchestrator", status="idle", current_task="Monitoring swarm", progress=100, last_update=datetime.now().isoformat(), color="#00f0ff"),
        AgentStatus(name="Architect", status="idle", current_task="Waiting for goal", progress=0, last_update=datetime.now().isoformat(), color="#a855f7"),
        AgentStatus(name="Coder", status="idle", current_task="No active coding", progress=0, last_update=datetime.now().isoformat(), color="#22c55e"),
        AgentStatus(name="Tester", status="idle", current_task="Ready for tests", progress=0, last_update=datetime.now().isoformat(), color="#eab308"),
        AgentStatus(name="Scribe", status="idle", current_task="Documenting knowledge", progress=0, last_update=datetime.now().isoformat(), color="#f472b6"),
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

app = FastAPI(title="KAIROS - AI Operating System", version="1.0")

@app.get("/")
async def root():
    return {"message": "KAIROS Backend is running. Visit /docs for API documentation."}

@app.get("/health")
async def health():
    return {"status": "ok"}

# CORS for React dev server (localhost:3000) and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve React build in production (when built) ---
DASHBOARD_BUILD = Path(__file__).parent.parent / "dashboard" / "dist"
if DASHBOARD_BUILD.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_BUILD), html=True), name="dashboard")

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        connected_clients.add(websocket)
        logger.info(f"Dashboard client connected. Total: {len(self.active_connections)}")
        # Send current state immediately
        await websocket.send_json(current_state.model_dump())

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        connected_clients.discard(websocket)
        logger.info(f"Dashboard client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast to all connected 3D dashboards"""
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

# --- API Endpoints ---

@app.get("/api/status")
async def get_status():
    """Current full dashboard state (for polling fallback)"""
    return current_state

@app.post("/api/update_agent")
async def update_agent(agent_update: AgentStatus):
    """Manual or internal update for an agent (used by orchestrator later)"""
    global current_state
    for i, agent in enumerate(current_state.agents):
        if agent.name.lower() == agent_update.name.lower():
            current_state.agents[i] = agent_update
            break
    else:
        current_state.agents.append(agent_update)

    current_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {agent_update.name}: {agent_update.status} - {agent_update.current_task[:50]}")

    # Broadcast to all 3D clients
    await manager.broadcast({
        "type": "agent_update",
        "data": agent_update.model_dump(),
        "full_state": current_state.model_dump()
    })
    return {"success": True}

@app.post("/api/trigger_goal")
async def trigger_goal(request: GoalRequest):
    """Trigger a NEW REAL swarm goal from dashboard. Runs actual orchestrator + shows live timer + real results."""
    goal = request.goal.strip()
    mode = request.mode or 'task'
    global current_state

    if mode == 'chat':
        current_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] DASHBOARD CHAT: {goal[:80]}")
        await manager.broadcast({"type": "new_goal", "goal": goal, "mode": mode, "full_state": current_state.model_dump()})
        return {"message": "Chat received", "goal": goal}

    # === REAL TASK START ===
    current_state.current_goal = goal
    current_state.active_task = goal
    current_state.started_at = datetime.now().isoformat()
    current_state.task_running = True
    current_state.task_completed = False
    current_state.real_artifacts = []
    current_state.real_result = "Running real swarm..."
    current_state.logs = current_state.logs[-8:] + [f"[{datetime.now().strftime('%H:%M:%S')}] REAL GOAL STARTED: {goal[:70]}"]

    # Smart ETA (30s for simple, up to 3min for complex)
    est = 45
    if len(goal) > 80: est = 90
    if any(k in goal.lower() for k in ["full", "app", "website", "system", "complete"]): est = 120
    if any(k in goal.lower() for k in ["html", "page", "function", "script"]): est = 35
    current_state.estimated_duration_seconds = est
    current_state.time_remaining_seconds = est

    # Reset agents to working state for this real task
    for agent in current_state.agents:
        agent.status = "thinking" if agent.name != "Orchestrator" else "working"
        agent.current_task = f"Starting: {goal[:35]}"
        agent.progress = 5
        agent.last_update = datetime.now().isoformat()

    await manager.broadcast({
        "type": "new_goal",
        "goal": goal,
        "full_state": current_state.model_dump()
    })

    # Run the REAL swarm in background thread (non-blocking)
    async def _run_real_swarm():
        try:
            from agents.orchestrator import run_swarm
            from core.dashboard_events import emit_agent_update, emit_log, emit_metrics
            import asyncio

            start_time = time.time()

            def progress_tick():
                remaining = max(0, int(est - (time.time() - start_time)))
                current_state.time_remaining_seconds = remaining

            emit_log(f"🚀 Starting real swarm for: {goal[:60]}...", "info", "Orchestrator")
            emit_agent_update("Orchestrator", "working", f"Coordinating: {goal[:45]}", 15)

            # This is the actual call that does real work
            result = await asyncio.to_thread(run_swarm, goal, project_root=str(Path(__file__).parent.parent))

            duration = time.time() - start_time
            current_state.time_remaining_seconds = 0
            current_state.task_running = False
            current_state.task_completed = True
            current_state.real_result = result.output
            current_state.real_artifacts = result.artifacts or []
            current_state.tasks_completed += 1 if result.success else 0

            # Final real updates
            emit_log(f"✅ REAL TASK COMPLETE in {duration:.1f}s", "success", "Orchestrator")
            emit_metrics(tasks_completed=current_state.tasks_completed)
            for a in current_state.agents:
                a.status = "completed"
                a.progress = 100
                a.current_task = "Task finished - see real output"

            await manager.broadcast({
                "type": "task_complete",
                "full_state": current_state.model_dump(),
                "result": result.output,
                "artifacts": current_state.real_artifacts,
            })

        except Exception as e:
            current_state.task_running = False
            current_state.real_result = f"ERROR: {str(e)}"
            emit_log(f"❌ Swarm failed: {e}", "error", "Orchestrator")
            await manager.broadcast({"type": "task_error", "error": str(e), "full_state": current_state.model_dump()})

    asyncio.create_task(_run_real_swarm())
    asyncio.create_task(_countdown_timer(est))

    return {"message": "REAL swarm started", "goal": goal, "estimated_seconds": est}

@app.get("/api/agents")
async def get_agents():
    return current_state.agents

# --- WebSocket Endpoint (Real-time 3D updates) ---
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

# --- Real countdown timer ---
async def _countdown_timer(estimated: int):
    """Live reverse timer broadcast every second while task is running."""
    global current_state
    for _ in range(estimated + 5):
        if not current_state.task_running:
            break
        await asyncio.sleep(1)
        current_state.time_remaining_seconds = max(0, current_state.time_remaining_seconds - 1)
        await manager.broadcast({
            "type": "timer_tick",
            "time_remaining": current_state.time_remaining_seconds,
            "full_state": current_state.model_dump()
        })

# --- Background simulation (visual flair when idle) ---
async def simulate_agent_activity():
    """Only for visual flair when idle. Real tasks disable this."""
    agent_names = ["Architect", "Coder", "Tester", "Scribe"]
    statuses = ["thinking", "working", "completed"]
    while True:
        await asyncio.sleep(5)
        if not connected_clients or current_state.task_running:
            continue

        import random
        agent_name = random.choice(agent_names)
        new_status = random.choice(statuses)

        for i, a in enumerate(current_state.agents):
            if a.name == agent_name:
                a.status = new_status
                a.current_task = {
                    "Architect": "Designing system architecture for new feature",
                    "Coder": "Writing and editing source files with Claw tools",
                    "Tester": "Running pytest and verifying edge cases",
                    "Scribe": "Generating documentation and auto-skills"
                }[agent_name]
                a.progress = random.randint(20, 95) if new_status != "completed" else 100
                a.last_update = datetime.now().isoformat()
                break

        current_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {agent_name} → {new_status}")
        if len(current_state.logs) > 12:
            current_state.logs = current_state.logs[-12:]

        await manager.broadcast({
            "type": "agent_update",
            "data": current_state.agents[i].model_dump() if 'i' in locals() else {},
            "full_state": current_state.model_dump()
        })

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Kairos 3D Futuristic Dashboard API starting...")
    
    broadcaster = get_broadcaster()
    if broadcaster:
        broadcaster.register_listener(manager.broadcast)
        broadcaster.enable()
    
    asyncio.create_task(simulate_agent_activity())
    logger.info("3D Dashboard WebSocket ready at /ws/dashboard")

# For direct run
if __name__ == "__main__":
    uvicorn.run(
        "backend.dashboard_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )