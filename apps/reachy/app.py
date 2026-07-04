"""Reachy Mini local-first operator surface for HuggingFace Space."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from apps.reachy.sim_bridge import router as sim_router
from apps.reachy.vscode import router as vscode_router
from apps.reachy.mcp_tools import router as mcp_tools_router
from apps.reachy.ae_coding_conductor import router as ae_coding_router
from hermes_cli.conductor import run_hermes

REPO = Path(__file__).resolve().parents[2]


def require_template(path: str) -> str:
    target = REPO / path
    if not target.exists():
        raise FileNotFoundError(f"template not found: {target}")
    return target.read_text(encoding="utf-8")


INDEX = require_template("templates/hermes-reachy-marketplace.html")
LOCAL_AI = require_template("templates/hermes-local-ai.html")


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class ReachyState:
    reachable: bool = False
    effector: str = "idle"
    telemetry: dict[str, object] | None = None
    updated: str = now_iso()


state = ReachyState()
app = FastAPI(title="Hermes Reachy", version="0.3")
app.include_router(sim_router)
app.include_router(vscode_router)
app.include_router(mcp_tools_router)
app.include_router(ae_coding_router)
from pathlib import Path
STATIC_ROOT = Path(__file__).resolve().parents[2] / "apps" / "reachy"
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="reachy-static")


@app.get("/", response_class=HTMLResponse)
async def index(_: Request) -> HTMLResponse:
    return HTMLResponse(INDEX)


@app.get("/local-ai", response_class=HTMLResponse)
async def local_ai(_: Request) -> HTMLResponse:
    return HTMLResponse(LOCAL_AI)


@app.post("/reachy/probe")
async def reachy_probe() -> JSONResponse:
    state.reachable = True
    state.effector = "idle"
    state.telemetry = {"command": "probe", "surface": "local"}
    state.updated = now_iso()
    return JSONResponse({"ok": True, "state": asdict(state)})


@app.post("/reachy/panel")
async def reachy_panel() -> JSONResponse:
    state.reachable = True
    state.effector = "panel_open"
    state.telemetry = {"command": "panel", "surface": "local"}
    state.updated = now_iso()
    return JSONResponse({"ok": True, "state": asdict(state)})


@app.post("/reachy/status")
async def reachy_status() -> JSONResponse:
    return JSONResponse({"ok": True, "state": asdict(state)})


@app.post("/conductor")
async def conductor(payload: dict) -> JSONResponse:
    result = run_hermes(payload)
    return JSONResponse(result)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "hermes-reachy"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
