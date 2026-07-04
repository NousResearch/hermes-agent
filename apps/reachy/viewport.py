"""Reachy Mini viewport computer — separate process, separate port."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi.staticfiles import StaticFiles

REPO = Path(__file__).resolve().parents[2]


def _repo_template(path: str) -> str:
    target = REPO / path
    if not target.exists():
        raise FileNotFoundError(f"template not found: {target}")
    return target.read_text(encoding="utf-8")


LOCAL_AI = _repo_template("templates/hermes-local-ai.html")
viewport_app = FastAPI(title="Hermes Viewport", version="0.1")
_STATIC_ROOT = REPO / "apps" / "reachy"
viewport_app.mount("/static", StaticFiles(directory=str(_STATIC_ROOT)), name="viewport-static")


@viewport_app.get("/", response_class=HTMLResponse)
async def index(_: Request) -> HTMLResponse:
    return HTMLResponse(LOCAL_AI)


@viewport_app.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "hermes-viewport"}


@viewport_app.post("/conductor")
async def conductor(request: Request) -> dict:
    body = await request.json()
    cmd = str(body.get("cmd", "")).strip()
    args = list(body.get("args", []) or [])
    if not cmd:
        return {"ok": False, "rc": 2, "stdout": "", "stderr": "missing cmd"}
    try:
        from hermes_cli.conductor import run as conductor_run
    except Exception as e:
        return {"ok": False, "rc": 3, "stdout": "", "stderr": f"conductor unavailable: {e}"}
    return conductor_run(cmd, args)


if __name__ == "__main__":
    uvicorn.run(viewport_app, host="0.0.0.0", port=5173)
