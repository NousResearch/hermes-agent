"""
Hermes Agent — Web UI server.

Provides a FastAPI backend serving the Vite/React frontend and REST API
endpoints for managing configuration, environment variables, and sessions.

Usage:
    python -m hermes_cli.main web          # Start on http://127.0.0.1:9119
    python -m hermes_cli.main web --port 8080
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli import __version__, __release_date__
from hermes_cli.config import (
    DEFAULT_CONFIG,
    OPTIONAL_ENV_VARS,
    get_config_path,
    get_env_path,
    get_hermes_home,
    load_config,
    load_env,
    save_config,
    save_env_value,
    delete_env_value,
    check_config_version,
    redact_key,
)
from gateway.status import get_running_pid, read_runtime_status

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    raise SystemExit(
        "Web UI requires fastapi and uvicorn.\n"
        "Run 'hermes web' to auto-install, or: pip install hermes-agent[web]"
    )

WEB_DIST = Path(__file__).parent / "web_dist"

app = FastAPI(title="Hermes Agent", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_SCHEMA = {
    "model": {
        "type": "string",
        "description": "Default model for chat",
        "category": "general",
    },
    "provider": {
        "type": "select",
        "description": "LLM provider",
        "options": ["auto", "openrouter", "nous", "anthropic", "openai", "codex", "custom"],
        "category": "general",
    },
    "system_prompt": {
        "type": "text",
        "description": "System prompt prepended to every conversation",
        "category": "general",
    },
    "toolsets": {
        "type": "list",
        "description": "Enabled toolsets",
        "category": "general",
    },
    "agent.max_turns": {
        "type": "number",
        "description": "Maximum agent turns per conversation",
        "category": "agent",
    },
    "terminal.backend": {
        "type": "select",
        "description": "Terminal execution backend",
        "options": ["local", "docker", "ssh", "modal", "daytona", "singularity"],
        "category": "terminal",
    },
    "terminal.timeout": {
        "type": "number",
        "description": "Command timeout (seconds)",
        "category": "terminal",
    },
    "terminal.cwd": {
        "type": "string",
        "description": "Working directory for terminal commands",
        "category": "terminal",
    },
    "browser.inactivity_timeout": {
        "type": "number",
        "description": "Browser inactivity timeout (seconds)",
        "category": "browser",
    },
    "compression.enabled": {
        "type": "boolean",
        "description": "Enable context compression",
        "category": "compression",
    },
    "compression.threshold": {
        "type": "number",
        "description": "Context window usage threshold to trigger compression (0-1)",
        "category": "compression",
    },
    "display.compact": {
        "type": "boolean",
        "description": "Compact display mode",
        "category": "display",
    },
    "display.personality": {
        "type": "select",
        "description": "Agent personality",
        "options": ["kawaii", "professional", "minimal", "hacker"],
        "category": "display",
    },
    "display.show_reasoning": {
        "type": "boolean",
        "description": "Show model reasoning/thinking",
        "category": "display",
    },
    "display.bell_on_complete": {
        "type": "boolean",
        "description": "Ring terminal bell when agent finishes",
        "category": "display",
    },
    "tts.provider": {
        "type": "select",
        "description": "Text-to-speech provider",
        "options": ["edge", "elevenlabs", "openai"],
        "category": "tts",
    },
    "checkpoints.enabled": {
        "type": "boolean",
        "description": "Enable filesystem checkpoints before destructive ops",
        "category": "checkpoints",
    },
    "checkpoints.max_snapshots": {
        "type": "number",
        "description": "Max checkpoint snapshots per directory",
        "category": "checkpoints",
    },
}


class ConfigUpdate(BaseModel):
    config: dict


class EnvVarUpdate(BaseModel):
    key: str
    value: str


class EnvVarDelete(BaseModel):
    key: str


@app.get("/api/status")
async def get_status():
    current_ver, latest_ver = check_config_version()

    gateway_pid = get_running_pid()
    gateway_running = gateway_pid is not None

    gateway_state = None
    gateway_platforms: dict = {}
    gateway_exit_reason = None
    gateway_updated_at = None
    runtime = read_runtime_status()
    if runtime:
        gateway_state = runtime.get("gateway_state")
        gateway_platforms = runtime.get("platforms") or {}
        gateway_exit_reason = runtime.get("exit_reason")
        gateway_updated_at = runtime.get("updated_at")
        if not gateway_running:
            gateway_state = gateway_state if gateway_state in ("stopped", "startup_failed") else "stopped"

    active_sessions = 0
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        sessions = db.list_sessions_rich(limit=50)
        now = time.time()
        active_sessions = sum(
            1 for s in sessions
            if s.get("ended_at") is None
            and (now - s.get("last_active", s.get("started_at", 0))) < 300
        )
    except Exception:
        pass

    return {
        "version": __version__,
        "release_date": __release_date__,
        "hermes_home": str(get_hermes_home()),
        "config_path": str(get_config_path()),
        "env_path": str(get_env_path()),
        "config_version": current_ver,
        "latest_config_version": latest_ver,
        "gateway_running": gateway_running,
        "gateway_pid": gateway_pid,
        "gateway_state": gateway_state,
        "gateway_platforms": gateway_platforms,
        "gateway_exit_reason": gateway_exit_reason,
        "gateway_updated_at": gateway_updated_at,
        "active_sessions": active_sessions,
    }


@app.get("/api/sessions")
async def get_sessions():
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        sessions = db.list_sessions_rich(limit=20)
        now = time.time()
        for s in sessions:
            s["is_active"] = (
                s.get("ended_at") is None
                and (now - s.get("last_active", s.get("started_at", 0))) < 300
            )
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config():
    return load_config()


@app.get("/api/config/defaults")
async def get_defaults():
    return DEFAULT_CONFIG


@app.get("/api/config/schema")
async def get_schema():
    return CONFIG_SCHEMA


@app.put("/api/config")
async def update_config(body: ConfigUpdate):
    try:
        save_config(body.config)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/env")
async def get_env_vars():
    env_on_disk = load_env()
    result = {}
    for var_name, info in OPTIONAL_ENV_VARS.items():
        value = env_on_disk.get(var_name)
        result[var_name] = {
            "is_set": bool(value),
            "redacted_value": redact_key(value) if value else None,
            "description": info.get("description", ""),
            "url": info.get("url"),
            "category": info.get("category", ""),
            "is_password": info.get("password", False),
            "tools": info.get("tools", []),
            "advanced": info.get("advanced", False),
        }
    return result


@app.put("/api/env")
async def set_env_var(body: EnvVarUpdate):
    try:
        save_env_value(body.key, body.value)
        return {"ok": True, "key": body.key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/env")
async def remove_env_var(body: EnvVarDelete):
    try:
        removed = delete_env_value(body.key)
        if not removed:
            raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")
        return {"ok": True, "key": body.key}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def mount_spa(application: FastAPI):
    """Mount the built SPA. Falls back to index.html for client-side routing."""
    if not WEB_DIST.exists():
        @application.get("/{full_path:path}")
        async def no_frontend(full_path: str):
            return JSONResponse(
                {"error": "Frontend not built. Run: cd web && npm run build"},
                status_code=404,
            )
        return

    application.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @application.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = WEB_DIST / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(WEB_DIST / "index.html")


mount_spa(app)


def start_server(host: str = "127.0.0.1", port: int = 9119, open_browser: bool = True):
    """Start the web UI server."""
    import uvicorn

    if open_browser:
        import threading
        import webbrowser

        def _open():
            import time as _t
            _t.sleep(1.0)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open, daemon=True).start()

    print(f"  Hermes Web UI → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
