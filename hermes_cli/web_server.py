"""
Hermes Agent — Web UI server.

Provides a FastAPI backend serving the Vite/React frontend and REST API
endpoints for managing configuration, environment variables, and sessions.

Usage:
    python -m hermes_cli.main web          # Start on http://127.0.0.1:9119
    python -m hermes_cli.main web --port 8080
"""

import logging
import os
import secrets
import sys
import time
import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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
    remove_env_value,
    check_config_version,
    redact_key,
)
from gateway.status import get_running_pid, read_runtime_status

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    raise SystemExit(
        "Web UI requires fastapi and uvicorn.\n"
        "Run 'hermes web' to auto-install, or: pip install hermes-agent[web]"
    )

WEB_DIST = Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)

app = FastAPI(title="Hermes Agent", version=__version__)

# ---------------------------------------------------------------------------
# Session token for protecting sensitive endpoints (reveal).
# Generated fresh on every server start — dies when the process exits.
# Injected into the SPA HTML so only the legitimate web UI can use it.
# ---------------------------------------------------------------------------
_SESSION_TOKEN = secrets.token_urlsafe(32)

# Simple rate limiter for the reveal endpoint
_reveal_timestamps: List[float] = []
_REVEAL_MAX_PER_WINDOW = 5
_REVEAL_WINDOW_SECONDS = 30

# Desk runs are local, browser-initiated agent turns.  They intentionally live
# in this process: if the web server stops, active runs stop too.
_DESK_RUNS: Dict[str, Dict[str, Any]] = {}
_DESK_ENV_LOCK = threading.Lock()

# CORS: restrict to localhost origins only.  The web UI is intended to run
# locally; binding to 0.0.0.0 with allow_origins=["*"] would let any website
# read/modify config and secrets.

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Config schema — auto-generated from DEFAULT_CONFIG
# ---------------------------------------------------------------------------

# Manual overrides for fields that need select options or custom types
_SCHEMA_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "model": {
        "type": "string",
        "description": "Default model (e.g. anthropic/claude-sonnet-4.6)",
        "category": "general",
    },
    "terminal.backend": {
        "type": "select",
        "description": "Terminal execution backend",
        "options": ["local", "docker", "ssh", "modal", "daytona", "singularity"],
    },
    "terminal.modal_mode": {
        "type": "select",
        "description": "Modal sandbox mode",
        "options": ["sandbox", "function"],
    },
    "tts.provider": {
        "type": "select",
        "description": "Text-to-speech provider",
        "options": ["edge", "elevenlabs", "openai", "neutts"],
    },
    "stt.provider": {
        "type": "select",
        "description": "Speech-to-text provider",
        "options": ["local", "openai", "mistral"],
    },
    "display.skin": {
        "type": "select",
        "description": "CLI visual theme",
        "options": ["default", "ares", "mono", "slate"],
    },
    "display.resume_display": {
        "type": "select",
        "description": "How resumed sessions display history",
        "options": ["minimal", "full", "off"],
    },
    "display.busy_input_mode": {
        "type": "select",
        "description": "Input behavior while agent is running",
        "options": ["queue", "interrupt", "block"],
    },
    "memory.provider": {
        "type": "select",
        "description": "Memory provider plugin",
        "options": ["builtin", "honcho"],
    },
    "approvals.mode": {
        "type": "select",
        "description": "Dangerous command approval mode",
        "options": ["ask", "yolo", "deny"],
    },
    "context.engine": {
        "type": "select",
        "description": "Context management engine",
        "options": ["default", "custom"],
    },
    "human_delay.mode": {
        "type": "select",
        "description": "Simulated typing delay mode",
        "options": ["off", "typing", "fixed"],
    },
    "logging.level": {
        "type": "select",
        "description": "Log level for agent.log",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
    },
    "agent.service_tier": {
        "type": "select",
        "description": "API service tier (OpenAI/Anthropic)",
        "options": ["", "auto", "default", "flex"],
    },
    "delegation.reasoning_effort": {
        "type": "select",
        "description": "Reasoning effort for delegated subagents",
        "options": ["", "low", "medium", "high"],
    },
}

# Categories with fewer fields get merged into "general" to avoid tab sprawl.
_CATEGORY_MERGE: Dict[str, str] = {
    "privacy": "security",
    "context": "agent",
    "skills": "agent",
    "cron": "agent",
    "network": "agent",
    "checkpoints": "agent",
    "approvals": "security",
    "human_delay": "display",
    "smart_model_routing": "agent",
}

# Display order for tabs — unlisted categories sort alphabetically after these.
_CATEGORY_ORDER = [
    "general", "agent", "terminal", "display", "delegation",
    "memory", "compression", "security", "browser", "voice",
    "tts", "stt", "logging", "discord", "auxiliary",
]


def _infer_type(value: Any) -> str:
    """Infer a UI field type from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _build_schema_from_config(
    config: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Walk DEFAULT_CONFIG and produce a flat dot-path → field schema dict."""
    schema: Dict[str, Dict[str, Any]] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        # Skip internal / version keys
        if full_key in ("_config_version",):
            continue

        # Category is the first path component for nested keys, or "general"
        # for top-level scalar fields (model, toolsets, timezone, etc.).
        if prefix:
            category = prefix.split(".")[0]
        elif isinstance(value, dict):
            category = key
        else:
            category = "general"

        if isinstance(value, dict):
            # Recurse into nested dicts
            schema.update(_build_schema_from_config(value, full_key))
        else:
            entry: Dict[str, Any] = {
                "type": _infer_type(value),
                "description": full_key.replace(".", " → ").replace("_", " ").title(),
                "category": category,
            }
            # Apply manual overrides
            if full_key in _SCHEMA_OVERRIDES:
                entry.update(_SCHEMA_OVERRIDES[full_key])
            # Merge small categories
            entry["category"] = _CATEGORY_MERGE.get(entry["category"], entry["category"])
            schema[full_key] = entry
    return schema


CONFIG_SCHEMA = _build_schema_from_config(DEFAULT_CONFIG)


class ConfigUpdate(BaseModel):
    config: dict


class EnvVarUpdate(BaseModel):
    key: str
    value: str


class EnvVarDelete(BaseModel):
    key: str


class EnvVarReveal(BaseModel):
    key: str


class ProjectCreate(BaseModel):
    name: str = ""
    path: str


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    path: Optional[str] = None
    notes: Optional[str] = None
    default_model: Optional[str] = None
    default_toolsets: Optional[List[str]] = None


class DeskSessionCreate(BaseModel):
    project_id: Optional[str] = None
    title: Optional[str] = None
    model: Optional[str] = None
    toolsets: Optional[List[str]] = None


class DeskRunCreate(BaseModel):
    session_id: str
    message: str


class BuiltinMemoryUpdate(BaseModel):
    target: str
    entries: List[str]


def _projects_path() -> Path:
    return get_hermes_home() / "projects.json"


def _read_projects() -> List[Dict[str, Any]]:
    path = _projects_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            projects = data.get("projects", [])
        else:
            projects = data
        return [p for p in projects if isinstance(p, dict)]
    except Exception:
        _log.exception("Failed to read projects registry")
        return []


def _write_projects(projects: List[Dict[str, Any]]) -> None:
    path = _projects_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"projects": projects}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _resolve_project_path(raw_path: str) -> Path:
    if not raw_path or not raw_path.strip():
        raise HTTPException(status_code=400, detail="Project path is required")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise HTTPException(status_code=400, detail="Project path must be absolute")
    try:
        path = path.resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail="Project path must be an existing directory")
    return path


def _project_metadata(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    exists = path.exists() and path.is_dir()
    metadata: Dict[str, Any] = {
        "exists": exists,
        "git_branch": None,
        "git_dirty": None,
        "context_files": [],
    }
    if not exists:
        return metadata

    for name in ("AGENTS.md", "agents.md", "HERMES.md", ".hermes.md", "CLAUDE.md", ".cursorrules"):
        if (path / name).is_file():
            metadata["context_files"].append(name)
    if (path / ".cursor" / "rules").is_dir():
        metadata["context_files"].append(".cursor/rules")

    try:
        import subprocess

        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if branch.returncode == 0:
            metadata["git_branch"] = branch.stdout.strip() or None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if dirty.returncode == 0:
            metadata["git_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        pass
    return metadata


def _project_with_metadata(project: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(project)
    enriched["metadata"] = _project_metadata(str(project.get("path", "")))
    return enriched


def _find_project(project_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not project_id:
        return None
    for project in _read_projects():
        if project.get("id") == project_id:
            return project
    return None


def _memory_store_snapshot() -> Dict[str, Any]:
    from tools.memory_tool import MemoryStore

    store = MemoryStore()
    store.load_from_disk()
    memory_usage = len("\n§\n".join(store.memory_entries)) if store.memory_entries else 0
    user_usage = len("\n§\n".join(store.user_entries)) if store.user_entries else 0
    return {
        "memory": {
            "entries": store.memory_entries,
            "usage": memory_usage,
            "limit": store.memory_char_limit,
        },
        "user": {
            "entries": store.user_entries,
            "usage": user_usage,
            "limit": store.user_char_limit,
        },
    }


async def _emit_desk_sse(queue: "asyncio.Queue[Optional[Dict[str, Any]]]"):
    while True:
        event = await queue.get()
        if event is None:
            yield ": stream closed\n\n"
            return
        yield f"data: {json.dumps(event, default=str)}\n\n"


def _desk_agent_kwargs(session_id: str, project: Optional[Dict[str, Any]], body: DeskSessionCreate | None = None) -> Dict[str, Any]:
    from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model
    from hermes_cli.tools_config import _get_platform_tools

    config = load_config()
    runtime_kwargs = _resolve_runtime_agent_kwargs()
    model = (body.model if body and body.model else None) or (project or {}).get("default_model") or _resolve_gateway_model(config)
    toolsets = body.toolsets if body and body.toolsets else (project or {}).get("default_toolsets")
    if not toolsets:
        toolsets = sorted(_get_platform_tools(config, "cli"))
    return {
        "model": model,
        "runtime_kwargs": runtime_kwargs,
        "enabled_toolsets": toolsets,
        "model_config": {
            "project_id": (project or {}).get("id"),
            "project_path": (project or {}).get("path"),
            "toolsets": toolsets,
        },
    }


@app.get("/api/status")
async def get_status():
    current_ver, latest_ver = check_config_version()

    gateway_pid = get_running_pid()
    gateway_running = gateway_pid is not None

    gateway_state = None
    gateway_platforms: dict = {}
    gateway_exit_reason = None
    gateway_updated_at = None
    configured_gateway_platforms: set[str] | None = None
    try:
        from gateway.config import load_gateway_config

        gateway_config = load_gateway_config()
        configured_gateway_platforms = {
            platform.value for platform in gateway_config.get_connected_platforms()
        }
    except Exception:
        configured_gateway_platforms = None

    runtime = read_runtime_status()
    if runtime:
        gateway_state = runtime.get("gateway_state")
        gateway_platforms = runtime.get("platforms") or {}
        if configured_gateway_platforms is not None:
            gateway_platforms = {
                key: value
                for key, value in gateway_platforms.items()
                if key in configured_gateway_platforms
            }
        gateway_exit_reason = runtime.get("exit_reason")
        gateway_updated_at = runtime.get("updated_at")
        if not gateway_running:
            gateway_state = gateway_state if gateway_state in ("stopped", "startup_failed") else "stopped"
            gateway_platforms = {}

    active_sessions = 0
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            sessions = db.list_sessions_rich(limit=50)
            now = time.time()
            active_sessions = sum(
                1 for s in sessions
                if s.get("ended_at") is None
                and (now - s.get("last_active", s.get("started_at", 0))) < 300
            )
        finally:
            db.close()
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
        try:
            sessions = db.list_sessions_rich(limit=20)
            now = time.time()
            for s in sessions:
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
            return sessions
        finally:
            db.close()
    except Exception as e:
        _log.exception("GET /api/sessions failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/sessions/search")
async def search_sessions(q: str = "", limit: int = 20):
    """Full-text search across session message content using FTS5."""
    if not q or not q.strip():
        return {"results": []}
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            # Auto-add prefix wildcards so partial words match
            # e.g. "nimb" → "nimb*" matches "nimby"
            # Preserve quoted phrases and existing wildcards as-is
            import re
            terms = []
            for token in re.findall(r'"[^"]*"|\S+', q.strip()):
                if token.startswith('"') or token.endswith("*"):
                    terms.append(token)
                else:
                    terms.append(token + "*")
            prefix_query = " ".join(terms)
            matches = db.search_messages(query=prefix_query, limit=limit)
            # Group by session_id — return unique sessions with their best snippet
            seen: dict = {}
            for m in matches:
                sid = m["session_id"]
                if sid not in seen:
                    seen[sid] = {
                        "session_id": sid,
                        "snippet": m.get("snippet", ""),
                        "role": m.get("role"),
                        "source": m.get("source"),
                        "model": m.get("model"),
                        "session_started": m.get("session_started"),
                    }
            return {"results": list(seen.values())}
        finally:
            db.close()
    except Exception:
        _log.exception("GET /api/sessions/search failed")
        raise HTTPException(status_code=500, detail="Search failed")


def _normalize_config_for_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config for the web UI.

    Hermes supports ``model`` as either a bare string (``"anthropic/claude-sonnet-4"``)
    or a dict (``{default: ..., provider: ..., base_url: ...}``).  The schema is built
    from DEFAULT_CONFIG where ``model`` is a string, but user configs often have the
    dict form.  Normalize to the string form so the frontend schema matches.
    """
    config = dict(config)  # shallow copy
    model_val = config.get("model")
    if isinstance(model_val, dict):
        config["model"] = model_val.get("default", model_val.get("name", ""))
    return config


@app.get("/api/config")
async def get_config():
    config = _normalize_config_for_web(load_config())
    # Strip internal keys that the frontend shouldn't see or send back
    return {k: v for k, v in config.items() if not k.startswith("_")}


@app.get("/api/config/defaults")
async def get_defaults():
    return DEFAULT_CONFIG


@app.get("/api/config/schema")
async def get_schema():
    return {"fields": CONFIG_SCHEMA, "category_order": _CATEGORY_ORDER}


def _denormalize_config_from_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse _normalize_config_for_web before saving.

    Reconstructs ``model`` as a dict by reading the current on-disk config
    to recover model subkeys (provider, base_url, api_mode, etc.) that were
    stripped from the GET response.  The frontend only sees model as a flat
    string; the rest is preserved transparently.
    """
    config = dict(config)
    # Remove any _model_meta that might have leaked in (shouldn't happen
    # with the stripped GET response, but be defensive)
    config.pop("_model_meta", None)

    model_val = config.get("model")
    if isinstance(model_val, str) and model_val:
        # Read the current disk config to recover model subkeys
        try:
            disk_config = load_config()
            disk_model = disk_config.get("model")
            if isinstance(disk_model, dict):
                # Preserve all subkeys, update default with the new value
                disk_model["default"] = model_val
                config["model"] = disk_model
        except Exception:
            pass  # can't read disk config — just use the string form
    return config


@app.put("/api/config")
async def update_config(body: ConfigUpdate):
    try:
        save_config(_denormalize_config_from_web(body.config))
        return {"ok": True}
    except Exception as e:
        _log.exception("PUT /api/config failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/auth/session-token")
async def get_session_token():
    """Return the ephemeral session token for this server instance.

    The token protects sensitive endpoints (reveal).  It's served to the SPA
    which stores it in memory — it's never persisted and dies when the server
    process exits.  CORS already restricts this to localhost origins.
    """
    return {"token": _SESSION_TOKEN}


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
        _log.exception("PUT /api/env failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/env")
async def remove_env_var(body: EnvVarDelete):
    try:
        removed = remove_env_value(body.key)
        if not removed:
            raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")
        return {"ok": True, "key": body.key}
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("DELETE /api/env failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/env/reveal")
async def reveal_env_var(body: EnvVarReveal, request: Request):
    """Return the real (unredacted) value of a single env var.

    Protected by:
    - Ephemeral session token (generated per server start, injected into SPA)
    - Rate limiting (max 5 reveals per 30s window)
    - Audit logging
    """
    # --- Token check ---
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {_SESSION_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # --- Rate limit ---
    now = time.time()
    cutoff = now - _REVEAL_WINDOW_SECONDS
    _reveal_timestamps[:] = [t for t in _reveal_timestamps if t > cutoff]
    if len(_reveal_timestamps) >= _REVEAL_MAX_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many reveal requests. Try again shortly.")
    _reveal_timestamps.append(now)

    # --- Reveal ---
    env_on_disk = load_env()
    value = env_on_disk.get(body.key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")

    _log.info("env/reveal: %s", body.key)
    return {"key": body.key, "value": value}


# ---------------------------------------------------------------------------
# Session detail endpoints
# ---------------------------------------------------------------------------


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        sid = db.resolve_session_id(session_id)
        session = db.get_session(sid) if sid else None
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    finally:
        db.close()


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = db.get_messages(sid)
        return {"session_id": sid, "messages": messages}
    finally:
        db.close()


@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        if not db.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Desk projects, runs, and memory endpoints
# ---------------------------------------------------------------------------


@app.get("/api/projects")
async def list_projects():
    projects = sorted(_read_projects(), key=lambda p: str(p.get("updated_at", "")), reverse=True)
    return {"projects": [_project_with_metadata(p) for p in projects]}


@app.post("/api/projects")
async def create_project(body: ProjectCreate):
    project_path = _resolve_project_path(body.path)
    now = time.time()
    projects = _read_projects()
    for project in projects:
        if project.get("path") == str(project_path):
            return _project_with_metadata(project)

    project = {
        "id": f"proj_{uuid.uuid4().hex[:12]}",
        "name": body.name.strip() or project_path.name or str(project_path),
        "path": str(project_path),
        "notes": "",
        "default_model": "",
        "default_toolsets": [],
        "last_session_id": "",
        "created_at": now,
        "updated_at": now,
    }
    projects.append(project)
    _write_projects(projects)
    return _project_with_metadata(project)


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, body: ProjectUpdate):
    projects = _read_projects()
    for project in projects:
        if project.get("id") != project_id:
            continue
        if body.name is not None:
            project["name"] = body.name.strip() or project.get("name") or "Untitled Project"
        if body.path is not None:
            project["path"] = str(_resolve_project_path(body.path))
        if body.notes is not None:
            project["notes"] = body.notes
        if body.default_model is not None:
            project["default_model"] = body.default_model
        if body.default_toolsets is not None:
            project["default_toolsets"] = [str(t) for t in body.default_toolsets if str(t).strip()]
        project["updated_at"] = time.time()
        _write_projects(projects)
        return _project_with_metadata(project)
    raise HTTPException(status_code=404, detail="Project not found")


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    projects = _read_projects()
    remaining = [p for p in projects if p.get("id") != project_id]
    if len(remaining) == len(projects):
        raise HTTPException(status_code=404, detail="Project not found")
    _write_projects(remaining)
    return {"ok": True}


@app.post("/api/desk/sessions")
async def create_desk_session(body: DeskSessionCreate):
    from hermes_state import SessionDB

    project = _find_project(body.project_id)
    if body.project_id and not project:
        raise HTTPException(status_code=404, detail="Project not found")

    session_id = f"desk_{uuid.uuid4().hex}"
    kwargs = _desk_agent_kwargs(session_id, project, body)
    db = SessionDB()
    try:
        db.create_session(
            session_id=session_id,
            source="desk",
            model=kwargs["model"],
            model_config=kwargs["model_config"],
        )
        if body.title:
            db.set_session_title(session_id, body.title)
    finally:
        db.close()

    if project:
        projects = _read_projects()
        for item in projects:
            if item.get("id") == project.get("id"):
                item["last_session_id"] = session_id
                item["updated_at"] = time.time()
                break
        _write_projects(projects)

    return {"session_id": session_id, "source": "desk", "project": project}


@app.post("/api/desk/runs")
async def start_desk_run(body: DeskRunCreate):
    from hermes_state import SessionDB

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    db = SessionDB()
    try:
        session = db.get_session(body.session_id)
    finally:
        db.close()
    if not session or session.get("source") != "desk":
        raise HTTPException(status_code=404, detail="Desk session not found")

    run_id = f"run_{uuid.uuid4().hex}"
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
    agent_ref: Dict[str, Any] = {"agent": None}
    _DESK_RUNS[run_id] = {
        "queue": queue,
        "session_id": body.session_id,
        "agent_ref": agent_ref,
        "started_at": time.time(),
    }

    async def _run_agent_task():
        def _push(event: Dict[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception:
                pass

        def _on_delta(delta: Optional[str]) -> None:
            if delta:
                _push({
                    "event": "message.delta",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "delta": delta,
                })

        def _on_tool(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
            if event_type == "tool.started":
                _push({
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "tool": tool_name,
                    "preview": preview or "",
                })
            elif event_type == "tool.completed":
                _push({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "tool": tool_name,
                    "duration": round(float(kwargs.get("duration", 0) or 0), 3),
                    "error": bool(kwargs.get("is_error", False)),
                })
            elif event_type == "reasoning.available":
                _push({
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "text": preview or "",
                })

        def _run_sync():
            from run_agent import AIAgent
            from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model
            from hermes_cli.tools_config import _get_platform_tools
            from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides

            run_db = SessionDB()
            terminal_cwd_previous = os.environ.get("TERMINAL_CWD")
            try:
                fresh_session = run_db.get_session(body.session_id) or session
                model_config = {}
                if fresh_session.get("model_config"):
                    try:
                        model_config = json.loads(fresh_session["model_config"])
                    except (TypeError, json.JSONDecodeError):
                        model_config = {}
                project_path = model_config.get("project_path") or None
                toolsets = model_config.get("toolsets") or sorted(_get_platform_tools(load_config(), "cli"))
                model = fresh_session.get("model") or _resolve_gateway_model(load_config())
                history = run_db.get_messages_as_conversation(body.session_id)

                if project_path:
                    register_task_env_overrides(body.session_id, {"cwd": project_path})

                with _DESK_ENV_LOCK:
                    if project_path:
                        os.environ["TERMINAL_CWD"] = project_path
                    elif terminal_cwd_previous is None:
                        os.environ.pop("TERMINAL_CWD", None)
                    runtime_kwargs = _resolve_runtime_agent_kwargs()
                    agent = AIAgent(
                        model=model,
                        **runtime_kwargs,
                        max_iterations=int(os.getenv("HERMES_MAX_ITERATIONS", "90")),
                        quiet_mode=True,
                        verbose_logging=False,
                        enabled_toolsets=toolsets,
                        session_id=body.session_id,
                        platform="desk",
                        stream_delta_callback=_on_delta,
                        tool_progress_callback=_on_tool,
                        session_db=run_db,
                    )
                    agent_ref["agent"] = agent
                    return agent.run_conversation(
                        user_message=message,
                        conversation_history=history,
                        task_id=body.session_id,
                    )
            finally:
                if terminal_cwd_previous is None:
                    os.environ.pop("TERMINAL_CWD", None)
                else:
                    os.environ["TERMINAL_CWD"] = terminal_cwd_previous
                clear_task_env_overrides(body.session_id)
                run_db.close()

        try:
            result = await loop.run_in_executor(None, _run_sync)
            _push({
                "event": "run.completed",
                "run_id": run_id,
                "timestamp": time.time(),
                "output": result.get("final_response", "") if isinstance(result, dict) else "",
            })
        except Exception as exc:
            _log.exception("Desk run failed")
            _push({
                "event": "run.failed",
                "run_id": run_id,
                "timestamp": time.time(),
                "error": str(exc),
            })
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(_run_agent_task())
    _DESK_RUNS[run_id]["task"] = task
    return {"run_id": run_id, "status": "started"}


@app.get("/api/desk/runs/{run_id}/events")
async def stream_desk_run_events(run_id: str):
    run = _DESK_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    async def _stream():
        try:
            async for chunk in _emit_desk_sse(run["queue"]):
                yield chunk
        finally:
            task = run.get("task")
            if task and task.done():
                _DESK_RUNS.pop(run_id, None)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/desk/runs/{run_id}/interrupt")
async def interrupt_desk_run(run_id: str):
    run = _DESK_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    agent = run.get("agent_ref", {}).get("agent")
    if agent is not None:
        try:
            agent.interrupt("Desk interrupt requested")
        except Exception:
            _log.exception("Failed to interrupt Desk run")
    return {"ok": True}


@app.get("/api/memory/builtin")
async def get_builtin_memory():
    return _memory_store_snapshot()


@app.put("/api/memory/builtin")
async def update_builtin_memory(body: BuiltinMemoryUpdate):
    if body.target not in ("memory", "user"):
        raise HTTPException(status_code=400, detail="target must be 'memory' or 'user'")
    from tools.memory_tool import ENTRY_DELIMITER, MemoryStore, _scan_memory_content

    store = MemoryStore()
    store.load_from_disk()
    clean_entries: List[str] = []
    for entry in body.entries:
        clean = str(entry).strip()
        if not clean:
            continue
        scan_error = _scan_memory_content(clean)
        if scan_error:
            raise HTTPException(status_code=400, detail=scan_error)
        if clean not in clean_entries:
            clean_entries.append(clean)
    limit = store.user_char_limit if body.target == "user" else store.memory_char_limit
    total = len(ENTRY_DELIMITER.join(clean_entries)) if clean_entries else 0
    if total > limit:
        raise HTTPException(status_code=400, detail=f"Memory would exceed limit ({total}/{limit} chars)")
    store._set_entries(body.target, clean_entries)
    store.save_to_disk(body.target)
    return _memory_store_snapshot()


@app.get("/api/memory/providers")
async def get_memory_providers():
    config = load_config()
    memory_cfg = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
    provider = memory_cfg.get("provider", "builtin")
    return {
        "active_provider": provider or "builtin",
        "builtin_enabled": bool(memory_cfg.get("memory_enabled") or memory_cfg.get("user_profile_enabled")),
        "external_configured": bool(provider and provider != "builtin"),
    }


# ---------------------------------------------------------------------------
# Log viewer endpoint
# ---------------------------------------------------------------------------


@app.get("/api/logs")
async def get_logs(
    file: str = "agent",
    lines: int = 100,
    level: Optional[str] = None,
    component: Optional[str] = None,
):
    from hermes_cli.logs import _read_tail, LOG_FILES

    log_name = LOG_FILES.get(file)
    if not log_name:
        raise HTTPException(status_code=400, detail=f"Unknown log file: {file}")
    log_path = get_hermes_home() / "logs" / log_name
    if not log_path.exists():
        return {"file": file, "lines": []}

    try:
        from hermes_logging import COMPONENT_PREFIXES
    except ImportError:
        COMPONENT_PREFIXES = {}

    has_filters = bool(level or component)
    comp_prefixes = COMPONENT_PREFIXES.get(component, ()) if component else ()
    result = _read_tail(
        log_path, min(lines, 500),
        has_filters=has_filters,
        min_level=level,
        component_prefixes=comp_prefixes,
    )
    return {"file": file, "lines": result}


# ---------------------------------------------------------------------------
# Cron job management endpoints
# ---------------------------------------------------------------------------


class CronJobCreate(BaseModel):
    prompt: str
    schedule: str
    name: str = ""
    deliver: str = "local"


class CronJobUpdate(BaseModel):
    updates: dict


@app.get("/api/cron/jobs")
async def list_cron_jobs():
    from cron.jobs import list_jobs
    return list_jobs(include_disabled=True)


@app.get("/api/cron/jobs/{job_id}")
async def get_cron_job(job_id: str):
    from cron.jobs import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs")
async def create_cron_job(body: CronJobCreate):
    from cron.jobs import create_job
    try:
        job = create_job(prompt=body.prompt, schedule=body.schedule,
                         name=body.name, deliver=body.deliver)
        return job
    except Exception as e:
        _log.exception("POST /api/cron/jobs failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate):
    from cron.jobs import update_job
    job = update_job(job_id, body.updates)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/pause")
async def pause_cron_job(job_id: str):
    from cron.jobs import pause_job
    job = pause_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/resume")
async def resume_cron_job(job_id: str):
    from cron.jobs import resume_job
    job = resume_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/trigger")
async def trigger_cron_job(job_id: str):
    from cron.jobs import trigger_job
    job = trigger_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str):
    from cron.jobs import remove_job
    if not remove_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Skills & Tools endpoints
# ---------------------------------------------------------------------------


class SkillToggle(BaseModel):
    name: str
    enabled: bool


@app.get("/api/skills")
async def get_skills():
    from tools.skills_tool import _find_all_skills
    from hermes_cli.skills_config import get_disabled_skills
    config = load_config()
    disabled = get_disabled_skills(config)
    skills = _find_all_skills(skip_disabled=True)
    for s in skills:
        s["enabled"] = s["name"] not in disabled
    return skills


@app.put("/api/skills/toggle")
async def toggle_skill(body: SkillToggle):
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    config = load_config()
    disabled = get_disabled_skills(config)
    if body.enabled:
        disabled.discard(body.name)
    else:
        disabled.add(body.name)
    save_disabled_skills(config, disabled)
    return {"ok": True, "name": body.name, "enabled": body.enabled}


@app.get("/api/tools/toolsets")
async def get_toolsets():
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        _get_platform_tools,
        _toolset_has_keys,
    )
    from toolsets import resolve_toolset

    config = load_config()
    enabled_toolsets = _get_platform_tools(
        config,
        "cli",
        include_default_mcp_servers=False,
    )
    result = []
    for name, label, desc in _get_effective_configurable_toolsets():
        try:
            tools = sorted(set(resolve_toolset(name)))
        except Exception:
            tools = []
        is_enabled = name in enabled_toolsets
        result.append({
            "name": name, "label": label, "description": desc,
            "enabled": is_enabled,
            "available": is_enabled,
            "configured": _toolset_has_keys(name, config),
            "tools": tools,
        })
    return result


# ---------------------------------------------------------------------------
# Raw YAML config endpoint
# ---------------------------------------------------------------------------


class RawConfigUpdate(BaseModel):
    yaml_text: str


@app.get("/api/config/raw")
async def get_config_raw():
    path = get_config_path()
    if not path.exists():
        return {"yaml": ""}
    return {"yaml": path.read_text(encoding="utf-8")}


@app.put("/api/config/raw")
async def update_config_raw(body: RawConfigUpdate):
    try:
        parsed = yaml.safe_load(body.yaml_text)
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML must be a mapping")
        save_config(parsed)
        return {"ok": True}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")


# ---------------------------------------------------------------------------
# Token / cost analytics endpoint
# ---------------------------------------------------------------------------


@app.get("/api/analytics/usage")
async def get_usage_analytics(days: int = 30):
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        cutoff = time.time() - (days * 86400)
        cur = db._conn.execute("""
            SELECT date(started_at, 'unixepoch') as day,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions
            FROM sessions WHERE started_at > ?
            GROUP BY day ORDER BY day
        """, (cutoff,))
        daily = [dict(r) for r in cur.fetchall()]

        cur2 = db._conn.execute("""
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions
            FROM sessions WHERE started_at > ? AND model IS NOT NULL
            GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, (cutoff,))
        by_model = [dict(r) for r in cur2.fetchall()]

        cur3 = db._conn.execute("""
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions
            FROM sessions WHERE started_at > ?
        """, (cutoff,))
        totals = dict(cur3.fetchone())

        return {"daily": daily, "by_model": by_model, "totals": totals, "period_days": days}
    finally:
        db.close()


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
        # Prevent path traversal via url-encoded sequences (%2e%2e/)
        if (
            full_path
            and file_path.resolve().is_relative_to(WEB_DIST.resolve())
            and file_path.exists()
            and file_path.is_file()
        ):
            return FileResponse(file_path)
        return FileResponse(
            WEB_DIST / "index.html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )


mount_spa(app)


def start_server(host: str = "127.0.0.1", port: int = 9119, open_browser: bool = True):
    """Start the web UI server."""
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        import logging
        logging.warning(
            "Binding to %s — the web UI exposes config and API keys. "
            "Only bind to non-localhost if you trust all users on the network.", host,
        )

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
