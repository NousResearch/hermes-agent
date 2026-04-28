"""
Hermes Agent — Web UI server.

Provides a FastAPI backend serving the Vite/React frontend and REST API
endpoints for managing configuration, environment variables, and sessions.

Usage:
    python -m hermes_cli.main web          # Start on http://127.0.0.1:9119
    python -m hermes_cli.main web --port 8080
"""

import asyncio
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
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
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    raise SystemExit(
        "Web UI requires fastapi and uvicorn.\n"
        "Run 'hermes web' to auto-install, or: pip install hermes-agent[web]"
    )

WEB_DIST = Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run registry — persistent execution tracking for GET /api/chat/runs/:runId
# ---------------------------------------------------------------------------
from hermes_cli.run_registry import RunRegistry as _RunRegistry
from hermes_cli.repository import get_repository_state as _get_repository_state

_RUN_REGISTRY = _RunRegistry()

# Background task references (kept alive to avoid GC)
_BACKGROUND_TASKS: set[asyncio.Task] = set()  # type: ignore[type-arg]

app = FastAPI(title="Hermes Agent", version=__version__)

# ---------------------------------------------------------------------------
# Session token for protecting sensitive endpoints (reveal).
# Generated fresh on every server start — dies when the process exits.
# Injected into the SPA HTML so only the legitimate web UI can use it.
# ---------------------------------------------------------------------------
_SESSION_TOKEN = os.getenv("HERMES_SESSION_TOKEN") or secrets.token_urlsafe(32)

# Simple rate limiter for the reveal endpoint
_reveal_timestamps: List[float] = []
_REVEAL_MAX_PER_WINDOW = 5
_REVEAL_WINDOW_SECONDS = 30

# CORS: restrict to localhost origins only.  The web UI is intended to run
# locally; binding to 0.0.0.0 with allow_origins=["*"] would let any website
# read/modify config and secrets.

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints that do NOT require the session token.  Everything else under
# /api/ is gated by the auth middleware below.  Keep this list minimal —
# only truly non-sensitive, read-only endpoints belong here.
# ---------------------------------------------------------------------------
_PUBLIC_API_PATHS: frozenset = frozenset(
    {
        "/api/status",
        "/api/config/defaults",
        "/api/config/schema",
        "/api/model/info",
        "/api/providers",
        "/api/workspaces/current",
        "/api/workspaces/recent",
        "/api/workspaces/validate",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _RealtimeHub:
    """Minimal in-process websocket fanout for HermesWeb operational events."""

    def __init__(self) -> None:
        self._connections: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self, websocket: WebSocket, session_id: Optional[str] = None
    ) -> str:
        await websocket.accept()
        connection_id = uuid.uuid4().hex[:12]
        async with self._lock:
            self._connections[connection_id] = {
                "websocket": websocket,
                "session_id": session_id or None,
            }
        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        async with self._lock:
            self._connections.pop(connection_id, None)

    async def broadcast(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        session_id: Optional[str] = None,
    ) -> None:
        async with self._lock:
            targets = list(self._connections.items())

        stale: list[str] = []
        for connection_id, connection in targets:
            subscribed_session = connection.get("session_id")
            if subscribed_session and session_id and subscribed_session != session_id:
                continue
            if subscribed_session and session_id is None:
                continue

            websocket: WebSocket = connection["websocket"]
            try:
                await websocket.send_json(
                    {
                        "type": event_type,
                        "sent_at": _utc_now_iso(),
                        **payload,
                    }
                )
            except Exception:
                stale.append(connection_id)

        for connection_id in stale:
            await self.disconnect(connection_id)


_REALTIME_HUB = _RealtimeHub()


def _make_code_event(
    event_type: str,
    payload: dict,
    workspace_id: Optional[str] = None,
    code_session_id: Optional[str] = None,
    version: int = 1,
) -> dict:
    """Build a normalized Code Mode WS event envelope."""
    return {
        "type": event_type,
        "version": version,
        "timestamp": _utc_now_iso(),
        "workspace_id": workspace_id,
        "code_session_id": code_session_id,
        "payload": payload,
    }


async def _broadcast_code_event(
    event_type: str,
    payload: dict,
    workspace_id: Optional[str] = None,
    code_session_id: Optional[str] = None,
) -> None:
    """Broadcast a normalized Code Mode event to all WS connections."""
    envelope = _make_code_event(
        event_type, payload,
        workspace_id=workspace_id,
        code_session_id=code_session_id,
    )
    try:
        await _REALTIME_HUB.broadcast(
            event_type,
            {k: v for k, v in envelope.items() if k != "type"},
            session_id=code_session_id,
        )
    except Exception:
        pass


class _StepEmitter:
    """Thread-safe bridge: fires run-step WS events from a sync thread executor.

    Agent callbacks run inside ``loop.run_in_executor`` (a plain thread), so
    they cannot ``await`` coroutines.  ``asyncio.run_coroutine_threadsafe``
    schedules the coroutine on the event loop from the thread.
    """

    def __init__(
        self,
        run_id: str,
        session_id: Optional[str],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._run_id = run_id
        self._session_id = session_id
        self._loop = loop

    def _fire(self, coro) -> None:
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception:
            pass

    def step_created(
        self,
        step_id: str,
        type_: str,
        title: str,
        content: Optional[str] = None,
        status: str = "running",
    ) -> None:
        try:
            _RUN_REGISTRY.create_step(
                run_id=self._run_id,
                step_id=step_id,
                type_=type_,
                title=title,
                content=content,
                status=status,
            )
        except Exception:
            pass
        now_iso = _utc_now_iso()
        step: Dict[str, Any] = {
            "id": step_id,
            "run_id": self._run_id,
            "type": type_,
            "title": title,
            "content": content,
            "status": status,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        self._fire(
            _REALTIME_HUB.broadcast(
                "step_created",
                {"run_id": self._run_id, "step": step},
                session_id=self._session_id,
            )
        )

    def step_updated(
        self,
        step_id: str,
        status: str,
        content: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        try:
            _RUN_REGISTRY.update_step(
                step_id=step_id, status=status, content=content, title=title
            )
        except Exception:
            pass
        now_iso = _utc_now_iso()
        step: Dict[str, Any] = {
            "id": step_id,
            "run_id": self._run_id,
            "status": status,
            "updated_at": now_iso,
        }
        if content is not None:
            step["content"] = content
        if title is not None:
            step["title"] = title
        self._fire(
            _REALTIME_HUB.broadcast(
                "step_updated",
                {"run_id": self._run_id, "step": step},
                session_id=self._session_id,
            )
        )

    def step_completed(
        self,
        step_id: str,
        content: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Emit a dedicated step_completed WS event and persist as completed."""
        try:
            _RUN_REGISTRY.update_step(
                step_id=step_id, status="completed", content=content, title=title
            )
        except Exception:
            pass
        now_iso = _utc_now_iso()
        step: Dict[str, Any] = {
            "id": step_id,
            "run_id": self._run_id,
            "status": "completed",
            "updated_at": now_iso,
        }
        if content is not None:
            step["content"] = content
        if title is not None:
            step["title"] = title
        self._fire(
            _REALTIME_HUB.broadcast(
                "step_completed",
                {"run_id": self._run_id, "step": step},
                session_id=self._session_id,
            )
        )

    def message_chunk(self, content: str) -> None:
        if not content:
            return
        self._fire(
            _REALTIME_HUB.broadcast(
                "message_chunk",
                {"run_id": self._run_id, "content": content},
                session_id=self._session_id,
            )
        )


def _on_approval_queued(session_key: str, approval_data: dict) -> None:
    try:
        from tools.approval import _gateway_queues, _lock

        with _lock:
            entries = _gateway_queues.get(session_key, [])
            idx = len(entries) - 1
        approval_id = _generate_approval_id(session_key, max(idx, 0))
        approval_data_copy = dict(approval_data)
        approval_data_copy.setdefault("session_id", session_key)
        approval_data_copy["created_at"] = _utc_now_iso()
        serialized = _serialize_approval_entry(
            approval_data_copy, approval_id, session_key
        )

        try:
            _approval_db.upsert_from_queue(
                approval_id=approval_id,
                session_id=serialized["session_id"],
                agent_id=serialized["agent_id"],
                title=serialized["title"],
                command=serialized["command"],
                created_at=serialized["created_at"],
                details=serialized.get("details"),
                kind=serialized.get("kind", "command"),
            )
        except Exception:
            pass

        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    _REALTIME_HUB.broadcast(
                        "approval.requested",
                        {"session_id": session_key, "approval": serialized},
                        session_id=session_key,
                    )
                )
        except RuntimeError:
            pass
    except Exception:
        pass


try:
    from tools.approval import register_approval_listener

    register_approval_listener(_on_approval_queued)
except Exception:
    pass


def _require_token(request: Request) -> None:
    """Validate the ephemeral session token.  Raises 401 on mismatch.

    Uses ``hmac.compare_digest`` to prevent timing side-channels.
    """
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {_SESSION_TOKEN}"
    if not hmac.compare_digest(auth.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the session token on all /api/ routes except the public list."""
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if path.startswith("/api/") and path not in _PUBLIC_API_PATHS:
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {_SESSION_TOKEN}"
        if not hmac.compare_digest(auth.encode(), expected.encode()):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )
    return await call_next(request)


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
    "model_context_length": {
        "type": "number",
        "description": "Context window override (0 = auto-detect from model metadata)",
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
    "general",
    "agent",
    "terminal",
    "display",
    "delegation",
    "memory",
    "compression",
    "security",
    "browser",
    "voice",
    "tts",
    "stt",
    "logging",
    "discord",
    "auxiliary",
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
            entry["category"] = _CATEGORY_MERGE.get(
                entry["category"], entry["category"]
            )
            schema[full_key] = entry
    return schema


CONFIG_SCHEMA = _build_schema_from_config(DEFAULT_CONFIG)

# Inject virtual fields that don't live in DEFAULT_CONFIG but are surfaced
# by the normalize/denormalize cycle.  Insert model_context_length right after
# the "model" key so it renders adjacent in the frontend.
_mcl_entry = _SCHEMA_OVERRIDES["model_context_length"]
_ordered_schema: Dict[str, Dict[str, Any]] = {}
for _k, _v in CONFIG_SCHEMA.items():
    _ordered_schema[_k] = _v
    if _k == "model":
        _ordered_schema["model_context_length"] = _mcl_entry
CONFIG_SCHEMA = _ordered_schema


class ConfigUpdate(BaseModel):
    config: dict


class EnvVarUpdate(BaseModel):
    key: str
    value: str


class EnvVarDelete(BaseModel):
    key: str


class EnvVarReveal(BaseModel):
    key: str


class ChatRequest(BaseModel):
    content: str
    session_id: Optional[str] = None
    sessionId: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    workspace_path: Optional[str] = None
    run_id: Optional[str] = None  # client-generated for frontend tracking

    @property
    def resolved_session_id(self) -> Optional[str]:
        return self.session_id or self.sessionId


_GATEWAY_HEALTH_URL = os.getenv("GATEWAY_HEALTH_URL")
_GATEWAY_HEALTH_TIMEOUT = float(os.getenv("GATEWAY_HEALTH_TIMEOUT", "3"))

# Max seconds a single chat run is allowed to execute.
# Increase via HERMES_RUN_TIMEOUT_SECONDS for long tasks (deploy, build, SSH…).
_RUN_TIMEOUT_SECONDS = float(
    os.getenv("HERMES_RUN_TIMEOUT_SECONDS", "3600")
)  # 1 hour default


def _probe_gateway_health() -> tuple[bool, dict | None]:
    """Probe the gateway via its HTTP health endpoint (cross-container).

    Uses ``/health/detailed`` first (returns full state), falling back to
    the simpler ``/health`` endpoint.  Returns ``(is_alive, body_dict)``.

    Accepts any of these as ``GATEWAY_HEALTH_URL``:
    - ``http://gateway:8642``                (base URL — recommended)
    - ``http://gateway:8642/health``         (explicit health path)
    - ``http://gateway:8642/health/detailed`` (explicit detailed path)

    This is a **blocking** call — run via ``run_in_executor`` from async code.
    """
    if not _GATEWAY_HEALTH_URL:
        return False, None

    # Normalise to base URL so we always probe the right paths regardless of
    # whether the user included /health or /health/detailed in the env var.
    base = _GATEWAY_HEALTH_URL.rstrip("/")
    if base.endswith("/health/detailed"):
        base = base[: -len("/health/detailed")]
    elif base.endswith("/health"):
        base = base[: -len("/health")]

    for path in (f"{base}/health/detailed", f"{base}/health"):
        try:
            req = urllib.request.Request(path, method="GET")
            with urllib.request.urlopen(req, timeout=_GATEWAY_HEALTH_TIMEOUT) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read())
                    return True, body
        except Exception:
            continue
    return False, None


def _parse_reasoning_config_for_web(config: Dict[str, Any]) -> dict | None:
    from hermes_constants import parse_reasoning_effort

    agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    raw = str(agent_cfg.get("reasoning_effort") or "").strip()
    if not raw:
        return None
    return parse_reasoning_effort(raw)


def _parse_service_tier_for_web(config: Dict[str, Any]) -> str | None:
    agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    value = str(agent_cfg.get("service_tier") or "").strip().lower()
    if not value or value in {"normal", "default", "standard", "off", "none"}:
        return None
    if value in {"fast", "priority", "on"}:
        return "priority"
    return None


def _configured_model_name(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return str(
            model_cfg.get("default")
            or model_cfg.get("name")
            or model_cfg.get("model")
            or ""
        ).strip()
    if isinstance(model_cfg, str):
        return model_cfg.strip()
    return ""


def _classify_message_kind(message: Dict[str, Any]) -> str:
    """Classify the kind of a message for frontend rendering."""
    role = message.get("role", "")
    content = message.get("content") or ""
    tool_name = message.get("tool_name")

    if role == "tool" or tool_name:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if "diff" in parsed or "files_modified" in parsed:
                    return "diff"
                if "path" in parsed and "content" in parsed:
                    return "artifact"
                return "tool_result"
        except (json.JSONDecodeError, TypeError):
            pass
        return "tool_result"

    if role == "assistant":
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if "diff" in parsed or "files_modified" in parsed:
                    return "diff"
                if "output" in parsed:
                    return "tool_result"
                if "path" in parsed and "content" in parsed:
                    return "artifact"
        except (json.JSONDecodeError, TypeError):
            pass

    if role == "user":
        return "user"
    if role == "system":
        return "system"
    if role == "error" or message.get("finish_reason") == "error":
        return "error"

    return "assistant"


def _serialize_chat_message(message: Dict[str, Any]) -> Dict[str, Any]:
    kind = _classify_message_kind(message)
    serialized = {
        "id": message.get("id"),
        "session_id": message.get("session_id"),
        "role": message.get("role"),
        "content": message.get("content"),
        "timestamp": message.get("timestamp"),
        "tool_name": message.get("tool_name"),
        "tool_calls": message.get("tool_calls"),
        "tool_call_id": message.get("tool_call_id"),
        "finish_reason": message.get("finish_reason"),
        "reasoning": message.get("reasoning"),
        "kind": kind,
    }

    if kind == "tool_result" and message.get("content"):
        try:
            parsed = json.loads(message["content"])
            if isinstance(parsed, dict):
                serialized["metadata"] = {
                    k: v
                    for k, v in parsed.items()
                    if k
                    in (
                        "output",
                        "command",
                        "exit_code",
                        "duration_ms",
                        "files_modified",
                        "diff",
                    )
                }
        except (json.JSONDecodeError, TypeError):
            if len(message["content"]) > 200:
                serialized["metadata"] = {
                    "truncated": True,
                    "preview": message["content"][:200],
                }

    return serialized


def _serialize_chat_session(
    session: Optional[Dict[str, Any]], *, latest_timestamp: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    if not session:
        return None

    last_active = latest_timestamp
    if last_active is None:
        last_active = session.get("ended_at") or session.get("started_at")
    now = time.time()
    return {
        "id": session.get("id"),
        "title": session.get("title"),
        "source": session.get("source"),
        "model": session.get("model"),
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "end_reason": session.get("end_reason"),
        "message_count": session.get("message_count"),
        "tool_call_count": session.get("tool_call_count"),
        "input_tokens": session.get("input_tokens"),
        "output_tokens": session.get("output_tokens"),
        "last_active": last_active,
        "is_active": (
            session.get("ended_at") is None
            and bool(last_active)
            and (now - float(last_active)) < 300
        ),
    }


def _resolve_chat_session_id(db, requested_session_id: Optional[str]) -> Optional[str]:
    if not requested_session_id:
        return None
    sid = db.resolve_session_id(requested_session_id)
    if not sid:
        raise HTTPException(status_code=404, detail="Session not found")
    return sid


def _resolve_chat_runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    from hermes_cli.runtime_provider import (
        format_runtime_provider_error,
        resolve_runtime_provider,
    )

    try:
        runtime = resolve_runtime_provider()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=format_runtime_provider_error(exc)
        ) from exc

    api_key = runtime.get("api_key")
    base_url = runtime.get("base_url")
    if not isinstance(api_key, str) or not api_key:
        has_custom_base = (
            isinstance(base_url, str) and base_url and "openrouter.ai" not in base_url
        )
        if has_custom_base:
            api_key = "no-key-required"
        else:
            raise HTTPException(
                status_code=503, detail="Runtime provider returned no usable API key"
            )
    if not isinstance(base_url, str) or not base_url:
        raise HTTPException(
            status_code=503, detail="Runtime provider returned no base URL"
        )

    model_name = str(
        runtime.get("model") or _configured_model_name(config) or ""
    ).strip()
    if not model_name:
        try:
            from hermes_cli.models import get_default_model_for_provider

            model_name = str(
                get_default_model_for_provider(runtime.get("provider")) or ""
            ).strip()
        except Exception:
            model_name = ""
    if not model_name:
        raise HTTPException(
            status_code=503, detail="No model configured for chat runtime"
        )

    runtime["api_key"] = api_key
    runtime["model"] = model_name
    return runtime


def _run_chat_turn(
    *,
    content: str,
    session_id: Optional[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    workspace_path: Optional[str] = None,
    step_emitter: Optional["_StepEmitter"] = None,
) -> Dict[str, Any]:
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from hermes_state import SessionDB
    from run_agent import AIAgent

    user_content = (content or "").strip()
    if not user_content:
        raise HTTPException(status_code=400, detail="content must not be empty")

    config = load_config()
    session_db = SessionDB()

    original_cwd = None
    if workspace_path:
        original_cwd = os.getcwd()
        try:
            os.chdir(workspace_path)
        except Exception:
            pass

    try:
        resolved_session_id = _resolve_chat_session_id(session_db, session_id)
        if resolved_session_id:
            try:
                session_db.reopen_session(resolved_session_id)
            except Exception:
                pass

        before_messages = (
            session_db.get_messages(resolved_session_id) if resolved_session_id else []
        )
        history = (
            session_db.get_messages_as_conversation(resolved_session_id)
            if resolved_session_id
            else []
        )
        history = [msg for msg in (history or []) if msg.get("role") != "session_meta"]

        if provider:
            runtime = resolve_runtime_provider(requested=provider)
            runtime["model"] = model or runtime.get("model") or ""
        else:
            runtime = _resolve_chat_runtime(config)

        agent_cfg = (
            config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
        )
        max_iterations = int(
            agent_cfg.get("max_turns") or config.get("max_turns") or 90
        )

        # Build step-emitter callbacks (fire-and-forget to the event loop)
        _tool_start_cb = None
        _tool_complete_cb = None
        _stream_delta_cb = None
        _status_cb = None

        # Step IDs for lifecycle steps that span across callbacks
        _agent_step_id = f"step-agent-{uuid.uuid4().hex[:8]}" if step_emitter else None
        _model_step_id = f"step-model-{uuid.uuid4().hex[:8]}" if step_emitter else None
        _model_step_fired = [False]  # mutable flag for closure

        if step_emitter:

            def _tool_start_cb(tool_call_id: str, name: str, args: Any) -> None:
                step_id = f"step-tool-{(tool_call_id or uuid.uuid4().hex)[:12]}"
                try:
                    args_preview = (
                        json.dumps(args, ensure_ascii=False)[:400]
                        if isinstance(args, dict)
                        else str(args)[:400]
                    )
                except Exception:
                    args_preview = str(args)[:400]
                step_emitter.step_created(
                    step_id=step_id,
                    type_="tool_call",
                    title=f"Chamando {name}",
                    content=args_preview or None,
                    status="running",
                )

            def _tool_complete_cb(
                tool_call_id: str, name: str, args: Any, result: Any
            ) -> None:
                # Mark tool_call step completed
                tc_step_id = f"step-tool-{(tool_call_id or '')[:12]}"
                step_emitter.step_completed(step_id=tc_step_id)
                # Emit separate tool_result step with result content
                result_str = str(result)[:600] if result is not None else None
                if result_str:
                    result_step_id = (
                        f"step-result-{(tool_call_id or uuid.uuid4().hex)[:12]}"
                    )
                    step_emitter.step_created(
                        step_id=result_step_id,
                        type_="tool_result",
                        title=f"Resultado: {name}",
                        content=result_str,
                        status="completed",
                    )

            def _stream_delta_cb(delta: str) -> None:
                if delta:
                    if not _model_step_fired[0]:
                        _model_step_fired[0] = True
                        step_emitter.step_created(
                            step_id=_model_step_id,
                            type_="message",
                            title="Modelo respondendo",
                            content=None,
                            status="running",
                        )
                    step_emitter.message_chunk(delta)

            def _status_cb(kind: str, message: str) -> None:
                if kind == "lifecycle" and message:
                    step_id = f"step-status-{uuid.uuid4().hex[:8]}"
                    step_emitter.step_created(
                        step_id=step_id,
                        type_="log",
                        title=message,
                        content=None,
                        status="completed",
                    )

        # Step: agent preparation
        if step_emitter and _agent_step_id:
            step_emitter.step_created(
                step_id=_agent_step_id,
                type_="log",
                title="Preparando agente",
                content=None,
                status="running",
            )

        agent = AIAgent(
            model=runtime["model"],
            api_key=runtime["api_key"],
            base_url=runtime.get("base_url"),
            provider=runtime.get("provider"),
            api_mode=runtime.get("api_mode"),
            acp_command=runtime.get("command"),
            acp_args=list(runtime.get("args") or []),
            credential_pool=runtime.get("credential_pool"),
            max_iterations=max_iterations,
            enabled_toolsets=sorted(_get_platform_tools(config, "cli")),
            verbose_logging=False,
            quiet_mode=True,
            reasoning_config=_parse_reasoning_config_for_web(config),
            service_tier=_parse_service_tier_for_web(config),
            session_id=resolved_session_id,
            platform="web",
            session_db=session_db,
            persist_session=True,
            tool_start_callback=_tool_start_cb,
            tool_complete_callback=_tool_complete_cb,
            stream_delta_callback=_stream_delta_cb,
            status_callback=_status_cb,
        )

        if step_emitter and _agent_step_id:
            step_emitter.step_completed(step_id=_agent_step_id)

        result = agent.run_conversation(
            user_message=user_content,
            conversation_history=history,
            persist_user_message=user_content,
        )

        # Mark model step completed if it was started
        if step_emitter and _model_step_fired[0] and _model_step_id:
            step_emitter.step_completed(step_id=_model_step_id)

        final_session_id = agent.session_id
        after_messages = session_db.get_messages(final_session_id)
        previous_max_id = max(
            (int(msg.get("id") or 0) for msg in before_messages), default=0
        )
        new_messages = [
            _serialize_chat_message(msg)
            for msg in after_messages
            if int(msg.get("id") or 0) > previous_max_id
        ]
        user_message = next(
            (msg for msg in new_messages if msg.get("role") == "user"), None
        )
        assistant_message = next(
            (msg for msg in reversed(new_messages) if msg.get("role") == "assistant"),
            None,
        )

        session = session_db.get_session(final_session_id)
        latest_timestamp = None
        if after_messages:
            latest_timestamp = after_messages[-1].get("timestamp")

        return {
            "session_id": final_session_id,
            "session": _serialize_chat_session(
                session, latest_timestamp=latest_timestamp
            ),
            "messages": new_messages,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "completed": bool(result.get("completed")),
            "partial": bool(result.get("partial")),
            "usage": {
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
            },
            "provider": runtime.get("provider"),
            "model": runtime.get("model"),
        }
    finally:
        session_db.close()
        if original_cwd is not None:
            try:
                os.chdir(original_cwd)
            except Exception:
                pass


async def _execute_chat_run(run_id: str, body: "ChatRequest") -> None:
    """Background task: run the agent, update the registry, emit WS events."""
    session_id = body.resolved_session_id
    loop = asyncio.get_event_loop()

    # --- Create linked task ---
    chat_task_id: Optional[str] = None
    try:
        chat_task_id = _generate_task_id()
        title = body.content[:80].strip() or "Chat run"
        task = _get_task_db().create_task(
            task_id=chat_task_id,
            title=title,
            description=body.content if len(body.content) > 80 else None,
            priority="medium",
            session_id=session_id,
            run_id=run_id,
        )
        task = _get_task_db().update_task(chat_task_id, {"status": "in_progress"})
        await _REALTIME_HUB.broadcast("task.created", {"task": task})
    except Exception as exc:
        _log.warning("[chat-run] task creation failed run_id=%s: %s", run_id, exc)
        chat_task_id = None

    # --- Mark as running ---
    _RUN_REGISTRY.start_run(run_id)
    _log.info("[chat-run] started  run_id=%s session_id=%s", run_id, session_id)
    await _REALTIME_HUB.broadcast(
        "chat_started",
        {"type": "chat_started", "run_id": run_id, "session_id": session_id},
    )

    # --- Create step emitter + emit start step ---
    step_emitter = _StepEmitter(run_id=run_id, session_id=session_id, loop=loop)
    step_emitter.step_created(
        step_id=f"step-start-{run_id[:12]}",
        type_="log",
        title="Iniciando execução",
        content=body.content[:200] if body.content else None,
        status="completed",
    )

    try:
        # Agent is blocking — run in thread executor so event loop stays responsive
        future = loop.run_in_executor(
            None,
            lambda: _run_chat_turn(
                content=body.content,
                session_id=session_id,
                provider=body.provider,
                model=body.model,
                workspace_path=body.workspace_path,
                step_emitter=step_emitter,
            ),
        )

        # Configurable run timeout (default 1h). Set HERMES_RUN_TIMEOUT_SECONDS to override.
        turn = await asyncio.wait_for(future, timeout=_RUN_TIMEOUT_SECONDS)

        messages: list = turn.get("messages") or []
        actual_session_id: Optional[str] = turn.get("session_id") or session_id
        assistant_msg = turn.get("assistant_message")

        # Persist completed run
        _RUN_REGISTRY.complete_run(
            run_id,
            messages=messages,
            session_id=actual_session_id,
            provider=turn.get("provider"),
            model=turn.get("model"),
        )
        _log.info(
            "[chat-run] completed run_id=%s session_id=%s msgs=%d",
            run_id,
            actual_session_id,
            len(messages),
        )

        # Emit completion step
        step_emitter.step_created(
            step_id=f"step-done-{run_id[:12]}",
            type_="message",
            title="Execução concluída",
            content=assistant_msg.get("content", "")[:300] if assistant_msg else None,
            status="completed",
        )

        # Update linked task to done
        if chat_task_id:
            try:
                updated_task = _get_task_db().update_task(
                    chat_task_id, {"status": "done"}
                )
                if updated_task:
                    await _REALTIME_HUB.broadcast(
                        "task.updated", {"task": updated_task}
                    )
            except Exception as exc:
                _log.warning(
                    "[chat-run] task update (done) failed run_id=%s: %s", run_id, exc
                )

        # Emit WS events (same as the old synchronous post_chat)
        for msg in messages:
            await _REALTIME_HUB.broadcast(
                "message.created",
                {"session_id": actual_session_id, "message": msg},
                session_id=actual_session_id,
            )

        await _REALTIME_HUB.broadcast(
            "chat_completed",
            {
                "type": "chat_completed",
                "run_id": run_id,
                "session_id": actual_session_id,
                "message": assistant_msg,
                "messages": messages,
            },
            session_id=actual_session_id,
        )

        if turn.get("session"):
            session_data = turn["session"]
            await _REALTIME_HUB.broadcast(
                "session.updated",
                {"session": session_data},
                session_id=actual_session_id,
            )
            now = time.time()
            agent_event = _session_to_agent(session_data, now)
            await _REALTIME_HUB.broadcast(
                "agent.updated",
                {"session_id": actual_session_id, "agent": agent_event},
                session_id=actual_session_id,
            )

    except asyncio.TimeoutError:
        _RUN_REGISTRY.timeout_run(run_id)
        _log.warning(
            "[chat-run] timeout  run_id=%s after %.0fs", run_id, _RUN_TIMEOUT_SECONDS
        )
        step_emitter.step_created(
            step_id=f"step-timeout-{run_id[:12]}",
            type_="log",
            title=f"Execução excedeu o tempo limite ({int(_RUN_TIMEOUT_SECONDS)}s)",
            status="failed",
        )
        if chat_task_id:
            try:
                updated_task = _get_task_db().update_task(
                    chat_task_id,
                    {
                        "status": "timeout",
                        "error_message": f"Execução excedeu o tempo limite ({int(_RUN_TIMEOUT_SECONDS)}s)",
                    },
                )
                if updated_task:
                    await _REALTIME_HUB.broadcast(
                        "task.updated", {"task": updated_task}
                    )
            except Exception as exc:
                _log.warning(
                    "[chat-run] task update (timeout) failed run_id=%s: %s", run_id, exc
                )
        await _REALTIME_HUB.broadcast(
            "chat_timeout",
            {"type": "chat_timeout", "run_id": run_id, "session_id": session_id},
        )

    except Exception as exc:
        error_msg = str(exc)
        _RUN_REGISTRY.fail_run(run_id, error_msg)
        _log.error("[chat-run] failed   run_id=%s error=%r", run_id, error_msg)
        step_emitter.step_created(
            step_id=f"step-failed-{run_id[:12]}",
            type_="log",
            title="Execução falhou",
            content=error_msg[:500],
            status="failed",
        )
        if chat_task_id:
            try:
                updated_task = _get_task_db().update_task(
                    chat_task_id,
                    {"status": "failed", "error_message": error_msg[:1000]},
                )
                if updated_task:
                    await _REALTIME_HUB.broadcast(
                        "task.updated", {"task": updated_task}
                    )
            except Exception as exc2:
                _log.warning(
                    "[chat-run] task update (failed) failed run_id=%s: %s", run_id, exc2
                )
        await _REALTIME_HUB.broadcast(
            "chat_failed",
            {
                "type": "chat_failed",
                "run_id": run_id,
                "session_id": session_id,
                "error": error_msg,
            },
        )


async def _orphan_sweep_loop() -> None:
    """Periodically mark stale running runs as timeout and purge old records.

    A run is considered orphaned when it has been in ``running`` state for
    longer than ``_RUN_TIMEOUT_SECONDS`` + a 60-second grace period.  This
    matches the ``asyncio.wait_for`` limit so only *truly* abandoned runs
    (e.g. server restarts mid-run) are swept, not legitimately long tasks.
    """
    orphan_threshold = _RUN_TIMEOUT_SECONDS + 60  # grace period
    while True:
        try:
            await asyncio.sleep(60)
            stale = _RUN_REGISTRY.get_stale_running_runs(
                max_age_seconds=orphan_threshold
            )
            for run_id in stale:
                _RUN_REGISTRY.timeout_run(run_id)
                _log.warning("[chat-run] orphan-sweep: timed out run_id=%s", run_id)
                await _REALTIME_HUB.broadcast(
                    "chat_timeout",
                    {"type": "chat_timeout", "run_id": run_id, "session_id": None},
                )
            deleted = _RUN_REGISTRY.delete_old_runs(max_age_seconds=7 * 86400)
            if deleted:
                _log.info("[chat-run] orphan-sweep: purged %d old runs", deleted)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            _log.error("[chat-run] orphan-sweep error: %s", exc)


@app.on_event("startup")
async def _start_orphan_sweep() -> None:
    task = asyncio.create_task(_orphan_sweep_loop())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    _log.info("[chat-run] orphan sweep started (interval=60s, timeout=600s)")


@app.on_event("startup")
async def _configure_artifact_ws_callback() -> None:
    """Register artifact WS broadcast callback so file tools emit artifact.created events."""
    loop = asyncio.get_event_loop()

    def _artifact_cb(artifact: dict, session_id: str) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                _REALTIME_HUB.broadcast(
                    "artifact.created",
                    {"payload": {"session_id": session_id, "artifact": artifact}},
                    session_id=session_id,
                ),
                loop,
            )
        except Exception:
            pass

    try:
        from tools.file_tools import set_artifact_created_callback

        set_artifact_created_callback(_artifact_cb)
    except Exception:
        pass


async def _require_ws_token(websocket: WebSocket) -> bool:
    auth = websocket.headers.get("authorization", "")
    token = websocket.query_params.get("token", "")
    expected = f"Bearer {_SESSION_TOKEN}"
    if hmac.compare_digest(auth.encode(), expected.encode()):
        return True
    if token and hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode()):
        return True
    await websocket.close(code=4401, reason="Unauthorized")
    return False


@app.get("/api/status")
async def get_status():
    current_ver, latest_ver = check_config_version()

    # --- Gateway liveness detection ---
    # Try local PID check first (same-host).  If that fails and a remote
    # GATEWAY_HEALTH_URL is configured, probe the gateway over HTTP so the
    # dashboard works when the gateway runs in a separate container.
    gateway_pid = get_running_pid()
    gateway_running = gateway_pid is not None
    remote_health_body: dict | None = None

    if not gateway_running and _GATEWAY_HEALTH_URL:
        loop = asyncio.get_event_loop()
        alive, remote_health_body = await loop.run_in_executor(
            None, _probe_gateway_health
        )
        if alive:
            gateway_running = True
            # PID from the remote container (display only — not locally valid)
            if remote_health_body:
                gateway_pid = remote_health_body.get("pid")

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

    # Prefer the detailed health endpoint response (has full state) when the
    # local runtime status file is absent or stale (cross-container).
    runtime = read_runtime_status()
    if (
        runtime is None
        and remote_health_body
        and remote_health_body.get("gateway_state")
    ):
        runtime = remote_health_body

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
            gateway_state = (
                gateway_state
                if gateway_state in ("stopped", "startup_failed")
                else "stopped"
            )
            gateway_platforms = {}
        elif gateway_running and remote_health_body is not None:
            # The health probe confirmed the gateway is alive, but the local
            # runtime status file may be stale (cross-container).  Override
            # stopped/None state so the dashboard shows the correct badge.
            if gateway_state in (None, "stopped"):
                gateway_state = "running"

    # If there was no runtime info at all but the health probe confirmed alive,
    # ensure we still report the gateway as running (no shared volume scenario).
    if gateway_running and gateway_state is None and remote_health_body is not None:
        gateway_state = "running"

    active_sessions = 0
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            sessions = db.list_sessions_rich(limit=50)
            now = time.time()
            active_sessions = sum(
                1
                for s in sessions
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
async def get_sessions(limit: int = 20, offset: int = 0):
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            sessions = db.list_sessions_rich(limit=limit, offset=offset)
            total = db.session_count()
            now = time.time()
            for s in sessions:
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
            return {
                "sessions": sessions,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
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

    Also surfaces ``model_context_length`` as a top-level field so the web UI can
    display and edit it.  A value of 0 means "auto-detect".
    """
    config = dict(config)  # shallow copy
    model_val = config.get("model")
    if isinstance(model_val, dict):
        # Extract context_length before flattening the dict
        ctx_len = model_val.get("context_length", 0)
        config["model"] = model_val.get("default", model_val.get("name", ""))
        config["model_context_length"] = ctx_len if isinstance(ctx_len, int) else 0
    else:
        config["model_context_length"] = 0
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


_EMPTY_MODEL_INFO: dict = {
    "model": "",
    "provider": "",
    "auto_context_length": 0,
    "config_context_length": 0,
    "effective_context_length": 0,
    "capabilities": {},
}


@app.get("/api/model/info")
def get_model_info():
    """Return resolved model metadata for the currently configured model.

    Calls the same context-length resolution chain the agent uses, so the
    frontend can display "Auto-detected: 200K" alongside the override field.
    Also returns model capabilities (vision, reasoning, tools) when available.
    """
    try:
        cfg = load_config()
        model_cfg = cfg.get("model", "")

        # Extract model name and provider from the config
        if isinstance(model_cfg, dict):
            model_name = model_cfg.get("default", model_cfg.get("name", ""))
            provider = model_cfg.get("provider", "")
            base_url = model_cfg.get("base_url", "")
            config_ctx = model_cfg.get("context_length")
        else:
            model_name = str(model_cfg) if model_cfg else ""
            provider = ""
            base_url = ""
            config_ctx = None

        if not model_name:
            return dict(_EMPTY_MODEL_INFO, provider=provider)

        # Resolve auto-detected context length (pass config_ctx=None to get
        # purely auto-detected value, then separately report the override)
        try:
            from agent.model_metadata import get_model_context_length

            auto_ctx = get_model_context_length(
                model=model_name,
                base_url=base_url,
                provider=provider,
                config_context_length=None,  # ignore override — we want auto value
            )
        except Exception:
            auto_ctx = 0

        config_ctx_int = 0
        if isinstance(config_ctx, int) and config_ctx > 0:
            config_ctx_int = config_ctx

        # Effective is what the agent actually uses
        effective_ctx = config_ctx_int if config_ctx_int > 0 else auto_ctx

        # Try to get model capabilities from models.dev
        caps = {}
        try:
            from agent.models_dev import get_model_capabilities

            mc = get_model_capabilities(provider=provider, model=model_name)
            if mc is not None:
                caps = {
                    "supports_tools": mc.supports_tools,
                    "supports_vision": mc.supports_vision,
                    "supports_reasoning": mc.supports_reasoning,
                    "context_window": mc.context_window,
                    "max_output_tokens": mc.max_output_tokens,
                    "model_family": mc.model_family,
                }
        except Exception:
            pass

        return {
            "model": model_name,
            "provider": provider,
            "auto_context_length": auto_ctx,
            "config_context_length": config_ctx_int,
            "effective_context_length": effective_ctx,
            "capabilities": caps,
        }
    except Exception:
        _log.exception("GET /api/model/info failed")
        return dict(_EMPTY_MODEL_INFO)


def _denormalize_config_from_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse _normalize_config_for_web before saving.

    Reconstructs ``model`` as a dict by reading the current on-disk config
    to recover model subkeys (provider, base_url, api_mode, etc.) that were
    stripped from the GET response.  The frontend only sees model as a flat
    string; the rest is preserved transparently.

    Also handles ``model_context_length`` — writes it back into the model dict
    as ``context_length``.  A value of 0 or absent means "auto-detect" (omitted
    from the dict so get_model_context_length() uses its normal resolution).
    """
    config = dict(config)
    # Remove any _model_meta that might have leaked in (shouldn't happen
    # with the stripped GET response, but be defensive)
    config.pop("_model_meta", None)

    # Extract and remove model_context_length before processing model
    ctx_override = config.pop("model_context_length", 0)
    if not isinstance(ctx_override, int):
        try:
            ctx_override = int(ctx_override)
        except (TypeError, ValueError):
            ctx_override = 0

    model_val = config.get("model")
    if isinstance(model_val, str) and model_val:
        # Read the current disk config to recover model subkeys
        try:
            disk_config = load_config()
            disk_model = disk_config.get("model")
            if isinstance(disk_model, dict):
                # Preserve all subkeys, update default with the new value
                disk_model["default"] = model_val
                # Write context_length into the model dict (0 = remove/auto)
                if ctx_override > 0:
                    disk_model["context_length"] = ctx_override
                else:
                    disk_model.pop("context_length", None)
                config["model"] = disk_model
            else:
                # Model was previously a bare string — upgrade to dict if
                # user is setting a context_length override
                if ctx_override > 0:
                    config["model"] = {
                        "default": model_val,
                        "context_length": ctx_override,
                    }
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
    _require_token(request)

    # --- Rate limit ---
    now = time.time()
    cutoff = now - _REVEAL_WINDOW_SECONDS
    _reveal_timestamps[:] = [t for t in _reveal_timestamps if t > cutoff]
    if len(_reveal_timestamps) >= _REVEAL_MAX_PER_WINDOW:
        raise HTTPException(
            status_code=429, detail="Too many reveal requests. Try again shortly."
        )
    _reveal_timestamps.append(now)

    # --- Reveal ---
    env_on_disk = load_env()
    value = env_on_disk.get(body.key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")

    _log.info("env/reveal: %s", body.key)
    return {"key": body.key, "value": value}


# ---------------------------------------------------------------------------
# OAuth provider endpoints — status + disconnect (Phase 1)
# ---------------------------------------------------------------------------
#
# Phase 1 surfaces *which OAuth providers exist* and whether each is
# connected, plus a disconnect button. The actual login flow (PKCE for
# Anthropic, device-code for Nous/Codex) still runs in the CLI for now;
# Phase 2 will add in-browser flows. For unconnected providers we return
# the canonical ``hermes auth add <provider>`` command so the dashboard
# can surface a one-click copy.


def _truncate_token(value: Optional[str], visible: int = 6) -> str:
    """Return ``...XXXXXX`` (last N chars) for safe display in the UI.

    We never expose more than the trailing ``visible`` characters of an
    OAuth access token. JWT prefixes (the part before the first dot) are
    stripped first when present so the visible suffix is always part of
    the signing region rather than a meaningless header chunk.
    """
    if not value:
        return ""
    s = str(value)
    if "." in s and s.count(".") >= 2:
        # Looks like a JWT — show the trailing piece of the signature only.
        s = s.rsplit(".", 1)[-1]
    if len(s) <= visible:
        return s
    return f"…{s[-visible:]}"


def _anthropic_oauth_status() -> Dict[str, Any]:
    """Combined status across the three Anthropic credential sources we read.

    Hermes resolves Anthropic creds in this order at runtime:
    1. ``~/.hermes/.anthropic_oauth.json`` — Hermes-managed PKCE flow
    2. ``~/.claude/.credentials.json`` — Claude Code CLI credentials (auto)
    3. ``ANTHROPIC_TOKEN`` / ``ANTHROPIC_API_KEY`` env vars
    The dashboard reports the highest-priority source that's actually present.
    """
    try:
        from agent.anthropic_adapter import (
            read_hermes_oauth_credentials,
            read_claude_code_credentials,
            _HERMES_OAUTH_FILE,
        )
    except ImportError:
        read_claude_code_credentials = None  # type: ignore
        read_hermes_oauth_credentials = None  # type: ignore
        _HERMES_OAUTH_FILE = None  # type: ignore

    hermes_creds = None
    if read_hermes_oauth_credentials:
        try:
            hermes_creds = read_hermes_oauth_credentials()
        except Exception:
            hermes_creds = None
    if hermes_creds and hermes_creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "hermes_pkce",
            "source_label": f"Hermes PKCE ({_HERMES_OAUTH_FILE})",
            "token_preview": _truncate_token(hermes_creds.get("accessToken")),
            "expires_at": hermes_creds.get("expiresAt"),
            "has_refresh_token": bool(hermes_creds.get("refreshToken")),
        }

    cc_creds = None
    if read_claude_code_credentials:
        try:
            cc_creds = read_claude_code_credentials()
        except Exception:
            cc_creds = None
    if cc_creds and cc_creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "claude_code",
            "source_label": "Claude Code (~/.claude/.credentials.json)",
            "token_preview": _truncate_token(cc_creds.get("accessToken")),
            "expires_at": cc_creds.get("expiresAt"),
            "has_refresh_token": bool(cc_creds.get("refreshToken")),
        }

    env_token = os.getenv("ANTHROPIC_TOKEN") or os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
    if env_token:
        return {
            "logged_in": True,
            "source": "env_var",
            "source_label": "ANTHROPIC_TOKEN environment variable",
            "token_preview": _truncate_token(env_token),
            "expires_at": None,
            "has_refresh_token": False,
        }
    return {"logged_in": False, "source": None}


def _claude_code_only_status() -> Dict[str, Any]:
    """Surface Claude Code CLI credentials as their own provider entry.

    Independent of the Anthropic entry above so users can see whether their
    Claude Code subscription tokens are actively flowing into Hermes even
    when they also have a separate Hermes-managed PKCE login.
    """
    try:
        from agent.anthropic_adapter import read_claude_code_credentials

        creds = read_claude_code_credentials()
    except Exception:
        creds = None
    if creds and creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "claude_code_cli",
            "source_label": "~/.claude/.credentials.json",
            "token_preview": _truncate_token(creds.get("accessToken")),
            "expires_at": creds.get("expiresAt"),
            "has_refresh_token": bool(creds.get("refreshToken")),
        }
    return {"logged_in": False, "source": None}


# Provider catalog. The order matters — it's how we render the UI list.
# ``cli_command`` is what the dashboard surfaces as the copy-to-clipboard
# fallback while Phase 2 (in-browser flows) isn't built yet.
# ``flow`` describes the OAuth shape so the future modal can pick the
# right UI: ``pkce`` = open URL + paste callback code, ``device_code`` =
# show code + verification URL + poll, ``external`` = read-only (delegated
# to a third-party CLI like Claude Code or Qwen).
_OAUTH_PROVIDER_CATALOG: tuple[Dict[str, Any], ...] = (
    {
        "id": "anthropic",
        "name": "Anthropic (Claude API)",
        "flow": "pkce",
        "cli_command": "hermes auth add anthropic",
        "docs_url": "https://docs.claude.com/en/api/getting-started",
        "status_fn": _anthropic_oauth_status,
    },
    {
        "id": "claude-code",
        "name": "Claude Code (subscription)",
        "flow": "external",
        "cli_command": "claude setup-token",
        "docs_url": "https://docs.claude.com/en/docs/claude-code",
        "status_fn": _claude_code_only_status,
    },
    {
        "id": "nous",
        "name": "Nous Portal",
        "flow": "device_code",
        "cli_command": "hermes auth add nous",
        "docs_url": "https://portal.nousresearch.com",
        "status_fn": None,  # dispatched via auth.get_nous_auth_status
    },
    {
        "id": "openai-codex",
        "name": "OpenAI Codex (ChatGPT)",
        "flow": "device_code",
        "cli_command": "hermes auth add openai-codex",
        "docs_url": "https://platform.openai.com/docs",
        "status_fn": None,  # dispatched via auth.get_codex_auth_status
    },
    {
        "id": "qwen-oauth",
        "name": "Qwen (via Qwen CLI)",
        "flow": "external",
        "cli_command": "hermes auth add qwen-oauth",
        "docs_url": "https://github.com/QwenLM/qwen-code",
        "status_fn": None,  # dispatched via auth.get_qwen_auth_status
    },
)


def _resolve_provider_status(provider_id: str, status_fn) -> Dict[str, Any]:
    """Dispatch to the right status helper for an OAuth provider entry."""
    if status_fn is not None:
        try:
            return status_fn()
        except Exception as e:
            return {"logged_in": False, "error": str(e)}
    try:
        from hermes_cli import auth as hauth

        if provider_id == "nous":
            raw = hauth.get_nous_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": "nous_portal",
                "source_label": raw.get("portal_base_url") or "Nous Portal",
                "token_preview": _truncate_token(raw.get("access_token")),
                "expires_at": raw.get("access_expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
        if provider_id == "openai-codex":
            raw = hauth.get_codex_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": raw.get("source") or "openai_codex",
                "source_label": raw.get("auth_mode") or "OpenAI Codex",
                "token_preview": _truncate_token(raw.get("api_key")),
                "expires_at": None,
                "has_refresh_token": False,
                "last_refresh": raw.get("last_refresh"),
            }
        if provider_id == "qwen-oauth":
            raw = hauth.get_qwen_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": "qwen_cli",
                "source_label": raw.get("auth_store_path") or "Qwen CLI",
                "token_preview": _truncate_token(raw.get("access_token")),
                "expires_at": raw.get("expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
    except Exception as e:
        return {"logged_in": False, "error": str(e)}
    return {"logged_in": False}


@app.get("/api/providers/oauth")
async def list_oauth_providers():
    """Enumerate every OAuth-capable LLM provider with current status.

    Response shape (per provider):
        id              stable identifier (used in DELETE path)
        name            human label
        flow            "pkce" | "device_code" | "external"
        cli_command     fallback CLI command for users to run manually
        docs_url        external docs/portal link for the "Learn more" link
        status:
          logged_in        bool — currently has usable creds
          source           short slug ("hermes_pkce", "claude_code", ...)
          source_label     human-readable origin (file path, env var name)
          token_preview    last N chars of the token, never the full token
          expires_at       ISO timestamp string or null
          has_refresh_token bool
    """
    providers = []
    for p in _OAUTH_PROVIDER_CATALOG:
        status = _resolve_provider_status(p["id"], p.get("status_fn"))
        providers.append(
            {
                "id": p["id"],
                "name": p["name"],
                "flow": p["flow"],
                "cli_command": p["cli_command"],
                "docs_url": p["docs_url"],
                "status": status,
            }
        )
    return {"providers": providers}


@app.delete("/api/providers/oauth/{provider_id}")
async def disconnect_oauth_provider(provider_id: str, request: Request):
    """Disconnect an OAuth provider. Token-protected (matches /env/reveal)."""
    _require_token(request)

    valid_ids = {p["id"] for p in _OAUTH_PROVIDER_CATALOG}
    if provider_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {provider_id}. "
            f"Available: {', '.join(sorted(valid_ids))}",
        )

    # Anthropic and claude-code clear the same Hermes-managed PKCE file
    # AND forget the Claude Code import. We don't touch ~/.claude/* directly
    # — that's owned by the Claude Code CLI; users can re-auth there if they
    # want to undo a disconnect.
    if provider_id in ("anthropic", "claude-code"):
        try:
            from agent.anthropic_adapter import _HERMES_OAUTH_FILE

            if _HERMES_OAUTH_FILE.exists():
                _HERMES_OAUTH_FILE.unlink()
        except Exception:
            pass
        # Also clear the credential pool entry if present.
        try:
            from hermes_cli.auth import clear_provider_auth

            clear_provider_auth("anthropic")
        except Exception:
            pass
        _log.info("oauth/disconnect: %s", provider_id)
        return {"ok": True, "provider": provider_id}

    try:
        from hermes_cli.auth import clear_provider_auth

        cleared = clear_provider_auth(provider_id)
        _log.info("oauth/disconnect: %s (cleared=%s)", provider_id, cleared)
        return {"ok": bool(cleared), "provider": provider_id}
    except Exception as e:
        _log.exception("disconnect %s failed", provider_id)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# OAuth Phase 2 — in-browser PKCE & device-code flows
# ---------------------------------------------------------------------------
#
# Two flow shapes are supported:
#
#   PKCE (Anthropic):
#     1. POST /api/providers/oauth/anthropic/start
#          → server generates code_verifier + challenge, builds claude.ai
#            authorize URL, stashes verifier in _oauth_sessions[session_id]
#          → returns { session_id, flow: "pkce", auth_url }
#     2. UI opens auth_url in a new tab. User authorizes, copies code.
#     3. POST /api/providers/oauth/anthropic/submit { session_id, code }
#          → server exchanges (code + verifier) → tokens at console.anthropic.com
#          → persists to ~/.hermes/.anthropic_oauth.json AND credential pool
#          → returns { ok: true, status: "approved" }
#
#   Device code (Nous, OpenAI Codex):
#     1. POST /api/providers/oauth/{nous|openai-codex}/start
#          → server hits provider's device-auth endpoint
#          → gets { user_code, verification_url, device_code, interval, expires_in }
#          → spawns background poller thread that polls the token endpoint
#            every `interval` seconds until approved/expired
#          → stores poll status in _oauth_sessions[session_id]
#          → returns { session_id, flow: "device_code", user_code,
#                      verification_url, expires_in, poll_interval }
#     2. UI opens verification_url in a new tab and shows user_code.
#     3. UI polls GET /api/providers/oauth/{provider}/poll/{session_id}
#          every 2s until status != "pending".
#     4. On "approved" the background thread has already saved creds; UI
#        refreshes the providers list.
#
# Sessions are kept in-memory only (single-process FastAPI) and time out
# after 15 minutes. A periodic cleanup runs on each /start call to GC
# expired sessions so the dict doesn't grow without bound.

_OAUTH_SESSION_TTL_SECONDS = 15 * 60
_oauth_sessions: Dict[str, Dict[str, Any]] = {}
_oauth_sessions_lock = threading.Lock()

# Import OAuth constants from canonical source instead of duplicating.
# Guarded so hermes web still starts if anthropic_adapter is unavailable;
# Phase 2 endpoints will return 501 in that case.
try:
    from agent.anthropic_adapter import (
        _OAUTH_CLIENT_ID as _ANTHROPIC_OAUTH_CLIENT_ID,
        _OAUTH_TOKEN_URL as _ANTHROPIC_OAUTH_TOKEN_URL,
        _OAUTH_REDIRECT_URI as _ANTHROPIC_OAUTH_REDIRECT_URI,
        _OAUTH_SCOPES as _ANTHROPIC_OAUTH_SCOPES,
        _generate_pkce as _generate_pkce_pair,
    )

    _ANTHROPIC_OAUTH_AVAILABLE = True
except ImportError:
    _ANTHROPIC_OAUTH_AVAILABLE = False
_ANTHROPIC_OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"


def _gc_oauth_sessions() -> None:
    """Drop expired sessions. Called opportunistically on /start."""
    cutoff = time.time() - _OAUTH_SESSION_TTL_SECONDS
    with _oauth_sessions_lock:
        stale = [
            sid for sid, sess in _oauth_sessions.items() if sess["created_at"] < cutoff
        ]
        for sid in stale:
            _oauth_sessions.pop(sid, None)


def _new_oauth_session(provider_id: str, flow: str) -> tuple[str, Dict[str, Any]]:
    """Create + register a new OAuth session, return (session_id, session_dict)."""
    sid = secrets.token_urlsafe(16)
    sess = {
        "session_id": sid,
        "provider": provider_id,
        "flow": flow,
        "created_at": time.time(),
        "status": "pending",  # pending | approved | denied | expired | error
        "error_message": None,
    }
    with _oauth_sessions_lock:
        _oauth_sessions[sid] = sess
    return sid, sess


def _save_anthropic_oauth_creds(
    access_token: str, refresh_token: str, expires_at_ms: int
) -> None:
    """Persist Anthropic PKCE creds to both Hermes file AND credential pool.

    Mirrors what auth_commands.add_command does so the dashboard flow leaves
    the system in the same state as ``hermes auth add anthropic``.
    """
    from agent.anthropic_adapter import _HERMES_OAUTH_FILE

    payload = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at_ms,
    }
    _HERMES_OAUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HERMES_OAUTH_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Best-effort credential-pool insert. Failure here doesn't invalidate
    # the file write — pool registration only matters for the rotation
    # strategy, not for runtime credential resolution.
    try:
        from agent.credential_pool import (
            PooledCredential,
            load_pool,
            AUTH_TYPE_OAUTH,
            SOURCE_MANUAL,
        )
        import uuid

        pool = load_pool("anthropic")
        # Avoid duplicate entries: delete any prior dashboard-issued OAuth entry
        existing = [
            e
            for e in pool.entries()
            if getattr(e, "source", "").startswith(f"{SOURCE_MANUAL}:dashboard_pkce")
        ]
        for e in existing:
            try:
                pool.remove_entry(getattr(e, "id", ""))
            except Exception:
                pass
        entry = PooledCredential(
            provider="anthropic",
            id=uuid.uuid4().hex[:6],
            label="dashboard PKCE",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:dashboard_pkce",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at_ms=expires_at_ms,
        )
        pool.add_entry(entry)
    except Exception as e:
        _log.warning("anthropic pool add (dashboard) failed: %s", e)


def _start_anthropic_pkce() -> Dict[str, Any]:
    """Begin PKCE flow. Returns the auth URL the UI should open."""
    if not _ANTHROPIC_OAUTH_AVAILABLE:
        raise HTTPException(
            status_code=501, detail="Anthropic OAuth not available (missing adapter)"
        )
    verifier, challenge = _generate_pkce_pair()
    sid, sess = _new_oauth_session("anthropic", "pkce")
    sess["verifier"] = verifier
    sess["state"] = verifier  # Anthropic round-trips verifier as state
    params = {
        "code": "true",
        "client_id": _ANTHROPIC_OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _ANTHROPIC_OAUTH_REDIRECT_URI,
        "scope": _ANTHROPIC_OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    auth_url = f"{_ANTHROPIC_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    return {
        "session_id": sid,
        "flow": "pkce",
        "auth_url": auth_url,
        "expires_in": _OAUTH_SESSION_TTL_SECONDS,
    }


def _submit_anthropic_pkce(session_id: str, code_input: str) -> Dict[str, Any]:
    """Exchange authorization code for tokens. Persists on success."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess or sess["provider"] != "anthropic" or sess["flow"] != "pkce":
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    if sess["status"] != "pending":
        return {
            "ok": False,
            "status": sess["status"],
            "message": sess.get("error_message"),
        }

    # Anthropic's redirect callback page formats the code as `<code>#<state>`.
    # Strip the state suffix if present (we already have the verifier server-side).
    parts = code_input.strip().split("#", 1)
    code = parts[0].strip()
    if not code:
        return {"ok": False, "status": "error", "message": "No code provided"}
    state_from_callback = parts[1] if len(parts) > 1 else ""

    exchange_data = json.dumps(
        {
            "grant_type": "authorization_code",
            "client_id": _ANTHROPIC_OAUTH_CLIENT_ID,
            "code": code,
            "state": state_from_callback or sess["state"],
            "redirect_uri": _ANTHROPIC_OAUTH_REDIRECT_URI,
            "code_verifier": sess["verifier"],
        }
    ).encode()
    req = urllib.request.Request(
        _ANTHROPIC_OAUTH_TOKEN_URL,
        data=exchange_data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "hermes-dashboard/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        sess["status"] = "error"
        sess["error_message"] = f"Token exchange failed: {e}"
        return {"ok": False, "status": "error", "message": sess["error_message"]}

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = int(result.get("expires_in") or 3600)
    if not access_token:
        sess["status"] = "error"
        sess["error_message"] = "No access token returned"
        return {"ok": False, "status": "error", "message": sess["error_message"]}

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    try:
        _save_anthropic_oauth_creds(access_token, refresh_token, expires_at_ms)
    except Exception as e:
        sess["status"] = "error"
        sess["error_message"] = f"Save failed: {e}"
        return {"ok": False, "status": "error", "message": sess["error_message"]}
    sess["status"] = "approved"
    _log.info("oauth/pkce: anthropic login completed (session=%s)", session_id)
    return {"ok": True, "status": "approved"}


async def _start_device_code_flow(provider_id: str) -> Dict[str, Any]:
    """Initiate a device-code flow (Nous or OpenAI Codex).

    Calls the provider's device-auth endpoint via the existing CLI helpers,
    then spawns a background poller. Returns the user-facing display fields
    so the UI can render the verification page link + user code.
    """
    from hermes_cli import auth as hauth

    if provider_id == "nous":
        from hermes_cli.auth import _request_device_code, PROVIDER_REGISTRY
        import httpx

        pconfig = PROVIDER_REGISTRY["nous"]
        portal_base_url = (
            os.getenv("HERMES_PORTAL_BASE_URL")
            or os.getenv("NOUS_PORTAL_BASE_URL")
            or pconfig.portal_base_url
        ).rstrip("/")
        client_id = pconfig.client_id
        scope = pconfig.scope

        def _do_nous_device_request():
            with httpx.Client(
                timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}
            ) as client:
                return _request_device_code(
                    client=client,
                    portal_base_url=portal_base_url,
                    client_id=client_id,
                    scope=scope,
                )

        device_data = await asyncio.get_event_loop().run_in_executor(
            None, _do_nous_device_request
        )
        sid, sess = _new_oauth_session("nous", "device_code")
        sess["device_code"] = str(device_data["device_code"])
        sess["interval"] = int(device_data["interval"])
        sess["expires_at"] = time.time() + int(device_data["expires_in"])
        sess["portal_base_url"] = portal_base_url
        sess["client_id"] = client_id
        threading.Thread(
            target=_nous_poller, args=(sid,), daemon=True, name=f"oauth-poll-{sid[:6]}"
        ).start()
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": str(device_data["user_code"]),
            "verification_url": str(device_data["verification_uri_complete"]),
            "expires_in": int(device_data["expires_in"]),
            "poll_interval": int(device_data["interval"]),
        }

    if provider_id == "openai-codex":
        # Codex uses fixed OpenAI device-auth endpoints; reuse the helper.
        sid, _ = _new_oauth_session("openai-codex", "device_code")
        # Use the helper but in a thread because it polls inline.
        # We can't extract just the start step without refactoring auth.py,
        # so we run the full helper in a worker and proxy the user_code +
        # verification_url back via the session dict. The helper prints
        # to stdout — we capture nothing here, just status.
        threading.Thread(
            target=_codex_full_login_worker,
            args=(sid,),
            daemon=True,
            name=f"oauth-codex-{sid[:6]}",
        ).start()
        # Block briefly until the worker has populated the user_code, OR error.
        deadline = time.time() + 10
        while time.time() < deadline:
            with _oauth_sessions_lock:
                s = _oauth_sessions.get(sid)
            if s and (s.get("user_code") or s["status"] != "pending"):
                break
            await asyncio.sleep(0.1)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(sid, {})
        if s.get("status") == "error":
            raise HTTPException(
                status_code=500, detail=s.get("error_message") or "device-auth failed"
            )
        if not s.get("user_code"):
            raise HTTPException(
                status_code=504,
                detail="device-auth timed out before returning a user code",
            )
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": s["user_code"],
            "verification_url": s["verification_url"],
            "expires_in": int(s.get("expires_in") or 900),
            "poll_interval": int(s.get("interval") or 5),
        }

    raise HTTPException(
        status_code=400,
        detail=f"Provider {provider_id} does not support device-code flow",
    )


def _nous_poller(session_id: str) -> None:
    """Background poller that drives a Nous device-code flow to completion."""
    from hermes_cli.auth import _poll_for_token, refresh_nous_oauth_from_state
    from datetime import datetime, timezone
    import httpx

    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        return
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    device_code = sess["device_code"]
    interval = sess["interval"]
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    try:
        with httpx.Client(
            timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}
        ) as client:
            token_data = _poll_for_token(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                device_code=device_code,
                expires_in=expires_in,
                poll_interval=interval,
            )
        # Same post-processing as _nous_device_code_login (mint agent key)
        now = datetime.now(timezone.utc)
        token_ttl = int(token_data.get("expires_in") or 0)
        auth_state = {
            "portal_base_url": portal_base_url,
            "inference_base_url": token_data.get("inference_base_url"),
            "client_id": client_id,
            "scope": token_data.get("scope"),
            "token_type": token_data.get("token_type", "Bearer"),
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "obtained_at": now.isoformat(),
            "expires_at": (
                datetime.fromtimestamp(
                    now.timestamp() + token_ttl, tz=timezone.utc
                ).isoformat()
                if token_ttl
                else None
            ),
            "expires_in": token_ttl,
        }
        full_state = refresh_nous_oauth_from_state(
            auth_state,
            min_key_ttl_seconds=300,
            timeout_seconds=15.0,
            force_refresh=False,
            force_mint=True,
        )
        # Save into credential pool same as auth_commands.py does
        from agent.credential_pool import (
            PooledCredential,
            load_pool,
            AUTH_TYPE_OAUTH,
            SOURCE_MANUAL,
        )

        pool = load_pool("nous")
        entry = PooledCredential.from_dict(
            "nous",
            {
                **full_state,
                "label": "dashboard device_code",
                "auth_type": AUTH_TYPE_OAUTH,
                "source": f"{SOURCE_MANUAL}:dashboard_device_code",
                "base_url": full_state.get("inference_base_url"),
            },
        )
        pool.add_entry(entry)
        # Also persist to auth store so get_nous_auth_status() sees it
        # (matches what _login_nous in auth.py does for the CLI flow).
        try:
            from hermes_cli.auth import (
                _load_auth_store,
                _save_provider_state,
                _save_auth_store,
                _auth_store_lock,
            )

            with _auth_store_lock():
                auth_store = _load_auth_store()
                _save_provider_state(auth_store, "nous", full_state)
                _save_auth_store(auth_store)
        except Exception as store_exc:
            _log.warning(
                "oauth/device: credential pool saved but auth store write failed "
                "(session=%s): %s",
                session_id,
                store_exc,
            )
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: nous login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("nous device-code poll failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = str(e)


def _codex_full_login_worker(session_id: str) -> None:
    """Run the complete OpenAI Codex device-code flow.

    Codex doesn't use the standard OAuth device-code endpoints; it has its
    own ``/api/accounts/deviceauth/usercode`` (JSON body, returns
    ``device_auth_id``) and ``/api/accounts/deviceauth/token`` (JSON body
    polled until 200). On success the response carries an
    ``authorization_code`` + ``code_verifier`` that get exchanged at
    CODEX_OAUTH_TOKEN_URL with grant_type=authorization_code.

    The flow is replicated inline (rather than calling
    _codex_device_code_login) because that helper prints/blocks/polls in a
    single function — we need to surface the user_code to the dashboard the
    moment we receive it, well before polling completes.
    """
    try:
        import httpx
        from hermes_cli.auth import (
            CODEX_OAUTH_CLIENT_ID,
            CODEX_OAUTH_TOKEN_URL,
            DEFAULT_CODEX_BASE_URL,
        )

        issuer = "https://auth.openai.com"

        # Step 1: request device code
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(
                f"{issuer}/api/accounts/deviceauth/usercode",
                json={"client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code != 200:
            raise RuntimeError(f"deviceauth/usercode returned {resp.status_code}")
        device_data = resp.json()
        user_code = device_data.get("user_code", "")
        device_auth_id = device_data.get("device_auth_id", "")
        poll_interval = max(3, int(device_data.get("interval", "5")))
        if not user_code or not device_auth_id:
            raise RuntimeError(
                "device-code response missing user_code or device_auth_id"
            )
        verification_url = f"{issuer}/codex/device"
        with _oauth_sessions_lock:
            sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            sess["user_code"] = user_code
            sess["verification_url"] = verification_url
            sess["device_auth_id"] = device_auth_id
            sess["interval"] = poll_interval
            sess["expires_in"] = 15 * 60  # OpenAI's effective limit
            sess["expires_at"] = time.time() + sess["expires_in"]

        # Step 2: poll until authorized
        deadline = time.time() + sess["expires_in"]
        code_resp = None
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            while time.time() < deadline:
                time.sleep(poll_interval)
                poll = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )
                if poll.status_code == 200:
                    code_resp = poll.json()
                    break
                if poll.status_code in (403, 404):
                    continue  # user hasn't authorized yet
                raise RuntimeError(f"deviceauth/token poll returned {poll.status_code}")

        if code_resp is None:
            with _oauth_sessions_lock:
                sess["status"] = "expired"
                sess["error_message"] = "Device code expired before approval"
            return

        # Step 3: exchange authorization_code for tokens
        authorization_code = code_resp.get("authorization_code", "")
        code_verifier = code_resp.get("code_verifier", "")
        if not authorization_code or not code_verifier:
            raise RuntimeError(
                "device-auth response missing authorization_code/code_verifier"
            )
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            token_resp = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": f"{issuer}/deviceauth/callback",
                    "client_id": CODEX_OAUTH_CLIENT_ID,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if token_resp.status_code != 200:
            raise RuntimeError(f"token exchange returned {token_resp.status_code}")
        tokens = token_resp.json()
        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")
        if not access_token:
            raise RuntimeError("token exchange did not return access_token")

        # Persist via credential pool — same shape as auth_commands.add_command
        from agent.credential_pool import (
            PooledCredential,
            load_pool,
            AUTH_TYPE_OAUTH,
            SOURCE_MANUAL,
        )
        import uuid as _uuid

        pool = load_pool("openai-codex")
        base_url = (
            os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
            or DEFAULT_CODEX_BASE_URL
        )
        entry = PooledCredential(
            provider="openai-codex",
            id=_uuid.uuid4().hex[:6],
            label="dashboard device_code",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:dashboard_device_code",
            access_token=access_token,
            refresh_token=refresh_token,
            base_url=base_url,
        )
        pool.add_entry(entry)
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: openai-codex login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("codex device-code worker failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(session_id)
            if s:
                s["status"] = "error"
                s["error_message"] = str(e)


@app.post("/api/providers/oauth/{provider_id}/start")
async def start_oauth_login(provider_id: str, request: Request):
    """Initiate an OAuth login flow. Token-protected."""
    _require_token(request)
    _gc_oauth_sessions()
    valid = {p["id"] for p in _OAUTH_PROVIDER_CATALOG}
    if provider_id not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider_id}")
    catalog_entry = next(p for p in _OAUTH_PROVIDER_CATALOG if p["id"] == provider_id)
    if catalog_entry["flow"] == "external":
        raise HTTPException(
            status_code=400,
            detail=f"{provider_id} uses an external CLI; run `{catalog_entry['cli_command']}` manually",
        )
    try:
        if catalog_entry["flow"] == "pkce":
            return _start_anthropic_pkce()
        if catalog_entry["flow"] == "device_code":
            return await _start_device_code_flow(provider_id)
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("oauth/start %s failed", provider_id)
        raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail="Unsupported flow")


class OAuthSubmitBody(BaseModel):
    session_id: str
    code: str


@app.post("/api/providers/oauth/{provider_id}/submit")
async def submit_oauth_code(provider_id: str, body: OAuthSubmitBody, request: Request):
    """Submit the auth code for PKCE flows. Token-protected."""
    _require_token(request)
    if provider_id == "anthropic":
        return await asyncio.get_event_loop().run_in_executor(
            None,
            _submit_anthropic_pkce,
            body.session_id,
            body.code,
        )
    raise HTTPException(
        status_code=400, detail=f"submit not supported for {provider_id}"
    )


@app.get("/api/providers/oauth/{provider_id}/poll/{session_id}")
async def poll_oauth_session(provider_id: str, session_id: str):
    """Poll a device-code session's status (no auth — read-only state)."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if sess["provider"] != provider_id:
        raise HTTPException(status_code=400, detail="Provider mismatch for session")
    return {
        "session_id": session_id,
        "status": sess["status"],
        "error_message": sess.get("error_message"),
        "expires_at": sess.get("expires_at"),
    }


@app.delete("/api/providers/oauth/sessions/{session_id}")
async def cancel_oauth_session(session_id: str, request: Request):
    """Cancel a pending OAuth session. Token-protected."""
    _require_token(request)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.pop(session_id, None)
    if sess is None:
        return {"ok": False, "message": "session not found"}
    return {"ok": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# HermesWeb chat + realtime endpoints
# ---------------------------------------------------------------------------


@app.get("/api/chat/history")
async def get_chat_history(
    session_id: Optional[str] = None, sessionId: Optional[str] = None
):
    requested_session_id = session_id or sessionId
    if not requested_session_id:
        raise HTTPException(
            status_code=400, detail="session_id query parameter is required"
        )

    from hermes_state import SessionDB

    db = SessionDB()
    try:
        sid = _resolve_chat_session_id(db, requested_session_id)
        messages = [_serialize_chat_message(msg) for msg in db.get_messages(sid)]
        return {
            "session_id": sid,
            "messages": messages,
            "total": len(messages),
        }
    finally:
        db.close()


@app.post("/api/chat")
async def post_chat(body: ChatRequest):
    """Start a chat run asynchronously.

    Returns {run_id, session_id, status: "queued"} immediately.
    The agent executes in a background task and emits WebSocket events:
      chat_started → chat_completed | chat_failed | chat_timeout

    Clients that support the run registry poll GET /api/chat/runs/{run_id}
    as a fallback if the WebSocket is unavailable.
    """
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must not be empty")

    # Accept client-provided run_id (for idempotency) or generate one
    run_id: str = body.run_id or f"run_{uuid.uuid4().hex}"
    session_id = body.resolved_session_id

    _log.info("[chat-run] queued   run_id=%s session_id=%s", run_id, session_id)
    _RUN_REGISTRY.create_run(
        run_id, session_id, content, provider=body.provider, model=body.model
    )

    # Start background execution (non-blocking)
    task = asyncio.create_task(_execute_chat_run(run_id, body))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    return {
        "run_id": run_id,
        "session_id": session_id,
        "status": "queued",
        # null fields preserve backward-compat with sync-expecting clients
        "message": None,
        "response": None,
        "messages": [],
    }


@app.get("/api/chat/runs/{run_id}/steps")
async def get_chat_run_steps(run_id: str):
    """Return timeline steps for a chat run (persisted).

    Response shape (snake_case primary, camelCase aliases for frontend compat):
    {
      "run_id": "...",
      "steps": [
        {
          "id": "...", "runId": "...", "run_id": "...",
          "type": "log | message | tool_call | tool_result",
          "title": "...", "content": "...",
          "status": "pending | running | completed | failed",
          "createdAt": "...", "updatedAt": "...",
          "created_at": "...", "updated_at": "..."
        }
      ]
    }
    """
    steps = _RUN_REGISTRY.get_steps(run_id)
    enriched = []
    for step in steps:
        enriched.append(
            {
                **step,
                "runId": step.get("run_id"),
                "createdAt": step.get("created_at"),
                "updatedAt": step.get("updated_at"),
            }
        )
    return {"run_id": run_id, "steps": enriched}


@app.get("/api/chat/runs/{run_id}")
async def get_chat_run(run_id: str):
    """Return the current state of a chat run.

    Response shape (camelCase aliases accepted by frontend hermes.ts parser):
    {
      "run_id": "...",
      "session_id": "...",
      "status": "queued | running | completed | failed | timeout",
      "messages": [...],
      "error": null,
      "started_at": "ISO8601",
      "updated_at": "ISO8601",
      "completed_at": "ISO8601 | null"
    }
    """
    run = _RUN_REGISTRY.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return run


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not await _require_ws_token(websocket):
        return

    session_id = websocket.query_params.get("session_id") or None
    connection_id = await _REALTIME_HUB.connect(websocket, session_id=session_id)

    try:
        await websocket.send_json(
            {
                "type": "hello",
                "connection_id": connection_id,
                "sent_at": _utc_now_iso(),
                "session_id": session_id,
                "status": {
                    "version": __version__,
                    "release_date": __release_date__,
                    "backend": "hermes-agent",
                },
            }
        )

        while True:
            try:
                incoming = await asyncio.wait_for(websocket.receive_text(), timeout=15)
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "connection_id": connection_id,
                        "sent_at": _utc_now_iso(),
                        "session_id": session_id,
                    }
                )
                continue

            if incoming.strip().lower() == "ping":
                await websocket.send_json(
                    {
                        "type": "pong",
                        "connection_id": connection_id,
                        "sent_at": _utc_now_iso(),
                        "session_id": session_id,
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "type": "info",
                        "connection_id": connection_id,
                        "sent_at": _utc_now_iso(),
                        "message": "Send 'ping' to keep the connection warm.",
                        "session_id": session_id,
                    }
                )
    except WebSocketDisconnect:
        pass
    finally:
        await _REALTIME_HUB.disconnect(connection_id)


# ---------------------------------------------------------------------------
# Agents endpoint
# ---------------------------------------------------------------------------


def _derive_agent_status(session: Dict[str, Any], now: float) -> str:
    if session.get("ended_at"):
        reason = (session.get("end_reason") or "").lower()
        if "error" in reason or "fail" in reason:
            return "error"
        return "ended"
    last_active = session.get("last_active") or session.get("started_at") or 0
    if now - float(last_active) < 300:
        return "active"
    return "idle"


def _session_to_agent(session: Dict[str, Any], now: float) -> Dict[str, Any]:
    session_id = session.get("id", "")
    parent_session_id = session.get("parent_session_id")
    kind = "subagent" if parent_session_id else "primary"
    title = session.get("title") or "Hermes"
    last_active = session.get("last_active") or session.get("started_at")
    updated_at = (
        _utc_now_iso()
        if not last_active
        else datetime.fromtimestamp(float(last_active), tz=timezone.utc).isoformat()
    )
    return {
        "id": f"agent-{session_id}" if kind == "primary" else f"subagent-{session_id}",
        "session_id": session_id,
        "name": title,
        "kind": kind,
        "status": _derive_agent_status(session, now),
        "model": session.get("model") or "",
        "platform": session.get("source") or "unknown",
        "updated_at": updated_at,
    }


@app.get("/api/agents")
async def get_agents(limit: int = 50):
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            sessions = db.list_sessions_rich(limit=limit)
            now = time.time()
            agents = [_session_to_agent(s, now) for s in sessions]
            return {"agents": agents}
        finally:
            db.close()
    except Exception:
        _log.exception("GET /api/agents failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Approvals endpoints
# ---------------------------------------------------------------------------

_approval_db = None


def _get_approval_db():
    global _approval_db
    if _approval_db is None:
        from hermes_state import ApprovalDB

        _approval_db = ApprovalDB()
    return _approval_db


def _generate_approval_id(session_key: str, index: int) -> str:
    raw = f"{session_key}:{index}:{time.time()}"
    import hashlib

    return f"approval-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _serialize_approval_entry(
    entry_data: Dict[str, Any],
    approval_id: str,
    session_key: str,
) -> Dict[str, Any]:
    command = entry_data.get("command", "")
    description = entry_data.get("description", "")
    pattern_keys = entry_data.get("pattern_keys", [])
    title = (
        description
        if description
        else pattern_keys[0]
        if pattern_keys
        else "dangerous command"
    )
    session_id = entry_data.get("session_id", session_key)
    agent_id = f"agent-{session_id}" if session_id else f"agent-{session_key}"
    created_at = entry_data.get("created_at")
    if created_at:
        if isinstance(created_at, (int, float)):
            created_at = datetime.fromtimestamp(
                float(created_at), tz=timezone.utc
            ).isoformat()
    else:
        created_at = _utc_now_iso()
    return {
        "id": approval_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "status": entry_data.get("_resolved_status", "pending"),
        "title": title,
        "kind": "command",
        "details": f"Command matches dangerous pattern: {description}"
        if description
        else "Dangerous command detected",
        "command": command,
        "created_at": created_at,
    }


def _collect_pending_approvals() -> List[Dict[str, Any]]:
    from tools.approval import _gateway_queues, _lock

    result = []
    with _lock:
        for session_key, entries in _gateway_queues.items():
            for i, entry in enumerate(entries):
                entry_data = dict(entry.data)
                approval_id = _generate_approval_id(session_key, i)
                entry_data["_resolved_status"] = "pending"
                entry_data.setdefault("session_id", session_key)
                if "created_at" not in entry_data:
                    entry_data["created_at"] = _utc_now_iso()
                result.append(
                    _serialize_approval_entry(entry_data, approval_id, session_key)
                )
    return result


@app.get("/api/approvals")
async def get_approvals(status: Optional[str] = None, limit: int = 50):
    try:
        approvals = _collect_pending_approvals()
        db_approvals = _get_approval_db().list_approvals(
            status=status if status and status != "pending" else None, limit=limit
        )
        if status and status != "pending":
            approvals.extend(db_approvals)
        elif status is None:
            approvals.extend(db_approvals)
        return {"approvals": approvals[:limit], "total": len(approvals)}
    except Exception:
        _log.exception("GET /api/approvals failed")
        raise HTTPException(status_code=500, detail="Internal server error")


def _resolve_approval_by_id(approval_id: str, choice: str) -> Dict[str, Any]:
    from tools.approval import _gateway_queues, _lock, resolve_gateway_approval

    resolved_status = "approved" if choice != "deny" else "rejected"

    with _lock:
        for session_key, entries in list(_gateway_queues.items()):
            for i, entry in enumerate(entries):
                candidate_id = _generate_approval_id(session_key, i)
                if candidate_id == approval_id:
                    if entry.result is not None:
                        return {
                            "ok": False,
                            "error": "already_resolved",
                            "approval_id": approval_id,
                            "current_status": "approved"
                            if entry.result != "deny"
                            else "rejected",
                        }
                    resolved_count = resolve_gateway_approval(session_key, choice)
                    try:
                        _get_approval_db().resolve_approval(
                            approval_id,
                            resolved_status,
                            resolved_by=session_key,
                            choice=choice,
                        )
                    except Exception:
                        pass
                    return {
                        "ok": resolved_count > 0,
                        "approval_id": approval_id,
                        "status": resolved_status,
                    }

    db_approval = _get_approval_db().get_approval(approval_id)
    if db_approval:
        if db_approval.get("status") != "pending":
            return {
                "ok": False,
                "error": "already_resolved",
                "approval_id": approval_id,
                "current_status": db_approval.get("status", "unknown"),
            }
        try:
            _get_approval_db().resolve_approval(
                approval_id, resolved_status, resolved_by="api", choice=choice
            )
        except Exception:
            pass
        return {
            "ok": True,
            "approval_id": approval_id,
            "status": resolved_status,
        }

    return None


@app.post("/api/approvals/{approval_id}/approve")
async def approve_approval(approval_id: str):
    result = _resolve_approval_by_id(approval_id, "session")
    if result is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    if result.get("error") == "already_resolved":
        raise HTTPException(
            status_code=409,
            detail=f"Approval already {result.get('current_status', 'resolved')}",
        )
    await _REALTIME_HUB.broadcast(
        "approval.resolved",
        {
            "approval_id": approval_id,
            "status": result.get("status", "approved"),
        },
    )
    return result


@app.post("/api/approvals/{approval_id}/reject")
async def reject_approval(approval_id: str):
    result = _resolve_approval_by_id(approval_id, "deny")
    if result is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    if result.get("error") == "already_resolved":
        raise HTTPException(
            status_code=409,
            detail=f"Approval already {result.get('current_status', 'resolved')}",
        )
    await _REALTIME_HUB.broadcast(
        "approval.resolved",
        {
            "approval_id": approval_id,
            "status": result.get("status", "rejected"),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Tasks endpoints
# ---------------------------------------------------------------------------

_task_db = None


def _get_task_db():
    global _task_db
    if _task_db is None:
        from hermes_state import TaskDB

        _task_db = TaskDB()
    return _task_db


def _generate_task_id() -> str:
    import hashlib

    raw = f"{time.time()}:{secrets.token_urlsafe(8)}"
    return f"task-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Optional[str] = "medium"
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None


@app.get("/api/tasks")
async def get_tasks(
    status: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 100,
):
    try:
        tasks = _get_task_db().list_tasks(
            status=status,
            agent_id=agent_id,
            session_id=session_id,
            limit=limit,
        )
        return {"tasks": tasks, "total": len(tasks)}
    except Exception:
        _log.exception("GET /api/tasks failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/tasks")
async def create_task(body: TaskCreate):
    try:
        task_id = _generate_task_id()
        task = _get_task_db().create_task(
            task_id=task_id,
            title=body.title,
            description=body.description,
            priority=body.priority or "medium",
            agent_id=body.agent_id,
            session_id=body.session_id,
            run_id=body.run_id,
        )
        await _REALTIME_HUB.broadcast(
            "task.created",
            {"task": task},
        )
        return task
    except Exception:
        _log.exception("POST /api/tasks failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.patch("/api/tasks/{task_id}")
async def update_task(task_id: str, body: TaskUpdate):
    try:
        updates = body.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        if body.status is not None:
            valid_statuses = {
                "todo",
                "in_progress",
                "review",
                "done",
                "archived",
                "failed",
                "timeout",
                "cancelled",
            }
            if body.status not in valid_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: {', '.join(sorted(valid_statuses))}",
                )

        if body.priority is not None:
            valid_priorities = {"low", "medium", "high"}
            if body.priority not in valid_priorities:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid priority. Must be one of: {', '.join(sorted(valid_priorities))}",
                )

        task = _get_task_db().update_task(task_id, updates)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        await _REALTIME_HUB.broadcast(
            "task.updated",
            {"task": task},
        )
        return task
    except HTTPException:
        raise
    except Exception:
        _log.exception("PATCH /api/tasks/%s failed", task_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    try:
        deleted = _get_task_db().delete_task(task_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Task not found")

        await _REALTIME_HUB.broadcast(
            "task.deleted",
            {"task_id": task_id},
        )
        return {"ok": True}
    except HTTPException:
        raise
    except Exception:
        _log.exception("DELETE /api/tasks/%s failed", task_id)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Summary endpoint
# ---------------------------------------------------------------------------


@app.get("/api/summary/daily")
async def get_summary_daily(date: Optional[str] = None):
    try:
        from hermes_state import SessionDB

        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use YYYY-MM-DD.",
                )

        session_db = SessionDB()
        task_db = _get_task_db()
        approval_db = _get_approval_db()

        try:
            sessions_total_today = session_db.sessions_today_count(date)
            sessions_active = session_db.active_sessions_count()

            task_counts = task_db.count_by_status()
            tasks_completed_today = task_db.completed_today_count(date)

            approvals_pending = approval_db.get_pending_count()
            approvals_approved_today = approval_db.resolved_today_count(
                "approved", date
            )
            approvals_rejected_today = approval_db.resolved_today_count(
                "rejected", date
            )

            now = time.time()
            cutoff = now - 300
            with session_db._lock:
                cursor = session_db._conn.execute(
                    """SELECT COUNT(*) FROM sessions s
                       WHERE s.ended_at IS NULL
                         AND COALESCE(
                             (SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = s.id),
                             s.started_at
                         ) < ?""",
                    (cutoff,),
                )
                agents_idle = cursor.fetchone()[0]

            return {
                "date": date,
                "sessions": {
                    "total_today": sessions_total_today,
                    "active": sessions_active,
                },
                "tasks": {
                    "total": sum(task_counts.values()),
                    "todo": task_counts.get("todo", 0),
                    "in_progress": task_counts.get("in_progress", 0),
                    "review": task_counts.get("review", 0),
                    "done": task_counts.get("done", 0),
                    "archived": task_counts.get("archived", 0),
                    "completed_today": tasks_completed_today,
                },
                "approvals": {
                    "pending": approvals_pending,
                    "approved_today": approvals_approved_today,
                    "rejected_today": approvals_rejected_today,
                },
                "agents": {
                    "active": sessions_active,
                    "idle": agents_idle,
                },
            }
        finally:
            session_db.close()
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/summary/daily failed")
        raise HTTPException(status_code=500, detail="Internal server error")


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


@app.get("/api/sessions/{session_id}/artifacts")
async def get_session_artifacts(session_id: str):
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        try:
            artifacts = db.get_artifacts_by_session(sid)
        except Exception as exc:
            _log.error(
                "get_artifacts_by_session failed for %s: %s", sid, exc, exc_info=True
            )
            artifacts = []
        return {"session_id": sid, "artifacts": artifacts, "total": len(artifacts)}
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
# Code Workspace endpoints
# ---------------------------------------------------------------------------


class _OpenWorkspaceBody(BaseModel):
    path: str


@app.get("/api/code/workspaces")
async def list_code_workspaces():
    from hermes_cli.code.workspace_service import CodeWorkspaceService

    try:
        svc = CodeWorkspaceService()
        workspaces = svc.list_workspaces()
        return {"workspaces": workspaces, "total": len(workspaces)}
    except Exception as exc:
        _log.error("list_code_workspaces failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/workspaces/open")
async def open_code_workspace(body: _OpenWorkspaceBody):
    from hermes_cli.code.workspace_service import CodeWorkspaceService

    try:
        svc = CodeWorkspaceService()
        workspace = svc.open_workspace(body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error(
            "open_code_workspace failed for %s: %s", body.path, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    # Emit WebSocket event (best-effort)
    try:
        await _REALTIME_HUB.broadcast(
            "code_workspace.updated",
            {"payload": {"workspace": workspace}},
        )
    except Exception:
        pass

    return {"workspace": workspace}


@app.get("/api/code/workspaces/{workspace_id}")
async def get_code_workspace(workspace_id: str):
    from hermes_cli.code.workspace_service import CodeWorkspaceService

    try:
        svc = CodeWorkspaceService()
        workspace = svc.get_workspace(workspace_id)
    except Exception as exc:
        _log.error(
            "get_code_workspace failed for %s: %s", workspace_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"workspace": workspace}


@app.post("/api/code/workspaces/{workspace_id}/refresh")
async def refresh_code_workspace(workspace_id: str):
    from hermes_cli.code.workspace_service import CodeWorkspaceService

    try:
        svc = CodeWorkspaceService()
        workspace = svc.refresh_workspace(workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "refresh_code_workspace failed for %s: %s", workspace_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    # Emit WebSocket event (best-effort)
    try:
        await _REALTIME_HUB.broadcast(
            "code_workspace.updated",
            {"payload": {"workspace": workspace}},
        )
    except Exception:
        pass

    return {"workspace": workspace}


# ---------------------------------------------------------------------------
# Code Session endpoints
# ---------------------------------------------------------------------------


class _CreateCodeSessionBody(BaseModel):
    workspace_id: str
    hermes_session_id: Optional[str] = None
    task_id: Optional[str] = None
    title: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[dict] = None


class _PatchCodeSessionBody(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    summary: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    task_id: Optional[str] = None
    hermes_session_id: Optional[str] = None
    metadata: Optional[dict] = None


class _CancelCodeSessionBody(BaseModel):
    reason: Optional[str] = None


class _CompleteCodeSessionBody(BaseModel):
    summary: Optional[str] = None


class _AddCodeSessionEventBody(BaseModel):
    type: str
    message: Optional[str] = None
    payload: Optional[dict] = None


class _RunCommandBody(BaseModel):
    command: str
    cwd: Optional[str] = None
    timeout_seconds: int = 120


@app.get("/api/code/sessions")
async def list_code_sessions(
    workspace_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        sessions = svc.list_sessions(
            workspace_id=workspace_id, status=status, limit=limit, offset=offset
        )
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as exc:
        _log.error("list_code_sessions failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/sessions")
async def create_code_session(body: _CreateCodeSessionBody):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        session = svc.create_session(
            workspace_id=body.workspace_id,
            hermes_session_id=body.hermes_session_id,
            task_id=body.task_id,
            title=body.title,
            provider=body.provider,
            model=body.model,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("create_code_session failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        await _REALTIME_HUB.broadcast(
            "code_session.created",
            {"payload": {"code_session": session}},
        )
    except Exception:
        pass

    return {"code_session": session}


@app.get("/api/code/sessions/{code_session_id}")
async def get_code_session(code_session_id: str):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        session = svc.get_session(code_session_id)
    except Exception as exc:
        _log.error(
            "get_code_session failed for %s: %s", code_session_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    if not session:
        raise HTTPException(status_code=404, detail="Code session not found")
    return {"code_session": session}


@app.patch("/api/code/sessions/{code_session_id}")
async def patch_code_session(code_session_id: str, body: _PatchCodeSessionBody):
    from hermes_cli.code.session_service import CodeSessionService

    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        svc = CodeSessionService()
        session = svc.update_session(code_session_id, **updates)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as exc:
        _log.error(
            "patch_code_session failed for %s: %s", code_session_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        await _REALTIME_HUB.broadcast(
            "code_session.updated",
            {"payload": {"code_session": session}},
        )
        if updates.get("status"):
            await _REALTIME_HUB.broadcast(
                "code_session.status_changed",
                {
                    "payload": {
                        "code_session_id": code_session_id,
                        "status": updates["status"],
                    }
                },
            )
    except Exception:
        pass

    return {"code_session": session}


@app.post("/api/code/sessions/{code_session_id}/cancel")
async def cancel_code_session(
    code_session_id: str, body: _CancelCodeSessionBody = _CancelCodeSessionBody()
):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        session = svc.cancel_session(code_session_id, reason=body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "cancel_code_session failed for %s: %s", code_session_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        await _REALTIME_HUB.broadcast(
            "code_session.cancelled",
            {"payload": {"code_session": session}},
        )
    except Exception:
        pass

    return {"code_session": session}


@app.post("/api/code/sessions/{code_session_id}/complete")
async def complete_code_session(
    code_session_id: str, body: _CompleteCodeSessionBody = _CompleteCodeSessionBody()
):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        session = svc.complete_session(code_session_id, summary=body.summary)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "complete_code_session failed for %s: %s",
            code_session_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        await _REALTIME_HUB.broadcast(
            "code_session.completed",
            {"payload": {"code_session": session}},
        )
    except Exception:
        pass

    return {"code_session": session}


@app.get("/api/code/sessions/{code_session_id}/events")
async def list_code_session_events(code_session_id: str):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        if not svc.get_session(code_session_id):
            raise HTTPException(status_code=404, detail="Code session not found")
        events = svc.list_events(code_session_id)
        return {"events": events, "total": len(events)}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error(
            "list_code_session_events failed for %s: %s",
            code_session_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/sessions/{code_session_id}/events")
async def add_code_session_event(code_session_id: str, body: _AddCodeSessionEventBody):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        event = svc.add_event(
            code_session_id,
            event_type=body.type,
            message=body.message,
            payload=body.payload,
        )
        return {"event": event}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "add_code_session_event failed for %s: %s",
            code_session_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/sessions/{code_session_id}/artifacts")
async def list_code_session_artifacts(code_session_id: str):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        artifacts = svc.list_artifacts(code_session_id)
        return {
            "code_session_id": code_session_id,
            "artifacts": artifacts,
            "total": len(artifacts),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "list_code_session_artifacts failed for %s: %s",
            code_session_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/sessions/{code_session_id}/artifacts/{artifact_id}/link")
async def link_artifact_to_code_session(code_session_id: str, artifact_id: str):
    from hermes_cli.code.session_service import CodeSessionService

    try:
        svc = CodeSessionService()
        artifact = svc.link_artifact(code_session_id, artifact_id)
        return {"artifact": artifact}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "link_artifact_to_code_session failed %s/%s: %s",
            code_session_id,
            artifact_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Log viewer endpoint
# ---------------------------------------------------------------------------


@app.get("/api/logs")
async def get_logs(
    file: str = "agent",
    lines: int = 100,
    level: Optional[str] = None,
    component: Optional[str] = None,
    search: Optional[str] = None,
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

    # Normalize "ALL" / "all" / empty → no filter. _matches_filters treats an
    # empty tuple as "must match a prefix" (startswith(()) is always False),
    # so passing () instead of None silently drops every line.
    min_level = level if level and level.upper() != "ALL" else None
    if component and component.lower() != "all":
        comp_prefixes = COMPONENT_PREFIXES.get(component)
        if comp_prefixes is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown component: {component}. "
                f"Available: {', '.join(sorted(COMPONENT_PREFIXES))}",
            )
    else:
        comp_prefixes = None

    has_filters = bool(min_level or comp_prefixes or search)
    result = _read_tail(
        log_path,
        min(lines, 500) if not search else 2000,
        has_filters=has_filters,
        min_level=min_level,
        component_prefixes=comp_prefixes,
    )
    # Post-filter by search term (case-insensitive substring match).
    # _read_tail doesn't support free-text search, so we filter here and
    # trim to the requested line count afterward.
    if search:
        needle = search.lower()
        result = [l for l in result if needle in l.lower()][-min(lines, 500) :]
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
        job = create_job(
            prompt=body.prompt,
            schedule=body.schedule,
            name=body.name,
            deliver=body.deliver,
        )
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
        result.append(
            {
                "name": name,
                "label": label,
                "description": desc,
                "enabled": is_enabled,
                "available": is_enabled,
                "configured": _toolset_has_keys(name, config),
                "tools": tools,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Raw YAML config endpoint
# ---------------------------------------------------------------------------


class RawConfigUpdate(BaseModel):
    yaml_text: str


class ProviderSelectRequest(BaseModel):
    provider_id: str
    model_id: Optional[str] = None


class ProviderTestRequest(BaseModel):
    provider_id: str
    model_id: Optional[str] = None


class ProviderAddRequest(BaseModel):
    id: str
    name: Optional[str] = None
    api_key: str
    base_url: Optional[str] = None
    default_model: Optional[str] = None


class WorkspacePathRequest(BaseModel):
    path: str


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
        cur = db._conn.execute(
            """
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
        """,
            (cutoff,),
        )
        daily = [dict(r) for r in cur.fetchall()]

        cur2 = db._conn.execute(
            """
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions
            FROM sessions WHERE started_at > ? AND model IS NOT NULL
            GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """,
            (cutoff,),
        )
        by_model = [dict(r) for r in cur2.fetchall()]

        cur3 = db._conn.execute(
            """
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions
            FROM sessions WHERE started_at > ?
        """,
            (cutoff,),
        )
        totals = dict(cur3.fetchone())

        return {
            "daily": daily,
            "by_model": by_model,
            "totals": totals,
            "period_days": days,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Provider endpoints for HermesWeb
# ---------------------------------------------------------------------------

# Allowlist of known providers to expose via /api/providers.
# Built from PROVIDER_REGISTRY in auth.py + openrouter + openai-codex.
_KNOWN_WEB_PROVIDERS: List[Dict[str, Any]] = [
    {"id": "minimax", "name": "MiniMax", "env_vars": ("MINIMAX_API_KEY",)},
    {
        "id": "anthropic",
        "name": "Anthropic (Claude)",
        "env_vars": ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "env_vars": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    },
    {
        "id": "zai",
        "name": "Z.AI / GLM",
        "env_vars": ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "env_vars": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    },
    {"id": "kimi-coding", "name": "Kimi / Moonshot", "env_vars": ("KIMI_API_KEY",)},
    {"id": "deepseek", "name": "DeepSeek", "env_vars": ("DEEPSEEK_API_KEY",)},
    {"id": "alibaba", "name": "Alibaba Cloud", "env_vars": ("DASHSCOPE_API_KEY",)},
    {
        "id": "minimax-cn",
        "name": "MiniMax (China)",
        "env_vars": ("MINIMAX_CN_API_KEY",),
    },
    {
        "id": "copilot",
        "name": "GitHub Copilot",
        "env_vars": ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
    },
    {"id": "xai", "name": "xAI", "env_vars": ("XAI_API_KEY",)},
    {"id": "xiaomi", "name": "Xiaomi MiMo", "env_vars": ("XIAOMI_API_KEY",)},
    {"id": "arcee", "name": "Arcee AI", "env_vars": ("ARCEEAI_API_KEY",)},
    {"id": "huggingface", "name": "Hugging Face", "env_vars": ("HF_TOKEN",)},
    {"id": "nous", "name": "Nous Portal", "env_vars": ()},
    {"id": "openai-codex", "name": "OpenAI Codex", "env_vars": ()},
]

# Known provider aliases for display purposes
_PROVIDER_DISPLAY_OVERRIDES: Dict[str, str] = {
    "minimax": "MiniMax",
    "anthropic": "Anthropic (Claude)",
    "openrouter": "OpenRouter",
}


def _get_loaded_env() -> Dict[str, str]:
    """Load .env file from active profile or default hermes home."""
    try:
        return load_env()
    except Exception:
        return {}


def _check_provider_status(
    provider_id: str,
    env_vars: tuple,
    env_data: Dict[str, str],
) -> str:
    """Check if a provider has usable credentials.

    Returns: 'configured', 'missing_token', 'invalid_token', or 'unknown'.
    """
    if not env_vars:
        # OAuth-based providers — check via auth system
        try:
            from hermes_cli.auth import has_usable_secret, get_provider_auth_state

            state = get_provider_auth_state(provider_id) or {}
            for key in ("api_key", "access_token", "agent_key"):
                val = state.get(key, "")
                if has_usable_secret(val):
                    return "configured"
        except Exception:
            pass
        return "missing_token"

    for var in env_vars:
        # Check process env first, then .env file
        val = os.getenv(var, "") or env_data.get(var, "")
        if (
            val
            and val.strip()
            and val.strip().lower() not in ("", "***", "your_key_here", "xxx")
        ):
            return "configured"
    return "missing_token"


def _get_provider_models_from_catalog(provider_id: str) -> List[Dict[str, Any]]:
    """Get models list from models.dev catalog for a provider."""
    try:
        from agent.models_dev import fetch_models_dev, PROVIDER_TO_MODELS_DEV
        from hermes_cli.providers import normalize_provider

        canonical = normalize_provider(provider_id)
        mdev_id = PROVIDER_TO_MODELS_DEV.get(canonical, canonical)
        data = fetch_models_dev()
        pdata = data.get(mdev_id)
        if not isinstance(pdata, dict):
            # Also try canonical directly
            pdata = data.get(canonical)
        if not isinstance(pdata, dict):
            return []

        models_raw = pdata.get("models", {})
        if not isinstance(models_raw, dict):
            return []

        models = []
        for mid, mdata in models_raw.items():
            if not isinstance(mdata, dict):
                continue
            model: Dict[str, Any] = {"id": mid, "name": mdata.get("name", mid)}
            ctx = (
                mdata.get("limit", {}).get("context")
                if isinstance(mdata.get("limit"), dict)
                else None
            )
            if ctx:
                model["contextWindow"] = ctx
            model["supportsTools"] = bool(mdata.get("tool_call", False))
            model["supportsVision"] = bool(mdata.get("attachment", False)) or (
                "image"
                in (
                    mdata.get("modalities", {}).get("input", [])
                    if isinstance(mdata.get("modalities"), dict)
                    else []
                )
            )
            models.append(model)

        return models
    except Exception:
        return []


def _get_workspace_state_path() -> Path:
    """Path to persistent workspace state file."""
    return get_hermes_home() / "workspace_state.json"


def _load_workspace_state() -> Dict[str, Any]:
    """Load workspace state from disk."""
    path = _get_workspace_state_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"current": None, "recent": []}


def _save_workspace_state(state: Dict[str, Any]) -> None:
    """Save workspace state to disk."""
    path = _get_workspace_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _is_path_allowed(resolved: Path) -> bool:
    """Check if a resolved path is within the workspace allowlist."""
    allowed_roots = [
        Path("/home/andrey/dev"),
        Path("/home/andrey/benchmarks-go"),
        Path("/home/andrey/projects"),
        Path("/home/andrey/work"),
        Path("/tmp/hermes-workspaces"),
    ]
    # Also allow hermes home subdirs
    hermes_home = get_hermes_home()
    allowed_roots.append(hermes_home)

    for root in allowed_roots:
        try:
            if resolved.is_relative_to(root.resolve()):
                return True
        except (ValueError, OSError):
            continue
    return False


def _is_dangerous_path(resolved: Path) -> bool:
    """Check if a path matches known dangerous patterns."""
    blocked = {"/etc", "/root", "/boot", "/sys", "/proc", "/dev"}
    resolved_str = str(resolved)
    for b in blocked:
        if resolved_str == b or resolved_str.startswith(b + "/"):
            return True
    # Check for sensitive dirs
    sensitive = ["/.ssh", "/.gnupg", "/.aws", "/.config/hermes"]
    for s in sensitive:
        if s in resolved_str:
            return True
    return False


def _normalize_workspace_path(path_str: str) -> Optional[str]:
    """Convert UNC Windows paths to Linux paths and validate.

    Returns normalized absolute path or None if invalid.
    """
    if not path_str:
        return None

    # Convert UNC Windows path: \\wsl.localhost\Ubuntu\home\andrey\dev\...
    if path_str.startswith("\\\\") or path_str.startswith("//"):
        # Normalize to forward slashes
        normalized = path_str.replace("\\", "/")
        # Strip \\wsl.localhost\Ubuntu prefix
        prefixes = [
            "//wsl.localhost/Ubuntu/",
            "//wsl.localhost/ubuntu/",
            "//wsl$/Ubuntu/",
            "//wsl$/ubuntu/",
        ]
        for prefix in prefixes:
            if normalized.lower().startswith(prefix.lower()):
                normalized = "/" + normalized[len(prefix) :]
                break
        else:
            # Generic UNC → just strip leading //
            normalized = normalized.lstrip("/")
    else:
        normalized = path_str.replace("\\", "/")

    # Resolve path traversal
    try:
        resolved = Path(normalized).resolve()
    except (OSError, ValueError):
        return None

    # Block path traversal with ..
    if ".." in Path(normalized).parts:
        return None

    resolved_str = str(resolved)

    # Block dangerous paths
    if _is_dangerous_path(resolved):
        return None

    # Check allowlist
    if not _is_path_allowed(resolved):
        return None

    return resolved_str


def _detect_stack(project_path: Path) -> List[str]:
    """Detect project technology stack from files in the directory."""
    stacks = []
    checks = {
        "typescript": ["tsconfig.json", "tsconfig.*.json"],
        "javascript": ["package.json"],
        "react": ["package.json"],
        "vue": ["vue.config.js", "nuxt.config.ts"],
        "nextjs": ["next.config.js", "next.config.mjs", "next.config.ts"],
        "vite": ["vite.config.ts", "vite.config.js"],
        "python": [
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "Pipfile",
        ],
        "go": ["go.mod"],
        "rust": ["Cargo.toml"],
        "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
        "ruby": ["Gemfile"],
        "php": ["composer.json"],
        "dotnet": ["*.csproj", "*.sln"],
        "docker": [
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            "compose.yml",
        ],
        "tailwind": ["tailwind.config.js", "tailwind.config.ts"],
        "prisma": ["prisma/schema.prisma"],
    }

    for stack, patterns in checks.items():
        for pattern in patterns:
            if "*" in pattern:
                matches = list(project_path.glob(pattern))
                if matches:
                    stacks.append(stack)
                    break
            else:
                if (project_path / pattern).exists():
                    stacks.append(stack)
                    break

    # Check if react is actually used in package.json
    if "react" in stacks:
        pkg = project_path / "package.json"
        if pkg.exists():
            try:
                pkg_data = json.loads(pkg.read_text())
                deps = {
                    **pkg_data.get("dependencies", {}),
                    **pkg_data.get("devDependencies", {}),
                }
                if "react" not in deps and "react-dom" not in deps:
                    stacks.remove("react")
            except Exception:
                pass

    return stacks


IGNORED_TREE_DIRS = {
    "node_modules",
    ".git",
    "dist",
    "build",
    ".venv",
    "__pycache__",
    ".next",
    "out",
    "coverage",
    ".cache",
}


def _build_file_tree(
    root: Path, max_depth: int = 3, max_entries: int = 500
) -> List[Dict[str, Any]]:
    """Build a file tree listing, ignoring common large/irrelevant dirs."""
    entries: List[Dict[str, Any]] = []

    def _scan(current: Path, depth: int) -> None:
        if depth > max_depth or len(entries) >= max_entries:
            return
        try:
            items = sorted(
                current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except (PermissionError, OSError):
            return

        for item in items:
            if len(entries) >= max_entries:
                break
            if item.name in IGNORED_TREE_DIRS:
                continue
            if item.name.startswith(".") and item.name not in (
                ".env",
                ".env.example",
                ".gitignore",
            ):
                continue
            entry: Dict[str, Any] = {
                "name": item.name,
                "path": str(item.relative_to(root)),
                "type": "directory" if item.is_dir() else "file",
            }
            if item.is_file():
                try:
                    entry["size"] = item.stat().st_size
                except OSError:
                    pass
            entries.append(entry)
            if item.is_dir():
                _scan(item, depth + 1)

    _scan(root, 1)
    return entries


# ---------------------------------------------------------------------------
# Provider API
# ---------------------------------------------------------------------------


@app.get("/api/providers")
async def list_providers():
    """List all known LLM providers with their configuration status."""
    env_data = _get_loaded_env()

    # Also load profile-specific env if active
    try:
        active_profile_path = get_hermes_home() / "active_profile"
        if active_profile_path.exists():
            profile_name = active_profile_path.read_text().strip()
            profile_env_path = get_hermes_home() / "profiles" / profile_name / ".env"
            if profile_env_path.exists():
                profile_env = {}
                for line in profile_env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        profile_env[k.strip()] = v.strip().strip("\"'")
                env_data.update(profile_env)
    except Exception:
        pass

    # Read custom providers from config
    config = load_config()
    user_providers = (
        config.get("providers") if isinstance(config.get("providers"), dict) else {}
    )

    providers: List[Dict[str, Any]] = []

    # Known built-in providers
    for wp in _KNOWN_WEB_PROVIDERS:
        pid = wp["id"]
        name = _PROVIDER_DISPLAY_OVERRIDES.get(pid, wp["name"])
        status = _check_provider_status(pid, wp["env_vars"], env_data)
        models = _get_provider_models_from_catalog(pid)
        default_model = ""
        if models:
            default_model = models[0]["id"]
        providers.append(
            {
                "id": pid,
                "name": name,
                "status": status,
                "models": models,
                "defaultModel": default_model,
            }
        )

    # Custom / user-defined providers
    for ep_name, entry in user_providers.items():
        if not isinstance(entry, dict):
            continue
        display_name = entry.get("name", ep_name)
        key_env = str(entry.get("key_env", "") or "").strip()
        api_key_inline = str(entry.get("api_key", "") or "").strip()
        has_key = False
        if key_env:
            has_key = bool(
                os.getenv(key_env, "").strip() or env_data.get(key_env, "").strip()
            )
        if not has_key and api_key_inline:
            has_key = True
        status = "configured" if has_key else "missing_token"
        default_model = entry.get("default_model", "")
        models = []
        if default_model:
            models = [
                {"id": default_model, "name": default_model, "supportsTools": True}
            ]
        providers.append(
            {
                "id": f"custom:{ep_name}",
                "name": display_name,
                "status": status,
                "models": models,
                "defaultModel": default_model,
            }
        )

    # Resolve current selection from config
    current_provider = ""
    current_model = ""
    model_cfg = config.get("model", "")
    if isinstance(model_cfg, dict):
        current_provider = str(model_cfg.get("provider", "") or "")
        current_model = str(model_cfg.get("default", model_cfg.get("name", "")) or "")
    elif isinstance(model_cfg, str) and model_cfg:
        current_model = model_cfg
        # Try to infer provider from model string (e.g. "anthropic/claude-...")
        if "/" in model_cfg:
            current_provider = model_cfg.split("/")[0]

    return {
        "providers": providers,
        "current": {"provider": current_provider, "model": current_model},
    }


@app.post("/api/providers/select")
async def select_provider_endpoint(body: ProviderSelectRequest, request: Request):
    """Save current provider/model selection."""
    _require_token(request)

    config = load_config()
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {"default": str(model_cfg)}

    # Update config with selection
    model_cfg["provider"] = body.provider_id
    if body.model_id:
        model_cfg["default"] = body.model_id
    config["model"] = model_cfg

    try:
        save_config(config)
        _log.info("Provider selection saved: %s / %s", body.provider_id, body.model_id)
        return {"ok": True, "provider": body.provider_id, "model": body.model_id}
    except Exception as e:
        _log.exception("Failed to save provider selection")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/providers/test")
async def test_provider_endpoint(body: ProviderTestRequest, request: Request):
    """Test connectivity to a provider/model.

    Makes a minimal API call to verify the provider is reachable and the
    credentials are valid. Does NOT expose the token.
    """
    _require_token(request)

    provider_id = body.provider_id
    model_id = body.model_id or ""

    # Strip custom: prefix for resolution
    raw_id = provider_id
    if provider_id.startswith("custom:"):
        raw_id = provider_id[7:]

    try:
        # Try to load .env into os.environ for credential resolution
        env_data = _get_loaded_env()
        for k, v in env_data.items():
            if v and k not in os.environ:
                os.environ[k] = v

        # Resolve runtime credentials
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import has_usable_secret

        runtime = resolve_runtime_provider(requested=raw_id)
        api_key = runtime.get("api_key", "")
        base_url = runtime.get("base_url", "")
        api_mode = runtime.get("api_mode", "chat_completions")

        if not has_usable_secret(api_key):
            return {
                "ok": False,
                "providerId": provider_id,
                "modelId": model_id,
                "message": "Token/API key not configured",
            }

        if not base_url:
            return {
                "ok": False,
                "providerId": provider_id,
                "modelId": model_id,
                "message": "No base URL configured for provider",
            }

        # Make a minimal test request
        import requests as http_requests

        if api_mode == "anthropic_messages":
            test_url = base_url.rstrip("/") + "/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            test_model = model_id or "claude-sonnet-4-20250514"
            payload = {
                "model": test_model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }
        else:
            # OpenAI-compatible
            test_url = base_url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            test_model = model_id or "gpt-4o-mini"
            payload = {
                "model": test_model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }

        try:
            resp = http_requests.post(
                test_url, json=payload, headers=headers, timeout=15
            )
            if resp.status_code < 400:
                return {
                    "ok": True,
                    "providerId": provider_id,
                    "modelId": model_id,
                    "message": "Connection successful",
                }
            elif resp.status_code == 401:
                return {
                    "ok": False,
                    "providerId": provider_id,
                    "modelId": model_id,
                    "message": "Authentication failed (invalid token)",
                }
            elif resp.status_code == 404:
                return {
                    "ok": False,
                    "providerId": provider_id,
                    "modelId": model_id,
                    "message": f"Model '{test_model}' not found or endpoint unreachable",
                }
            else:
                return {
                    "ok": False,
                    "providerId": provider_id,
                    "modelId": model_id,
                    "message": f"Provider returned HTTP {resp.status_code}",
                }
        except http_requests.ConnectionError:
            return {
                "ok": False,
                "providerId": provider_id,
                "modelId": model_id,
                "message": "Connection refused — provider endpoint unreachable",
            }
        except http_requests.Timeout:
            return {
                "ok": False,
                "providerId": provider_id,
                "modelId": model_id,
                "message": "Connection timed out (15s)",
            }

    except Exception as e:
        _log.exception("Provider test failed for %s", provider_id)
        return {
            "ok": False,
            "providerId": provider_id,
            "modelId": model_id,
            "message": str(e)[:200],
        }


@app.post("/api/providers/add")
async def add_provider_endpoint(body: ProviderAddRequest, request: Request):
    """Add a new custom provider with API key.

    Saves provider to config.yaml and optionally saves the API key to .env.
    """
    _require_token(request)

    if not body.api_key or not body.api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required")

    provider_name = body.name or body.id
    slug = body.id.strip().lower().replace(" ", "-")

    config = load_config()
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        providers = {}

    # Store provider config
    providers[slug] = {
        "name": provider_name,
        "api": body.base_url or "",
        "key_env": f"CUSTOM_{slug.upper().replace('-', '_')}_API_KEY",
        "default_model": body.default_model or "",
        "transport": "openai_chat",
    }
    config["providers"] = providers

    try:
        save_config(config)
    except Exception as e:
        _log.exception("Failed to save provider config")
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    # Save API key to .env
    env_key = f"CUSTOM_{slug.upper().replace('-', '_')}_API_KEY"
    try:
        save_env_value(env_key, body.api_key.strip())
    except Exception as e:
        _log.warning("Failed to save API key to .env: %s", e)

    return {
        "ok": True,
        "provider": {
            "id": f"custom:{slug}",
            "name": provider_name,
            "status": "configured",
            "defaultModel": body.default_model or "",
        },
    }


# ---------------------------------------------------------------------------
# Workspace API
# ---------------------------------------------------------------------------


@app.get("/api/workspaces/current")
async def get_current_workspace():
    """Get the currently active workspace."""
    state = _load_workspace_state()
    current = state.get("current")
    if not current or not isinstance(current, dict):
        return {
            "path": "",
            "name": "",
            "exists": False,
            "writable": False,
            "isGitRepo": False,
            "validationSource": "unvalidated",
        }

    ws_path = Path(current.get("path", ""))
    info = _build_workspace_info(ws_path)
    info["lastOpenedAt"] = current.get("lastOpenedAt", "")
    info["validationSource"] = "backend"
    return info


@app.get("/api/workspaces/recent")
async def get_recent_workspaces():
    """Get list of recently opened workspace paths."""
    state = _load_workspace_state()
    recent = state.get("recent", [])
    if not isinstance(recent, list):
        recent = []
    # Filter to only valid existing paths
    valid = []
    for item in recent:
        if isinstance(item, str):
            p = Path(item)
            if p.exists() and p.is_dir():
                valid.append(item)
    return {"paths": valid[:10]}


@app.post("/api/workspaces/validate")
async def validate_workspace_endpoint(body: WorkspacePathRequest):
    """Validate a workspace path."""
    normalized = _normalize_workspace_path(body.path)
    if not normalized:
        return {
            "path": body.path,
            "name": Path(body.path).name or body.path,
            "exists": False,
            "writable": False,
            "isGitRepo": False,
            "valid": False,
            "message": "Path is not allowed, does not exist, or is outside the workspace allowlist",
            "validationSource": "backend",
        }

    ws_path = Path(normalized)
    info = _build_workspace_info(ws_path)
    info["valid"] = info["exists"]
    if not info["exists"]:
        info["message"] = "Directory does not exist"
    info["validationSource"] = "backend"
    return info


@app.post("/api/workspaces/open")
async def open_workspace_endpoint(body: WorkspacePathRequest, request: Request):
    """Open/set a workspace and save it as current + add to recents."""
    _require_token(request)

    normalized = _normalize_workspace_path(body.path)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="Path is not allowed or outside the workspace allowlist",
        )

    ws_path = Path(normalized)
    info = _build_workspace_info(ws_path)
    info["validationSource"] = "backend"
    info["lastOpenedAt"] = _utc_now_iso()

    # Save state
    state = _load_workspace_state()
    state["current"] = {"path": normalized, "lastOpenedAt": info["lastOpenedAt"]}
    recent = state.get("recent", [])
    if not isinstance(recent, list):
        recent = []
    # Move to front, deduplicate
    recent = [normalized] + [
        r for r in recent if r != normalized and isinstance(r, str)
    ]
    state["recent"] = recent[:10]
    _save_workspace_state(state)

    return info


@app.get("/api/workspaces/tree")
async def get_workspace_tree(path: str = "", depth: int = 3):
    """List files/folders of a workspace, ignoring common build artifacts."""
    target_path = path
    if not target_path:
        state = _load_workspace_state()
        current = state.get("current")
        if isinstance(current, dict):
            target_path = current.get("path", "")

    if not target_path:
        raise HTTPException(
            status_code=400,
            detail="No workspace path provided and no current workspace set",
        )

    normalized = _normalize_workspace_path(target_path)
    if not normalized:
        raise HTTPException(status_code=400, detail="Path is not allowed")

    ws_path = Path(normalized)
    if not ws_path.exists() or not ws_path.is_dir():
        raise HTTPException(
            status_code=404, detail="Workspace directory does not exist"
        )

    tree = _build_file_tree(ws_path, max_depth=min(depth, 5))
    return {"path": normalized, "entries": tree, "total": len(tree)}


def _build_workspace_info(ws_path: Path) -> Dict[str, Any]:
    """Build workspace info dict from a path."""
    exists = ws_path.exists() and ws_path.is_dir()
    writable = False
    is_git = False
    branch = ""

    if exists:
        try:
            writable = os.access(ws_path, os.W_OK)
        except OSError:
            pass

        git_dir = ws_path / ".git"
        if git_dir.exists():
            is_git = True
            head_file = git_dir / "HEAD"
            if head_file.exists():
                try:
                    head_content = head_file.read_text().strip()
                    if head_content.startswith("ref: refs/heads/"):
                        branch = head_content[len("ref: refs/heads/") :]
                except Exception:
                    pass

    stack = _detect_stack(ws_path) if exists else []

    return {
        "path": str(ws_path),
        "name": ws_path.name,
        "exists": exists,
        "writable": writable,
        "isGitRepo": is_git,
        "branch": branch,
        "detectedStack": stack,
    }


# ---------------------------------------------------------------------------
# Repository endpoints — read-only, Phase 1
# ---------------------------------------------------------------------------


@app.get("/api/repositories")
async def api_get_repositories(request: Request):
    """Return all configured repositories."""
    try:
        state = _get_repository_state()
        return JSONResponse({"repositories": [state]})
    except Exception as exc:
        _log.error("Failed to read repository state: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read repository state")


@app.get("/api/repositories/current/status")
async def api_get_current_repository_status(request: Request):
    """Return state of the primary configured repository."""
    try:
        state = _get_repository_state()
        return JSONResponse({"repository": state})
    except Exception as exc:
        _log.error("Failed to read repository status: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read repository status")


@app.post("/api/repositories/current/refresh")
async def api_refresh_current_repository(request: Request):
    """Re-read repository state (Phase 1: read-only, no git fetch/pull)."""
    try:
        state = _get_repository_state()
        return JSONResponse({"repository": state})
    except Exception as exc:
        _log.error("Failed to refresh repository: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to refresh repository")


@app.get("/api/code/sessions/{code_session_id}/commands")
async def list_commands(code_session_id: str):
    from hermes_cli.code.command_runner import CommandRunnerService

    try:
        svc = CommandRunnerService(realtime_hub=_REALTIME_HUB)
        commands = svc.list_commands(code_session_id)
        return {"commands": commands, "total": len(commands)}
    except Exception as exc:
        _log.error(
            "list_commands failed for %s: %s", code_session_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/sessions/{code_session_id}/commands/run")
async def run_command(code_session_id: str, body: _RunCommandBody):
    from hermes_cli.code.command_runner import CommandRunnerService

    try:
        svc = CommandRunnerService(realtime_hub=_REALTIME_HUB)

        # 1. Create the command record
        cmd = svc.create_command(
            code_session_id=code_session_id,
            command=body.command,
            cwd=body.cwd,
            timeout_seconds=body.timeout_seconds,
        )

        # Emit command.started
        try:
            await _REALTIME_HUB.broadcast(
                "command.started", {"payload": {"command": cmd}}
            )
        except Exception:
            pass

        # 2. If it's blocked or needs approval, return it immediately without running
        if cmd["status"] in ("blocked", "needs_approval"):
            return {"command": cmd}

        # 3. Run the command sync
        updated_cmd = svc.run_command_sync(cmd["id"])

        # 4. Emit completed/failed/timeout
        try:
            await _REALTIME_HUB.broadcast(
                f"command.{updated_cmd['status']}",
                {"payload": {"command": updated_cmd}},
            )
            # Also emit command.output
            if updated_cmd.get("stdout") or updated_cmd.get("stderr"):
                await _REALTIME_HUB.broadcast(
                    "command.output",
                    {
                        "payload": {
                            "command_id": updated_cmd["id"],
                            "stdout": updated_cmd.get("stdout", ""),
                            "stderr": updated_cmd.get("stderr", ""),
                        }
                    },
                )
        except Exception:
            pass

        return {"command": updated_cmd}

    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as exc:
        _log.error("run_command failed for %s: %s", code_session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/commands/{command_id}")
async def get_command(command_id: str):
    from hermes_cli.code.command_runner import CommandRunnerService

    try:
        svc = CommandRunnerService(realtime_hub=_REALTIME_HUB)
        cmd = svc.get_command(command_id)
        if not cmd:
            raise HTTPException(status_code=404, detail="Command not found")
        return {"command": cmd}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("get_command failed for %s: %s", command_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/commands/{command_id}/cancel")
async def cancel_command(command_id: str):
    from hermes_cli.code.command_runner import CommandRunnerService

    try:
        svc = CommandRunnerService(realtime_hub=_REALTIME_HUB)
        cmd = svc.get_command(command_id)
        if not cmd:
            raise HTTPException(status_code=404, detail="Command not found")

        if cmd["status"] in ("completed", "failed", "timeout", "cancelled"):
            return {"command": cmd, "message": "Command is not running"}

        updated_cmd = svc.cancel_command(command_id)

        try:
            await _REALTIME_HUB.broadcast(
                "command.cancelled", {"payload": {"command": updated_cmd}}
            )
        except Exception:
            pass

        return {"command": updated_cmd}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("cancel_command failed for %s: %s", command_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Git endpoints
# ---------------------------------------------------------------------------


class _PrepareBranchBody(BaseModel):
    branch_name: str


class _CreateBranchBody(BaseModel):
    branch_name: str
    code_session_id: Optional[str] = None


class _PrepareCommitBody(BaseModel):
    message: str
    code_session_id: Optional[str] = None


class _CreateSnapshotBody(BaseModel):
    code_session_id: Optional[str] = None


@app.get("/api/code/workspaces/{workspace_id}/git/status")
async def git_status(workspace_id: str, code_session_id: Optional[str] = None):
    from hermes_cli.code.git_service import GitService

    try:
        svc = GitService()
        status = svc.get_status(workspace_id, code_session_id=code_session_id)
        try:
            await _REALTIME_HUB.broadcast(
                "git.status_changed",
                {"payload": {"workspace_id": workspace_id, "status": status}},
            )
        except Exception:
            pass
        return {"status": status}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("git_status failed for %s: %s", workspace_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/workspaces/{workspace_id}/git/diff")
async def git_diff(workspace_id: str, path: Optional[str] = None):
    from hermes_cli.code.git_service import GitService

    try:
        svc = GitService()
        diff = svc.get_diff(workspace_id, path=path)
        return {"diff": diff}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("git_diff failed for %s: %s", workspace_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/workspaces/{workspace_id}/git/branch")
async def git_branch(workspace_id: str):
    from hermes_cli.code.git_service import GitService

    try:
        svc = GitService()
        branch = svc.get_branch(workspace_id)
        return branch
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("git_branch failed for %s: %s", workspace_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/workspaces/{workspace_id}/git/remote")
async def git_remote(workspace_id: str):
    from hermes_cli.code.git_service import GitService

    try:
        svc = GitService()
        remote = svc.get_remote(workspace_id)
        return remote
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("git_remote failed for %s: %s", workspace_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/workspaces/{workspace_id}/git/snapshot")
async def git_snapshot(
    workspace_id: str, body: _CreateSnapshotBody = _CreateSnapshotBody()
):
    from hermes_cli.code.git_service import GitService

    try:
        svc = GitService()
        snapshot = svc.create_snapshot(
            workspace_id, code_session_id=body.code_session_id
        )
        try:
            await _REALTIME_HUB.broadcast(
                "git.snapshot.created",
                {"payload": {"workspace_id": workspace_id, "snapshot": snapshot}},
            )
        except Exception:
            pass
        return {"snapshot": snapshot}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("git_snapshot failed for %s: %s", workspace_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/workspaces/{workspace_id}/git/branch/prepare")
async def git_prepare_branch(workspace_id: str, body: _PrepareBranchBody):
    from hermes_cli.code.git_service import GitService

    try:
        svc = GitService()
        result = svc.prepare_branch(workspace_id, body.branch_name)
        try:
            await _REALTIME_HUB.broadcast(
                "git.branch.prepared",
                {"payload": {"workspace_id": workspace_id, "result": result}},
            )
        except Exception:
            pass
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error(
            "git_prepare_branch failed for %s: %s", workspace_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/workspaces/{workspace_id}/git/branch")
async def git_create_branch(workspace_id: str, body: _CreateBranchBody):
    from hermes_cli.code.git_service import GitService

    try:
        svc = GitService()
        result = svc.create_branch(
            workspace_id, body.branch_name, code_session_id=body.code_session_id
        )
        try:
            if result.get("result", {}).get("executed"):
                await _REALTIME_HUB.broadcast(
                    "git.branch.created",
                    {"payload": {"workspace_id": workspace_id, "result": result}},
                )
        except Exception:
            pass
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error(
            "git_create_branch failed for %s: %s", workspace_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/workspaces/{workspace_id}/git/commit/prepare")
async def git_prepare_commit(workspace_id: str, body: _PrepareCommitBody):
    from hermes_cli.code.git_service import GitService

    try:
        svc = GitService()
        result = svc.prepare_commit(
            workspace_id, body.message, code_session_id=body.code_session_id
        )
        try:
            await _REALTIME_HUB.broadcast(
                "git.commit.prepared",
                {"payload": {"workspace_id": workspace_id, "result": result}},
            )
        except Exception:
            pass
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error(
            "git_prepare_commit failed for %s: %s", workspace_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Provider Router endpoints
# ---------------------------------------------------------------------------


class _SelectModelBody(BaseModel):
    task_type: str


class _UpdateModelBody(BaseModel):
    provider: str
    model: str


class _CreatePresetBody(BaseModel):
    name: str
    provider: str
    model: str
    metadata: Optional[dict] = None


class _TrackCostBody(BaseModel):
    provider: str
    model: str
    task_type: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    metadata: Optional[dict] = None


@app.get("/api/code/sessions/{code_session_id}/model")
async def get_session_model(code_session_id: str):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        result = router.get_session_model(code_session_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("get_session_model failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/sessions/{code_session_id}/model/select")
async def select_session_model(code_session_id: str, body: _SelectModelBody):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        result = router.select_model(code_session_id, body.task_type)
        try:
            await _REALTIME_HUB.broadcast(
                "provider.model_selected",
                {"payload": {"code_session_id": code_session_id, "result": result}},
            )
        except Exception:
            pass
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("select_session_model failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.put("/api/code/sessions/{code_session_id}/model")
async def update_session_model(code_session_id: str, body: _UpdateModelBody):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        result = router.update_session_model(code_session_id, body.provider, body.model)
        try:
            await _REALTIME_HUB.broadcast(
                "provider.model_updated",
                {"payload": {"code_session_id": code_session_id, "result": result}},
            )
        except Exception:
            pass
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("update_session_model failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/sessions/{code_session_id}/presets")
async def list_session_presets(code_session_id: str):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        result = router.get_presets_summary(code_session_id)
        return {"presets": result}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("list_session_presets failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/sessions/{code_session_id}/presets")
async def create_session_preset(code_session_id: str, body: _CreatePresetBody):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        preset = router.create_preset(
            code_session_id,
            body.name,
            body.provider,
            body.model,
            metadata=body.metadata,
        )
        return {"preset": preset}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("create_session_preset failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/code/sessions/{code_session_id}/presets/{preset_id}")
async def delete_session_preset(code_session_id: str, preset_id: str):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        ok = router.delete_preset(code_session_id, preset_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("delete_session_preset failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/sessions/{code_session_id}/cost")
async def track_session_cost(code_session_id: str, body: _TrackCostBody):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        entry = router.track_cost(
            code_session_id,
            provider=body.provider,
            model=body.model,
            task_type=body.task_type,
            input_tokens=body.input_tokens,
            output_tokens=body.output_tokens,
            cache_read_tokens=body.cache_read_tokens,
            cache_write_tokens=body.cache_write_tokens,
            cost_usd=body.cost_usd,
            metadata=body.metadata,
        )
        return {"entry": entry}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("track_session_cost failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/sessions/{code_session_id}/cost")
async def get_session_cost(code_session_id: str):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        summary = router.get_session_cost_summary(code_session_id)
        return {"summary": summary}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("get_session_cost failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/sessions/{code_session_id}/cost/entries")
async def list_session_cost_entries(
    code_session_id: str,
    limit: int = 100,
    offset: int = 0,
):
    from hermes_cli.code.provider_router import ProviderRouter

    try:
        router = ProviderRouter()
        entries = router.list_cost_entries(code_session_id, limit=limit, offset=offset)
        return {"entries": entries, "total": len(entries)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("list_session_cost_entries failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Code Intelligence / LSP endpoints
# ---------------------------------------------------------------------------


@app.get("/api/code/workspaces/{workspace_id}/diagnostics")
async def get_workspace_diagnostics(
    workspace_id: str, code_session_id: Optional[str] = None
):
    from hermes_cli.code.lsp_service import CodeIntelligenceService

    try:
        svc = CodeIntelligenceService(realtime_hub=_REALTIME_HUB)
        result = svc.get_workspace_diagnostics(
            workspace_id, code_session_id=code_session_id
        )
        try:
            await _REALTIME_HUB.broadcast(
                "diagnostics.completed",
                {
                    "payload": {
                        "workspace_id": workspace_id,
                        "summary": result.get("summary", {}),
                    }
                },
            )
        except Exception:
            pass
        return {"result": result}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "get_workspace_diagnostics failed for %s: %s",
            workspace_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/workspaces/{workspace_id}/diagnostics/file")
async def get_file_diagnostics(
    workspace_id: str,
    path: str,
    code_session_id: Optional[str] = None,
):
    from hermes_cli.code.lsp_service import CodeIntelligenceService

    try:
        svc = CodeIntelligenceService(realtime_hub=_REALTIME_HUB)
        result = svc.get_file_diagnostics(
            workspace_id, file_path=path, code_session_id=code_session_id
        )
        return {"result": result}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "get_file_diagnostics failed for %s: %s",
            workspace_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/workspaces/{workspace_id}/languages")
async def get_supported_languages(workspace_id: str):
    from hermes_cli.code.lsp_service import CodeIntelligenceService

    try:
        svc = CodeIntelligenceService()
        languages = svc.get_supported_languages(workspace_id)
        return {"languages": languages}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "get_supported_languages failed for %s: %s",
            workspace_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/workspaces/{workspace_id}/lsp/restart")
async def restart_language_services(workspace_id: str):
    from hermes_cli.code.lsp_service import CodeIntelligenceService

    try:
        svc = CodeIntelligenceService()
        result = svc.restart_language_services(workspace_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error(
            "restart_language_services failed for %s: %s",
            workspace_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Multi-Agent Coding Flow endpoints  (Phase 8)
# ---------------------------------------------------------------------------


class _CreateAgentFlowBody(BaseModel):
    workspace_id: str
    code_session_id: str
    task_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    preset: Optional[str] = None


@app.get("/api/code/agent-flows")
async def list_agent_flows(
    code_session_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
):
    from hermes_cli.code.multi_agent_coding import MultiAgentCodingService

    try:
        svc = MultiAgentCodingService(realtime_hub=_REALTIME_HUB)
        flows = svc.list_flows(
            code_session_id=code_session_id,
            workspace_id=workspace_id,
            status=status,
            limit=limit,
        )
        return {"flows": flows}
    except Exception as exc:
        _log.error("list_agent_flows failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/agent-flows")
async def create_agent_flow(body: _CreateAgentFlowBody):
    from hermes_cli.code.multi_agent_coding import MultiAgentCodingService

    try:
        svc = MultiAgentCodingService(realtime_hub=_REALTIME_HUB)
        flow = svc.create_flow(
            code_session_id=body.code_session_id,
            workspace_id=body.workspace_id,
            task_id=body.task_id,
            title=body.title,
            description=body.description,
            provider=body.provider,
            model=body.model,
            preset=body.preset,
        )
        await _REALTIME_HUB.broadcast(
            "code_flow.created",
            {"payload": {"flow": flow}},
        )
        return {"flow": flow}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("create_agent_flow failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/agent-flows/{flow_id}")
async def get_agent_flow(flow_id: str):
    from hermes_cli.code.multi_agent_coding import MultiAgentCodingService

    try:
        svc = MultiAgentCodingService(realtime_hub=_REALTIME_HUB)
        flow = svc.get_flow(flow_id)
        if not flow:
            raise HTTPException(status_code=404, detail=f"Flow not found: {flow_id}")
        return {"flow": flow}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("get_agent_flow failed for %s: %s", flow_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/agent-flows/{flow_id}/run")
async def run_agent_flow(flow_id: str, background_tasks=None):
    from hermes_cli.code.multi_agent_coding import MultiAgentCodingService

    try:
        svc = MultiAgentCodingService(realtime_hub=_REALTIME_HUB)
        flow = svc.run_flow(flow_id)
        await _REALTIME_HUB.broadcast(
            "code_flow.updated",
            {"payload": {"flow": flow}},
        )
        return {"flow": flow}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("run_agent_flow failed for %s: %s", flow_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/agent-flows/{flow_id}/cancel")
async def cancel_agent_flow(flow_id: str, reason: Optional[str] = None):
    from hermes_cli.code.multi_agent_coding import MultiAgentCodingService

    try:
        svc = MultiAgentCodingService(realtime_hub=_REALTIME_HUB)
        flow = svc.cancel_flow(flow_id, reason=reason)
        await _REALTIME_HUB.broadcast(
            "code_flow.cancelled",
            {"payload": {"flow": flow}},
        )
        return {"flow": flow}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("cancel_agent_flow failed for %s: %s", flow_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/agent-flows/{flow_id}/resume")
async def resume_agent_flow(flow_id: str):
    from hermes_cli.code.multi_agent_coding import MultiAgentCodingService

    try:
        svc = MultiAgentCodingService(realtime_hub=_REALTIME_HUB)
        flow = svc.resume_flow(flow_id)
        await _REALTIME_HUB.broadcast(
            "code_flow.updated",
            {"payload": {"flow": flow}},
        )
        return {"flow": flow}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("resume_agent_flow failed for %s: %s", flow_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/sessions/{code_session_id}/agent-flows")
async def list_session_agent_flows(code_session_id: str, limit: int = 50):
    from hermes_cli.code.multi_agent_coding import MultiAgentCodingService

    try:
        svc = MultiAgentCodingService(realtime_hub=_REALTIME_HUB)
        flows = svc.list_flows(code_session_id=code_session_id, limit=limit)
        return {"flows": flows}
    except Exception as exc:
        _log.error(
            "list_session_agent_flows failed for %s: %s", code_session_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Coding Skills endpoints  (Phase 9)
# ---------------------------------------------------------------------------


class _CreateSkillRunBody(BaseModel):
    skill_name: str
    workspace_id: str
    code_session_id: Optional[str] = None
    task_id: Optional[str] = None
    input: Optional[dict] = None


@app.get("/api/code/skills")
async def list_coding_skills():
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        return {"skills": svc.list_skills()}
    except Exception as exc:
        _log.error("list_coding_skills failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/skill-runs")
async def list_skill_runs(
    workspace_id: Optional[str] = None,
    code_session_id: Optional[str] = None,
    skill_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
):
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        runs = svc.list_runs(
            workspace_id=workspace_id,
            code_session_id=code_session_id,
            skill_name=skill_name,
            status=status,
            limit=limit,
        )
        return {"runs": runs}
    except Exception as exc:
        _log.error("list_skill_runs failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/skill-runs")
async def create_skill_run(body: _CreateSkillRunBody):
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        run = svc.create_run(
            skill_name=body.skill_name,
            workspace_id=body.workspace_id,
            code_session_id=body.code_session_id,
            task_id=body.task_id,
            input_data=body.input,
        )
        await _REALTIME_HUB.broadcast("skill.started", {"payload": {"run": run}})
        return {"run": run}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        _log.error("create_skill_run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/skill-runs/{run_id}")
async def get_skill_run(run_id: str):
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        run = svc.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Skill run not found: {run_id}")
        return {"run": run}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("get_skill_run failed for %s: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/skill-runs/{run_id}/run")
async def run_skill(run_id: str):
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        run = svc.run_skill(run_id)
        await _REALTIME_HUB.broadcast("skill.updated", {"payload": {"run": run}})
        return {"run": run}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("run_skill failed for %s: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/skill-runs/{run_id}/cancel")
async def cancel_skill_run(run_id: str, reason: Optional[str] = None):
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        run = svc.cancel_run(run_id, reason=reason)
        await _REALTIME_HUB.broadcast("skill.cancelled", {"payload": {"run": run}})
        return {"run": run}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("cancel_skill_run failed for %s: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/skill-runs/{run_id}/resume")
async def resume_skill_run(run_id: str):
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        run = svc.resume_run(run_id)
        await _REALTIME_HUB.broadcast("skill.updated", {"payload": {"run": run}})
        return {"run": run}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("resume_skill_run failed for %s: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/sessions/{code_session_id}/skill-runs")
async def list_session_skill_runs(code_session_id: str, limit: int = 50):
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        runs = svc.list_runs(code_session_id=code_session_id, limit=limit)
        return {"runs": runs}
    except Exception as exc:
        _log.error("list_session_skill_runs failed for %s: %s", code_session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/code/skills/{skill_name}/run")
async def run_skill_shortcut(skill_name: str, body: _CreateSkillRunBody):
    """Shortcut: create + run a skill in one request."""
    from hermes_cli.code.coding_skills import CodingSkillsService

    try:
        svc = CodingSkillsService(realtime_hub=_REALTIME_HUB)
        run = svc.create_run(
            skill_name=skill_name,
            workspace_id=body.workspace_id,
            code_session_id=body.code_session_id,
            task_id=body.task_id,
            input_data=body.input,
        )
        run = svc.run_skill(run["id"])
        await _REALTIME_HUB.broadcast("skill.updated", {"payload": {"run": run}})
        return {"run": run}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("run_skill_shortcut failed for %s: %s", skill_name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def mount_spa(application: FastAPI):
    """Mount the built SPA. Falls back to index.html for client-side routing.

    The session token is injected into index.html via a ``<script>`` tag so
    the SPA can authenticate against protected API endpoints without a
    separate (unauthenticated) token-dispensing endpoint.
    """
    if not WEB_DIST.exists():

        @application.get("/{full_path:path}")
        async def no_frontend(full_path: str):
            return JSONResponse(
                {"error": "Frontend not built. Run: cd web && npm run build"},
                status_code=404,
            )

        return

    _index_path = WEB_DIST / "index.html"

    def _serve_index():
        """Return index.html with the session token injected."""
        html = _index_path.read_text()
        token_script = (
            f'<script>window.__HERMES_SESSION_TOKEN__="{_SESSION_TOKEN}";</script>'
        )
        html = html.replace("</head>", f"{token_script}</head>", 1)
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    application.mount(
        "/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets"
    )

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
        return _serve_index()


# ---------------------------------------------------------------------------
# P0: ArtifactLedger endpoints
# ---------------------------------------------------------------------------

class _CreateLedgerArtifactBody(BaseModel):
    category: str
    content: str
    title: Optional[str] = None
    format: str = "markdown"
    workspace_id: Optional[str] = None
    code_session_id: Optional[str] = None
    flow_id: Optional[str] = None
    command_id: Optional[str] = None
    orchestrated_run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/code/ledger/artifacts")
async def create_ledger_artifact(body: _CreateLedgerArtifactBody):
    from hermes_cli.code.artifact_ledger import ArtifactLedger
    try:
        ledger = ArtifactLedger(realtime_hub=_REALTIME_HUB)
        artifact = ledger.create_artifact(
            category=body.category,
            content=body.content,
            title=body.title,
            format=body.format,
            workspace_id=body.workspace_id,
            code_session_id=body.code_session_id,
            flow_id=body.flow_id,
            command_id=body.command_id,
            orchestrated_run_id=body.orchestrated_run_id,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("create_ledger_artifact failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        await _broadcast_code_event(
            "code.artifact.created",
            {"artifact": artifact},
            workspace_id=body.workspace_id,
            code_session_id=body.code_session_id,
        )
    except Exception:
        pass

    return {"artifact": artifact}


@app.get("/api/code/ledger/artifacts")
async def list_ledger_artifacts(
    code_session_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    orchestrated_run_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    from hermes_cli.code.artifact_ledger import ArtifactLedger
    try:
        ledger = ArtifactLedger()
        artifacts = ledger.list_artifacts(
            code_session_id=code_session_id,
            workspace_id=workspace_id,
            orchestrated_run_id=orchestrated_run_id,
            category=category,
            limit=limit,
            offset=offset,
        )
        return {"artifacts": artifacts, "total": len(artifacts)}
    except Exception as exc:
        _log.error("list_ledger_artifacts failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/ledger/artifacts/{artifact_id}")
async def get_ledger_artifact(artifact_id: str):
    from hermes_cli.code.artifact_ledger import ArtifactLedger
    try:
        ledger = ArtifactLedger()
        artifact = ledger.get_artifact(artifact_id)
    except Exception as exc:
        _log.error("get_ledger_artifact failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return {"artifact": artifact}


@app.get("/api/code/ledger/categories")
async def list_ledger_categories():
    from hermes_cli.code.artifact_ledger import ARTIFACT_CATEGORIES
    return {"categories": list(ARTIFACT_CATEGORIES)}


# ---------------------------------------------------------------------------
# P0: AgentOrchestrator endpoints
# ---------------------------------------------------------------------------

class _CreateOrchestratedRunBody(BaseModel):
    workspace_id: Optional[str] = None
    code_session_id: Optional[str] = None
    title: Optional[str] = None
    task_description: Optional[str] = None
    branch: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class _TransitionOrchestratedRunBody(BaseModel):
    to_state: str
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


@app.post("/api/code/orchestrator/runs")
async def create_orchestrated_run(body: _CreateOrchestratedRunBody):
    from hermes_cli.code.agent_orchestrator import AgentOrchestrator
    try:
        orch = AgentOrchestrator(realtime_hub=_REALTIME_HUB)
        run = orch.create_run(
            workspace_id=body.workspace_id,
            code_session_id=body.code_session_id,
            title=body.title,
            task_description=body.task_description,
            branch=body.branch,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _log.error("create_orchestrated_run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        await _broadcast_code_event(
            "orchestrator.run.created",
            {"run": run},
            workspace_id=body.workspace_id,
            code_session_id=body.code_session_id,
        )
    except Exception:
        pass

    return {"run": run}


@app.get("/api/code/orchestrator/runs")
async def list_orchestrated_runs(
    workspace_id: Optional[str] = None,
    code_session_id: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
):
    from hermes_cli.code.agent_orchestrator import AgentOrchestrator
    try:
        orch = AgentOrchestrator()
        runs = orch.list_runs(
            workspace_id=workspace_id,
            code_session_id=code_session_id,
            state=state,
            limit=limit,
        )
        return {"runs": runs, "total": len(runs)}
    except Exception as exc:
        _log.error("list_orchestrated_runs failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/orchestrator/runs/{run_id}")
async def get_orchestrated_run(run_id: str):
    from hermes_cli.code.agent_orchestrator import AgentOrchestrator
    try:
        orch = AgentOrchestrator()
        run = orch.get_run(run_id)
    except Exception as exc:
        _log.error("get_orchestrated_run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    if not run:
        raise HTTPException(status_code=404, detail="Orchestrated run not found")
    return {"run": run}


@app.post("/api/code/orchestrator/runs/{run_id}/transition")
async def transition_orchestrated_run(run_id: str, body: _TransitionOrchestratedRunBody):
    from hermes_cli.code.agent_orchestrator import AgentOrchestrator
    try:
        orch = AgentOrchestrator(realtime_hub=_REALTIME_HUB)
        run = orch.transition(
            run_id=run_id,
            to_state=body.to_state,
            message=body.message,
            payload=body.payload,
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as exc:
        _log.error("transition_orchestrated_run failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        await _broadcast_code_event(
            "orchestrator.run.transitioned",
            {"run": run, "to_state": body.to_state},
            workspace_id=run.get("workspace_id"),
            code_session_id=run.get("code_session_id"),
        )
    except Exception:
        pass

    return {"run": run}


@app.get("/api/code/orchestrator/runs/{run_id}/events")
async def list_orchestrated_run_events(run_id: str):
    from hermes_cli.code.agent_orchestrator import AgentOrchestrator
    try:
        orch = AgentOrchestrator()
        events = orch.list_events(run_id)
        return {"events": events, "total": len(events)}
    except Exception as exc:
        _log.error("list_orchestrated_run_events failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/orchestrator/states")
async def list_orchestrator_states():
    from hermes_cli.code.agent_orchestrator import AgentOrchestrator, TRANSITIONS
    states = AgentOrchestrator.valid_states()
    transitions = {s: sorted(TRANSITIONS.get(s, frozenset())) for s in states}
    return {"states": states, "transitions": transitions}


# ---------------------------------------------------------------------------
# P0: ExecutionPolicyEngine endpoints
# ---------------------------------------------------------------------------

class _AssessCommandBody(BaseModel):
    command: str


@app.post("/api/code/policy/assess")
async def assess_command_policy(body: _AssessCommandBody):
    from hermes_cli.code.execution_policy import policy_engine
    assessment = policy_engine.assess(body.command)
    return {"assessment": assessment}


@app.get("/api/code/policy/risk-classes")
async def list_risk_classes():
    from hermes_cli.code.execution_policy import RiskClass
    classes = [
        v for k, v in vars(RiskClass).items()
        if not k.startswith("_") and isinstance(v, str)
    ]
    return {"risk_classes": sorted(classes)}


# ---------------------------------------------------------------------------
# P0: WorktreeService endpoints
# ---------------------------------------------------------------------------

@app.get("/api/code/workspaces/{workspace_id}/git-capabilities")
async def get_git_capabilities(workspace_id: str):
    from hermes_cli.code.worktree_service import WorktreeService
    try:
        svc = WorktreeService()
        caps = svc.detect_capabilities(workspace_id)
        return {"capabilities": caps}
    except Exception as exc:
        _log.error("get_git_capabilities failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/code/workspaces/{workspace_id}/worktrees")
async def list_worktrees(workspace_id: str):
    from hermes_cli.code.worktree_service import WorktreeService
    try:
        svc = WorktreeService()
        worktrees = svc.list_worktrees(workspace_id)
        return {"worktrees": worktrees}
    except Exception as exc:
        _log.error("list_worktrees failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# P0: SkillDiscovery endpoints
# ---------------------------------------------------------------------------

@app.get("/api/code/skills/catalog")
async def list_skill_catalog(workspace_id: Optional[str] = None):
    from hermes_cli.code.skill_discovery import SkillDiscoveryService
    from pathlib import Path as _Path
    try:
        svc = SkillDiscoveryService()
        workspace_path = None
        if workspace_id:
            from hermes_state import WorkspaceDB
            wdb = WorkspaceDB()
            try:
                ws = wdb.get_workspace(workspace_id)
                if ws:
                    workspace_path = _Path(ws["path"])
            finally:
                wdb.close()
        skills = svc.list_skills(workspace_path=workspace_path)
        return {"skills": skills, "total": len(skills)}
    except Exception as exc:
        _log.error("list_skill_catalog failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# P0: RepoKnowledge / AGENTS.md endpoints
# ---------------------------------------------------------------------------

@app.get("/api/code/workspaces/{workspace_id}/repo-knowledge")
async def get_repo_knowledge(workspace_id: str):
    from hermes_cli.code.repo_knowledge import RepoKnowledgeService
    from pathlib import Path as _Path
    try:
        from hermes_state import WorkspaceDB
        wdb = WorkspaceDB()
        try:
            ws = wdb.get_workspace(workspace_id)
        finally:
            wdb.close()

        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")

        svc = RepoKnowledgeService()
        workspace_path = _Path(ws["path"])
        guidance = svc.detect(workspace_path)
        return {"guidance": guidance}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("get_repo_knowledge failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


mount_spa(app)


def start_server(
    host: str = "127.0.0.1",
    port: int = 9119,
    open_browser: bool = True,
    allow_public: bool = False,
):
    """Start the web UI server."""
    import uvicorn

    _LOCALHOST = ("127.0.0.1", "localhost", "::1")
    if host not in _LOCALHOST and not allow_public:
        raise SystemExit(
            f"Refusing to bind to {host} — the dashboard exposes API keys "
            f"and config without robust authentication.\n"
            f"Use --insecure to override (NOT recommended on untrusted networks)."
        )
    if host not in _LOCALHOST:
        _log.warning(
            "Binding to %s with --insecure — the dashboard has no robust "
            "authentication. Only use on trusted networks.",
            host,
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
