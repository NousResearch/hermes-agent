"""
Hermes Agent — Web UI server.

Provides a FastAPI backend serving the Vite/React frontend and REST API
endpoints for managing configuration, environment variables, and sessions.

Usage:
    python -m hermes_cli.main web          # Start on http://127.0.0.1:9119
    python -m hermes_cli.main web --port 8080
"""

from contextlib import asynccontextmanager, contextmanager

import asyncio
import atexit
import base64
import binascii
import concurrent.futures
import functools
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import inspect
import importlib.util
import json
import logging
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import zipfile

from hermes_cli._subprocess_compat import windows_detach_flags, windows_hide_flags
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli import __version__, __release_date__
from hermes_cli.config import (
    cfg_get,
    DEFAULT_CONFIG,
    OPTIONAL_ENV_VARS,
    clear_model_endpoint_credentials,
    get_config_path,
    get_env_path,
    get_hermes_home,
    load_config,
    load_env,
    read_raw_config,
    save_config,
    save_env_value,
    remove_env_value,
    check_config_version,
    detect_install_method,
    format_docker_update_message,
    recommended_update_command_for_method,
    redact_key,
    write_platform_config_field,
    _deep_merge,
)
from gateway.status import (
    derive_gateway_busy,
    derive_gateway_drainable,
    get_running_pid_cached,
    get_running_pid,
    get_runtime_status_running_pid,
    parse_active_agents,
    read_runtime_status,
)
from utils import env_var_enabled

try:
    from fastapi import (
        FastAPI, File, Form, HTTPException, Request, UploadFile,
        WebSocket, WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    from starlette.concurrency import run_in_threadpool
except ImportError:
    # First try lazy-installing the dashboard extras. Only the user actually
    # running `hermes dashboard` needs fastapi+uvicorn; lazy install keeps
    # them out of every other install path. After install, re-import.
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.dashboard", prompt=False)
        from fastapi import (
            FastAPI, File, Form, HTTPException, Request, UploadFile,
            WebSocket, WebSocketDisconnect,
        )
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
        from starlette.concurrency import run_in_threadpool
    except Exception:
        raise SystemExit(
            "Web UI requires fastapi and uvicorn.\n"
            f"Install with: {sys.executable} -m pip install 'fastapi' 'uvicorn[standard]'"
        )

WEB_DIST = Path(os.environ["HERMES_WEB_DIST"]) if "HERMES_WEB_DIST" in os.environ else Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-channel subscriber registry used by /api/pub (PTY-side gateway → dashboard)
# and /api/events (dashboard → browser sidebar).  Keyed by an opaque channel id
# the chat tab generates on mount; entries auto-evict when the last subscriber
# drops AND the publisher has disconnected.
#
# State lives on app.state (not module-level globals) so that asyncio.Lock is
# created on the running event loop during lifespan startup.  A module-level
# asyncio.Lock() binds to whatever loop was active at import time, which breaks
# when the same module is used across TestClient instances or uvicorn reloads.
# ---------------------------------------------------------------------------

def _start_desktop_cron_ticker(stop_event: "threading.Event", interval: int = 60) -> None:
    """Tick the cron scheduler from inside the desktop dashboard backend.

    The scheduler tick loop normally lives in ``hermes gateway run`` — but the
    desktop app spawns a ``hermes dashboard`` backend, not a gateway, so a cron
    a user creates in the app would never fire. We run the resolved cron
    scheduler provider here (no live adapters; delivery falls back to the
    per-platform send path).

    Cross-process safe: the built-in provider's ``cron.scheduler.tick`` takes
    the ``cron/.tick.lock`` file lock, so this never double-fires alongside a
    real gateway on the same HERMES_HOME — whichever process grabs the lock
    first wins the tick.
    """
    from cron.scheduler_provider import resolve_cron_scheduler

    provider = resolve_cron_scheduler()
    _log.info("Desktop cron scheduler started (provider=%s, interval=%ds)", provider.name, interval)
    provider.start(stop_event, interval=interval)


def _warm_gateway_module() -> None:
    try:
        import hermes_cli.gateway  # noqa: F401
    except Exception:
        pass


def _resolve_restart_drain_timeout() -> float:
    try:
        from hermes_cli.gateway import _get_restart_drain_timeout
        return _get_restart_drain_timeout()
    except ImportError:
        from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT


@asynccontextmanager
async def _lifespan(app: "FastAPI"):
    app.state.event_channels = {}  # dict[str, set]
    app.state.event_lock = asyncio.Lock()
    app.state.pty_active_session_files = {}  # dict[str, Path]
    # Serializes chat-argv resolution so concurrent /api/pty connections
    # don't trigger overlapping ``npm install`` / ``npm run build`` work.
    # On app.state (not a module global) so the Lock binds to the running
    # event loop during lifespan startup — see _get_event_state's docstring.
    app.state.chat_argv_lock = asyncio.Lock()

    # Fire hermes_cli.gateway import into a background thread so the event
    # loop is not blocked and HERMES_DASHBOARD_READY fires without delay.
    # On a cold Windows install the module chain triggers .pyc compilation
    # and Defender real-time scans that can stall the event loop for 15-30s.
    # Running in an executor means the cost is paid in a worker thread while
    # the server socket is already open and accepting probes.
    asyncio.get_event_loop().run_in_executor(None, _warm_gateway_module)

    # Desktop-spawned backends (HERMES_DESKTOP=1) fire cron jobs themselves,
    # since the app has no gateway running the scheduler. Server `hermes
    # dashboard` is unaffected — it relies on its own gateway.
    cron_stop: "threading.Event | None" = None
    cron_thread: "threading.Thread | None" = None
    if os.getenv("HERMES_DESKTOP") == "1":
        cron_stop = threading.Event()
        cron_thread = threading.Thread(
            target=_start_desktop_cron_ticker,
            args=(cron_stop,),
            daemon=True,
            name="desktop-cron-ticker",
        )
        cron_thread.start()

    # Reap idle/dead keep-alive PTY sessions in the background (30-min TTL).
    pty_reaper_task = asyncio.create_task(run_reaper(PTY_REGISTRY))

    try:
        yield
    finally:
        pty_reaper_task.cancel()
        await PTY_REGISTRY.close_all()
        if cron_stop is not None:
            cron_stop.set()


def _get_event_state(app: "FastAPI"):
    """Return (event_channels, event_lock) from app.state.

    Lazily initialises the state if the lifespan hasn't run (e.g. when
    TestClient is constructed without a ``with`` block).  The lifespan
    path is preferred because it guarantees the Lock is created on the
    correct event loop, but the lazy path lets existing non-``with``
    TestClient usages keep working.
    """
    try:
        return app.state.event_channels, app.state.event_lock
    except AttributeError:
        app.state.event_channels = {}
        app.state.event_lock = asyncio.Lock()
        return app.state.event_channels, app.state.event_lock


def _get_chat_argv_lock(app: "FastAPI") -> asyncio.Lock:
    """Return the chat-argv resolution lock from app.state.

    Mirrors :func:`_get_event_state`: prefers the lifespan-initialised Lock
    (created on the correct event loop) but lazily initialises it for
    non-``with`` TestClient usages.
    """
    try:
        return app.state.chat_argv_lock
    except AttributeError:
        app.state.chat_argv_lock = asyncio.Lock()
        return app.state.chat_argv_lock


def _get_pty_active_session_files(app: "FastAPI") -> dict[str, Path]:
    """Return channel -> active-session-file state for dashboard PTYs."""
    try:
        return app.state.pty_active_session_files
    except AttributeError:
        app.state.pty_active_session_files = {}
        return app.state.pty_active_session_files


app = FastAPI(title="Hermes Agent", version=__version__, lifespan=_lifespan)

# Memory-provider OAuth connect routes live in the memory layer, not here.
from hermes_cli.memory_oauth import router as _memory_oauth_router  # noqa: E402

app.include_router(_memory_oauth_router)

# ---------------------------------------------------------------------------
# Session token for protecting sensitive endpoints (reveal).
# The desktop shell mints the token and injects it via
# HERMES_DASHBOARD_SESSION_TOKEN so its main process can authenticate the
# /api calls it makes on the user's behalf; otherwise we generate one fresh
# on every server start. Either way it dies when the process exits and is
# injected into the SPA HTML so only the legitimate web UI can use it.
# ---------------------------------------------------------------------------
_SESSION_TOKEN = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or secrets.token_urlsafe(32)
_SESSION_HEADER_NAME = "X-Hermes-Session-Token"

# In-browser Chat tab (/chat, /api/pty, /api/ws, …).  Always enabled: the
# desktop app and the dashboard's own Chat tab both drive the agent over the
# `/api/ws` + `/api/pty` WebSockets, so the embedded-chat surface is an
# unconditional part of the dashboard.  Kept as a module-level constant (rather
# than inlining ``True`` at every gate) so the WS endpoints and the SPA token
# injection share a single, testable seam.
_DASHBOARD_EMBEDDED_CHAT_ENABLED = True

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
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints that do NOT require the session token.  Everything else under
# /api/ is gated by the auth middleware below.
#
# This list is defined in ``hermes_cli.dashboard_auth.public_paths`` so the
# OAuth gate middleware can honour the same allowlist — keeping the two
# gates in lockstep avoids drift like the wildcard-subdomain regression
# where ``/api/status`` was public under the legacy gate but 401'd under
# the OAuth gate (breaking the portal's liveness probe).
#
# Keep the upstream list minimal — only truly non-sensitive, read-only
# endpoints belong there.
# ---------------------------------------------------------------------------
from hermes_cli.dashboard_auth.public_paths import (
    PUBLIC_API_PATHS as _PUBLIC_API_PATHS,
)


def _has_valid_session_token(request: Request) -> bool:
    """True if the request carries a valid dashboard session token.

    The dedicated session header avoids collisions with reverse proxies that
    already use ``Authorization`` (for example Caddy ``basic_auth``). We still
    accept the legacy Bearer path for backward compatibility with older
    dashboard bundles.
    """
    session_header = request.headers.get(_SESSION_HEADER_NAME, "")
    if session_header and hmac.compare_digest(
        session_header.encode(),
        _SESSION_TOKEN.encode(),
    ):
        return True

    auth = request.headers.get("authorization", "")
    expected = f"Bearer {_SESSION_TOKEN}"
    return hmac.compare_digest(auth.encode(), expected.encode())


# Routes that may also authenticate via a ``?token=`` query param, for download
# links opened by the OS shell or a new browser tab where the session header
# can't be set. Kept narrow — same query-token tradeoff as the /api/pty WS.
_QUERY_TOKEN_API_PATHS: frozenset[str] = frozenset({"/api/files/download"})


def _has_valid_query_token(request: Request, path: str) -> bool:
    if path not in _QUERY_TOKEN_API_PATHS:
        return False
    token = request.query_params.get("token", "")
    return bool(token) and hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode())


def _require_token(request: Request) -> None:
    """Authorize a sensitive endpoint, raising 401 if the caller isn't allowed.

    Two auth schemes protect the dashboard, exactly one active per bind:

    * **Loopback / ``--insecure`` mode** (``auth_required`` False): the
      ephemeral ``_SESSION_TOKEN`` is injected into the SPA HTML and echoed
      back via ``X-Hermes-Session-Token`` (or the legacy ``Bearer`` header).
      Validate it here.
    * **Gated / OAuth mode** (``auth_required`` True): ``_SESSION_TOKEN`` is
      NOT injected (the SPA authenticates with a session cookie), so there is
      no token to check. The ``gated_auth_middleware`` has already verified the
      cookie before the request reached this handler — any non-public ``/api/``
      route it lets through carries a verified ``request.state.session``. The
      legacy ``auth_middleware`` likewise short-circuits in this mode. Requiring
      the (absent) token here would 401 every cookie-authenticated request,
      making plugin install/enable/disable and the other ``_require_token``
      endpoints permanently unreachable behind the gate. Defer to the gate.
    """
    if getattr(request.app.state, "auth_required", False):
        # Gate is authoritative. It attaches ``request.state.session`` on
        # success and 401s otherwise, so a request that reached us is already
        # authenticated. Belt-and-braces: confirm the session is present.
        if getattr(request.state, "session", None) is not None:
            return
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _has_valid_session_token(request):
        raise HTTPException(status_code=401, detail="Unauthorized")


# Accepted Host header values for loopback binds. DNS rebinding attacks
# point a victim browser at an attacker-controlled hostname (evil.test)
# which resolves to 127.0.0.1 after a TTL flip — bypassing same-origin
# checks because the browser now considers evil.test and our dashboard
# "same origin". Validating the Host header at the app layer rejects any
# request whose Host isn't one we bound for. See GHSA-ppp5-vxwm-4cf7.
_LOOPBACK_HOST_VALUES: frozenset = frozenset({
    "localhost", "127.0.0.1", "::1",
})


def should_require_auth(host: str, allow_public: bool = False) -> bool:
    """Return True iff the dashboard auth gate must be active.

    Truth table:
      host == loopback        → False (no auth — local-only, trusted operator)
      host != loopback        → True  (gate engages — OAuth or password required)

    "Loopback" is 127.0.0.1, localhost, ::1. RFC1918 / CGNAT / link-local are
    deliberately treated as PUBLIC — a hostile device on the same LAN is exactly
    the threat model the gate is designed for.

    ``allow_public`` (the legacy ``--insecure`` escape hatch) NO LONGER disables
    the gate. It is accepted for backward-compat with old launch scripts and
    desktop shells but is ignored: a non-loopback bind ALWAYS requires an auth
    provider (OAuth or the bundled password provider). This closes the
    unauthenticated-public-dashboard hole behind the June 2026 ``hermes-0day``
    MCP-persistence campaign, where ``--insecure --host 0.0.0.0`` left the
    config/MCP/agent surface open to internet scanners.
    """
    return host not in _LOOPBACK_HOST_VALUES


def _is_accepted_host(host_header: str, bound_host: str) -> bool:
    """True if the Host header targets the interface we bound to.

    Accepts:
    - Exact bound host (with or without port suffix)
    - Loopback aliases when bound to loopback
    - Any host when bound to 0.0.0.0 (explicit opt-in to non-loopback,
      no protection possible at this layer)
    """
    if not host_header:
        return False
    # Strip port suffix. IPv6 addresses use bracket notation:
    #   [::1]         — no port
    #   [::1]:9119    — with port
    # Plain hosts/v4:
    #   localhost:9119
    #   127.0.0.1:9119
    h = host_header.strip()
    if h.startswith("["):
        # IPv6 bracketed — port (if any) follows "]:"
        close = h.find("]")
        if close != -1:
            host_only = h[1:close]  # strip brackets
        else:
            host_only = h.strip("[]")
    else:
        host_only = h.rsplit(":", 1)[0] if ":" in h else h
    host_only = host_only.lower()

    # 0.0.0.0 bind means operator explicitly opted into all-interfaces
    # (requires --insecure per web_server.start_server). No Host-layer
    # defence can protect that mode; rely on operator network controls.
    if bound_host in {"0.0.0.0", "::"}:
        return True

    # Loopback bind: accept the loopback names
    bound_lc = bound_host.lower()
    if bound_lc in _LOOPBACK_HOST_VALUES:
        return host_only in _LOOPBACK_HOST_VALUES

    # Explicit non-loopback bind: require exact host match
    return host_only == bound_lc


@app.middleware("http")
async def host_header_middleware(request: Request, call_next):
    """Reject requests whose Host header doesn't match the bound interface.

    Defends against DNS rebinding: a victim browser on a localhost
    dashboard is tricked into fetching from an attacker hostname that
    TTL-flips to 127.0.0.1. CORS and same-origin checks don't help —
    the browser now treats the attacker origin as same-origin with the
    dashboard. Host-header validation at the app layer catches it.

    See GHSA-ppp5-vxwm-4cf7.
    """
    # Store the bound host on app.state so this middleware can read it —
    # set by start_server() at listen time.
    bound_host = getattr(app.state, "bound_host", None)
    if bound_host:
        host_header = request.headers.get("host", "")
        if not _is_accepted_host(host_header, bound_host):
            return JSONResponse(
                status_code=400,
                content={
                    "detail": (
                        "Invalid Host header. Dashboard requests must use "
                        "the hostname the server was bound to."
                    ),
                },
            )
    return await call_next(request)


@app.middleware("http")
async def _plugin_api_runtime_gate(request: Request, call_next):
    """Block requests to disabled plugin API routes at request time.

    :func:`_mount_plugin_api_routes` gates at import time, but if a plugin
    is disabled *after* the dashboard is already running, its FastAPI router
    remains mounted until restart.  This middleware enforces the enabled/
    disabled policy on every request to ``/api/plugins/{name}/...`` so that
    runtime config changes take effect immediately.

    Registered BEFORE the auth middlewares (so it executes AFTER them): a
    request that hasn't cleared auth must get auth's 401 first, never this
    gate's 404 — otherwise an unauthenticated caller could fingerprint which
    plugins are installed/enabled by reading the status code. We only reach
    the enabled/disabled check for a request that auth already let through.
    """
    path = request.url.path
    if path.startswith("/api/plugins/"):
        # Only gate authenticated requests. Unauthenticated ones fall
        # through so auth_middleware / the OAuth gate return 401 first and
        # this route can't be used as a plugin-name oracle.
        _authed = (
            getattr(request.state, "token_authenticated", False)
            or getattr(request.app.state, "auth_required", False)
            or _has_valid_session_token(request)
            or _has_valid_query_token(request, path)
        )
        if _authed:
            # Extract plugin name from /api/plugins/<name>/...
            parts = path.split("/")
            # parts: ['', 'api', 'plugins', '<name>', ...]
            if len(parts) >= 4:
                plugin_name = parts[3]
                if plugin_name:
                    try:
                        from hermes_cli.plugins_cmd import (
                            _get_enabled_set,
                            _get_disabled_set,
                        )
                        enabled_set = _get_enabled_set()
                        disabled_set = _get_disabled_set()
                    except Exception:
                        enabled_set = set()
                        disabled_set = set()
                    # Determine plugin source.  Check the cached plugin list;
                    # if not found, assume user plugin (safe default — blocks).
                    plugins = _get_dashboard_plugins()
                    plugin = next(
                        (p for p in plugins if p.get("name") == plugin_name),
                        None,
                    )
                    source = plugin.get("source") if plugin else "user"
                    if source == "user":
                        if plugin_name in disabled_set or plugin_name not in enabled_set:
                            return JSONResponse(
                                status_code=404,
                                content={"detail": "Plugin not found"},
                            )
                    elif source == "bundled":
                        if plugin_name in disabled_set:
                            return JSONResponse(
                                status_code=404,
                                content={"detail": "Plugin not found"},
                            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Dashboard OAuth auth gate — engaged only when start_server flags the
# bind as non-loopback-without-insecure.  No-op pass-through in loopback
# mode so the legacy auth_middleware (below) handles those binds via
# the injected ``_SESSION_TOKEN``.  Registered between host_header and
# auth_middleware so the order is: host check → cookie auth → token auth.
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _dashboard_auth_gate(request: Request, call_next):
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware
    return await gated_auth_middleware(request, call_next)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the session token on all /api/ routes except the public list."""
    # A request already authenticated by the token-auth seam (a service caller
    # presenting a bearer token on a registered token route) carries
    # ``token_authenticated`` — never bounce it through the cookie/session gate.
    if getattr(request.state, "token_authenticated", False):
        return await call_next(request)
    # When the OAuth gate is active, cookie-based auth (gated_auth_middleware
    # above) is authoritative.  The legacy _SESSION_TOKEN path is loopback-only
    # and is skipped here so the gate's session attachment isn't overridden.
    if getattr(request.app.state, "auth_required", False):
        return await call_next(request)
    path = request.url.path
    if path.startswith("/api/") and path not in _PUBLIC_API_PATHS:
        if not _has_valid_session_token(request) and not _has_valid_query_token(request, path):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )
    return await call_next(request)


@app.middleware("http")
async def _token_auth_seam(request: Request, call_next):
    """Outermost auth seam: non-interactive bearer-token auth for opted-in routes.

    Registered LAST so it runs FIRST (Starlette middleware is outermost-last).
    A registered token route is fully owned here — authenticate by token,
    attach the principal + ``token_authenticated`` flag, and let the downstream
    cookie/session gates skip enforcement. Non-token routes pass straight
    through untouched.
    """
    from hermes_cli.dashboard_auth.token_auth import token_auth_middleware
    return await token_auth_middleware(request, call_next)


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
        # "mistral" temporarily removed — mistralai PyPI package quarantined
        # (malicious 2.4.6 release on 2026-05-12). Restore once available.
        "options": ["local", "groq", "openai", "xai", "elevenlabs"],
    },
    "stt.elevenlabs.model_id": {
        "type": "select",
        "description": "ElevenLabs Scribe model",
        "options": ["scribe_v2", "scribe_v1"],
    },
    "display.skin": {
        "type": "select",
        "description": "CLI visual theme",
        "options": ["default", "ares", "mono", "slate"],
    },
    "dashboard.theme": {
        "type": "select",
        "description": "Web dashboard visual theme",
        "options": ["default", "midnight", "ember", "mono", "cyberpunk", "rose"],
    },
    "display.resume_display": {
        "type": "select",
        "description": "How resumed sessions display history",
        "options": ["minimal", "full", "off"],
    },
    "display.busy_input_mode": {
        "type": "select",
        "description": "Input behavior while agent is running",
        "options": ["interrupt", "queue", "steer"],
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
    "updates.non_interactive_local_changes": {
        "type": "select",
        "description": (
            "When the chat app / gateway updates Hermes (no terminal prompt), "
            "what to do with uncommitted local source edits. 'stash' keeps them "
            "and re-applies them after the update; 'discard' throws them away. "
            "Terminal updates always ask, regardless of this setting."
        ),
        "options": ["stash", "discard"],
    },
    "updates.refresh_cua_driver": {
        "type": "bool",
        "description": (
            "Refresh an already-installed cua-driver during hermes update. "
            "Disable this on non-admin macOS accounts where /Applications is "
            "not writable."
        ),
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
    "dashboard": "display",
    "code_execution": "agent",
    "prompt_caching": "agent",
    "goals": "agent",
    "updates": "general",
    # `onboarding.profile_build` is the only schema-surfaced onboarding field
    # (`onboarding.seen` is an internal latch dict, not a user setting), so fold
    # it into the agent tab rather than spawning a one-field orphan category.
    "onboarding": "agent",
    # Only `telegram.reactions` currently lives under telegram — fold it in
    # with the other messaging-platform config (discord) so it isn't an
    # orphan tab of one field.
    "telegram": "discord",
    # `computer_use.cua_telemetry` is the only schema-surfaced computer_use
    # field — fold it into the agent tab rather than spawning a one-field
    # orphan category.
    "computer_use": "agent",
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
        if full_key in {"_config_version", "memory.provider"}:
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
    profile: Optional[str] = None


class EnvVarUpdate(BaseModel):
    key: str
    value: str
    profile: Optional[str] = None
    # Optional bearer key for the connectivity probe of a custom/local endpoint
    # (``key == "OPENAI_BASE_URL"``). Self-hosted endpoints that gate
    # ``/v1/models`` behind auth otherwise look "reachable but empty"; sending
    # the key lets the probe enumerate the served models. Ignored for the
    # regular PUT /api/env path (which only reads key/value).
    api_key: str = ""


class EnvVarDelete(BaseModel):
    key: str
    profile: Optional[str] = None


class EnvVarReveal(BaseModel):
    key: str
    profile: Optional[str] = None


class MemoryProviderConfigUpdate(BaseModel):
    values: Dict[str, Any] = {}


class MemoryProviderSetupRequest(BaseModel):
    values: Dict[str, Any] = {}


class MessagingPlatformUpdate(BaseModel):
    enabled: Optional[bool] = None
    env: Dict[str, str] = {}
    clear_env: List[str] = []
    # Explicit body profile beats the query param injected by the global
    # dashboard profile switcher (same precedence as other scoped writes).
    profile: Optional[str] = None


class TelegramOnboardingStart(BaseModel):
    bot_name: Optional[str] = None


class TelegramOnboardingApply(BaseModel):
    allowed_user_ids: List[str]
    profile: Optional[str] = None


class WeixinOnboardingStart(BaseModel):
    profile: Optional[str] = None


class WeixinOnboardingApply(BaseModel):
    profile: Optional[str] = None


class WhatsAppOnboardingStart(BaseModel):
    mode: Optional[str] = "bot"
    allowed_users: Optional[str] = ""
    profile: Optional[str] = None


class WhatsAppOnboardingApply(BaseModel):
    mode: Optional[str] = None
    allowed_users: Optional[str] = None
    profile: Optional[str] = None


class AudioTranscriptionRequest(BaseModel):
    data_url: str
    mime_type: Optional[str] = None


class ManagedFileUpload(BaseModel):
    path: str
    data_url: str
    overwrite: bool = True


class ChatImageUpload(BaseModel):
    data_url: str
    filename: Optional[str] = None


class ManagedDirectoryCreate(BaseModel):
    path: str


class ManagedFileDelete(BaseModel):
    path: str
    recursive: bool = False


_AUDIO_MIME_EXTENSIONS: Dict[str, str] = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp3": ".mp3",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
    "video/webm": ".webm",
}
_MAX_TRANSCRIPTION_UPLOAD_BYTES = 25 * 1024 * 1024


def _audio_extension_for_mime(mime_type: str) -> str:
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    return _AUDIO_MIME_EXTENSIONS.get(normalized, ".webm")


class ModelAssignment(BaseModel):
    """Payload for POST /api/model/set — assign a provider/model to a slot.

    scope="main"        → writes model.provider + model.default
    scope="auxiliary"   → writes auxiliary.<task>.provider + auxiliary.<task>.model
    scope="auxiliary" with task=""  → applied to every auxiliary.* slot
    scope="auxiliary" with task="__reset__"  → resets every slot to provider="auto"
    """
    scope: str
    provider: str
    model: str
    task: str = ""
    # Optional OpenAI-compatible endpoint URL. Only honored for custom/local
    # providers on the main slot — lets the GUI configure a self-hosted endpoint
    # (vLLM, llama.cpp, Ollama, …) that needs no API key. The runtime resolver
    # reads model.base_url from config (it ignores OPENAI_BASE_URL), so this is
    # the path that actually wires a local endpoint into resolution.
    base_url: str = ""
    # Optional API key for a custom/local endpoint. Persisted to
    # ``model.api_key`` (where the runtime resolver reads it) so a self-hosted
    # endpoint that requires auth works from the GUI — mirrors the key the
    # ``hermes model`` custom flow collects. Honored only on the main slot for
    # custom/local providers.
    api_key: str = ""
    confirm_expensive_model: bool = False
    profile: Optional[str] = None


class MoaModelSlot(BaseModel):
    provider: str = ""
    model: str = ""


class MoaPresetPayload(BaseModel):
    reference_models: list[MoaModelSlot] = []
    aggregator: MoaModelSlot = MoaModelSlot()
    # None = temperature omitted from API calls (provider default), matching
    # single-model agent behavior.
    reference_temperature: Optional[float] = None
    aggregator_temperature: Optional[float] = None
    max_tokens: int = 4096
    enabled: bool = True


class MoaConfigPayload(BaseModel):
    default_preset: str = "default"
    active_preset: str = ""
    presets: dict[str, MoaPresetPayload] = {}
    # Backward-compatible flat payload fields used by older dashboard/desktop
    # clients during this PR's transition window.
    reference_models: list[MoaModelSlot] = []
    aggregator: MoaModelSlot = MoaModelSlot()
    reference_temperature: Optional[float] = None
    aggregator_temperature: Optional[float] = None
    max_tokens: int = 4096
    enabled: bool = True
    profile: Optional[str] = None


def _normalize_main_model_assignment(provider: str, model: str) -> tuple[str, str]:
    """Normalize a main-slot (provider, model) pair before persisting.

    The Models page has two assignment paths and only one of them was safe:

    - The "Change" picker sends a real Hermes provider slug — fine.
    - The per-card "Use as → Main model" menu sends ``entry.provider``
      from the analytics rows, falling back to the model's VENDOR prefix
      (``modelVendor("anthropic/claude-opus-4.6") == "anthropic"``) when
      the session row has no ``billing_provider`` (older sessions, NULL
      rows).  That wrote ``provider: anthropic`` +
      ``default: anthropic/claude-opus-4.6`` to config — a vendor-prefixed
      OpenRouter slug on the NATIVE Anthropic provider.  New sessions then
      400 against api.anthropic.com ("model: anthropic/claude-opus-4.6 not
      found") and the user reads it as "changing models does nothing".

    Two repairs, both at this single chokepoint so every caller inherits:

    1. Vendor-name → Hermes-provider mapping: when the provider string is
       not a known Hermes provider/alias (e.g. ``moonshotai``, ``x-ai`` is
       known but ``poolside`` isn't) but the model is a vendor-prefixed
       aggregator slug, keep the user's CURRENT aggregator if they're on
       one, else fall back to openrouter.
    2. Model-format normalization for the resolved provider via
       ``normalize_model_for_provider`` (e.g. ``anthropic/claude-opus-4.6``
       on native anthropic → ``claude-opus-4-6``).
    """
    from hermes_cli.models import _KNOWN_PROVIDER_NAMES, normalize_provider
    from hermes_cli.model_normalize import normalize_model_for_provider

    prov_in = (provider or "").strip()
    model_in = (model or "").strip()
    canonical = normalize_provider(prov_in)

    if canonical not in _KNOWN_PROVIDER_NAMES and "/" in model_in:
        # Vendor prefix posing as a provider (analytics fallback). Resolve
        # against the user's current provider when it's an aggregator that
        # serves vendor-prefixed slugs; otherwise default to openrouter.
        try:
            cur_cfg = load_config().get("model", {})
            cur_provider = (
                str(cur_cfg.get("provider", "") or "").strip().lower()
                if isinstance(cur_cfg, dict) else ""
            )
        except Exception:
            cur_provider = ""
        from hermes_cli.models import _AGGREGATOR_PROVIDERS
        if cur_provider and normalize_provider(cur_provider) in _AGGREGATOR_PROVIDERS:
            canonical = normalize_provider(cur_provider)
            prov_in = cur_provider
        else:
            canonical = "openrouter"
            prov_in = "openrouter"

    # Custom/user-config providers keep the model verbatim — the registry
    # normalizer doesn't know their namespaces.
    if canonical in _KNOWN_PROVIDER_NAMES and not canonical.startswith("custom"):
        try:
            normalized_model = normalize_model_for_provider(model_in, canonical)
            if normalized_model:
                model_in = normalized_model
        except Exception:
            _log.debug("model normalization failed for %s/%s", prov_in, model_in, exc_info=True)

    return prov_in, model_in


def _apply_main_model_assignment(
    model_cfg: "Any", provider: str, model: str, base_url: str = "", api_key: str = ""
) -> dict:
    """Apply a main-slot model assignment to a ``model`` config dict in place.

    Sets ``provider``/``default``, then reconciles ``base_url``:

    - An explicitly supplied ``base_url`` is always persisted (covers
      ``custom``/local endpoints and any provider whose key is bound to a
      non-default host).
    - Otherwise, a stale ``base_url`` is cleared ONLY when switching to a
      *different* provider — that URL belonged to the old provider. When the
      provider is unchanged and no new URL is supplied, the existing
      ``base_url`` is preserved. This keeps a user's custom endpoint (e.g. a
      Xiaomi MiMo Token Plan host, ``https://token-plan-*.xiaomimimo.com/v1``)
      alive when they merely re-pick a model under the same provider — picking
      a model previously wiped it, forcing the registry default and breaking
      Token Plan keys.

    The runtime resolver reads ``model.base_url`` from config (it ignores
    ``OPENAI_BASE_URL``) and only honors it when the configured provider matches
    and the pool entry is on the registry default, so preserving it here is what
    lets the override actually route. The hardcoded ``context_length`` override
    is always dropped since the new model may have a different context window.

    Returns the same dict (coerced to a fresh dict if the input wasn't one) so
    callers can assign it straight back onto the model config.
    """
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    prev_provider = str(model_cfg.get("provider") or "").strip().lower()
    new_provider = provider.strip().lower()
    model_cfg["provider"] = provider
    model_cfg["default"] = model
    if base_url.strip():
        model_cfg["base_url"] = base_url.strip()
    elif model_cfg.get("base_url") and new_provider != prev_provider:
        # Switching providers: the old URL belonged to the old provider, drop
        # it so the new provider's default endpoint is used. Same-provider
        # re-assignment keeps the user's configured base_url intact.
        model_cfg["base_url"] = ""
    # The endpoint key follows the same lifecycle as base_url: an explicit key
    # is always persisted; an existing key is dropped only when switching to a
    # different provider (it belonged to the old endpoint), and preserved on a
    # same-provider re-pick so re-selecting a model doesn't wipe the key.
    if api_key.strip():
        model_cfg["api_key"] = api_key.strip()
        model_cfg.pop("api", None)
    elif model_cfg.get("api_key") and new_provider != prev_provider:
        clear_model_endpoint_credentials(model_cfg, clear_api_mode=False)
    if new_provider != prev_provider:
        clear_model_endpoint_credentials(model_cfg, clear_api_key=False)
    model_cfg.pop("context_length", None)
    return model_cfg


_GATEWAY_HEALTH_URL = os.getenv("GATEWAY_HEALTH_URL")
try:
    _GATEWAY_HEALTH_TIMEOUT = float(os.getenv("GATEWAY_HEALTH_TIMEOUT", "3"))
except (ValueError, TypeError):
    _log.warning(
        "Invalid GATEWAY_HEALTH_TIMEOUT value %r — using default 3.0s",
        os.getenv("GATEWAY_HEALTH_TIMEOUT"),
    )
    _GATEWAY_HEALTH_TIMEOUT = 3.0

_STATUS_ACTIVE_SESSIONS_TIMEOUT = 0.75

# DEPRECATED (scheduled for removal): GATEWAY_HEALTH_URL / GATEWAY_HEALTH_TIMEOUT.
# Cross-container / cross-host gateway liveness detection will be folded into a
# first-class dashboard config key so it's no longer Docker-adjacent lore buried
# in env vars.  The env vars still work for now so existing Compose deployments
# don't break.  Do not add new callers — wire new uses through the planned
# config surface.


def _probe_gateway_health() -> tuple[bool, dict | None]:
    """Probe the gateway via its HTTP health endpoint (cross-container).

    .. deprecated::
        Driven by the deprecated ``GATEWAY_HEALTH_URL`` /
        ``GATEWAY_HEALTH_TIMEOUT`` env vars.  Scheduled for removal alongside
        a move to a first-class dashboard config key.  See
        :data:`_GATEWAY_HEALTH_URL` for context.

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


def _count_status_active_sessions() -> int:
    """Return the dashboard status active-session count.

    This is best-effort status garnish, not a critical path.  Use a read-only
    connection so /api/status never tries to initialise or migrate state.db
    while another Hermes process is writing to it.
    """
    from hermes_state import DEFAULT_DB_PATH, SessionDB

    # read_only opens require the DB to already exist (see SessionDB.__init__
    # read_only contract) — on a fresh install every /api/status poll would
    # otherwise pay an OperationalError until the first session is written.
    if not DEFAULT_DB_PATH.exists():
        return 0

    db = SessionDB(read_only=True)
    try:
        sessions = db.list_sessions_rich(limit=50, compact_rows=True)
        now = time.time()
        return sum(
            1 for s in sessions
            if s.get("ended_at") is None
            and (now - s.get("last_active", s.get("started_at", 0))) < 300
        )
    finally:
        db.close()


async def _status_active_sessions() -> int:
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _count_status_active_sessions),
            timeout=_STATUS_ACTIVE_SESSIONS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        _log.debug(
            "/api/status active session count exceeded %.2fs; returning 0",
            _STATUS_ACTIVE_SESSIONS_TIMEOUT,
        )
    except Exception as exc:
        _log.debug("/api/status active session count unavailable: %s", exc)
    return 0


# Image MIME types this endpoint will serve. Extension-allowlisted so an
# authenticated caller can't pull non-image files through it.
_MEDIA_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
}
_MEDIA_MAX_BYTES = 25 * 1024 * 1024
_MANAGED_FILES_ROOT_ENV = "HERMES_DASHBOARD_FILES_ROOT"
_MANAGED_FILE_MAX_BYTES = 100 * 1024 * 1024
_HOSTED_MANAGED_FILES_ROOT = Path("/opt/data")


@dataclass(frozen=True)
class ManagedFilesPolicy:
    default_path: Path
    locked_root: Path | None
    can_change_path: bool


_FS_READDIR_HIDDEN = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".next",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}

# Filenames that must never be listed, read, or downloaded through the
# managed-files API.  These typically contain credentials (API keys, tokens)
# and exposing them through the dashboard file browser is a security leak —
# see issue #57505. The set mirrors the credential-file basenames of the two
# canonical credential guards elsewhere in the codebase
# (agent.file_safety.get_read_block_error and
# gateway.platforms.base._ROOT_CREDENTIAL_FILES) so the dashboard Files tab
# doesn't lag behind them — an operator can point the managed root at
# HERMES_HOME itself, at which point every one of these basenames is a live
# secret store sitting in the browsable tree.
_SENSITIVE_MANAGED_FILE_BASENAMES = frozenset({
    "auth.json",
    "auth.lock",
    "credentials",
    "config.yaml",
    ".anthropic_oauth.json",
    "google_token.json",
    "google_oauth_pending.json",
    "google_oauth.json",
    "webhook_subscriptions.json",
    "bws_cache.json",
    # git's credential-store helper cache (agent.file_safety blocks this too).
    ".git-credentials",
})

# Directory names whose entire subtree is credential material. Both canonical
# guards deny these as directory trees, not basenames:
#   * gateway.platforms.base._ROOT_CREDENTIAL_DIRS = {"pairing", "mcp-tokens"}
#   * agent.file_safety.get_read_block_error (mcp-tokens/ prefix match)
# The managed-files API lets the browser descend into subdirs, so a
# basename-only guard would still expose e.g. ``mcp-tokens/<server>.json``
# (live MCP OAuth tokens) and ``pairing/<x>``. We match on ANY path component
# so these trees are blocked wherever they appear under the browsable root,
# without needing to resolve them relative to HERMES_HOME.
_SENSITIVE_MANAGED_DIR_NAMES = frozenset({
    "mcp-tokens",
    "pairing",
})


def _is_sensitive_filename(name: str) -> bool:
    """Return True for a basename the managed-files API must never expose.

    Covers ``.env`` / ``.env.<suffix>`` / ``.envrc`` variants plus the
    canonical Hermes credential-store basenames (see
    ``_SENSITIVE_MANAGED_FILE_BASENAMES`` above).

    Case-insensitive so ``.ENV`` / ``.Env.local`` / ``Auth.JSON`` on
    case-insensitive filesystems (macOS/Windows mounts) can't slip past
    the guard.

    Basename-only: for the directory-tree credential stores
    (``mcp-tokens/``, ``pairing/``) that the canonical guards also deny,
    use :func:`_is_sensitive_path`, which the API call sites route through.
    """
    lowered = name.lower()
    if lowered == ".env" or lowered.startswith(".env.") or lowered == ".envrc":
        return True
    return lowered in _SENSITIVE_MANAGED_FILE_BASENAMES


def _is_sensitive_path(path: Path) -> bool:
    """Return True for any path the managed-files API must never expose.

    Combines the basename denylist (:func:`_is_sensitive_filename`) with a
    credential-directory-tree check: a path is sensitive if its own basename
    is sensitive OR any of its path components is a credential directory
    (``mcp-tokens`` / ``pairing``). The component match is case-insensitive
    and needs no HERMES_HOME resolution, so it blocks these trees wherever
    they sit under the operator-configured managed root — closing the gap
    the canonical guards cover as directory trees but a basename-only check
    would miss.

    Read-side only: this guards list/read/download (the #57505 exfil surface).
    The write endpoints (upload/mkdir/delete) are a separate threat class
    handled by the write-path checks; extending this guard to them is out of
    scope for this fix.
    """
    if _is_sensitive_filename(path.name):
        return True
    return any(part.lower() in _SENSITIVE_MANAGED_DIR_NAMES for part in path.parts)


_FS_DATA_URL_MAX_BYTES = 16 * 1024 * 1024
_FS_TEXT_SOURCE_MAX_BYTES = 64 * 1024 * 1024
_FS_TEXT_PREVIEW_MAX_BYTES = 512 * 1024
# Upper bound for the in-app spot editor's save. The editor only opens
# non-truncated text (<= the preview cap), so this is a safety ceiling against
# a pasted-in megablob, not the expected payload size.
_FS_TEXT_WRITE_MAX_BYTES = 8 * 1024 * 1024
_FS_PREVIEW_LANGUAGE_BY_EXT = {
    ".c": "c",
    ".conf": "ini",
    ".cpp": "cpp",
    ".css": "css",
    ".csv": "csv",
    ".go": "go",
    ".graphql": "graphql",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "jsx",
    ".kt": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".mjs": "javascript",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "shell",
    ".sql": "sql",
    ".svg": "xml",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".txt": "text",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "shell",
}
_FS_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".bmp": "image/bmp",
    ".flac": "audio/flac",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4a": "audio/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg; codecs=opus",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".webp": "image/webp",
}


def _fs_path(raw_path: str) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Path is required")
    if "\0" in raw:
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        if raw.lower().startswith("file:"):
            parsed = urllib.parse.urlparse(raw)
            if parsed.netloc and parsed.netloc not in {"", "localhost"}:
                raise ValueError
            raw = urllib.request.url2pathname(parsed.path)
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid path")


def _fs_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _FS_MIME_TYPES:
        return _FS_MIME_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _fs_looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\0" in data:
        return True
    suspicious = sum(1 for byte in data if byte < 32 and byte not in {9, 10, 13})
    return suspicious / len(data) > 0.12


def _fs_regular_file(path: Path) -> tuple[Path, os.stat_result]:
    target = _fs_path(str(path))
    try:
        st = target.stat()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except NotADirectoryError:
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")
    if stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be read")
    return target, st


def _fs_find_git_root(start: Path) -> str | None:
    directory = start
    for _ in range(50):
        try:
            if (directory / ".git").exists():
                return str(directory)
        except OSError:
            return None
        parent = directory.parent
        if parent == directory:
            return None
        directory = parent
    return None


def _fs_default_cwd() -> str:
    cfg_terminal = load_config().get("terminal") or {}
    raw = str(cfg_terminal.get("cwd") or os.environ.get("TERMINAL_CWD") or "").strip()
    if raw and raw not in {".", "auto", "cwd"}:
        try:
            candidate = Path(raw).expanduser().resolve(strict=False)
            if candidate.is_dir():
                return str(candidate)
        except (OSError, RuntimeError):
            pass
    return str(Path.cwd())


def _fs_git_branch(cwd: str) -> str:
    try:
        run_kwargs: Dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 2,
            "check": False,
        }
        if sys.platform == "win32":
            run_kwargs["creationflags"] = windows_hide_flags()
        result = subprocess.run(
            ["git", "-C", cwd, "branch", "--show-current"],
            **run_kwargs,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _media_serve_roots() -> list[Path]:
    """Directories ``GET /api/media`` is allowed to read from.

    Confined to where the agent and attach pipeline actually write media on the
    gateway host — its images dir and cache subtree. This stops an authenticated
    client from reading image-extension files anywhere on disk (e.g. a renamed
    key or a screenshot outside the cache) merely because the suffix passes the
    allowlist.
    """
    home = get_hermes_home()
    roots = [home / "images", home / "screenshots", home / "cache"]
    out: list[Path] = []
    for root in roots:
        try:
            out.append(root.resolve())
        except (OSError, RuntimeError):
            continue
    return out


@app.get("/api/media")
async def get_media(path: str):
    """Return a gateway-local image file as a base64 data URL.

    Lets remote clients (the desktop app over the network, or the web dashboard
    in a browser) display images the agent wrote to *this* machine's filesystem
    — they can't read the gateway's local disk directly.

    Auth-gated by the session token like every other /api route. Restricted to
    an image-extension allowlist, a size cap, AND the gateway's own media roots
    (resolved, symlink-safe) so it can't be used to read arbitrary files.
    """
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")

    if target.suffix.lower() not in _MEDIA_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    roots = _media_serve_roots()
    if not any(target == root or root in target.parents for root in roots):
        raise HTTPException(status_code=403, detail="Path outside media roots")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.stat().st_size > _MEDIA_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return {"data_url": f"data:{_MEDIA_CONTENT_TYPES[target.suffix.lower()]};base64,{encoded}"}


def _canonical_path(path: Path, *, require_exists: bool = False) -> Path:
    try:
        return path.expanduser().resolve(strict=require_exists)
    except FileNotFoundError:
        if require_exists:
            raise HTTPException(status_code=404, detail="Path not found")
        raise
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")


def _ensure_managed_root(raw_path: str | Path) -> Path:
    root = Path(raw_path).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Managed files root is unavailable: {exc}")
    if not resolved.is_dir():
        raise HTTPException(status_code=500, detail="Managed files root is not a directory")
    return resolved


def _path_is_under(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _path_text(raw_path: str | None) -> str:
    text = str(raw_path or "").strip()
    if "\x00" in text:
        raise HTTPException(status_code=400, detail="Invalid path")
    return text


def _local_dashboard_request(request: Request) -> bool:
    if getattr(request.app.state, "auth_required", False):
        return False
    host = (request.url.hostname or "").lower()
    client_host = (request.client.host if request.client else "").lower()
    local_hosts = {"", "localhost", "127.0.0.1", "::1", "testserver", "testclient"}
    return host in local_hosts or client_host in local_hosts


def _default_hermes_root_is_opt_data() -> bool:
    raw = os.environ.get("HERMES_HOME", "").strip()
    if not raw:
        return False
    try:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root().expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        root = Path(raw).expanduser().resolve(strict=False)
    return root == _HOSTED_MANAGED_FILES_ROOT


def _dashboard_local_update_managed_externally() -> bool:
    """Return true when the dashboard should not offer ``hermes update``.

    Containerized dashboards are updated by the outer launcher/image, not by an
    in-browser local update action. Keep this dashboard capability separate
    from install-method detection: manual git/pip installs inside containers can
    still behave like their actual install method in the CLI.

    However, when the install method is ``git`` (a bind-mounted checkout inside
    a container — e.g. the hermes-webui image sharing the Hermes source tree),
    the dashboard's ``hermes update`` button is the correct update path and
    should not be suppressed. Other containerized install methods remain
    externally managed unless their apply path is proven safe inside the
    running container filesystem.
    """
    if _default_hermes_root_is_opt_data():
        return True
    try:
        from hermes_constants import is_container

        if not is_container():
            return False
    except Exception:
        return False
    # We are inside a container, but the install may still be self-managed.
    # If the install method is git, the dashboard update button works against
    # the mounted checkout and should be offered. Keep pip blocked inside
    # containers: its apply path mutates the running container filesystem and
    # is not the bind-mounted checkout case this gate is meant to recover.
    try:
        method = detect_install_method(PROJECT_ROOT)
        if method == "git":
            return False
    except Exception:
        pass
    return True


def _managed_files_policy(request: Request, *, create_root: bool = True) -> ManagedFilesPolicy:
    raw_forced_root = os.environ.get(_MANAGED_FILES_ROOT_ENV, "").strip()
    if raw_forced_root:
        root = _ensure_managed_root(raw_forced_root) if create_root else _canonical_path(Path(raw_forced_root))
        return ManagedFilesPolicy(default_path=root, locked_root=root, can_change_path=False)

    # Remote/OAuth access does not imply a hosted container. Users can expose a
    # local dashboard through the auth gate (for example a macOS launchd install)
    # and still expect the Files page to browse their local home directory. Lock
    # to /opt/data only when the installation's Hermes root is actually /opt/data
    # (the container/hosted layout) or when HERMES_DASHBOARD_FILES_ROOT is set.
    if _default_hermes_root_is_opt_data():
        root = _ensure_managed_root(_HOSTED_MANAGED_FILES_ROOT) if create_root else _HOSTED_MANAGED_FILES_ROOT
        return ManagedFilesPolicy(default_path=root, locked_root=root, can_change_path=False)

    home = _canonical_path(Path.home())
    return ManagedFilesPolicy(default_path=home, locked_root=None, can_change_path=True)


def _resolve_managed_path(
    raw_path: str | None,
    request: Request,
    *,
    for_write: bool = False,
) -> tuple[ManagedFilesPolicy, Path, str]:
    policy = _managed_files_policy(request)
    text = _path_text(raw_path)
    root = policy.locked_root

    if root is not None and (not text or text in {".", "/"}):
        candidate = root
    elif not text:
        candidate = policy.default_path
    else:
        candidate = Path(text).expanduser()
        if root is not None and not candidate.is_absolute():
            if any(part == ".." for part in candidate.parts):
                raise HTTPException(status_code=400, detail="Path cannot contain '..'")
            candidate = root / candidate
        elif not candidate.is_absolute():
            raise HTTPException(status_code=400, detail="Path must be absolute")

    if ".." in candidate.parts:
        raise HTTPException(status_code=400, detail="Path cannot contain '..'")

    if for_write and not candidate.exists():
        parent = _canonical_path(candidate.parent)
        resolved = parent / candidate.name
    else:
        resolved = _canonical_path(candidate, require_exists=not for_write)

    if root is not None and not _path_is_under(root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    return policy, resolved, str(resolved)


def _managed_response_meta(policy: ManagedFilesPolicy) -> Dict[str, Any]:
    locked_root = str(policy.locked_root) if policy.locked_root is not None else None
    return {
        "root": locked_root,
        "locked_root": locked_root,
        "can_change_path": policy.can_change_path,
    }


def _managed_file_entry(policy: ManagedFilesPolicy, target: Path) -> Dict[str, Any]:
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")
    if policy.locked_root is not None and not _path_is_under(policy.locked_root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    try:
        st = resolved.stat()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat path: {exc}")

    is_dir = resolved.is_dir()
    mime_type = None if is_dir else (mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
    return {
        "name": target.name or resolved.name or str(resolved),
        "path": str(resolved),
        "is_directory": is_dir,
        "size": None if is_dir else st.st_size,
        "mtime": st.st_mtime,
        "mime_type": mime_type,
    }


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    text = (data_url or "").strip()
    if not text.startswith("data:") or "," not in text:
        raise HTTPException(status_code=400, detail="Upload payload must be a data URL")
    header, encoded = text.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Upload payload must be base64 encoded")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Upload payload is not valid base64")
    if len(data) > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")
    return data, mime_type


_CHAT_IMAGE_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_CHAT_IMAGE_ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_CHAT_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)


def _sanitize_chat_image_filename(filename: str | None) -> str:
    candidate = Path(str(filename or "").strip()).name
    candidate = re.sub(r"[\x00-\x1f]+", "_", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or "pasted-image"


def _chat_image_extension(data: bytes) -> str | None:
    head = data[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _CHAT_IMAGE_MAGIC:
        if head.startswith(sig):
            return ext
    return None


def _decode_chat_image_upload(payload: ChatImageUpload) -> tuple[bytes, str, str]:
    data, mime_type = _decode_data_url(payload.data_url)
    if not mime_type.lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload payload must be an image")
    if len(data) > _CHAT_IMAGE_UPLOAD_MAX_BYTES:
        mb = _CHAT_IMAGE_UPLOAD_MAX_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Image is too large; cap is {mb} MB")

    ext = _chat_image_extension(data)
    if ext not in _CHAT_IMAGE_ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    return data, mime_type, ext


@app.post("/api/chat/image-upload")
async def upload_chat_image(payload: ChatImageUpload, profile: Optional[str] = None):
    """Persist a browser-provided chat image where the embedded TUI can read it.

    The dashboard /chat page runs Hermes inside an xterm.js PTY. Browser
    clipboard image bytes are not visible to the server-side clipboard, so the
    page uploads them here, then drives the TUI's ``/image <path>`` command
    with the returned gateway-visible path. Files land under
    ``HERMES_HOME/images/`` — the same directory ``clipboard.paste`` /
    ``image.attach`` already use.
    """
    data, mime_type, ext = _decode_chat_image_upload(payload)
    with _profile_scope(profile) as scoped_home:
        home = scoped_home or get_hermes_home()
        img_dir = Path(home) / "images"
        try:
            img_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Image directory is not writable")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not create image directory: {exc}")

        stem = Path(_sanitize_chat_image_filename(payload.filename)).stem or "pasted-image"
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "pasted-image"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = img_dir / f"dashboard_{ts}_{secrets.token_hex(4)}_{stem}{ext}"

        try:
            target.write_bytes(data)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Image directory is not writable")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not write image: {exc}")

    return {
        "ok": True,
        "path": str(target),
        "name": target.name,
        "bytes": len(data),
        "mime_type": mime_type,
    }


@app.get("/api/files")
async def list_managed_files(request: Request, path: Optional[str] = None):
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    try:
        entries = [
            _managed_file_entry(policy, child)
            for child in target.iterdir()
            if not _is_sensitive_path(child)
        ]
    except PermissionError:
        raise HTTPException(status_code=403, detail="Directory is not readable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read directory: {exc}")

    entries.sort(key=lambda item: (not item["is_directory"], str(item["name"]).lower()))
    locked_root = policy.locked_root
    parent = None
    if target.parent != target and (locked_root is None or target != locked_root):
        parent = str(target.parent)
    return {
        "path": display_path,
        "parent": parent,
        "entries": entries,
        **_managed_response_meta(policy),
    }


@app.get("/api/files/read")
async def read_managed_file(request: Request, path: str):
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")

    try:
        size = target.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat file: {exc}")
    if size > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")

    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    try:
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read file: {exc}")

    return {
        "name": target.name,
        "path": display_path,
        "size": size,
        "mime_type": mime_type,
        "data_url": f"data:{mime_type};base64,{encoded}",
        **_managed_response_meta(policy),
    }


@app.get("/api/files/download")
async def download_managed_file(request: Request, path: str):
    """Stream a managed file as an attachment download.

    Remote clients (desktop app, browser dashboard) open agent-written files
    that live on *this* gateway's disk, not theirs. Auth-gated like every other
    managed-files route — ``auth_middleware`` additionally accepts the session
    token as a ``?token=`` query param here so a shell/browser-opened download
    (which can't set the session header) still authenticates. See ``/api/pty``
    for the same query-token precedent.
    """
    policy, target, _display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")

    try:
        size = target.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat file: {exc}")
    if size > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")

    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"

    return FileResponse(
        path=str(target),
        media_type=mime_type,
        filename=target.name,
        content_disposition_type="attachment",
    )


@app.post("/api/files/upload")
async def upload_managed_file(payload: ManagedFileUpload, request: Request):
    policy, target, display_path = _resolve_managed_path(payload.path, request, for_write=True)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=409, detail="A directory already exists at that path")
    if target.exists() and not payload.overwrite:
        raise HTTPException(status_code=409, detail="File already exists")

    data, _mime_type = _decode_data_url(payload.data_url)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


# Stream uploads to disk in fixed-size chunks. The legacy JSON endpoint above
# buffers the whole file as a base64 data URL in a JSON body, which (a) inflates
# the payload ~33%, (b) holds the entire file (plus its decoded copy) in memory,
# and (c) reliably trips upstream proxy body-size/timeout limits with a 502 on
# large backup archives (NS-501). This multipart endpoint reads the request body
# in 1 MiB chunks straight to a temp file, enforces the size cap as it goes, and
# atomically renames into place — constant memory, no base64 inflation.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


@app.post("/api/files/upload-stream")
async def upload_managed_file_stream(
    request: Request,
    file: UploadFile = File(...),
    path: str = Form(...),
    overwrite: bool = Form(True),
):
    policy, target, display_path = _resolve_managed_path(path, request, for_write=True)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=409, detail="A directory already exists at that path")
    if target.exists() and not overwrite:
        raise HTTPException(status_code=409, detail="File already exists")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create parent directory: {exc}")

    # Write to a sibling temp file first so a partial/aborted upload never
    # clobbers an existing file, then atomically rename into place.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".upload", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    total = 0
    renamed = False
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANAGED_FILE_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="File is too large")
                out.write(chunk)
        os.replace(tmp_path, target)
        renamed = True
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")
    finally:
        # Clean up the temp file on every non-success exit, including
        # BaseException paths the `except` clauses above don't catch — most
        # importantly asyncio.CancelledError when a browser aborts a large
        # upload mid-stream (the exact NS-501 scenario). os.replace clears
        # tmp_path on success, so only unlink when the rename didn't happen.
        if not renamed:
            tmp_path.unlink(missing_ok=True)
        await file.close()

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


@app.post("/api/files/mkdir")
async def create_managed_directory(payload: ManagedDirectoryCreate, request: Request):
    policy, target, display_path = _resolve_managed_path(payload.path, request, for_write=True)
    if target.exists() and not target.is_dir():
        raise HTTPException(status_code=409, detail="A file already exists at that path")

    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Directory is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create directory: {exc}")

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


@app.delete("/api/files")
async def delete_managed_file(payload: ManagedFileDelete, request: Request):
    policy, target, display_path = _resolve_managed_path(payload.path, request)
    if policy.locked_root is not None and target == policy.locked_root:
        raise HTTPException(status_code=400, detail="Cannot delete the managed files root")
    if target.parent == target:
        raise HTTPException(status_code=400, detail="Cannot delete the filesystem root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    try:
        if target.is_dir():
            if payload.recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
    except OSError as exc:
        status_code = 409 if target.is_dir() and not payload.recursive else 500
        raise HTTPException(status_code=status_code, detail=f"Could not delete path: {exc}")

    return {"ok": True, "path": display_path, **_managed_response_meta(policy)}


@app.get("/api/fs/list")
async def fs_list(path: str):
    target = _fs_path(path)
    try:
        entries = []
        with os.scandir(target) as scan:
            for entry in scan:
                if entry.name in _FS_READDIR_HIDDEN:
                    continue
                entries.append({
                    "name": entry.name,
                    "path": str(target / entry.name),
                    "isDirectory": entry.is_dir(follow_symlinks=False),
                })
        entries.sort(key=lambda item: (not item["isDirectory"], item["name"].lower(), item["name"]))
        return {"entries": entries}
    except FileNotFoundError:
        return {"entries": [], "error": "ENOENT"}
    except NotADirectoryError:
        return {"entries": [], "error": "ENOTDIR"}
    except PermissionError:
        return {"entries": [], "error": "EACCES"}
    except OSError as exc:
        return {"entries": [], "error": getattr(exc, "strerror", None) or "read-error"}


@app.get("/api/fs/read-text")
async def fs_read_text(path: str):
    target, st = _fs_regular_file(_fs_path(path))
    if st.st_size > _FS_TEXT_SOURCE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    bytes_to_read = min(st.st_size, _FS_TEXT_PREVIEW_MAX_BYTES)
    try:
        with target.open("rb") as handle:
            data = handle.read(bytes_to_read)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "File read failed")
    return {
        "binary": _fs_looks_binary(data[:4096]),
        "byteSize": st.st_size,
        "language": _FS_PREVIEW_LANGUAGE_BY_EXT.get(target.suffix.lower(), "text"),
        "mimeType": _fs_mime_type(target),
        "path": str(target),
        "text": data.decode("utf-8", errors="replace"),
        "truncated": st.st_size > _FS_TEXT_PREVIEW_MAX_BYTES,
    }


class FsWriteText(BaseModel):
    path: str
    content: str


@app.post("/api/fs/write-text")
async def fs_write_text(payload: FsWriteText):
    """Overwrite (or create) a UTF-8 text file for the in-app spot editor.

    Mirrors the local Electron ``hermes:fs:writeText`` hardening: the path is
    resolved + validated by ``_fs_path``, the parent directory must already
    exist (we never build directory trees), only regular files may be replaced,
    and the payload is size-capped. The write is staged to a sibling temp file
    and ``os.replace``-d into place so a crash mid-write can't truncate the
    original. Stale-on-disk detection is the client's job (re-read before save),
    so both transports behave identically.
    """
    target = _fs_path(payload.path)
    text = payload.content or ""
    if len(text.encode("utf-8")) > _FS_TEXT_WRITE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content too large")

    try:
        st: Optional[os.stat_result] = target.stat()
    except FileNotFoundError:
        st = None
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")

    if st is not None and stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if st is not None and not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be written")
    if not target.parent.is_dir():
        raise HTTPException(status_code=400, detail="Parent directory does not exist")

    tmp = target.with_name(f".{target.name}.hermes-tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")

    return {"ok": True, "path": str(target), "byteSize": len(text.encode("utf-8"))}


@app.get("/api/fs/read-data-url")
async def fs_read_data_url(path: str):
    target, st = _fs_regular_file(_fs_path(path))
    if st.st_size > _FS_DATA_URL_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "File read failed")
    return {"dataUrl": f"data:{_fs_mime_type(target)};base64,{encoded}"}


@app.get("/api/fs/git-root")
async def fs_git_root(path: str):
    target = _fs_path(path)
    try:
        st = target.stat()
        start = target if stat.S_ISDIR(st.st_mode) else target.parent
    except OSError:
        start = target
    return {"root": _fs_find_git_root(start)}


@app.get("/api/fs/default-cwd")
async def fs_default_cwd():
    cwd = _fs_default_cwd()
    return {"cwd": cwd, "branch": _fs_git_branch(cwd)}


# ---------------------------------------------------------------------------
# Git ops — the remote half of the desktop coding rail + review pane.
#
# The desktop runs these as Electron-local git on the user's machine; over a
# remote gateway that's the wrong filesystem, so we mirror them here (same auth
# gate + path hardening as /api/fs). Logic lives in ``hermes_cli.web_git``;
# these are thin, executor-offloaded wrappers (git/gh can block).
# ---------------------------------------------------------------------------

from hermes_cli import web_git as _web_git  # noqa: E402


async def _git_op(fn, *args):
    """Run a (blocking) git op off the event loop; map a failed mutation to 400."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, fn, *args)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "git operation failed")


def _git_path(path: str) -> str:
    return str(_fs_path(path))


class GitPathBody(BaseModel):
    path: str

class GitFileBody(BaseModel):
    path: str
    file: Optional[str] = None


class GitCommitBody(BaseModel):
    path: str
    message: str
    push: bool = False


class GitWorktreeAddBody(BaseModel):
    path: str
    name: Optional[str] = None
    branch: Optional[str] = None
    base: Optional[str] = None
    existingBranch: Optional[str] = None


class GitWorktreeRemoveBody(BaseModel):
    path: str
    worktreePath: str
    force: bool = False


class GitBranchSwitchBody(BaseModel):
    path: str
    branch: str


@app.get("/api/git/status")
async def git_status_route(path: str):
    return await _git_op(_web_git.repo_status, _git_path(path))


@app.get("/api/git/worktrees")
async def git_worktrees_route(path: str):
    return {"worktrees": await _git_op(_web_git.worktree_list, _git_path(path))}


@app.get("/api/git/branches")
async def git_branches_route(path: str):
    return {"branches": await _git_op(_web_git.branch_list, _git_path(path))}


@app.get("/api/git/review/list")
async def git_review_list_route(path: str, scope: str = "uncommitted", base: Optional[str] = None):
    return await _git_op(_web_git.review_list, _git_path(path), scope, base)


@app.get("/api/git/review/diff")
async def git_review_diff_route(
    path: str, file: str, scope: str = "uncommitted", base: Optional[str] = None, staged: bool = False
):
    return {"diff": await _git_op(_web_git.review_diff, _git_path(path), file, scope, base, staged)}


@app.get("/api/git/file-diff")
async def git_file_diff_route(path: str, file: str):
    return {"diff": await _git_op(_web_git.file_diff_vs_head, _git_path(path), file)}


@app.get("/api/git/review/commit-context")
async def git_commit_context_route(path: str):
    return await _git_op(_web_git.review_commit_context, _git_path(path))


@app.get("/api/git/review/rev-parse")
async def git_rev_parse_route(path: str, ref: Optional[str] = None):
    return {"sha": await _git_op(_web_git.review_rev_parse, _git_path(path), ref)}


@app.get("/api/git/review/ship-info")
async def git_ship_info_route(path: str):
    return await _git_op(_web_git.review_ship_info, _git_path(path))


@app.post("/api/git/review/stage")
async def git_stage_route(body: GitFileBody):
    return await _git_op(_web_git.review_stage, _git_path(body.path), body.file)


@app.post("/api/git/review/unstage")
async def git_unstage_route(body: GitFileBody):
    return await _git_op(_web_git.review_unstage, _git_path(body.path), body.file)


@app.post("/api/git/review/revert")
async def git_revert_route(body: GitFileBody):
    return await _git_op(_web_git.review_revert, _git_path(body.path), body.file)


@app.post("/api/git/review/commit")
async def git_commit_route(body: GitCommitBody):
    return await _git_op(_web_git.review_commit, _git_path(body.path), body.message, body.push)


@app.post("/api/git/review/push")
async def git_push_route(body: GitPathBody):
    return await _git_op(_web_git.review_push, _git_path(body.path))


@app.post("/api/git/review/create-pr")
async def git_create_pr_route(body: GitPathBody):
    return await _git_op(_web_git.review_create_pr, _git_path(body.path))


@app.post("/api/git/worktree/add")
async def git_worktree_add_route(body: GitWorktreeAddBody):
    options = {
        key: value
        for key, value in {
            "name": body.name,
            "branch": body.branch,
            "base": body.base,
            "existingBranch": body.existingBranch,
        }.items()
        if value
    }
    return await _git_op(_web_git.worktree_add, _git_path(body.path), options)


@app.post("/api/git/worktree/remove")
async def git_worktree_remove_route(body: GitWorktreeRemoveBody):
    return await _git_op(
        _web_git.worktree_remove, _git_path(body.path), _git_path(body.worktreePath), body.force
    )


@app.post("/api/git/branch/switch")
async def git_branch_switch_route(body: GitBranchSwitchBody):
    return await _git_op(_web_git.branch_switch, _git_path(body.path), body.branch)


# Host TCP ports each port-binding gateway platform listens on, as
# ``platform-name -> (config port key, adapter default)``.  Mirrors
# ``_PORT_BINDING_PLATFORM_VALUES`` in gateway/run.py and each adapter's
# DEFAULT_PORT / DEFAULT_WEBHOOK_PORT constant.  Used only for the dashboard's
# gateway-topology readout — best-effort display data, not a bind source.
_PORT_BINDING_PLATFORM_PORTS: Dict[str, Tuple[str, int]] = {
    "webhook": ("port", 8644),
    "api_server": ("port", 8642),
    "msgraph_webhook": ("port", 8646),
    "feishu": ("webhook_port", 8765),
    "wecom_callback": ("port", 8645),
    "bluebubbles": ("webhook_port", 8645),
    "sms": ("webhook_port", 8080),
    "whatsapp_cloud": ("webhook_port", 8090),
    "line": ("port", 8646),
}

# Platform states that mean the adapter is NOT serving its port right now.
_PLATFORM_DEAD_STATES = frozenset({"fatal", "disconnected", "stopped"})


def _profile_platform_ports(profile_home: Path, runtime: Optional[dict]) -> Dict[str, int]:
    """Best-effort map of ``platform -> host TCP port`` for one profile's gateway.

    Reads the platforms the running gateway reported in its
    ``gateway_state.json`` and resolves each port-binding platform's port from
    the profile's ``config.yaml`` (top-level ``platforms:`` wins over
    ``gateway.platforms:``, matching ``load_gateway_config`` precedence),
    falling back to the adapter default.  Display-only: env-var port overrides
    (e.g. ``WEBHOOK_PORT`` in that profile's .env) are not resolved here.
    """
    platforms = (runtime or {}).get("platforms") or {}
    active = [
        name for name, state in platforms.items()
        if name in _PORT_BINDING_PLATFORM_PORTS
        and isinstance(state, dict)
        and state.get("state") not in _PLATFORM_DEAD_STATES
    ]
    if not active:
        return {}

    blocks: Dict[str, dict] = {}
    try:
        with open(profile_home / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        gateway_cfg = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
        # gateway.platforms first, top-level platforms second — later wins,
        # matching the precedence in gateway.config.load_gateway_config().
        for src in ((gateway_cfg or {}).get("platforms"), cfg.get("platforms")):
            if not isinstance(src, dict):
                continue
            for plat_name, plat_block in src.items():
                if isinstance(plat_block, dict):
                    blocks.setdefault(plat_name, {}).update(plat_block)
    except Exception:
        blocks = {}

    ports: Dict[str, int] = {}
    for name in active:
        port_key, default_port = _PORT_BINDING_PLATFORM_PORTS[name]
        block = blocks.get(name) or {}
        extra = block.get("extra") if isinstance(block.get("extra"), dict) else {}
        raw = block.get(port_key, (extra or {}).get(port_key, default_port))
        try:
            ports[name] = int(raw)
        except (TypeError, ValueError):
            ports[name] = default_port
    return ports


def _collect_profile_gateway_topology() -> Dict[str, Any]:
    """Enumerate profiles and the gateways serving them for ``/api/status``.

    Returns ``{"profiles": [...], "gateway_mode": ..., "gateways": [...]}``:

    * ``profiles`` — every profile on the host (default + named), from
      ``profiles_to_serve(True)`` (the cheap enumeration chokepoint — no
      per-profile config reads or skill counts).
    * ``gateways`` — one entry per profile with a LIVE gateway process:
      ``{"profile", "ports", "served_profiles"?}``.  Liveness reuses
      ``_check_gateway_running`` so this agrees with the profiles sidebar.
    * ``gateway_mode`` — ``"multiplex"`` when the default gateway serves
      multiple profiles (gateway.multiplex_profiles), ``"single"`` for one
      live gateway, ``"multiple"`` for independent per-profile gateways,
      ``"none"`` when nothing is running.
    """
    try:
        from hermes_cli.profiles import _check_gateway_running, profiles_to_serve
        from gateway.status import read_runtime_status
        homes = profiles_to_serve(True)
    except Exception:
        _log.debug("profile/gateway topology enumeration failed", exc_info=True)
        return {"profiles": [], "gateway_mode": "unknown", "gateways": []}

    profile_names = [name for name, _home in homes]
    gateways: List[Dict[str, Any]] = []
    multiplex = False
    for name, home in homes:
        try:
            if not _check_gateway_running(home):
                continue
        except Exception:
            continue
        try:
            runtime = read_runtime_status(home / "gateway_state.json")
        except Exception:
            runtime = None
        served = [str(p) for p in ((runtime or {}).get("served_profiles") or [])]
        if name == "default" and len(served) > 1:
            multiplex = True
        entry: Dict[str, Any] = {
            "profile": name,
            "ports": _profile_platform_ports(home, runtime),
        }
        if served:
            entry["served_profiles"] = served
        gateways.append(entry)

    if multiplex:
        mode = "multiplex"
    elif len(gateways) > 1:
        mode = "multiple"
    elif len(gateways) == 1:
        mode = "single"
    else:
        mode = "none"

    return {"profiles": profile_names, "gateway_mode": mode, "gateways": gateways}


@app.get("/api/status")
async def get_status(profile: Optional[str] = None):
    status_scope = None
    requested_profile = (profile or "").strip()
    # Plain /api/status stays the machine-level public liveness probe. The
    # dashboard adds ?profile= when its management switcher targets another
    # profile, so its gateway badge reflects the selected profile.
    #
    # Use the config-only (contextvar) scope, NOT _profile_scope: this handler
    # awaits the remote-health probe, and _profile_scope swaps process-global
    # skills-module attributes that a concurrent request would cross-restore
    # across that await. Status only resolves get_hermes_home() at call time
    # (config/env/gateway state), which the task-local contextvar covers.
    if requested_profile and requested_profile.lower() != "current":
        status_scope = _config_profile_scope(requested_profile)
        status_scope.__enter__()

    try:
        current_ver, latest_ver = check_config_version()
        # --- Gateway liveness detection ---
        # Try local PID check first (same-host).  If that fails and a remote
        # GATEWAY_HEALTH_URL is configured, probe the gateway over HTTP so the
        # dashboard works when the gateway runs in a separate container.
        gateway_pid = get_running_pid_cached()
        gateway_running = gateway_pid is not None
        remote_health_body: dict | None = None

        if not gateway_running and _GATEWAY_HEALTH_URL:
            loop = asyncio.get_running_loop()
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
        local_runtime = read_runtime_status()
        runtime = local_runtime
        if runtime is None and remote_health_body and remote_health_body.get("gateway_state"):
            runtime = remote_health_body
        # The runtime-status PID fallback validates liveness with a local
        # os.kill() probe, so it must only run against the LOCAL status file —
        # never the remote health body, whose PID belongs to another host and
        # is display-only. (Running os.kill on a remote PID is both wrong and
        # trips the test live-system guard.)
        if not gateway_running and local_runtime is not None:
            runtime_pid = get_runtime_status_running_pid(local_runtime)
            if runtime_pid is not None:
                gateway_running = True
                gateway_pid = runtime_pid

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
                gateway_state = gateway_state if gateway_state in {"stopped", "startup_failed"} else "stopped"
                gateway_platforms = {}
            elif gateway_running and remote_health_body is not None:
                # The health probe confirmed the gateway is alive, but the local
                # runtime status file may be stale (cross-container).  Override
                # stopped/None state so the dashboard shows the correct badge.
                if gateway_state in {None, "stopped"}:
                    gateway_state = "running"

        # If there was no runtime info at all but the health probe confirmed alive,
        # ensure we still report the gateway as running (no shared volume scenario).
        if gateway_running and gateway_state is None and remote_health_body is not None:
            gateway_state = "running"

        active_sessions = await _status_active_sessions()

        # Busy/drainable readout (NAS lifecycle-safety gate).  active_agents is
        # the in-flight gateway-turn count the gateway now persists at every
        # turn boundary; gateway_busy/gateway_drainable are derived from it +
        # liveness via the single shared contract in gateway.status.  Liveness
        # keys off gateway_running (a live PID/health probe), NEVER
        # gateway_updated_at — a healthy idle gateway never advances that.
        active_agents = parse_active_agents((runtime or {}).get("active_agents", 0))
        gateway_busy = derive_gateway_busy(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
            active_agents=active_agents,
        )
        gateway_drainable = derive_gateway_drainable(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
        )
        # Resolved drain timeout (seconds) so NAS can size its poll deadline
        # without out-of-band knowledge.  Offload to a thread: on a cold
        # Windows install the first import of hermes_cli.gateway blocks the
        # asyncio event loop for 15-30s (.pyc compilation + Defender scans),
        # exceeding the desktop handshake's 15s socket timeout.  After the
        # first call the module is in sys.modules and run_in_executor returns
        # in microseconds.
        restart_drain_timeout = await asyncio.get_running_loop().run_in_executor(
            None, _resolve_restart_drain_timeout
        )

        # Dashboard auth gate (Phase 7): surface whether the gate is engaged
        # and which providers are registered so ``hermes status`` and the
        # SPA's StatusPage can show "OAuth gate ON via Nous Research" or
        # "loopback only — no auth gate" with no extra round trips.
        auth_required = bool(getattr(app.state, "auth_required", False))
        auth_providers: list[str] = []
        try:
            from hermes_cli.dashboard_auth import list_providers as _list_providers
            auth_providers = [p.name for p in _list_providers()]
        except Exception:
            # Module not importable yet (early startup) — leave as [].
            pass

        # Nous bootstrap-session validity for the NAS health sweep. A hosted
        # agent whose Nous auth dies terminally (invalid_grant / quarantine)
        # looks HEALTHY to every liveness/connectivity probe — the machine,
        # relay, and this dashboard all stay up — yet every inference turn
        # fails. This is the ONLY signal that surfaces that condition, and it
        # is determinable with no working token (local auth-store state). NAS
        # re-mints the bootstrap session when it reads "terminal". Best-effort:
        # never let auth classification break the public liveness probe.
        nous_session_valid = "unknown"
        try:
            from hermes_cli.auth import get_nous_session_validity
            nous_session_valid = get_nous_session_validity()
        except Exception:
            nous_session_valid = "unknown"

        # Always-public liveness + auth-gate shape. Safe for external uptime
        # probes (NAS's wildcard-subdomain liveness probe), the SPA's pre-login
        # bootstrap, and anyone who can curl the host — i.e. exactly the audience
        # ``PUBLIC_API_PATHS`` documents this endpoint as serving.
        status = {
            "version": __version__,
            "release_date": __release_date__,
            "config_version": current_ver,
            "latest_config_version": latest_ver,
            "can_update_hermes": not _dashboard_local_update_managed_externally(),
            "gateway_running": gateway_running,
            "gateway_state": gateway_state,
            "gateway_platforms": gateway_platforms,
            "gateway_exit_reason": gateway_exit_reason,
            "gateway_updated_at": gateway_updated_at,
            "active_agents": active_agents,
            "gateway_busy": gateway_busy,
            "gateway_drainable": gateway_drainable,
            "restart_drain_timeout": restart_drain_timeout,
            "active_sessions": active_sessions,
            "auth_required": auth_required,
            "auth_providers": auth_providers,
            "nous_session_valid": nous_session_valid,
        }

        # Profile + gateway topology: which profiles exist, whether one
        # multiplexed gateway or several per-profile gateways serve them, and
        # (gated) which host ports the live gateways' port-binding platforms
        # listen on.  Enumerating profiles walks the filesystem and probes the
        # process table, so keep it off the event loop.
        #
        # Split by sensitivity: profile NAMES (``profiles``) and the gateway
        # ``gateway_mode`` are low-sensitivity PRODUCT surface — Hermes Cloud
        # renders the profile list in the Portal, which reads this endpoint over
        # the network (a gated bind), so they must survive the auth gate. The
        # per-gateway ``gateways[]`` detail carries host ports (deployment
        # recon), so it stays gated with the host paths / PID below.
        topology = await asyncio.get_running_loop().run_in_executor(
            None, _collect_profile_gateway_topology
        )
        status["profiles"] = topology["profiles"]
        status["gateway_mode"] = topology["gateway_mode"]

        # Absolute host paths, the gateway PID, the internal gateway health
        # URL, and per-gateway ports are deployment recon a liveness probe never
        # needs. ``/api/status`` is in ``PUBLIC_API_PATHS`` so it bypasses
        # dashboard auth; on a network-exposed (gated) bind that means *any*
        # unauthenticated caller reaches it, and leaking host metadata there
        # contradicts the allowlist's own contract ("version, gateway state,
        # active session count, and the dashboard auth-gate shape. No bodies, no
        # session content, no secrets"). Surface this detail only on a loopback
        # / ``--insecure`` bind, where the dashboard is local-only and the
        # caller is already inside the trust envelope — the same loopback/gated
        # split ``should_require_auth`` draws.
        if not auth_required:
            status.update({
                "hermes_home": str(get_hermes_home()),
                "config_path": str(get_config_path()),
                "env_path": str(get_env_path()),
                "gateway_pid": gateway_pid,
                "gateway_health_url": _GATEWAY_HEALTH_URL,
                "gateways": topology["gateways"],
            })

        return status
    finally:
        if status_scope is not None:
            status_scope.__exit__(*sys.exc_info())


_WINDOWS_11_MIN_BUILD = 22000


def _windows_build_number(version: str, platform_label: str) -> Optional[int]:
    """Extract the Windows NT build number from stdlib platform strings."""
    for value in (version or "", platform_label or ""):
        match = re.search(r"(?:^|[^\d])10\.0\.(\d{5,})(?:[^\d]|$)", value)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def _display_system_platform(
    *,
    system: str,
    release: str,
    version: str,
    platform_label: str,
) -> Dict[str, str]:
    """Return host OS fields for display while preserving stdlib detail."""
    if system == "Windows" and release == "10":
        build = _windows_build_number(version, platform_label)
        if build is not None and build >= _WINDOWS_11_MIN_BUILD:
            platform_label = re.sub(
                r"^Windows-10(?=-)",
                "Windows-11",
                platform_label,
                count=1,
            )
            release = "11"

    return {
        "os": system,
        "os_release": release,
        "os_version": version,
        "platform": platform_label,
    }


@app.get("/api/system/stats")
async def get_system_stats():
    """Host + process system stats for the System page.

    OS / Python / host identity from stdlib; CPU / memory / disk / uptime from
    psutil when available, with graceful degradation when it isn't.  Read-only
    and non-sensitive (no env values, no paths beyond the hermes home root).
    """
    import platform as _platform

    info: Dict[str, Any] = {
        **_display_system_platform(
            system=_platform.system(),
            release=_platform.release(),
            version=_platform.version(),
            platform_label=_platform.platform(),
        ),
        "arch": _platform.machine(),
        "hostname": _platform.node(),
        "python_version": _platform.python_version(),
        "python_impl": _platform.python_implementation(),
        "hermes_version": __version__,
        "cpu_count": os.cpu_count(),
    }

    # psutil enriches the picture when present; everything below is optional.
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["memory"] = {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        }
        try:
            du = psutil.disk_usage(str(get_hermes_home()))
            info["disk"] = {
                "total": du.total,
                "used": du.used,
                "free": du.free,
                "percent": du.percent,
            }
        except Exception:
            pass
        try:
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            la = getattr(psutil, "getloadavg", None)
            if la:
                info["load_avg"] = list(la())
        except Exception:
            pass
        try:
            boot = psutil.boot_time()
            info["uptime_seconds"] = int(time.time() - boot)
        except Exception:
            pass
        try:
            proc = psutil.Process()
            info["process"] = {
                "pid": proc.pid,
                "rss": proc.memory_info().rss,
                "create_time": int(proc.create_time()),
                "num_threads": proc.num_threads(),
            }
        except Exception:
            pass
        info["psutil"] = True
    except Exception:
        info["psutil"] = False
        # stdlib-only fallbacks for load average + uptime where the kernel
        # exposes them.
        try:
            info["load_avg"] = list(os.getloadavg())
        except (OSError, AttributeError):
            pass

    return info


# ---------------------------------------------------------------------------
# Curator endpoints — background skill-maintenance status + controls.
#
# The curator periodically reviews skills (archive stale, prune, pin).  The
# dashboard surfaces its state and the pause/resume/run-now controls that
# `hermes curator` exposes.
# ---------------------------------------------------------------------------


@app.get("/api/curator")
async def get_curator_status():
    try:
        from agent import curator
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Curator unavailable: {exc}")
    try:
        state = curator.load_state()
    except Exception:
        state = {}
    return {
        "enabled": _safe_call(curator, "is_enabled", True),
        "paused": _safe_call(curator, "is_paused", False),
        "interval_hours": _safe_call(curator, "get_interval_hours", None),
        "last_run_at": state.get("last_run_at"),
        "min_idle_hours": _safe_call(curator, "get_min_idle_hours", None),
        "stale_after_days": _safe_call(curator, "get_stale_after_days", None),
        "archive_after_days": _safe_call(curator, "get_archive_after_days", None),
    }


class CuratorPause(BaseModel):
    paused: bool


@app.put("/api/curator/paused")
async def set_curator_paused(body: CuratorPause):
    from agent import curator

    curator.set_paused(bool(body.paused))
    return {"ok": True, "paused": bool(body.paused)}


@app.post("/api/curator/run")
async def run_curator():
    """Trigger a curator review now (backgrounded; tail via action status)."""
    try:
        proc = _spawn_hermes_action(["curator", "run"], "curator-run")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run curator: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "curator-run"}


@app.get("/api/learning/graph")
async def get_learning_graph(profile: Optional[str] = None):
    """Learning graph payload for the desktop panel.

    Profile-scoped view of learned, non-base skills plus memory chunks, with
    graph links derived from skill relations and memory-skill overlap.
    """
    try:
        from agent.learning_graph import build_learning_graph

        with _profile_scope(profile):
            return build_learning_graph()
    except Exception:
        _log.exception("GET /api/learning/graph failed")
        raise HTTPException(status_code=500, detail="Failed to build learning graph")


class LearningNodeRef(BaseModel):
    id: str
    profile: Optional[str] = None


class LearningNodeEdit(BaseModel):
    id: str
    content: str
    profile: Optional[str] = None


@app.get("/api/learning/node")
async def get_learning_node(id: str, profile: Optional[str] = None):
    """Current content of a journey node (skill SKILL.md or memory chunk), for an edit prefill."""
    from agent.learning_mutations import node_detail

    with _profile_scope(profile):
        res = node_detail(id)
    if not res.get("ok"):
        raise HTTPException(status_code=404, detail=res.get("message", "not found"))
    return res


@app.delete("/api/learning/node")
async def delete_learning_node(body: LearningNodeRef):
    """Delete a journey node — skills are archived (restorable), memories removed."""
    from agent.learning_mutations import delete_node

    with _profile_scope(body.profile):
        res = delete_node(body.id)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("message", "delete failed"))
    return res


@app.put("/api/learning/node")
async def update_learning_node(body: LearningNodeEdit):
    """Rewrite a journey node's content (SKILL.md or memory chunk)."""
    from agent.learning_mutations import edit_node

    with _profile_scope(body.profile):
        res = edit_node(body.id, body.content)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("message", "edit failed"))
    return res


def _safe_call(mod, fn_name: str, default):
    try:
        fn = getattr(mod, fn_name, None)
        return fn() if callable(fn) else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Portal endpoint — Nous Portal auth + Tool Gateway routing status (read-only).
# ---------------------------------------------------------------------------


@app.get("/api/portal")
async def get_portal_status():
    cfg = load_config() or {}
    auth: Dict[str, Any] = {}
    try:
        from hermes_cli.auth import get_nous_auth_status

        auth = get_nous_auth_status() or {}
    except Exception:
        auth = {}

    features = []
    try:
        from hermes_cli.nous_subscription import get_nous_subscription_features

        feats = get_nous_subscription_features(cfg)
        if feats is not None:
            for feat in feats.items():
                if getattr(feat, "managed_by_nous", False):
                    state = "via Nous Portal"
                elif getattr(feat, "active", False) and getattr(feat, "current_provider", None):
                    state = feat.current_provider
                elif getattr(feat, "active", False):
                    state = "active"
                else:
                    state = "not configured"
                features.append({"label": getattr(feat, "label", ""), "state": state})
    except Exception:
        _log.exception("portal features failed")

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return {
        "logged_in": bool(auth.get("logged_in")),
        "portal_url": auth.get("portal_base_url"),
        "inference_url": auth.get("inference_base_url"),
        "provider": str((model_cfg or {}).get("provider") or ""),
        "subscription_url": "https://portal.nousresearch.com/manage-subscription",
        "features": features,
    }


# ---------------------------------------------------------------------------
# Diagnostics: prompt-size, support dump, debug upload, config migrate.
# All produce text output, so they spawn background actions tailed via
# /api/actions/<name>/status.
# ---------------------------------------------------------------------------


@app.post("/api/ops/prompt-size")
async def run_prompt_size():
    try:
        proc = _spawn_hermes_action(["prompt-size"], "prompt-size")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "prompt-size"}


@app.post("/api/ops/dump")
async def run_dump():
    try:
        proc = _spawn_hermes_action(["dump"], "dump")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "dump"}


@app.post("/api/ops/config-migrate")
async def run_config_migrate():
    try:
        proc = _spawn_hermes_action(["config", "migrate"], "config-migrate")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "config-migrate"}


class DebugShareRequest(BaseModel):
    # Redaction is ON by default — force-mode scrubs credential-shaped tokens
    # out of log content before it leaves the machine. The toggle exists so an
    # operator who knows the logs are clean can opt out for fuller fidelity.
    redact: bool = True
    # Recent log lines included in the summary tail (full logs are separate).
    lines: int = 200


@app.post("/api/ops/debug-share")
async def run_debug_share_endpoint(body: DebugShareRequest | None = None):
    """Upload a redacted debug report + full logs and return the paste URLs.

    Unlike the other diagnostics actions (doctor, dump, prompt-size) this is
    *synchronous*: the whole point of ``debug share`` is the set of shareable
    URLs it produces, so we run the upload in a worker thread and return the
    structured ``{urls, failures, redacted, ...}`` payload directly. The
    dashboard renders those as real, copyable links instead of scraping a log
    tail. Pastes auto-delete after 6 hours (handled inside the share core).
    """
    from hermes_cli.debug import build_debug_share

    req = body or DebugShareRequest()
    try:
        result = await asyncio.to_thread(
            build_debug_share,
            log_lines=max(1, min(int(req.lines), 5000)),
            redact=bool(req.redact),
        )
    except RuntimeError as exc:
        # Required summary-report upload failed (offline / paste service down).
        raise HTTPException(status_code=502, detail=f"Upload failed: {exc}")
    except Exception as exc:
        _log.exception("debug share failed")
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")

    return {
        "ok": True,
        "urls": result.urls,
        "failures": result.failures,
        "redacted": result.redacted,
        "auto_delete_seconds": result.auto_delete_seconds,
    }


# ---------------------------------------------------------------------------
# Gateway + update actions (invoked from the Status page).
#
# Both commands are spawned as detached subprocesses so the HTTP request
# returns immediately.  stdin is closed (``DEVNULL``) so any stray ``input()``
# calls fail fast with EOF rather than hanging forever.  stdout/stderr are
# streamed to a per-action log file under ``~/.hermes/logs/<action>.log`` so
# the dashboard can tail them back to the user.
# ---------------------------------------------------------------------------

_ACTION_LOG_DIR: Path = get_hermes_home() / "logs"
_ACTION_LOG_TAIL_MAX_BYTES = 256 * 1024
_ACTION_LOG_TAIL_INITIAL_CHUNK_BYTES = 8 * 1024
_ACTION_LOG_TAIL_MAX_CHUNK_BYTES = 64 * 1024

# Short ``name`` (from the URL) → absolute log file path.
_ACTION_LOG_FILES: Dict[str, str] = {
    "gateway-restart": "gateway-restart.log",
    "gateway-start": "gateway-start.log",
    "gateway-stop": "gateway-stop.log",
    "hermes-update": "hermes-update.log",
    "doctor": "action-doctor.log",
    "security-audit": "action-security-audit.log",
    "backup": "action-backup.log",
    "import": "action-import.log",
    "checkpoints-prune": "action-checkpoints-prune.log",
    "skills-install": "action-skills-install.log",
    "skills-uninstall": "action-skills-uninstall.log",
    "skills-update": "action-skills-update.log",
    "curator-run": "action-curator-run.log",
    "prompt-size": "action-prompt-size.log",
    "dump": "action-dump.log",
    "config-migrate": "action-config-migrate.log",
    "tools-post-setup": "action-tools-post-setup.log",
}

# ``name`` → most recently spawned Popen handle.  Used so ``status`` can
# report liveness and exit code without shelling out to ``ps``.
_ACTION_PROCS: Dict[str, subprocess.Popen] = {}
_ACTION_COMMANDS: Dict[str, Tuple[str, ...]] = {}

# ``name`` → completed synthetic action result for actions the server handled
# without spawning a subprocess (for example, unsupported Docker updates).
_ACTION_RESULTS: Dict[str, Dict[str, Any]] = {}


def _record_completed_action(name: str, message: str, exit_code: int = 1) -> None:
    """Record a non-spawned action result and write it to the action log."""
    log_file_name = _ACTION_LOG_FILES[name]
    _ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _ACTION_LOG_DIR / log_file_name
    with open(log_path, "ab", buffering=0) as log_file:
        log_file.write(
            f"\n=== {name} completed {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
        )
        log_file.write(message.encode("utf-8", errors="replace"))
        if not message.endswith("\n"):
            log_file.write(b"\n")
    _ACTION_PROCS.pop(name, None)
    _ACTION_COMMANDS.pop(name, None)
    _ACTION_RESULTS[name] = {"exit_code": exit_code, "pid": None}


def _dashboard_spawn_executable() -> str:
    """Prefer pythonw.exe for detached dashboard actions on Windows."""
    if sys.platform != "win32":
        return sys.executable
    exe = sys.executable
    if exe.lower().endswith("python.exe"):
        pythonw = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.isfile(pythonw):
            return pythonw
    return exe


def _spawn_hermes_action(subcommand: List[str], name: str) -> subprocess.Popen:
    """Spawn ``hermes <subcommand>`` detached and record the Popen handle.

    Uses the running interpreter's ``hermes_cli.main`` module so the action
    inherits the same venv/PYTHONPATH the web server is using.
    """
    log_file_name = _ACTION_LOG_FILES[name]
    _ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _ACTION_LOG_DIR / log_file_name
    log_file = open(log_path, "ab", buffering=0)
    log_file.write(
        f"\n=== {name} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
    )

    cmd = [_dashboard_spawn_executable(), "-m", "hermes_cli.main", *subcommand]

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": {**os.environ, "HERMES_NONINTERACTIVE": "1"},
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = windows_detach_flags()
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    # The child inherits its own duplicated fd for stdout/stderr, so the
    # parent's handle can be released immediately — otherwise we leak one
    # fd per spawned action.
    log_file.close()
    _ACTION_RESULTS.pop(name, None)
    _ACTION_COMMANDS[name] = tuple(subcommand)
    _ACTION_PROCS[name] = proc
    return proc


def _tail_lines(path: Path, n: int) -> List[str]:
    """Return the last ``n`` lines of ``path`` without loading huge logs."""
    if n <= 0 or not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size <= 0:
        return []

    min_offset = max(0, size - _ACTION_LOG_TAIL_MAX_BYTES)
    offset = size
    chunk_size = _ACTION_LOG_TAIL_INITIAL_CHUNK_BYTES
    newline_count = 0
    chunks: List[bytes] = []
    drop_partial_first_line = False

    try:
        with path.open("rb") as handle:
            while offset > min_offset and newline_count <= n:
                read_size = min(chunk_size, offset - min_offset)
                offset -= read_size
                handle.seek(offset)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
                chunk_size = min(
                    chunk_size * 2,
                    _ACTION_LOG_TAIL_MAX_CHUNK_BYTES,
                )
            if offset > 0:
                handle.seek(offset - 1)
                drop_partial_first_line = handle.read(1) != b"\n"
    except OSError:
        return []

    lines = (
        b"".join(reversed(chunks))
        .decode("utf-8", errors="replace")
        .splitlines()
    )
    if drop_partial_first_line and lines:
        lines = lines[1:]
    return lines[-n:]


def _gateway_subcommand(profile: Optional[str], verb: str) -> List[str]:
    return _profile_cli_args(profile) + ["gateway", verb]


def _gateway_display_command(profile: Optional[str], verb: str) -> str:
    return " ".join(["hermes", *_gateway_subcommand(profile, verb)])


# Slack member IDs (users U..., Enterprise Grid W...). Kept in sync with the
# frontend SLACK_MEMBER_ID_RE in web/src/pages/ChannelsPage.tsx.
_SLACK_MEMBER_ID_RE = re.compile(r"[UW][A-Z0-9]{2,}")


def _validate_messaging_env_value(platform_id: str, key: str, value: str) -> None:
    """Reject platform credentials that are clearly in the wrong field."""
    if platform_id != "slack" or not value:
        return

    if key == "SLACK_BOT_TOKEN" and not value.startswith("xoxb-"):
        raise HTTPException(
            status_code=400,
            detail="Slack Bot Token must start with xoxb-. Paste the bot token from OAuth & Permissions.",
        )
    if key == "SLACK_APP_TOKEN" and not value.startswith("xapp-"):
        raise HTTPException(
            status_code=400,
            detail="Slack App Token must start with xapp-. Paste the app-level token from Basic Information > App-Level Tokens.",
        )
    if key == "SLACK_ALLOWED_USERS":
        # Mirror the gateway's parse (gateway/platforms/slack.py): split on comma,
        # strip, and drop empty entries so a trailing/interior comma isn't rejected
        # here when the runtime would accept it. "*" is the allow-all wildcard.
        user_ids = [part.strip() for part in value.split(",") if part.strip()]
        invalid = [
            user_id
            for user_id in user_ids
            if user_id != "*" and not _SLACK_MEMBER_ID_RE.fullmatch(user_id)
        ]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail="Slack allowed user IDs must be comma-separated member IDs like U01ABC2DEF3.",
            )


def _spawn_gateway_restart(profile: Optional[str] = None) -> Tuple[subprocess.Popen, bool]:
    """Spawn ``hermes gateway restart``, reusing an in-flight restart.

    Multiple dashboard paths can request a restart in quick succession
    (restart button double-click, or a stale cached frontend firing its own
    restart after the server already auto-restarted post-onboarding). Two
    concurrent ``hermes gateway restart`` children race each other on the
    manual kill-and-start path, so reuse the live one instead.

    Returns ``(proc, reused)``.
    """
    subcommand = _gateway_subcommand(profile, "restart")
    existing = _ACTION_PROCS.get("gateway-restart")
    if existing is not None and existing.poll() is None:
        existing_command = _ACTION_COMMANDS.get("gateway-restart")
        if existing_command is None or existing_command == tuple(subcommand):
            return existing, True
        raise RuntimeError("gateway restart already in progress for another profile")
    return _spawn_hermes_action(subcommand, "gateway-restart"), False


def _restart_gateway_after_webhook_enable(profile: Optional[str] = None) -> dict[str, Any]:
    """Best-effort gateway restart after enabling the webhook platform."""
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after enabling webhooks")
        return {
            "restart_started": False,
            "restart_error": str(exc),
        }
    if reused:
        _log.info(
            "Webhook enable: reusing in-flight gateway restart (pid %s)",
            proc.pid,
        )
    return {
        "restart_started": True,
        "restart_action": "gateway-restart",
        "restart_pid": proc.pid,
    }


@app.post("/api/gateway/restart")
async def restart_gateway(profile: Optional[str] = None):
    """Kick off a ``hermes gateway restart`` in the background."""
    try:
        proc, _reused = _spawn_gateway_restart(profile)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway restart")
        raise HTTPException(status_code=500, detail=f"Failed to restart gateway: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "gateway-restart",
    }


@app.post("/api/gateway/drain")
async def gateway_drain(request: Request):
    """Begin or cancel an external (NAS-driven) gateway drain.

    Authenticated by the non-interactive token-auth seam: the
    ``dashboard_auth/drain`` plugin registers this exact path as a token route
    and verifies the ``Authorization`` bearer secret. If that plugin isn't
    active (no ``HERMES_DASHBOARD_DRAIN_SECRET``), the route is NOT a token
    route, so on a gated bind the cookie gate handles it (a browser session can
    still drive it from the dashboard) and on a loopback bind the legacy
    session-token gate applies — either way it is never unauthenticated on a
    network-exposed bind.

    Body: ``{"action": "drain"}`` (begin) or ``{"action": "cancel"}`` (cancel).
    Begin writes the ``.drain_request.json`` marker the gateway's
    ``_drain_control_watcher`` observes (flip to ``draining`` + refuse new
    turns); cancel removes it (revert to ``running`` + re-accept). Idempotent
    on both sides. This endpoint only writes/removes the marker — the gateway
    process owns the actual state transition (there is no HTTP control channel
    into the running gateway; the marker IS the channel, decisions.md Q-B).

    The force-override (D6: "unless a user commands it") is NOT here — an
    immediate, drain-skipping action maps onto the existing
    ``POST /api/gateway/restart`` force path, which supersedes a drain.
    """
    from gateway.drain_control import (
        clear_drain_request,
        drain_requested,
        write_drain_request,
    )

    try:
        body = await request.json()
    except Exception:
        body = {}
    action = str((body or {}).get("action", "drain")).strip().lower()

    # Attribute the request to the verified token principal when present
    # (token-auth seam attaches it); fall back to a generic label otherwise.
    principal_obj = getattr(request.state, "token_principal", None)
    principal = getattr(principal_obj, "principal", None) or "dashboard"

    if action == "cancel":
        existed = clear_drain_request()
        _log.info("Gateway drain CANCEL requested by %s (existed=%s)", principal, existed)
        return {"ok": True, "action": "cancel", "was_draining": existed}

    if action != "drain":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown drain action {action!r}; expected 'drain' or 'cancel'",
        )

    payload = write_drain_request(
        principal=str(principal),
        suppress_notification=bool((body or {}).get("suppress_notification", False)),
    )
    _log.info(
        "Gateway drain BEGIN requested by %s (suppress_notification=%s)",
        principal,
        payload["suppress_notification"],
    )
    return {
        "ok": True,
        "action": "drain",
        "requested_at": payload["requested_at"],
        # Echo so a caller polling /api/status knows the marker is now set;
        # the gateway watcher flips gateway_state -> draining within ~1s.
        "draining": drain_requested(),
        "suppress_notification": payload["suppress_notification"],
    }


@app.post("/api/hermes/update")
async def update_hermes():
    """Kick off ``hermes update`` in the background."""
    if _dashboard_local_update_managed_externally():
        message = (
            "Hermes updates are managed outside this dashboard in "
            "containerized environments. The built-in local updater is "
            "disabled here."
        )
        _record_completed_action("hermes-update", message, exit_code=1)
        return {
            "ok": False,
            "pid": None,
            "name": "hermes-update",
            "error": "dashboard_update_managed_externally",
            "message": message,
            "update_command": "managed outside dashboard",
        }

    install_method = detect_install_method(PROJECT_ROOT)
    if install_method == "docker":
        message = format_docker_update_message()
        _record_completed_action("hermes-update", message, exit_code=1)
        return {
            "ok": False,
            "pid": None,
            "name": "hermes-update",
            "error": "docker_update_unsupported",
            "message": message,
            "update_command": recommended_update_command_for_method(install_method),
        }

    try:
        proc = _spawn_hermes_action(["update"], "hermes-update")
    except Exception as exc:
        _log.exception("Failed to spawn hermes update")
        raise HTTPException(status_code=500, detail=f"Failed to start update: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "hermes-update",
    }


def _recent_upstream_commits(n: int = 20) -> List[Dict[str, Any]]:
    """Commits the local checkout is behind ``origin/main`` by, newest first.

    Logs the SAME range the behind-count uses (``HEAD..origin/main`` — see
    ``banner._check_via_local_git``), NOT the branch's ``@{upstream}``. On a
    feature-branch checkout ``@{upstream}`` is the branch's own tip (zero
    commits), which would leave the changelog empty even though the count is
    non-zero. Pinning to ``origin/main`` keeps count and changelog consistent.

    Best-effort: returns [] if not a git checkout, origin/main is unreachable,
    or git is unavailable. Never raises into the request path.
    """
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "log",
                "--format=%H%x1f%s%x1f%an%x1f%ct",
                "HEAD..origin/main",
                f"-n{int(n)}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return []
        rows: List[Dict[str, Any]] = []
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            parts = (line.split("\x1f") + ["", "", "", "0"])[:4]
            sha, summary, author, at = parts
            rows.append(
                {
                    "sha": sha[:7],
                    "summary": summary,
                    "author": author,
                    "at": int(at or 0),
                }
            )
        return rows
    except Exception:
        return []


@app.get("/api/hermes/update/check")
async def check_hermes_update(force: bool = False):
    """Report whether a Hermes update is available, without applying it.

    Powers the dashboard's "check before you update" flow: the System page
    shows the commit-behind count and asks the user to confirm before
    ``POST /api/hermes/update`` actually runs ``hermes update``.

    Returns:
        install_method: 'git' | 'pip' | 'docker' | 'nixos' | 'homebrew' | ...
        current_version: installed Hermes version string
        behind: commits behind upstream (>=1), 0 if up to date,
                -1 if behind by an unknown count (nix/pypi), or null if the
                check could not run (offline, no remote, etc.)
        update_available: convenience bool (behind is non-zero and not null)
        can_apply: True when the dashboard's update button can apply it
                   in place (git/pip); False for docker/nix/homebrew where the
                   user must update out-of-band
        update_command: the recommended command for this install method
        message: human-readable guidance for non-applyable methods
        commits: for git/pip installs that are behind, a list of the commits
                 the local checkout is behind upstream by — each
                 {sha, summary, author, at}. Absent/empty otherwise. The
                 desktop's remote update overlay renders this as "what's
                 changed". Additive: existing consumers ignore it.
    """
    if _dashboard_local_update_managed_externally():
        return {
            "install_method": "managed-runtime",
            "current_version": __version__,
            "behind": None,
            "update_available": False,
            "can_apply": False,
            "update_command": "managed outside dashboard",
            "message": (
                "Hermes updates are managed outside this dashboard in "
                "containerized environments."
            ),
        }

    install_method = detect_install_method(PROJECT_ROOT)
    update_command = recommended_update_command_for_method(install_method)

    payload: Dict[str, Any] = {
        "install_method": install_method,
        "current_version": __version__,
        "behind": None,
        "update_available": False,
        "can_apply": install_method in ("git", "pip"),
        "update_command": update_command,
        "message": None,
    }

    if install_method == "docker":
        payload["message"] = format_docker_update_message()
        return payload

    # banner.check_for_updates() handles git / pypi / nix-revision paths and
    # caches the result for 6h. ``force`` busts the cache so the "Check now"
    # button reflects reality immediately.
    try:
        from hermes_cli.banner import check_for_updates

        if force:
            try:
                (get_hermes_home() / ".update_check").unlink()
            except OSError:
                pass

        behind = await asyncio.to_thread(check_for_updates)
    except Exception:
        _log.exception("Update check failed")
        behind = None

    payload["behind"] = behind
    if behind is None:
        payload["message"] = "Couldn't reach the update source — try again later."
    elif behind == 0:
        payload["message"] = "You're on the latest version."
    else:
        payload["update_available"] = True
        # Enrich with the actual commits we're behind by, so the desktop's
        # remote update overlay can show "what's changed". git/pip only;
        # best-effort (empty list on any failure).
        if install_method in ("git", "pip"):
            payload["commits"] = await asyncio.to_thread(_recent_upstream_commits)

    return payload


@app.post("/api/audio/transcribe")
async def transcribe_audio_upload(payload: AudioTranscriptionRequest):
    data_url = (payload.data_url or "").strip()
    if not data_url.startswith("data:") or "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid audio payload")

    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise HTTPException(
            status_code=400, detail="Audio payload must be base64 encoded"
        )

    mime_type = (
        payload.mime_type or header[5:].split(";", 1)[0] or "audio/webm"
    ).strip()
    normalized_mime_type = mime_type.split(";", 1)[0].lower()
    if not (
        normalized_mime_type.startswith("audio/")
        or normalized_mime_type == "video/webm"
    ):
        raise HTTPException(
            status_code=400, detail="Payload must be an audio recording"
        )

    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Audio payload is not valid base64")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio recording is empty")
    if len(audio_bytes) > _MAX_TRANSCRIPTION_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio recording is too large")

    temp_path = ""
    try:
        suffix = _audio_extension_for_mime(mime_type)
        with tempfile.NamedTemporaryFile(
            prefix="hermes-desktop-voice-",
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        from tools.transcription_tools import transcribe_audio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, transcribe_audio, temp_path)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Desktop voice transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error") or "Transcription failed",
        )

    return {
        "ok": True,
        "transcript": str(result.get("transcript") or "").strip(),
        "provider": result.get("provider"),
    }


class TTSSpeakRequest(BaseModel):
    text: str


def _elevenlabs_voice_label(voice: Dict[str, Any]) -> str:
    name = str(voice.get("name") or voice.get("voice_id") or "Voice").strip()
    category = str(voice.get("category") or "").strip()

    return f"{name} ({category})" if category else name


# Collapses repeated identical ElevenLabs voice-list failures (the desktop
# re-polls on every settings open/focus) to a single log line. Re-arms on
# success or when the error signature changes, so a real new failure is seen.
_voice_list_last_error: Optional[str] = None


def _voice_list_error_logged_once(signature: Optional[str]) -> bool:
    """Return True if ``signature`` is new and should be logged now.

    Passing ``None`` clears the latch (call on success). Idempotent per
    signature: the same error logs once until it changes.
    """
    global _voice_list_last_error
    if signature is None:
        _voice_list_last_error = None
        return False
    if signature == _voice_list_last_error:
        return False
    _voice_list_last_error = signature
    return True


@app.get("/api/audio/elevenlabs/voices")
async def get_elevenlabs_voices():
    """Return ElevenLabs voices when an API key is configured.

    The desktop UI uses this for the ``tts.elevenlabs.voice_id`` dropdown.
    Only non-secret voice metadata is returned; the API key stays server-side.
    """
    api_key = (load_env().get("ELEVENLABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        return {"available": False, "voices": []}

    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/voices",
        headers={
            "Accept": "application/json",
            "xi-api-key": api_key,
        },
    )

    try:
        loop = asyncio.get_running_loop()

        def _fetch() -> Dict[str, Any]:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))

        payload = await loop.run_in_executor(None, _fetch)
    except urllib.error.HTTPError as exc:
        # An auth failure (bad/expired/scoped key) is a persistent,
        # user-fixable state, not a transient blip — the desktop polls this on
        # every settings open/focus, so a per-poll WARNING floods the log
        # (#voice-list-401-spam). Treat 401/403 as "integration unavailable":
        # report it to the UI with a 200 and log at most once until the error
        # signature changes (see _voice_list_error_logged_once).
        if exc.code in (401, 403):
            if _voice_list_error_logged_once(f"http-{exc.code}"):
                _log.info(
                    "ElevenLabs voices unavailable: %s — check ELEVENLABS_API_KEY", exc
                )
            return {"available": False, "voices": [], "error": "unauthorized"}
        if _voice_list_error_logged_once(f"http-{exc.code}"):
            _log.warning("ElevenLabs voice list failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not load ElevenLabs voices")
    except Exception as exc:
        if _voice_list_error_logged_once(str(exc)):
            _log.warning("ElevenLabs voice list failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not load ElevenLabs voices")
    _voice_list_error_logged_once(None)  # success — re-arm logging for next failure

    voices = []
    for voice in payload.get("voices") or []:
        if not isinstance(voice, dict):
            continue

        voice_id = str(voice.get("voice_id") or "").strip()
        if not voice_id:
            continue

        voices.append({
            "voice_id": voice_id,
            "name": str(voice.get("name") or voice_id),
            "label": _elevenlabs_voice_label(voice),
        })

    voices.sort(key=lambda item: str(item.get("label") or "").lower())
    return {"available": True, "voices": voices}


@app.post("/api/audio/speak")
async def speak_text(payload: TTSSpeakRequest):
    """Synthesize speech and return audio as base64 data URL.

    Used by the desktop voice-conversation mode to play back assistant
    responses without exposing the on-disk file path. Reuses the
    existing TTS provider chain (Edge / OpenAI / ElevenLabs / etc.)
    configured in ``~/.hermes/config.yaml`` under ``tts.``.
    """
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        from tools.tts_tool import text_to_speech_tool
        loop = asyncio.get_running_loop()
        result_json = await loop.run_in_executor(None, text_to_speech_tool, text)
    except Exception as exc:
        _log.exception("Desktop voice TTS failed")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}")

    try:
        result = json.loads(result_json) if isinstance(result_json, str) else result_json
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid TTS response")

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error") or "Speech synthesis failed",
        )

    file_path = result.get("file_path")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=500, detail="Audio file missing")

    ext = os.path.splitext(file_path)[1].lower()
    mime_type = {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }.get(ext, "audio/mpeg")

    try:
        with open(file_path, "rb") as fh:
            audio_bytes = fh.read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read audio: {exc}")
    finally:
        try:
            os.unlink(file_path)
        except OSError:
            pass

    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return {
        "ok": True,
        "data_url": f"data:{mime_type};base64,{encoded}",
        "mime_type": mime_type,
        "provider": result.get("provider"),
    }


@app.get("/api/actions/{name}/status")
async def get_action_status(name: str, lines: int = 200):
    """Tail an action log and report whether the process is still running."""
    log_file_name = _ACTION_LOG_FILES.get(name)
    if log_file_name is None:
        raise HTTPException(status_code=404, detail=f"Unknown action: {name}")

    log_path = _ACTION_LOG_DIR / log_file_name
    tail = _tail_lines(log_path, min(max(lines, 1), 2000))

    proc = _ACTION_PROCS.get(name)
    if proc is None:
        result = _ACTION_RESULTS.get(name)
        running = False
        exit_code = result.get("exit_code") if result else None
        pid = result.get("pid") if result else None
    else:
        exit_code = proc.poll()
        running = exit_code is None
        pid = proc.pid
        if exit_code is not None:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
            _ACTION_RESULTS[name] = {"exit_code": exit_code, "pid": pid}
            _ACTION_PROCS.pop(name, None)
            _ACTION_COMMANDS.pop(name, None)

    return {
        "name": name,
        "running": running,
        "exit_code": exit_code,
        "pid": pid,
        "lines": tail,
    }


# Per-row fields that no session LIST consumer reads but that dominate the
# payload. ``system_prompt`` is the fully rendered prompt — tens of KB per
# row — and made a 21-row /api/sessions response 528KB (96% dead weight),
# re-fetched by the desktop sidebar on every refresh. The desktop's
# SessionInfo type doesn't declare either field and the web UI never touches
# them; ``GET /api/sessions/{id}`` detail reads stay complete. List callers
# that genuinely need the full rows can pass ``?full=1``.
_SESSION_LIST_HEAVY_FIELDS = ("system_prompt", "model_config")


def _strip_session_list_rows(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for s in sessions:
        for key in _SESSION_LIST_HEAVY_FIELDS:
            s.pop(key, None)
    return sessions


@app.get("/api/sessions")
def get_sessions(
    limit: int = 20,
    offset: int = 0,
    min_messages: int = 0,
    archived: str = "exclude",
    order: str = "created",
    source: str = None,
    exclude_sources: str = None,
    cwd_prefix: str = None,
    full: bool = False,
    profile: Optional[str] = None,
):
    """List sessions.

    ``archived`` controls how soft-archived sessions are treated:
    ``exclude`` (default) hides them, ``only`` returns just the archived ones
    (used by the desktop "Archived sessions" settings panel), and ``include``
    returns both.

    ``order`` controls pagination order: ``created`` (default, by original
    start time) or ``recent`` (by latest activity across the compression
    chain). ``recent`` keeps a long-running conversation on the first page
    after it auto-compresses into a fresh continuation id.

    Rows omit ``system_prompt``/``model_config`` (the payload-dominating
    fields no list UI reads) unless ``full=1`` is passed.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(
            status_code=400,
            detail="archived must be one of: exclude, only, include",
        )
    if order not in ("created", "recent"):
        raise HTTPException(
            status_code=400,
            detail="order must be one of: created, recent",
        )
    profile_name: Optional[str] = None
    if profile:
        profile_name, _ = _cron_profile_home(profile)
    try:
        db = _open_session_db_for_profile(profile)
        try:
            min_message_count = max(0, min_messages)
            archived_only = archived == "only"
            include_archived = archived == "include"
            # Optional source scoping: ``source`` includes a single class,
            # ``exclude_sources`` (comma-separated) drops classes. The desktop
            # uses these to split recents (exclude=cron) from the cron-jobs
            # section (source=cron) into two independent lists.
            exclude_list = [s for s in (exclude_sources or "").split(",") if s.strip()]
            sessions = db.list_sessions_rich(
                source=source or None,
                exclude_sources=exclude_list or None,
                cwd_prefix=(cwd_prefix or None),
                limit=limit,
                offset=offset,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                order_by_last_active=order == "recent",
                # SQL-level projection: when the caller didn't ask for full
                # rows, skip the system_prompt blob inside SQLite too (pairs
                # with the API-level _strip_session_list_rows below).
                compact_rows=not full,
            )
            total = db.session_count(
                source=source or None,
                cwd_prefix=(cwd_prefix or None),
                exclude_sources=exclude_list or None,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                exclude_children=True,
            )
            now = time.time()
            for s in sessions:
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
                if profile_name:
                    s["profile"] = profile_name
                    s["is_default_profile"] = profile_name == "default"
                # SQLite stores the flag as 0/1; expose a real JSON boolean.
                s["archived"] = bool(s.get("archived"))
            if not full:
                _strip_session_list_rows(sessions)
            return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/sessions failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/profiles/sessions")
def get_profiles_sessions(
    limit: int = 20,
    offset: int = 0,
    min_messages: int = 0,
    archived: str = "exclude",
    order: str = "recent",
    profile: str = "all",
    source: str = None,
    exclude_sources: str = None,
    full: bool = False,
):
    """Unified, read-only session list aggregated across ALL profiles.

    Intentionally process-light: this opens each profile's ``state.db`` directly
    from disk — it does NOT spawn a dashboard backend per profile. Each returned
    session is tagged with its owning ``profile`` so the desktop renders one
    browsable list and only spins up a profile's backend when the user actually
    interacts (sends a message). A user with a single (default) profile gets the
    same rows as ``/api/sessions``, just tagged ``profile="default"``.

    Rows omit ``system_prompt``/``model_config`` unless ``full=1`` — same
    list projection as ``/api/sessions``.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(status_code=400, detail="archived must be one of: exclude, only, include")
    if order not in ("created", "recent"):
        raise HTTPException(status_code=400, detail="order must be one of: created, recent")

    from hermes_state import SessionDB
    from hermes_cli import profiles as profiles_mod

    targets: List[Tuple[str, Path]] = []
    if profile and profile != "all":
        name, home = _cron_profile_home(profile)
        targets.append((name, home))
    else:
        try:
            infos = profiles_mod.list_profiles()
            targets = [(info.name, info.path) for info in infos]
        except Exception:
            _log.exception("GET /api/profiles/sessions: list_profiles failed")
            targets = []
        if not targets:
            targets.append(("default", profiles_mod.get_profile_dir("default")))

    min_message_count = max(0, min_messages)
    archived_only = archived == "only"
    include_archived = archived == "include"
    # Source scoping (see /api/sessions): recents pass exclude_sources=cron,
    # the cron-jobs section passes source=cron — two independent lists so
    # newest cron sessions can't starve the recents page.
    source_filter = source or None
    exclude_list = [s for s in (exclude_sources or "").split(",") if s.strip()]
    # Over-fetch per profile so the merged+sorted window is correct for the
    # requested page. Capped so a huge profile can't blow up the response.
    per_profile = min(max(limit + offset, limit), 500)

    merged: List[Dict[str, Any]] = []
    total = 0
    profile_totals: Dict[str, int] = {}
    errors: List[Dict[str, str]] = []
    now = time.time()
    for name, home in targets:
        db_path = Path(home) / "state.db"
        if not db_path.exists():
            continue
        try:
            # Read-only: this loop runs on every sidebar refresh, so it must
            # never DDL/write-lock another profile's live DB (see SessionDB
            # read_only docstring).
            db = SessionDB(db_path=db_path, read_only=True)
        except Exception as exc:
            errors.append({"profile": name, "error": str(exc)})
            continue
        try:
            rows = db.list_sessions_rich(
                source=source_filter,
                exclude_sources=exclude_list or None,
                limit=per_profile,
                offset=0,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                order_by_last_active=order == "recent",
                # Same SQL-level blob skip as /api/sessions (see above).
                compact_rows=not full,
            )
            profile_total = db.session_count(
                source=source_filter,
                exclude_sources=exclude_list or None,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                exclude_children=True,
            )
            total += profile_total
            profile_totals[name] = profile_total
            for s in rows:
                s["profile"] = name
                s["is_default_profile"] = name == "default"
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
                s["archived"] = bool(s.get("archived"))
                merged.append(s)
        except Exception as exc:
            errors.append({"profile": name, "error": str(exc)})
        finally:
            db.close()

    sort_key = "last_active" if order == "recent" else "started_at"
    merged.sort(key=lambda s: s.get(sort_key) or s.get("started_at") or 0, reverse=True)
    window = merged[offset:offset + limit]
    if not full:
        _strip_session_list_rows(window)
    return {
        "sessions": window,
        "total": total,
        "profile_totals": profile_totals,
        "limit": limit,
        "offset": offset,
        "errors": errors,
    }


@app.get("/api/sessions/search")
async def search_sessions(q: str = "", limit: int = 20, profile: Optional[str] = None):
    """Search sessions by ID plus full-text message content using FTS5.

    Direct session-id matches are surfaced first, then FTS message-content
    matches. Results are deduped by compression lineage, not by raw
    ``session_id``. Auto-compression rotates a conversation onto a fresh
    session id (and leaves the old segment's messages in the FTS index), so one
    logical chat can own many ``sessions`` rows that all match the same query.
    Branches also use ``parent_session_id``, but they are real alternate
    conversations; don't collapse branch-specific hits back into the parent.
    """
    if not q or not q.strip():
        return {"results": []}
    try:
        db = _open_session_db_for_profile(profile)
        try:
            safe_limit = max(1, min(int(limit or 20), 100))

            # Walk parent_session_id to the compression root, memoized so a
            # chain of compression segments only costs one walk. We deliberately
            # stop at branch/delegate edges: those sessions may diverge from the
            # parent and should remain searchable on their own.
            root_cache: dict = {}

            def compression_root(session_id: str) -> str:
                if not session_id:
                    return session_id
                if session_id in root_cache:
                    return root_cache[session_id]
                chain = []
                cur = session_id
                visited = set()
                root = session_id
                while cur and cur not in visited:
                    visited.add(cur)
                    chain.append(cur)
                    if cur in root_cache:
                        root = root_cache[cur]
                        break
                    try:
                        s = db.get_session(cur)
                    except Exception:
                        s = None
                    if not s:
                        root = cur
                        break
                    parent = s.get("parent_session_id") if isinstance(s, dict) else None
                    if not parent:
                        root = cur
                        break
                    try:
                        parent_session = db.get_session(parent)
                    except Exception:
                        parent_session = None
                    if not parent_session:
                        root = cur
                        break
                    parent_ended_at = parent_session.get("ended_at")
                    started_at = s.get("started_at")
                    is_compression_edge = (
                        parent_session.get("end_reason") == "compression"
                        and parent_ended_at is not None
                        and started_at is not None
                        and started_at >= parent_ended_at
                    )
                    if not is_compression_edge:
                        root = cur
                        break
                    cur = parent
                for node in chain:
                    root_cache[node] = root
                return root

            tip_cache: dict = {}

            def lineage_tip(root_id: str) -> str:
                if root_id in tip_cache:
                    return tip_cache[root_id]
                tip = root_id
                try:
                    resolved = db.get_compression_tip(root_id)
                    if resolved:
                        tip = resolved
                except Exception:
                    pass
                tip_cache[root_id] = tip
                return tip

            # Both ID matches and content matches share one keyspace, keyed by
            # compression lineage root, so an id-hit and a content-hit on the
            # same logical conversation collapse to a single result. The first
            # hit for a lineage wins; ID matches run first and take priority.
            seen: dict = {}

            def add_lineage_result(raw_sid: str, payload: dict) -> None:
                if not raw_sid:
                    return
                root = compression_root(raw_sid)
                if root in seen or len(seen) >= safe_limit:
                    return
                payload = dict(payload)
                payload["session_id"] = lineage_tip(root)
                payload["lineage_root"] = root
                seen[root] = payload

            # Direct ID matches first: users often paste a session id from CLI,
            # logs, or another Hermes surface. FTS can't find those unless the
            # id happens to appear in message text. search_sessions_by_id is
            # SQL-bounded, so this stays cheap even with thousands of sessions.
            for row in db.search_sessions_by_id(q, limit=safe_limit, include_archived=True):
                sid = row.get("id")
                preview = (row.get("preview") or "").strip()
                snippet = preview or f"Session ID: {sid}"
                add_lineage_result(
                    sid,
                    {
                        "snippet": snippet,
                        "role": None,
                        "source": row.get("source"),
                        "model": row.get("model"),
                        "session_started": row.get("started_at"),
                    },
                )

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
            # Over-fetch so lineage dedup can still surface `limit` distinct
            # conversations even when several hits collapse onto one root.
            fetch_limit = max(safe_limit * 5, 50)
            matches = db.search_messages(query=prefix_query, limit=fetch_limit)

            for m in matches:
                if len(seen) >= safe_limit:
                    break
                add_lineage_result(
                    m["session_id"],
                    {
                        "snippet": m.get("snippet", ""),
                        "role": m.get("role"),
                        "source": m.get("source"),
                        "model": m.get("model"),
                        "session_started": m.get("session_started"),
                    },
                )
            return {"results": list(seen.values())}
        finally:
            db.close()
    except HTTPException:
        raise
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


def _memory_provider_label(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def _normalize_memory_provider_name(name: Any) -> str:
    provider = str(name or "").strip()
    if provider.lower() in {"built-in", "builtin", "none"}:
        return ""
    return provider


def _load_memory_provider(name: str):
    try:
        from plugins.memory import load_memory_provider

        return load_memory_provider(name)
    except Exception:
        _log.debug("Failed to load memory provider %s", name, exc_info=True)
        return None


def _memory_provider_manifest(name: str) -> Dict[str, Any]:
    try:
        from plugins.memory import find_provider_dir

        provider_dir = find_provider_dir(name)
        if provider_dir is None:
            return {}
        manifest_path = provider_dir / "plugin.yaml"
        if not manifest_path.exists():
            return {}
        with manifest_path.open(encoding="utf-8-sig") as handle:
            manifest = yaml.safe_load(handle) or {}
        return manifest if isinstance(manifest, dict) else {}
    except Exception:
        _log.debug("Failed to read memory provider manifest for %s", name, exc_info=True)
        return {}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _memory_provider_setup_manifest(name: str) -> Dict[str, Any]:
    manifest = _memory_provider_manifest(name)
    external_dependencies: List[Dict[str, str]] = []
    for raw in manifest.get("external_dependencies") or []:
        if not isinstance(raw, dict):
            continue
        dep = {
            "name": str(raw.get("name") or "").strip(),
            "install": str(raw.get("install") or "").strip(),
            "check": str(raw.get("check") or "").strip(),
        }
        if dep["name"] or dep["install"] or dep["check"]:
            external_dependencies.append(dep)

    return {
        "pip_dependencies": _string_list(manifest.get("pip_dependencies")),
        "external_dependencies": external_dependencies,
        "required_env": _string_list(manifest.get("requires_env")),
    }


def _memory_provider_setup_info(name: str) -> Dict[str, Any]:
    setup = _memory_provider_setup_manifest(name)
    setup["dependencies_installed"] = _memory_provider_dependencies_installed(setup)
    return setup


_MEMORY_PROVIDER_IMPORT_NAMES = {
    "honcho-ai": "honcho",
    "mem0ai": "mem0",
    "hindsight-client": "hindsight_client",
    "hindsight-all": "hindsight",
}


def _memory_provider_dependency_package(dep: str) -> str:
    return re.split(r"[\[<>=!~;]", dep, maxsplit=1)[0].strip()


def _memory_provider_import_name(dep: str) -> str:
    package = _memory_provider_dependency_package(dep)
    return _MEMORY_PROVIDER_IMPORT_NAMES.get(package, package.replace("-", "_"))


def _dependency_importable(dep: str) -> bool:
    import_name = _memory_provider_import_name(dep)
    if not import_name:
        return False
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def _trim_setup_output(value: Optional[str], limit: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... truncated ..."


def _memory_provider_setup_env() -> Dict[str, str]:
    env = os.environ.copy()
    home = Path.home()
    extra_bins = [
        home / ".brv-cli" / "bin",
        home / ".local" / "bin",
        home / ".npm-global" / "bin",
        Path("/usr/local/bin"),
    ]
    existing_path = env.get("PATH", "")
    prefix = os.pathsep.join(str(path) for path in extra_bins if path.exists())
    if prefix:
        env["PATH"] = prefix + os.pathsep + existing_path
    return env


def _command_result(
    *,
    kind: str,
    name: str,
    status: str,
    command: str = "",
    completed: Optional[subprocess.CompletedProcess] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "status": status,
        "command": command,
        "returncode": None if completed is None else completed.returncode,
        "stdout": "" if completed is None else _trim_setup_output(completed.stdout),
        "stderr": _trim_setup_output(error or ("" if completed is None else completed.stderr)),
    }


def _run_setup_command(
    command: Any,
    *,
    display: str,
    shell: bool = False,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=shell,
        executable="/bin/bash" if shell else None,
        env=_memory_provider_setup_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _memory_provider_dependencies_installed(setup: Dict[str, Any]) -> bool:
    pip_dependencies = _string_list(setup.get("pip_dependencies"))
    external_dependencies = setup.get("external_dependencies") or []

    pip_ok = all(_dependency_importable(dep) for dep in pip_dependencies)
    external_ok = True
    for dep in external_dependencies:
        if not isinstance(dep, dict):
            continue
        check_cmd = str(dep.get("check") or "").strip()
        install_cmd = str(dep.get("install") or "").strip()
        if not check_cmd:
            if install_cmd:
                external_ok = False
            continue
        try:
            completed = _run_setup_command(
                shlex.split(check_cmd),
                display=check_cmd,
                timeout=20,
            )
        except Exception:
            external_ok = False
            continue
        if completed.returncode != 0:
            external_ok = False

    return pip_ok and external_ok


def _install_memory_provider_pip_dependencies(dependencies: List[str]) -> List[Dict[str, Any]]:
    missing = [dep for dep in dependencies if not _dependency_importable(dep)]
    if not dependencies:
        return []
    if not missing:
        return [
            _command_result(kind="pip", name=", ".join(dependencies), status="already_installed")
        ]

    uv_path = shutil.which("uv")
    if uv_path:
        command: Any = [uv_path, "pip", "install", "--python", sys.executable, "--quiet", *missing]
        display = f"uv pip install --python {sys.executable} {' '.join(missing)}"
    else:
        command = [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        display = f"{sys.executable} -m pip install {' '.join(missing)}"

    try:
        completed = _run_setup_command(command, display=display, timeout=240)
    except Exception as exc:
        return [
            _command_result(
                kind="pip",
                name=", ".join(missing),
                status="failed",
                command=display,
                error=str(exc),
            )
        ]

    return [
        _command_result(
            kind="pip",
            name=", ".join(missing),
            status="installed" if completed.returncode == 0 else "failed",
            command=display,
            completed=completed,
        )
    ]


def _install_memory_provider_external_dependencies(
    dependencies: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for dep in dependencies:
        name = dep.get("name") or "dependency"
        check_cmd = dep.get("check") or ""
        install_cmd = dep.get("install") or ""

        if check_cmd:
            try:
                check = _run_setup_command(
                    shlex.split(check_cmd),
                    display=check_cmd,
                    timeout=20,
                )
            except Exception as exc:
                results.append(
                    _command_result(
                        kind="external_check",
                        name=name,
                        status="missing" if install_cmd else "failed",
                        command=check_cmd,
                        error=str(exc),
                    )
                )
            else:
                if check.returncode == 0:
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="already_installed",
                            command=check_cmd,
                            completed=check,
                        )
                    )
                    continue
                results.append(
                    _command_result(
                        kind="external_check",
                        name=name,
                        status="missing" if install_cmd else "failed",
                        command=check_cmd,
                        completed=check,
                    )
                )

            if not install_cmd:
                continue

        if install_cmd:
            try:
                install = _run_setup_command(
                    install_cmd,
                    display=install_cmd,
                    shell=True,
                    timeout=300,
                )
            except Exception as exc:
                results.append(
                    _command_result(
                        kind="external_install",
                        name=name,
                        status="failed",
                        command=install_cmd,
                        error=str(exc),
                    )
                )
                continue

            results.append(
                _command_result(
                    kind="external_install",
                    name=name,
                    status="installed" if install.returncode == 0 else "failed",
                    command=install_cmd,
                    completed=install,
                )
            )

            if check_cmd and install.returncode == 0:
                try:
                    post_check = _run_setup_command(
                        shlex.split(check_cmd),
                        display=check_cmd,
                        timeout=20,
                    )
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="verified" if post_check.returncode == 0 else "failed",
                            command=check_cmd,
                            completed=post_check,
                        )
                    )
                except Exception as exc:
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="failed",
                            command=check_cmd,
                            error=str(exc),
                        )
                    )

    return results


def _install_memory_provider_setup(name: str) -> Dict[str, Any]:
    provider = _load_memory_provider(name)
    manifest = _memory_provider_manifest(name)
    if provider is None and not manifest:
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")

    setup = _memory_provider_setup_manifest(name)
    results = []
    results.extend(_install_memory_provider_pip_dependencies(setup["pip_dependencies"]))
    results.extend(
        _install_memory_provider_external_dependencies(setup["external_dependencies"])
    )

    if not results:
        results.append(
            _command_result(
                kind="setup",
                name=name,
                status="no_declared_steps",
            )
        )

    ok = all(result["status"] not in {"failed"} for result in results)
    statuses = {row["name"]: row for row in _discover_memory_provider_statuses()}
    return {
        "ok": ok,
        "provider": name,
        "results": results,
        "status": statuses.get(name),
    }


def _normalize_memory_provider_schema(name: str, provider: Any) -> List[Dict[str, Any]]:
    raw_schema: List[Dict[str, Any]] = []
    if provider is not None and hasattr(provider, "get_config_schema"):
        try:
            raw = provider.get_config_schema()
            if isinstance(raw, list):
                raw_schema = [field for field in raw if isinstance(field, dict)]
        except Exception:
            _log.warning("Failed to read memory provider schema for %s", name, exc_info=True)

    fields: List[Dict[str, Any]] = []
    for raw in raw_schema:
        key = str(raw.get("key") or "").strip()
        if not key:
            continue

        choices = raw.get("choices") or raw.get("options") or []
        if not isinstance(choices, list):
            choices = []

        explicit_kind = str(raw.get("kind") or raw.get("type") or "").strip().lower()
        if raw.get("secret"):
            kind = "secret"
        elif choices:
            kind = "select"
        elif explicit_kind in {"bool", "boolean"} or isinstance(raw.get("default"), bool):
            kind = "boolean"
        else:
            kind = "text"

        options = []
        for choice in choices:
            value = str(choice)
            options.append({"value": value, "label": value, "description": ""})

        description = str(raw.get("description") or "")
        fields.append({
            "key": key,
            "label": str(raw.get("label") or key.replace("_", " ").title()),
            "kind": kind,
            "description": description,
            "placeholder": str(raw.get("placeholder") or ""),
            "required": bool(raw.get("required", False)),
            "default": raw.get("default", ""),
            "options": options,
            "url": str(raw.get("url") or ""),
            "when": raw.get("when") if isinstance(raw.get("when"), dict) else None,
            "_env_key": str(raw.get("env_var") or "") or None,
        })

    return fields


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _log.debug("Failed to read JSON config from %s", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _read_memory_provider_existing_values(name: str) -> Dict[str, Any]:
    """Best-effort read of existing provider config across legacy/native stores."""

    hermes_home = get_hermes_home()
    values: Dict[str, Any] = {}

    # Common native provider stores.
    for path in (
        hermes_home / f"{name}.json",
        hermes_home / name / "config.json",
    ):
        values.update(_read_json_file(path))

    try:
        cfg = load_config()
    except Exception:
        cfg = {}

    memory_cfg = cfg.get("memory") if isinstance(cfg, dict) else {}
    if isinstance(memory_cfg, dict):
        provider_cfg = memory_cfg.get(name)
        if isinstance(provider_cfg, dict):
            values.update(provider_cfg)
        legacy_cfg = memory_cfg.get("provider_config")
        if isinstance(legacy_cfg, dict):
            values = {**legacy_cfg, **values}

    # Holographic stores under plugins.hermes-memory-store.
    plugins_cfg = cfg.get("plugins") if isinstance(cfg, dict) else {}
    if name == "holographic" and isinstance(plugins_cfg, dict):
        holographic_cfg = plugins_cfg.get("hermes-memory-store")
        if isinstance(holographic_cfg, dict):
            values.update(holographic_cfg)

    return values


def _env_lookup(env_key: Optional[str]) -> str:
    if not env_key:
        return ""
    env_on_disk = load_env()
    return str(env_on_disk.get(env_key) or os.environ.get(env_key) or "")


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _field_default(field: Dict[str, Any]) -> Any:
    default = field.get("default", "")
    if field["kind"] == "boolean":
        return _coerce_bool(default, default=False)
    return default


def _field_value(field: Dict[str, Any], data: Dict[str, Any]) -> Any:
    if field["kind"] == "secret":
        return ""

    value = data.get(field["key"])
    if value in (None, ""):
        value = _env_lookup(field.get("_env_key"))
    if value in (None, ""):
        value = _field_default(field)

    if field["kind"] == "select":
        allowed = {opt["value"] for opt in field.get("options", [])}
        value = str(value)
        return value if value in allowed else str(_field_default(field))
    if field["kind"] == "boolean":
        return _coerce_bool(value, default=_coerce_bool(_field_default(field), default=False))
    return str(value)


def _field_is_set(field: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if field["kind"] == "secret":
        return bool(_env_lookup(field.get("_env_key")) or data.get(field["key"]))
    value = _field_value(field, data)
    return value not in (None, "")


def _field_visible(
    field: Dict[str, Any],
    data: Dict[str, Any],
    fields_by_key: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    when = field.get("when")
    if not isinstance(when, dict) or not when:
        return True
    for dep_key, expected in when.items():
        dep_field = (fields_by_key or {}).get(str(dep_key)) or {
            "key": str(dep_key),
            "kind": "text",
            "default": "",
            "_env_key": None,
        }
        actual = _field_value(dep_field, data)
        if str(actual) != str(expected):
            return False
    return True


def _public_memory_provider_field(field: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "key": field["key"],
        "label": field["label"],
        "kind": field["kind"],
        "description": field["description"],
        "placeholder": field["placeholder"],
        "required": field["required"],
        "value": "" if field["kind"] == "secret" else _field_value(field, data),
        "is_set": _field_is_set(field, data),
        "options": field.get("options", []),
        "url": field.get("url", ""),
        "when": field.get("when"),
    }
    return entry


def _memory_provider_payload(name: str, provider: Any) -> Dict[str, Any]:
    data = _read_memory_provider_existing_values(name)
    fields = [
        _public_memory_provider_field(field, data)
        for field in _normalize_memory_provider_schema(name, provider)
    ]
    return {
        "name": name,
        "label": _memory_provider_label(name),
        "fields": fields,
        "setup": _memory_provider_setup_info(name),
    }


def _coerce_schema_field(field: Dict[str, Any], raw: Any) -> Any:
    if field["kind"] == "boolean":
        return _coerce_bool(raw, default=_coerce_bool(_field_default(field), default=False))

    value = str(raw if raw is not None else "").strip()
    if field["kind"] == "select":
        if not value:
            value = str(_field_default(field))
        allowed = {opt["value"] for opt in field.get("options", [])}
        if value not in allowed:
            raise ValueError(f"Invalid value for '{field['key']}'")
        return value

    return value or _field_default(field)


def _save_memory_provider_native_config(name: str, provider: Any, values: Dict[str, Any]) -> None:
    if provider is not None and hasattr(provider, "save_config"):
        try:
            from agent.memory_provider import MemoryProvider as _BaseMemoryProvider
        except Exception:
            provider.save_config(values, str(get_hermes_home()))
            return
        if type(provider).save_config is not _BaseMemoryProvider.save_config:
            provider.save_config(values, str(get_hermes_home()))
            return

    cfg = load_config()
    memory_cfg = cfg.get("memory")
    if not isinstance(memory_cfg, dict):
        memory_cfg = {}
        cfg["memory"] = memory_cfg
    current = memory_cfg.get(name)
    if not isinstance(current, dict):
        current = {}
    current.update(values)
    memory_cfg[name] = current
    save_config(cfg)


def _memory_provider_is_configured(name: str, provider: Any) -> bool:
    data = _read_memory_provider_existing_values(name)
    fields = _normalize_memory_provider_schema(name, provider)
    fields_by_key = {field["key"]: field for field in fields}
    visible_fields = [
        field for field in fields if _field_visible(field, data, fields_by_key)
    ]
    required_fields = [field for field in visible_fields if field.get("required")]
    if not required_fields:
        return True
    return all(_field_is_set(field, data) for field in required_fields)


def _discover_memory_provider_statuses() -> List[Dict[str, Any]]:
    discovered: Dict[str, Dict[str, Any]] = {}
    try:
        from plugins.memory import discover_memory_providers

        for name, description, available in discover_memory_providers():
            discovered[str(name)] = {
                "name": str(name),
                "description": str(description or ""),
                "available": bool(available),
                "missing": False,
            }
    except Exception:
        _log.exception("discover_memory_providers failed")

    cfg = load_config()
    active = ""
    mem = cfg.get("memory")
    if isinstance(mem, dict):
        active = _normalize_memory_provider_name(mem.get("provider"))
    if active and active not in discovered:
        discovered[active] = {
            "name": active,
            "description": "Configured provider was not found.",
            "available": False,
            "missing": True,
        }

    providers: List[Dict[str, Any]] = []
    for name in sorted(discovered):
        row = discovered[name]
        provider = None if row["missing"] else _load_memory_provider(name)
        setup = _memory_provider_setup_info(name)
        configured = False if row["missing"] else _memory_provider_is_configured(name, provider)
        schema_fields = [] if row["missing"] else _normalize_memory_provider_schema(name, provider)
        if row["missing"]:
            status = "missing"
        elif not row["available"] and not setup.get("dependencies_installed", True):
            status = "unavailable"
        elif not configured:
            status = "needs_config"
        elif not row["available"] and schema_fields:
            status = "needs_config"
        elif not row["available"]:
            status = "unavailable"
        else:
            status = "ready"
        providers.append({
            "name": name,
            "description": row["description"],
            "available": row["available"],
            "configured": configured,
            "status": status,
            "setup": setup,
        })
    return providers


def _require_memory_provider_ready(name: str) -> None:
    if not name:
        return
    statuses = {row["name"]: row for row in _discover_memory_provider_statuses()}
    row = statuses.get(name)
    if row is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown memory provider '{name}'.",
        )
    if row["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Memory provider '{name}' is not ready "
                f"({row['status'].replace('_', ' ')}). Configure it in the dashboard first."
            ),
        )


def _write_memory_provider_config_values(
    name: str,
    provider: Any,
    values: Dict[str, Any],
) -> None:
    existing = _read_memory_provider_existing_values(name)
    fields = _normalize_memory_provider_schema(name, provider)
    fields_by_key = {field["key"]: field for field in fields}
    config_values: Dict[str, Any] = {}
    secrets: Dict[str, str] = {}

    for field in fields:
        if not _field_visible(field, {**existing, **config_values}, fields_by_key):
            continue

        if field["kind"] == "secret":
            submitted = str(values.get(field["key"]) or "").strip()
            if submitted and field.get("_env_key"):
                secrets[str(field["_env_key"])] = submitted
            continue

        raw = (
            values[field["key"]]
            if field["key"] in values
            else existing.get(field["key"], _field_default(field))
        )
        config_values[field["key"]] = _coerce_schema_field(field, raw)

    _save_memory_provider_native_config(name, provider, config_values)

    for env_key, secret in secrets.items():
        save_env_value(env_key, secret)


_MEMORY_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _require_valid_memory_provider_name(name: str) -> None:
    """Reject provider names that could traverse outside the plugin dirs.

    ``name`` is interpolated into filesystem paths by ``find_provider_dir()``
    and gates which plugin manifest's setup commands run. A strict charset
    allowlist (no path separators, no dots) makes traversal impossible
    regardless of how the downstream lookup evolves.
    """
    if not _MEMORY_PROVIDER_NAME_RE.fullmatch(name or ""):
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")


@app.get("/api/memory/providers/{name}/config")
async def get_memory_provider_config(name: str):
    _require_valid_memory_provider_name(name)
    provider = _load_memory_provider(name)
    if provider is None:
        # Undeclared providers (e.g. builtin) have no config surface. Return an
        # empty schema so the generic panel simply renders nothing.
        return {"name": name, "label": name, "fields": [], "setup": _memory_provider_setup_info(name)}
    return _memory_provider_payload(name, provider)


@app.post("/api/memory/providers/{name}/setup")
async def setup_memory_provider(name: str, body: MemoryProviderSetupRequest):
    _require_valid_memory_provider_name(name)
    provider = _load_memory_provider(name)
    if provider is None and not _memory_provider_manifest(name):
        # No discoverable plugin directory → nothing whose manifest could
        # legitimately declare setup commands. Refuse before the
        # command-running path. (provider may be None with a manifest present
        # when its pip deps aren't installed yet — that's the setup use case.)
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")
    if provider is not None and body.values:
        try:
            _write_memory_provider_config_values(name, provider, body.values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            _log.exception("Failed to persist memory provider setup values for %s", name)
            raise HTTPException(status_code=500, detail="Internal server error")
    return _install_memory_provider_setup(name)


@app.put("/api/memory/providers/{name}/config")
async def update_memory_provider_config(name: str, body: MemoryProviderConfigUpdate):
    _require_valid_memory_provider_name(name)
    provider = _load_memory_provider(name)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")

    values = body.values or {}

    try:
        _write_memory_provider_config_values(name, provider, values)
        _require_memory_provider_ready(name)

        config = load_config()
        memory_config = config.get("memory")
        if not isinstance(memory_config, dict):
            memory_config = {}
            config["memory"] = memory_config
        memory_config["provider"] = name
        save_config(config)

        return {"ok": True, "active": name}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception("PUT /api/memory/providers/%s/config failed", name)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/config")
async def get_config(profile: Optional[str] = None):
    with _profile_scope(profile):
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
def get_model_info(profile: Optional[str] = None):
    """Return resolved model metadata for the currently configured model.

    Calls the same context-length resolution chain the agent uses, so the
    frontend can display "Auto-detected: 200K" alongside the override field.
    Also returns model capabilities (vision, reasoning, tools) when available.
    """
    try:
        with _profile_scope(profile):
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
    except HTTPException:
        # Unknown/invalid profile must surface as 404, not degrade into a
        # 200 with empty model info (which would render as "no model set").
        raise
    except Exception:
        _log.exception("GET /api/model/info failed")
        return dict(_EMPTY_MODEL_INFO)


# ---------------------------------------------------------------------------
# Model assignment — pick provider+model for main slot or auxiliary slots.
# Mirrors the model.options JSON-RPC from tui_gateway but uses REST so the
# Models page (which has no chat PTY open) can drive it.
# ---------------------------------------------------------------------------

# Canonical auxiliary task slots. Keep in sync with DEFAULT_CONFIG["auxiliary"]
# in hermes_cli/config.py — listed here for deterministic ordering in the UI.
_AUX_TASK_SLOTS: Tuple[str, ...] = (
    "vision",
    "web_extract",
    "compression",
    "skills_hub",
    "approval",
    "mcp",
    "title_generation",
    "triage_specifier",
    "kanban_decomposer",
    "profile_describer",
    "curator",
)


@app.get("/api/model/options")
def get_model_options(
    profile: Optional[str] = None,
    refresh: bool = False,
    include_unconfigured: bool = False,
    explicit_only: bool = False,
):
    """Return authenticated providers + their curated model lists.

    REST equivalent of the ``model.options`` JSON-RPC on tui_gateway, so the
    dashboard Models page can render the picker without a live chat session.
    The response shape matches ``model.options`` 1:1 so ``ModelPickerDialog``
    can share the same types.

    ``profile`` scopes the picker context (current model/provider, custom
    providers from config, per-profile .env auth state) so the Models page
    reads the SAME profile /api/model/set writes.

    ``refresh`` busts the per-provider model-id disk cache so every row
    re-fetches its live catalog — used by the picker's explicit "Refresh
    Models" control. Normal opens leave it false to stay on the 1h cache.
    """
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        # Most desktop surfaces should only list providers the user has already
        # configured. Onboarding opts into the full provider universe via
        # include_unconfigured=1 so it can still render setup affordances for
        # providers that are not yet authenticated.
        with _profile_scope(profile):
            return build_models_payload(
                load_picker_context(),
                explicit_only=bool(explicit_only),
                include_unconfigured=bool(include_unconfigured),
                picker_hints=True,
                canonical_order=True,
                pricing=True,
                capabilities=True,
                refresh=bool(refresh),
                probe_custom_providers=bool(refresh),
                probe_current_custom_provider=not bool(refresh),
            )
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/model/options failed")
        raise HTTPException(status_code=500, detail="Failed to list model options")


@app.get("/api/model/recommended-default")
def get_recommended_default_model(provider: str = ""):
    """Return the recommended default model for a freshly-authenticated provider.

    Mirrors the model-curation `hermes model` does so GUI onboarding lands on a
    sensible default instead of blindly taking the first curated entry. For
    Nous this honors the user's free/paid tier: free users get a free model,
    paid users get the full curated default. For any other provider it falls
    back to the first curated model (same as before).

    Response: {"provider": str, "model": str, "free_tier": bool | None}
    where free_tier is True/False for Nous and None otherwise. `model` may be
    empty if nothing could be resolved (caller degrades gracefully).
    """
    slug = (provider or "").strip().lower()

    if slug == "nous":
        try:
            from hermes_cli.models import (
                get_curated_nous_model_ids,
                get_pricing_for_provider,
                check_nous_free_tier,
                partition_nous_models_by_tier,
                union_with_portal_free_recommendations,
                union_with_portal_paid_recommendations,
            )
            from hermes_cli.auth import get_provider_auth_state

            model_ids = get_curated_nous_model_ids()
            pricing = get_pricing_for_provider("nous") or {}
            free_tier = check_nous_free_tier(force_fresh=True)

            portal_url = ""
            try:
                state = get_provider_auth_state("nous") or {}
                portal_url = state.get("portal_base_url", "") or ""
            except Exception:
                portal_url = ""

            if free_tier:
                model_ids, pricing = union_with_portal_free_recommendations(
                    model_ids, pricing, portal_url
                )
                model_ids, _unavailable = partition_nous_models_by_tier(
                    model_ids, pricing, free_tier=True
                )
            else:
                model_ids, pricing = union_with_portal_paid_recommendations(
                    model_ids, pricing, portal_url
                )

            model = model_ids[0] if model_ids else ""
            return {"provider": "nous", "model": model, "free_tier": bool(free_tier)}
        except Exception:
            _log.exception("GET /api/model/recommended-default (nous) failed")
            return {"provider": "nous", "model": "", "free_tier": None}

    # Non-Nous: first curated model for the provider, matching prior behaviour.
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        payload = build_models_payload(load_picker_context())
        for row in payload.get("providers", []):
            if str(row.get("slug", "")).lower() == slug:
                models = row.get("models") or []
                return {"provider": slug, "model": models[0] if models else "", "free_tier": None}
        return {"provider": slug, "model": "", "free_tier": None}
    except Exception:
        _log.exception("GET /api/model/recommended-default failed")
        return {"provider": slug, "model": "", "free_tier": None}


@app.get("/api/model/auxiliary")
def get_auxiliary_models(profile: Optional[str] = None):
    """Return current auxiliary task assignments.

    Shape:
      {
        "tasks": [
          {"task": "vision", "provider": "auto", "model": "", "base_url": ""},
          ...
        ],
        "main": {"provider": "openrouter", "model": "anthropic/claude-opus-4.7"},
      }

    ``profile`` scopes the read — without it, the Models page would show
    the dashboard profile's auxiliary pins while /api/model/set wrote the
    selected profile's (read/write asymmetry).
    """
    try:
        with _profile_scope(profile):
            cfg = load_config()
        aux_cfg = cfg.get("auxiliary", {})
        if not isinstance(aux_cfg, dict):
            aux_cfg = {}

        tasks = []
        for slot in _AUX_TASK_SLOTS:
            slot_cfg = aux_cfg.get(slot, {}) if isinstance(aux_cfg.get(slot), dict) else {}
            tasks.append({
                "task": slot,
                "provider": str(slot_cfg.get("provider", "auto") or "auto"),
                "model": str(slot_cfg.get("model", "") or ""),
                "base_url": str(slot_cfg.get("base_url", "") or ""),
            })

        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            main = {
                "provider": str(model_cfg.get("provider", "") or ""),
                "model": str(model_cfg.get("default", model_cfg.get("name", "")) or ""),
            }
        else:
            main = {"provider": "", "model": str(model_cfg) if model_cfg else ""}

        return {"tasks": tasks, "main": main}
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/model/auxiliary failed")
        raise HTTPException(status_code=500, detail="Failed to read auxiliary config")


@app.get("/api/model/moa")
def get_moa_models(profile: Optional[str] = None):
    """Return the configured Mixture-of-Agents provider/model slots."""
    try:
        from hermes_cli.moa_config import normalize_moa_config

        with _profile_scope(profile):
            cfg = load_config()
            return normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/model/moa failed")
        raise HTTPException(status_code=500, detail="Failed to read MoA config")


@app.put("/api/model/moa")
def set_moa_models(body: MoaConfigPayload, profile: Optional[str] = None):
    """Persist the Mixture-of-Agents provider/model slots."""
    try:
        from hermes_cli.moa_config import normalize_moa_config

        with _profile_scope(body.profile or profile):
            cfg = load_config()
            if body.presets:
                raw = {
                    "default_preset": body.default_preset,
                    "active_preset": body.active_preset,
                    "presets": {
                        name: {
                            "reference_models": [slot.dict() for slot in preset.reference_models],
                            "aggregator": preset.aggregator.dict(),
                            "reference_temperature": preset.reference_temperature,
                            "aggregator_temperature": preset.aggregator_temperature,
                            "max_tokens": preset.max_tokens,
                            "enabled": preset.enabled,
                        }
                        for name, preset in body.presets.items()
                    },
                }
            else:
                raw = {
                    "reference_models": [slot.dict() for slot in body.reference_models],
                    "aggregator": body.aggregator.dict(),
                    "reference_temperature": body.reference_temperature,
                    "aggregator_temperature": body.aggregator_temperature,
                    "max_tokens": body.max_tokens,
                    "enabled": body.enabled,
                }
            normalized = normalize_moa_config(raw)
            cfg["moa"] = normalized
            save_config(cfg)
            return {"ok": True, **normalized}
    except HTTPException:
        raise
    except Exception:
        _log.exception("PUT /api/model/moa failed")
        raise HTTPException(status_code=500, detail="Failed to save MoA config")


@app.post("/api/model/set")
async def set_model_assignment(body: ModelAssignment, profile: Optional[str] = None):
    """Assign a model to the main slot or an auxiliary task slot.

    Writes to ``~/.hermes/config.yaml`` — applies to **new** sessions only.
    The currently running chat PTY (if any) is not affected; use the
    ``/model`` slash command inside a chat to hot-swap that specific session.
    """
    scope = (body.scope or "").strip().lower()
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    task = (body.task or "").strip().lower()
    base_url = (body.base_url or "").strip()
    api_key = (body.api_key or "").strip()

    if scope not in {"main", "auxiliary"}:
        raise HTTPException(status_code=400, detail="scope must be 'main' or 'auxiliary'")

    try:
        # Expensive-model warning runs BEFORE the profile scope is entered:
        # _profile_scope must never be held across an await (the RLock is
        # reentrant per-thread, so a second coroutine interleaving on the
        # event-loop thread could cross-restore the module globals).
        if model and not body.confirm_expensive_model:
            try:
                from hermes_cli.model_cost_guard import expensive_model_warning

                # Pricing lookup can hit models.dev / a /models endpoint on a
                # cache miss — keep it off the event loop.
                warning = await asyncio.to_thread(
                    expensive_model_warning,
                    model,
                    provider=provider,
                    base_url=base_url,
                )
            except Exception:
                warning = None
            if warning is not None:
                return {
                    "ok": False,
                    "scope": scope,
                    "provider": provider,
                    "model": model,
                    "confirm_required": True,
                    "confirm_message": warning.message,
                }

        def _apply_assignment():
            with _profile_scope(body.profile or profile):
                return _apply_model_assignment_sync(
                    scope, provider, model, task, base_url, api_key
                )

        return await asyncio.to_thread(_apply_assignment)
    except HTTPException:
        raise
    except Exception:
        _log.exception("POST /api/model/set failed")
        raise HTTPException(status_code=500, detail="Failed to save model assignment")


def _apply_model_assignment_sync(
    scope: str, provider: str, model: str, task: str, base_url: str, api_key: str = ""
):
    """Synchronous body of POST /api/model/set.

    Runs inside ``_profile_scope`` (in a worker thread) so every
    load_config/save_config lands in the requested profile.  Raises
    HTTPException for validation errors — the async wrapper re-raises them.
    """
    cfg = load_config()

    if scope == "main":
        if not provider or not model:
            raise HTTPException(status_code=400, detail="provider and model required for main")
        provider, model = _normalize_main_model_assignment(provider, model)
        model_cfg = _apply_main_model_assignment(
            cfg.get("model", {}), provider, model, base_url, api_key
        )
        cfg["model"] = model_cfg

        # When switching the main provider to Nous, mirror the CLI's
        # post-model-selection behaviour (hermes_cli/main.py
        # prompt_enable_tool_gateway / tools_config apply_nous_managed_defaults):
        # auto-route any *unconfigured* tools through the Nous Tool Gateway.
        # This is purely additive — apply_nous_managed_defaults skips every
        # tool where the user already has a direct key (FIRECRAWL_API_KEY,
        # FAL_KEY, etc.) or an explicit backend/provider in config, so it
        # never overwrites a user's own setup. GUI users thus land on the
        # gateway the same way CLI users do, without a separate prompt.
        gateway_tools: list[str] = []
        if provider.strip().lower() == "nous":
            try:
                from hermes_cli.nous_subscription import apply_nous_managed_defaults
                from hermes_cli.tools_config import _get_platform_tools

                enabled = _get_platform_tools(
                    cfg, "cli", include_default_mcp_servers=False
                )
                changed = apply_nous_managed_defaults(
                    cfg,
                    enabled_toolsets=enabled,
                    force_fresh=True,
                )
                gateway_tools = sorted(changed)
            except Exception:
                # Portal lookup hiccups / non-subscriber / non-nous gating
                # must never block saving the model assignment.
                _log.debug("apply_nous_managed_defaults skipped", exc_info=True)

        save_config(cfg)

        # Register a named ``custom_providers`` entry for a custom/local
        # endpoint, mirroring the ``hermes model`` custom flow
        # (_save_custom_provider). Without this the endpoint only lives in
        # ``model.*`` and the picker has no proper ready row for it — the
        # GUI then surfaces a "needs setup" dead-end on the bare ``custom``
        # provider. Dedups by base_url, so re-saving is idempotent.
        if provider.strip().lower() in {"custom", "local"} and base_url:
            try:
                from hermes_cli.main import _auto_provider_name, _save_custom_provider

                _save_custom_provider(
                    base_url,
                    api_key,
                    model,
                    name=_auto_provider_name(base_url),
                )
            except Exception:
                # Never block the assignment on the bookkeeping write —
                # model.* is already persisted and routable.
                _log.debug("custom_providers registration skipped", exc_info=True)

        # Surface auxiliary slots still pinned to a *different* provider than
        # the new main one. Switching the main model does NOT touch aux pins
        # (they're independent, sticky per-task overrides — see
        # auxiliary_client._resolve_auto). A user who switches main away from
        # a now-unpaid provider (e.g. nous with $0 balance) keeps paying 402s
        # on every background aux call until they reset those pins. We never
        # auto-clear them — pinning aux to a cheaper/different model is a
        # legitimate config — but we tell the caller so the UI can offer a
        # "reset to main" nudge instead of silently burning credits.
        new_provider = provider.strip().lower()
        stale_aux: list[dict] = []
        aux_cfg = cfg.get("auxiliary", {})
        if isinstance(aux_cfg, dict):
            for slot in _AUX_TASK_SLOTS:
                slot_cfg = aux_cfg.get(slot)
                if not isinstance(slot_cfg, dict):
                    continue
                slot_provider = str(slot_cfg.get("provider", "") or "").strip()
                if (
                    slot_provider
                    and slot_provider.lower() not in {"auto", ""}
                    and slot_provider.lower() != new_provider
                ):
                    stale_aux.append({
                        "task": slot,
                        "provider": slot_provider,
                        "model": str(slot_cfg.get("model", "") or ""),
                    })

        return {
            "ok": True,
            "scope": "main",
            "provider": provider,
            "model": model,
            "base_url": model_cfg.get("base_url", ""),
            "gateway_tools": gateway_tools,
            "stale_aux": stale_aux,
        }

    # scope == "auxiliary"
    aux = cfg.get("auxiliary")
    if not isinstance(aux, dict):
        aux = {}

    if task == "__reset__":
        # Reset every slot to provider="auto", model="" — keeps other fields intact.
        for slot in _AUX_TASK_SLOTS:
            slot_cfg = aux.get(slot)
            if not isinstance(slot_cfg, dict):
                slot_cfg = {}
            slot_cfg["provider"] = "auto"
            slot_cfg["model"] = ""
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
            aux[slot] = slot_cfg
        cfg["auxiliary"] = aux
        save_config(cfg)
        return {"ok": True, "scope": "auxiliary", "reset": True}

    if not provider:
        raise HTTPException(status_code=400, detail="provider required for auxiliary")

    targets = [task] if task else list(_AUX_TASK_SLOTS)
    for slot in targets:
        if slot not in _AUX_TASK_SLOTS:
            raise HTTPException(status_code=400, detail=f"unknown auxiliary task: {slot}")
        slot_cfg = aux.get(slot)
        if not isinstance(slot_cfg, dict):
            slot_cfg = {}
        prev_provider = str(slot_cfg.get("provider") or "").strip().lower()
        new_provider = provider.strip().lower()
        slot_cfg["provider"] = provider
        slot_cfg["model"] = model
        if new_provider != prev_provider and new_provider != "custom":
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
        aux[slot] = slot_cfg

    cfg["auxiliary"] = aux
    save_config(cfg)
    return {
        "ok": True,
        "scope": "auxiliary",
        "tasks": targets,
        "provider": provider,
        "model": model,
    }




def _infer_provider_on_model_change(model_val: str, prev_provider: str) -> tuple[str, str]:
    """Infer which provider serves ``model_val`` when the flat Config-page Model
    field changes, given the previously-saved ``prev_provider``.

    Returns ``(provider, model)``; ``provider`` is empty when no switch is
    warranted (leave the existing provider untouched). Two signals, in order:

    1. Curated-catalog detection (``detect_provider_for_model``) — handles the
       ~28 OpenRouter-curated models and direct provider-static catalogs.
    2. Vendor-slug heuristic — a ``vendor/model`` slug cannot belong to a
       single-model / non-aggregator provider (e.g. ``ollama-local``). When the
       current provider is not an aggregator that serves vendor-prefixed slugs,
       route to an aggregator. ``_normalize_main_model_assignment`` (called by
       the caller) keeps the user's current aggregator when they're already on
       one, else falls back to openrouter — the same chokepoint logic as
       ``POST /api/model/set``.
    """
    name = (model_val or "").strip()
    if not name:
        return "", name
    try:
        from hermes_cli.models import (
            _AGGREGATOR_PROVIDERS,
            detect_provider_for_model,
            normalize_provider,
        )
    except Exception:
        return "", name

    try:
        detected = detect_provider_for_model(name, prev_provider)
    except Exception:
        detected = None
    if detected:
        return detected[0], detected[1]

    # Vendor-prefixed slug under a non-aggregator provider → reassign. Use a
    # sentinel "openrouter" here; _normalize_main_model_assignment resolves the
    # real aggregator (keeps a current aggregator, else openrouter).
    if "/" in name:
        try:
            cur_is_aggregator = normalize_provider(prev_provider) in _AGGREGATOR_PROVIDERS
        except Exception:
            cur_is_aggregator = False
        if not cur_is_aggregator:
            return "openrouter", name

    return "", name


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
                prev_default = str(disk_model.get("default") or "").strip()
                prev_provider = str(disk_model.get("provider") or "").strip()
                # When the model name actually changed, re-detect which
                # provider serves it. The Config-page Model field is a flat
                # string with no provider info, so without this a user who
                # picks an OpenRouter model while their default provider is
                # ollama-local keeps the stale provider and 404s. Only fires
                # on a real model change so saving unrelated config fields
                # never overwrites an explicit provider.
                if model_val != prev_default and prev_provider:
                    new_provider, resolved_model = _infer_provider_on_model_change(
                        model_val, prev_provider
                    )
                    if new_provider and new_provider.strip().lower() != prev_provider.lower():
                        # Route through the canonical assignment chokepoints so
                        # the model is normalized for the new provider and stale
                        # base_url/api_mode/api_key are cleared on the switch
                        # (and preserved on a same-provider re-pick).
                        norm_provider, norm_model = _normalize_main_model_assignment(
                            new_provider, resolved_model
                        )
                        disk_model = _apply_main_model_assignment(
                            disk_model, norm_provider, norm_model
                        )
                        model_val = norm_model
                # Preserve all subkeys, update default with the new value
                disk_model["default"] = model_val
                # Write context_length into the model dict (0 = remove/auto)
                if ctx_override > 0:
                    disk_model["context_length"] = ctx_override
                else:
                    disk_model.pop("context_length", None)
                config["model"] = disk_model
            # Model was previously a bare string — upgrade to dict if
            # user is setting a context_length override
            elif ctx_override > 0:
                config["model"] = {
                    "default": model_val,
                    "context_length": ctx_override,
                }
        except Exception:
            pass  # can't read disk config — just use the string form
    return config


@app.put("/api/config")
async def update_config(body: ConfigUpdate, profile: Optional[str] = None):
    try:
        with _profile_scope(body.profile or profile):
            # The dashboard form is schema-driven (see CONFIG_SCHEMA). Any root
            # key absent from the schema — most visibly ``custom_providers``, but
            # also ``agent.personalities``, ``terminal.lifetime_seconds``, etc. —
            # is not sent in the PUT body. A full-replace save would silently
            # drop those keys. Deep-merge incoming over what's on disk so the
            # frontend can only overwrite what it explicitly sends.
            existing = read_raw_config()
            incoming = _denormalize_config_from_web(body.config)
            save_config(_deep_merge(existing, incoming))
        return {"ok": True}
    except HTTPException:
        raise
    except Exception:
        _log.exception("PUT /api/config failed")
        raise HTTPException(status_code=500, detail="Internal server error")


def _catalog_provider_env_metadata() -> dict:
    """Map provider env vars → desktop card metadata, derived from the catalog.

    Returns ``{env_var: {provider, provider_label, description, url, is_password,
    advanced}}`` for every API-key provider in the unified ``provider_catalog()``
    (i.e. the ``hermes model`` universe). This is what lets the desktop Keys tab
    render a card for a provider even when its env var was never hand-added to
    ``OPTIONAL_ENV_VARS`` — closing the drift where CLI-configurable providers
    (openai-api, kilocode, novita, tencent-tokenhub, copilot, …) were missing
    from the GUI.

    Hand ``OPTIONAL_ENV_VARS`` prose is layered ON TOP of this in the endpoint;
    this only supplies membership + grouping + sensible fallbacks.
    """
    try:
        from hermes_cli.provider_catalog import provider_catalog
    except Exception:
        return {}

    # Env vars already declared with a NON-provider category (e.g. the shared
    # GITHUB_TOKEN, which is a Skills-Hub "tool" credential) must not be
    # promoted into a provider card. Copilot lists GITHUB_TOKEN among its auth
    # aliases, but its provider card uses the provider-owned COPILOT_GITHUB_TOKEN.
    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS as _OPT
    except Exception:
        _OPT = {}
    _non_provider_keys = {
        k for k, v in _OPT.items()
        if (v or {}).get("category") and (v or {}).get("category") != "provider"
    }

    meta: dict = {}
    for d in provider_catalog():
        if d.tab != "keys":
            continue
        # API-key vars: the first is the primary (password) field; any aliases
        # are kept as additional password fields so users can clear them too.
        for env_var in d.api_key_env_vars:
            if env_var in _non_provider_keys:
                continue  # don't hijack a shared tool/messaging credential
            meta.setdefault(
                env_var,
                {
                    "provider": d.slug,
                    "provider_label": d.label,
                    "description": d.description,
                    "url": d.signup_url or None,
                    "is_password": True,
                    "advanced": False,
                    "category": "provider",
                },
            )
        # Base-URL override is an advanced, non-secret field for the same card.
        if d.base_url_env_var:
            meta.setdefault(
                d.base_url_env_var,
                {
                    "provider": d.slug,
                    "provider_label": d.label,
                    "description": f"{d.label} base URL override",
                    "url": None,
                    "is_password": False,
                    "advanced": True,
                    "category": "provider",
                },
            )

        # AWS-SDK providers (Bedrock) authenticate via the AWS credential chain
        # rather than a pasted API key, so they have no api_key_env_vars. Tag
        # their AWS_* settings to the provider card so they still appear on the
        # Keys tab (otherwise Bedrock — a `hermes model` provider — would be
        # invisible in the desktop app).
        if d.auth_type == "aws_sdk":
            for aws_var in ("AWS_REGION", "AWS_PROFILE"):
                existing = meta.get(aws_var, {})
                meta[aws_var] = {
                    "provider": d.slug,
                    "provider_label": d.label,
                    "description": existing.get("description") or f"{d.label} ({aws_var})",
                    "url": existing.get("url"),
                    "is_password": False,
                    "advanced": existing.get("advanced", True),
                    "category": "provider",
                }

        # Vertex AI authenticates via OAuth2 (service-account JSON or ADC), not a
        # pasted API key, so it also has no api_key_env_vars. Tag its credential
        # env var to the provider card so it appears on the Keys tab (otherwise
        # Vertex — a `hermes model` provider — would be invisible in the desktop
        # app). The value is a filesystem path, not a secret string, so it is
        # not a password field.
        if d.auth_type == "vertex":
            existing = meta.get("VERTEX_CREDENTIALS_PATH", {})
            meta["VERTEX_CREDENTIALS_PATH"] = {
                "provider": d.slug,
                "provider_label": d.label,
                "description": existing.get("description")
                or f"{d.label} — service account JSON path (or use ADC)",
                "url": existing.get("url"),
                "is_password": False,
                "advanced": existing.get("advanced", True),
                "category": "provider",
            }
    return meta


@app.get("/api/env")
async def get_env_vars(profile: Optional[str] = None):
    with _profile_scope(profile):
        env_on_disk = load_env()
    channel_keys = _channel_managed_env_keys()
    catalog_meta = _catalog_provider_env_metadata()

    def _row(var_name: str, info: dict, *, custom: bool = False) -> dict:
        value = env_on_disk.get(var_name)
        cat_meta = catalog_meta.get(var_name) or {}
        # Hand OPTIONAL_ENV_VARS prose wins where present; the catalog fills any
        # gaps (description/url) and always supplies provider grouping hints.
        return {
            "is_set": bool(value),
            "redacted_value": redact_key(value) if value else None,
            "description": info.get("description") or cat_meta.get("description", ""),
            "url": info.get("url") if info.get("url") is not None else cat_meta.get("url"),
            "category": info.get("category") or cat_meta.get("category", ""),
            "is_password": info.get("password", cat_meta.get("is_password", False)),
            "tools": info.get("tools", []),
            "advanced": info.get("advanced", cat_meta.get("advanced", False)),
            # True when this var is a messaging-platform credential owned by a
            # Channels page card. The Keys/Env page uses this to hide it and
            # avoid duplicating the (richer) Channels configuration UI.
            "channel_managed": var_name in channel_keys,
            # Provider grouping hints derived from the unified provider catalog
            # so the desktop Keys tab groups by the SAME provider identity the
            # CLI `hermes model` picker uses (not desktop-only prefix guesses).
            "provider": cat_meta.get("provider", ""),
            "provider_label": cat_meta.get("provider_label", ""),
            # True when this key exists in the user's .env but is NOT in any
            # catalog (OPTIONAL_ENV_VARS or the provider catalog) — an
            # arbitrary/custom env var the user added directly. Surfaced so the
            # Keys page can list (and let the user manage) them instead of
            # hiding everything it doesn't recognise.
            "custom": custom,
        }

    result = {}
    for var_name, info in OPTIONAL_ENV_VARS.items():
        result[var_name] = _row(var_name, info)
    # Synthesize rows for catalog provider env vars that have no hand entry in
    # OPTIONAL_ENV_VARS — these are the providers that were CLI-configurable but
    # invisible in the desktop app until now.
    for var_name in catalog_meta:
        if var_name not in result:
            result[var_name] = _row(var_name, {})
    # Surface arbitrary/custom keys the user set in .env that aren't in any
    # catalog. These are always "set" (they're on disk). Treated as secrets by
    # default (is_password=True → redacted, reveal-gated) since an unrecognised
    # key could hold anything. Channel-managed credentials are excluded — those
    # belong to the Channels page. This makes the "add a custom key" surface
    # round-trip: a key added there reappears here under its own section.
    for var_name in env_on_disk:
        if var_name in result or var_name in channel_keys:
            continue
        row = _row(var_name, {}, custom=True)
        row["category"] = "custom"
        row["is_password"] = True
        result[var_name] = row
    return result


@app.put("/api/env")
async def set_env_var(body: EnvVarUpdate, profile: Optional[str] = None):
    try:
        with _profile_scope(body.profile or profile):
            save_env_value(body.key, body.value)
        return {"ok": True, "key": body.key}
    except ValueError as exc:
        # save_env_value raises ValueError for invalid names and for keys
        # on the denylist (LD_PRELOAD, PATH, PYTHONPATH, …). Surface the
        # message to the SPA so the user understands why the write was
        # refused instead of seeing an opaque 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception("PUT /api/env failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# Live credential probes keyed by env var. Each entry is (method, url, auth)
# where auth is "bearer" (Authorization header) or "query" (?key=). A cheap
# read-only models/key call that 401s on a bad token — enough to catch a
# mistyped key before it's persisted. Providers absent from this map (or local
# endpoints) are not network-validated; the client treats those as "unknown".
_CREDENTIAL_PROBES: dict[str, tuple[str, str]] = {
    "OPENROUTER_API_KEY": ("https://openrouter.ai/api/v1/key", "bearer"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1/models", "bearer"),
    "XAI_API_KEY": ("https://api.x.ai/v1/models", "bearer"),
    "GEMINI_API_KEY": ("https://generativelanguage.googleapis.com/v1beta/models", "query"),
}


def _parse_model_ids(resp: "Any") -> List[str]:
    """Extract model ids from an OpenAI-compatible ``/v1/models`` response.

    Tolerant of the common shapes: ``{"data": [{"id": ...}]}`` (OpenAI / vLLM /
    llama.cpp) and a bare ``{"data": ["id", ...]}``. Returns ``[]`` on any
    parse/HTTP error so a slightly non-standard endpoint never hard-blocks.
    """
    try:
        if not resp.is_success:
            return []
        payload = resp.json()
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict):
            mid = str(item.get("id") or "").strip()
        else:
            mid = str(item or "").strip()
        if mid:
            ids.append(mid)
    return ids


@app.post("/api/providers/validate")
async def validate_provider_credential(body: EnvVarUpdate, request: Request):
    """Live-probe a provider credential before it's saved.

    Returns {ok, reachable, message}. ok=True means the provider accepted the
    key; ok=False + reachable=True means the key is bad (caller should block);
    reachable=False means the network probe couldn't run (caller may save with
    a warning rather than hard-blocking offline users).
    """
    _require_token(request)
    import httpx

    key = (body.key or "").strip()
    value = (body.value or "").strip()
    if not value:
        return {"ok": False, "reachable": True, "message": "Enter a value first."}

    # Local / custom endpoint: validate connectivity, not auth — any HTTP
    # response (even 401) proves the endpoint is up. Also surface the model
    # ids the endpoint advertises (OpenAI ``/v1/models`` shape) so the GUI can
    # auto-pick a default without asking the user to type a model name.
    if key == "OPENAI_BASE_URL":
        url = value.rstrip("/") + "/models"
        # Send the optional API key so endpoints that require auth on
        # ``/v1/models`` (many hosted OpenAI-compatible servers) still enumerate
        # their models instead of returning an empty list behind a 401.
        api_key = (body.api_key or "").strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            with httpx.Client(timeout=httpx.Timeout(8.0)) as client:
                resp = client.get(url, headers=headers)
            return {"ok": True, "reachable": True, "message": "", "models": _parse_model_ids(resp)}
        except Exception:
            return {"ok": False, "reachable": False, "message": f"Could not reach {url}."}

    probe = _CREDENTIAL_PROBES.get(key)
    if not probe:
        # No probe for this provider — can't validate, don't block.
        return {"ok": True, "reachable": False, "message": ""}

    url, auth = probe
    headers = {"Accept": "application/json"}
    params = {}
    if auth == "bearer":
        headers["Authorization"] = f"Bearer {value}"
    else:
        params["key"] = value

    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            resp = client.get(url, headers=headers, params=params)
    except Exception:
        return {"ok": False, "reachable": False, "message": "Could not reach the provider to verify the key."}

    if resp.status_code in (401, 403):
        return {"ok": False, "reachable": True, "message": "That API key was rejected. Double-check it and try again."}
    if resp.status_code == 429 or resp.is_success:
        # 429 = key is valid but rate-limited; success = valid.
        return {"ok": True, "reachable": True, "message": ""}
    return {"ok": False, "reachable": True, "message": f"Provider returned HTTP {resp.status_code} for this key."}


@app.delete("/api/env")
async def remove_env_var(body: EnvVarDelete, profile: Optional[str] = None):
    try:
        with _profile_scope(body.profile or profile):
            removed = remove_env_value(body.key)
        if not removed:
            raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")
        return {"ok": True, "key": body.key}
    except HTTPException:
        raise
    except ValueError as exc:
        # remove_env_value raises ValueError for invalid key names. Surface
        # the message to the SPA so the user understands why the delete was
        # refused instead of seeing an opaque 500. Mirrors PUT /api/env.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception("DELETE /api/env failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/env/reveal")
async def reveal_env_var(
    body: EnvVarReveal, request: Request, profile: Optional[str] = None
):
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
        raise HTTPException(status_code=429, detail="Too many reveal requests. Try again shortly.")
    _reveal_timestamps.append(now)

    # --- Reveal ---
    with _profile_scope(body.profile or profile):
        env_on_disk = load_env()
    value = env_on_disk.get(body.key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")

    _log.info("env/reveal: %s", body.key)
    return {"key": body.key, "value": value}


# Entries omit fields they don't need to override; the catalog builder fills
# in env_vars from OPTIONAL_ENV_VARS via prefix matching when not specified,
# and pulls required_env from a plugin's PlatformEntry when available.
_PLATFORM_OVERRIDES: dict[str, dict[str, Any]] = {
    "telegram": {
        "name": "Telegram",
        "description": "Run Hermes from Telegram DMs, groups, and topics.",
        "docs_url": "https://core.telegram.org/bots/features#botfather",
        "env_vars": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_PROXY"),
        "required_env": ("TELEGRAM_BOT_TOKEN",),
    },
    "discord": {
        "name": "Discord",
        "description": "Connect Hermes to Discord DMs, channels, and threads.",
        "docs_url": "https://discord.com/developers/applications",
        "env_vars": (
            "DISCORD_BOT_TOKEN",
            "DISCORD_ALLOWED_USERS",
            "DISCORD_REPLY_TO_MODE",
        ),
        "required_env": ("DISCORD_BOT_TOKEN",),
    },
    "slack": {
        "name": "Slack",
        "description": "Use Hermes from Slack via Socket Mode. Add allowed Slack member IDs so connected bots can respond.",
        "docs_url": "https://api.slack.com/apps",
        "env_vars": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS"),
        "required_env": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"),
    },
    "mattermost": {
        "name": "Mattermost",
        "description": "Connect Hermes to Mattermost channels and direct messages.",
        "docs_url": "https://mattermost.com/deploy/",
        "env_vars": ("MATTERMOST_URL", "MATTERMOST_TOKEN", "MATTERMOST_ALLOWED_USERS"),
        "required_env": ("MATTERMOST_URL", "MATTERMOST_TOKEN"),
    },
    "matrix": {
        "name": "Matrix",
        "description": "Use Hermes in Matrix rooms and direct messages.",
        "docs_url": "https://matrix.org/ecosystem/servers/",
        "env_vars": (
            "MATRIX_HOMESERVER",
            "MATRIX_ACCESS_TOKEN",
            "MATRIX_USER_ID",
            "MATRIX_ALLOWED_USERS",
        ),
        "required_env": ("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN", "MATRIX_USER_ID"),
    },
    "signal": {
        "name": "Signal",
        "description": "Connect through a signal-cli REST bridge.",
        "docs_url": "https://github.com/bbernhard/signal-cli-rest-api",
        "env_vars": ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT", "SIGNAL_ALLOWED_USERS"),
        "required_env": ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"),
    },
    "whatsapp": {
        "name": "WhatsApp",
        "description": "Use Hermes through the bundled WhatsApp bridge with QR-based auth.",
        "docs_url": "https://github.com/tulir/whatsmeow",
        "env_vars": (
            "WHATSAPP_ENABLED",
            "WHATSAPP_MODE",
            "WHATSAPP_DM_POLICY",
            "WHATSAPP_ALLOWED_USERS",
        ),
        "required_env": (),
    },
    "homeassistant": {
        "name": "Home Assistant",
        "description": "Control your smart home from Hermes via Home Assistant.",
        "docs_url": "https://www.home-assistant.io/docs/authentication/",
        "env_vars": ("HASS_URL", "HASS_TOKEN"),
        "required_env": ("HASS_URL", "HASS_TOKEN"),
    },
    "email": {
        "name": "Email",
        "description": "Talk to Hermes through an IMAP/SMTP mailbox.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        "env_vars": (
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
        ),
        "required_env": (
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
        ),
    },
    "sms": {
        "name": "SMS (Twilio)",
        "description": "Send and receive text messages via Twilio.",
        "docs_url": "https://www.twilio.com/console",
        "env_vars": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
        "required_env": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
    },
    "dingtalk": {
        "name": "DingTalk",
        "description": "Connect Hermes to DingTalk groups (钉钉).",
        "docs_url": "https://open.dingtalk.com/document/orgapp/the-robot-development-process",
        "env_vars": ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
        "required_env": ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
    },
    "feishu": {
        "name": "Feishu / Lark",
        "description": "Use Hermes inside Feishu / Lark.",
        "docs_url": "https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/intro",
        "env_vars": (
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_ENCRYPT_KEY",
            "FEISHU_VERIFICATION_TOKEN",
        ),
        "required_env": ("FEISHU_APP_ID", "FEISHU_APP_SECRET"),
    },
    "google_chat": {
        "name": "Google Chat",
        "description": "Connect Hermes to Google Chat via Cloud Pub/Sub.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/google_chat",
    },
    "wecom": {
        "name": "WeCom (group bot)",
        "description": "Send-only WeCom group bot via webhook.",
        "docs_url": "https://developer.work.weixin.qq.com/document/path/91770",
        "env_vars": ("WECOM_BOT_ID", "WECOM_SECRET"),
        "required_env": ("WECOM_BOT_ID",),
    },
    "wecom_callback": {
        "name": "WeCom (app)",
        "description": "Two-way WeCom integration via callback app.",
        "docs_url": "https://developer.work.weixin.qq.com/document/path/90930",
        "env_vars": (
            "WECOM_CALLBACK_CORP_ID",
            "WECOM_CALLBACK_CORP_SECRET",
            "WECOM_CALLBACK_AGENT_ID",
            "WECOM_CALLBACK_TOKEN",
            "WECOM_CALLBACK_ENCODING_AES_KEY",
        ),
        "required_env": (
            "WECOM_CALLBACK_CORP_ID",
            "WECOM_CALLBACK_CORP_SECRET",
            "WECOM_CALLBACK_AGENT_ID",
        ),
    },
    "weixin": {
        "name": "Weixin / WeChat (Personal)",
        "description": "Connect a personal WeChat account through Tencent's iLink Bot API.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin/",
        "env_vars": ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN", "WEIXIN_BASE_URL"),
        "required_env": ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN"),
    },
    "bluebubbles": {
        "name": "BlueBubbles (iMessage)",
        "description": "Use Hermes through iMessage via a BlueBubbles server.",
        "docs_url": "https://bluebubbles.app/",
        "env_vars": (
            "BLUEBUBBLES_SERVER_URL",
            "BLUEBUBBLES_PASSWORD",
            "BLUEBUBBLES_ALLOWED_USERS",
        ),
        "required_env": ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"),
    },
    "qqbot": {
        "name": "QQ Bot",
        "description": "Connect Hermes to a QQ Bot from the QQ Open Platform.",
        "docs_url": "https://q.qq.com",
        "env_vars": ("QQ_APP_ID", "QQ_CLIENT_SECRET", "QQ_ALLOWED_USERS"),
        "required_env": ("QQ_APP_ID", "QQ_CLIENT_SECRET"),
    },
    # Teams ships as a platform plugin, so its name/env vars come from the
    # plugin registry. Only the docs link needs an override here so the
    # Channels page can point at the Microsoft Teams setup guide.
    "teams": {
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/teams",
    },
    "yuanbao": {
        "name": "Yuanbao (元宝)",
        "description": "Connect Hermes to Tencent Yuanbao.",
        "docs_url": "",
        "required_env": (),
    },
    "api_server": {
        "name": "API server",
        "description": "Expose Hermes as an OpenAI-compatible HTTP API for tools like Open WebUI.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        "env_vars": (
            "API_SERVER_ENABLED",
            "API_SERVER_KEY",
            "API_SERVER_PORT",
            "API_SERVER_HOST",
            "API_SERVER_MODEL_NAME",
        ),
        "required_env": (),
    },
    "webhook": {
        "name": "Webhooks",
        "description": "Receive events from GitHub, GitLab, and other webhook sources.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/",
        "env_vars": ("WEBHOOK_ENABLED", "WEBHOOK_PORT", "WEBHOOK_SECRET"),
        "required_env": (),
    },
}

# Display order: well-known platforms surface first; unknown plugins fall to
# the end alphabetically.
_PLATFORM_ORDER: tuple[str, ...] = (
    "telegram",
    "discord",
    "slack",
    "mattermost",
    "matrix",
    "whatsapp",
    "signal",
    "bluebubbles",
    "homeassistant",
    "email",
    "sms",
    "dingtalk",
    "feishu",
    "google_chat",
    "wecom",
    "wecom_callback",
    "weixin",
    "qqbot",
    "yuanbao",
    "api_server",
    "webhook",
)

# Display labels for env vars not in OPTIONAL_ENV_VARS (HOME_CHANNEL_*, bridge
# toggles, Twilio, HASS, Email, etc.). Anything missing from OPTIONAL_ENV_VARS
# falls back here so the UI can still render a friendly label.
_MESSAGING_ENV_FALLBACKS: dict[str, dict[str, Any]] = {
    "SIGNAL_HTTP_URL": {
        "description": "signal-cli REST API base URL, e.g. http://127.0.0.1:8080",
        "prompt": "Signal bridge URL",
        "url": "https://github.com/bbernhard/signal-cli-rest-api",
    },
    "SIGNAL_ACCOUNT": {
        "description": "Signal account phone number registered with the bridge",
        "prompt": "Signal account",
    },
    "SIGNAL_ALLOWED_USERS": {
        "description": "Comma-separated Signal users allowed to use the bot",
        "prompt": "Allowed Signal users",
    },
    "WHATSAPP_ENABLED": {
        "description": "Enable the WhatsApp gateway adapter",
        "prompt": "Enable WhatsApp",
        "advanced": True,
    },
    "WHATSAPP_MODE": {
        "description": "WhatsApp bridge mode",
        "prompt": "WhatsApp mode",
        "advanced": True,
    },
    "WHATSAPP_DM_POLICY": {
        "description": "How WhatsApp direct messages are authorized",
        "prompt": "WhatsApp DM policy",
        "advanced": True,
    },
    "WHATSAPP_ALLOWED_USERS": {
        "description": "Comma-separated WhatsApp users allowed to use the bot",
        "prompt": "Allowed WhatsApp users",
    },
    "HASS_URL": {
        "description": "Home Assistant base URL, e.g. https://homeassistant.local:8123",
        "prompt": "Home Assistant URL",
    },
    "HASS_TOKEN": {
        "description": "Long-lived access token from Home Assistant (Profile → Security)",
        "prompt": "Home Assistant access token",
        "password": True,
    },
    "EMAIL_ADDRESS": {
        "description": "Email address to send and receive from",
        "prompt": "Email address",
    },
    "EMAIL_PASSWORD": {
        "description": "Email account password or app password",
        "prompt": "Email password",
        "password": True,
    },
    "EMAIL_IMAP_HOST": {
        "description": "IMAP server host (e.g. imap.gmail.com)",
        "prompt": "IMAP host",
    },
    "EMAIL_SMTP_HOST": {
        "description": "SMTP server host (e.g. smtp.gmail.com)",
        "prompt": "SMTP host",
    },
    "TWILIO_ACCOUNT_SID": {
        "description": "Twilio Account SID",
        "prompt": "Twilio Account SID",
        "url": "https://www.twilio.com/console",
    },
    "TWILIO_AUTH_TOKEN": {
        "description": "Twilio Auth Token",
        "prompt": "Twilio Auth Token",
        "password": True,
    },
    "WECOM_BOT_ID": {"description": "WeCom group bot ID", "prompt": "WeCom Bot ID"},
    "WECOM_SECRET": {
        "description": "WeCom group bot secret",
        "prompt": "WeCom Secret",
        "password": True,
    },
    "WECOM_CALLBACK_CORP_ID": {
        "description": "WeCom corp ID",
        "prompt": "WeCom Corp ID",
    },
    "WECOM_CALLBACK_CORP_SECRET": {
        "description": "WeCom app corp secret",
        "prompt": "WeCom Corp Secret",
        "password": True,
    },
    "WECOM_CALLBACK_AGENT_ID": {
        "description": "WeCom app agent ID",
        "prompt": "WeCom Agent ID",
    },
    "WECOM_CALLBACK_TOKEN": {
        "description": "WeCom callback verification token",
        "prompt": "WeCom Token",
    },
    "WECOM_CALLBACK_ENCODING_AES_KEY": {
        "description": "WeCom callback AES encoding key",
        "prompt": "WeCom AES Key",
        "password": True,
    },
    "WEIXIN_ACCOUNT_ID": {
        "description": "iLink Bot account ID obtained through QR login in hermes gateway setup",
        "prompt": "iLink Bot account ID",
    },
    "WEIXIN_TOKEN": {
        "description": "iLink Bot token obtained through QR login in hermes gateway setup",
        "prompt": "iLink Bot token",
        "password": True,
    },
    "WEIXIN_BASE_URL": {
        "description": "iLink API base URL saved by QR login (default: https://ilinkai.weixin.qq.com)",
        "prompt": "iLink API base URL",
    },
    "FEISHU_APP_ID": {"description": "Feishu / Lark app ID", "prompt": "App ID"},
    "FEISHU_APP_SECRET": {
        "description": "Feishu / Lark app secret",
        "prompt": "App secret",
        "password": True,
    },
    "FEISHU_ENCRYPT_KEY": {
        "description": "Feishu / Lark encrypt key",
        "prompt": "Encrypt key",
        "password": True,
    },
    "FEISHU_VERIFICATION_TOKEN": {
        "description": "Feishu / Lark verification token",
        "prompt": "Verification token",
        "password": True,
    },
    "DINGTALK_CLIENT_ID": {
        "description": "DingTalk client ID (App key)",
        "prompt": "Client ID",
    },
    "DINGTALK_CLIENT_SECRET": {
        "description": "DingTalk client secret (App secret)",
        "prompt": "Client secret",
        "password": True,
    },
}


def _messaging_platform_catalog() -> tuple[dict[str, Any], ...]:
    """Build the messaging catalog from the gateway's Platform enum + plugin registry.

    Built-in platforms come from ``gateway.config.Platform`` (LOCAL is excluded).
    Plugin platforms come from ``gateway.platform_registry.plugin_entries()``,
    which lets newly installed adapters (e.g. IRC) appear without a code change
    here. Per-platform UI metadata (description, docs URL, env-var picks) lives
    in :data:`_PLATFORM_OVERRIDES`; anything not overridden gets reasonable
    defaults derived from the platform id and required_env.
    """
    from gateway.config import Platform

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []

    for member in Platform.__members__.values():
        if member.value == "local":
            continue
        if member.value in seen:
            continue
        seen.add(member.value)
        entries.append(_build_catalog_entry(member.value))

    try:
        from gateway.platform_registry import platform_registry

        for plugin_entry in platform_registry.plugin_entries():
            if plugin_entry.name in seen:
                continue
            seen.add(plugin_entry.name)
            entries.append(_build_catalog_entry(plugin_entry.name, plugin_entry))
    except Exception:
        _log.debug("plugin platform registry unavailable", exc_info=True)

    order = {pid: idx for idx, pid in enumerate(_PLATFORM_ORDER)}
    entries.sort(
        key=lambda e: (order.get(e["id"], len(_PLATFORM_ORDER)), e["name"].lower())
    )
    return tuple(entries)


def _channel_managed_env_keys() -> frozenset[str]:
    """Env-var keys owned by a Channels page platform card.

    The Channels page is the canonical surface for configuring messaging
    platform credentials (with connection status, test, enable toggle and
    gateway restart). The Keys/Env page consults this set to hide those vars
    so the same fields aren't duplicated in a plainer UI. Best-effort: if the
    gateway catalog can't be built, nothing is flagged and Keys shows it all.
    """
    try:
        keys: set[str] = set()
        for entry in _messaging_platform_catalog():
            keys.update(entry.get("env_vars", ()))
        return frozenset(keys)
    except Exception:
        _log.debug("could not build channel-managed env key set", exc_info=True)
        return frozenset()


# Cross-cutting gateway / relay knobs stay on the Keys → Settings tab even though
# they use the ``messaging`` category in OPTIONAL_ENV_VARS. Platform-scoped vars
# (``DISCORD_*``, ``MATRIX_*``, …) are owned by the Messaging UI instead.
_MESSAGING_KEYS_PAGE_KEYS = frozenset({
    "GATEWAY_ALLOW_ALL_USERS",
    "GATEWAY_PROXY_KEY",
    "GATEWAY_PROXY_URL",
})


def _platform_env_prefixes(platform_id: str) -> tuple[str, ...]:
    """Env-var prefixes owned by a messaging platform card."""
    aliases: dict[str, tuple[str, ...]] = {
        "email": ("EMAIL_",),
        "homeassistant": ("HASS_",),
        "qqbot": ("QQ_", "QQBOT_"),
        "sms": ("TWILIO_",),
        "wecom": ("WECOM_BOT_", "WECOM_SECRET"),
        "wecom_callback": ("WECOM_CALLBACK_",),
    }
    if platform_id in aliases:
        return aliases[platform_id]
    return (platform_id.upper().replace("-", "_") + "_",)


def _discover_platform_env_vars(platform_id: str) -> tuple[str, ...]:
    """All messaging-category env vars for a platform (override + plugin + prefix)."""
    prefixes = _platform_env_prefixes(platform_id)
    keys: list[str] = []
    for name, info in OPTIONAL_ENV_VARS.items():
        if info.get("category") != "messaging":
            continue
        if name in _MESSAGING_KEYS_PAGE_KEYS:
            continue
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        keys.append(name)
    return tuple(sorted(set(keys)))


def _merge_platform_env_vars(
    platform_id: str,
    override: dict[str, Any],
    plugin_entry: Any | None,
) -> tuple[str, ...]:
    """Canonical env-var list for a messaging platform card."""
    discovered = _discover_platform_env_vars(platform_id)
    if "env_vars" in override:
        return tuple(dict.fromkeys((*override["env_vars"], *discovered)))
    if plugin_entry is not None and plugin_entry.required_env:
        return tuple(dict.fromkeys((*tuple(plugin_entry.required_env), *discovered)))
    return discovered


def _build_catalog_entry(
    platform_id: str, plugin_entry: Any | None = None
) -> dict[str, Any]:
    override = _PLATFORM_OVERRIDES.get(platform_id, {})

    env_vars = _merge_platform_env_vars(platform_id, override, plugin_entry)

    if "required_env" in override:
        required_env = tuple(override["required_env"])
    elif plugin_entry is not None:
        required_env = tuple(plugin_entry.required_env or ())
    else:
        required_env = ()

    if override.get("name"):
        name = override["name"]
    elif plugin_entry is not None and plugin_entry.label:
        name = plugin_entry.label
    else:
        name = platform_id.replace("_", " ").title()

    description = override.get("description")
    if not description and plugin_entry is not None:
        description = plugin_entry.install_hint or ""

    return {
        "id": platform_id,
        "name": name,
        "description": description or "",
        "docs_url": override.get("docs_url", ""),
        "env_vars": env_vars,
        "required_env": required_env,
    }


def _catalog_lookup(platform_id: str) -> dict[str, Any] | None:
    for entry in _messaging_platform_catalog():
        if entry["id"] == platform_id:
            return entry
    return None


def _messaging_env_info(key: str) -> dict[str, Any]:
    info = OPTIONAL_ENV_VARS.get(key) or _MESSAGING_ENV_FALLBACKS.get(key) or {}
    return {
        "description": info.get("description", ""),
        "prompt": info.get("prompt", key),
        "help": info.get("help", ""),
        "url": info.get("url"),
        "is_password": info.get("password", False),
        "advanced": info.get("advanced", False),
    }


def _gateway_platform_config(platform_id: str):
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    platform = Platform(platform_id)
    platform_config = config.platforms.get(platform)
    return config, platform, platform_config


def _messaging_platform_payload(
    entry: dict[str, Any],
    env_on_disk: dict[str, str],
    runtime: dict | None,
    scoped: bool = False,
) -> dict[str, Any]:
    platform_id = entry["id"]
    runtime_platforms = runtime.get("platforms") if runtime else {}
    runtime_platform = (
        runtime_platforms.get(platform_id, {})
        if isinstance(runtime_platforms, dict)
        else {}
    )
    gateway_running = (
        get_running_pid() is not None
        or get_runtime_status_running_pid(runtime) is not None
    )
    env_vars = []

    for key in entry["env_vars"]:
        # When profile-scoped, judge only the profile's own .env — the
        # dashboard process's os.environ carries the ROOT install's .env
        # (loaded at startup) and would falsely report the root credentials
        # as the profile's.
        value = env_on_disk.get(key) or ("" if scoped else os.getenv(key, ""))
        env_vars.append(
            {
                "key": key,
                "required": key in entry["required_env"],
                "is_set": bool(value),
                "redacted_value": redact_key(value) if value else None,
                **_messaging_env_info(key),
            }
        )

    if scoped:
        # Profile-scoped view: derive enablement/configuration from the
        # profile's config.yaml + .env only. load_gateway_config()'s
        # env-override layer reads os.environ and would leak the root
        # install's tokens into the profile's reported state.
        try:
            cfg = load_config()
            platforms_cfg = cfg.get("platforms") or {}
            plat_cfg = platforms_cfg.get(platform_id)
            if not isinstance(plat_cfg, dict):
                plat_cfg = {}
            enabled = bool(plat_cfg.get("enabled"))
            hc = plat_cfg.get("home_channel")
            home_channel = hc if isinstance(hc, dict) else None
        except Exception:
            enabled = False
            home_channel = None
        configured = all(env_on_disk.get(key) for key in entry["required_env"])
    else:
        try:
            gateway_config, platform, platform_config = _gateway_platform_config(
                platform_id
            )
            enabled = bool(platform_config and platform_config.enabled)
            configured = bool(
                platform_config
                and gateway_config._is_platform_connected(platform, platform_config)
            )
            home_channel = (
                platform_config.home_channel.to_dict()
                if platform_config and platform_config.home_channel
                else None
            )
        except Exception:
            enabled = False
            configured = all(
                env_on_disk.get(key) or os.getenv(key, "")
                for key in entry["required_env"]
            )
            home_channel = None

    state = (
        runtime_platform.get("state") if isinstance(runtime_platform, dict) else None
    )
    runtime_gateway_state = runtime.get("gateway_state") if isinstance(runtime, dict) else None
    runtime_gateway_error = runtime.get("exit_reason") if isinstance(runtime, dict) else None
    if not enabled:
        state = "disabled"
    elif not configured:
        state = "not_configured"
    elif gateway_running and not state:
        state = "pending_restart"
    elif (
        not gateway_running
        and not state
        and runtime_gateway_state == "startup_failed"
    ):
        state = "startup_failed"
    elif not gateway_running and not state:
        state = "gateway_stopped"

    error_code = (
        runtime_platform.get("error_code")
        if isinstance(runtime_platform, dict)
        else None
    )
    error_message = (
        runtime_platform.get("error_message")
        if isinstance(runtime_platform, dict)
        else None
    )
    if state == "startup_failed":
        error_code = error_code or "startup_failed"
        error_message = error_message or runtime_gateway_error

    whatsapp_setup = None
    if platform_id == "whatsapp":
        whatsapp_mode = (
            env_on_disk.get("WHATSAPP_MODE")
            or ("" if scoped else os.getenv("WHATSAPP_MODE", ""))
        ).strip()
        allowed_users_value = (
            env_on_disk.get("WHATSAPP_ALLOWED_USERS")
            or ("" if scoped else os.getenv("WHATSAPP_ALLOWED_USERS", ""))
        ).strip()
        whatsapp_setup = {
            "mode": whatsapp_mode if whatsapp_mode in {"bot", "self-chat"} else "",
            "allowed_users_set": bool(allowed_users_value),
            "home_channel_set": bool(home_channel),
        }

    payload = {
        "id": platform_id,
        "name": entry["name"],
        "description": entry["description"],
        "docs_url": entry["docs_url"],
        "enabled": enabled,
        "configured": configured,
        "gateway_running": gateway_running,
        "state": state,
        "error_code": error_code,
        "error_message": error_message,
        "updated_at": (
            runtime_platform.get("updated_at")
            if isinstance(runtime_platform, dict)
            else None
        ),
        "home_channel": home_channel,
        "env_vars": env_vars,
    }
    if whatsapp_setup is not None:
        payload["whatsapp_setup"] = whatsapp_setup
    return payload


def _write_platform_enabled(platform_id: str, enabled: bool) -> None:
    write_platform_config_field(platform_id, "enabled", enabled)


_WHATSAPP_ONBOARDING_TTL_SECONDS = 600
_WHATSAPP_ONBOARDING_TERMINAL_STATUSES = {"connected", "error", "expired", "cancelled"}


@dataclass
class _WhatsAppOnboardingSession:
    proc: subprocess.Popen | None
    mode: str
    allowed_users: str
    session_path: str
    expires_at: str
    expires_at_ts: float
    profile: str | None = None
    status: str = "starting"
    qr_payload: str | None = None
    account_id: str | None = None
    account_name: str | None = None
    account_phone: str | None = None
    error: str | None = None


_whatsapp_onboarding_sessions: dict[str, _WhatsAppOnboardingSession] = {}
_whatsapp_onboarding_lock = threading.RLock()


def _utc_iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_whatsapp_onboarding_mode(value: Any) -> str:
    mode = str(value or "bot").strip().lower()
    if mode not in {"bot", "self-chat"}:
        raise HTTPException(status_code=400, detail="WhatsApp mode must be 'bot' or 'self-chat'.")
    return mode


def _normalize_whatsapp_allowed_users(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return ",".join(part.replace(" ", "") for part in raw.split(",") if part.strip())


def _whatsapp_session_path() -> Path:
    from hermes_constants import get_hermes_dir

    return get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")


def _whatsapp_phone_from_identifier(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = raw.split("@", 1)[0].split(":", 1)[0]
    digits = re.sub(r"\D+", "", candidate)
    return digits or None


def _whatsapp_linked_account_from_session(session_path: Path) -> tuple[str | None, str | None, str | None]:
    creds_path = session_path / "creds.json"
    try:
        payload = json.loads(creds_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None

    account_id: str | None = None
    account_name: str | None = None

    def collect(candidate: Any) -> None:
        nonlocal account_id, account_name
        if not isinstance(candidate, dict):
            return
        if account_id is None:
            for key in ("id", "jid", "lid"):
                value = str(candidate.get(key) or "").strip()
                if value:
                    account_id = value
                    break
        if account_name is None:
            for key in ("name", "verifiedName", "notify", "pushName"):
                value = str(candidate.get(key) or "").strip()
                if value:
                    account_name = value
                    break

    collect(payload.get("me"))
    collect(payload.get("account"))
    collect(payload)
    return account_id, account_name, _whatsapp_phone_from_identifier(account_id)


def _ensure_whatsapp_bridge_dependencies(bridge_dir: Path) -> None:
    """Install bridge dependencies when the dashboard is the setup surface."""
    if (bridge_dir / "node_modules").exists():
        return

    from hermes_constants import find_node_executable, with_hermes_node_path
    from utils import env_int

    npm = find_node_executable("npm")
    if not npm:
        raise HTTPException(
            status_code=500,
            detail="npm was not found. WhatsApp setup needs Node.js and npm.",
        )

    timeout = env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300)
    try:
        result = subprocess.run(
            [npm, "install", "--silent"],
            cwd=str(bridge_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=with_hermes_node_path(),
            creationflags=windows_hide_flags(),
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=500,
            detail="Installing WhatsApp bridge dependencies timed out.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to install WhatsApp bridge dependencies: {exc}",
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            detail = "\n".join(detail.splitlines()[-10:])
        raise HTTPException(
            status_code=500,
            detail=f"npm install failed for WhatsApp bridge: {detail or 'no output'}",
        )


def _spawn_whatsapp_pairing_process(session_path: Path, mode: str) -> subprocess.Popen:
    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    from hermes_constants import find_node_executable, with_hermes_node_path

    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"
    if not bridge_script.exists():
        raise HTTPException(
            status_code=500,
            detail=f"WhatsApp bridge script was not found at {bridge_script}.",
        )
    node = find_node_executable("node")
    if not node:
        raise HTTPException(
            status_code=500,
            detail="Node.js was not found. WhatsApp setup needs Node.js.",
        )

    _ensure_whatsapp_bridge_dependencies(bridge_dir)
    session_path.mkdir(parents=True, exist_ok=True)

    env = with_hermes_node_path()
    env["WHATSAPP_MODE"] = mode
    env["WHATSAPP_DM_POLICY"] = "pairing"
    return subprocess.Popen(
        [
            node,
            str(bridge_script),
            "--pair-only",
            "--pair-json",
            "--session",
            str(session_path),
        ],
        cwd=str(bridge_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        env=env,
        creationflags=windows_hide_flags(),
    )


def _terminate_whatsapp_pairing(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _watch_whatsapp_pairing(pairing_id: str, proc: subprocess.Popen) -> None:
    try:
        stream = proc.stdout
        if stream is not None:
            for line in stream:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event = str(payload.get("event") or "").strip()
                with _whatsapp_onboarding_lock:
                    record = _whatsapp_onboarding_sessions.get(pairing_id)
                    if not record or record.proc is not proc:
                        return
                    if event == "qr":
                        qr = str(payload.get("qr") or "").strip()
                        if qr:
                            record.qr_payload = qr
                            record.status = "waiting"
                            record.error = None
                    elif event == "connected":
                        user = payload.get("user")
                        if isinstance(user, dict):
                            account_id = str(user.get("id") or "").strip()
                            account_name = str(user.get("name") or "").strip()
                            record.account_id = account_id or None
                            record.account_name = account_name or None
                            record.account_phone = _whatsapp_phone_from_identifier(account_id)
                        record.status = "connected"
                        record.error = None
                    elif event == "error":
                        record.status = "error"
                        record.error = str(payload.get("error") or "WhatsApp pairing failed.")
                    elif event == "disconnected" and record.status == "starting":
                        record.status = "waiting"
        returncode = proc.wait()
    except Exception as exc:
        with _whatsapp_onboarding_lock:
            record = _whatsapp_onboarding_sessions.get(pairing_id)
            if record and record.proc is proc and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
                record.status = "error"
                record.error = str(exc)
        return

    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.proc is not proc:
            return
        if record.status in {"connected", "cancelled", "expired"}:
            return
        record.status = "error"
        record.error = (
            "WhatsApp pairing process exited before pairing completed."
            if returncode == 0
            else f"WhatsApp pairing process exited with code {returncode}."
        )


def _run_whatsapp_pairing(pairing_id: str, session_path: Path, mode: str) -> None:
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            return
        record.status = "installing"

    try:
        proc = _spawn_whatsapp_pairing_process(session_path, mode)
    except Exception as exc:
        with _whatsapp_onboarding_lock:
            record = _whatsapp_onboarding_sessions.get(pairing_id)
            if record and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
                record.status = "error"
                record.error = str(exc)
        return

    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            _terminate_whatsapp_pairing(proc)
            return
        record.proc = proc
        record.status = "starting"

    _watch_whatsapp_pairing(pairing_id, proc)


def _prune_whatsapp_onboarding_sessions() -> None:
    now = time.time()
    remove_ids: list[str] = []
    for pairing_id, record in _whatsapp_onboarding_sessions.items():
        if (
            record.proc is not None
            and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES
            and record.proc.poll() is not None
        ):
            record.status = "error"
            record.error = "WhatsApp pairing process exited before pairing completed."
        if record.expires_at_ts <= now and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            _terminate_whatsapp_pairing(record.proc)
            record.status = "expired"
            record.error = "WhatsApp QR setup expired. Start a new setup."
        if record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES and record.expires_at_ts + 300 <= now:
            remove_ids.append(pairing_id)
    for pairing_id in remove_ids:
        _whatsapp_onboarding_sessions.pop(pairing_id, None)


def _supersede_whatsapp_onboarding_sessions(session_path: Path) -> None:
    for existing in _whatsapp_onboarding_sessions.values():
        if existing.session_path == str(session_path) and existing.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            existing.status = "cancelled"
            existing.error = "Superseded by a newer WhatsApp setup session."
            _terminate_whatsapp_pairing(existing.proc)


def _whatsapp_onboarding_payload(pairing_id: str, record: _WhatsAppOnboardingSession) -> dict[str, Any]:
    return {
        "pairing_id": pairing_id,
        "status": record.status,
        "qr_payload": record.qr_payload,
        "expires_at": record.expires_at,
        "mode": record.mode,
        "allowed_users": record.allowed_users,
        "account_id": record.account_id,
        "account_name": record.account_name,
        "account_phone": record.account_phone,
        "error": record.error,
    }


def _restart_gateway_after_whatsapp_onboarding(profile: Optional[str] = None) -> dict[str, Any]:
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after WhatsApp onboarding")
        return {
            "restart_started": False,
            "restart_error": str(exc),
        }
    if reused:
        _log.info(
            "WhatsApp onboarding: reusing in-flight gateway restart (pid %s)",
            proc.pid,
        )
    return {
        "restart_started": True,
        "restart_action": "gateway-restart",
        "restart_pid": proc.pid,
    }


@app.post("/api/messaging/whatsapp/onboarding/start")
async def start_whatsapp_onboarding(body: WhatsAppOnboardingStart):
    mode = _normalize_whatsapp_onboarding_mode(body.mode)
    allowed_users = _normalize_whatsapp_allowed_users(body.allowed_users)
    effective_profile = body.profile

    with _config_profile_scope(effective_profile):
        session_path = _whatsapp_session_path()
        expires_at_ts = time.time() + _WHATSAPP_ONBOARDING_TTL_SECONDS
        expires_at = _utc_iso_from_ts(expires_at_ts)
        if (session_path / "creds.json").exists():
            pairing_id = secrets.token_urlsafe(16)
            account_id, account_name, account_phone = _whatsapp_linked_account_from_session(session_path)
            record = _WhatsAppOnboardingSession(
                proc=None,
                mode=mode,
                allowed_users=allowed_users,
                session_path=str(session_path),
                expires_at=expires_at,
                expires_at_ts=expires_at_ts,
                profile=effective_profile,
                status="connected",
                account_id=account_id,
                account_name=account_name,
                account_phone=account_phone,
            )
            with _whatsapp_onboarding_lock:
                _prune_whatsapp_onboarding_sessions()
                _supersede_whatsapp_onboarding_sessions(session_path)
                _whatsapp_onboarding_sessions[pairing_id] = record
            return _whatsapp_onboarding_payload(pairing_id, record)

    pairing_id = secrets.token_urlsafe(16)
    record = _WhatsAppOnboardingSession(
        proc=None,
        mode=mode,
        allowed_users=allowed_users,
        session_path=str(session_path),
        expires_at=expires_at,
        expires_at_ts=expires_at_ts,
        profile=effective_profile,
    )

    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        _supersede_whatsapp_onboarding_sessions(session_path)
        _whatsapp_onboarding_sessions[pairing_id] = record

    threading.Thread(
        target=_run_whatsapp_pairing,
        args=(pairing_id, session_path, mode),
        daemon=True,
    ).start()

    return _whatsapp_onboarding_payload(pairing_id, record)


@app.get("/api/messaging/whatsapp/onboarding/{pairing_id}")
async def get_whatsapp_onboarding_status(pairing_id: str):
    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="WhatsApp setup session was not found. Start a new setup.",
            )
        if record.status == "expired":
            raise HTTPException(status_code=410, detail=record.error or "WhatsApp setup expired.")
        return _whatsapp_onboarding_payload(pairing_id, record)


@app.post("/api/messaging/whatsapp/onboarding/{pairing_id}/apply")
async def apply_whatsapp_onboarding(
    pairing_id: str, body: WhatsAppOnboardingApply, profile: Optional[str] = None
):
    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="WhatsApp setup session was not found. Start a new setup.",
            )
        if record.status != "connected":
            raise HTTPException(status_code=409, detail="WhatsApp setup is not connected yet.")
        mode = _normalize_whatsapp_onboarding_mode(body.mode or record.mode)
        allowed_users = _normalize_whatsapp_allowed_users(
            record.allowed_users if body.allowed_users is None else body.allowed_users
        )
        if mode == "self-chat" and not allowed_users:
            allowed_users = record.account_phone or record.account_id or ""
        record_profile = record.profile

    effective_profile = body.profile or profile or record_profile
    try:
        with _config_profile_scope(effective_profile):
            save_env_value("WHATSAPP_MODE", mode)
            save_env_value("WHATSAPP_DM_POLICY", "pairing")
            if allowed_users:
                save_env_value("WHATSAPP_ALLOWED_USERS", allowed_users)
            # Blank means "keep the existing allowlist"; explicit clearing
            # still lives in the normal config editor where the field is visible.
            save_env_value("WHATSAPP_ENABLED", "true")
            _write_platform_enabled("whatsapp", True)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("WhatsApp onboarding apply failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to save WhatsApp setup.",
        ) from exc

    with _whatsapp_onboarding_lock:
        _whatsapp_onboarding_sessions.pop(pairing_id, None)

    restart_result = _restart_gateway_after_whatsapp_onboarding(effective_profile)
    return {
        "ok": True,
        "platform": "whatsapp",
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@app.delete("/api/messaging/whatsapp/onboarding/{pairing_id}")
async def cancel_whatsapp_onboarding(pairing_id: str):
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.pop(pairing_id, None)
    if record:
        record.status = "cancelled"
        _terminate_whatsapp_pairing(record.proc)
    return {"ok": True}


_TELEGRAM_ONBOARDING_DEFAULT_URL = "https://setup.hermes-agent.nousresearch.com"
_TELEGRAM_ONBOARDING_USER_AGENT = f"HermesDashboard/{__version__}"
_TELEGRAM_USER_ID_RE = re.compile(r"^\d+$")


@dataclass
class _TelegramOnboardingPairing:
    poll_token: str
    expires_at: str
    expires_at_ts: float
    bot_token: str | None = None
    bot_username: str | None = None
    owner_user_id: str | None = None


_telegram_onboarding_pairings: dict[str, _TelegramOnboardingPairing] = {}
_telegram_onboarding_lock = threading.RLock()


def _telegram_onboarding_base_url() -> str:
    return (
        os.getenv("TELEGRAM_ONBOARDING_URL", _TELEGRAM_ONBOARDING_DEFAULT_URL)
        .strip()
        .rstrip("/")
    )


def _parse_expiry_ts(value: str) -> float:
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return time.time() + 600


def _prune_telegram_onboarding_pairings() -> None:
    now = time.time()
    expired = [
        pairing_id
        for pairing_id, record in _telegram_onboarding_pairings.items()
        if record.expires_at_ts <= now
    ]
    for pairing_id in expired:
        _telegram_onboarding_pairings.pop(pairing_id, None)


def _normalize_telegram_user_id(value: Any) -> str | None:
    normalized = str(value or "").strip()
    if _TELEGRAM_USER_ID_RE.fullmatch(normalized):
        return normalized
    return None


def _telegram_onboarding_error_message(error: str, fallback: str) -> str:
    return {
        "not_found": "Telegram pairing was not found. Start a new setup.",
        "expired": "Telegram setup expired. Start a new setup.",
        "claimed": "Telegram setup was already claimed. Start a new setup.",
        "unauthorized": "Telegram setup service rejected this request.",
        "telegram_manager_bot_token_not_configured": "Telegram setup service is not configured.",
        "telegram_token_fetch_failed": "Telegram could not finish bot setup. Try again.",
    }.get(error, fallback)


def _telegram_onboarding_request_sync(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    import httpx

    headers = {
        "Accept": "application/json",
        "User-Agent": _TELEGRAM_ONBOARDING_USER_AGENT,
    }
    request_kwargs: dict[str, Any] = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
        request_kwargs["json"] = body
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    url = f"{_telegram_onboarding_base_url()}{path}"
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.request(
                method,
                url,
                headers=headers,
                **request_kwargs,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            parsed = exc.response.json()
        except Exception:
            parsed = {}
        error = str(parsed.get("error") or parsed.get("status") or "")
        detail = _telegram_onboarding_error_message(
            error,
            "Telegram setup service returned an error.",
        )
        status_code = 404 if exc.response.status_code == 404 else 502
        if error in {"expired", "claimed"}:
            status_code = 410
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service is unavailable. Try again shortly.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service is unavailable. Try again shortly.",
        ) from exc

    try:
        parsed = response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an invalid response.",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an invalid response.",
        )
    return parsed


async def _telegram_onboarding_request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _telegram_onboarding_request_sync,
        method,
        path,
        body=body,
        bearer_token=bearer_token,
    )


@app.post("/api/messaging/telegram/onboarding/start")
async def start_telegram_onboarding(body: TelegramOnboardingStart):
    bot_name = (body.bot_name or "Hermes Agent").strip() or "Hermes Agent"
    payload = await _telegram_onboarding_request(
        "POST",
        "/v1/telegram/pairings",
        body={"bot_name": bot_name},
    )

    pairing_id = str(payload.get("pairing_id") or "").strip()
    poll_token = str(payload.get("poll_token") or "").strip()
    expires_at = str(payload.get("expires_at") or "").strip()
    deep_link = str(payload.get("deep_link") or "").strip()
    qr_payload = str(payload.get("qr_payload") or deep_link).strip()
    suggested_username = str(payload.get("suggested_username") or "").strip()
    if not pairing_id or not poll_token or not expires_at or not deep_link:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an incomplete response.",
        )

    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        _telegram_onboarding_pairings[pairing_id] = _TelegramOnboardingPairing(
            poll_token=poll_token,
            expires_at=expires_at,
            expires_at_ts=_parse_expiry_ts(expires_at),
        )

    return {
        "pairing_id": pairing_id,
        "suggested_username": suggested_username,
        "deep_link": deep_link,
        "qr_payload": qr_payload,
        "expires_at": expires_at,
    }


@app.get("/api/messaging/telegram/onboarding/{pairing_id}")
async def get_telegram_onboarding_status(pairing_id: str):
    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        record = _telegram_onboarding_pairings.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="Telegram setup session was not found. Start a new setup.",
            )
        if record.bot_token:
            return {
                "status": "ready",
                "bot_username": record.bot_username,
                "owner_user_id": record.owner_user_id,
                "expires_at": record.expires_at,
            }
        poll_token = record.poll_token

    payload = await _telegram_onboarding_request(
        "GET",
        f"/v1/telegram/pairings/{urllib.parse.quote(pairing_id, safe='')}",
        bearer_token=poll_token,
    )
    status = str(payload.get("status") or "").strip()
    if status == "waiting":
        with _telegram_onboarding_lock:
            current = _telegram_onboarding_pairings.get(pairing_id)
            expires_at = current.expires_at if current else ""
        return {"status": "waiting", "expires_at": expires_at}

    if status == "ready":
        bot_token = str(payload.get("token") or "").strip()
        bot_username = str(payload.get("bot_username") or "").strip()
        if not bot_token:
            raise HTTPException(
                status_code=502,
                detail="Telegram setup service returned an incomplete response.",
            )
        owner_user_id = _normalize_telegram_user_id(payload.get("owner_user_id"))
        with _telegram_onboarding_lock:
            record = _telegram_onboarding_pairings.get(pairing_id)
            if not record:
                raise HTTPException(
                    status_code=404,
                    detail="Telegram setup session was not found. Start a new setup.",
                )
            record.bot_token = bot_token
            record.bot_username = bot_username or None
            record.owner_user_id = owner_user_id
            return {
                "status": "ready",
                "bot_username": record.bot_username,
                "owner_user_id": record.owner_user_id,
                "expires_at": record.expires_at,
            }

    if status in {"expired", "claimed"}:
        with _telegram_onboarding_lock:
            _telegram_onboarding_pairings.pop(pairing_id, None)
        raise HTTPException(
            status_code=410,
            detail=_telegram_onboarding_error_message(
                status,
                "Telegram setup is no longer available. Start a new setup.",
            ),
        )

    raise HTTPException(
        status_code=502,
        detail="Telegram setup service returned an unknown status.",
    )


def _restart_gateway_after_telegram_onboarding(profile: Optional[str] = None) -> dict[str, Any]:
    """Best-effort gateway restart after saving Telegram QR onboarding.

    The QR flow naturally pulls users into Telegram on another device. If the
    saved token waits on a separate dashboard restart click, Hermes appears
    broken from the chat side. Keep the config save authoritative, but report
    restart failures so the UI can fall back to the existing manual banner.
    """
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after Telegram onboarding")
        return {
            "restart_started": False,
            "restart_error": str(exc),
        }
    if reused:
        _log.info(
            "Telegram onboarding: reusing in-flight gateway restart (pid %s)",
            proc.pid,
        )
    return {
        "restart_started": True,
        "restart_action": "gateway-restart",
        "restart_pid": proc.pid,
    }


@app.post("/api/messaging/telegram/onboarding/{pairing_id}/apply")
async def apply_telegram_onboarding(
    pairing_id: str, body: TelegramOnboardingApply, profile: Optional[str] = None
):
    allowed_user_ids = []
    seen = set()
    for raw_id in body.allowed_user_ids:
        normalized = _normalize_telegram_user_id(raw_id)
        if not normalized:
            raise HTTPException(
                status_code=400,
                detail="Allowed Telegram user IDs must be numeric.",
            )
        if normalized not in seen:
            seen.add(normalized)
            allowed_user_ids.append(normalized)
    if not allowed_user_ids:
        raise HTTPException(
            status_code=400,
            detail="Add at least one allowed Telegram user ID.",
        )

    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        record = _telegram_onboarding_pairings.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="Telegram setup session was not found. Start a new setup.",
            )
        bot_token = record.bot_token
        bot_username = record.bot_username
        if not bot_token:
            raise HTTPException(
                status_code=409,
                detail="Telegram setup is not ready yet.",
            )

    effective_profile = body.profile or profile
    try:
        with _profile_scope(effective_profile):
            save_env_value("TELEGRAM_BOT_TOKEN", bot_token)
            save_env_value("TELEGRAM_ALLOWED_USERS", ",".join(allowed_user_ids))
            _write_platform_enabled("telegram", True)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("Telegram onboarding apply failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to save Telegram setup.",
        ) from exc

    with _telegram_onboarding_lock:
        _telegram_onboarding_pairings.pop(pairing_id, None)

    restart_result = _restart_gateway_after_telegram_onboarding(effective_profile)

    return {
        "ok": True,
        "platform": "telegram",
        "bot_username": bot_username,
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@app.delete("/api/messaging/telegram/onboarding/{pairing_id}")
async def cancel_telegram_onboarding(pairing_id: str):
    with _telegram_onboarding_lock:
        _telegram_onboarding_pairings.pop(pairing_id, None)
    return {"ok": True}


# =============================================================================
# Weixin (WeChat personal) QR onboarding — mirrors Telegram onboarding shape
# =============================================================================

def _weixin_qr_manager():
    """Lazy import + bind to the dashboard event loop on first use."""
    from gateway.platforms.weixin_qr_session import get_weixin_qr_session_manager

    manager = get_weixin_qr_session_manager()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            manager.set_event_loop(loop)
    except RuntimeError:
        pass
    return manager


@app.post("/api/messaging/weixin/onboarding/start")
async def start_weixin_onboarding(body: WeixinOnboardingStart, profile: Optional[str] = None):
    """Start a Weixin QR login session.

    Returns ``{session_id, state, expires_at}`` with ``state="starting"``.
    The frontend polls ``/status`` until ``state`` transitions to
    ``waiting`` (QR ready), ``confirmed`` (scan complete), or ``failed``.
    """
    effective_profile = body.profile or profile
    try:
        with _profile_scope(effective_profile):
            hermes_home = str(get_hermes_home())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    manager = _weixin_qr_manager()
    return manager.start_session(hermes_home)


@app.get("/api/messaging/weixin/onboarding/{session_id}/status")
async def get_weixin_onboarding_status(session_id: str):
    """Poll a Weixin QR login session.

    Returns ``{session_id, state, expires_at, qr_image_base64?, error?, account_id?, base_url?}``.
    ``qr_image_base64`` is present when ``state`` is ``waiting`` or ``scanned``
    and contains a base64-encoded PNG suitable for ``<img src="data:image/png;base64,...">``.
    """
    manager = _weixin_qr_manager()
    status = manager.get_status(session_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail="WeChat setup session was not found. Start a new setup.",
        )
    return status


@app.post("/api/messaging/weixin/onboarding/{session_id}/apply")
async def apply_weixin_onboarding(
    session_id: str, body: WeixinOnboardingApply, profile: Optional[str] = None
):
    """Save confirmed Weixin credentials to .env and enable the platform.

    Called by the frontend after ``status`` returns ``state="confirmed"``.
    Writes ``WEIXIN_ACCOUNT_ID``, ``WEIXIN_TOKEN``, ``WEIXIN_BASE_URL`` to
    ``.env``, sets ``platforms.weixin.enabled: true`` in config.yaml, and
    triggers a gateway restart — mirroring ``apply_telegram_onboarding``.
    """
    manager = _weixin_qr_manager()
    credentials = manager.get_credentials(session_id)
    if not credentials:
        raise HTTPException(
            status_code=409,
            detail="WeChat setup is not ready yet. Scan the QR code first.",
        )

    effective_profile = body.profile or profile
    try:
        with _profile_scope(effective_profile):
            save_env_value("WEIXIN_ACCOUNT_ID", credentials["account_id"])
            save_env_value("WEIXIN_TOKEN", credentials["token"])
            # Defensive: ensure the base URL is always the canonical iLink
            # endpoint, even if the QR session somehow captured a stale one.
            base_url = credentials["base_url"]
            if "ilinkai.weixin.qq.com" not in base_url:
                base_url = "https://ilinkai.weixin.qq.com"
            save_env_value("WEIXIN_BASE_URL", base_url)
            _write_platform_enabled("weixin", True)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("Weixin onboarding apply failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to save WeChat setup.",
        ) from exc

    # Remove the session now that credentials are persisted.
    manager.cancel_session(session_id)

    restart_result = _restart_gateway_after_telegram_onboarding(effective_profile)

    return {
        "ok": True,
        "platform": "weixin",
        "account_id": credentials["account_id"],
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@app.delete("/api/messaging/weixin/onboarding/{session_id}")
async def cancel_weixin_onboarding(session_id: str):
    """Cancel a Weixin QR login session."""
    manager = _weixin_qr_manager()
    manager.cancel_session(session_id)
    return {"ok": True}


@app.get("/api/messaging/platforms")
async def get_messaging_platforms(profile: Optional[str] = None):
    # Profile-scoped so the dashboard's global profile switcher shows the
    # TARGET profile's channel credentials/state, not the root install's.
    # Inside _profile_scope, load_env()/read_runtime_status()/get_running_pid()
    # all resolve against the requested profile's HERMES_HOME.
    with _profile_scope(profile) as scoped_dir:
        env_on_disk = load_env()
        runtime = read_runtime_status()
        return {
            "env_path": str(get_env_path()),
            "gateway_start_command": _gateway_display_command(profile, "start"),
            "platforms": [
                _messaging_platform_payload(
                    entry, env_on_disk, runtime, scoped=scoped_dir is not None
                )
                for entry in _messaging_platform_catalog()
            ]
        }


@app.put("/api/messaging/platforms/{platform_id}")
async def update_messaging_platform(
    platform_id: str, body: MessagingPlatformUpdate, profile: Optional[str] = None
):
    entry = _catalog_lookup(platform_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"Unknown messaging platform: {platform_id}"
        )

    allowed_env = set(entry["env_vars"])
    try:
        with _profile_scope(body.profile or profile):
            for key in body.clear_env:
                if key not in allowed_env:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key} is not configurable for {entry['name']}",
                    )
                remove_env_value(key)

            for key, value in body.env.items():
                if key not in allowed_env:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key} is not configurable for {entry['name']}",
                    )
                trimmed = value.strip()
                if trimmed:
                    _validate_messaging_env_value(platform_id, key, trimmed)
                    save_env_value(key, trimmed)

            if body.enabled is not None:
                _write_platform_enabled(platform_id, body.enabled)

        return {"ok": True, "platform": platform_id}
    except HTTPException:
        raise
    except Exception:
        _log.exception("PUT /api/messaging/platforms/%s failed", platform_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/messaging/platforms/{platform_id}/test")
async def test_messaging_platform(platform_id: str, profile: Optional[str] = None):
    entry = _catalog_lookup(platform_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"Unknown messaging platform: {platform_id}"
        )

    with _profile_scope(profile) as scoped_dir:
        env_on_disk = load_env()
        payload = _messaging_platform_payload(
            entry, env_on_disk, read_runtime_status(), scoped=scoped_dir is not None
        )
    if not payload["enabled"]:
        message = f"{entry['name']} is disabled. Enable it, then restart the gateway."
        return {"ok": False, "state": payload["state"], "message": message}
    if not payload["configured"]:
        missing = [
            field["key"]
            for field in payload["env_vars"]
            if field["required"] and not field["is_set"]
        ]
        message = (
            f"Missing required setup: {', '.join(missing)}"
            if missing
            else "Platform setup is incomplete."
        )
        return {"ok": False, "state": payload["state"], "message": message}
    if not payload["gateway_running"]:
        return {
            "ok": False,
            "state": payload["state"],
            "message": "Gateway is not running. Restart the gateway to connect this platform.",
        }
    if payload["state"] == "connected":
        return {
            "ok": True,
            "state": payload["state"],
            "message": f"{entry['name']} is connected.",
        }
    if payload.get("error_message"):
        return {
            "ok": False,
            "state": payload["state"],
            "message": payload["error_message"],
        }
    return {
        "ok": False,
        "state": payload["state"],
        "message": "Setup looks complete, but the gateway has not reported a connection yet. Restart the gateway.",
    }


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

    Returns the Entra-ID placeholder when handed a callable (Azure Foundry
    bearer provider) — the callable is NEVER invoked here.
    """
    if not value:
        return ""
    if callable(value) and not isinstance(value, str):
        # Entra ID bearer provider — never reveal a minted token in the UI.
        return "<entra-id-bearer>"
    s = str(value)
    if "." in s and s.count(".") >= 2:
        # Looks like a JWT — show the trailing piece of the signature only.
        s = s.rsplit(".", 1)[-1]
    if len(s) <= visible:
        return s
    return f"…{s[-visible:]}"


def _anthropic_oauth_status() -> Dict[str, Any]:
    """Status for the "Anthropic API Key" catalog entry.

    Two sources, in priority order:
    1. ``~/.hermes/.anthropic_oauth.json`` — Hermes-managed PKCE flow (what
       this entry's Connect button writes)
    2. ``ANTHROPIC_API_KEY`` → ``ANTHROPIC_TOKEN`` → ``CLAUDE_CODE_OAUTH_TOKEN``
       env vars (registry order) — from ``.env``, the shell, or an external
       secret source like Bitwarden (whose keys are injected into the process
       env during ``load_hermes_dotenv()``, so the same check covers them)

    Claude Code's ``~/.claude/.credentials.json`` is deliberately NOT read
    here — it has its own dedicated catalog entry (``claude-code`` →
    ``_claude_code_only_status``). Reporting it under the API-key entry
    double-counts the token and shadows a real ANTHROPIC_API_KEY.
    """
    try:
        from agent.anthropic_adapter import (
            read_hermes_oauth_credentials,
            _get_hermes_oauth_file,
        )
    except ImportError:
        read_hermes_oauth_credentials = None  # type: ignore
        _get_hermes_oauth_file = None  # type: ignore

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
            "source_label": f"Hermes PKCE ({_get_hermes_oauth_file() if _get_hermes_oauth_file else None})",
            "token_preview": _truncate_token(hermes_creds.get("accessToken")),
            "expires_at": hermes_creds.get("expiresAt"),
            "has_refresh_token": bool(hermes_creds.get("refreshToken")),
        }

    # Env-var / secret-source path. ``get_env_value`` checks the process
    # environment first (where Bitwarden-sourced secrets land) then .env.
    env_var_order: tuple = ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        env_var_order = PROVIDER_REGISTRY["anthropic"].api_key_env_vars
    except (ImportError, KeyError):
        pass
    try:
        from hermes_cli.config import get_env_value
    except ImportError:
        get_env_value = None  # type: ignore
    try:
        from hermes_cli.env_loader import format_secret_source_suffix
    except ImportError:
        format_secret_source_suffix = None  # type: ignore

    for var in env_var_order:
        value = (get_env_value(var) if get_env_value else None) or os.getenv(var)
        if not value:
            continue
        suffix = format_secret_source_suffix(var) if format_secret_source_suffix else ""
        return {
            "logged_in": True,
            "source": "env_var",
            "source_label": f"{var}{suffix}",
            "token_preview": _truncate_token(value),
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


def _copilot_acp_status() -> Dict[str, Any]:
    """Status for copilot-acp — credentials are owned by the Copilot CLI.

    There is no cheap programmatic credential probe for the ACP subprocess, so
    this is a read-only "managed by the Copilot CLI" card (like claude-code):
    Hermes never claims a login state it can't verify.
    """
    return {
        "logged_in": False,
        "source": "copilot_cli",
        "source_label": "Managed by the GitHub Copilot CLI",
        "token_preview": None,
        "expires_at": None,
        "has_refresh_token": False,
    }


# Explicit, hand-tuned OAuth/account provider cards. These carry the bits that
# can't be derived from the unified provider catalog: the OAuth ``flow`` shape,
# the per-provider ``status_fn``, the ``cli_command`` fallback, and curated
# display order. They are the OVERRIDE BASE for ``_build_oauth_catalog()``,
# which unions them with every accounts-tab provider in ``provider_catalog()``
# so newly-added OAuth/external providers appear automatically (no hand edit).
# This tuple also still includes two entries that are NOT catalog providers but
# must show on the Accounts tab: the api-key Anthropic PKCE card and the
# synthetic ``claude-code`` subscription row.
# ``flow`` describes the OAuth shape so the modal can pick the right UI:
# ``pkce`` = open URL + paste callback code, ``device_code`` = show code +
# verification URL + poll, ``external`` = read-only (delegated to a third-party
# CLI like Claude Code or Qwen).
_OAUTH_PROVIDER_CATALOG: tuple[Dict[str, Any], ...] = (
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
        "name": "OpenAI OAuth (ChatGPT)",
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
    {
        "id": "minimax-oauth",
        "name": "MiniMax (OAuth)",
        # MiniMax's flow is structurally device-code (verification URI +
        # user code, backend polls the token endpoint) with a PKCE
        # extension for code-binding. The dashboard renders the same UX
        # as Nous's device-code flow; the PKCE bit is a security
        # extension that doesn't change the operator experience.
        "flow": "device_code",
        "cli_command": "hermes auth add minimax-oauth",
        "docs_url": "https://www.minimax.io",
        "status_fn": None,  # dispatched via auth.get_minimax_oauth_auth_status
    },
    {
        "id": "xai-oauth",
        "name": "xAI Grok OAuth (SuperGrok / Premium+)",
        # Device code is the default because it works in remote shells,
        # containers, and desktop installs without requiring a reachable
        # 127.0.0.1 callback.
        "flow": "device_code",
        "cli_command": "hermes auth add xai-oauth",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth",
        "status_fn": None,  # dispatched via auth.get_xai_oauth_auth_status
    },
    {
        "id": "copilot-acp",
        "name": "GitHub Copilot (ACP)",
        "flow": "external",
        "cli_command": "copilot /login",
        "docs_url": "https://docs.github.com/en/copilot",
        "status_fn": _copilot_acp_status,
    },
    # ── Anthropic / Claude entries sit at the bottom: the API-key path
    # first, then the subscription OAuth path (which only works with extra
    # usage credits on top of a Claude Max plan — see disclaimer in name).
    {
        "id": "anthropic",
        "name": "Anthropic API Key",
        "flow": "pkce",
        "cli_command": "hermes auth add anthropic",
        "docs_url": "https://docs.claude.com/en/api/getting-started",
        "status_fn": _anthropic_oauth_status,
    },
    {
        "id": "claude-code",
        "name": "Anthropic OAuth: Required Extra Usage Credits to Use Subscription",
        "flow": "external",
        "cli_command": "claude setup-token",
        "docs_url": "https://docs.claude.com/en/docs/claude-code",
        "status_fn": _claude_code_only_status,
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
        if provider_id == "minimax-oauth":
            raw = hauth.get_minimax_oauth_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": "minimax_oauth",
                "source_label": f"MiniMax ({raw.get('region', 'global')})",
                "token_preview": None,
                "expires_at": raw.get("expires_at"),
                "has_refresh_token": True,
            }
        if provider_id == "xai-oauth":
            raw = hauth.get_xai_oauth_auth_status()
            # source_label is meant to be a human-readable origin (auth-store
            # path / credential source), not the internal auth_mode string
            # ("oauth_pkce"). Prefer the store path, then the source slug.
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": raw.get("source") or "xai_oauth",
                "source_label": raw.get("auth_store") or raw.get("source") or "xAI Grok OAuth",
                "token_preview": _truncate_token(raw.get("api_key")),
                "expires_at": None,
                "has_refresh_token": True,
                "last_refresh": raw.get("last_refresh"),
            }
        # No hand-written branch for this provider id: fall through to the
        # canonical slug-driven dispatcher so accounts-tab providers derived
        # from the unified catalog (which carry status_fn=None) still reflect
        # real login state instead of rendering permanently logged-out. This
        # closes the membership-auto-extends-but-status-doesn't gap: add an
        # OAuth/account provider plugin and its card shows the right state.
        raw = hauth.get_auth_status(provider_id)
        if isinstance(raw, dict) and "logged_in" in raw:
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": raw.get("source") or raw.get("provider") or provider_id,
                "source_label": (
                    raw.get("source_label")
                    or raw.get("auth_store")
                    or raw.get("auth_store_path")
                    or raw.get("base_url")
                    or raw.get("name")
                    or ""
                ),
                "token_preview": _truncate_token(
                    raw.get("access_token") or raw.get("api_key")
                ),
                "expires_at": raw.get("expires_at") or raw.get("access_expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
    except Exception as e:
        return {"logged_in": False, "error": str(e)}
    return {"logged_in": False}


def _oauth_provider_disconnect_command(provider: Dict[str, Any]) -> Optional[str]:
    """Shell command that clears an external provider's credentials.

    External providers store their credentials outside Hermes, so the disconnect
    API deliberately refuses them (we never delete files another CLI owns on the
    user's behalf via a silent API call). For the ones we know how to clear we
    instead hand the GUI a command it can *run in the embedded terminal* — the
    user sees exactly what executes, and Hermes then stops resolving the token.

    Claude Code has no scriptable logout (only the interactive ``/logout``), so
    we remove the credential the same way logout does: the macOS Keychain entry
    (``Claude Code-credentials``) and/or the ``~/.claude/.credentials.json``
    file — the two sources ``read_claude_code_credentials()`` consults. Returns
    None for providers we can't safely clear (the GUI shows a manual hint).
    """
    if provider.get("flow") != "external":
        return None
    if provider.get("id") == "claude-code":
        rm_file = "rm -f ~/.claude/.credentials.json"
        if sys.platform == "darwin":
            return f'security delete-generic-password -s "Claude Code-credentials" 2>/dev/null; {rm_file}'
        return rm_file
    return None


def _oauth_provider_disconnect_hint(provider: Dict[str, Any], status: Dict[str, Any]) -> Optional[str]:
    """Return the manual disconnect path when the API cannot clear this provider."""
    if provider.get("flow") == "external":
        if _oauth_provider_disconnect_command(provider):
            # The GUI offers a one-click "run in terminal" path; this hint is the
            # fallback wording for surfaces that only show text.
            return "Managed outside Hermes — run the disconnect command to remove it."
        return "Managed by that provider's CLI; remove it there."
    if status.get("source") == "env_var":
        return "Remove the API key from Settings → Keys instead."
    return None


def _build_oauth_catalog() -> list[Dict[str, Any]]:
    """Build the Accounts-tab provider list.

    MEMBERSHIP is the union of:
      1. ``_OAUTH_PROVIDER_CATALOG`` — the explicit, hand-tuned cards that carry
         bespoke flow / status_fn / cli_command (including the api-key Anthropic
         PKCE card and the synthetic claude-code subscription row, which are not
         catalog providers), and
      2. every accounts-tab provider in the unified ``provider_catalog()`` (the
         ``hermes model`` universe) — so any OAuth/external provider added as a
         plugin appears automatically, with sensible defaults, even if no
         explicit card was written for it.

    The explicit catalog wins on metadata; the unified catalog guarantees we
    never silently drop a provider the CLI picker offers. Order: explicit cards
    first (their curated order), then any catalog-only providers appended in
    ``hermes model`` order.
    """
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()

    # 1. Explicit hand-tuned cards (authoritative metadata + curated order).
    for entry in _OAUTH_PROVIDER_CATALOG:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        rows.append(dict(entry))

    # 2. Catalog accounts-providers not already covered — keeps the Accounts tab
    #    in lockstep with the `hermes model` universe (zero-edit for new plugins).
    try:
        from hermes_cli.provider_catalog import provider_catalog
        for d in provider_catalog():
            if d.tab != "accounts" or d.slug in seen:
                continue
            seen.add(d.slug)
            rows.append({
                "id": d.slug,
                "name": d.label,
                "flow": "external",
                "cli_command": f"hermes auth add {d.slug}",
                "docs_url": d.signup_url or "",
                "status_fn": None,
            })
    except Exception:
        pass

    return rows


@app.get("/api/providers/oauth")
async def list_oauth_providers(profile: Optional[str] = None):
    """Enumerate every OAuth-capable LLM provider with current status.

    Response shape (per provider):
        id              stable identifier (used in DELETE path)
        name            human label
        flow            "pkce" | "device_code" | "external"
        cli_command     fallback CLI command for users to run manually
        disconnect_command  shell command that clears an external provider's
                            creds (run in the embedded terminal), else null
        docs_url        external docs/portal link for the "Learn more" link
        status:
          logged_in        bool — currently has usable creds
          source           short slug ("hermes_pkce", "claude_code", ...)
          source_label     human-readable origin (file path, env var name)
          token_preview    last N chars of the token, never the full token
          expires_at       ISO timestamp string or null
          has_refresh_token bool

    Membership is derived from the unified provider_catalog() so this stays in
    sync with the `hermes model` picker; _OAUTH_OVERRIDES supplies per-provider
    flow/status/cli metadata.
    """
    with _profile_scope(profile):
        providers = []
        for p in _build_oauth_catalog():
            status = _resolve_provider_status(p["id"], p.get("status_fn"))
            disconnect_hint = _oauth_provider_disconnect_hint(p, status)
            providers.append({
                "id": p["id"],
                "name": p["name"],
                "flow": p["flow"],
                "cli_command": p["cli_command"],
                "docs_url": p["docs_url"],
                "disconnect_hint": disconnect_hint,
                "disconnect_command": _oauth_provider_disconnect_command(p),
                "disconnectable": disconnect_hint is None,
                "status": status,
            })
        return {"providers": providers}


@app.delete("/api/providers/oauth/{provider_id}")
async def disconnect_oauth_provider(
    provider_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Disconnect an OAuth provider. Token-protected (matches /env/reveal)."""
    _require_token(request)

    with _profile_scope(profile):
        catalog_by_id = {p["id"]: p for p in _build_oauth_catalog()}
        provider = catalog_by_id.get(provider_id)
        if provider is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider: {provider_id}. "
                       f"Available: {', '.join(sorted(catalog_by_id))}",
            )

        disconnect_hint = _oauth_provider_disconnect_hint(provider, {})
        if disconnect_hint:
            raise HTTPException(
                status_code=400,
                detail=f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}",
            )

        status = _resolve_provider_status(provider_id, provider.get("status_fn"))
        disconnect_hint = _oauth_provider_disconnect_hint(provider, status)
        if disconnect_hint:
            raise HTTPException(
                status_code=400,
                detail=f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}",
            )

        # Anthropic clears only the Hermes-managed PKCE file and auth-store entry.
        # The separate claude-code catalog row is external/read-only and rejected
        # above so we never pretend to remove ~/.claude/* credentials owned by the CLI.
        if provider_id == "anthropic":
            cleared = False
            try:
                from agent.anthropic_adapter import _get_hermes_oauth_file
                oauth_file = _get_hermes_oauth_file()
                if oauth_file.exists():
                    oauth_file.unlink()
                    cleared = True
            except Exception:
                pass
            # Also clear the credential pool entry if present.
            try:
                from hermes_cli.auth import clear_provider_auth
                cleared = clear_provider_auth("anthropic") or cleared
            except Exception:
                pass
            _log.info("oauth/disconnect: %s", provider_id)
            return {"ok": bool(cleared), "provider": provider_id}

        try:
            from hermes_cli.auth import clear_provider_auth, invalidate_nous_auth_status_cache
            cleared = clear_provider_auth(provider_id)
            if provider_id == "nous":
                invalidate_nous_auth_status_cache()
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
        _OAUTH_TOKEN_URLS as _ANTHROPIC_OAUTH_TOKEN_URLS,
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
        stale = [sid for sid, sess in _oauth_sessions.items() if sess["created_at"] < cutoff]
        for sid in stale:
            _oauth_sessions.pop(sid, None)


def _oauth_profile_name(profile: Optional[str]) -> Optional[str]:
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return None
    return requested


def _validate_oauth_profile(profile: Optional[str]) -> None:
    profile_name = _oauth_profile_name(profile)
    if profile_name:
        _resolve_profile_dir(profile_name)


def _new_oauth_session(
    provider_id: str,
    flow: str,
    profile: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """Create + register a new OAuth session, return (session_id, session_dict)."""
    sid = secrets.token_urlsafe(16)
    profile_name = _oauth_profile_name(profile)
    sess = {
        "session_id": sid,
        "provider": provider_id,
        "flow": flow,
        "profile": profile_name,
        "created_at": time.time(),
        "status": "pending",  # pending | approved | denied | expired | error
        "error_message": None,
    }
    with _oauth_sessions_lock:
        _oauth_sessions[sid] = sess
    return sid, sess


def _oauth_session_profile(
    session_id: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Return the profile that owns an OAuth session, if one was provided."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        profile = sess.get("profile") if sess else None
    return profile or _oauth_profile_name(fallback)


def _save_anthropic_oauth_creds(access_token: str, refresh_token: str, expires_at_ms: int) -> None:
    """Persist Anthropic PKCE creds to both Hermes file AND credential pool.

    Mirrors what auth_commands.add_command does so the dashboard flow leaves
    the system in the same state as ``hermes auth add anthropic``.
    """
    from agent.anthropic_adapter import _get_hermes_oauth_file
    oauth_file = _get_hermes_oauth_file()
    payload = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at_ms,
    }
    # atomic_json_write creates the temp with mode 0o600 (via mkstemp) *before*
    # any content is written, then fsyncs and atomically replaces the target.
    # The previous os.replace + post-hoc chmod left a TOCTOU window in which the
    # OAuth token file was world-readable at the default umask (0o644 on most
    # hosts) between the rename and the chmod. atomic_json_write also preserves
    # the existing file's owner and cleans up its temp on failure.
    from utils import atomic_json_write

    atomic_json_write(oauth_file, payload, indent=2, mode=0o600)
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
        existing = [e for e in pool.entries() if getattr(e, "source", "").startswith(f"{SOURCE_MANUAL}:dashboard_pkce")]
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


def _start_anthropic_pkce(profile: Optional[str] = None) -> Dict[str, Any]:
    """Begin PKCE flow. Returns the auth URL the UI should open."""
    if not _ANTHROPIC_OAUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Anthropic OAuth not available (missing adapter)")
    verifier, challenge = _generate_pkce_pair()
    sid, sess = _new_oauth_session("anthropic", "pkce", profile=profile)
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


def _submit_anthropic_pkce(
    session_id: str,
    code_input: str,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Exchange authorization code for tokens. Persists on success."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess or sess["provider"] != "anthropic" or sess["flow"] != "pkce":
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    if sess["status"] != "pending":
        return {"ok": False, "status": sess["status"], "message": sess.get("error_message")}

    # Anthropic's redirect callback page formats the code as `<code>#<state>`.
    # Strip the state suffix if present (we already have the verifier server-side).
    parts = code_input.strip().split("#", 1)
    code = parts[0].strip()
    if not code:
        return {"ok": False, "status": "error", "message": "No code provided"}
    state_from_callback = parts[1] if len(parts) > 1 else ""

    exchange_data = json.dumps({
        "grant_type": "authorization_code",
        "client_id": _ANTHROPIC_OAUTH_CLIENT_ID,
        "code": code,
        "state": state_from_callback or sess["state"],
        "redirect_uri": _ANTHROPIC_OAUTH_REDIRECT_URI,
        "code_verifier": sess["verifier"],
    }).encode()
    # Anthropic migrated the OAuth token endpoint to platform.claude.com;
    # console.anthropic.com now 404s. Try the new host first, then fall back.
    result = None
    last_exc = None
    for _endpoint in _ANTHROPIC_OAUTH_TOKEN_URLS:
        req = urllib.request.Request(
            _endpoint,
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
            break
        except Exception as e:
            last_exc = e
            continue
    if result is None:
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = f"Token exchange failed: {last_exc}"
        return {"ok": False, "status": "error", "message": sess["error_message"]}

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = int(result.get("expires_in") or 3600)
    if not access_token:
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = "No access token returned"
        return {"ok": False, "status": "error", "message": sess["error_message"]}

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    try:
        with _profile_scope(_oauth_session_profile(session_id, profile)):
            _save_anthropic_oauth_creds(access_token, refresh_token, expires_at_ms)
    except Exception as e:
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = f"Save failed: {e}"
        return {"ok": False, "status": "error", "message": sess["error_message"]}
    with _oauth_sessions_lock:
        sess["status"] = "approved"
    _log.info("oauth/pkce: anthropic login completed (session=%s)", session_id)
    return {"ok": True, "status": "approved"}


async def _start_device_code_flow(
    provider_id: str,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Initiate a device-code flow (Nous, OpenAI Codex, MiniMax, or xAI).

    Calls the provider's device-auth endpoint via the existing CLI helpers,
    then spawns a background poller. Returns the user-facing display fields
    so the UI can render the verification page link + user code.
    """
    if provider_id == "nous":
        from hermes_cli.auth import (
            _request_device_code,
            PROVIDER_REGISTRY,
        )
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
                timeout=httpx.Timeout(15.0),
                headers={"Accept": "application/json"},
            ) as client:
                return (
                    _request_device_code(
                        client=client,
                        portal_base_url=portal_base_url,
                        client_id=client_id,
                        scope=scope,
                    ),
                    scope,
                )

        device_data, effective_scope = await asyncio.get_running_loop().run_in_executor(
            None, _do_nous_device_request
        )
        sid, sess = _new_oauth_session("nous", "device_code", profile=profile)
        sess["device_code"] = str(device_data["device_code"])
        sess["interval"] = int(device_data["interval"])
        sess["expires_at"] = time.time() + int(device_data["expires_in"])
        sess["portal_base_url"] = portal_base_url
        sess["client_id"] = client_id
        sess["scope"] = effective_scope
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
        sid, _ = _new_oauth_session("openai-codex", "device_code", profile=profile)
        # Use the helper but in a thread because it polls inline.
        # We can't extract just the start step without refactoring auth.py,
        # so we run the full helper in a worker and proxy the user_code +
        # verification_url back via the session dict. The helper prints
        # to stdout — we capture nothing here, just status.
        threading.Thread(
            target=_codex_full_login_worker, args=(sid,), daemon=True,
            name=f"oauth-codex-{sid[:6]}",
        ).start()
        # Block briefly until the worker has populated the user_code, OR error.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with _oauth_sessions_lock:
                s = _oauth_sessions.get(sid)
            if s and (s.get("user_code") or s["status"] != "pending"):
                break
            await asyncio.sleep(0.1)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(sid, {})
        if s.get("status") == "error":
            raise HTTPException(status_code=500, detail=s.get("error_message") or "device-auth failed")
        if not s.get("user_code"):
            raise HTTPException(status_code=504, detail="device-auth timed out before returning a user code")
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": s["user_code"],
            "verification_url": s["verification_url"],
            "expires_in": int(s.get("expires_in") or 900),
            "poll_interval": int(s.get("interval") or 5),
        }

    if provider_id == "minimax-oauth":
        # MiniMax uses a device-code-style flow (verification URI + user
        # code + background poll) with a PKCE extension on top. From the
        # operator's perspective it's identical to Nous's device-code
        # flow; the PKCE bit (verifier + challenge from
        # _minimax_pkce_pair) is a security extension that binds the
        # token exchange to the original session.
        from hermes_cli.auth import (
            _minimax_pkce_pair,
            _minimax_request_user_code,
            MINIMAX_OAUTH_CLIENT_ID,
            MINIMAX_OAUTH_GLOBAL_BASE,
        )
        import httpx
        verifier, challenge, state = _minimax_pkce_pair()
        portal_base_url = (
            os.getenv("MINIMAX_PORTAL_BASE_URL") or MINIMAX_OAUTH_GLOBAL_BASE
        ).rstrip("/")
        def _do_minimax_request():
            with httpx.Client(
                timeout=httpx.Timeout(15.0),
                headers={"Accept": "application/json"},
                follow_redirects=True,
            ) as client:
                return _minimax_request_user_code(
                    client=client,
                    portal_base_url=portal_base_url,
                    client_id=MINIMAX_OAUTH_CLIENT_ID,
                    code_challenge=challenge,
                    state=state,
                )
        device_data = await asyncio.get_event_loop().run_in_executor(
            None, _do_minimax_request
        )
        sid, sess = _new_oauth_session("minimax-oauth", "device_code", profile=profile)
        # The CLI flow names this `interval_ms` because MiniMax's
        # `interval` field is in milliseconds (defensive default 2000ms
        # in _minimax_poll_token).
        interval_raw = device_data.get("interval")
        sess["interval_ms"] = (
            int(interval_raw) if interval_raw is not None else None
        )
        sess["user_code"] = str(device_data["user_code"])
        sess["code_verifier"] = verifier
        sess["state"] = state
        sess["portal_base_url"] = portal_base_url
        sess["client_id"] = MINIMAX_OAUTH_CLIENT_ID
        sess["region"] = "global"
        # `expired_in` from MiniMax is overloaded — could be a unix-ms
        # timestamp OR a seconds-from-now duration. Mirror the heuristic
        # in _minimax_poll_token. Stash the raw value for the poller;
        # compute a derived expires_at + UI-friendly expires_in seconds.
        expired_in_raw = int(device_data["expired_in"])
        sess["expired_in_raw"] = expired_in_raw
        if expired_in_raw > 1_000_000_000_000:  # likely unix-ms
            expires_at_ts = expired_in_raw / 1000.0
            expires_in_seconds = max(0, int(expires_at_ts - time.time()))
        else:
            expires_at_ts = time.time() + expired_in_raw
            expires_in_seconds = expired_in_raw
        sess["expires_at"] = expires_at_ts
        threading.Thread(
            target=_minimax_poller,
            args=(sid,),
            daemon=True,
            name=f"oauth-poll-{sid[:6]}",
        ).start()
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": str(device_data["user_code"]),
            "verification_url": str(device_data["verification_uri"]),
            "expires_in": expires_in_seconds,
            "poll_interval": max(2, (sess["interval_ms"] or 2000) // 1000),
        }

    if provider_id == "xai-oauth":
        from hermes_cli.auth import _xai_oauth_request_device_code
        import httpx

        def _do_xai_device_request():
            with httpx.Client(
                timeout=httpx.Timeout(20.0),
                headers={"Accept": "application/json"},
            ) as client:
                return _xai_oauth_request_device_code(client)

        device_data = await asyncio.get_running_loop().run_in_executor(
            None, _do_xai_device_request
        )
        sid, sess = _new_oauth_session("xai-oauth", "device_code", profile=profile)
        sess["device_code"] = str(device_data["device_code"])
        sess["interval"] = int(device_data["interval"])
        sess["expires_at"] = time.time() + int(device_data["expires_in"])
        threading.Thread(
            target=_xai_device_poller,
            args=(sid,),
            daemon=True,
            name=f"oauth-poll-{sid[:6]}",
        ).start()
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": str(device_data["user_code"]),
            "verification_url": str(
                device_data.get("verification_uri_complete")
                or device_data["verification_uri"]
            ),
            "expires_in": int(device_data["expires_in"]),
            "poll_interval": int(device_data["interval"]),
        }

    raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support device-code flow")


def _nous_poller(session_id: str) -> None:
    """Background poller that drives a Nous device-code flow to completion."""
    from hermes_cli.auth import (
        _poll_for_token,
        refresh_nous_oauth_from_state,
    )
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
    scope = sess.get("scope")
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
            token_data = _poll_for_token(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                device_code=device_code,
                expires_in=expires_in,
                poll_interval=interval,
            )
        # Same post-processing as _nous_device_code_login (validate/refresh JWT)
        now = datetime.now(timezone.utc)
        token_ttl = int(token_data.get("expires_in") or 0)
        auth_state = {
            "portal_base_url": portal_base_url,
            "inference_base_url": token_data.get("inference_base_url"),
            "client_id": client_id,
            "scope": token_data.get("scope") or scope,
            "token_type": token_data.get("token_type", "Bearer"),
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "obtained_at": now.isoformat(),
            "expires_at": (
                datetime.fromtimestamp(now.timestamp() + token_ttl, tz=timezone.utc).isoformat()
                if token_ttl else None
            ),
            "expires_in": token_ttl,
        }
        with _profile_scope(_oauth_session_profile(session_id)):
            full_state = refresh_nous_oauth_from_state(
                auth_state,
                timeout_seconds=15.0,
                force_refresh=False,
            )
            from hermes_cli.auth import persist_nous_credentials
            persist_nous_credentials(full_state)
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: nous login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("nous device-code poll failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = str(e)


def _minimax_poller(session_id: str) -> None:
    """Background poller that drives a MiniMax OAuth flow to completion.

    Mirrors `_nous_poller` but calls the MiniMax-specific token endpoint,
    which uses a PKCE-style ``code_verifier`` + ``user_code`` rather than
    the ``device_code`` field used by Nous. On success, builds the same
    auth_state dict that ``_minimax_oauth_login`` (the CLI flow) builds
    and persists via ``_minimax_save_auth_state`` — so the dashboard
    path leaves the system in the same state as
    ``hermes auth add minimax-oauth``.
    """
    from hermes_cli.auth import (
        _minimax_poll_token,
        _minimax_resolve_token_expiry_unix,
        _minimax_save_auth_state,
        MINIMAX_OAUTH_GLOBAL_INFERENCE,
        MINIMAX_OAUTH_SCOPE,
    )
    from datetime import datetime, timezone
    import httpx
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        return
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    user_code = sess["user_code"]
    code_verifier = sess["code_verifier"]
    interval_ms = sess.get("interval_ms")
    expired_in_raw = sess["expired_in_raw"]
    try:
        with httpx.Client(
            timeout=httpx.Timeout(15.0),
            headers={"Accept": "application/json"},
            follow_redirects=True,
        ) as client:
            token_data = _minimax_poll_token(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                user_code=user_code,
                code_verifier=code_verifier,
                expired_in=expired_in_raw,
                interval_ms=interval_ms,
            )
        # Build the auth_state dict in the same shape as the CLI flow's
        # `_minimax_oauth_login` so `_minimax_save_auth_state` writes
        # the canonical record. Region is fixed to "global" for the
        # dashboard path; cn-region operators can still use the CLI
        # flow which supports `--region cn`.
        now = datetime.now(timezone.utc)
        expires_at_ts = _minimax_resolve_token_expiry_unix(
            int(token_data["expired_in"]), now=now,
        )
        expires_in_s = max(0, int(expires_at_ts - now.timestamp()))
        auth_state = {
            "provider": "minimax-oauth",
            "region": sess.get("region", "global"),
            "portal_base_url": portal_base_url,
            "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
            "client_id": client_id,
            "scope": MINIMAX_OAUTH_SCOPE,
            "token_type": token_data.get("token_type", "Bearer"),
            "access_token": token_data["access_token"],
            "refresh_token": token_data["refresh_token"],
            "resource_url": token_data.get("resource_url"),
            "obtained_at": now.isoformat(),
            "expires_at": datetime.fromtimestamp(
                expires_at_ts, tz=timezone.utc
            ).isoformat(),
            "expires_in": expires_in_s,
        }
        with _profile_scope(_oauth_session_profile(session_id)):
            _minimax_save_auth_state(auth_state)
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: minimax login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("minimax device-code poll failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = str(e)


def _xai_device_poller(session_id: str) -> None:
    """Background poller for xAI's OAuth device-code flow."""
    import httpx
    from hermes_cli.auth import (
        _save_xai_oauth_tokens,
        _xai_oauth_discovery,
        _xai_oauth_poll_device_token,
        unsuppress_credential_source,
    )

    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        return
    device_code = sess["device_code"]
    interval = int(sess["interval"])
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    try:
        discovery = _xai_oauth_discovery(20.0)
        with httpx.Client(
            timeout=httpx.Timeout(20.0),
            headers={"Accept": "application/json"},
        ) as client:
            token_data = _xai_oauth_poll_device_token(
                client,
                token_endpoint=discovery["token_endpoint"],
                device_code=device_code,
                expires_in=expires_in,
                poll_interval=interval,
            )
        tokens = {
            "access_token": str(token_data.get("access_token", "") or "").strip(),
            "refresh_token": str(token_data.get("refresh_token", "") or "").strip(),
            "id_token": str(token_data.get("id_token", "") or "").strip(),
            "expires_in": token_data.get("expires_in"),
            "token_type": str(token_data.get("token_type") or "Bearer").strip() or "Bearer",
        }
        with _profile_scope(_oauth_session_profile(session_id)):
            _save_xai_oauth_tokens(
                tokens,
                discovery=discovery,
                last_refresh=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                auth_mode="oauth_device_code",
            )
            # The singleton write above is the single source of truth: the
            # credential-pool load seeds it as the canonical ``device_code``
            # entry. Do NOT also insert a parallel ``manual:dashboard_*`` pool
            # entry — that duplicates the single-use refresh token across two
            # entries and triggers rotation churn / ``refresh_token_reused``.
            # An interactive dashboard login is also an explicit re-enable
            # signal, so clear any ``device_code`` suppression left by a
            # prior ``hermes auth remove xai-oauth`` (mirrors auth_add_command
            # and the ``hermes model`` re-login path in _login_xai_oauth).
            unsuppress_credential_source("xai-oauth", "device_code")
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: xai login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("xai device-code poll failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = str(e)


def _http_response_error_detail(resp: Any) -> str:
    """Best-effort extraction of a short provider error detail."""
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            parts = [
                str(error.get(key, "")).strip()
                for key in ("message", "error_description", "code", "type")
                if str(error.get(key, "")).strip()
            ]
            if parts:
                return ": ".join(parts)
        if isinstance(error, str) and error.strip():
            return error.strip()
        for key in ("detail", "message", "error_description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    text = str(getattr(resp, "text", "") or "").strip()
    return text[:500]


def _codex_device_code_start_error(resp: Any) -> str:
    """Dashboard-facing OpenAI Codex device-code start failure."""
    status = getattr(resp, "status_code", "unknown")
    detail = _http_response_error_detail(resp)
    lower = detail.lower()
    if "device" in lower and ("authori" in lower or "enable" in lower):
        message = (
            "OpenAI rejected the device-code login request. Your OpenAI "
            "account may need device-code authorization enabled before Hermes "
            "can start this dashboard login. Enable device-code authorization "
            "in OpenAI, then return here and click Login again."
        )
    else:
        message = (
            "OpenAI rejected the device-code login request. Please try Login "
            "again from the dashboard after checking your OpenAI account settings."
        )
    if detail:
        return f"{message} (HTTP {status}: {detail})"
    return f"{message} (HTTP {status})"


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
            raise RuntimeError(_codex_device_code_start_error(resp))
        device_data = resp.json()
        user_code = device_data.get("user_code", "")
        device_auth_id = device_data.get("device_auth_id", "")
        poll_interval = max(3, int(device_data.get("interval", "5")))
        if not user_code or not device_auth_id:
            raise RuntimeError("device-code response missing user_code or device_auth_id")
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
        deadline = time.monotonic() + sess["expires_in"]
        code_resp = None
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            while time.monotonic() < deadline:
                time.sleep(poll_interval)
                poll = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )
                if poll.status_code == 200:
                    code_resp = poll.json()
                    break
                if poll.status_code in {403, 404}:
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
            raise RuntimeError("device-auth response missing authorization_code/code_verifier")
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

        from hermes_cli.auth import _save_codex_tokens

        with _profile_scope(_oauth_session_profile(session_id)):
            _save_codex_tokens({
                "access_token": access_token,
                "refresh_token": refresh_token,
            })
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
async def start_oauth_login(
    provider_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Initiate an OAuth login flow. Token-protected."""
    _require_token(request)
    _gc_oauth_sessions()
    _validate_oauth_profile(profile)
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
        # The pkce branch is gated on provider_id == "anthropic" because
        # `_start_anthropic_pkce()` is hardcoded to the Anthropic flow.
        # Routing any other future pkce-flagged provider through it would
        # silently launch the Anthropic OAuth flow (the bug fixed in this
        # change for MiniMax). New PKCE providers must add their own
        # start function and an explicit branch here.
        if catalog_entry["flow"] == "pkce" and provider_id == "anthropic":
            return _start_anthropic_pkce(profile=profile)
        if catalog_entry["flow"] == "device_code":
            return await _start_device_code_flow(provider_id, profile=profile)
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
async def submit_oauth_code(
    provider_id: str,
    body: OAuthSubmitBody,
    request: Request,
    profile: Optional[str] = None,
):
    """Submit the auth code for PKCE flows. Token-protected."""
    _require_token(request)
    if provider_id == "anthropic":
        return await asyncio.get_running_loop().run_in_executor(
            None, _submit_anthropic_pkce, body.session_id, body.code, profile,
        )
    raise HTTPException(status_code=400, detail=f"submit not supported for {provider_id}")


@app.get("/api/providers/oauth/{provider_id}/poll/{session_id}")
async def poll_oauth_session(
    provider_id: str,
    session_id: str,
    profile: Optional[str] = None,
):
    """Poll a session's status (no auth — read-only state).

    Shared by the device-code flows (Nous, OpenAI Codex, MiniMax, xAI).
    Each surfaces progress through the same background-worker-updated
    ``status`` field, so a single poll endpoint serves them all.
    """
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
async def cancel_oauth_session(
    session_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Cancel a pending OAuth session. Token-protected."""
    _require_token(request)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.pop(session_id, None)
    if sess is None:
        return {"ok": False, "message": "session not found"}
    return {"ok": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# Session detail endpoints
# ---------------------------------------------------------------------------



def _session_latest_descendant(session_id: str, db):
    """Resolve a session id to the newest child leaf session.

    /model may create child sessions. Dashboard refresh should continue the
    newest child instead of reopening the old parent.
    """
    def row_get(row, key, index):
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            try:
                return row[index]
            except Exception:
                return None

    sid = db.resolve_session_id(session_id)
    if not sid or not db.get_session(sid):
        return None, []

    conn = (
        getattr(db, "conn", None)
        or getattr(db, "_conn", None)
        or getattr(db, "connection", None)
        or getattr(db, "_connection", None)
    )

    rows = []
    if conn is not None:
        raw_rows = conn.execute(
            """
            WITH RECURSIVE descendants(id, parent_session_id, started_at) AS (
                SELECT id, parent_session_id, started_at FROM sessions WHERE id = ?
                UNION
                SELECT s.id, s.parent_session_id, s.started_at
                FROM sessions s
                JOIN descendants d ON s.parent_session_id = d.id
            )
            SELECT id, parent_session_id, started_at FROM descendants
            """,
            (sid,),
        ).fetchall()
        for row in raw_rows:
            rows.append({
                "id": row_get(row, "id", 0),
                "parent_session_id": row_get(row, "parent_session_id", 1),
                "started_at": row_get(row, "started_at", 2),
            })
    else:
        rows = db.list_sessions_rich(limit=10000, offset=0, compact_rows=True)

    children = {}
    for row in rows:
        rid = row.get("id")
        parent = row.get("parent_session_id")
        if rid and parent:
            children.setdefault(parent, []).append(row)

    def started(row):
        try:
            return float(row.get("started_at") or 0)
        except Exception:
            return 0.0

    current = sid
    path = [sid]
    seen = {sid}

    while children.get(current):
        candidates = [r for r in children[current] if r.get("id") not in seen]
        if not candidates:
            break
        candidates.sort(key=started, reverse=True)
        current = candidates[0]["id"]
        path.append(current)
        seen.add(current)

    return current, path


# CRITICAL — every literal-path route below MUST be declared BEFORE the
# templated ``/api/sessions/{session_id}`` family that follows. FastAPI/
# Starlette match routes in registration order, and the ``{session_id}``
# pattern is unconstrained — it would otherwise swallow e.g.
# ``DELETE /api/sessions/empty``, ``POST /api/sessions/bulk-delete``, or
# ``GET /api/sessions/stats`` as "operate on the session with id
# 'empty'" / "'bulk-delete'" / "'stats'", which would 404 (or worse,
# succeed and delete the wrong row). Same story as the older
# ``/api/sessions/search`` endpoint up at line ~1191. If you split or
# reorder this block, move every route in it together.
class BulkDeleteSessions(BaseModel):
    ids: List[str]
    profile: Optional[str] = None


@app.post("/api/sessions/bulk-delete")
async def bulk_delete_sessions_endpoint(body: BulkDeleteSessions):
    """Delete every session in ``body.ids`` in a single DB transaction.

    Backs the dashboard's bulk-select-and-delete flow on the sessions
    page. POST (not DELETE) because most HTTP clients refuse to send a
    request body on DELETE and a body is the natural shape for a list
    of IDs — Starlette accepts both, but POSTing a list keeps proxies,
    curl, and the browser ``fetch`` API consistent.

    Per-row contract matches :meth:`SessionDB.delete_sessions`:

    * Unknown IDs are silently skipped (the response ``deleted`` count
      reflects what really happened, not the input length). This is
      deliberate — UI selection state can race against another tab's
      delete, and we'd rather succeed-on-the-rest than fail-the-whole-
      batch.
    * Children of every deleted parent are orphaned, not cascade-
      deleted.
    * Active and archived sessions ARE deleted when explicitly
      selected — unlike ``DELETE /api/sessions/empty``, the user
      hand-picked the rows so we trust the selection.
    * Like the other session-delete endpoints, this does NOT pass a
      ``sessions_dir`` through; on-disk transcript / request-dump
      cleanup runs at the CLI/agent layer on the next prune pass.

    The response carries the actual deleted count, so the dashboard
    can surface it in a toast. The IDs that were removed are not
    echoed back because the client already knows what it asked to
    delete (unknown IDs are silently skipped — see contract above)
    and can prune its in-memory list directly from the request.
    """
    # Enforce a hard cap so a runaway/typo'd selection can't lock the
    # DB writer for an extended window. The dashboard pages 20 rows
    # at a time; 500 covers a "select all on every page in a
    # reasonable scrollback" worst case without opening the door to
    # multi-thousand-row transactions.
    if len(body.ids) > 500:
        raise HTTPException(
            status_code=400,
            detail="ids must contain at most 500 entries",
        )
    db = _open_session_db_for_profile(body.profile)
    try:
        deleted = db.delete_sessions(body.ids)
        return {"ok": True, "deleted": deleted}
    finally:
        db.close()


@app.get("/api/sessions/empty/count")
async def count_empty_sessions_endpoint(profile: Optional[str] = None):
    """Return the number of empty, ended, non-archived sessions.

    Drives the dashboard's "Delete empty (N)" button — when N is 0 the
    UI hides the affordance so users aren't presented with a button
    that does nothing. Cheap, single-COUNT query.
    """
    db = _open_session_db_for_profile(profile)
    try:
        return {"count": db.count_empty_sessions()}
    finally:
        db.close()


@app.delete("/api/sessions/empty")
async def delete_empty_sessions_endpoint(profile: Optional[str] = None):
    """Delete every empty (``message_count == 0``), ended,
    non-archived session in a single transaction.

    Safety contract mirrors :meth:`SessionDB.delete_empty_sessions`:

    * Active sessions are skipped (``ended_at IS NULL``) so a live
      agent isn't yanked mid-handshake.
    * Archived sessions are skipped — the user explicitly chose to
      keep those rows.
    * Children of deleted parents are orphaned, not cascade-deleted.

    Like the single-session ``DELETE /api/sessions/{id}`` endpoint
    below, this doesn't pass a ``sessions_dir`` through — the on-disk
    transcript / request-dump cleanup is wired at the CLI/agent layer
    but the web server historically leaves file cleanup to the next
    prune-on-startup pass. Matching that pre-existing trade-off keeps
    the two delete endpoints' DB-vs-disk behaviour consistent.
    """
    db = _open_session_db_for_profile(profile)
    try:
        deleted = db.delete_empty_sessions()
        return {"ok": True, "deleted": deleted}
    finally:
        db.close()


@app.get("/api/sessions/stats")
async def get_session_stats(profile: Optional[str] = None):
    """Session-store statistics for the Sessions page (mirrors `hermes sessions stats`).

    Registered before ``/api/sessions/{session_id}`` so the literal ``stats``
    path isn't captured as a session id by the parameterized route.
    """
    db = _open_session_db_for_profile(profile)
    try:
        total = db.session_count(include_archived=True)
        active_store = db.session_count(include_archived=False)
        archived = db.session_count(archived_only=True)
        messages = db.message_count()
        by_source: Dict[str, int] = {}
        try:
            for s in db.list_sessions_rich(limit=10000, include_archived=True, compact_rows=True):
                src = str(s.get("source") or "cli")
                by_source[src] = by_source.get(src, 0) + 1
        except Exception:
            pass
        return {
            "total": total,
            "active_store": active_store,
            "archived": archived,
            "messages": messages,
            "by_source": by_source,
        }
    finally:
        db.close()


def _open_session_db_for_profile(profile: Optional[str]):
    """Open a SessionDB for read paths, optionally for another profile.

    ``profile`` None/empty → this process's own ``state.db`` (the common,
    single-profile case). A named profile opens that profile's on-disk
    ``state.db`` directly so the primary backend can serve cross-profile reads
    (transcripts, detail) without spawning that profile's backend.
    """
    from hermes_state import SessionDB
    if not profile:
        return SessionDB()
    _name, home = _cron_profile_home(profile)
    return SessionDB(db_path=Path(home) / "state.db")


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str, profile: Optional[str] = None):
    db = _open_session_db_for_profile(profile)
    try:
        sid = db.resolve_session_id(session_id)
        session = db.get_session(sid) if sid else None
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if profile:
            session["profile"] = _cron_profile_home(profile)[0]
        return session
    finally:
        db.close()



@app.get("/api/sessions/{session_id}/latest-descendant")
async def get_session_latest_descendant(
    session_id: str,
    profile: Optional[str] = None,
):
    db = _open_session_db_for_profile(profile)
    try:
        latest, path = _session_latest_descendant(session_id, db)
        if not latest:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "requested_session_id": path[0] if path else session_id,
            "session_id": latest,
            "path": path,
            "changed": bool(path and latest != path[0]),
        }
    finally:
        db.close()

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    profile: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
):
    db = _open_session_db_for_profile(profile)
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        sid = db.resolve_resume_session_id(sid)
        # Clamp limit to prevent abuse (max 500 per page)
        _limit = min(limit, 500) if limit is not None else None
        messages = db.get_messages(sid, limit=_limit, offset=offset)
        return {
            "session_id": sid,
            "messages": messages,
            "pagination": {
                "limit": _limit,
                "offset": offset,
                "returned": len(messages),
            },
        }
    finally:
        db.close()


@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, profile: Optional[str] = None):
    # ``profile`` deletes a session belonging to another (local) profile by
    # opening its state.db directly. Remote profiles never reach here — the
    # desktop routes their DELETE to the remote backend. Omit for current/default.
    db = _open_session_db_for_profile(profile)
    try:
        # Resolve exact ids / unique prefixes like every other session endpoint
        # (detail, messages, rename, export all do). A session that no longer
        # exists is an idempotent success: DELETE's contract is "ensure it's
        # gone", and the desktop optimistically removes the row then RESTORES it
        # on any error — so a 404 on an already-absent row resurrected a ghost
        # row and surfaced "session not found". /goal + auto-compression churn
        # leaves transient empty rows (reaped by empty-session hygiene) that
        # race the sidebar snapshot, which is exactly when this fired. Mirrors
        # the bulk-delete endpoint, which already treats ghost ids as success.
        sid = db.resolve_session_id(session_id)
        if not sid:
            return {"ok": True, "already_absent": True}
        db.delete_session(sid)
        return {"ok": True}
    finally:
        db.close()


class SessionRename(BaseModel):
    title: Optional[str] = None
    archived: Optional[bool] = None
    # Mutate a session belonging to another profile (opens its state.db). Omit
    # for the current/default profile.
    profile: Optional[str] = None


@app.patch("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, body: SessionRename):
    """Update a session: rename (or clear its title) and/or archive it.

    ``title`` renames (empty/null clears the title); ``archived`` soft-hides or
    restores the session. Either field may be omitted. ``profile`` targets
    another profile's session.
    """
    db = _open_session_db_for_profile(body.profile)
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        if body.title is None and body.archived is None:
            raise HTTPException(
                status_code=400,
                detail="Nothing to update; provide 'title' and/or 'archived'.",
            )
        if body.title is not None:
            try:
                db.set_session_title(sid, body.title or "")
            except ValueError as e:
                # Title too long, invalid characters, or already in use.
                raise HTTPException(status_code=400, detail=str(e))
        if body.archived is not None:
            db.set_session_archived(sid, body.archived)
        result = {"ok": True, "title": db.get_session_title(sid) or ""}
        if body.archived is not None:
            result["archived"] = bool(body.archived)
        return result
    finally:
        db.close()


@app.get("/api/sessions/{session_id}/export")
async def export_session_endpoint(session_id: str, profile: Optional[str] = None):
    """Export a single session (metadata + messages) as JSON."""
    db = _open_session_db_for_profile(profile)
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        data = db.export_session(sid)
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return data
    finally:
        db.close()


class SessionPrune(BaseModel):
    older_than_days: Optional[float] = 90
    source: Optional[str] = None
    profile: Optional[str] = None
    # Extended filters (all optional, AND together — mirrors the CLI flags)
    started_before: Optional[float] = None  # epoch seconds
    started_after: Optional[float] = None  # epoch seconds
    title_like: Optional[str] = None
    end_reason: Optional[str] = None
    cwd_prefix: Optional[str] = None
    min_messages: Optional[int] = None
    max_messages: Optional[int] = None
    model_like: Optional[str] = None
    provider: Optional[str] = None
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None
    branch_like: Optional[str] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    min_cost: Optional[float] = None
    max_cost: Optional[float] = None
    min_tool_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    include_archived: bool = False
    dry_run: bool = False


@app.post("/api/sessions/prune")
async def prune_sessions_endpoint(body: SessionPrune):
    """Delete ended sessions matching filters (mirrors `hermes sessions prune`)."""
    has_window = (
        body.started_before is not None or body.started_after is not None
    )
    if body.older_than_days is not None and body.older_than_days < 1 and not has_window:
        raise HTTPException(status_code=400, detail="older_than_days must be >= 1")
    # Mirror the CLI: the implicit 90-day cutoff only applies to a truly bare
    # prune. Any attribute filter (source, title, model, ...) suppresses it
    # unless the caller explicitly sent older_than_days.
    _attr_filters_set = any(
        getattr(body, f) is not None
        for f in (
            "source", "title_like", "end_reason", "cwd_prefix",
            "min_messages", "max_messages", "model_like", "provider",
            "user_id", "chat_id", "chat_type", "branch_like",
            "min_tokens", "max_tokens", "min_cost", "max_cost",
            "min_tool_calls", "max_tool_calls",
        )
    )
    _older_than_explicit = "older_than_days" in body.model_fields_set
    _effective_older_than = body.older_than_days
    if has_window or (_attr_filters_set and not _older_than_explicit):
        _effective_older_than = None
    profile_home = _cron_profile_home(body.profile)[1] if body.profile else get_hermes_home()
    db = _open_session_db_for_profile(body.profile)
    try:
        filters = dict(
            older_than_days=_effective_older_than,
            source=(body.source or None),
            started_before=body.started_before,
            started_after=body.started_after,
            title_like=(body.title_like or None),
            end_reason=(body.end_reason or None),
            cwd_prefix=(body.cwd_prefix or None),
            min_messages=body.min_messages,
            max_messages=body.max_messages,
            model_like=(body.model_like or None),
            provider=(body.provider or None),
            user_id=(body.user_id or None),
            chat_id=(body.chat_id or None),
            chat_type=(body.chat_type or None),
            branch_like=(body.branch_like or None),
            min_tokens=body.min_tokens,
            max_tokens=body.max_tokens,
            min_cost=body.min_cost,
            max_cost=body.max_cost,
            min_tool_calls=body.min_tool_calls,
            max_tool_calls=body.max_tool_calls,
            archived=None if body.include_archived else False,
        )
        if body.dry_run:
            rows = db.list_prune_candidates(**filters)
            return {
                "ok": True,
                "removed": 0,
                "matched": len(rows),
                # Rows are ordered oldest-first.
                "oldest_started_at": rows[0]["started_at"] if rows else None,
                "newest_started_at": rows[-1]["started_at"] if rows else None,
                "sessions": [
                    {
                        "id": r["id"],
                        "source": r["source"],
                        "title": r.get("title"),
                        "model": r.get("model"),
                        "started_at": r["started_at"],
                        "message_count": r["message_count"],
                    }
                    for r in rows
                ],
            }
        sessions_dir = profile_home / "sessions"
        removed = db.prune_sessions(
            sessions_dir=sessions_dir if sessions_dir.exists() else None,
            **filters,
        )
        return {"ok": True, "removed": removed}
    finally:
        db.close()


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
        log_path, min(lines, 500) if not search else 2000,
        has_filters=has_filters,
        min_level=min_level,
        component_prefixes=comp_prefixes,
    )
    # Post-filter by search term (case-insensitive substring match).
    # _read_tail doesn't support free-text search, so we filter here and
    # trim to the requested line count afterward.
    if search:
        needle = search.lower()
        result = [l for l in result if needle in l.lower()][-min(lines, 500):]
    return {"file": file, "lines": result}


# ---------------------------------------------------------------------------
# Cron job management endpoints
# ---------------------------------------------------------------------------


class CronJobCreate(BaseModel):
    prompt: str = ""
    schedule: str
    name: str = ""
    deliver: str = "local"
    skills: Optional[List[str]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    script: Optional[str] = None
    context_from: Optional[Any] = None
    enabled_toolsets: Optional[List[str]] = None
    workdir: Optional[str] = None
    no_agent: bool = False


class CronJobUpdate(BaseModel):
    updates: dict


def _cron_optional_text(value: Any, *, strip_trailing_slash: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _cron_string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = re.split(r"[\n,]", value)
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        return None
    items = [str(item).strip() for item in raw_items if str(item).strip()]
    return items or None


def _normalize_dashboard_cron_script(value: Any, profile_home: Path) -> Optional[str]:
    """Validate a dashboard-selected cron script against the profile sandbox."""
    text = _cron_optional_text(value)
    if not text:
        return None

    scripts_root = (profile_home / "scripts").resolve()
    raw_path = Path(text).expanduser()
    candidate = raw_path.resolve() if raw_path.is_absolute() else (scripts_root / raw_path).resolve()
    try:
        relative = candidate.relative_to(scripts_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"script must be inside {scripts_root}",
        ) from exc
    if not candidate.exists():
        raise HTTPException(status_code=400, detail=f"script does not exist: {candidate}")
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail=f"script is not a file: {candidate}")
    return str(relative)


def _validate_dashboard_cron_effective_job(job: Dict[str, Any]) -> None:
    prompt = _cron_optional_text(job.get("prompt"))
    script = _cron_optional_text(job.get("script"))
    skills = _cron_string_list(job.get("skills")) or _cron_string_list(job.get("skill"))
    no_agent = bool(job.get("no_agent"))

    if no_agent:
        if not script:
            raise HTTPException(
                status_code=400,
                detail="no_agent=True requires a script",
            )
        return

    if not (prompt or skills or script):
        raise HTTPException(
            status_code=400,
            detail="agent cron jobs require a prompt, skill, or script",
        )


def _normalize_dashboard_cron_updates(
    updates: Dict[str, Any],
    profile_home: Path,
) -> Dict[str, Any]:
    """Normalize dashboard JSON into cron.jobs.update_job's storage shape.

    This intentionally stays in the dashboard adapter layer: cron/jobs.py is the
    source of truth for scheduling behaviour; the dashboard only translates form
    payloads into the shapes that existing core functions already accept.
    """
    normalized = dict(updates or {})

    for key in ("model", "provider", "workdir"):
        if key in normalized:
            normalized[key] = _cron_optional_text(normalized[key])
    if "script" in normalized:
        normalized["script"] = _normalize_dashboard_cron_script(
            normalized["script"],
            profile_home,
        )
    if "base_url" in normalized:
        normalized["base_url"] = _cron_optional_text(
            normalized["base_url"], strip_trailing_slash=True
        )
    if "deliver" in normalized:
        normalized["deliver"] = _cron_optional_text(normalized["deliver"]) or "local"
    if "context_from" in normalized:
        normalized["context_from"] = _cron_string_list(normalized["context_from"])
    if "enabled_toolsets" in normalized:
        normalized["enabled_toolsets"] = _cron_string_list(normalized["enabled_toolsets"])
    return normalized


def _validate_dashboard_cron_context_from(
    refs: Optional[List[str]],
    profile_name: str,
) -> None:
    if not refs:
        return
    for ref in refs:
        if not _call_cron_for_profile(profile_name, "get_job", ref):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"context_from job '{ref}' not found in profile "
                    f"'{profile_name}'"
                ),
            )


def _cron_profile_dicts() -> List[Dict[str, Any]]:
    """Return dashboard profile records, falling back to a directory scan."""
    from hermes_cli import profiles as profiles_mod
    try:
        return [_profile_to_dict(p) for p in profiles_mod.list_profiles()]
    except Exception:
        _log.exception("Failed to list profiles for cron dashboard; falling back to directory scan")
        return _fallback_profile_dicts(profiles_mod)


def _cron_profile_home(profile: Optional[str]) -> Tuple[str, Path]:
    """Resolve a profile query value to (profile_name, HERMES_HOME)."""
    from hermes_cli import profiles as profiles_mod

    raw = (profile or "default").strip() or "default"
    try:
        canon = profiles_mod.normalize_profile_name(raw)
        profiles_mod.validate_profile_name(canon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(canon):
        raise HTTPException(status_code=404, detail=f"Profile '{canon}' does not exist.")
    return canon, profiles_mod.get_profile_dir(canon)


def _annotate_cron_job(job: Dict[str, Any], profile: str, home: Path) -> Dict[str, Any]:
    annotated = dict(job)
    annotated["profile"] = profile
    annotated["profile_name"] = profile
    annotated["hermes_home"] = str(home)
    annotated["is_default_profile"] = profile == "default"
    return annotated


def _call_cron_for_profile(target_profile: Optional[str], func_name: str, *args, **kwargs):
    """Run cron.jobs helpers against the selected profile's cron directory.

    The dashboard is a single process that can inspect many profiles. Route
    storage through cron.jobs' execution-context override so dashboard calls
    cannot retarget a concurrent desktop ticker's load/save transaction.
    """
    profile_name, home = _cron_profile_home(target_profile)
    from cron import jobs as cron_jobs
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(str(home))
    try:
        with cron_jobs.use_cron_store(home):
            result = getattr(cron_jobs, func_name)(*args, **kwargs)
    finally:
        reset_hermes_home_override(token)

    if isinstance(result, list):
        return [_annotate_cron_job(j, profile_name, home) for j in result]
    if isinstance(result, dict):
        return _annotate_cron_job(result, profile_name, home)
    return result


def _find_cron_job_profile(job_id: str) -> Optional[str]:
    for profile in _cron_profile_dicts():
        name = str(profile.get("name") or "")
        if not name:
            continue
        jobs = _call_cron_for_profile(name, "list_jobs", True)
        if any(j.get("id") == job_id or j.get("name") == job_id for j in jobs):
            return name
    return None


def _list_cron_jobs_sync(profile: str = "all"):
    requested = (profile or "all").strip()
    if requested.lower() != "all":
        return _call_cron_for_profile(requested, "list_jobs", True)

    jobs: List[Dict[str, Any]] = []
    for item in _cron_profile_dicts():
        name = str(item.get("name") or "")
        if not name:
            continue
        try:
            jobs.extend(_call_cron_for_profile(name, "list_jobs", True))
        except Exception:
            _log.exception("Failed to list cron jobs for profile %s", name)
    return jobs


async def _run_cron_dashboard_io(func, *args, **kwargs):
    """Run cron dashboard profile/job I/O outside the FastAPI event loop."""
    if inspect.iscoroutinefunction(func):
        raise TypeError("_run_cron_dashboard_io only accepts sync callables")
    result = await run_in_threadpool(func, *args, **kwargs)
    if inspect.isawaitable(result):
        raise TypeError("_run_cron_dashboard_io sync callable returned an awaitable")
    return result


@app.get("/api/cron/jobs")
async def list_cron_jobs(profile: str = "all"):
    return await _run_cron_dashboard_io(_list_cron_jobs_sync, profile)


def _get_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _call_cron_for_profile(selected, "get_job", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/cron/jobs/{job_id}")
async def get_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_get_cron_job_sync, job_id, profile)


def _list_cron_job_runs_sync(job_id: str, profile: Optional[str] = None, limit: int = 20):
    """Run sessions produced by a cron job, newest first.

    Cron runs are stored as ordinary sessions whose id is
    ``cron_{job_id}_{timestamp}`` (see cron/scheduler.run_job). A job's history
    is therefore every session whose id carries that prefix; ``source='cron'``
    narrows it and the id prefix binds it to this job. Powers the run-history
    list under each job in the desktop cron detail. Same row shape as
    ``/api/sessions`` so the frontend can reuse SessionInfo.

    Backed by ``SessionDB.list_cron_job_runs`` — a bounded ``[prefix, hi)``
    id-range scan, not the compression-chain CTE used for the recents list,
    so the cost scales with the requested window and not the (unbounded) total
    cron history.
    """
    selected = profile or _find_cron_job_profile(job_id)
    # job_id may be a human name; resolve to the canonical id used in run-session ids.
    canonical = job_id
    if selected:
        job = _call_cron_for_profile(selected, "get_job", job_id)
        if job and job.get("id"):
            canonical = str(job["id"])

    try:
        limit_n = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit_n = 20

    db = _open_session_db_for_profile(selected)
    try:
        runs = db.list_cron_job_runs(canonical, limit=limit_n, offset=0)
        now = time.time()
        for s in runs:
            s["is_active"] = (
                s.get("ended_at") is None
                and (now - s.get("last_active", s.get("started_at", 0))) < 300
            )
            s["archived"] = bool(s.get("archived"))
            if selected:
                s["profile"] = selected
        return {"runs": runs, "limit": limit_n}
    finally:
        db.close()


@app.get("/api/cron/jobs/{job_id}/runs")
async def list_cron_job_runs(job_id: str, profile: Optional[str] = None, limit: int = 20):
    return await _run_cron_dashboard_io(_list_cron_job_runs_sync, job_id, profile, limit)


def _create_cron_job_sync(body: CronJobCreate, profile: str = "default"):
    try:
        profile_name, profile_home = _cron_profile_home(profile)
        script = _normalize_dashboard_cron_script(body.script, profile_home)
        skills = _cron_string_list(body.skills)
        context_from = _cron_string_list(body.context_from)
        _validate_dashboard_cron_context_from(context_from, profile_name)
        no_agent = bool(body.no_agent)
        _validate_dashboard_cron_effective_job({
            "prompt": body.prompt,
            "skills": skills,
            "script": script,
            "no_agent": no_agent,
        })
        return _call_cron_for_profile(
            profile_name,
            "create_job",
            prompt=body.prompt or "",
            schedule=body.schedule,
            name=body.name,
            deliver=_cron_optional_text(body.deliver) or "local",
            skills=skills,
            model=_cron_optional_text(body.model),
            provider=_cron_optional_text(body.provider),
            base_url=_cron_optional_text(body.base_url, strip_trailing_slash=True),
            script=script,
            context_from=context_from,
            enabled_toolsets=_cron_string_list(body.enabled_toolsets),
            workdir=_cron_optional_text(body.workdir),
            no_agent=no_agent,
        )
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("POST /api/cron/jobs failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/cron/jobs")
async def create_cron_job(body: CronJobCreate, profile: str = "default"):
    return await _run_cron_dashboard_io(_create_cron_job_sync, body, profile)


@app.get("/api/cron/delivery-targets")
async def get_cron_delivery_targets():
    """Delivery targets the cron dropdown should offer.

    Always includes the implicit ``local`` option. Beyond that, the list is
    derived dynamically from the configured gateway platforms via
    ``cron.scheduler.cron_delivery_targets()`` — no hardcoded platform list. A
    configured platform that hasn't set its cron home channel is still returned
    with ``home_target_set: false`` so the UI can surface it as "configure a
    home channel first" rather than hiding it.
    """
    targets = [
        {
            "id": "local",
            "name": "Local (save only)",
            "home_target_set": True,
            "home_env_var": None,
        }
    ]
    try:
        from cron.scheduler import cron_delivery_targets

        targets.extend(cron_delivery_targets())
    except Exception:
        _log.exception("GET /api/cron/delivery-targets failed")
    return {"targets": targets}


def _update_cron_job_sync(job_id: str, body: CronJobUpdate, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        profile_name, profile_home = _cron_profile_home(selected)
        existing = _call_cron_for_profile(profile_name, "get_job", job_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Job not found")
        updates = _normalize_dashboard_cron_updates(
            body.updates,
            profile_home,
        )
        if "context_from" in updates:
            _validate_dashboard_cron_context_from(
                updates.get("context_from"),
                profile_name,
            )
        execution_fields = {"prompt", "skill", "skills", "script", "no_agent"}
        if execution_fields.intersection(updates):
            effective = {**existing, **updates}
            if "skills" in updates and "skill" not in updates:
                effective["skill"] = None
            _validate_dashboard_cron_effective_job(effective)
        job = _call_cron_for_profile(profile_name, "update_job", job_id, updates)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.put("/api/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_update_cron_job_sync, job_id, body, profile)


def _pause_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _call_cron_for_profile(selected, "pause_job", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/pause")
async def pause_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_pause_cron_job_sync, job_id, profile)


def _resume_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _call_cron_for_profile(selected, "resume_job", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/resume")
async def resume_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_resume_cron_job_sync, job_id, profile)


def _trigger_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _call_cron_for_profile(selected, "trigger_job", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/trigger")
async def trigger_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_trigger_cron_job_sync, job_id, profile)


def _delete_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        removed = _call_cron_for_profile(selected, "remove_job", job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


@app.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_delete_cron_job_sync, job_id, profile)


def _fire_cron_job_for_profile(profile: str, job_id: str) -> bool:
    """Run ONE due cron job end-to-end for ``profile`` via the resolved
    scheduler provider's ``fire_due`` (store CAS claim + ``run_one_job``).

    Scope both cron storage and the runtime Hermes home so the job's store,
    config, credentials, scripts, skills, and output all belong to the selected
    profile. Runs with no live adapters; delivery falls back to the per-platform
    send path.
    """
    _profile_name, home = _cron_profile_home(profile)
    from cron import jobs as cron_jobs
    from cron.scheduler_provider import resolve_cron_scheduler
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(str(home))
    try:
        with cron_jobs.use_cron_store(home):
            provider = resolve_cron_scheduler()
            return bool(provider.fire_due(job_id, adapters=None, loop=None))
    finally:
        reset_hermes_home_override(token)


@app.post("/api/cron/fire")
async def cron_fire_webhook(request: Request):
    """Chronos managed-cron fire webhook (NAS -> agent).

    Authenticated by a short-lived NAS-minted JWT (verified by the pluggable
    Chronos fire-verifier), NOT the dashboard session cookie — so this path is
    in ``PUBLIC_API_PATHS`` to bypass the dashboard auth gate, and the JWT is
    the real gate. This is the inbound half of scale-to-zero managed cron: NAS
    POSTs here at fire time, the agent verifies, claims the job (store CAS, so
    at-most-once across replicas / on a NAS retry), runs it, and re-arms the
    next one-shot.

    Lives on the dashboard app (not the api_server adapter) because the
    dashboard is the agent's always-reachable public HTTP surface on hosted
    deployments; the gateway may be idle/scaled down.

    Returns 202 immediately and runs the job in the background so a long agent
    turn never trips NAS's HTTP timeout.
    """
    from plugins.cron_providers.chronos.verify import get_fire_verifier

    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.startswith("Bearer ") else ""

    cfg = load_config()
    claims = get_fire_verifier()(
        token=token,
        expected_audience=cfg_get(cfg, "cron", "chronos", "expected_audience", default=""),
        jwks_or_key=cfg_get(cfg, "cron", "chronos", "nas_jwks_url", default="") or None,
        issuer=cfg_get(cfg, "cron", "chronos", "portal_url", default="") or None,
    )
    if claims is None:
        return JSONResponse({"error": "invalid fire token"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    job_id = (body or {}).get("job_id") if isinstance(body, dict) else None
    if not job_id:
        return JSONResponse({"error": "missing job_id"}, status_code=400)

    # _find_cron_job_profile walks every profile and lists its jobs (file
    # I/O per profile) — run it off the event loop like the other cron
    # dashboard endpoints.
    profile = await _run_cron_dashboard_io(_find_cron_job_profile, job_id)
    if not profile:
        # Job is gone (cancelled / completed) — nothing to fire. 200 so NAS
        # does not retry a fire that is intentionally absent.
        return JSONResponse({"status": "gone", "job_id": job_id}, status_code=200)

    # Run in the background; the store CAS claim inside fire_due de-dupes a
    # NAS/scheduler retry that arrives while this is in flight.
    asyncio.create_task(
        asyncio.to_thread(_fire_cron_job_for_profile, profile, job_id)
    )
    return JSONResponse({"status": "accepted", "job_id": job_id}, status_code=202)


# ---------------------------------------------------------------------------
# Automation Blueprints — parameterized automation blueprints. The dashboard renders the
# slot schema as a form; submitting instantiates a real cron job via the same
# create_job path. See cron/blueprint_catalog.py for the single source of truth.
# ---------------------------------------------------------------------------
class AutomationBlueprintInstantiate(BaseModel):
    blueprint: str                      # blueprint key, e.g. "morning-brief"
    values: Dict[str, Any] = {}      # filled slot values from the form


@app.get("/api/cron/blueprints")
async def list_cron_blueprints():
    """Return the blueprint catalog as form schemas for the dashboard gallery.

    The ``deliver`` slot's options are rewritten from the user's actually
    configured gateway platforms (plus the universal origin/local/all), so the
    form never offers a platform that isn't connected.
    """
    try:
        from cron.blueprint_catalog import CATALOG, blueprint_catalog_entry

        deliver_options = None
        try:
            from cron.scheduler import cron_delivery_targets

            platforms = [t["id"] for t in cron_delivery_targets() if t.get("id")]
            deliver_options = ["origin", "local", *platforms]
        except Exception:
            _log.debug("cron_delivery_targets unavailable; using static deliver options", exc_info=True)

        entries = []
        for r in CATALOG:
            entry = blueprint_catalog_entry(r)
            if deliver_options:
                for f in entry.get("fields", []):
                    if f.get("name") == "deliver":
                        f["options"] = deliver_options
            entries.append(entry)
        return {"blueprints": entries}
    except Exception as e:
        _log.exception("GET /api/cron/blueprints failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cron/blueprints/instantiate")
async def instantiate_blueprint(body: AutomationBlueprintInstantiate, profile: str = "default"):
    """Fill a blueprint's slots and create the cron job (form-submit path)."""
    try:
        from cron.blueprint_catalog import fill_blueprint, get_blueprint, BlueprintFillError

        blueprint = get_blueprint(body.blueprint)
        if blueprint is None:
            raise HTTPException(status_code=404, detail=f"Unknown blueprint: {body.blueprint}")
        try:
            spec = fill_blueprint(blueprint, body.values)
        except BlueprintFillError as exc:
            # Field-level validation error — 422 so the form can show it inline.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Blueprint-created jobs deliver to the dashboard's configured target by
        # default; the form's deliver slot overrides via spec["deliver"].
        spec.pop("origin", None)
        # create_job does per-profile file I/O — keep it off the event loop
        # like the sibling cron endpoints (partial avoids **spec keys ever
        # colliding with the wrapper's own parameters).
        _create = functools.partial(_call_cron_for_profile, profile, "create_job", **spec)
        return await _run_cron_dashboard_io(_create)
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("POST /api/cron/blueprints/instantiate failed")
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# MCP server endpoints — list / add / remove / test.
#
# Wraps the same config data layer the CLI uses (hermes_cli.mcp_config), so
# servers managed here show up under `hermes mcp list` and vice versa.  Secrets
# in stdio `env` blocks are redacted on read; the agent picks them up from
# config.yaml at session start exactly as with CLI-added servers.
# ---------------------------------------------------------------------------


class MCPServerCreate(BaseModel):
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = []
    # env: KEY=VALUE map for stdio servers (API keys, etc.)
    env: Dict[str, str] = {}
    # auth: "oauth" | "header" | None
    auth: Optional[str] = None
    profile: Optional[str] = None


class MCPServersReplace(BaseModel):
    # Whole-map replace (name → raw server config) for the GUI mcp.json editor.
    servers: Dict[str, Dict[str, Any]] = {}
    profile: Optional[str] = None


def _redact_mcp_env(env: Dict[str, Any]) -> Dict[str, str]:
    """Mask secret-shaped MCP env values for read responses."""
    out: Dict[str, str] = {}
    for k, v in (env or {}).items():
        try:
            out[str(k)] = redact_key(str(v)) if v else ""
        except Exception:
            out[str(k)] = "***"
    return out


def _mcp_server_summary(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    transport = "http" if cfg.get("url") else ("stdio" if cfg.get("command") else "unknown")
    return {
        "name": name,
        "transport": transport,
        "url": cfg.get("url"),
        "command": cfg.get("command"),
        "args": list(cfg.get("args") or []),
        "env": _redact_mcp_env(cfg.get("env") or {}),
        "auth": cfg.get("auth"),
        "enabled": cfg.get("enabled", True) is not False,
        # Tool selection: list of enabled tool names, or None = all.
        "tools": cfg.get("tools"),
    }


@app.get("/api/mcp/servers")
async def list_mcp_servers(profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers

    with _profile_scope(profile):
        servers = _get_mcp_servers()
    return {
        "servers": [
            _mcp_server_summary(name, cfg) for name, cfg in sorted(servers.items())
        ]
    }


@app.post("/api/mcp/servers")
async def add_mcp_server(body: MCPServerCreate, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers, _save_mcp_server

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Server name is required")
    with _profile_scope(body.profile or profile):
        existing = _get_mcp_servers()
    if name in existing:
        raise HTTPException(status_code=409, detail=f"Server '{name}' already exists")
    if not body.url and not body.command:
        raise HTTPException(
            status_code=400,
            detail="Provide either a URL (HTTP/SSE server) or a command (stdio server)",
        )

    server_config: Dict[str, Any] = {}
    if body.url:
        server_config["url"] = body.url.strip()
    if body.command:
        server_config["command"] = body.command.strip()
        if body.args:
            server_config["args"] = list(body.args)
    if body.env:
        server_config["env"] = dict(body.env)
    if body.auth:
        server_config["auth"] = body.auth

    try:
        with _profile_scope(body.profile or profile):
            if not _save_mcp_server(name, server_config):
                raise HTTPException(
                    status_code=400,
                    detail=f"Server '{name}' rejected: suspicious command/args configuration",
                )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("POST /api/mcp/servers failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _mcp_server_summary(name, server_config)


@app.put("/api/mcp/servers")
async def replace_mcp_servers(body: MCPServersReplace, profile: Optional[str] = None):
    """Replace the entire ``mcp_servers`` map (the GUI mcp.json editor's save).

    The generic ``/api/config`` endpoint deep-merges maps, so it can never
    delete a server key, drop an ``enabled: false`` flag, or remove a nested
    field — edits looked saved but the stale entry survived on disk.  This
    endpoint sets the whole map so removals actually persist.  Storage stays
    the config.yaml ``mcp_servers`` key the CLI/TUI already read.
    """
    from hermes_cli.mcp_config import _replace_mcp_servers

    with _profile_scope(body.profile or profile):
        ok, issues = _replace_mcp_servers(body.servers)
    if not ok:
        raise HTTPException(status_code=400, detail="; ".join(issues))
    return {"ok": True}


@app.delete("/api/mcp/servers/{name}")
async def remove_mcp_server(name: str, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _remove_mcp_server

    with _profile_scope(profile):
        removed = _remove_mcp_server(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    return {"ok": True}


@app.post("/api/mcp/servers/{name}/test")
async def test_mcp_server(name: str, profile: Optional[str] = None):
    """Connect to the server, list its tools, disconnect.  Returns tool list."""
    from hermes_cli.mcp_config import (
        _get_mcp_servers,
        _oauth_tokens_present,
        _probe_single_server,
    )

    with _profile_scope(profile):
        servers = _get_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    details: Dict[str, Any] = {}
    # An `auth: oauth` server that serves tools/list anonymously would probe OK
    # with no token — a false green. Require a token on disk for it, matching the
    # /auth verification (some providers don't enforce auth on tools/list).
    needs_oauth_token = servers[name].get("auth") == "oauth"

    def _probe_scoped():
        # Home-only scope (contextvar), NOT _profile_scope. A probe blocks for
        # as long as the server takes to spawn/connect — a stdio `npx` cold
        # start is many seconds — and _profile_scope holds a process-global
        # skills lock for its ENTIRE body. Holding that across the probe
        # serialized every other endpoint (config/skills/toolsets all take the
        # same lock), so a slow server made unrelated requests time out at 15s.
        # The probe touches no skills globals; it only needs the HERMES_HOME
        # override for .env interpolation + OAuth token resolution, which the
        # contextvar provides (copied into this to_thread worker; and
        # _run_on_mcp_loop re-wraps it onto the MCP event-loop thread).
        with _config_profile_scope(profile):
            tools = _probe_single_server(name, servers[name], details=details)
            token_present = _oauth_tokens_present(name) if needs_oauth_token else True
            return tools, token_present

    try:
        # Probe blocks on a dedicated MCP event loop — run in a thread so the
        # FastAPI event loop is never blocked.
        tools, token_present = await asyncio.to_thread(_probe_scoped)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "tools": [],
        }
    if not token_present:
        return {
            "ok": False,
            "error": "OAuth authentication required — no token found.",
            "tools": [],
        }
    return {
        "ok": True,
        "tools": [{"name": t, "description": d} for t, d in tools],
        "prompts": details.get("prompts", 0),
        "resources": details.get("resources", 0),
    }


@app.post("/api/mcp/servers/{name}/auth")
async def auth_mcp_server(name: str, profile: Optional[str] = None):
    """Run the OAuth flow for an HTTP MCP server (opens the system browser).

    Mirrors ``hermes mcp login``: wipe cached OAuth state so the probe forces
    a fresh browser flow, connect, then verify a token actually landed on disk
    (some providers serve tools/list unauthenticated — see
    ``_reauth_oauth_server``).  Blocks until the browser flow completes, so it
    runs in a worker thread.  ``auth: oauth`` is persisted only on success.
    """
    from hermes_cli.mcp_config import (
        _get_mcp_servers,
        _oauth_tokens_present,
        _probe_single_server,
        _save_mcp_server,
    )

    with _profile_scope(profile):
        servers = _get_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    cfg = dict(servers[name])
    if not cfg.get("url"):
        raise HTTPException(
            status_code=400,
            detail="stdio servers authenticate via env keys, not OAuth",
        )
    # A server carrying `headers` uses API-key/bearer auth; a 401 there is a bad
    # key, not an OAuth prompt. Refuse rather than rewrite it to `auth: oauth`
    # and corrupt a working header-auth config. (Explicit `auth: oauth` is fine.)
    if cfg.get("headers") and cfg.get("auth") != "oauth":
        raise HTTPException(
            status_code=400,
            detail="This server uses header/API-key auth, not OAuth — check its key.",
        )
    cfg["auth"] = "oauth"

    def _run():
        from tools.mcp_oauth import HermesTokenStorage, force_interactive_oauth

        # Home-only scope, not _profile_scope: this blocks on the browser flow
        # for up to minutes; holding the shared skills lock that whole time
        # would freeze every other endpoint. Config writes here (_save_mcp_server)
        # resolve HERMES_HOME via the contextvar override, which is all they need.
        with _config_profile_scope(profile), force_interactive_oauth():
            storage = HermesTokenStorage(name)
            # Snapshot before clearing: a re-auth wipes cached state to force a
            # fresh consent, but if the flow fails we must NOT leave the user
            # worse off than before — restore the working token on any failure.
            backup = storage.snapshot()
            try:
                from tools.mcp_oauth_manager import get_manager

                get_manager().remove(name)
            except Exception:
                pass  # No cached state to clear — fine.
            try:
                # The default 30s connect timeout would kill the flow while the
                # user is still on the consent screen — give the browser
                # round-trip the full callback window (300s in mcp_oauth) plus
                # headroom so the connect wrapper can't pre-empt it. Honor a
                # larger configured connect_timeout when the user set one.
                try:
                    _cfg_timeout = float(cfg.get("connect_timeout", 0))
                except (TypeError, ValueError):
                    _cfg_timeout = 0.0
                tools = _probe_single_server(
                    name, cfg, connect_timeout=max(_cfg_timeout, 315)
                )
            except Exception:
                storage.restore(backup)
                raise
            if not _oauth_tokens_present(name):
                storage.restore(backup)
                return {
                    "ok": False,
                    "error": (
                        "The server responded, but no OAuth token was obtained — "
                        "this provider may require a manually-registered OAuth "
                        "client (see `hermes mcp login`)."
                    ),
                    "tools": [],
                }
            _save_mcp_server(name, cfg)
            return {
                "ok": True,
                "tools": [{"name": t, "description": d} for t, d in tools],
            }

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        msg = str(exc)
        # Providers that gate RFC 7591 registration to pre-approved clients
        # (Figma's MCP catalog, etc.) 403 the register call before any
        # authorization URL exists — surface what's actually happening
        # instead of a bare "403 Forbidden".
        lowered = msg.lower()
        if "403" in msg and ("regist" in lowered or "forbidden" in lowered):
            msg = (
                f"'{name}' only allows pre-approved OAuth clients — it rejected "
                "client registration (403), so no browser flow can start. "
                "Options: add a pre-registered client to this server's entry "
                "(oauth: {client_id: ..., client_secret: ...}), or use the "
                "provider's stdio / API-key server instead."
            )
        return {"ok": False, "error": msg, "tools": []}


class MCPEnabledToggle(BaseModel):
    enabled: bool
    profile: Optional[str] = None


@app.put("/api/mcp/servers/{name}/enabled")
async def set_mcp_server_enabled(
    name: str, body: MCPEnabledToggle, profile: Optional[str] = None
):
    """Enable or disable an MCP server (takes effect on next session/gateway).

    Toggles the ``enabled`` key on the server's config.yaml entry — the same
    flag the agent reads at startup.  Disabled servers stay in config so they
    can be re-enabled without re-entering their settings.
    """
    with _profile_scope(body.profile or profile):
        cfg = load_config()
        servers = cfg.get("mcp_servers")
        if not isinstance(servers, dict) or name not in servers:
            raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
        if not isinstance(servers[name], dict):
            raise HTTPException(status_code=400, detail="Malformed server config")
        servers[name]["enabled"] = bool(body.enabled)
        save_config(cfg)
    return {"ok": True, "name": name, "enabled": bool(body.enabled)}


@app.get("/api/mcp/catalog")
async def list_mcp_catalog(profile: Optional[str] = None):
    """Browse the Nous-approved MCP catalog (the optional-mcps/ manifests).

    Each entry reports whether it's already installed and enabled so the UI
    can show install / enabled state inline.  This is the same catalog
    `hermes mcp catalog` / `hermes mcp install` read.  ``profile`` scopes
    the installed/enabled annotations (the catalog itself is repo-shipped
    and identical for every profile).
    """
    try:
        from hermes_cli import mcp_catalog
    except Exception as exc:
        _log.exception("mcp_catalog import failed")
        raise HTTPException(status_code=500, detail=f"Catalog unavailable: {exc}")

    entries = []
    try:
        with _profile_scope(profile):
            catalog_entries = list(mcp_catalog.list_catalog())
            installed_state = {
                e.name: (mcp_catalog.is_installed(e.name), mcp_catalog.is_enabled(e.name))
                for e in catalog_entries
            }
        for entry in catalog_entries:
            auth = entry.auth
            transport = entry.transport
            install = entry.install
            entries.append({
                "name": entry.name,
                "description": entry.description,
                "source": entry.source,
                "transport": transport.type,
                "auth_type": getattr(auth, "type", "none"),
                # Env vars the user must supply (names + prompts only, never values).
                "required_env": [
                    {"name": e.name, "prompt": e.prompt, "required": e.required}
                    for e in getattr(auth, "env", []) or []
                ],
                # Transport details so the UI can show exactly what connects/runs.
                # The trust model (docs: user-guide/features/mcp) tells users to
                # inspect command/args/url and the install bootstrap before
                # installing — surface them rather than hiding them in the repo.
                "command": transport.command,
                "args": list(transport.args or []),
                "url": transport.url,
                # Git bootstrap (present only for entries that clone + build).
                "install_url": install.url if install else None,
                "install_ref": install.ref if install else None,
                "bootstrap": list(install.bootstrap) if install else [],
                # Default tool pre-selection hint and post-install guidance.
                "default_enabled": list(entry.tools.default_enabled)
                if entry.tools.default_enabled is not None
                else None,
                "post_install": entry.post_install or "",
                "needs_install": entry.install is not None,
                "installed": installed_state.get(entry.name, (False, False))[0],
                "enabled": installed_state.get(entry.name, (False, False))[1],
            })
    except HTTPException:
        # Unknown/invalid profile → 404, not a silently-empty catalog.
        raise
    except Exception:
        _log.exception("list_mcp_catalog failed")

    diagnostics = []
    try:
        diagnostics = [
            {"name": n, "kind": k, "message": m}
            for (n, k, m) in mcp_catalog.catalog_diagnostics()
        ]
    except Exception:
        pass

    return {"entries": entries, "diagnostics": diagnostics}


class MCPCatalogInstall(BaseModel):
    name: str
    # env: KEY=VALUE map for catalog entries that declare required env vars.
    env: Dict[str, str] = {}
    enable: bool = True
    profile: Optional[str] = None


@app.post("/api/mcp/catalog/install")
async def install_mcp_catalog_entry(body: MCPCatalogInstall, profile: Optional[str] = None):
    """Install a catalog MCP into config.yaml.

    For HTTP/stdio entries with required env vars, those are written to .env
    via the standard env path so the agent can read them at session start.
    Entries that need a git bootstrap (``needs_install``) are installed via
    the CLI action path because the clone can take time.
    """
    from hermes_cli import mcp_catalog

    name = (body.name or "").strip()
    entry = mcp_catalog.get_entry(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No catalog entry '{name}'")

    # Persist any supplied env vars first (catalog entries declare which names
    # they need; we only write the ones the user provided).
    effective_profile = body.profile or profile
    if body.env:
        with _profile_scope(effective_profile):
            for k, v in body.env.items():
                if v:
                    save_env_value(k, v)

    # Git-bootstrap entries can take a while to clone — run via the background
    # action path so the request returns immediately and the UI can tail logs.
    # The -p subprocess rebinds HERMES_HOME-derived paths in the child.
    if entry.install is not None:
        # Unique per-entry action name: a shared "mcp-install" would let a
        # re-click (or a second entry) overwrite the tracked process/log while
        # the first clone is still running.
        action = _mcp_install_action_name(name)
        try:
            proc = _spawn_hermes_action(
                _profile_cli_args(effective_profile) + ["mcp", "install", name],
                action,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Install failed: {exc}")
        return {"ok": True, "name": name, "background": True, "action": action}

    # No git step — install synchronously via the catalog API. install_entry
    # routes through load_config/save_config + save_env_value, all call-time
    # resolvers, so the context override scopes it. Wrap the to_thread body
    # in the scope INSIDE the thread (contextvars don't propagate into
    # to_thread the other way around — asyncio.to_thread copies context, so
    # setting it here works; keep it explicit for clarity).
    def _install_scoped():
        with _profile_scope(effective_profile):
            mcp_catalog.install_entry(entry, enable=body.enable)

    try:
        await asyncio.to_thread(_install_scoped)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("install_mcp_catalog_entry failed")
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "name": name, "background": False}


def _mcp_install_action_name(name: str) -> str:
    """Unique per-entry mcp-install action name (+ registered log file), so a
    re-click or a second catalog install doesn't overwrite the first's tracked
    process/log while its git clone is still running."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "server"
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    action = f"mcp-install-{slug}-{digest}"
    _ACTION_LOG_FILES.setdefault(action, f"action-{action}.log")
    return action


_ACTION_LOG_FILES.setdefault("computer-use-grant", "action-computer-use-grant.log")


# ---------------------------------------------------------------------------
# Pairing endpoints — approve / revoke / list messaging pairing codes.
#
# These are how a remote admin onboards messaging users (Telegram, Discord, …)
# without shell access.  Wraps gateway.pairing.PairingStore directly.
# ---------------------------------------------------------------------------


class PairingApprove(BaseModel):
    platform: str
    code: str


class PairingRevoke(BaseModel):
    platform: str
    user_id: str


def _pairing_store():
    from gateway.pairing import PairingStore

    return PairingStore()


@app.get("/api/pairing")
async def list_pairing():
    store = _pairing_store()
    return {
        "pending": store.list_pending(),
        "approved": store.list_approved(),
    }


@app.post("/api/pairing/approve")
async def approve_pairing(body: PairingApprove):
    store = _pairing_store()
    platform = (body.platform or "").lower().strip()
    code = (body.code or "").upper().strip()
    if not platform or not code:
        raise HTTPException(status_code=400, detail="platform and code are required")

    result = store.approve_code(platform, code)
    if result:
        return {"ok": True, "user": result}
    if store._is_locked_out(platform):
        raise HTTPException(
            status_code=429,
            detail=f"Platform '{platform}' is locked out after too many failed approvals.",
        )
    raise HTTPException(
        status_code=404,
        detail=f"Code '{code}' not found or expired for platform '{platform}'.",
    )


@app.post("/api/pairing/revoke")
async def revoke_pairing(body: PairingRevoke):
    store = _pairing_store()
    platform = (body.platform or "").lower().strip()
    if not platform or not body.user_id:
        raise HTTPException(status_code=400, detail="platform and user_id are required")
    if store.revoke(platform, body.user_id):
        return {"ok": True}
    raise HTTPException(
        status_code=404,
        detail=f"User {body.user_id} not found in approved list for {platform}.",
    )


@app.post("/api/pairing/clear-pending")
async def clear_pending_pairing():
    store = _pairing_store()
    count = store.clear_pending()
    return {"ok": True, "cleared": count}


# ---------------------------------------------------------------------------
# Webhook subscription endpoints — list / subscribe / remove.
#
# Wraps the same JSON store the CLI uses (hermes_cli.webhook); the webhook
# adapter hot-reloads it without a gateway restart.  Per-route HMAC secrets
# are redacted on read and surfaced once on create.
# ---------------------------------------------------------------------------


class WebhookCreate(BaseModel):
    name: str
    description: Optional[str] = None
    events: List[str] = []
    prompt: Optional[str] = None
    script: Optional[str] = None
    skills: List[str] = []
    deliver: str = "log"
    deliver_only: bool = False
    deliver_chat_id: Optional[str] = None
    # secret: omit to auto-generate
    secret: Optional[str] = None


def _webhook_route_summary(name: str, route: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": route.get("description", ""),
        "events": list(route.get("events") or []),
        "deliver": route.get("deliver", "log"),
        "deliver_only": bool(route.get("deliver_only")),
        "prompt": route.get("prompt", ""),
        "script": route.get("script", ""),
        "skills": list(route.get("skills") or []),
        "created_at": route.get("created_at"),
        "url": f"{base_url}/webhooks/{name}",
        # Secret is masked on read; full value only returned on create.
        "secret_set": bool(route.get("secret")),
        # Default-enabled; only an explicit enabled:false turns a route off.
        "enabled": route.get("enabled", True) is not False,
    }


@app.get("/api/webhooks")
async def list_webhooks():
    import hermes_cli.webhook as wh

    base_url = wh._get_webhook_base_url()
    subs = wh._load_subscriptions()
    return {
        "enabled": wh._is_webhook_enabled(),
        "base_url": base_url,
        "subscriptions": [
            _webhook_route_summary(name, route, base_url)
            for name, route in subs.items()
        ],
    }


@app.post("/api/webhooks/enable")
async def enable_webhooks():
    try:
        _write_platform_enabled("webhook", True)
    except Exception as exc:
        _log.exception("Failed to enable webhook platform from dashboard")
        raise HTTPException(
            status_code=500,
            detail="Failed to enable webhook platform.",
        ) from exc

    restart_result = _restart_gateway_after_webhook_enable()
    return {
        "ok": True,
        "platform": "webhook",
        "enabled": True,
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@app.post("/api/webhooks")
async def create_webhook(body: WebhookCreate):
    import re as _re
    import secrets as _secrets
    import time as _time
    import hermes_cli.webhook as wh

    if not wh._is_webhook_enabled():
        raise HTTPException(
            status_code=400,
            detail="Webhook platform is not enabled. Enable it from the Webhooks page first.",
        )

    name = (body.name or "").strip().lower().replace(" ", "-")
    if not _re.match(r"^[a-z0-9][a-z0-9_-]*$", name):
        raise HTTPException(
            status_code=400,
            detail="Invalid name. Use lowercase alphanumeric with hyphens/underscores.",
        )

    if body.deliver_only and body.deliver == "log":
        raise HTTPException(
            status_code=400,
            detail="Direct delivery requires a real target (telegram, discord, …), not 'log'.",
        )

    secret = body.secret or _secrets.token_urlsafe(32)
    route: Dict[str, Any] = {
        "description": body.description or f"Dashboard-created subscription: {name}",
        "events": [e.strip() for e in body.events if e.strip()],
        "secret": secret,
        "prompt": body.prompt or "",
        "skills": [s.strip() for s in body.skills if s.strip()],
        "deliver": body.deliver or "log",
        "created_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    if body.script and body.script.strip():
        route["script"] = body.script.strip()
    if body.deliver_only:
        route["deliver_only"] = True
    if body.deliver_chat_id:
        route["deliver_extra"] = {"chat_id": body.deliver_chat_id}

    subs = wh._load_subscriptions()
    subs[name] = route
    wh._save_subscriptions(subs)

    base_url = wh._get_webhook_base_url()
    summary = _webhook_route_summary(name, route, base_url)
    # Surface the secret exactly once, on create.
    summary["secret"] = secret
    return summary


@app.delete("/api/webhooks/{name}")
async def delete_webhook(name: str):
    import hermes_cli.webhook as wh

    key = (name or "").strip().lower()
    subs = wh._load_subscriptions()
    if key not in subs:
        raise HTTPException(status_code=404, detail=f"No subscription named '{key}'")
    del subs[key]
    wh._save_subscriptions(subs)
    return {"ok": True}


class WebhookEnabledToggle(BaseModel):
    enabled: bool


@app.put("/api/webhooks/{name}/enabled")
async def set_webhook_enabled(name: str, body: WebhookEnabledToggle):
    """Enable or disable a webhook route.

    Disabled routes stay in the subscriptions file (so they can be
    re-enabled) but the gateway rejects incoming events with 403.  The
    gateway hot-reloads the subscriptions file, so this takes effect on the
    next event without a restart.
    """
    import hermes_cli.webhook as wh

    key = (name or "").strip().lower()
    subs = wh._load_subscriptions()
    if key not in subs:
        raise HTTPException(status_code=404, detail=f"No subscription named '{key}'")
    subs[key]["enabled"] = bool(body.enabled)
    wh._save_subscriptions(subs)
    return {"ok": True, "name": key, "enabled": bool(body.enabled)}


# ---------------------------------------------------------------------------
# Gateway lifecycle endpoints — start / stop.
#
# restart + update already exist above; these complete the lifecycle so a
# remote admin can bring the gateway up or down without shell access.  Both
# spawn the real `hermes gateway <verb>` so behaviour matches the CLI exactly.
# Status is already surfaced by /api/status (gateway_running/state/platforms).
# ---------------------------------------------------------------------------


@app.post("/api/gateway/start")
async def start_gateway(profile: Optional[str] = None):
    try:
        proc = _spawn_hermes_action(_gateway_subcommand(profile, "start"), "gateway-start")
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway start")
        raise HTTPException(status_code=500, detail=f"Failed to start gateway: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "gateway-start"}


@app.post("/api/gateway/stop")
async def stop_gateway(profile: Optional[str] = None):
    try:
        proc = _spawn_hermes_action(_gateway_subcommand(profile, "stop"), "gateway-stop")
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway stop")
        raise HTTPException(status_code=500, detail=f"Failed to stop gateway: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "gateway-stop"}


# ---------------------------------------------------------------------------
# Credential pool endpoints — list / add / remove rotation keys.
#
# The credential pool (auth.json -> credential_pool.<provider>[]) holds the
# rotating API keys the agent round-robins through.  Secrets are redacted on
# read; only the agent ever sees the raw values at session start.
# ---------------------------------------------------------------------------


class CredentialPoolAdd(BaseModel):
    provider: str
    # api_key for API-key providers; OAuth pooling stays CLI-only (it needs
    # an interactive browser flow that doesn't belong in a single POST).
    api_key: str
    label: Optional[str] = None


def _pool_entry_summary(entry: Any, index: int) -> Dict[str, Any]:
    """Redacted, display-safe view of one PooledCredential.

    ``index`` is 1-based to match CredentialPool.remove_index().
    """
    token = getattr(entry, "access_token", "") or ""
    return {
        "index": index,
        "id": getattr(entry, "id", None),
        "label": getattr(entry, "label", None),
        "auth_type": getattr(entry, "auth_type", None),
        "source": getattr(entry, "source", None),
        "priority": getattr(entry, "priority", 0),
        "last_status": getattr(entry, "last_status", None),
        "request_count": getattr(entry, "request_count", 0),
        "token_preview": redact_key(token) if token else "",
        "has_refresh": bool(getattr(entry, "refresh_token", None)),
    }


@app.get("/api/credentials/pool")
async def list_credential_pool():
    from agent.credential_pool import load_pool
    from hermes_cli.auth import read_credential_pool

    providers = []
    # read_credential_pool(None) lists every provider that has pooled entries;
    # load_pool() then gives us the rich PooledCredential objects per provider.
    raw_pool = read_credential_pool()
    for provider_id in sorted(raw_pool.keys()):
        try:
            pool = load_pool(provider_id)
        except Exception:
            _log.exception("load_pool(%s) failed", provider_id)
            continue
        entries = pool.entries()
        if not entries:
            continue
        providers.append({
            "provider": provider_id,
            "entries": [
                _pool_entry_summary(e, i) for i, e in enumerate(entries, start=1)
            ],
        })
    return {"providers": providers}


@app.post("/api/credentials/pool")
async def add_credential_pool_entry(body: CredentialPoolAdd):
    import uuid as _uuid
    from agent.credential_pool import (
        load_pool,
        PooledCredential,
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
    )

    provider = (body.provider or "").strip().lower()
    api_key = (body.api_key or "").strip()
    if not provider or not api_key:
        raise HTTPException(status_code=400, detail="provider and api_key are required")

    try:
        pool = load_pool(provider)
        label = (body.label or "").strip() or f"key #{len(pool.entries()) + 1}"
        entry = PooledCredential(
            provider=provider,
            id=_uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=api_key,
        )
        pool.add_entry(entry)
    except Exception as exc:
        _log.exception("POST /api/credentials/pool failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "provider": provider, "count": len(pool.entries())}


@app.delete("/api/credentials/pool/{provider}/{index}")
async def remove_credential_pool_entry(provider: str, index: int):
    """Remove a pool entry.  ``index`` is 1-based (matches the list response)."""
    from agent.credential_pool import load_pool

    provider = (provider or "").strip().lower()
    try:
        pool = load_pool(provider)
        removed = pool.remove_index(index)
    except Exception as exc:
        _log.exception("DELETE /api/credentials/pool failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if removed is None:
        raise HTTPException(status_code=404, detail="No pool entry at that index")
    return {"ok": True, "provider": provider, "count": len(pool.entries())}


# ---------------------------------------------------------------------------
# Memory provider endpoints — status / list providers / select / disable / reset.
#
# Provider setup is dashboard-native when a provider exposes get_config_schema().
# The dashboard never runs interactive provider setup hooks; activation is only
# allowed once the provider is discoverable, available, and has required config.
# ---------------------------------------------------------------------------


class MemoryProviderSelect(BaseModel):
    # "" or "built-in" disables the external provider (built-in only).
    provider: str


class MemoryReset(BaseModel):
    # "all" | "memory" | "user"
    target: str = "all"


@app.get("/api/memory")
async def get_memory_status():
    cfg = load_config()
    active = ""
    mem = cfg.get("memory")
    if isinstance(mem, dict):
        active = _normalize_memory_provider_name(mem.get("provider"))

    # Built-in memory file sizes (so the UI can show what a reset would erase).
    mem_dir = get_hermes_home() / "memories"
    files = {}
    for fname, key in (("MEMORY.md", "memory"), ("USER.md", "user")):
        path = mem_dir / fname
        files[key] = path.stat().st_size if path.exists() else 0

    return {
        "active": active,
        "providers": _discover_memory_provider_statuses(),
        "builtin_files": files,
    }


@app.put("/api/memory/provider")
async def set_memory_provider(body: MemoryProviderSelect):
    provider = _normalize_memory_provider_name(body.provider)

    _require_memory_provider_ready(provider)

    cfg = load_config()
    if not isinstance(cfg.get("memory"), dict):
        cfg["memory"] = {}
    cfg["memory"]["provider"] = provider
    save_config(cfg)
    return {"ok": True, "active": provider}


@app.post("/api/memory/reset")
async def reset_memory(body: MemoryReset):
    target = (body.target or "all").strip().lower()
    if target not in {"all", "memory", "user"}:
        raise HTTPException(status_code=400, detail="target must be all, memory, or user")

    mem_dir = get_hermes_home() / "memories"
    deleted = []
    targets = []
    if target in {"all", "memory"}:
        targets.append("MEMORY.md")
    if target in {"all", "user"}:
        targets.append("USER.md")
    for fname in targets:
        path = mem_dir / fname
        if path.exists():
            try:
                path.unlink()
                deleted.append(fname)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not delete {fname}: {exc}")
    return {"ok": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Operations endpoints — doctor / security audit / backup / import /
# checkpoints / hooks.
#
# Diagnostic and maintenance commands.  The long-running / text-output ones
# (doctor, security audit, backup, import, skills install) are spawned as
# background actions whose logs the dashboard tails via
# /api/actions/{name}/status — same pattern as gateway restart and update.
# The cheap, structured reads (hooks list, checkpoints list) return JSON
# directly.
# ---------------------------------------------------------------------------


@app.post("/api/ops/doctor")
async def run_doctor():
    try:
        proc = _spawn_hermes_action(["doctor"], "doctor")
    except Exception as exc:
        _log.exception("Failed to spawn doctor")
        raise HTTPException(status_code=500, detail=f"Failed to run doctor: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "doctor"}


@app.post("/api/ops/security-audit")
async def run_security_audit():
    try:
        proc = _spawn_hermes_action(["security", "audit"], "security-audit")
    except Exception as exc:
        _log.exception("Failed to spawn security audit")
        raise HTTPException(status_code=500, detail=f"Failed to run security audit: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "security-audit"}


class BackupRequest(BaseModel):
    # Optional output path; defaults to a timestamped zip in the home dir.
    output: Optional[str] = None


def _dashboard_backup_dir() -> Path:
    return get_hermes_home() / "backups"


def _new_dashboard_backup_path() -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return _dashboard_backup_dir() / f"hermes-backup-{stamp}-{secrets.token_hex(4)}.zip"


@app.post("/api/ops/backup")
async def run_backup(body: BackupRequest):
    args = ["backup"]
    archive: Optional[Path] = None
    if body.output:
        args.append(body.output.strip())
    else:
        archive = _new_dashboard_backup_path()
        try:
            archive.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not create backup directory: {exc}",
            )
        args.append(str(archive))
    try:
        proc = _spawn_hermes_action(args, "backup")
    except Exception as exc:
        _log.exception("Failed to spawn backup")
        raise HTTPException(status_code=500, detail=f"Failed to run backup: {exc}")
    response = {"ok": True, "pid": proc.pid, "name": "backup"}
    if archive is not None:
        response["archive"] = str(archive)
    return response


@app.get("/api/ops/backup/download")
async def download_dashboard_backup(archive: str):
    try:
        backup_dir = _dashboard_backup_dir().expanduser().resolve(strict=False)
        target = Path(archive).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backup not found")
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid backup path")

    if not _path_is_under(backup_dir, target):
        raise HTTPException(status_code=403, detail="Backup is outside the dashboard backup directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Backup not found")

    return FileResponse(
        path=str(target),
        media_type="application/zip",
        filename=target.name,
        content_disposition_type="attachment",
    )


class ImportRequest(BaseModel):
    archive: str
    # Pass --force to `hermes import`. The spawned action runs with
    # stdin=DEVNULL, so the CLI's interactive "Continue? [y/N]" overwrite
    # prompt hits EOF and auto-aborts ("Aborted.", exit 1) whenever the
    # target already has a config — which it always does when the dashboard
    # itself is running from it. The dashboard shows its own confirm modal
    # before calling this endpoint, then sends force=True so the restore
    # proceeds non-interactively.
    force: bool = False


@app.post("/api/ops/import")
async def run_import(body: ImportRequest):
    archive = (body.archive or "").strip()
    if not archive:
        raise HTTPException(status_code=400, detail="archive path is required")
    if not os.path.isfile(archive):
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive}")
    args = ["import", archive]
    if body.force:
        args.append("--force")
    try:
        proc = _spawn_hermes_action(args, "import")
    except Exception as exc:
        _log.exception("Failed to spawn import")
        raise HTTPException(status_code=500, detail=f"Failed to run import: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "import"}


def _safe_backup_upload_name(filename: str | None) -> str:
    name = Path(filename or "backup.zip").name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not name:
        name = "backup.zip"
    if not name.lower().endswith(".zip"):
        name = f"{name}.zip"
    return name


@app.post("/api/ops/import-upload")
async def run_import_upload(
    file: UploadFile = File(...),
    force: bool = Form(False),
):
    staging_dir = _dashboard_backup_dir()
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create import staging directory: {exc}",
        )

    safe_name = _safe_backup_upload_name(file.filename)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = staging_dir / f"dashboard-import-{stamp}-{secrets.token_hex(4)}-{safe_name}"
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".upload",
        dir=str(staging_dir),
    )
    tmp_path = Path(tmp_name)
    total = 0
    renamed = False
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANAGED_FILE_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Archive is too large")
                out.write(chunk)
        os.replace(tmp_path, target)
        renamed = True
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Import staging directory is not writable",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not write uploaded archive: {exc}",
        )
    finally:
        if not renamed:
            tmp_path.unlink(missing_ok=True)
        await file.close()

    if not zipfile.is_zipfile(target):
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded archive is not a valid zip file",
        )

    args = ["import", str(target)]
    if force:
        args.append("--force")
    try:
        proc = _spawn_hermes_action(args, "import")
    except Exception as exc:
        _log.exception("Failed to spawn import")
        raise HTTPException(status_code=500, detail=f"Failed to run import: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "import",
        "archive": str(target),
        "uploaded_bytes": total,
    }


@app.get("/api/ops/hooks")
async def list_hooks():
    """List configured shell hooks from config.yaml with consent + health.

    Reports each hook's allowlist (consent) status and whether the script is
    currently executable, plus the set of valid hook events so the create
    form can offer them.
    """
    from hermes_cli.config import load_config as _load_config
    from agent import shell_hooks

    try:
        from hermes_cli.plugins import VALID_HOOKS
        valid_events = sorted(VALID_HOOKS)
    except Exception:
        valid_events = []

    specs = []
    try:
        specs = shell_hooks.iter_configured_hooks(_load_config())
    except Exception:
        _log.exception("iter_configured_hooks failed")

    out = []
    for spec in specs:
        entry = None
        try:
            entry = shell_hooks.allowlist_entry_for(spec.event, spec.command)
        except Exception:
            pass
        executable = False
        try:
            executable = shell_hooks.script_is_executable(spec.command)
        except Exception:
            pass
        out.append({
            "event": spec.event,
            "matcher": spec.matcher,
            "command": spec.command,
            "timeout": spec.timeout,
            "allowed": entry is not None,
            "approved_at": (entry or {}).get("approved_at"),
            "executable": executable,
        })

    return {"hooks": out, "valid_events": valid_events}


class HookCreate(BaseModel):
    event: str
    command: str
    matcher: Optional[str] = None
    timeout: Optional[int] = None
    # approve: write the consent allowlist entry too (the operator using the
    # authenticated dashboard is giving consent). Without it the hook is
    # configured but won't fire until approved.
    approve: bool = True


@app.post("/api/ops/hooks")
async def create_hook(body: HookCreate):
    """Add a shell hook to config.yaml (and optionally approve it).

    Shell hooks run arbitrary commands, so this is a privileged action: it
    writes to the ``hooks:`` config block and, when ``approve`` is set, records
    consent in the allowlist so the hook actually fires.  Takes effect on the
    next session / gateway restart.
    """
    from agent import shell_hooks

    event = (body.event or "").strip()
    command = (body.command or "").strip()
    if not event or not command:
        raise HTTPException(status_code=400, detail="event and command are required")

    try:
        from hermes_cli.plugins import VALID_HOOKS
        if event not in VALID_HOOKS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown event '{event}'. Valid: {', '.join(sorted(VALID_HOOKS))}",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    cfg = load_config()
    hooks_cfg = cfg.get("hooks")
    if not isinstance(hooks_cfg, dict):
        hooks_cfg = {}
        cfg["hooks"] = hooks_cfg
    entries = hooks_cfg.get(event)
    if not isinstance(entries, list):
        entries = []
        hooks_cfg[event] = entries

    new_entry: Dict[str, Any] = {"command": command}
    if body.matcher:
        new_entry["matcher"] = body.matcher
    if body.timeout is not None:
        new_entry["timeout"] = int(body.timeout)
    entries.append(new_entry)
    save_config(cfg)

    approved = False
    if body.approve:
        try:
            shell_hooks._record_approval(event, command)
            approved = True
        except Exception:
            _log.exception("hook consent record failed")

    return {"ok": True, "event": event, "command": command, "approved": approved}


class HookDelete(BaseModel):
    event: str
    command: str


@app.delete("/api/ops/hooks")
async def delete_hook(body: HookDelete):
    """Remove a hook from config.yaml and revoke its consent allowlist entry."""
    from agent import shell_hooks

    event = (body.event or "").strip()
    command = (body.command or "").strip()
    if not event or not command:
        raise HTTPException(status_code=400, detail="event and command are required")

    cfg = load_config()
    hooks_cfg = cfg.get("hooks")
    removed = False
    if isinstance(hooks_cfg, dict) and isinstance(hooks_cfg.get(event), list):
        before = len(hooks_cfg[event])
        hooks_cfg[event] = [
            e for e in hooks_cfg[event]
            if not (isinstance(e, dict) and e.get("command") == command)
        ]
        removed = len(hooks_cfg[event]) < before
        if not hooks_cfg[event]:
            del hooks_cfg[event]
        if not hooks_cfg:
            cfg.pop("hooks", None)
        save_config(cfg)

    # Revoke consent regardless so a re-add re-prompts.
    try:
        shell_hooks.revoke(command)
    except Exception:
        pass

    if not removed:
        raise HTTPException(status_code=404, detail="No matching hook found")
    return {"ok": True}


@app.get("/api/ops/checkpoints")
async def list_checkpoints():
    """List the /rollback shadow store checkpoints (read-only)."""
    # Checkpoints live under <hermes_home>/checkpoints/.  Surface a count +
    # total size so the dashboard can show what a prune would reclaim; the
    # actual prune is a spawned action so confirmation/pruning logic stays
    # in one place (the CLI).
    cp_dir = get_hermes_home() / "checkpoints"
    sessions = []
    total_bytes = 0
    if cp_dir.is_dir():
        for child in sorted(cp_dir.iterdir()):
            if not child.is_dir():
                continue
            size = 0
            count = 0
            for f in child.rglob("*"):
                if f.is_file():
                    try:
                        size += f.stat().st_size
                        count += 1
                    except OSError:
                        pass
            total_bytes += size
            sessions.append({
                "session": child.name,
                "files": count,
                "bytes": size,
            })
    return {"sessions": sessions, "total_bytes": total_bytes}


@app.post("/api/ops/checkpoints/prune")
async def prune_checkpoints():
    try:
        proc = _spawn_hermes_action(["checkpoints", "prune"], "checkpoints-prune")
    except Exception as exc:
        _log.exception("Failed to spawn checkpoints prune")
        raise HTTPException(status_code=500, detail=f"Failed to prune checkpoints: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "checkpoints-prune"}


# ---------------------------------------------------------------------------
# Skills hub endpoints — search / install / uninstall / update.
#
# Search and install touch the network (GitHub, hub sources) and run the same
# complex source-router pipeline the CLI uses, so they're spawned as background
# actions whose logs the dashboard tails.  The already-installed skill list +
# enable/disable toggle live in the existing /api/skills endpoints.
# ---------------------------------------------------------------------------


class SkillInstallRequest(BaseModel):
    identifier: str
    profile: Optional[str] = None


def _profile_cli_args(profile: Optional[str]) -> List[str]:
    """Return ``["-p", <name>]`` for a validated non-default profile.

    Hub install/uninstall/update run in a fresh ``hermes`` subprocess, and
    ``_apply_profile_override()`` reads ``-p`` from argv in the child — the
    only mechanism that reaches import-time-bound globals like
    ``skills_hub.SKILLS_DIR``. Empty/"current" means the dashboard's own
    profile (no args, legacy behavior).
    """
    requested = (profile or "").strip()
    if not requested or requested.lower() in {"current", "default"}:
        return []
    from hermes_cli import profiles as profiles_mod
    _resolve_profile_dir(requested)
    return ["-p", profiles_mod.normalize_profile_name(requested)]


def _hub_action_name(verb: str, key: str) -> str:
    """Unique per-skill hub action name (+ registered log file).

    ``_spawn_hermes_action`` tracks one process/log per name, so a shared
    "skills-install"/"skills-uninstall" would make concurrent row-level actions
    overwrite each other's status/log while the UI polls per identifier. Slug
    (readable) + hash (collision-proof) keys each action to its own row.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")[:48] or "skill"
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    name = f"skills-{verb}-{slug}-{digest}"
    _ACTION_LOG_FILES.setdefault(name, f"action-{name}.log")
    return name


@app.post("/api/skills/hub/install")
async def install_skill_hub(body: SkillInstallRequest, profile: Optional[str] = None):
    identifier = (body.identifier or "").strip()
    if not identifier:
        raise HTTPException(status_code=400, detail="identifier is required")
    name = _hub_action_name("install", identifier)
    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(body.profile or profile)
            + ["skills", "install", identifier, "--yes"],
            name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn skills install")
        raise HTTPException(status_code=500, detail=f"Failed to install skill: {exc}")
    return {"ok": True, "pid": proc.pid, "name": name}


class SkillUninstallRequest(BaseModel):
    name: str
    profile: Optional[str] = None


@app.post("/api/skills/hub/uninstall")
async def uninstall_skill_hub(body: SkillUninstallRequest, profile: Optional[str] = None):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    action = _hub_action_name("uninstall", name)
    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(body.profile or profile) + ["skills", "uninstall", name, "--yes"],
            action,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn skills uninstall")
        raise HTTPException(status_code=500, detail=f"Failed to uninstall skill: {exc}")
    return {"ok": True, "pid": proc.pid, "name": action}


class SkillsUpdateRequest(BaseModel):
    profile: Optional[str] = None


@app.post("/api/skills/hub/update")
async def update_skills_hub(
    body: Optional[SkillsUpdateRequest] = None, profile: Optional[str] = None
):
    try:
        effective = (body.profile if body else None) or profile
        proc = _spawn_hermes_action(
            _profile_cli_args(effective) + ["skills", "update"], "skills-update"
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn skills update")
        raise HTTPException(status_code=500, detail=f"Failed to update skills: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "skills-update"}


# Human-readable labels for each hub source id (matches `hermes skills search`
# provenance).  Keep in sync with create_source_router()'s source list.
_SKILL_HUB_SOURCE_LABELS = {
    "official": "Official (Nous)",
    "hermes-index": "Hermes Index",
    "skills-sh": "skills.sh",
    "well-known": "Well-Known",
    "url": "Direct URL",
    "github": "GitHub",
    "clawhub": "ClawHub",
    "claude-marketplace": "Claude Marketplace",
    "lobehub": "LobeHub",
    "browse-sh": "browse.sh",
}


def _skill_meta_to_payload(m) -> dict:
    return {
        "name": m.name,
        "description": m.description,
        "source": m.source,
        "identifier": m.identifier,
        "trust_level": m.trust_level,
        "repo": m.repo,
        "tags": list(m.tags or []),
    }


def _installed_hub_identifiers(profile: Optional[str] = None) -> dict:
    """Map identifier -> installed lock entry for hub-installed skills.

    Lets the UI mark search results that are already installed.  Scoped to
    ``profile``'s skills/.hub/lock.json when provided (HubLockFile takes an
    explicit path, sidestepping the import-time LOCK_FILE binding).
    Best-effort: returns an empty dict if the lock file can't be read.
    """
    try:
        from tools.skills_hub import HubLockFile

        requested = (profile or "").strip()
        if requested and requested.lower() != "current":
            profile_dir = _resolve_profile_dir(requested)
            lock = HubLockFile(profile_dir / "skills" / ".hub" / "lock.json")
        else:
            lock = HubLockFile()
        out = {}
        for entry in lock.list_installed():
            ident = entry.get("identifier")
            if ident:
                out[ident] = {
                    "name": entry.get("name"),
                    "trust_level": entry.get("trust_level"),
                    "scan_verdict": entry.get("scan_verdict"),
                }
        return out
    except Exception:
        return {}


@app.get("/api/skills/hub/sources")
async def list_skills_hub_sources(profile: Optional[str] = None):
    """List the configured skill-hub sources and installed-skill provenance.

    Gives the dashboard something to show BEFORE a search runs — which hubs
    are wired up, their trust tier, and a set of featured skills pulled from
    the centralized index (zero extra API calls).  Without this the Browse-hub
    tab is a blank page with no indication it's even connected to anything.
    ``profile`` scopes the installed-skill provenance to that profile.
    """

    def _run():
        from tools.skills_hub import create_source_router

        with _config_profile_scope(profile):
            sources = create_source_router()
        out = []
        index_available = False
        featured = []
        for src in sources:
            sid = src.source_id()
            entry = {
                "id": sid,
                "label": _SKILL_HUB_SOURCE_LABELS.get(sid, sid),
            }
            # GitHub exposes a rate-limit flag; the index an availability flag.
            if sid == "github":
                try:
                    entry["rate_limited"] = bool(getattr(src, "is_rate_limited", False))
                except Exception:
                    entry["rate_limited"] = False
            if sid == "hermes-index":
                try:
                    index_available = bool(getattr(src, "is_available", False))
                except Exception:
                    index_available = False
                entry["available"] = index_available
                # Empty-query search on the index returns featured/popular skills.
                if index_available:
                    try:
                        featured = [
                            _skill_meta_to_payload(m) for m in src.search("", limit=12)
                        ]
                    except Exception:
                        featured = []
            out.append(entry)
        # Tell the UI which sources are worth searching individually (for its
        # progressive per-source fan-out). Mirror parallel_search_sources: when
        # the centralized index is available it already subsumes the external
        # API sources, so they're redundant — skipping them avoids ~70 GitHub
        # calls per keystroke. Keep this set in sync with that function's
        # ``_api_source_ids``.
        _api_source_ids = frozenset(
            {"github", "skills-sh", "clawhub", "claude-marketplace", "lobehub", "well-known"}
        )
        for entry in out:
            entry["searchable"] = not (index_available and entry["id"] in _api_source_ids)
        return {
            "sources": out,
            "index_available": index_available,
            "featured": featured,
            "installed": _installed_hub_identifiers(profile),
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("skills hub sources listing failed")
        raise HTTPException(status_code=502, detail=f"Hub sources failed: {exc}")


@app.get("/api/skills/hub/search")
async def search_skills_hub(
    q: str = "", source: str = "all", limit: int = 20, profile: Optional[str] = None
):
    """Search the skill hub across all configured sources.

    Network-bound (parallel source search); runs in a thread so the FastAPI
    loop isn't blocked.  Returns structured results the UI installs by
    identifier via POST /api/skills/hub/install, previews via
    /api/skills/hub/preview, and scans via /api/skills/hub/scan.
    """
    query = (q or "").strip()
    if not query:
        return {"results": [], "source_counts": {}, "timed_out": [], "installed": {}}

    def _run():
        from tools.skills_hub import create_source_router, parallel_search_sources

        with _config_profile_scope(profile):
            sources = create_source_router()
        capped = min(max(limit, 1), 50)
        all_results, source_counts, timed_out = parallel_search_sources(
            sources, query=query, source_filter=source or "all", overall_timeout=30
        )

        # Dedupe by identifier, preferring higher trust (mirrors unified_search).
        _rank = {"builtin": 2, "trusted": 1, "community": 0}
        seen = {}
        for r in all_results:
            if r.identifier not in seen:
                seen[r.identifier] = r
            elif _rank.get(r.trust_level, 0) > _rank.get(seen[r.identifier].trust_level, 0):
                seen[r.identifier] = r
        deduped = list(seen.values())[:capped]

        return {
            "results": [_skill_meta_to_payload(m) for m in deduped],
            "source_counts": source_counts,
            "timed_out": timed_out,
            "installed": _installed_hub_identifiers(profile),
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("skills hub search failed")
        raise HTTPException(status_code=502, detail=f"Hub search failed: {exc}")


@app.get("/api/skills/hub/preview")
async def preview_skill_hub(identifier: str = "", profile: Optional[str] = None):
    """Fetch a hub skill's SKILL.md content + metadata for in-dashboard reading.

    Resolves the identifier across configured sources (same path the CLI
    installer uses), then returns the rendered SKILL.md text and the file
    manifest WITHOUT installing anything.  This is the 'read the actual skill
    before installing' affordance the Browse-hub tab was missing.

    Scoped to ``profile`` so a non-default profile with different hub taps
    resolves against ITS source router, not the default profile's.
    """
    ident = (identifier or "").strip()
    if not ident:
        raise HTTPException(status_code=400, detail="identifier is required")

    def _run():
        from hermes_cli.skills_hub import _resolve_source_meta_and_bundle
        from tools.skills_hub import create_source_router

        with _config_profile_scope(profile):
            sources = create_source_router()
            meta, bundle, _src = _resolve_source_meta_and_bundle(ident, sources)
        if not bundle and not meta:
            return None

        files = {}
        skill_md = ""
        if bundle:
            for rel, content in (bundle.files or {}).items():
                if isinstance(content, bytes):
                    # Some sources (e.g. official optional skills) store every
                    # file as bytes.  Decode text so SKILL.md / docs render;
                    # only fall back to a placeholder for genuinely-binary data.
                    try:
                        files[rel] = content.decode("utf-8")
                    except UnicodeDecodeError:
                        files[rel] = "(binary file)"
                else:
                    files[rel] = content
            skill_md = files.get("SKILL.md", "") or ""

        m = meta or bundle
        return {
            "name": getattr(m, "name", ident),
            "description": getattr(m, "description", "") or "",
            "source": getattr(m, "source", "") or "",
            "identifier": getattr(m, "identifier", ident) or ident,
            "trust_level": getattr(m, "trust_level", "community") or "community",
            "repo": getattr(m, "repo", None),
            "tags": list(getattr(m, "tags", None) or []),
            "skill_md": skill_md,
            "files": sorted(files.keys()),
        }

    try:
        result = await asyncio.to_thread(_run)
    except Exception as exc:
        _log.exception("skills hub preview failed")
        raise HTTPException(status_code=502, detail=f"Hub preview failed: {exc}")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {ident}")
    return result


@app.get("/api/skills/hub/scan")
async def scan_skill_hub(identifier: str = "", profile: Optional[str] = None):
    """Run the install-time security scan on a hub skill WITHOUT installing it.

    Fetches the bundle, quarantines it, and runs the same `scan_skill` /
    `should_allow_install` pipeline the CLI installer uses — then cleans up the
    quarantine.  Returns the verdict, per-finding detail, trust tier, and the
    install-policy decision so the dashboard can show a visual safety result
    on demand (the 'scan' button the Browse-hub tab was missing).

    Scoped to ``profile`` so the bundle resolves against that profile's hub
    source router, matching where an install would pull it from.
    """
    ident = (identifier or "").strip()
    if not ident:
        raise HTTPException(status_code=400, detail="identifier is required")

    def _run():
        import shutil as _shutil

        from hermes_cli.skills_hub import _resolve_source_meta_and_bundle
        from tools.skills_hub import create_source_router, quarantine_bundle
        from tools.skills_guard import scan_skill, should_allow_install

        with _config_profile_scope(profile):
            sources = create_source_router()
            meta, bundle, _src = _resolve_source_meta_and_bundle(ident, sources)
        if not bundle:
            return None

        if bundle.source == "official":
            scan_source = "official"
        else:
            scan_source = (
                getattr(bundle, "identifier", "")
                or getattr(meta, "identifier", "")
                or ident
            )

        q_path = None
        try:
            q_path = quarantine_bundle(bundle)
            result = scan_skill(q_path, source=scan_source)
        finally:
            if q_path is not None:
                _shutil.rmtree(q_path, ignore_errors=True)

        allowed, reason = should_allow_install(result, force=False)
        # `allowed` may be None ("ask") for agent-created/dangerous gates.
        if allowed is True:
            policy = "allow"
        elif allowed is None:
            policy = "ask"
        else:
            policy = "block"

        findings = [
            {
                "severity": f.severity,
                "category": f.category,
                "file": f.file,
                "line": f.line,
                "description": f.description,
            }
            for f in result.findings
        ]
        # Per-severity tally for an at-a-glance summary.
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in result.findings:
            if f.severity in counts:
                counts[f.severity] += 1

        return {
            "name": result.skill_name,
            "identifier": ident,
            "source": result.source,
            "trust_level": result.trust_level,
            "verdict": result.verdict,
            "summary": result.summary,
            "policy": policy,
            "policy_reason": reason,
            "findings": findings,
            "severity_counts": counts,
        }

    try:
        result = await asyncio.to_thread(_run)
    except Exception as exc:
        _log.exception("skills hub scan failed")
        raise HTTPException(status_code=502, detail=f"Hub scan failed: {exc}")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {ident}")
    return result


# ---------------------------------------------------------------------------
# Profile management endpoints (minimal — list/create/rename/delete + SOUL.md)
# ---------------------------------------------------------------------------


class ProfileCreate(BaseModel):
    name: str
    clone_from: Optional[str] = None
    # Backward compatibility for older dashboard/desktop clients. New clients
    # send clone_from="default" (or another profile name) explicitly.
    clone_from_default: bool = False
    clone_all: bool = False
    no_skills: bool = False
    description: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    # Profile-builder additions — all optional, all applied best-effort AFTER
    # the profile directory exists, so a hiccup in any of them never 500s the
    # create (the user can fix it from the relevant dashboard page afterward).
    # MCP servers to write into the new profile's config.yaml.
    mcp_servers: List["MCPServerCreate"] = []
    # Built-in / optional skills to KEEP active. When this list is non-empty,
    # the builder uses "replace" semantics: the bundle is seeded, then every
    # seeded skill NOT in this list is added to the profile's disabled list.
    # Empty list = leave the seeded bundle untouched (legacy behaviour).
    keep_skills: List[str] = []
    # Skills-hub identifiers to install into the new profile. Installed async
    # via a subprocess scoped to the profile (`hermes -p <name> skills install`)
    # because skills_hub.SKILLS_DIR is import-time-bound and the HERMES_HOME
    # override can't redirect it. Returns spawned PIDs for the UI to poll.
    hub_skills: List[str] = []


class ProfileRename(BaseModel):
    new_name: str


class ProfileSoulUpdate(BaseModel):
    content: str


class ProfileActiveUpdate(BaseModel):
    name: str


class ProfileDescriptionUpdate(BaseModel):
    description: str = ""


class ProfileModelUpdate(BaseModel):
    provider: str
    model: str


class ProfileDescribeAuto(BaseModel):
    overwrite: bool = False


def _profile_attr(info, name: str, default: Any = None) -> Any:
    try:
        return getattr(info, name)
    except Exception:
        return default


def _profile_to_dict(info) -> Dict[str, Any]:
    return {
        "name": _profile_attr(info, "name", ""),
        "path": str(_profile_attr(info, "path", "")),
        "is_default": bool(_profile_attr(info, "is_default", False)),
        "model": _profile_attr(info, "model"),
        "provider": _profile_attr(info, "provider"),
        "has_env": bool(_profile_attr(info, "has_env", False)),
        "skill_count": int(_profile_attr(info, "skill_count", 0) or 0),
        "gateway_running": bool(_profile_attr(info, "gateway_running", False)),
        "description": _profile_attr(info, "description", "") or "",
        "description_auto": bool(_profile_attr(info, "description_auto", False)),
        "distribution_name": _profile_attr(info, "distribution_name"),
        "distribution_version": _profile_attr(info, "distribution_version"),
        "distribution_source": _profile_attr(info, "distribution_source"),
        "has_alias": _profile_attr(info, "alias_path") is not None,
    }


def _fallback_profile_dicts(profiles_mod) -> List[Dict[str, Any]]:
    def _safe(callable_, default):
        try:
            return callable_()
        except Exception:
            return default

    profiles: List[Dict[str, Any]] = []
    default_home = profiles_mod._get_default_hermes_home()
    if default_home.is_dir():
        model, provider = _safe(lambda: profiles_mod._read_config_model(default_home), (None, None))
        profiles.append({
            "name": "default",
            "path": str(default_home),
            "is_default": True,
            "model": model,
            "provider": provider,
            "has_env": (default_home / ".env").exists(),
            "skill_count": _safe(lambda: profiles_mod._count_skills(default_home), 0),
            "gateway_running": _safe(lambda: profiles_mod._check_gateway_running(default_home), False),
            "description": _safe(lambda: profiles_mod.read_profile_meta(default_home).get("description", ""), ""),
            "description_auto": _safe(lambda: profiles_mod.read_profile_meta(default_home).get("description_auto", False), False),
            "distribution_name": None,
            "distribution_version": None,
            "distribution_source": None,
            "has_alias": False,
        })

    profiles_root = profiles_mod._get_profiles_root()
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if not entry.is_dir() or not profiles_mod._PROFILE_ID_RE.match(entry.name):
                continue
            model, provider = _safe(lambda entry=entry: profiles_mod._read_config_model(entry), (None, None))
            profiles.append({
                "name": entry.name,
                "path": str(entry),
                "is_default": False,
                "model": model,
                "provider": provider,
                "has_env": (entry / ".env").exists(),
                "skill_count": _safe(lambda entry=entry: profiles_mod._count_skills(entry), 0),
                "gateway_running": _safe(lambda entry=entry: profiles_mod._check_gateway_running(entry), False),
                "description": _safe(lambda entry=entry: profiles_mod.read_profile_meta(entry).get("description", ""), ""),
                "description_auto": _safe(lambda entry=entry: profiles_mod.read_profile_meta(entry).get("description_auto", False), False),
                "distribution_name": None,
                "distribution_version": None,
                "distribution_source": None,
                "has_alias": False,
            })

    return profiles


def _resolve_profile_dir(name: str) -> Path:
    """Validate ``name`` and resolve to its directory or raise an HTTPException."""
    from hermes_cli import profiles as profiles_mod
    try:
        profiles_mod.validate_profile_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(name):
        raise HTTPException(status_code=404, detail=f"Profile '{name}' does not exist.")
    return profiles_mod.get_profile_dir(name)


def _profile_setup_command(name: str) -> str:
    """Return the shell command used to configure a profile in the CLI."""
    _resolve_profile_dir(name)
    return "hermes setup" if name == "default" else f"{name} setup"


def _write_profile_model(profile_dir: Path, provider: str, model: str) -> None:
    """Write the main model assignment into a specific profile's config.yaml.

    Scopes ``load_config``/``save_config`` to ``profile_dir`` via the
    context-local HERMES_HOME override so the write lands in the target
    profile's config rather than the dashboard process's active profile.
    Clears any stale ``base_url`` / ``context_length`` the same way
    ``POST /api/model/set`` does, since the new model may differ.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(str(profile_dir))
    try:
        provider, model = _normalize_main_model_assignment(provider, model)
        cfg = load_config()
        cfg["model"] = _apply_main_model_assignment(cfg.get("model", {}), provider, model)
        save_config(cfg)
    finally:
        reset_hermes_home_override(token)


def _write_profile_mcp_servers(profile_dir: Path, servers: List["MCPServerCreate"]) -> int:
    """Write MCP server entries into a specific profile's config.yaml.

    Scopes ``load_config``/``save_config`` to ``profile_dir`` via the
    context-local HERMES_HOME override (same mechanism as
    ``_write_profile_model``) so the entries land in the target profile's
    config rather than the dashboard process's active profile.

    Mirrors the per-server shape the ``POST /api/mcp/servers`` endpoint builds,
    but batched so the whole profile-create write is a single config save.
    Returns the number of servers written.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.mcp_security import validate_mcp_server_entry

    written = 0
    token = set_hermes_home_override(str(profile_dir))
    try:
        cfg = load_config()
        mcp = cfg.setdefault("mcp_servers", {})
        for server in servers:
            name = (server.name or "").strip()
            if not name:
                continue
            entry: Dict[str, Any] = {}
            if server.url:
                entry["url"] = server.url
            if server.command:
                entry["command"] = server.command
            if server.args:
                entry["args"] = list(server.args)
            if server.env:
                entry["env"] = dict(server.env)
            if server.auth:
                entry["auth"] = server.auth
            if not entry:
                # Nothing usable to write (neither url nor command) — skip
                # rather than persist an empty, unusable server stanza.
                continue
            issues = validate_mcp_server_entry(name, entry)
            if issues:
                _log.warning("Profile-create: skipping MCP server '%s': %s", name, "; ".join(issues))
                continue
            mcp[name] = entry
            written += 1
        if written:
            save_config(cfg)
        elif not mcp:
            # We created an empty mcp_servers dict but wrote nothing — don't
            # leave a stray empty key in the new profile's config.
            cfg.pop("mcp_servers", None)
            save_config(cfg)
    finally:
        reset_hermes_home_override(token)
    return written


def _disable_unselected_skills(profile_dir: Path, keep: List[str]) -> int:
    """Disable every installed skill in ``profile_dir`` not in ``keep``.

    Profiles manage skill activation via a *disabled* list — all installed
    skills are active by default and users opt out. The builder's skill step
    uses "replace" semantics: the user picks exactly which seeded built-in /
    optional skills stay active, and everything else gets added to the disabled
    list. (Hub skills are installed separately via subprocess and are active on
    install.) Scoped to the profile via the HERMES_HOME override. Returns the
    number of skills newly disabled.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills

    keep_set = {s.strip() for s in keep if s and s.strip()}
    disabled_count = 0
    token = set_hermes_home_override(str(profile_dir))
    try:
        installed: List[str] = []
        skills_root = profile_dir / "skills"
        if skills_root.is_dir():
            for md in skills_root.rglob("SKILL.md"):
                installed.append(md.parent.name)
        cfg = load_config()
        disabled = get_disabled_skills(cfg)
        for name in installed:
            if name not in keep_set and name not in disabled:
                disabled.add(name)
                disabled_count += 1
        if disabled_count:
            save_disabled_skills(cfg, disabled)
    finally:
        reset_hermes_home_override(token)
    return disabled_count


@app.get("/api/profiles")
async def list_profiles_endpoint():
    from hermes_cli import profiles as profiles_mod
    try:
        loop = asyncio.get_running_loop()
        profiles = await loop.run_in_executor(None, profiles_mod.list_profiles)
        return {"profiles": [_profile_to_dict(p) for p in profiles]}
    except Exception:
        _log.exception("GET /api/profiles failed; falling back to profile directory scan")
        return {"profiles": _fallback_profile_dicts(profiles_mod)}


@app.post("/api/profiles")
async def create_profile_endpoint(body: ProfileCreate):
    from hermes_cli import profiles as profiles_mod
    explicit_source = (body.clone_from or "").strip()
    if explicit_source:
        # Duplicating a specific profile: clone its config/skills/SOUL (or full
        # state when clone_all) from the named source rather than "default".
        clone = True
        clone_from = explicit_source
        clone_config = not body.clone_all
    elif body.clone_all:
        # Preserve the dashboard's historical clone-all behavior: a full-copy
        # request with no explicit dropdown source copies from default.
        clone = True
        clone_from = "default"
        clone_config = False
    else:
        clone = body.clone_from_default
        clone_from = "default" if clone else None
        clone_config = clone
    try:
        path = profiles_mod.create_profile(
            name=body.name,
            clone_from=clone_from,
            clone_all=body.clone_all,
            clone_config=clone_config,
            no_skills=body.no_skills,
            description=body.description,
        )
        # Match the CLI's profile-create flow: fresh named profiles get the
        # bundled skills installed. When cloning from default, create_profile()
        # has already copied the source profile's skills, including any
        # user-installed skills. When no_skills=True, create_profile() wrote
        # the opt-out marker and seed_profile_skills() will no-op.
        if not clone:
            profiles_mod.seed_profile_skills(path, quiet=True)

        # Match the CLI's profile-create flow: named profiles should get a
        # wrapper in ~/.local/bin when the alias is safe to create.
        collision = profiles_mod.check_alias_collision(body.name)
        if not collision:
            profiles_mod.create_wrapper_script(body.name)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("POST /api/profiles failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Optional explicit model assignment for the new profile. Best-effort:
    # the profile already exists, so a model-write hiccup must not 500 the
    # whole create — the user can set the model later from the Models page
    # or `<profile> setup`.
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    model_set = False
    if provider and model:
        try:
            _write_profile_model(path, provider, model)
            model_set = True
        except Exception:
            _log.exception("Setting model for new profile %s failed", body.name)

    # Optional MCP servers. Best-effort, same rationale as model assignment.
    mcp_written = 0
    if body.mcp_servers:
        try:
            mcp_written = _write_profile_mcp_servers(path, body.mcp_servers)
        except Exception:
            _log.exception("Writing MCP servers for new profile %s failed", body.name)

    # Optional "keep" skill selection — replace semantics. When the builder
    # sends an explicit keep list, disable every seeded skill not in it.
    # Best-effort. Skipped when keep_skills is empty (legacy: keep the bundle).
    skills_disabled = 0
    if body.keep_skills:
        try:
            skills_disabled = _disable_unselected_skills(path, body.keep_skills)
        except Exception:
            _log.exception("Applying skill selection for new profile %s failed", body.name)

    # Optional skills-hub installs. Spawned async, scoped to the new profile
    # via `-p <name>` (a fresh subprocess re-binds skills_hub.SKILLS_DIR to the
    # profile's HERMES_HOME at import). Returns PIDs for the UI to poll.
    hub_installs: List[Dict[str, Any]] = []
    for identifier in body.hub_skills:
        ident = (identifier or "").strip()
        if not ident:
            continue
        try:
            proc = _spawn_hermes_action(
                ["-p", body.name, "skills", "install", ident, "--yes"],
                _hub_action_name("install", ident),
            )
            hub_installs.append({"identifier": ident, "pid": proc.pid})
        except Exception:
            _log.exception(
                "Spawning hub-skill install %s for new profile %s failed",
                ident,
                body.name,
            )
            hub_installs.append({"identifier": ident, "pid": None})

    return {
        "ok": True,
        "name": body.name,
        "path": str(path),
        "model_set": model_set,
        "mcp_written": mcp_written,
        "skills_disabled": skills_disabled,
        "hub_installs": hub_installs,
    }


@app.get("/api/profiles/active")
async def get_active_profile_endpoint():
    """Return the sticky active profile and the profile this dashboard
    process is currently running as.

    ``active`` is the sticky default written by ``hermes profile use`` —
    the profile new CLI invocations pick up. ``current`` is the profile
    the running dashboard/gateway is scoped to (derived from HERMES_HOME).
    """
    from hermes_cli import profiles as profiles_mod
    try:
        active = profiles_mod.get_active_profile() or "default"
    except Exception:
        active = "default"
    try:
        current = profiles_mod.get_active_profile_name() or "default"
    except Exception:
        current = "default"
    return {"active": active, "current": current}


@app.post("/api/profiles/active")
async def set_active_profile_endpoint(body: ProfileActiveUpdate):
    """Set the sticky active profile (mirrors ``hermes profile use``).

    Note: this does not retarget the already-running dashboard process —
    it changes which profile subsequent CLI commands and gateways use.
    """
    from hermes_cli import profiles as profiles_mod
    try:
        profiles_mod.set_active_profile(body.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("POST /api/profiles/active failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "active": profiles_mod.normalize_profile_name(body.name)}


@app.get("/api/profiles/{name}/setup-command")
async def get_profile_setup_command(name: str):
    return {"command": _profile_setup_command(name)}


@app.post("/api/profiles/{name}/open-terminal")
async def open_profile_terminal_endpoint(name: str):
    try:
        command = _profile_setup_command(name)

        if sys.platform.startswith("win"):
            subprocess.Popen(["cmd.exe", "/c", "start", "", command])
        elif sys.platform == "darwin":
            escaped = command.replace("\\", "\\\\").replace('"', '\\"')
            applescript = (
                'tell application "Terminal"\n'
                "activate\n"
                f'do script "{escaped}"\n'
                "end tell"
            )
            subprocess.Popen(["osascript", "-e", applescript])
        else:
            terminal_commands = [
                ("x-terminal-emulator", ["x-terminal-emulator", "-e", "sh", "-lc", command]),
                ("gnome-terminal", ["gnome-terminal", "--", "sh", "-lc", command]),
                ("konsole", ["konsole", "-e", "sh", "-lc", command]),
                ("xfce4-terminal", ["xfce4-terminal", "-e", f"sh -lc '{command}'"]),
                ("mate-terminal", ["mate-terminal", "-e", f"sh -lc '{command}'"]),
                ("lxterminal", ["lxterminal", "-e", f"sh -lc '{command}'"]),
                ("tilix", ["tilix", "-e", "sh", "-lc", command]),
                ("alacritty", ["alacritty", "-e", "sh", "-lc", command]),
                ("kitty", ["kitty", "sh", "-lc", command]),
                ("xterm", ["xterm", "-e", "sh", "-lc", command]),
            ]
            for executable, popen_args in terminal_commands:
                if subprocess.call(
                    ["which", executable],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ) == 0:
                    subprocess.Popen(popen_args)
                    break
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No supported terminal emulator found",
                )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("POST /api/profiles/%s/open-terminal failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "command": command}


@app.patch("/api/profiles/{name}")
async def rename_profile_endpoint(name: str, body: ProfileRename):
    from hermes_cli import profiles as profiles_mod
    try:
        path = profiles_mod.rename_profile(name, body.new_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, FileExistsError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("PATCH /api/profiles/%s failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "name": body.new_name, "path": str(path)}


@app.delete("/api/profiles/{name}")
async def delete_profile_endpoint(name: str):
    """Delete a profile. The dashboard collects the user's confirmation in
    its own dialog before this request, so we always pass ``yes=True`` to
    skip the CLI's interactive prompt."""
    from hermes_cli import profiles as profiles_mod
    try:
        path = profiles_mod.delete_profile(name, yes=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("DELETE /api/profiles/%s failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "path": str(path)}


@app.get("/api/profiles/{name}/soul")
async def get_profile_soul(name: str):
    soul_path = _resolve_profile_dir(name) / "SOUL.md"
    if soul_path.exists():
        try:
            return {"content": soul_path.read_text(encoding="utf-8"), "exists": True}
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Could not read SOUL.md: {e}")
    return {"content": "", "exists": False}


@app.put("/api/profiles/{name}/soul")
async def update_profile_soul(name: str, body: ProfileSoulUpdate):
    soul_path = _resolve_profile_dir(name) / "SOUL.md"
    try:
        soul_path.write_text(body.content, encoding="utf-8")
    except OSError as e:
        _log.exception("PUT /api/profiles/%s/soul failed", name)
        raise HTTPException(status_code=500, detail=f"Could not write SOUL.md: {e}")
    return {"ok": True}


@app.put("/api/profiles/{name}/description")
async def update_profile_description_endpoint(name: str, body: ProfileDescriptionUpdate):
    """Set or clear a profile's role description (kanban routing signal).

    Empty string clears the description. Non-empty stores it as a
    user-authored description (``description_auto: false``) so the
    auto-describer won't overwrite it on a sweep.
    """
    from hermes_cli import profiles as profiles_mod
    profile_dir = _resolve_profile_dir(name)
    text = (body.description or "").strip()
    try:
        profiles_mod.write_profile_meta(
            profile_dir,
            description=text,
            description_auto=False,
        )
    except Exception as e:
        _log.exception("PUT /api/profiles/%s/description failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "description": text, "description_auto": False}


@app.put("/api/profiles/{name}/model")
async def update_profile_model_endpoint(name: str, body: ProfileModelUpdate):
    """Set the main model (``model.default`` + ``model.provider``) for a
    specific profile's config.yaml, without touching the dashboard's own
    active profile. Mirrors ``POST /api/model/set`` (main scope) but scoped
    to the named profile via the HERMES_HOME override.
    """
    profile_dir = _resolve_profile_dir(name)
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    if not provider or not model:
        raise HTTPException(status_code=400, detail="provider and model are required")
    try:
        _write_profile_model(profile_dir, provider, model)
    except Exception as e:
        _log.exception("PUT /api/profiles/%s/model failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "provider": provider, "model": model}


@app.post("/api/profiles/{name}/describe-auto")
async def describe_profile_auto_endpoint(name: str, body: ProfileDescribeAuto):
    """Auto-generate a profile's description via the auxiliary LLM
    (``auxiliary.profile_describer``). Mirrors ``hermes profile describe
    <name> --auto``.

    A failed generation (no aux client, LLM error, …) is returned as
    ``ok: false`` with a reason rather than an HTTP error so the UI can
    surface it inline and let the operator fix config and retry.
    """
    _resolve_profile_dir(name)
    try:
        from hermes_cli import profile_describer
        outcome = profile_describer.describe_profile(name, overwrite=bool(body.overwrite))
    except Exception as e:
        _log.exception("POST /api/profiles/%s/describe-auto failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ok": bool(outcome.ok),
        "reason": outcome.reason,
        "description": outcome.description,
        # Only a successful generation is an auto-authored description. A failed
        # sweep leaves any existing description untouched, so don't claim it's
        # auto-generated.
        "description_auto": bool(outcome.ok),
    }


# ---------------------------------------------------------------------------
# Skills & Tools endpoints
#
# Every read/write below accepts an optional ``profile`` query param so the
# dashboard can manage ANY profile's skills/toolsets, not just the profile
# the dashboard process happens to be running under. Without this, "Set as
# active" on the Profiles page (which only flips the sticky ``active_profile``
# file for FUTURE CLI/gateway invocations) misled users into thinking skill
# toggles would land in the activated profile — they silently wrote into the
# dashboard's own config instead. See _profile_scope() for the mechanism.
# ---------------------------------------------------------------------------


_SKILLS_PROFILE_LOCK = threading.RLock()


@contextmanager
def _profile_scope(profile: Optional[str]):
    """Scope config + skill-directory resolution to ``profile`` for one request.

    Two seams must be redirected for skills/toolsets endpoints:

    1. ``load_config``/``save_config`` resolve ``get_hermes_home()`` at call
       time — the context-local override from ``set_hermes_home_override``
       reaches them (same pattern as ``_write_profile_model``).
    2. ``tools.skills_tool`` and ``tools.skill_manager_tool`` bind
       ``SKILLS_DIR`` at import time, so the override CANNOT reach them.
       Like ``_call_cron_for_profile`` does for cron's module globals,
       temporarily retarget both under a lock and restore them
       immediately after.

    ``profile`` of None/""/"current" means "the dashboard's own profile" —
    config resolution is untouched, but the skill-module globals are still
    retargeted to the *current* ``get_hermes_home()`` so writes land in the
    live home even when the import-time binding is stale (e.g. the process
    imported the modules before a HERMES_HOME override, or under test
    isolation).
    """
    requested = (profile or "").strip()

    from hermes_constants import (
        get_hermes_home,
        set_hermes_home_override,
        reset_hermes_home_override,
    )
    from tools import skills_tool as _skills_tool
    from tools import skill_manager_tool as _skill_mgr

    token = None
    if not requested or requested.lower() == "current":
        profile_dir = get_hermes_home()
    else:
        profile_dir = _resolve_profile_dir(requested)
        token = set_hermes_home_override(str(profile_dir))

    with _SKILLS_PROFILE_LOCK:
        old_home = _skills_tool.HERMES_HOME
        old_skills_dir = _skills_tool.SKILLS_DIR
        old_mgr_home = _skill_mgr.HERMES_HOME
        old_mgr_skills_dir = _skill_mgr.SKILLS_DIR
        _skills_tool.HERMES_HOME = profile_dir
        _skills_tool.SKILLS_DIR = profile_dir / "skills"
        _skill_mgr.HERMES_HOME = profile_dir
        _skill_mgr.SKILLS_DIR = profile_dir / "skills"
        try:
            yield profile_dir if token is not None else None
        finally:
            _skills_tool.HERMES_HOME = old_home
            _skills_tool.SKILLS_DIR = old_skills_dir
            _skill_mgr.HERMES_HOME = old_mgr_home
            _skill_mgr.SKILLS_DIR = old_mgr_skills_dir
            if token is not None:
                reset_hermes_home_override(token)


@contextmanager
def _config_profile_scope(profile: Optional[str]):
    """Await-safe, config-only profile scope for handlers that ``await``.

    Unlike ``_profile_scope`` this touches ONLY the context-local
    ``set_hermes_home_override`` contextvar — it does NOT swap the
    process-global ``skills_tool``/``skill_manager`` module attributes.
    Those globals are shared across all event-loop tasks, so holding them
    across an ``await`` lets a concurrent skills request restore THIS
    request's profile dir on its ``finally`` (cross-contamination). The
    contextvar override is task-local and survives an ``await`` cleanly,
    which is all endpoints that resolve ``get_hermes_home()`` at call time
    (config, env, gateway status) actually need.

    None/""/"current" means the dashboard's own profile — no override.
    """
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        yield None
        return

    from hermes_constants import (
        set_hermes_home_override,
        reset_hermes_home_override,
    )

    profile_dir = _resolve_profile_dir(requested)
    token = set_hermes_home_override(str(profile_dir))
    try:
        yield profile_dir
    finally:
        reset_hermes_home_override(token)


class SkillToggle(BaseModel):
    name: str
    enabled: bool
    profile: Optional[str] = None


@app.get("/api/skills")
async def get_skills(profile: Optional[str] = None):
    from tools.skills_tool import _find_all_skills
    from hermes_cli.skills_config import get_disabled_skills
    from tools.skill_usage import (
        _read_bundled_manifest_names,
        _read_hub_installed_names,
        activity_count,
        load_usage,
    )
    with _profile_scope(profile):
        config = load_config()
        disabled = get_disabled_skills(config)
        skills = _find_all_skills(skip_disabled=True)
        usage = load_usage()
        # Set-based provenance (same classification as skill_usage.provenance,
        # without a per-skill manifest read): hub > bundled > agent, where
        # "agent" covers agent-authored AND local hand-made skills — the ones
        # the user may edit/delete from the UI.
        bundled_names = _read_bundled_manifest_names()
        hub_names = _read_hub_installed_names()
    for s in skills:
        s["enabled"] = s["name"] not in disabled
        s["usage"] = activity_count(usage.get(s["name"], {}))
        s["provenance"] = (
            "hub" if s["name"] in hub_names
            else "bundled" if s["name"] in bundled_names
            else "agent"
        )
    return skills


@app.put("/api/skills/toggle")
async def toggle_skill(body: SkillToggle, profile: Optional[str] = None):
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    with _profile_scope(body.profile or profile):
        config = load_config()
        disabled = get_disabled_skills(config)
        if body.enabled:
            disabled.discard(body.name)
        else:
            disabled.add(body.name)
        save_disabled_skills(config, disabled)
    return {"ok": True, "name": body.name, "enabled": body.enabled}


class SkillCreate(BaseModel):
    name: str
    content: str
    category: Optional[str] = None
    profile: Optional[str] = None


class SkillContentUpdate(BaseModel):
    name: str
    content: str
    profile: Optional[str] = None


def _clear_skills_prompt_cache() -> None:
    """Best-effort: invalidate the skills system-prompt snapshot after a write.

    Mirrors what ``skill_manage`` does so a dashboard-authored skill is picked
    up by the next session without a manual cache reset.
    """
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass


@app.get("/api/skills/content")
async def get_skill_content(name: str, profile: Optional[str] = None):
    """Return the raw SKILL.md text for a skill, for the dashboard editor."""
    from tools.skill_manager_tool import _find_skill

    with _profile_scope(profile):
        found = _find_skill(name)
        if not found:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found.")
        skill_md = found["path"] / "SKILL.md"
        if not skill_md.exists():
            raise HTTPException(status_code=404, detail=f"Skill '{name}' has no SKILL.md.")
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"name": name, "content": content, "path": str(skill_md)}


@app.post("/api/skills")
async def create_skill(body: SkillCreate):
    """Create a new custom skill (SKILL.md) from the dashboard editor.

    Calls the same validated write path as the agent's ``skill_manage``
    tool (frontmatter validation, name/category validation, size limit,
    optional security scan) — but bypasses the agent write-approval gate:
    a write from the authenticated dashboard IS the user acting directly.
    """
    from tools.skill_manager_tool import _create_skill

    with _profile_scope(body.profile):
        result = _create_skill(body.name, body.content, body.category or None)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create skill."))
    _clear_skills_prompt_cache()
    return result


@app.put("/api/skills/content")
async def update_skill_content(body: SkillContentUpdate):
    """Replace the SKILL.md of an existing skill (full rewrite) from the editor."""
    from tools.skill_manager_tool import _edit_skill

    with _profile_scope(body.profile):
        result = _edit_skill(body.name, body.content)
    if not result.get("success"):
        err = result.get("error", "Failed to update skill.")
        status = 404 if "not found" in str(err).lower() else 400
        raise HTTPException(status_code=status, detail=err)
    _clear_skills_prompt_cache()
    return result


@app.get("/api/tools/toolsets")
async def get_toolsets(profile: Optional[str] = None):
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        _get_platform_tools,
        _toolset_has_keys,
        gui_toolset_label,
    )
    from toolsets import resolve_toolset

    with _profile_scope(profile):
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
            "name": name,
            "label": gui_toolset_label(label),
            "description": desc,
            "enabled": is_enabled,
            "available": is_enabled,
            "configured": _toolset_has_keys(name, config),
            "tools": tools,
        })
    return result


class ToolsetToggle(BaseModel):
    enabled: bool
    profile: Optional[str] = None


@app.put("/api/tools/toolsets/{name}")
async def toggle_toolset(name: str, body: ToolsetToggle, profile: Optional[str] = None):
    """Enable/disable a configurable toolset for the desktop (cli) platform.

    Persists to ``platform_toolsets.cli`` via the same ``_save_platform_tools``
    helper the CLI ``hermes tools`` picker uses, so the GUI and CLI stay in
    lockstep. Scoped to ``body.profile`` when provided. Returns 400 for
    unknown toolset keys.
    """
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        _get_platform_tools,
        _save_platform_tools,
    )

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    with _profile_scope(body.profile or profile):
        config = load_config()
        enabled = set(
            _get_platform_tools(config, "cli", include_default_mcp_servers=False)
        )
        if body.enabled:
            enabled.add(name)
        else:
            enabled.discard(name)
        _save_platform_tools(config, "cli", enabled)
    return {"ok": True, "name": name, "enabled": body.enabled}


@app.get("/api/tools/toolsets/{name}/config")
async def get_toolset_config(name: str, profile: Optional[str] = None):
    """Return the provider matrix + key status for a toolset's config panel.

    Surfaces the same provider rows the CLI ``hermes tools`` picker shows
    (via ``_visible_providers``), each with its ``env_vars`` annotated with
    current ``is_set`` state so the GUI can render provider selection + key
    entry. Toolsets without a ``TOOL_CATEGORIES`` entry return an empty
    provider list and ``has_category: false``. Returns 400 for unknown keys.
    """
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES,
        _get_effective_configurable_toolsets,
        _is_provider_active,
        _visible_providers,
    )
    from hermes_cli.config import get_env_value

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    with _profile_scope(profile):
        config = load_config()
        cat = TOOL_CATEGORIES.get(name)
        providers = []
        active_provider = None
        if cat:
            for prov in _visible_providers(cat, config, force_fresh=True):
                env_vars = [
                    {
                        "key": e["key"],
                        "prompt": e.get("prompt", e["key"]),
                        "url": e.get("url"),
                        "default": e.get("default"),
                        "is_set": bool(get_env_value(e["key"])),
                    }
                    for e in prov.get("env_vars", [])
                ]
                # Surface the same active-provider determination the CLI picker
                # uses (``_is_provider_active``) so the GUI highlights the provider
                # actually written to config (e.g. web.backend), not just the first
                # keyless one in the list.
                is_active = _is_provider_active(prov, config, force_fresh=True)
                if is_active and active_provider is None:
                    active_provider = prov["name"]
                providers.append({
                    "name": prov["name"],
                    "badge": prov.get("badge", ""),
                    "tag": prov.get("tag", ""),
                    "env_vars": env_vars,
                    "post_setup": prov.get("post_setup"),
                    "requires_nous_auth": bool(prov.get("requires_nous_auth")),
                    "is_active": is_active,
                })
    return {
        "name": name,
        "has_category": cat is not None,
        "providers": providers,
        "active_provider": active_provider,
    }


class ToolsetProviderSelect(BaseModel):
    provider: str
    profile: Optional[str] = None


# Toolsets whose backends carry a selectable model catalog, mapped to the
# config.yaml section their `model` key lives in. Mirrors the CLI's
# post-selection model pickers (`_configure_imagegen_model_for_plugin` /
# `_configure_videogen_model_for_plugin` in tools_config.py).
_MODEL_CATALOG_TOOLSETS = {
    "image_gen": "image_gen",
    "video_gen": "video_gen",
}


def _resolve_toolset_model_plugin(ts_key: str, provider_row: dict) -> Optional[str]:
    """Map a provider picker row to its model-catalog plugin name.

    Plugin-backed rows carry ``image_gen_plugin_name`` / ``video_gen_plugin_name``;
    the managed "Nous Subscription" image row instead carries the legacy
    ``imagegen_backend: "fal"`` marker (same underlying FAL catalog).
    """
    if ts_key == "image_gen":
        return provider_row.get("image_gen_plugin_name") or (
            "fal" if provider_row.get("imagegen_backend") else None
        )
    if ts_key == "video_gen":
        return provider_row.get("video_gen_plugin_name")
    return None


def _toolset_model_catalog(ts_key: str, plugin_name: str):
    """Return ``(catalog_dict, default_model)`` for a toolset's plugin backend."""
    from hermes_cli.tools_config import (
        _plugin_image_gen_catalog,
        _plugin_video_gen_catalog,
    )

    if ts_key == "image_gen":
        return _plugin_image_gen_catalog(plugin_name)
    return _plugin_video_gen_catalog(plugin_name)


def _find_toolset_provider_row(ts_key: str, config: dict, provider: Optional[str]) -> Optional[dict]:
    """Resolve a provider picker row by name, or the active row when omitted."""
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES,
        _is_provider_active,
        _visible_providers,
    )

    cat = TOOL_CATEGORIES.get(ts_key)
    if cat is None:
        return None
    rows = _visible_providers(cat, config, force_fresh=True)
    if provider:
        return next((p for p in rows if p.get("name") == provider), None)
    return next(
        (p for p in rows if _is_provider_active(p, config, force_fresh=True)), None
    )


@app.get("/api/tools/toolsets/{name}/models")
async def get_toolset_models(
    name: str, provider: Optional[str] = None, profile: Optional[str] = None
):
    """Return the model catalog for a toolset backend (image/video gen).

    The GUI counterpart of the model picker `hermes tools` runs after a
    backend is selected — e.g. FAL's multi-model catalog (speed / strengths /
    price per model). ``provider`` names a picker row; omitted, the currently
    active provider is used. Toolsets without model catalogs return
    ``has_models: false``.
    """
    section = _MODEL_CATALOG_TOOLSETS.get(name)
    if section is None:
        return {"name": name, "has_models": False, "models": [], "current": None, "default": None}

    with _profile_scope(profile):
        config = load_config()
        row = _find_toolset_provider_row(name, config, provider)
        plugin = _resolve_toolset_model_plugin(name, row) if row else None
        if not plugin:
            return {
                "name": name,
                "has_models": False,
                "models": [],
                "current": None,
                "default": None,
            }

        catalog, default_model = _toolset_model_catalog(name, plugin)
        section_cfg = config.get(section)
        current = None
        if isinstance(section_cfg, dict):
            raw = section_cfg.get("model")
            if isinstance(raw, str) and raw.strip():
                current = raw.strip()
        if current not in catalog:
            current = default_model if default_model in catalog else None

    models = [
        {
            "id": model_id,
            "display": meta.get("display", model_id),
            "speed": meta.get("speed", ""),
            "strengths": meta.get("strengths", ""),
            "price": meta.get("price", ""),
        }
        for model_id, meta in catalog.items()
    ]
    return {
        "name": name,
        "has_models": bool(models),
        "provider": row.get("name") if row else None,
        "plugin": plugin,
        "models": models,
        "current": current,
        "default": default_model,
    }


class ToolsetModelSelect(BaseModel):
    model: str
    provider: Optional[str] = None
    profile: Optional[str] = None


@app.put("/api/tools/toolsets/{name}/model")
async def select_toolset_model(
    name: str, body: ToolsetModelSelect, profile: Optional[str] = None
):
    """Persist a backend model selection (``image_gen.model`` / ``video_gen.model``).

    Validates the model against the resolved backend's catalog — the same
    write the CLI's post-selection model picker performs. Returns 400 for
    toolsets without model catalogs or unknown model ids.
    """
    section = _MODEL_CATALOG_TOOLSETS.get(name)
    if section is None:
        raise HTTPException(
            status_code=400, detail=f"Toolset has no model catalog: {name}"
        )

    model_id = (body.model or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model is required")

    with _profile_scope(body.profile or profile):
        config = load_config()
        row = _find_toolset_provider_row(name, config, body.provider)
        plugin = _resolve_toolset_model_plugin(name, row) if row else None
        if not plugin:
            raise HTTPException(
                status_code=400,
                detail=f"No model-capable backend is active for {name}",
            )

        catalog, _default = _toolset_model_catalog(name, plugin)
        if model_id not in catalog:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model {model_id!r} for backend {plugin!r}",
            )

        section_cfg = config.setdefault(section, {})
        if not isinstance(section_cfg, dict):
            section_cfg = {}
            config[section] = section_cfg
        section_cfg["model"] = model_id
        save_config(config)

    return {"ok": True, "name": name, "model": model_id, "plugin": plugin}


@app.put("/api/tools/toolsets/{name}/provider")
async def select_toolset_provider(
    name: str, body: ToolsetProviderSelect, profile: Optional[str] = None
):
    """Persist a provider selection for a toolset (no key prompting).

    Delegates to ``apply_provider_selection`` — the shared, non-interactive
    core extracted from the CLI configurator — so the GUI and ``hermes tools``
    write identical config keys (``web.backend``, ``tts.provider``, etc.).
    API keys and post-setup flows are handled by separate endpoints. Returns
    400 for unknown toolset or provider names.
    """
    from hermes_cli.tools_config import (
        apply_provider_selection,
        _get_effective_configurable_toolsets,
    )

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    with _profile_scope(body.profile or profile):
        config = load_config()
        try:
            apply_provider_selection(name, body.provider, config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc).strip('"'))
        save_config(config)
    return {"ok": True, "name": name, "provider": body.provider}


class ToolsetEnvUpdate(BaseModel):
    env: Dict[str, str]
    profile: Optional[str] = None


@app.put("/api/tools/toolsets/{name}/env")
async def save_toolset_env(name: str, body: ToolsetEnvUpdate, profile: Optional[str] = None):
    """Persist API keys for a toolset's provider env vars.

    Writes each ``key: value`` to ``~/.hermes/.env`` via ``save_env_value`` —
    the same store ``hermes tools`` writes when it prompts for keys. Keys are
    validated against the env-var allowlist for the toolset's category (the
    union of every visible provider's ``env_vars``), so the GUI can't write an
    arbitrary env var through this endpoint. A blank value is treated as
    "leave unchanged" and skipped. Returns the saved/skipped key lists and the
    refreshed ``is_set`` status. Returns 400 for unknown toolset or env keys.
    """
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES,
        _get_effective_configurable_toolsets,
        _visible_providers,
    )
    from hermes_cli.config import get_env_value, save_env_value

    valid_ts = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid_ts:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    with _profile_scope(body.profile or profile):
        config = load_config()
        cat = TOOL_CATEGORIES.get(name)
        allowed: set[str] = set()
        if cat:
            for prov in _visible_providers(cat, config, force_fresh=True):
                for e in prov.get("env_vars", []):
                    allowed.add(e["key"])

        unknown = [k for k in body.env if k not in allowed]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown env var(s) for toolset {name}: {', '.join(sorted(unknown))}",
            )

        saved: List[str] = []
        skipped: List[str] = []
        for key, value in body.env.items():
            if value and value.strip():
                try:
                    save_env_value(key, value.strip())
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                saved.append(key)
            else:
                skipped.append(key)

        status = {k: bool(get_env_value(k)) for k in allowed}
    return {"ok": True, "name": name, "saved": saved, "skipped": skipped, "is_set": status}


class ToolsetPostSetup(BaseModel):
    key: str
    profile: Optional[str] = None


@app.post("/api/tools/toolsets/{name}/post-setup")
async def run_toolset_post_setup(
    name: str, body: ToolsetPostSetup, profile: Optional[str] = None
):
    """Spawn a provider's post-setup install hook as a background action.

    Post-setup hooks (npm install for browser/Camofox, pip install for
    KittenTTS/Piper/ddgs, cua-driver fetch, etc.) are long-running and
    text-output, so this follows the spawn-action pattern: it launches
    ``hermes tools post-setup <key>`` and the frontend tails the log via
    ``GET /api/actions/tools-post-setup/status``. The ``key`` is validated
    against the declared post-setup allowlist before spawning. Returns 400
    for unknown toolset or post-setup key.

    ``profile`` spawns the hook as ``hermes -p <profile> tools post-setup``.
    Most hooks install machine-level artifacts (repo node_modules, shared
    pip packages) where the scope is inert, but hooks that read config or
    write per-profile state must see the same HERMES_HOME the rest of the
    drawer's writes targeted — so the scope is threaded for consistency.
    """
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        valid_post_setup_keys,
    )

    valid_ts = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid_ts:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    if body.key not in valid_post_setup_keys():
        raise HTTPException(
            status_code=400, detail=f"Unknown post-setup key: {body.key}"
        )

    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(body.profile or profile)
            + ["tools", "post-setup", body.key],
            "tools-post-setup",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn tools post-setup")
        raise HTTPException(
            status_code=500, detail=f"Failed to run post-setup: {exc}"
        )
    return {"ok": True, "pid": proc.pid, "name": "tools-post-setup", "key": body.key}


# ---------------------------------------------------------------------------
# Computer Use (cua-driver) — cross-platform readiness + macOS permission grant
#
# cua-driver runs on macOS, Windows, and Linux. The desktop card reflects
# per-OS readiness: on macOS the Accessibility + Screen Recording TCC grants
# (which attach to cua-driver's OWN identity, com.trycua.driver — not Hermes,
# so no app entitlement is involved); elsewhere, driver health from
# `cua-driver doctor`. The grant flow is macOS-only (no TCC toggles to request
# on Windows/Linux).
# ---------------------------------------------------------------------------


@app.get("/api/tools/computer-use/status")
async def get_computer_use_status(profile: Optional[str] = None):
    """Cross-platform Computer Use readiness for the desktop card.

    See ``tools.computer_use.permissions.computer_use_status`` for the payload
    shape. Read-only and fast (shells ``cua-driver doctor`` + macOS
    ``permissions status``).
    """
    from tools.computer_use.permissions import computer_use_status

    with _profile_scope(profile):
        return computer_use_status()


@app.post("/api/tools/computer-use/permissions/grant")
async def grant_computer_use_permissions(profile: Optional[str] = None):
    """Spawn ``hermes computer-use permissions grant`` as a background action.

    macOS-only: ``cua-driver permissions grant`` launches CuaDriver via
    LaunchServices so the TCC dialog is attributed to com.trycua.driver, then
    waits for approval. The frontend polls ``GET /api/actions/computer-use-
    grant/status`` and re-reads ``/status`` once it exits. Windows/Linux have
    no TCC toggles to grant, so this returns 400 there.
    """
    if sys.platform != "darwin":
        raise HTTPException(
            status_code=400,
            detail="Computer Use permission grants are a macOS concept.",
        )
    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(profile)
            + ["computer-use", "permissions", "grant"],
            "computer-use-grant",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn computer-use permissions grant")
        raise HTTPException(
            status_code=500, detail=f"Failed to request permissions: {exc}"
        )
    return {"ok": True, "pid": proc.pid, "name": "computer-use-grant"}


# ---------------------------------------------------------------------------
# Raw YAML config endpoint
# ---------------------------------------------------------------------------


class RawConfigUpdate(BaseModel):
    yaml_text: str
    profile: Optional[str] = None


@app.get("/api/config/raw")
async def get_config_raw(profile: Optional[str] = None):
    """Raw config.yaml text plus its resolved path.

    ``path`` is resolved inside ``_profile_scope`` so the Config page header
    shows the file the switched profile actually reads/writes — /api/status's
    ``config_path`` is machine-global and always reports the dashboard
    process's own profile, which is wrong under the global profile switcher.
    """
    with _profile_scope(profile):
        path = get_config_path()
    if not path.exists():
        return {"yaml": "", "path": str(path)}
    return {"yaml": path.read_text(encoding="utf-8"), "path": str(path)}


@app.put("/api/config/raw")
async def update_config_raw(body: RawConfigUpdate, profile: Optional[str] = None):
    try:
        parsed = yaml.safe_load(body.yaml_text)
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML must be a mapping")
        with _profile_scope(body.profile or profile):
            save_config(parsed)
        return {"ok": True}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")


# ---------------------------------------------------------------------------
# Token / cost analytics endpoint
# ---------------------------------------------------------------------------


@app.get("/api/analytics/usage")
async def get_usage_analytics(days: int = 30, profile: Optional[str] = None):
    from agent.insights import InsightsEngine

    db = _open_session_db_for_profile(profile)
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
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ?
            GROUP BY day ORDER BY day
        """, (cutoff,))
        daily = [dict(r) for r in cur.fetchall()]

        cur2 = db._conn.execute("""
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
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
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ?
        """, (cutoff,))
        totals = dict(cur3.fetchone())
        insights_report = InsightsEngine(db).generate(days=days)
        skills = insights_report.get("skills", {
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        })

        return {
            "daily": daily,
            "by_model": by_model,
            "totals": totals,
            "period_days": days,
            "skills": skills,
            # Per-tool-name call counts (already computed by InsightsEngine);
            # the desktop Capabilities page aggregates these per toolset.
            "tools": insights_report.get("tools", []),
        }
    finally:
        db.close()


@app.get("/api/analytics/models")
async def get_models_analytics(days: int = 30, profile: Optional[str] = None):
    """Rich per-model analytics for the Models dashboard page.

    Returns token/cost/session breakdown per model plus capability metadata
    from models.dev (context window, vision, tools, reasoning, etc.).
    """
    db = _open_session_db_for_profile(profile)
    try:
        cutoff = time.time() - (days * 86400)

        cur = db._conn.execute("""
            SELECT model,
                   billing_provider,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls,
                   SUM(tool_call_count) as tool_calls,
                   MAX(started_at) as last_used_at,
                   AVG(input_tokens + output_tokens) as avg_tokens_per_session
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
            GROUP BY model, billing_provider
            ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, (cutoff,))
        raw_rows = [dict(r) for r in cur.fetchall()]

        # Session rows can be created before the first billable provider call
        # finishes. If that early row records only the model name, and a later
        # row for the same model has real accounting + billing_provider, the
        # Models page used to show a duplicate "0 tokens / — API calls" card
        # next to the real provider card. Fold those session-only rows into
        # the single accounted provider row when the ownership is unambiguous.
        rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
        for row in raw_rows:
            rows_by_model.setdefault(row.get("model") or "", []).append(row)

        rows: List[Dict[str, Any]] = []
        for model_rows in rows_by_model.values():
            provider_rows = [r for r in model_rows if r.get("billing_provider")]
            if len(provider_rows) == 1:
                target = provider_rows[0]
                for row in model_rows:
                    if row is target or row.get("billing_provider"):
                        continue
                    has_usage = any(
                        (row.get(key) or 0) != 0
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                            "reasoning_tokens",
                            "estimated_cost",
                            "actual_cost",
                            "api_calls",
                            "tool_calls",
                        )
                    )
                    if has_usage:
                        continue
                    target["sessions"] = (target.get("sessions") or 0) + (row.get("sessions") or 0)
                    target["last_used_at"] = max(target.get("last_used_at") or 0, row.get("last_used_at") or 0)
                    total_tokens = (target.get("input_tokens") or 0) + (target.get("output_tokens") or 0)
                    sessions = target.get("sessions") or 0
                    target["avg_tokens_per_session"] = total_tokens / sessions if sessions else 0
                rows.append(target)
                rows.extend(
                    r for r in model_rows
                    if r is not target
                    and (r.get("billing_provider") or any(
                        (r.get(key) or 0) != 0
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                            "reasoning_tokens",
                            "estimated_cost",
                            "actual_cost",
                            "api_calls",
                            "tool_calls",
                        )
                    ))
                )
            else:
                rows.extend(model_rows)

        rows.sort(
            key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
            reverse=True,
        )

        models = []
        for row in rows:
            provider = row.get("billing_provider") or ""
            model_name = row["model"]
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

            models.append({
                "model": model_name,
                "provider": provider,
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cache_read_tokens": row["cache_read_tokens"],
                "reasoning_tokens": row["reasoning_tokens"],
                "estimated_cost": row["estimated_cost"],
                "actual_cost": row["actual_cost"],
                "sessions": row["sessions"],
                "api_calls": row["api_calls"],
                "tool_calls": row["tool_calls"],
                "last_used_at": row["last_used_at"],
                "avg_tokens_per_session": row["avg_tokens_per_session"],
                "capabilities": caps,
            })

        totals_cur = db._conn.execute("""
            SELECT COUNT(DISTINCT model) as distinct_models,
                   SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        """, (cutoff,))
        totals = dict(totals_cur.fetchone())

        return {
            "models": models,
            "totals": totals,
            "period_days": days,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# /api/pty — PTY-over-WebSocket bridge for the dashboard "Chat" tab.
#
# The endpoint spawns the same ``hermes --tui`` binary the CLI uses, behind
# a POSIX pseudo-terminal, and forwards bytes + resize escapes across a
# WebSocket.  The browser renders the ANSI through xterm.js (see
# web/src/pages/ChatPage.tsx).
#
# Auth: ``?token=<session_token>`` query param (browsers can't set
# Authorization on the WS upgrade).  Same ephemeral ``_SESSION_TOKEN`` as
# REST.  Localhost-only — we defensively reject non-loopback clients even
# though uvicorn binds to 127.0.0.1.
# ---------------------------------------------------------------------------

# PTY bridge: POSIX uses pty_bridge (fcntl/termios/ptyprocess); native Windows
# uses win_pty_bridge (pywinpty/ConPTY, already a declared dependency).  Both
# expose the same public surface — spawn/read/write/resize/close/is_available —
# so the /api/pty WebSocket handler needs no platform guards.
if sys.platform.startswith("win"):
    try:
        from hermes_cli.win_pty_bridge import WinPtyBridge as PtyBridge, PtyUnavailableError
        _PTY_BRIDGE_AVAILABLE = True
    except ImportError:  # pragma: no cover - pywinpty missing
        PtyBridge = None  # type: ignore[assignment]
        _PTY_BRIDGE_AVAILABLE = False

        class PtyUnavailableError(RuntimeError):  # type: ignore[no-redef]
            """Stub when win_pty_bridge cannot be imported."""
            pass
else:
    try:
        from hermes_cli.pty_bridge import PtyBridge, PtyUnavailableError
        _PTY_BRIDGE_AVAILABLE = True
    except ImportError:  # pragma: no cover - dev env without ptyprocess
        PtyBridge = None  # type: ignore[assignment]
        _PTY_BRIDGE_AVAILABLE = False

        class PtyUnavailableError(RuntimeError):  # type: ignore[no-redef]
            """Stub on platforms where pty_bridge can't be imported."""
            pass

_RESIZE_RE = re.compile(rb"\x1b\[RESIZE:(\d+);(\d+)\]")
_PTY_READ_CHUNK_TIMEOUT = 0.2

# Keep-alive PTY sessions: a terminal connecting with ``?attach=<token>`` is
# bound to a process that survives disconnect/refresh and is reattachable.
from hermes_cli.pty_session import PtySessionRegistry, RegistryFull, run_reaper  # noqa: E402

PTY_REGISTRY = PtySessionRegistry(
    ttl=30 * 60,
    max_sessions=16,
    buffer_cap=1 * 1024 * 1024,
    read_timeout=_PTY_READ_CHUNK_TIMEOUT,
)


async def _legacy_pump(ws: "WebSocket", bridge) -> None:
    """Original 1:1 socket<->PTY pump: stream until disconnect, then close the
    bridge. Used when no ``?attach=`` token is supplied (keep-alive opt-in).

    Behavior is identical to the pre-keep-alive ``pty_ws`` body, including the
    #54028 half-open-socket protection (reader EOF → close the WS so the
    writer's ``ws.receive()`` unparks) and the #53227 ``to_thread`` offloads
    for the blocking ``bridge.close()``.
    """
    loop = asyncio.get_running_loop()

    # --- reader task: PTY master → WebSocket ----------------------------
    async def pump_pty_to_ws() -> None:
        try:
            while True:
                chunk = await loop.run_in_executor(
                    None, bridge.read, _PTY_READ_CHUNK_TIMEOUT
                )
                if chunk is None:  # EOF
                    return
                if not chunk:  # no data this tick; yield control and retry
                    await asyncio.sleep(0)
                    continue
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    return
        finally:
            # The child has exited (EOF) or the send side broke.  Close the
            # WebSocket so the writer loop's ``ws.receive()`` returns instead
            # of blocking forever — otherwise, when the browser's socket is
            # half-open (no FIN delivered, common on macOS/launchd) the
            # handler never reaches its ``finally`` and the PTY's fds leak.
            # With dashboard auto-reconnect (#52962) every dropped socket then
            # stacks a fresh PTY on top of the orphaned one, exhausting fds.
            #
            # Reap the bridge here too (close() is idempotent): on child EOF the
            # writer loop's ``finally`` is the usual closer, but if the handler
            # task is cancelled the instant we close the WS, that ``finally``
            # can be skipped, leaking the PTY. Closing from the EOF path makes
            # the reap independent of that cancellation race (#54028).
            try:
                await asyncio.to_thread(bridge.close)
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass

    reader_task = asyncio.create_task(pump_pty_to_ws())

    # --- writer loop: WebSocket → PTY master ----------------------------
    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                # Raised when ws.receive() is called after the socket is
                # already disconnected (e.g. closed by the reader task above).
                break
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is None:
                text = msg.get("text")
                raw = text.encode("utf-8") if isinstance(text, str) else b""
            if not raw:
                continue
            # Resize escape is consumed locally, never written to the PTY.
            match = _RESIZE_RE.match(raw)
            if match and match.end() == len(raw):
                bridge.resize(cols=int(match.group(1)), rows=int(match.group(2)))
                continue
            bridge.write(raw)
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
        await asyncio.to_thread(bridge.close)


_VALID_CHANNEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# Starlette's TestClient reports the peer as "testclient"; treat it as
# loopback so tests don't need to rewrite request scope.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _ws_client_reason(ws: "WebSocket") -> Optional[str]:
    """Return a rejection reason for the client IP, or None when allowed.

    Reasons are short machine-parseable tokens logged on the rejection path
    so a "WS keeps closing" report can be diagnosed from agent.log without a
    repro. ``None`` means the peer IP passed this gate.

    See :func:`_ws_client_is_allowed` for the full policy rationale.
    """
    if getattr(app.state, "auth_required", False):
        return None
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return None
    client_host = ws.client.host if ws.client else ""
    if not client_host:
        # Fail-closed: a loopback-bound dashboard with auth disabled must
        # not accept a WebSocket with no identifiable peer. ASGI servers
        # behind a misconfigured proxy or unix socket can deliver
        # ws.client == None or "" — treating that as "allowed" would let
        # an unidentified peer reach a loopback-only surface.
        return f"missing_or_empty_peer bound={bound_host or '?'}"
    if client_host in _LOOPBACK_HOSTS:
        return None
    return f"peer_not_loopback peer={client_host} bound={bound_host or '?'}"


def _ws_client_is_allowed(ws: "WebSocket") -> bool:
    """Check if the WebSocket client IP is acceptable.

    Loopback bind: only loopback clients allowed — the legacy
    ``?token=<_SESSION_TOKEN>`` path is the only auth we have, so we
    don't want LAN hosts guessing tokens.

    Explicit non-loopback bind (``--host 0.0.0.0``, ``--host ::``, or a
    specific address such as a Tailscale/LAN IP, always with
    ``--insecure``): allow any peer. The operator explicitly opted into
    non-loopback exposure, so the loopback-only peer restriction does not
    apply. DNS-rebinding is still blocked by the Host/Origin guard in
    :func:`_ws_host_origin_is_allowed`, which mirrors the HTTP layer and
    requires the Host header to match the bound interface — the same
    defence ``_is_accepted_host`` applies to non-loopback HTTP requests.

    Gated mode: any peer is allowed — uvicorn's ``proxy_headers=True``
    (enabled when the OAuth gate is active so cookies can pick up
    ``X-Forwarded-Proto``) rewrites ``ws.client.host`` to the
    X-Forwarded-For value, which is the real internet client IP. The
    OAuth gate + single-use ``?ticket=`` is the auth at that point; the
    Host/Origin guard in :func:`_ws_host_origin_is_allowed` is what
    blocks DNS-rebinding here, not the peer IP.
    """
    if getattr(app.state, "auth_required", False):
        return True
    # Any explicit non-loopback bind (0.0.0.0, ::, or a specific LAN /
    # Tailscale address) means the operator opted into non-loopback
    # access via --insecure.  The loopback-only peer gate only applies to
    # an actual loopback bind; otherwise the WS handshake is rejected even
    # though same-bind HTTP requests pass _is_accepted_host.
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return True
    client_host = ws.client.host if ws.client else ""
    if not client_host:
        # Fail-closed: see _ws_client_reason for rationale. An empty
        # client_host on a loopback-bound dashboard with auth disabled
        # must be rejected, not accepted as a default-allow.
        return False
    return client_host in _LOOPBACK_HOSTS


def _ws_host_origin_reason(ws: "WebSocket") -> Optional[str]:
    """Return a Host/Origin rejection reason, or None when allowed.

    Mirrors :func:`_ws_host_origin_is_allowed` but yields a short
    machine-parseable token (``host_mismatch …`` / ``origin_mismatch …``)
    on rejection so the close path can log *why* the upgrade was refused.
    """
    bound_host = getattr(app.state, "bound_host", None)
    if not bound_host:
        return None

    host_header = ws.headers.get("host", "")
    if not _is_accepted_host(host_header, bound_host):
        return f"host_mismatch host={host_header or '?'} bound={bound_host}"

    origin = ws.headers.get("origin", "")
    if not origin:
        return None

    parsed = urllib.parse.urlparse(origin)
    if parsed.scheme not in {"http", "https"}:
        # Non-web origin (packaged Electron: file://, null, app://). The
        # upstream credential check is the real auth boundary; trust it.
        # See _ws_host_origin_is_allowed for the full rationale.
        return None

    if not parsed.netloc:
        return f"origin_mismatch origin={origin} bound={bound_host}"

    if not _is_accepted_host(parsed.netloc, bound_host):
        return f"origin_mismatch origin={origin} bound={bound_host}"
    return None


def _ws_host_origin_is_allowed(ws: "WebSocket") -> bool:
    """Apply the dashboard Host/Origin guard to WebSocket upgrades.

    FastAPI HTTP middleware does not run for WebSocket routes, so the
    DNS-rebinding Host check used for normal dashboard HTTP requests must be
    repeated here before accepting the upgrade.  Browsers also send an Origin
    header on WebSocket handshakes; when present, require it to target the
    same bound dashboard host.
    """
    return _ws_host_origin_reason(ws) is None


def _ws_request_reason(ws: "WebSocket") -> Optional[str]:
    """First Host/Origin or peer-IP rejection reason, or None when allowed."""
    return _ws_host_origin_reason(ws) or _ws_client_reason(ws)


def _ws_request_is_allowed(ws: "WebSocket") -> bool:
    """Return True when the WebSocket upgrade matches dashboard boundaries."""
    return _ws_host_origin_is_allowed(ws) and _ws_client_is_allowed(ws)


def _ws_auth_mode() -> str:
    """Short label for the active WS auth mode — logged on every connection."""
    if getattr(app.state, "auth_required", False):
        return "gated"
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return "insecure"
    return "loopback"


def _ws_auth_reason(ws: "WebSocket") -> tuple[Optional[str], str]:
    """Validate WS-upgrade auth; return ``(reason, credential)``.

    ``reason`` is None when the credential is accepted, else a short
    machine-parseable token explaining the rejection (``no_credential``,
    ``token_mismatch``, ``ticket_invalid``, ``internal_invalid``).
    ``credential`` names which credential type was presented (``ticket``,
    ``internal``, ``token``, or ``none``) so the accepted path can log *how*
    a peer authed, not just that it did.

    Loopback / ``--insecure``: legacy ``?token=<_SESSION_TOKEN>`` query
    parameter, constant-time compared.

    Gated (public bind, no ``--insecure``): one of two credentials —

    * ``?ticket=<single-use>`` — a browser-minted, single-use, 30s-TTL ticket
      consumed against the dashboard-auth ticket store. This is what the SPA
      (and native clients) use.
    * ``?internal=<process-credential>`` — the process-lifetime internal
      credential, used only by WS clients the server spawns itself (the
      embedded-TUI PTY child attaching to ``/api/ws`` and ``/api/pub``). It
      is multi-use and never expires so the child can reconnect, and is never
      injected into the SPA — see ``dashboard_auth.ws_tickets`` for the
      threat model.

    The legacy ``?token=`` path is unconditionally rejected in gated mode
    (the SPA bundle isn't carrying the token any longer, and a leaked
    ``_SESSION_TOKEN`` must not grant WS access once the gate is engaged).

    Audit-logs the rejection so operators can debug "WS keeps closing"
    issues from the log.
    """
    auth_required = bool(getattr(app.state, "auth_required", False))
    if auth_required:
        # Lazy import — keeps this function importable in test harnesses
        # that don't bring in the dashboard_auth layer.
        from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
        from hermes_cli.dashboard_auth.ws_tickets import (
            TicketInvalid,
            consume_internal_credential,
            consume_ticket,
        )

        # Server-spawned children (PTY child → /api/ws, /api/pub) present the
        # multi-use internal credential rather than a single-use ticket, so
        # they survive reconnects and slow cold boots.
        internal = ws.query_params.get("internal", "")
        if internal:
            try:
                consume_internal_credential(internal)
                return None, "internal"
            except TicketInvalid as exc:
                audit_log(
                    AuditEvent.WS_TICKET_REJECTED,
                    reason=f"internal: {exc}",
                    ip=(ws.client.host if ws.client else ""),
                    path=ws.url.path,
                )
                return "internal_invalid", "internal"

        ticket = ws.query_params.get("ticket", "")
        if not ticket:
            return "no_credential", "none"

        try:
            consume_ticket(ticket)
            return None, "ticket"
        except TicketInvalid as exc:
            audit_log(
                AuditEvent.WS_TICKET_REJECTED,
                reason=str(exc),
                ip=(ws.client.host if ws.client else ""),
                path=ws.url.path,
            )
            return "ticket_invalid", "ticket"

    token = ws.query_params.get("token", "")
    if not token:
        return "no_credential", "none"
    if hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode()):
        return None, "token"
    return "token_mismatch", "token"


def _ws_auth_ok(ws: "WebSocket") -> bool:
    """True when the WS-upgrade credential is accepted. See _ws_auth_reason."""
    return _ws_auth_reason(ws)[0] is None

# Per-channel subscriber registry used by /api/pub (PTY-side gateway → dashboard)
# and /api/events (dashboard → browser sidebar).  Keyed by an opaque channel id
# the chat tab generates on mount; entries auto-evict when the last subscriber
# drops AND the publisher has disconnected.
# (Channel state and the chat-argv lock are initialised in _lifespan on app
# startup — see _get_event_state / _get_chat_argv_lock above.)


def _resolve_chat_argv(
    resume: Optional[str] = None,
    sidecar_url: Optional[str] = None,
    profile: Optional[str] = None,
    active_session_file: Optional[str] = None,
) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve the argv + cwd + env for the chat PTY.

    Default: whatever ``hermes --tui`` would run.  Tests monkeypatch this
    function to inject a tiny fake command (``cat``, ``sh -c 'printf …'``)
    so nothing has to build Node or the TUI bundle.

    Session resume is propagated via the ``HERMES_TUI_RESUME`` env var —
    matching what ``hermes_cli.main._launch_tui`` does for the CLI path.
    Appending ``--resume <id>`` to argv doesn't work because ``ui-tui`` does
    not parse its argv.

    ``HERMES_TUI_GATEWAY_URL`` is injected so the PTY child can attach to
    this process's in-memory ``tui_gateway`` instance instead of spawning
    its own Python gateway subprocess.

    `sidecar_url` (when set) is forwarded as ``HERMES_TUI_SIDECAR_URL`` so
    the spawned ``tui_gateway.entry`` can mirror dispatcher emits to the
    dashboard's ``/api/pub`` endpoint (see :func:`pub_ws`).

    `active_session_file` (when set) is forwarded as
    ``HERMES_TUI_ACTIVE_SESSION_FILE``. The TUI writes the current session id
    there whenever it creates/resumes/switches sessions, giving the dashboard a
    small cross-process breadcrumb for reconnecting after an unexpected browser
    WebSocket close.

    `profile` (when set) scopes the ENTIRE chat to that profile by pointing
    ``HERMES_HOME`` at the profile dir in the child env. Every spawned
    process (the TUI and the ``tui_gateway.entry`` it launches) resolves
    ``get_hermes_home()`` from that env var at its own import, so the child
    binds the profile's config, skills, memory, and state.db from the start
    — the same propagation ``hermes -p <name>`` performs. The in-process
    ``HERMES_TUI_GATEWAY_URL`` attach is SKIPPED for scoped chats: the
    dashboard's in-memory gateway runs under the dashboard's own profile,
    so a profile-scoped chat must spawn its own gateway subprocess.
    """
    from hermes_cli.main import PROJECT_ROOT, _make_tui_argv

    profile_dir: Optional[Path] = None
    requested = (profile or "").strip()
    if requested and requested.lower() != "current":
        profile_dir = _resolve_profile_dir(requested)

    argv, cwd = _make_tui_argv(PROJECT_ROOT / "ui-tui", tui_dev=False)
    env = os.environ.copy()
    try:
        from hermes_cli.config import apply_terminal_config_to_env
        apply_terminal_config_to_env(env=env)
    except Exception:
        _log.debug("Failed to apply terminal config bridge for dashboard chat", exc_info=True)
    env.setdefault("NODE_ENV", "production")
    # Browser-embedded chat should prefer stable wheel-based scrollback over
    # native terminal mouse tracking. When mouse tracking is enabled, wheel
    # events are consumed by the TUI and forwarded as terminal input, which
    # makes browser-side transcript scrolling feel broken. Keep the terminal
    # build unchanged for native CLI usage; only disable mouse tracking for
    # the dashboard PTY path.
    env.setdefault("HERMES_TUI_DISABLE_MOUSE", "1")
    env.setdefault("HERMES_TUI_INLINE", "1")
    # The dashboard terminal is xterm.js, which always renders 24-bit RGB.
    # But chalk inside the TUI child decides its color depth from the
    # SERVER process env — and hosted/cloud deploys run the dashboard under
    # a process manager (container init, systemd) with no COLORTERM, so
    # chalk downgrades every hex color to the xterm 256 palette. The skin's
    # bronze border #CD7F32 snaps to palette 173 (#D7875F, salmon-red) and
    # the banner reads red/yellow instead of gold. Local launches dodge
    # this only because the operator's interactive terminal leaks
    # COLORTERM=truecolor into os.environ. Backfill it for the PTY child;
    # setdefault so an explicit operator value still wins.
    env.setdefault("COLORTERM", "truecolor")
    env["HERMES_TUI_DASHBOARD"] = "1"

    if profile_dir is not None:
        env["HERMES_HOME"] = str(profile_dir)

    if resume:
        _resume_db = _open_session_db_for_profile(
            requested if profile_dir is not None else None
        )
        try:
            latest_resume, _latest_path = _session_latest_descendant(resume, _resume_db)
        finally:
            _resume_db.close()
        if latest_resume:
            resume = latest_resume
        env["HERMES_TUI_RESUME"] = resume

    if sidecar_url:
        env["HERMES_TUI_SIDECAR_URL"] = sidecar_url

    if active_session_file:
        env["HERMES_TUI_ACTIVE_SESSION_FILE"] = active_session_file

    # Profile-scoped chats must NOT attach to the dashboard's in-memory
    # gateway — it runs under the dashboard's own profile. Without the
    # attach URL, gatewayClient spawns its own `tui_gateway.entry`, which
    # inherits the profile HERMES_HOME set above.
    if profile_dir is None:
        if gateway_ws_url := _build_gateway_ws_url():
            env["HERMES_TUI_GATEWAY_URL"] = gateway_ws_url

    return list(argv), str(cwd) if cwd else None, env


# Hosts that mean "listen on every interface" — the server should bind to
# them, but an in-container client must NOT dial them: dialing 0.0.0.0
# resolves to "any local interface", which on most platforms routes through
# the kernel's wildcard stack and behind a forward proxy (HTTPS_PROXY with
# a NO_PROXY that doesn't list 0.0.0.0) gets MITM'd into a failed handshake
# (issue #58993).  The fix is to use a loopback address for the client
# netloc while leaving the bind host alone.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})


def _resolve_client_ws_host() -> Optional[str]:
    """Return the host the in-container WS client should dial.

    Resolution order:

    1. Explicit ``HERMES_DASHBOARD_WS_HOST`` env var — wins always. Operators
       running the dashboard behind a forward proxy can pin a routable host
       (e.g. ``127.0.0.1``, the container's internal IP, or a sidecar DNS
       name) and bypass auto-detection entirely.
    2. The configured bind host — if it's a wildcard (``0.0.0.0`` / ``::``),
       substitute ``127.0.0.1`` since both the dashboard and its TUI child
       run in the same container.
    3. Any other bind host (loopback or LAN IP) — preserved verbatim.
    """
    explicit = os.environ.get("HERMES_DASHBOARD_WS_HOST", "").strip()
    if explicit:
        return explicit

    host = getattr(app.state, "bound_host", None)
    if not host:
        return None

    if host in _WILDCARD_HOSTS:
        return "127.0.0.1"

    return host


def _build_gateway_ws_url() -> Optional[str]:
    """ws:// URL the PTY child should attach to for JSON-RPC gateway traffic.

    Loopback / ``--insecure``: ``?token=<_SESSION_TOKEN>``.

    Gated mode: the legacy token path is rejected by ``_ws_auth_ok``, so the
    server-spawned PTY child authenticates with the process-lifetime internal
    credential (``?internal=``). It must NOT use a single-use browser ticket:
    the child reads this URL once at startup and reuses it on every reconnect,
    and a 30s-TTL ticket can expire before a slow cold boot even dials.
    """
    host = _resolve_client_ws_host()
    port = getattr(app.state, "bound_port", None)

    if not host or not port:
        return None

    netloc = (
        f"[{host}]:{port}"
        if ":" in host and not host.startswith("[")
        else f"{host}:{port}"
    )

    if getattr(app.state, "auth_required", False):
        from hermes_cli.dashboard_auth.ws_tickets import internal_ws_credential

        qs = urllib.parse.urlencode({"internal": internal_ws_credential()})
    else:
        qs = urllib.parse.urlencode({"token": _SESSION_TOKEN})

    return f"ws://{netloc}/api/ws?{qs}"


async def _resolve_chat_argv_async(
    resume: Optional[str] = None,
    sidecar_url: Optional[str] = None,
    profile: Optional[str] = None,
    active_session_file: Optional[str] = None,
) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve chat argv without blocking the dashboard event loop.

    ``_resolve_chat_argv`` may run ``npm install`` / ``npm run build`` through
    ``_make_tui_argv``.  Keep that synchronous work off the WebSocket event
    loop so reverse proxies and existing dashboard connections can continue
    to exchange keepalives while the TUI launch command is prepared.  The
    async lock preserves the previous one-build-at-a-time behavior when
    multiple browser tabs connect at once without occupying worker threads
    while queued connections wait.
    """
    kwargs = {
        "resume": resume,
        "sidecar_url": sidecar_url,
        "profile": profile,
    }
    if active_session_file is not None:
        kwargs["active_session_file"] = active_session_file

    async with _get_chat_argv_lock(app):
        return await asyncio.to_thread(
            _resolve_chat_argv,
            **kwargs,
        )


def _build_sidecar_url(channel: str) -> Optional[str]:
    """ws:// URL the PTY child should publish events to, or None when unbound.

    Loopback / ``--insecure``: uses ``?token=<_SESSION_TOKEN>``.

    Gated mode: authenticates with the process-lifetime internal credential
    (``?internal=``), the same one ``_build_gateway_ws_url`` uses. The PTY
    child is a server-spawned process we trust; the credential is multi-use
    and never expires, so the child can reconnect ``/api/pub`` without a new
    URL. (This previously minted a single-use 30s ticket, which meant the
    child could not reconnect and could miss the window on a slow cold boot.)
    Connections authenticated this way are recorded under the
    ``server-internal`` identity in the audit log.
    """
    host = _resolve_client_ws_host()
    port = getattr(app.state, "bound_port", None)

    if not host or not port:
        return None

    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"

    if getattr(app.state, "auth_required", False):
        # Gated mode — use the internal credential so the WS upgrade survives
        # _ws_auth_ok and the child can reconnect.
        from hermes_cli.dashboard_auth.ws_tickets import internal_ws_credential

        qs = urllib.parse.urlencode(
            {"internal": internal_ws_credential(), "channel": channel}
        )
    else:
        qs = urllib.parse.urlencode({"token": _SESSION_TOKEN, "channel": channel})

    return f"ws://{netloc}/api/pub?{qs}"


async def _broadcast_event(app: Any, channel: str, payload: str) -> None:
    """Fan out one publisher frame to every subscriber on `channel`."""
    event_channels, event_lock = _get_event_state(app)
    async with event_lock:
        subs = list(event_channels.get(channel, ()))

    for sub in subs:
        try:
            await sub.send_text(payload)
        except Exception:
            # Subscriber went away mid-send; the /api/events finally clause
            # will remove it from the registry on its next iteration.
            _log.warning("broadcast send failed for subscriber on %s", channel, exc_info=True)


def _channel_or_close_code(ws: WebSocket) -> Optional[str]:
    """Return the channel id from the query string or None if invalid."""
    channel = ws.query_params.get("channel", "")

    return channel if _VALID_CHANNEL_RE.match(channel) else None


def _active_session_file_for_channel(app: "FastAPI", channel: str) -> Path:
    """Return the per-channel file where a dashboard TUI writes its active sid."""
    files = _get_pty_active_session_files(app)
    existing = files.get(channel)
    if existing is not None:
        return existing

    fd, raw_path = tempfile.mkstemp(prefix="hermes-pty-active-", suffix=".json")
    os.close(fd)
    path = Path(raw_path)
    files[channel] = path
    return path


def _read_active_session_file(path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    session_id = str(data.get("session_id") or "").strip()
    return session_id or None


def _forget_active_session_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _ws_close_reason(text: str) -> str:
    """Clamp a WS close reason to the protocol's 123-byte UTF-8 limit.

    RFC 6455 caps the close-frame reason at 123 bytes; uvicorn raises if a
    longer string is passed. Our reasons embed an attacker-controlled origin,
    so truncate defensively rather than crash the close handler.
    """
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= 123:
        return text
    return encoded[:120].decode("utf-8", "ignore") + "..."


# ---------------------------------------------------------------------------
# /api/console — safe Hermes Console command WebSocket.
#
# Unlike /api/pty, this endpoint never spawns a PTY, shell, or full Hermes CLI
# subprocess. It runs the curated console engine in-process and exchanges
# structured JSON frames with the dashboard xterm overlay.
# ---------------------------------------------------------------------------

_CONSOLE_PROMPT = "hermes> "
_CONSOLE_COMMAND_TIMEOUT_SECONDS = 60.0
_CONSOLE_OUTPUT_LIMIT = 50000

# Console commands run in a worker thread. On a timeout, asyncio.wait_for cancels
# the *awaitable*, but Python threads aren't preemptible, so a genuinely stuck
# worker keeps running to completion. To keep that from exhausting the shared
# default thread pool (asyncio.to_thread), we run console commands on a small
# dedicated, bounded pool: a leaked worker is capped, and concurrent console
# execution is bounded to a fixed number of threads regardless of reconnects.
_CONSOLE_EXECUTOR_MAX_WORKERS = 4
_console_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_console_executor_lock = threading.Lock()


def _get_console_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily create the bounded console worker pool (once per process)."""
    global _console_executor
    if _console_executor is None:
        with _console_executor_lock:
            if _console_executor is None:
                _console_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_CONSOLE_EXECUTOR_MAX_WORKERS,
                    thread_name_prefix="hermes-console",
                )
                # Ensure the pool is torn down on interpreter exit. Don't wait on
                # in-flight workers: a stuck 60s console command must not block
                # shutdown (cancel_futures drops anything not yet started).
                atexit.register(
                    lambda: _console_executor
                    and _console_executor.shutdown(wait=False, cancel_futures=True)
                )
    return _console_executor


def _dashboard_console_context() -> str:
    """Choose local vs hosted command policy for the dashboard console."""
    return "hosted" if _default_hermes_root_is_opt_data() else "local"


def _console_profile_from_ws(ws: WebSocket) -> Optional[str]:
    profile = (ws.query_params.get("profile") or "").strip()
    return profile or None


def _execute_console_line(
    engine: Any,
    line: str,
    *,
    confirmed: bool,
    profile: Optional[str],
) -> Any:
    # _profile_scope swaps process-global skill module paths; keep it inside
    # the worker thread and never hold it across awaits.
    with _profile_scope(profile):
        return engine.execute(line, confirmed=confirmed)


async def _console_send(
    ws: WebSocket,
    send_lock: asyncio.Lock,
    payload: Dict[str, Any],
) -> None:
    async with send_lock:
        await ws.send_json(payload)


async def _console_send_result(
    ws: WebSocket,
    send_lock: asyncio.Lock,
    result: Any,
    *,
    command_id: int,
) -> None:
    command = result.command or ""
    status = result.status
    if status == "ok":
        if result.output:
            await _console_send(
                ws,
                send_lock,
                {
                    "type": "output",
                    "id": command_id,
                    "stream": "stdout",
                    "data": result.output,
                    "command": command,
                },
            )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "ok",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "error":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "id": command_id,
                "message": result.output or "Command failed.",
                "command": command,
            },
        )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "error",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "confirm_required":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "confirm_required",
                "id": command_id,
                "command": command,
                "message": result.confirmation_message or f"Run `{command}`?",
                "prompt": _CONSOLE_PROMPT,
            },
        )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "confirm_required",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "clear":
        await _console_send(ws, send_lock, {"type": "clear", "id": command_id})
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "clear",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "exit":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "exit",
                "command": command,
                "prompt": "",
            },
        )
        return

    await _console_send(
        ws,
        send_lock,
        {
            "type": "error",
            "id": command_id,
            "message": f"Unknown console result status: {status}",
            "command": command,
        },
    )


def _console_json_payload(msg: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    raw: str | bytes | None = msg.get("text")
    if raw is None:
        raw = msg.get("bytes")
    if raw is None:
        return None, None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "Console frames must be UTF-8 JSON."
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, "Console frames must be JSON objects."
    if not isinstance(payload, dict):
        return None, "Console frames must be JSON objects."
    return payload, None


@app.websocket("/api/console")
async def console_ws(ws: WebSocket) -> None:
    peer = ws.client.host if ws.client else "?"

    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        _log.info("console refused: embedded chat disabled peer=%s", peer)
        await ws.close(code=4404, reason="embedded chat disabled")
        return

    auth_reason, cred = _ws_auth_reason(ws)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning(
            "console auth rejected reason=%s mode=%s cred=%s peer=%s",
            auth_reason, mode, cred, peer,
        )
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return

    host_origin_reason = _ws_host_origin_reason(ws)
    if host_origin_reason is not None:
        _log.warning("console refused: %s peer=%s", host_origin_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(host_origin_reason))
        return

    client_reason = _ws_client_reason(ws)
    if client_reason is not None:
        _log.warning("console refused: %s", client_reason)
        await ws.close(code=4408, reason=_ws_close_reason(client_reason))
        return

    await ws.accept()

    profile = _console_profile_from_ws(ws)
    context = _dashboard_console_context()
    send_lock = asyncio.Lock()

    try:
        from hermes_cli.console_engine import HermesConsoleEngine

        engine = HermesConsoleEngine(
            output_limit=_CONSOLE_OUTPUT_LIMIT,
            context=context,  # type: ignore[arg-type]
        )
        if profile and profile.lower() != "current":
            _resolve_profile_dir(profile)
    except HTTPException as exc:
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "message": str(exc.detail),
                "prompt": "",
            },
        )
        await ws.close(code=4400, reason=_ws_close_reason(str(exc.detail)))
        return
    except Exception as exc:
        _log.exception("console failed to initialize")
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "message": f"Console unavailable: {exc}",
                "prompt": "",
            },
        )
        await ws.close(code=1011)
        return

    _log.info(
        "console accepted peer=%s mode=%s cred=%s context=%s profile=%s",
        peer,
        mode,
        cred,
        context,
        profile or "current",
    )
    await _console_send(
        ws,
        send_lock,
        {
            "type": "ready",
            "context": context,
            "profile": profile or "current",
            "prompt": _CONSOLE_PROMPT,
        },
    )

    active_task: asyncio.Task | None = None
    pending_confirmation: Optional[str] = None
    command_generation = 0

    async def run_command(line: str, *, confirmed: bool, command_id: int) -> None:
        nonlocal active_task, pending_confirmation, command_generation
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _get_console_executor(),
                    functools.partial(
                        _execute_console_line,
                        engine,
                        line,
                        confirmed=confirmed,
                        profile=profile,
                    ),
                ),
                timeout=_CONSOLE_COMMAND_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            if command_id == command_generation:
                pending_confirmation = None
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "id": command_id,
                        "message": (
                            "Command timed out. Hermes Console returned to the prompt."
                        ),
                        "command": line,
                    },
                )
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "complete",
                        "id": command_id,
                        "status": "timeout",
                        "command": line,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
        except Exception as exc:
            if command_id == command_generation:
                pending_confirmation = None
                _log.exception("console command failed")
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "id": command_id,
                        "message": str(exc) or exc.__class__.__name__,
                        "command": line,
                    },
                )
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "complete",
                        "id": command_id,
                        "status": "error",
                        "command": line,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
        else:
            if command_id != command_generation:
                return
            pending_confirmation = (
                result.command if result.status == "confirm_required" else None
            )
            await _console_send_result(
                ws,
                send_lock,
                result,
                command_id=command_id,
            )
            if result.status == "exit":
                await ws.close(code=1000)
        finally:
            if command_id == command_generation:
                active_task = None

    async def start_command(line: str, *, confirmed: bool = False) -> None:
        nonlocal active_task, command_generation
        command_generation += 1
        command_id = command_generation
        active_task = asyncio.create_task(
            run_command(line, confirmed=confirmed, command_id=command_id)
        )

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                break
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break

            payload, error = _console_json_payload(msg)
            if error:
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "message": error,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue
            if payload is None:
                continue

            frame_type = str(payload.get("type") or "").strip().lower()
            if frame_type == "ping":
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "pong",
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue

            if frame_type == "cancel":
                if active_task and not active_task.done():
                    command_generation += 1
                    active_task.cancel()
                    active_task = None
                    pending_confirmation = None
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "cancelled",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                elif pending_confirmation:
                    pending_confirmation = None
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "cancelled",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                else:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "idle",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                continue

            if active_task and not active_task.done():
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "message": "A console command is already running.",
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue

            if frame_type == "confirm":
                command = str(payload.get("command") or pending_confirmation or "").strip()
                if not pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": "No command is waiting for confirmation.",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                if command != pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Confirmation does not match the pending command.",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                pending_confirmation = None
                await start_command(command, confirmed=True)
                continue

            if frame_type in {"input", "command"}:
                line = str(payload.get("line") or payload.get("command") or "").strip()
                if not line:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "ok",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                if pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": (
                                "Confirm or cancel the pending command before "
                                "running another one."
                            ),
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                await start_command(line)
                continue

            await _console_send(
                ws,
                send_lock,
                {
                    "type": "error",
                    "message": f"Unsupported console frame: {frame_type or '?'}",
                    "prompt": _CONSOLE_PROMPT,
                },
            )
    except WebSocketDisconnect:
        pass
    finally:
        if active_task and not active_task.done():
            active_task.cancel()
            try:
                await active_task
            except (asyncio.CancelledError, Exception):
                pass


@app.websocket("/api/pty")
async def pty_ws(ws: WebSocket) -> None:
    peer = ws.client.host if ws.client else "?"

    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        _log.info("pty refused: embedded chat disabled peer=%s", peer)
        await ws.close(code=4404, reason="embedded chat disabled")
        return

    # --- auth + host/origin/peer check (before accept so we can close
    #     cleanly AND tell the client WHY via the close code + reason).
    #     Each gate maps to a distinct close code so the log and the
    #     browser banner agree on the cause:
    #       4401 bad credential   4403 host/origin mismatch
    #       4408 peer not allowed  4404 chat disabled
    auth_reason, cred = _ws_auth_reason(ws)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning(
            "pty auth rejected reason=%s mode=%s cred=%s peer=%s",
            auth_reason, mode, cred, peer,
        )
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return

    host_origin_reason = _ws_host_origin_reason(ws)
    if host_origin_reason is not None:
        _log.warning("pty refused: %s peer=%s", host_origin_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(host_origin_reason))
        return

    client_reason = _ws_client_reason(ws)
    if client_reason is not None:
        _log.warning("pty refused: %s", client_reason)
        await ws.close(code=4408, reason=_ws_close_reason(client_reason))
        return

    await ws.accept()
    _log.info("pty accepted peer=%s mode=%s cred=%s", peer, mode, cred)

    # On native Windows, the POSIX PTY bridge can't be imported.  Tell the
    # client and close cleanly rather than pretending the feature works.
    if not _PTY_BRIDGE_AVAILABLE:
        await ws.send_text(
            "\r\n\x1b[31mChat unavailable: the embedded terminal requires a "
            "POSIX PTY, which native Windows Python doesn't provide.\x1b[0m\r\n"
            "\x1b[33mInstall Hermes inside WSL2 to use the dashboard's /chat "
            "tab — the rest of the dashboard works here.\x1b[0m\r\n"
        )
        await ws.close(code=1011)
        return

    # --- spawn PTY ------------------------------------------------------
    resume = ws.query_params.get("resume") or None
    profile = ws.query_params.get("profile") or None
    channel = _channel_or_close_code(ws)
    sidecar_url = _build_sidecar_url(channel) if channel else None
    force_fresh = (ws.query_params.get("fresh") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    active_session_file: Optional[Path] = None

    if channel:
        active_session_file = _active_session_file_for_channel(ws.app, channel)
        if force_fresh:
            resume = None
            _forget_active_session_file(active_session_file)
        elif not resume:
            resume = _read_active_session_file(active_session_file)

    resolve_kwargs = {
        "resume": resume,
        "sidecar_url": sidecar_url,
        "profile": profile,
    }
    if active_session_file is not None:
        resolve_kwargs["active_session_file"] = str(active_session_file)

    try:
        argv, cwd, env = await _resolve_chat_argv_async(**resolve_kwargs)
    except HTTPException as exc:
        # Unknown/invalid profile from _resolve_profile_dir.
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc.detail}\x1b[0m\r\n")
        await ws.close(code=1011)
        return
    except SystemExit as exc:
        # _make_tui_argv calls sys.exit(1) when node/npm is missing.
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return


    attach_token = ws.query_params.get("attach") or None

    def _spawn():
        return PtyBridge.spawn(argv, cwd=cwd, env=env)

    if attach_token is None:
        # Legacy path: 1:1 socket<->PTY, killed on disconnect (unchanged).
        try:
            bridge = _spawn()
        except PtyUnavailableError as exc:
            await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
            await ws.close(code=1011)
            return
        except (FileNotFoundError, OSError) as exc:
            await ws.send_text(f"\r\n\x1b[31mChat failed to start: {exc}\x1b[0m\r\n")
            await ws.close(code=1011)
            return
        await _legacy_pump(ws, bridge)
        return

    # Keep-alive path: the PTY outlives this socket; reattach by token.
    try:
        session, _created = await PTY_REGISTRY.attach_or_spawn(
            attach_token, spawn=_spawn
        )
    except PtyUnavailableError as exc:
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return
    except (FileNotFoundError, OSError, RegistryFull) as exc:
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return

    await session.attach(ws)

    # --- writer loop: WebSocket → PTY master ----------------------------
    # No reader task here: the session's drain task (spawned once per PTY,
    # inside the registry) forwards PTY output to whichever socket is
    # attached and rings-buffers it while detached.  On child EOF the drain
    # closes the attached socket with 4410, which unparks ``ws.receive()``
    # below — same half-open-socket protection the legacy pump has (#54028).
    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                # ws.receive() after the socket is already disconnected
                # (e.g. closed by the drain task on process exit).
                break
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is None:
                text = msg.get("text")
                raw = text.encode("utf-8") if isinstance(text, str) else b""
            if not raw:
                continue

            # Resize escape is consumed locally, never written to the PTY.
            match = _RESIZE_RE.match(raw)
            if match and match.end() == len(raw):
                session.bridge.resize(cols=int(match.group(1)), rows=int(match.group(2)))
                continue

            session.bridge.write(raw)
    except WebSocketDisconnect:
        pass
    finally:
        # Detach only — the PTY keeps running for a reattach; the registry
        # reaper closes it after the TTL (or immediately on process exit).
        PTY_REGISTRY.detach(attach_token, ws)


# ---------------------------------------------------------------------------
# /api/ws — JSON-RPC WebSocket sidecar for the dashboard "Chat" tab.
#
# Drives the same `tui_gateway.dispatch` surface Ink uses over stdio, so the
# dashboard can render structured metadata (model badge, tool-call sidebar,
# slash launcher, session info) alongside the xterm.js terminal that PTY
# already paints. Both transports bind to the same session id when one is
# active, so a tool.start emitted by the agent fans out to both sinks.
# ---------------------------------------------------------------------------


@app.websocket("/api/ws")
async def gateway_ws(ws: WebSocket) -> None:
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    from tui_gateway.ws import handle_ws

    await handle_ws(ws)


# ---------------------------------------------------------------------------
# /api/pub + /api/events — chat-tab event broadcast.
#
# The PTY-side ``tui_gateway.entry`` opens /api/pub at startup (driven by
# HERMES_TUI_SIDECAR_URL set in /api/pty's PTY env) and writes every
# dispatcher emit through it.  The dashboard fans those frames out to any
# subscriber that opened /api/events on the same channel id.  This is what
# gives the React sidebar its tool-call feed without breaking the PTY
# child's stdio handshake with Ink.
# ---------------------------------------------------------------------------


@app.websocket("/api/pub")
async def pub_ws(ws: WebSocket) -> None:
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    channel = _channel_or_close_code(ws)
    if not channel:
        await ws.close(code=4400)
        return

    await ws.accept()

    try:
        while True:
            await _broadcast_event(ws.app, channel, await ws.receive_text())
    except WebSocketDisconnect:
        pass


@app.websocket("/api/events")
async def events_ws(ws: WebSocket) -> None:
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    channel = _channel_or_close_code(ws)
    if not channel:
        await ws.close(code=4400)
        return

    await ws.accept()

    event_channels, event_lock = _get_event_state(ws.app)
    async with event_lock:
        event_channels.setdefault(channel, set()).add(ws)

    try:
        while True:
            # Subscribers don't speak — the receive() just blocks until
            # disconnect so the connection stays open as long as the
            # browser holds it.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with event_lock:
            subs = event_channels.get(channel)

            if subs is not None:
                subs.discard(ws)

                if not subs:
                    event_channels.pop(channel, None)


def _normalise_prefix(raw: Optional[str]) -> str:
    """Normalise an X-Forwarded-Prefix header value.

    Thin re-export of :func:`hermes_cli.dashboard_auth.prefix.normalise_prefix`
    — the single source of truth lives in the dashboard_auth package so
    the gate middleware, the OAuth routes, the cookie helpers, and the
    SPA mount all agree on validation rules.
    """
    from hermes_cli.dashboard_auth.prefix import normalise_prefix
    return normalise_prefix(raw)


def mount_spa(application: FastAPI):
    """Mount the built SPA. Falls back to index.html for client-side routing.

    The session token is injected into index.html via a ``<script>`` tag so
    the SPA can authenticate against protected API endpoints without a
    separate (unauthenticated) token-dispensing endpoint.

    When served behind a path-prefix reverse proxy (e.g.
    ``mission-control.tilos.com/hermes/*`` -> local Caddy -> :9119), the
    proxy injects ``X-Forwarded-Prefix: /hermes`` on every request. We
    rewrite the served ``index.html`` so absolute asset URLs (``/assets/...``)
    and the SPA's runtime ``__HERMES_BASE_PATH__`` honour that prefix
    without rebuilding the bundle.
    """
    # `hermes serve` is the headless backend: it must NEVER serve the browser
    # SPA, even if a dist is lying around from a prior `dashboard`/build. Take
    # the no-frontend path so only the JSON-RPC/WS/API surface is reachable.
    _headless = os.environ.get("HERMES_SERVE_HEADLESS") == "1"
    if _headless or not WEB_DIST.exists():
        _msg = (
            "Headless backend (hermes serve): web UI disabled — use "
            "`hermes dashboard` for the browser UI."
            if _headless
            else "Frontend not built. Run: cd web && npm run build"
        )

        @application.get("/{full_path:path}")
        async def no_frontend(full_path: str):
            return JSONResponse({"error": _msg}, status_code=404)
        return

    _index_path = WEB_DIST / "index.html"

    def _serve_index(prefix: str = ""):
        """Return index.html with the session token + base-path injected.

        ``prefix`` is the normalised ``X-Forwarded-Prefix`` (e.g. ``/hermes``)
        or empty string when served at root.

        When the OAuth auth gate is active (``app.state.auth_required``),
        the legacy ``_SESSION_TOKEN`` is NOT injected — the SPA reads
        identity from ``/api/auth/me`` over cookie auth instead.  The
        ``__HERMES_AUTH_REQUIRED__`` flag lets the SPA pick the right
        auth scheme for /api/pty and /api/ws (ticket vs token).
        """
        html = _index_path.read_text(encoding="utf-8")
        chat_js = "true" if _DASHBOARD_EMBEDDED_CHAT_ENABLED else "false"
        gated = bool(getattr(app.state, "auth_required", False))
        gated_js = "true" if gated else "false"
        if gated:
            bootstrap_script = (
                f"<script>"
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__={gated_js};"
                f"</script>"
            )
        else:
            bootstrap_script = (
                f'<script>window.__HERMES_SESSION_TOKEN__="{_SESSION_TOKEN}";'
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__={gated_js};"
                f"</script>"
            )
        if prefix:
            # Rewrite absolute asset URLs baked into the Vite build so the
            # browser fetches them through the same proxy prefix.
            html = html.replace('href="/assets/', f'href="{prefix}/assets/')
            html = html.replace('src="/assets/', f'src="{prefix}/assets/')
            html = html.replace('href="/favicon.ico"', f'href="{prefix}/favicon.ico"')
            html = html.replace('href="/fonts/', f'href="{prefix}/fonts/')
            html = html.replace('href="/ds-assets/', f'href="{prefix}/ds-assets/')
            html = html.replace('src="/ds-assets/', f'src="{prefix}/ds-assets/')
        html = html.replace("</head>", f"{bootstrap_script}</head>", 1)
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    # When served behind a path-prefix proxy, the built CSS contains
    # absolute ``url(/fonts/...)`` and ``url(/ds-assets/...)`` references.
    # Browsers resolve those against the document origin, which means
    # under ``/hermes`` they'd hit ``mission-control.tilos.com/fonts/...``
    # (the MC Pages app), not the Hermes backend. Intercept CSS asset
    # requests BEFORE the StaticFiles mount and rewrite the absolute paths
    # when a prefix is in play.
    @application.get("/assets/{filename}.css")
    async def serve_css(filename: str, request: Request):
        css_path = WEB_DIST / "assets" / f"{filename}.css"
        if not css_path.is_file() or not css_path.resolve().is_relative_to(
            WEB_DIST.resolve()
        ):
            return JSONResponse({"error": "not found"}, status_code=404)
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        css = css_path.read_text(encoding="utf-8")
        if prefix:
            for asset_dir in ("/fonts/", "/fonts-terminal/", "/ds-assets/", "/assets/"):
                css = css.replace(f"url({asset_dir}", f"url({prefix}{asset_dir}")
                css = css.replace(f"url(\"{asset_dir}", f"url(\"{prefix}{asset_dir}")
                css = css.replace(f"url('{asset_dir}", f"url('{prefix}{asset_dir}")
        return Response(content=css, media_type="text/css")

    application.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @application.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        # An unmatched /api/* path is a missing/renamed endpoint, NOT a
        # client-side route. Falling through to index.html here returns
        # `<!doctype html>` with status 200, which makes JSON clients (the
        # desktop app's fetchJson, dashboard fetch wrappers) blow up with an
        # opaque `SyntaxError: Unexpected token '<'`. Return a real 404 JSON
        # so the caller sees a clear "no such endpoint" instead.
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse(
                {"detail": f"No such API endpoint: /{full_path}"},
                status_code=404,
            )
        file_path = WEB_DIST / full_path
        # Prevent path traversal via url-encoded sequences (%2e%2e/)
        if (
            full_path
            and file_path.resolve().is_relative_to(WEB_DIST.resolve())
            and file_path.exists()
            and file_path.is_file()
        ):
            return FileResponse(file_path)
        return _serve_index(prefix)


# ---------------------------------------------------------------------------
# Dashboard theme endpoints
# ---------------------------------------------------------------------------

# Built-in dashboard themes — label + description only.  The actual color
# definitions live in the frontend (web/src/themes/presets.ts).
_BUILTIN_DASHBOARD_THEMES = [
    {"name": "default",       "label": "Hermes Teal",         "description": "Classic dark teal — the canonical Hermes look"},
    {"name": "default-large", "label": "Hermes Teal (Large)", "description": "Hermes Teal with bigger fonts and roomier spacing"},
    {"name": "nous-blue",     "label": "Nous Blue",           "description": "Light mode — vivid Nous-blue accents on cream canvas"},
    {"name": "midnight",      "label": "Midnight",            "description": "Deep blue-violet with cool accents"},
    {"name": "ember",     "label": "Ember",          "description": "Warm crimson and bronze — forge vibes"},
    {"name": "mono",      "label": "Mono",           "description": "Clean grayscale — minimal and focused"},
    {"name": "cyberpunk", "label": "Cyberpunk",      "description": "Neon green on black — matrix terminal"},
    {"name": "rose",      "label": "Rosé",           "description": "Soft pink and warm ivory — easy on the eyes"},
]


def _parse_theme_layer(value: Any, default_hex: str, default_alpha: float = 1.0) -> Optional[Dict[str, Any]]:
    """Normalise a theme layer spec from YAML into `{hex, alpha}` form.

    Accepts shorthand (a bare hex string) or full dict form.  Returns
    ``None`` on garbage input so the caller can fall back to a built-in
    default rather than blowing up.
    """
    if value is None:
        return {"hex": default_hex, "alpha": default_alpha}
    if isinstance(value, str):
        return {"hex": value, "alpha": default_alpha}
    if isinstance(value, dict):
        hex_val = value.get("hex", default_hex)
        alpha_val = value.get("alpha", default_alpha)
        if not isinstance(hex_val, str):
            return None
        try:
            alpha_f = float(alpha_val)
        except (TypeError, ValueError):
            alpha_f = default_alpha
        return {"hex": hex_val, "alpha": max(0.0, min(1.0, alpha_f))}
    return None


_THEME_DEFAULT_TYPOGRAPHY: Dict[str, str] = {
    "fontSans": 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    "fontMono": 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace',
    "baseSize": "15px",
    "lineHeight": "1.55",
    "letterSpacing": "0",
}

_THEME_DEFAULT_LAYOUT: Dict[str, str] = {
    "radius": "0.5rem",
    "density": "comfortable",
}

_THEME_OVERRIDE_KEYS = {
    "card", "cardForeground", "popover", "popoverForeground",
    "primary", "primaryForeground", "secondary", "secondaryForeground",
    "muted", "mutedForeground", "accent", "accentForeground",
    "destructive", "destructiveForeground", "success", "warning",
    "border", "input", "ring",
}

# Well-known named asset slots themes can populate.  Any other keys under
# ``assets.custom`` are exposed as ``--theme-asset-custom-<key>`` CSS vars
# for plugin/shell use.
_THEME_NAMED_ASSET_KEYS = {"bg", "hero", "logo", "crest", "sidebar", "header"}

# Component-style buckets themes can override.  The value under each bucket
# is a mapping from camelCase property name to CSS string; each pair emits
# ``--component-<bucket>-<kebab-property>`` on :root.  The frontend's shell
# components (Card, App header, Backdrop, etc.) consume these vars so themes
# can restyle chrome (clip-path, border-image, segmented progress, etc.)
# without shipping their own CSS.
_THEME_COMPONENT_BUCKETS = {
    "card", "header", "footer", "sidebar", "tab",
    "progress", "badge", "backdrop", "page",
}

_THEME_LAYOUT_VARIANTS = {"standard", "cockpit", "tiled"}

# Cap on customCSS length so a malformed/oversized theme YAML can't blow up
# the response payload or the <style> tag.  32 KiB is plenty for every
# practical reskin (the Strike Freedom demo is ~2 KiB).
_THEME_CUSTOM_CSS_MAX = 32 * 1024


def _normalise_theme_definition(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise a user theme YAML into the wire format `ThemeProvider`
    expects.  Returns ``None`` if the theme is unusable.

    Accepts both the full schema (palette/typography/layout) and a loose
    form with bare hex strings, so hand-written YAMLs stay friendly.
    """
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    # Palette
    palette_src = data.get("palette", {}) if isinstance(data.get("palette"), dict) else {}
    # Allow top-level `colors.background` as a shorthand too.
    colors_src = data.get("colors", {}) if isinstance(data.get("colors"), dict) else {}

    def _layer(key: str, default_hex: str, default_alpha: float = 1.0) -> Dict[str, Any]:
        spec = palette_src.get(key, colors_src.get(key))
        parsed = _parse_theme_layer(spec, default_hex, default_alpha)
        return parsed if parsed is not None else {"hex": default_hex, "alpha": default_alpha}

    palette = {
        "background": _layer("background", "#041c1c", 1.0),
        "midground": _layer("midground", "#ffe6cb", 1.0),
        "foreground": _layer("foreground", "#ffffff", 0.0),
        "warmGlow": palette_src.get("warmGlow") or data.get("warmGlow") or "rgba(255, 189, 56, 0.35)",
        "noiseOpacity": 1.0,
    }
    raw_noise = palette_src.get("noiseOpacity", data.get("noiseOpacity"))
    try:
        palette["noiseOpacity"] = float(raw_noise) if raw_noise is not None else 1.0
    except (TypeError, ValueError):
        palette["noiseOpacity"] = 1.0

    # Typography
    typo_src = data.get("typography", {}) if isinstance(data.get("typography"), dict) else {}
    typography = dict(_THEME_DEFAULT_TYPOGRAPHY)
    for key in ("fontSans", "fontMono", "fontDisplay", "fontUrl", "baseSize", "lineHeight", "letterSpacing"):
        val = typo_src.get(key)
        if isinstance(val, str) and val.strip():
            typography[key] = val

    # Layout
    layout_src = data.get("layout", {}) if isinstance(data.get("layout"), dict) else {}
    layout = dict(_THEME_DEFAULT_LAYOUT)
    radius = layout_src.get("radius")
    if isinstance(radius, str) and radius.strip():
        layout["radius"] = radius
    density = layout_src.get("density")
    if isinstance(density, str) and density in {"compact", "comfortable", "spacious"}:
        layout["density"] = density

    # Color overrides — keep only valid keys with string values.
    overrides_src = data.get("colorOverrides", {})
    color_overrides: Dict[str, str] = {}
    if isinstance(overrides_src, dict):
        for key, val in overrides_src.items():
            if key in _THEME_OVERRIDE_KEYS and isinstance(val, str) and val.strip():
                color_overrides[key] = val

    # Assets — named slots + arbitrary user-defined keys.  Values must be
    # strings (URLs or CSS ``url(...)``/``linear-gradient(...)`` expressions).
    # We don't fetch remote assets here; the frontend just injects them as
    # CSS vars.  Empty values are dropped so a theme can explicitly clear a
    # slot by setting ``hero: ""``.
    assets_out: Dict[str, Any] = {}
    assets_src = data.get("assets", {}) if isinstance(data.get("assets"), dict) else {}
    for key in _THEME_NAMED_ASSET_KEYS:
        val = assets_src.get(key)
        if isinstance(val, str) and val.strip():
            assets_out[key] = val
    custom_assets_src = assets_src.get("custom")
    if isinstance(custom_assets_src, dict):
        custom_assets: Dict[str, str] = {}
        for key, val in custom_assets_src.items():
            if (
                isinstance(key, str)
                and key.replace("-", "").replace("_", "").isalnum()
                and isinstance(val, str)
                and val.strip()
            ):
                custom_assets[key] = val
        if custom_assets:
            assets_out["custom"] = custom_assets

    # Custom CSS — raw CSS text the frontend injects as a scoped <style>
    # tag on theme apply.  Clipped to _THEME_CUSTOM_CSS_MAX to keep the
    # payload bounded.  We intentionally do NOT parse/sanitise the CSS
    # here — the dashboard is localhost-only and themes are user-authored
    # YAML in ~/.hermes/, same trust level as the config file itself.
    custom_css_val = data.get("customCSS")
    custom_css: Optional[str] = None
    if isinstance(custom_css_val, str) and custom_css_val.strip():
        custom_css = custom_css_val[:_THEME_CUSTOM_CSS_MAX]

    # Component style overrides — per-bucket dicts of camelCase CSS
    # property -> CSS string.  The frontend converts these into CSS vars
    # that shell components (Card, App header, Backdrop) consume.
    component_styles_src = data.get("componentStyles", {})
    component_styles: Dict[str, Dict[str, str]] = {}
    if isinstance(component_styles_src, dict):
        for bucket, props in component_styles_src.items():
            if bucket not in _THEME_COMPONENT_BUCKETS or not isinstance(props, dict):
                continue
            clean: Dict[str, str] = {}
            for prop, value in props.items():
                if (
                    isinstance(prop, str)
                    and prop.replace("-", "").replace("_", "").isalnum()
                    and isinstance(value, (str, int, float))
                    and str(value).strip()
                ):
                    clean[prop] = str(value)
            if clean:
                component_styles[bucket] = clean

    layout_variant_src = data.get("layoutVariant")
    layout_variant = (
        layout_variant_src
        if isinstance(layout_variant_src, str) and layout_variant_src in _THEME_LAYOUT_VARIANTS
        else "standard"
    )

    result: Dict[str, Any] = {
        "name": name,
        "label": data.get("label") or name,
        "description": data.get("description", ""),
        "palette": palette,
        "typography": typography,
        "layout": layout,
        "layoutVariant": layout_variant,
    }
    if color_overrides:
        result["colorOverrides"] = color_overrides
    if assets_out:
        result["assets"] = assets_out
    if custom_css is not None:
        result["customCSS"] = custom_css
    if component_styles:
        result["componentStyles"] = component_styles
    return result


def _discover_user_themes() -> list:
    """Scan ~/.hermes/dashboard-themes/*.yaml for user-created themes.

    Returns a list of fully-normalised theme definitions ready to ship
    to the frontend, so the client can apply them without a secondary
    round-trip or a built-in stub.
    """
    themes_dir = get_hermes_home() / "dashboard-themes"
    if not themes_dir.is_dir():
        return []
    result = []
    for f in sorted(themes_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        normalised = _normalise_theme_definition(data)
        if normalised is not None:
            result.append(normalised)
    return result


@app.get("/api/dashboard/themes")
async def get_dashboard_themes():
    """Return available themes and the currently active one.

    Built-in entries ship name/label/description only (the frontend owns
    their full definitions in `web/src/themes/presets.ts`).  User themes
    from `~/.hermes/dashboard-themes/*.yaml` ship with their full
    normalised definition under `definition`, so the client can apply
    them without a stub.
    """
    config = load_config()
    active = cfg_get(config, "dashboard", "theme", default="default")
    user_themes = _discover_user_themes()
    seen = set()
    themes = []
    for t in _BUILTIN_DASHBOARD_THEMES:
        seen.add(t["name"])
        themes.append(t)
    for t in user_themes:
        if t["name"] in seen:
            continue
        themes.append({
            "name": t["name"],
            "label": t["label"],
            "description": t["description"],
            "definition": t,
        })
        seen.add(t["name"])
    return {"themes": themes, "active": active}


class ThemeSetBody(BaseModel):
    name: str


@app.put("/api/dashboard/theme")
async def set_dashboard_theme(body: ThemeSetBody):
    """Set the active dashboard theme (persists to config.yaml)."""
    config = load_config()
    if "dashboard" not in config:
        config["dashboard"] = {}
    config["dashboard"]["theme"] = body.name
    save_config(config)
    return {"ok": True, "theme": body.name}


# Curated font-override ids. Kept in sync with FONT_CHOICES in
# web/src/themes/fonts.ts — the frontend owns the stacks + webfont URLs;
# the backend only needs the id allow-list so it can reject anything not
# in the vetted catalog (the font's webfont URL is injected as a <link>,
# so we never accept an arbitrary user-supplied id/URL here).
_FONT_DEFAULT_ID = "theme"
_FONT_CHOICES = frozenset({
    "system-sans", "system-serif", "system-mono",
    "inter", "ibm-plex-sans", "work-sans", "atkinson-hyperlegible", "dm-sans",
    "spectral", "fraunces", "source-serif",
    "jetbrains-mono", "ibm-plex-mono", "space-mono",
})


@app.get("/api/dashboard/font")
async def get_dashboard_font():
    """Return the active font override (``"theme"`` = use the theme's font)."""
    config = load_config()
    font = cfg_get(config, "dashboard", "font", default=_FONT_DEFAULT_ID)
    if font not in _FONT_CHOICES:
        font = _FONT_DEFAULT_ID
    return {"font": font}


class FontSetBody(BaseModel):
    font: str


@app.put("/api/dashboard/font")
async def set_dashboard_font(body: FontSetBody):
    """Set the dashboard font override (persists to config.yaml).

    Accepts any id in the curated catalog, or ``"theme"`` to clear the
    override and fall back to the active theme's own font. Unknown ids are
    coerced to ``"theme"`` rather than 400'd so a stale client can't wedge
    the picker.
    """
    font = body.font if body.font in _FONT_CHOICES else _FONT_DEFAULT_ID
    config = load_config()
    if "dashboard" not in config:
        config["dashboard"] = {}
    config["dashboard"]["font"] = font
    save_config(config)
    return {"ok": True, "font": font}


# ---------------------------------------------------------------------------
# Dashboard plugin system
# ---------------------------------------------------------------------------

def _safe_plugin_api_relpath(api_field: Any, *, dashboard_dir: Path) -> Optional[str]:
    """Validate the manifest's ``api`` field for the plugin loader.

    The web server later imports this file as a Python module via
    ``importlib.util.spec_from_file_location`` (arbitrary code
    execution by design — that's how plugins extend the backend).
    Pre-#29156 the field was used as-is, which meant:

    * An absolute path swallowed the plugin's dashboard directory
      entirely — ``Path('safe/dashboard') / '/tmp/evil.py'`` resolves
      to ``/tmp/evil.py``, so any attacker-controlled manifest could
      point the import at any Python file on disk (GHSA-5qr3-c538-wm9j).
    * A ``../..`` traversal could climb out of the plugin into
      neighbouring directories on the search path.

    Return the original string when the resolved path stays under
    ``dashboard_dir``; return ``None`` (with a warning logged at the
    call site) otherwise so the plugin still loads its static JS/CSS
    but its backend ``api`` is rejected.
    """
    if not isinstance(api_field, str) or not api_field.strip():
        return None
    candidate = Path(api_field)
    if candidate.is_absolute():
        return None
    try:
        resolved = (dashboard_dir / candidate).resolve()
        base = dashboard_dir.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return api_field


def _discover_dashboard_plugins() -> list:
    """Scan plugins/*/dashboard/manifest.json for dashboard extensions.

    Checks three plugin sources (same as hermes_cli.plugins):
    1. User plugins:    ~/.hermes/plugins/<name>/dashboard/manifest.json
    2. Bundled plugins: <repo>/plugins/<name>/dashboard/manifest.json  (memory/, etc.)
    3. Project plugins: ./.hermes/plugins/  (only if HERMES_ENABLE_PROJECT_PLUGINS)
    """
    plugins = []
    seen_names: set = set()

    from hermes_cli.plugins import get_bundled_plugins_dir
    bundled_root = get_bundled_plugins_dir()
    search_dirs = [
        (get_hermes_home() / "plugins", "user"),
        (bundled_root / "memory", "bundled"),
        (bundled_root, "bundled"),
    ]
    # GHSA-5qr3-c538-wm9j (#29156): the previous ``os.environ.get(...)``
    # check treated *any* non-empty string as truthy, so ``=0``, ``=false``,
    # and ``=no`` — all of which the agent loader and operators correctly
    # read as "disabled" — silently *enabled* the untrusted project source
    # in the web server.  Combined with the absolute-path RCE primitive on
    # the manifest's ``api`` field (now patched below), this turned the
    # opt-in into a sticky always-on switch.  Use the shared truthy
    # semantics (``1`` / ``true`` / ``yes`` / ``on``) so the gate matches
    # ``hermes_cli/plugins.py`` and the documented user contract.
    if env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        search_dirs.append((Path.cwd() / ".hermes" / "plugins", "project"))

    for plugins_root, source in search_dirs:
        if not plugins_root.is_dir():
            continue
        for child in sorted(plugins_root.iterdir()):
            if not child.is_dir():
                continue
            manifest_file = child / "dashboard" / "manifest.json"
            if not manifest_file.exists():
                continue
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                name = data.get("name", child.name)
                if name in seen_names:
                    continue
                seen_names.add(name)
                # Tab options: ``path`` + ``position`` for a new tab, optional
                # ``override`` to replace a built-in route, and ``hidden`` to
                # register the plugin component/slots without adding a tab
                # (useful for slot-only plugins like a header-crest injector).
                raw_tab = data.get("tab", {}) if isinstance(data.get("tab"), dict) else {}
                tab_info = {
                    "path": raw_tab.get("path", f"/{name}"),
                    "position": raw_tab.get("position", "end"),
                }
                override_path = raw_tab.get("override")
                if isinstance(override_path, str) and override_path.startswith("/"):
                    tab_info["override"] = override_path
                if bool(raw_tab.get("hidden")):
                    tab_info["hidden"] = True
                # Slots: list of named slot locations this plugin populates.
                # The frontend exposes ``registerSlot(pluginName, slotName, Component)``
                # on window; plugins with non-empty slots call it from their JS bundle.
                slots_src = data.get("slots")
                slots: List[str] = []
                if isinstance(slots_src, list):
                    slots = [s for s in slots_src if isinstance(s, str) and s]
                # Validate ``api`` at discovery time so the value cached
                # on the plugin entry is already safe to feed into the
                # importer.  An attacker-controlled manifest can name
                # any absolute path or ``..`` traversal here — the
                # web server then imports that file as a Python module
                # (RCE, GHSA-5qr3-c538-wm9j).
                raw_api = data.get("api")
                dashboard_dir = child / "dashboard"
                safe_api = _safe_plugin_api_relpath(raw_api, dashboard_dir=dashboard_dir)
                if raw_api and safe_api is None:
                    _log.warning(
                        "Plugin %s: refusing unsafe api path %r (must be a "
                        "relative file inside the plugin's dashboard/ "
                        "directory); backend routes from this plugin will "
                        "not be mounted",
                        name, raw_api,
                    )
                plugins.append({
                    "name": name,
                    "label": data.get("label", name),
                    "description": data.get("description", ""),
                    "icon": data.get("icon", "Puzzle"),
                    "version": data.get("version", "0.0.0"),
                    "tab": tab_info,
                    "slots": slots,
                    "entry": data.get("entry", "dist/index.js"),
                    "css": data.get("css"),
                    "has_api": bool(safe_api),
                    "source": source,
                    "_dir": str(dashboard_dir),
                    "_api_file": safe_api,
                })
            except Exception as exc:
                _log.warning("Bad dashboard plugin manifest %s: %s", manifest_file, exc)
                continue
    return plugins


# Cache discovered plugins per-process (refresh on explicit re-scan).
_dashboard_plugins_cache: Optional[list] = None


def _get_dashboard_plugins(force_rescan: bool = False) -> list:
    global _dashboard_plugins_cache
    if _dashboard_plugins_cache is None or force_rescan:
        _dashboard_plugins_cache = _discover_dashboard_plugins()
    elif _dashboard_plugins_cache:
        if any(not Path(p["_dir"]).is_dir() for p in _dashboard_plugins_cache):
            _dashboard_plugins_cache = _discover_dashboard_plugins()
    return _dashboard_plugins_cache


@app.get("/api/dashboard/plugins")
async def get_dashboard_plugins():
    """Return discovered dashboard plugins (excludes user-hidden and non-enabled ones)."""
    plugins = _get_dashboard_plugins()
    # Read user's hidden plugins list from config.
    config = load_config()
    hidden: list = cfg_get(config, "dashboard", "hidden_plugins", default=[]) or []
    # Gate: only serve user plugins that are in plugins.enabled and not
    # in plugins.disabled.  This prevents the frontend from loading JS/CSS
    # from plugins the user has not explicitly activated.  (#46435)
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()

    def _is_active(p: dict) -> bool:
        name = p.get("name", "")
        if name in hidden:
            return False
        if p.get("source") == "user":
            if name in disabled_set:
                return False
            if name not in enabled_set:
                return False
        elif p.get("source") == "bundled":
            if name in disabled_set:
                return False
        return True

    # Strip internal fields before sending to frontend.
    return [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in plugins
        if _is_active(p)
    ]


@app.get("/api/dashboard/plugins/rescan")
async def rescan_dashboard_plugins():
    """Force re-scan of dashboard plugins."""
    plugins = _get_dashboard_plugins(force_rescan=True)
    return {"ok": True, "count": len(plugins)}


class _AgentPluginInstallBody(BaseModel):
    identifier: str
    force: bool = False
    enable: bool = True


def _strip_dashboard_manifest(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in p.items() if not k.startswith("_")}


def _merged_plugins_hub() -> Dict[str, Any]:
    """Agent discovery + dashboard manifests + optional provider picker metadata."""
    from hermes_cli.plugins_cmd import (
        _discover_all_plugins,
        _get_current_context_engine,
        _get_current_memory_provider,
        _discover_context_engines,
        _get_disabled_set,
        _get_enabled_set,
        _read_manifest as _read_plugin_manifest_at,
    )

    dashboard_list = _get_dashboard_plugins()
    dash_by_name = {str(p["name"]): p for p in dashboard_list}

    disabled_set = _get_disabled_set()
    enabled_set = _get_enabled_set()

    # Read user-hidden plugins from config for the user_hidden field.
    config = load_config()
    hidden_plugins: list = cfg_get(config, "dashboard", "hidden_plugins", default=[]) or []

    plugins_root_resolved = (get_hermes_home() / "plugins").resolve()
    rows: List[Dict[str, Any]] = []

    for name, version, description, source, dir_str, key in _discover_all_plugins():
        # Both the path-derived key (nested category plugins) and the bare
        # manifest name count for enabled/disabled state, matching the runtime
        # loader's back-compat lookup.
        aliases = {name}
        if key:
            aliases.add(key)
        if aliases & disabled_set:
            runtime_status = "disabled"
        elif aliases & enabled_set:
            runtime_status = "enabled"
        else:
            runtime_status = "inactive"

        dir_path = Path(dir_str)
        dm = dash_by_name.get(name)
        has_dash_manifest = dm is not None or (dir_path / "dashboard" / "manifest.json").exists()

        under_user_tree = False
        try:
            dir_path.resolve().relative_to(plugins_root_resolved)
            under_user_tree = True
        except ValueError:
            pass

        can_remove_update = (
            source in {"user", "git"} and under_user_tree and Path(dir_str).is_dir()
        )

        # Check if this plugin provides tools that require auth
        auth_required = False
        auth_command = ""
        manifest_data = _read_plugin_manifest_at(dir_path)
        provides_tools = manifest_data.get("provides_tools") or []
        if provides_tools:
            try:
                from tools.registry import registry
                for tname in provides_tools:
                    entry = registry.get_entry(tname)
                    if entry and entry.check_fn and not entry.check_fn():
                        auth_required = True
                        auth_command = f"hermes auth {name}"
                        break
            except Exception:
                pass

        rows.append({
            "name": name,
            "version": version or "",
            "description": description or "",
            "source": source,
            "runtime_status": runtime_status,
            "has_dashboard_manifest": has_dash_manifest,
            "dashboard_manifest": _strip_dashboard_manifest(dm) if dm else None,
            "path": dir_str,
            "can_remove": can_remove_update,
            "can_update_git": can_remove_update and (Path(dir_str) / ".git").exists(),
            "auth_required": auth_required,
            "auth_command": auth_command,
            "user_hidden": name in hidden_plugins,
        })

    agent_names = {r["name"] for r in rows}
    orphan_dashboard = [
        _strip_dashboard_manifest(p)
        for p in dashboard_list
        if str(p["name"]) not in agent_names
    ]

    memory_providers = _discover_memory_provider_statuses()

    context_engines: List[Dict[str, str]] = []
    try:
        for n, desc in _discover_context_engines():
            context_engines.append({"name": n, "description": desc})
    except Exception:
        context_engines = []

    return {
        "plugins": rows,
        "orphan_dashboard_plugins": orphan_dashboard,
        "providers": {
            "memory_provider": _normalize_memory_provider_name(_get_current_memory_provider()),
            "memory_options": memory_providers,
            "context_engine": _get_current_context_engine(),
            "context_options": context_engines,
        },
    }


@app.get("/api/dashboard/plugins/hub")
async def get_plugins_hub(request: Request):
    """Unified agent plugins + dashboard extension metadata (session protected)."""
    _require_token(request)
    try:
        return _merged_plugins_hub()
    except Exception as exc:
        _log.warning("plugins/hub failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build plugins hub.") from exc


@app.post("/api/dashboard/agent-plugins/install")
async def post_agent_plugin_install(request: Request, body: _AgentPluginInstallBody):
    _require_token(request)
    from hermes_cli.plugins_cmd import dashboard_install_plugin

    result = dashboard_install_plugin(
        body.identifier.strip(),
        force=body.force,
        enable=body.enable,
    )
    if not result.get("ok"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error") or "Install failed.",
        )
    _get_dashboard_plugins(force_rescan=True)
    # Strip internal paths from the response
    result.pop("after_install_path", None)
    return result


def _validate_plugin_name(name: str) -> str:
    """Reject path-traversal attempts in plugin name URL parameters."""
    name = name.strip("/")
    if not name or ".." in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid plugin name.")
    return name


@app.post("/api/dashboard/agent-plugins/{name:path}/enable")
async def post_agent_plugin_enable(request: Request, name: str):
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    result = dashboard_set_agent_plugin_enabled(name, enabled=True)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Enable failed.")
    return result


@app.post("/api/dashboard/agent-plugins/{name:path}/disable")
async def post_agent_plugin_disable(request: Request, name: str):
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    result = dashboard_set_agent_plugin_enabled(name, enabled=False)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Disable failed.")
    return result


@app.post("/api/dashboard/agent-plugins/{name:path}/update")
async def post_agent_plugin_update(request: Request, name: str):
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_update_user_plugin

    result = dashboard_update_user_plugin(name)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Update failed.")
    _get_dashboard_plugins(force_rescan=True)
    return result


@app.delete("/api/dashboard/agent-plugins/{name:path}")
async def delete_agent_plugin(request: Request, name: str):
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_remove_user_plugin

    result = dashboard_remove_user_plugin(name)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Remove failed.")
    _get_dashboard_plugins(force_rescan=True)
    return result


class _PluginProvidersPutBody(BaseModel):
    memory_provider: Optional[str] = None
    context_engine: Optional[str] = None


@app.put("/api/dashboard/plugin-providers")
async def put_plugin_providers(request: Request, body: _PluginProvidersPutBody):
    """Persist memory provider / context engine selection (writes config.yaml)."""
    _require_token(request)
    from hermes_cli.plugins_cmd import (
        _save_context_engine,
        _save_memory_provider,
    )

    if body.memory_provider is not None:
        memory_provider = _normalize_memory_provider_name(body.memory_provider)
        _require_memory_provider_ready(memory_provider)
        _save_memory_provider(memory_provider)
    if body.context_engine is not None:
        _save_context_engine(body.context_engine)
    return {"ok": True}


class _PluginVisibilityBody(BaseModel):
    hidden: bool


@app.post("/api/dashboard/plugins/{name:path}/visibility")
async def post_plugin_visibility(request: Request, name: str, body: _PluginVisibilityBody):
    """Toggle a plugin's sidebar visibility (persists to config.yaml dashboard.hidden_plugins)."""
    _require_token(request)
    name = _validate_plugin_name(name)

    config = load_config()
    if "dashboard" not in config or not isinstance(config.get("dashboard"), dict):
        config["dashboard"] = {}
    hidden_list: list = config["dashboard"].get("hidden_plugins") or []
    if not isinstance(hidden_list, list):
        hidden_list = []

    if body.hidden and name not in hidden_list:
        hidden_list.append(name)
    elif not body.hidden and name in hidden_list:
        hidden_list.remove(name)

    config["dashboard"]["hidden_plugins"] = hidden_list
    save_config(config)
    return {"ok": True, "name": name, "hidden": body.hidden}


@app.get("/dashboard-plugins/{plugin_name}/{file_path:path}")
async def serve_plugin_asset(plugin_name: str, file_path: str):
    """Serve static assets from a dashboard plugin directory.

    Only serves files from the plugin's ``dashboard/`` subdirectory.
    Path traversal is blocked by checking ``resolve().is_relative_to()``.

    Restricted to a browser-fetchable suffix allowlist (JS/CSS/JSON/HTML/
    SVG/PNG/JPG/WOFF). The dashboard loads plugin JS via ``<script src>``
    and CSS via ``<link href>``, neither of which can attach a custom
    auth header — so this route stays unauthenticated to keep the SPA
    working. But user-installed plugins ship a ``plugin_api.py``
    backend module that the browser never fetches; it's only imported
    by :func:`_mount_plugin_api_routes` at startup. Without a suffix
    allowlist, anyone on the loopback port can curl the ``.py`` source
    of a private third-party plugin. Reject everything outside the
    browser-asset set.

    User plugins must be in plugins.enabled before their assets are
    served. (#46435, GHSA-mcfc-hp25-cjv7)
    """
    plugins = _get_dashboard_plugins()
    plugin = next((p for p in plugins if p["name"] == plugin_name), None)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    # Gate: user plugins must be enabled to serve assets;
    # bundled plugins must not be explicitly disabled.
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()
    if plugin.get("source") == "user":
        if plugin_name in disabled_set or plugin_name not in enabled_set:
            raise HTTPException(status_code=404, detail="Plugin not found")
    elif plugin.get("source") == "bundled":
        if plugin_name in disabled_set:
            raise HTTPException(status_code=404, detail="Plugin not found")

    base = Path(plugin["_dir"])
    target = (base / file_path).resolve()

    if not target.is_relative_to(base.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal blocked")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Browser-asset suffix allowlist. Everything outside this set is
    # rejected with 404 so we don't leak ``.py`` backend sources, README
    # files, ``.env.example`` templates, etc. — none of which the SPA
    # actually fetches. Add to this set deliberately when a new asset
    # type comes up; do NOT change the default fallback.
    suffix = target.suffix.lower()
    content_types = {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".html": "text/html",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".ico": "image/x-icon",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
        ".otf": "font/otf",
        ".map": "application/json",
    }
    if suffix not in content_types:
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )
    media_type = content_types[suffix]
    return FileResponse(
        target,
        media_type=media_type,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


def _mount_plugin_api_routes():
    """Import and mount backend API routes from plugins that declare them.

    Each plugin's ``api`` field points to a Python file that must expose
    a ``router`` (FastAPI APIRouter).  Routes are mounted under
    ``/api/plugins/<name>/``.

    Backend import is restricted to ``bundled`` and ``user`` sources.
    Project plugins (``./.hermes/plugins/``) ship with the CWD and are
    therefore attacker-controlled in any threat model where the user
    opens a malicious repo; they can extend the dashboard UI via
    static JS/CSS but their Python ``api`` file is never auto-imported
    by the web server.  See GHSA-5qr3-c538-wm9j (#29156).

    Additionally, user plugins must be explicitly enabled via the
    ``plugins.enabled`` allow-list in config.yaml before their backend
    code is imported. Without this gate, an installed-but-not-enabled
    plugin's Python code would execute at dashboard startup — a code
    execution vector that bypasses the user's intent. (#46435,
    GHSA-mcfc-hp25-cjv7)
    """
    # Load the enabled/disabled sets once for the loop.
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()

    for plugin in _get_dashboard_plugins():
        api_file_name = plugin.get("_api_file")
        if not api_file_name:
            continue
        plugin_name = plugin.get("name", "")
        # Gate: user plugins must be in plugins.enabled and not in
        # plugins.disabled before we import their Python code.
        # Bundled plugins are trusted (they ship with the release) but
        # still respect an explicit disable.
        if plugin.get("source") == "user":
            if plugin_name in disabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (explicitly disabled)",
                    plugin_name,
                )
                continue
            if plugin_name not in enabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (not in plugins.enabled)",
                    plugin_name,
                )
                continue
        elif plugin.get("source") == "bundled":
            if plugin_name in disabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (explicitly disabled)",
                    plugin_name,
                )
                continue
        if plugin.get("source") == "project":
            _log.warning(
                "Plugin %s: ignoring backend api=%s (project plugins may "
                "not auto-import Python code; move the plugin to "
                "~/.hermes/plugins/ if you trust it)",
                plugin["name"], api_file_name,
            )
            continue
        dashboard_dir = Path(plugin["_dir"])
        api_path = dashboard_dir / api_file_name
        try:
            resolved_api = api_path.resolve()
            resolved_base = dashboard_dir.resolve()
            resolved_api.relative_to(resolved_base)
        except (OSError, RuntimeError, ValueError):
            # Discovery already filters this, but re-check here in case
            # ``_dir`` was tampered with after caching or a future caller
            # bypasses the validator.  Defence in depth keeps the import
            # primitive contained even if the upstream check regresses.
            _log.warning(
                "Plugin %s: refusing to import api file outside its "
                "dashboard directory (%s)", plugin["name"], api_path,
            )
            continue
        if not api_path.exists():
            _log.warning("Plugin %s declares api=%s but file not found", plugin["name"], api_file_name)
            continue
        try:
            module_name = f"hermes_dashboard_plugin_{plugin['name']}"
            spec = importlib.util.spec_from_file_location(module_name, api_path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            # Register in sys.modules BEFORE exec_module so pydantic/FastAPI
            # can resolve forward references (e.g. models defined in a file
            # that uses `from __future__ import annotations`). Without this,
            # TypeAdapter lazy-build fails at first request with
            # "is not fully defined" because the module namespace isn't
            # reachable by name for string-annotation resolution.
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            router = getattr(mod, "router", None)
            if router is None:
                _log.warning("Plugin %s api file has no 'router' attribute", plugin["name"])
                continue
            app.include_router(router, prefix=f"/api/plugins/{plugin['name']}")
            _log.info("Mounted plugin API routes: /api/plugins/%s/", plugin["name"])
        except Exception as exc:
            _log.warning("Failed to load plugin %s API routes: %s", plugin["name"], exc)


# Mount plugin API routes before the SPA catch-all.
_mount_plugin_api_routes()

# Mount the dashboard auth routes (/login, /auth/*, /api/auth/*) before the
# SPA catch-all so /{full_path:path} doesn't swallow them.  These are
# always mounted — the gate middleware decides whether to enforce auth,
# not whether the routes exist.
from hermes_cli.dashboard_auth.routes import router as _dashboard_auth_router  # noqa: E402
app.include_router(_dashboard_auth_router)

mount_spa(app)


def _read_bound_port(server: "uvicorn.Server", fallback: int) -> int:
    """Read the OS-assigned port from a live uvicorn server socket.

    After ``server.startup()`` the socket is bound.  Returns the actual
    port so ephemeral (port-0) discovery works without a pre-bind TOCTOU.
    Falls back to *fallback* if the socket list is empty (shouldn't happen
    but guards against uvicorn internals changing).
    """
    if server.servers and server.servers[0].sockets:
        return server.servers[0].sockets[0].getsockname()[1]
    return fallback


def _write_dashboard_ready_file(actual_port: int) -> None:
    """Optionally publish the dashboard port through an atomic ready file.

    Windows Desktop can launch dashboard backends with ``pythonw.exe`` to avoid
    console flashes. That path cannot rely on stdout for the port announcement,
    so Electron passes ``HERMES_DESKTOP_READY_FILE`` and waits for this JSON.
    Normal CLI/dashboard launches still use the stdout READY line below.
    """
    target = os.environ.get("HERMES_DESKTOP_READY_FILE")
    if not target:
        return

    tmp_name = ""
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"port": int(actual_port)}, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
            tmp_name = fh.name
        os.replace(tmp_name, path)
    except Exception as exc:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
        _log.warning("Failed to write dashboard ready file %r: %s", target, exc)


def _maybe_open_browser(
    host: str, actual_port: int, open_browser: bool, initial_profile: str
) -> None:
    """Open the dashboard URL in the user's browser if appropriate.

    Skips on headless Linux (no ``DISPLAY`` / ``WAYLAND_DISPLAY``) to avoid
    TUI browsers (links, lynx) that would SIGHUP the server process.
    Maps ``0.0.0.0`` / ``::`` binds to ``127.0.0.1`` so the browser opens
    a reachable URL.
    """
    if not open_browser:
        return

    import webbrowser

    _has_display = (
        sys.platform != "linux"
        or bool(os.environ.get("DISPLAY"))
        or bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    if not _has_display:
        _log.debug(
            "Skipping browser-open: no DISPLAY or WAYLAND_DISPLAY detected "
            "(headless Linux). Pass --no-open to suppress this detection."
        )
        return

    _display_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
    _open_url = f"http://{_display_host}:{actual_port}"
    if initial_profile:
        from urllib.parse import quote
        _open_url += f"/?profile={quote(initial_profile)}"

    def _open():
        try:
            time.sleep(1.0)
            webbrowser.open(_open_url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


def start_server(
    host: str = "127.0.0.1",
    port: int = 9119,
    open_browser: bool = True,
    allow_public: bool = False,
    initial_profile: str = "",
    headless: bool = False,
):
    """Start the web UI server.

    ``initial_profile`` (when set) is appended to the auto-opened browser
    URL as ``?profile=<name>`` so the SPA's profile switcher preselects it
    — used when a profile alias (``<profile> dashboard``) routes to the
    machine dashboard.

    ``headless`` is the ``serve`` path: the JSON-RPC/WS backend with no UI
    build and no SPA mount (mount_spa() honours ``HERMES_SERVE_HEADLESS``), so
    the banner announces the bind rather than a browser URL.
    """
    import uvicorn

    try:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive

        start_nous_auth_keepalive()
    except Exception as exc:
        _log.debug("Nous auth keepalive did not start: %s", exc)

    # Phase 0: stash the auth-gate flag on app.state so middleware / SPA-token
    # injection / WS-auth paths can branch on it consistently.  Phase 3.5
    # uses this to decide whether to refuse the bind, log the gate-on
    # banner, and enable uvicorn proxy_headers.
    app.state.auth_required = should_require_auth(host)

    # ``--insecure`` no longer disables the auth gate (June 2026 hardening:
    # the hermes-0day MCP-persistence campaign abused unauthenticated public
    # dashboards). If a caller still passes it, warn that it is now a no-op
    # rather than silently changing their expectation of an open bind.
    if allow_public and host not in _LOOPBACK_HOST_VALUES:
        _log.warning(
            "--insecure no longer bypasses dashboard authentication. A "
            "non-loopback bind (%s) now ALWAYS requires an auth provider "
            "(OAuth or the bundled password provider). Configure one — see "
            "below — or bind to 127.0.0.1 and reach it over an SSH tunnel / "
            "Tailscale.", host,
        )

    if app.state.auth_required:
        # The gate engages on every non-loopback bind. Require at least one
        # provider to be registered, else fail closed — there is no longer an
        # escape hatch that serves the dashboard without authentication.
        from hermes_cli.dashboard_auth import list_providers
        if not list_providers():
            # Surface the *specific* reason any bundled provider declined
            # to register (e.g. missing HERMES_DASHBOARD_OAUTH_CLIENT_ID).
            # Each provider plugin that ships with Hermes Agent exposes a
            # module-level ``LAST_SKIP_REASON`` string for this purpose;
            # without it the operator would only see "no providers" which
            # is misleading when the provider IS installed but unconfigured.
            skip_reasons: list[str] = []
            try:
                from plugins.dashboard_auth import nous as _nous_plugin

                if _nous_plugin.LAST_SKIP_REASON:
                    skip_reasons.append(
                        f"  • nous: {_nous_plugin.LAST_SKIP_REASON}"
                    )
            except Exception:
                pass

            _fix_hint = (
                "Configure an auth provider before exposing the dashboard:\n"
                "  • Password: set dashboard.basic_auth.username + "
                "password_hash in config.yaml\n"
                "    (hash with: python -c \"from "
                "plugins.dashboard_auth.basic import hash_password; "
                "print(hash_password('your-password'))\")\n"
                "  • OAuth: run `hermes dashboard register` (Nous Portal) or "
                "install a DashboardAuthProvider plugin.\n"
                "There is no unauthenticated public-bind option — to keep it "
                "local, bind 127.0.0.1 and tunnel in (SSH / Tailscale)."
            )
            if skip_reasons:
                raise SystemExit(
                    f"Refusing to bind dashboard to {host} — the auth gate "
                    f"engages on non-loopback binds, but no auth providers "
                    f"are registered.\n\n"
                    f"Bundled providers reported these issues:\n"
                    + "\n".join(skip_reasons)
                    + "\n\n"
                    + _fix_hint
                )
            raise SystemExit(
                f"Refusing to bind dashboard to {host} — the auth gate "
                f"engages on non-loopback binds, but no auth providers are "
                f"registered.\n\n" + _fix_hint
            )
        _log.info(
            "Dashboard binding to %s with auth gate enabled. Providers: %s",
            host,
            ", ".join(p.name for p in list_providers()),
        )

    # Record the bound host so host_header_middleware can validate incoming
    # Host headers against it. Defends against DNS rebinding (GHSA-ppp5-vxwm-4cf7).
    app.state.bound_host = host

    # ── Start uvicorn with direct Server API ─────────────────────────
    # We use uvicorn.Server directly (not uvicorn.run) so we can split
    # startup from the main loop.  After startup() the socket is actually
    # bound — we read the OS-assigned port from the live socket, print
    # HERMES_DASHBOARD_READY, open the browser, *then* serve.
    #
    # This eliminates the TOCTOU of the old pre-bind-then-close approach
    # (bind port 0 → close → uvicorn rebind): the socket is held by
    # uvicorn the entire time, so no other process can steal the port.
    #
    # For explicit non-zero ports, if the port is taken uvicorn catches
    # OSError inside create_server() and exits with a clear error — no
    # separate preflight probe needed.
    # Loopback binds are the Desktop case: a single local client, no reverse
    # proxy in front. uvicorn's ws keepalive ping runs ON the same event loop
    # as agent turns, and a single synchronous GIL-holding call on a worker
    # thread (e.g. a regex/scrub over a large model output, or a long
    # delegate_task subagent turn) can starve that loop for *minutes* — the
    # loop cannot process the incoming pong, so uvicorn declares the socket
    # dead and closes it, dropping an otherwise-healthy local connection
    # (#53773: "event loop stalled 226.3s"; #48445/#50005). A longer timeout
    # only raises the threshold — a multi-minute stall sails past any finite
    # window. The keepalive ping exists to detect *half-open* connections
    # (reverse-proxy 524, dropped tunnels), which cannot happen on loopback:
    # there is no network or proxy in the path, and a dead local client tears
    # the socket down with a real FIN/RST that starlette surfaces as
    # WebSocketDisconnect regardless of the ping. So on loopback the ping
    # provides ~no liveness value while actively killing recoverable stalls —
    # disable it entirely. Non-loopback binds sit behind a Cloudflare Tunnel
    # (idle timeout ~100s) where half-open IS a real failure mode, so keep the
    # ping at 20/20 to detect it promptly and stay under the tunnel's idle
    # window.
    _is_loopback = host in ("127.0.0.1", "localhost", "::1")
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning",
        # proxy_headers defaults to False so _ws_client_is_allowed sees
        # the real connection peer rather than X-Forwarded-For's rewritten
        # value (which would defeat the loopback gate when behind a reverse
        # proxy).  When the OAuth gate is active we are explicitly running
        # behind a TLS terminator (Fly.io) and need X-Forwarded-Proto to
        # decide cookie Secure flags, so we flip proxy_headers on for that
        # mode.
        proxy_headers=bool(app.state.auth_required),
        # Half-open detection for public binds only (see above). Loopback
        # disables the protocol ping (None) so an event-loop stall can never
        # trigger a false disconnect; a genuinely dead local client is still
        # reaped via the WebSocketDisconnect → disconnect/reap path.
        ws_ping_interval=None if _is_loopback else 20.0,
        ws_ping_timeout=None if _is_loopback else 20.0,
    )
    server = uvicorn.Server(config)

    async def _serve():
        # Split startup from main_loop so we can read the bound port
        # after the socket is live (ephemeral port discovery).
        if not config.loaded:
            config.load()
        server.lifespan = config.lifespan_class(config)
        with server.capture_signals():
            await server.startup()
            if server.should_exit:
                return

            actual_port = _read_bound_port(server, fallback=port)
            app.state.bound_port = actual_port

            _write_dashboard_ready_file(actual_port)
            # Port-discovery sentinel parsed by the desktop spawn. `serve` is a
            # plain backend, not a dashboard, so it announces a neutral token;
            # `dashboard` keeps the legacy one. The desktop matches either.
            ready_token = "HERMES_BACKEND_READY" if headless else "HERMES_DASHBOARD_READY"
            print(f"{ready_token} port={actual_port}", flush=True)
            if headless:
                # No SPA, and the JSON-RPC/WS endpoints are auth-gated — don't
                # advertise a paste-and-connect URL, just announce the bind.
                print(f"  Hermes backend listening on {host}:{actual_port}")
            else:
                print(f"  Hermes Web UI → http://{host}:{actual_port}")
            _maybe_open_browser(host, actual_port, open_browser, initial_profile)

            # Collapse the peer-hangup teardown flood (#50005). When the Desktop
            # forcibly closes its WebSocket mid-write, asyncio logs a full
            # traceback per pending connection-lost callback — 50+ identical
            # WinError 10054 (ConnectionResetError) lines per disconnect on
            # Windows. This filter downgrades exactly that class to one debug
            # line and passes every other loop error through unchanged.
            try:
                from tui_gateway.loop_noise import install_loop_noise_filter

                install_loop_noise_filter(asyncio.get_running_loop())
            except Exception as exc:  # pragma: no cover - best-effort
                _log.debug("loop noise filter install skipped: %s", exc)

            # ── Loop heartbeat watchdog (CF-1) ───────────────────────────
            # Confirm the GIL-pressure hypothesis in production. Re-arm a 2s
            # tick and measure the drift between when it *should* fire and
            # when it actually does: a healthy loop drifts ~0, but a turn that
            # holds the GIL blocks the loop and the next tick fires late by the
            # stall duration. We log that so a stalled-loop WS drop is
            # diagnosable from the gateway log. Uses loop.time() (monotonic)
            # for drift, and call_later (not a task) so it dies with the loop —
            # nothing to cancel on shutdown.
            _hb_interval = 2.0
            _hb_stall_threshold = 5.0
            _hb_loop = asyncio.get_running_loop()

            def _loop_heartbeat(expected: float) -> None:
                now = _hb_loop.time()
                drift = now - expected
                if drift > _hb_stall_threshold:
                    _log.warning(
                        "event loop stalled %.1fs (GIL pressure suspected)",
                        drift,
                    )
                _hb_loop.call_later(
                    _hb_interval, _loop_heartbeat, now + _hb_interval
                )

            _hb_loop.call_later(
                _hb_interval, _loop_heartbeat, _hb_loop.time() + _hb_interval
            )

            await server.main_loop()
            if server.started:
                await server.shutdown()

    # On POSIX, keep the long-standing ``asyncio.run(_serve())`` behavior
    # unchanged — Python's default loop there is already a SelectorEventLoop
    # (or uvloop when uvicorn[standard] installs it), which is exactly what
    # uvicorn serves on. Touching that path would only widen the blast radius
    # for no benefit.
    #
    # On Windows it is broken: ``asyncio.run`` defaults to a ProactorEventLoop,
    # but uvicorn's socket-serving stack assumes a SelectorEventLoop on win32
    # (``uvicorn/loops/asyncio.py`` forces it, and ``uvicorn.Server.run`` threads
    # ``config.get_loop_factory()`` into its runner for exactly this reason).
    # Driving uvicorn on the proactor loop makes ``server.startup()`` bind a
    # socket that never accepts — the dashboard / desktop backend prints
    # "Skipping web UI build" and then hangs forever with the port LISTENING but
    # no TCP handshake completing (#50641). So *only on Windows* we mirror
    # uvicorn's own machinery and run on the loop factory it picks.
    if sys.platform != "win32":
        asyncio.run(_serve())
        return

    # Windows-only path. Resolve the runner + loop factory FIRST (and fall back
    # to a hand-installed Windows selector policy only when uvicorn predates the
    # loop-factory API, < 0.36). The actual serve call is then OUTSIDE the
    # try/except so genuine serve-time errors (port in use, KeyboardInterrupt)
    # propagate normally instead of being swallowed and double-run.
    try:
        from uvicorn._compat import asyncio_run as _runner

        _loop_factory = config.get_loop_factory()
    except Exception:
        _runner = None
        _loop_factory = None
        try:
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined]
            )
        except Exception:
            pass

    if _runner is not None:
        _runner(_serve(), loop_factory=_loop_factory)
    else:
        asyncio.run(_serve())
