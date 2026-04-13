"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless; opt-in session continuity via X-Hermes-Session-Id header)
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent as an available model
- POST /v1/runs                    — start a run, returns run_id immediately (202)
- GET  /v1/runs/{run_id}/events    — SSE stream of structured lifecycle events
- GET  /health                     — health check

Any OpenAI-compatible frontend (Open WebUI, LobeChat, LibreChat,
AnythingLLM, NextChat, ChatBox, etc.) can connect to hermes-agent
through this adapter by pointing at http://localhost:8642/v1.

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "hermes-agent"
    stream: bool = False
class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list


from gateway.config import Platform, PlatformConfig
def json_response(data, status=200, headers=None):
    return JSONResponse(content=data, status_code=status, headers=headers)

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = 1_000_000  # 1 MB default limit for POST bodies


# The gateway now uses FastAPI + uvicorn (aiohttp is no longer required),
# but /v1/runs/{run_id}/events still uses aiohttp.web.StreamResponse and
# the test suite patches AIOHTTP_AVAILABLE to simulate a missing runtime.
# Keep both accessible at module scope so callers/tests can patch them.
try:  # pragma: no cover - import-time feature detection
    import aiohttp as _aiohttp  # noqa: F401
    from aiohttp import web  # noqa: F401
    AIOHTTP_AVAILABLE = True
except Exception:  # pragma: no cover
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]


def check_api_server_requirements() -> bool:
    """Return True when the aiohttp runtime dependency is importable."""
    return bool(AIOHTTP_AVAILABLE)


def _parse_council_header(header_value: str) -> Optional[bool]:
    """Parse the X-Hermes-Council request header into an override bool.

    Returns:
        True  if header is "on"/"true"/"1"/"yes"
        False if header is "off"/"false"/"0"/"no"
        None  if header is missing/empty/unrecognized (use config default)
    """
    if not header_value:
        return None
    normalized = header_value.strip().lower()
    if normalized in ("on", "true", "1", "yes"):
        return True
    if normalized in ("off", "false", "0", "no"):
        return False
    return None


class ResponseStore:
    """
    SQLite-backed LRU store for Responses API state.

    Each stored response includes the full internal conversation history
    (with tool calls and results) so it can be reconstructed on subsequent
    requests via previous_response_id.

    Persists across gateway restarts.  Falls back to in-memory SQLite
    if the on-disk path is unavailable.
    """

    def __init__(self, max_size: int = MAX_STORED_RESPONSES, db_path: str = None):
        self._max_size = max_size
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
            except Exception:
                db_path = ":memory:"
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                name TEXT PRIMARY KEY,
                response_id TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", (response_id,)
        ).fetchone()
        if row is None:
            return None
        import time
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id),
        )
        self._conn.commit()
        return json.loads(row[0])

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        import time
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()),
        )
        # Evict oldest entries beyond max_size
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            self._conn.execute(
                "DELETE FROM responses WHERE response_id IN "
                "(SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?)",
                (count - self._max_size,),
            )
        self._conn.commit()

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        cursor = self._conn.execute(
            "DELETE FROM responses WHERE response_id = ?", (response_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        row = self._conn.execute(
            "SELECT response_id FROM conversations WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)",
            (name, response_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_CORS_HEADERS = {
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, Idempotency-Key",
}




def _openai_error(message: str, err_type: str = "invalid_request_error", param: str = None, code: str = None) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }



_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}




class _RateLimiter:
    """Simple per-IP sliding-window rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._buckets: Dict[str, List[float]] = {}

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        cutoff = now - self._window
        bucket = self._buckets.get(ip)
        if bucket is None:
            self._buckets[ip] = [now]
            return True
        # Trim old entries
        bucket[:] = [t for t in bucket if t > cutoff]
        if len(bucket) >= self._max:
            return False
        bucket.append(now)
        return True

    def cleanup(self) -> None:
        """Periodically remove stale IPs (call from a slow path)."""
        now = time.time()
        cutoff = now - self._window * 2
        stale = [ip for ip, b in self._buckets.items() if not b or b[-1] < cutoff]
        for ip in stale:
            self._buckets.pop(ip, None)


_rate_limiter = _RateLimiter(
    max_requests=int(os.getenv("API_RATE_LIMIT_MAX", "60")),
    window_seconds=int(os.getenv("API_RATE_LIMIT_WINDOW", "60")),
)




class _IdempotencyCache:
    """In-memory idempotency cache with TTL and basic LRU semantics."""
    def __init__(self, max_items: int = 1000, ttl_seconds: int = 300):
        from collections import OrderedDict
        self._store = OrderedDict()
        self._ttl = ttl_seconds
        self._max = max_items

    def _purge(self):
        import time as _t
        now = _t.time()
        expired = [k for k, v in self._store.items() if now - v["ts"] > self._ttl]
        for k in expired:
            self._store.pop(k, None)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    async def get_or_set(self, key: str, fingerprint: str, compute_coro):
        self._purge()
        item = self._store.get(key)
        if item and item["fp"] == fingerprint:
            return item["resp"]
        resp = await compute_coro()
        import time as _t
        self._store[key] = {"resp": resp, "fp": fingerprint, "ts": _t.time()}
        self._purge()
        return resp


_idem_cache = _IdempotencyCache()


def _make_request_fingerprint(body: Dict[str, Any], keys: List[str]) -> str:
    from hashlib import sha256
    subset = {k: body.get(k) for k in keys}
    return sha256(repr(subset).encode("utf-8")).hexdigest()


class APIServerAdapter(BasePlatformAdapter):
    """
    OpenAI-compatible HTTP API server adapter.

    Runs an aiohttp web server that accepts OpenAI-format requests
    and routes them through hermes-agent's AIAgent.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port", os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))))
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        # Refuse to start unauthenticated on non-loopback hosts. Localhost-only
        # bindings are trusted; anything reachable from other machines MUST
        # require a Bearer token. Override with API_SERVER_ALLOW_INSECURE=1.
        _loopback_hosts = {"127.0.0.1", "localhost", "::1"}
        if self._host not in _loopback_hosts and not self._api_key:
            if os.getenv("API_SERVER_ALLOW_INSECURE", "").lower() not in ("1", "true", "yes"):
                raise RuntimeError(
                    f"API server bound to non-loopback host {self._host!r} without "
                    "API_SERVER_KEY set. Set API_SERVER_KEY to enable Bearer auth "
                    "or API_SERVER_ALLOW_INSECURE=1 to override."
                )
            logger.warning(
                "API server bound to %s without auth (API_SERVER_ALLOW_INSECURE set)",
                self._host,
            )
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")),
        )
        self._app: Optional[FastAPI] = None
        
        
        self._response_store = ResponseStore()
        # Active run streams: run_id -> asyncio.Queue of SSE event dicts
        self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
        # Creation timestamps for orphaned-run TTL sweep
        self._run_streams_created: Dict[str, float] = {}
        self._session_db: Optional[Any] = None  # Lazy-init SessionDB for session continuity
        # Approval gate: pending requests keyed by request_id
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}
        # Read timeout from config, fall back to extra, fall back to 30s
        try:
            from tools.approval import _get_approval_timeout, _get_companion_gate
            self._approval_timeout: int = _get_approval_timeout()
            self._companion_gate: bool = _get_companion_gate()
        except Exception:
            self._approval_timeout = int(extra.get("approval_timeout", 30))
            self._companion_gate = True

    @staticmethod
    def _parse_cors_origins(value: Any) -> tuple[str, ...]:
        """Normalize configured CORS origins into a stable tuple."""
        if not value:
            return ()

        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]

        return tuple(str(item).strip() for item in items if str(item).strip())

    def _cors_headers_for_origin(self, origin: str) -> Optional[Dict[str, str]]:
        """Return CORS headers for an allowed browser origin."""
        if not origin or not self._cors_origins:
            return None

        if "*" in self._cors_origins:
            headers = dict(_CORS_HEADERS)
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Max-Age"] = "600"
            return headers

        if origin not in self._cors_origins:
            return None

        headers = dict(_CORS_HEADERS)
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
        headers["Access-Control-Max-Age"] = "600"
        return headers

    def _origin_allowed(self, origin: str) -> bool:
        """Allow non-browser clients and explicitly configured browser origins."""
        if not origin:
            return True

        if not self._cors_origins:
            return False

        return "*" in self._cors_origins or origin in self._cors_origins

    def install_cors_middleware(self, app: FastAPI) -> None:
        """Attach CORS handling that is strict-by-default.

        Browser requests (Origin header present) against an adapter with no
        configured allowlist are rejected with 403. Allowed origins get
        explicit CORS headers (including Vary: Origin, Max-Age: 600, and
        the documented method/header list) and OPTIONS preflights short-
        circuit with a 200.

        Called from both `_build_app()` (production) and test helpers so
        the browser-facing contract is exercised end-to-end.
        """
        @app.middleware("http")
        async def _cors_middleware(request: Request, call_next):
            origin = request.headers.get("origin", "")
            if origin and not self._origin_allowed(origin):
                return JSONResponse(
                    {"error": {"message": "CORS: origin not allowed", "type": "forbidden"}},
                    status_code=403,
                )

            cors_headers = self._cors_headers_for_origin(origin) if origin else None

            if request.method == "OPTIONS" and origin and cors_headers is not None:
                # Echo requested headers so Authorization/Idempotency-Key etc
                # aren't stripped when the spec header list is asked for.
                requested = request.headers.get("access-control-request-headers", "")
                merged_headers = dict(cors_headers)
                if requested:
                    merged_headers["Access-Control-Allow-Headers"] = (
                        merged_headers.get("Access-Control-Allow-Headers", "") + ", " + requested
                    ).strip(", ")
                return Response(status_code=200, headers=merged_headers)

            response = await call_next(request)
            if cors_headers is not None:
                for k, v in cors_headers.items():
                    response.headers[k] = v
            return response

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    # F-009: per-client-IP auth-failure rate limiter. The upstream finding
    # called for a full multi-key ring with per-key revocation; that is
    # out-of-scope for this batch (schema + CLI plumbing). As a meaningful
    # partial mitigation we add a 5-fail/60s IP lockout — the common attacker
    # path (credential stuffing on a leaked static API key) now burns out
    # quickly without requiring operator rotation.
    _AUTH_FAIL_WINDOW_SEC = 60
    _AUTH_FAIL_LIMIT = 5

    def _auth_fail_state(self) -> Dict[str, List[float]]:
        state = getattr(self, "_auth_fail_state_dict", None)
        if state is None:
            state = {}
            self._auth_fail_state_dict = state
        return state

    def _record_auth_fail(self, ip: str) -> bool:
        """Record one auth failure for `ip`; return True if IP is now locked."""
        now = time.time()
        state = self._auth_fail_state()
        timestamps = state.get(ip, [])
        cutoff = now - self._AUTH_FAIL_WINDOW_SEC
        timestamps = [t for t in timestamps if t > cutoff]
        timestamps.append(now)
        state[ip] = timestamps
        return len(timestamps) >= self._AUTH_FAIL_LIMIT

    def _is_locked_out(self, ip: str) -> bool:
        now = time.time()
        state = self._auth_fail_state()
        cutoff = now - self._AUTH_FAIL_WINDOW_SEC
        fresh = [t for t in state.get(ip, []) if t > cutoff]
        return len(fresh) >= self._AUTH_FAIL_LIMIT

    def _check_auth(self, request: Request) -> Optional[Response]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        If no API key is configured, all requests are allowed.
        """
        if not self._api_key:
            return None  # No key configured — allow all (local-only use)

        peer_ip = ""
        if getattr(request, "client", None):
            peer_ip = request.client.host or ""
        if peer_ip and self._is_locked_out(peer_ip):
            return json_response(
                {"error": {"message": "Too many failed auth attempts, retry later.", "type": "rate_limited", "code": "auth_rate_limited"}},
                status=429,
            )

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._api_key):
                return None  # Auth OK

        if peer_ip:
            self._record_auth_fail(peer_ip)
        return json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    def _caller_fingerprint(self, request: Request) -> Optional[str]:
        """F-003: derive an owner fingerprint from the caller's Bearer token.

        Returns None when no API key is configured on the server (local-only
        mode) — in that case, sessions are "anonymous" and cross-owner checks
        are a no-op. Otherwise returns a 16-hex-char prefix of the sha256 of
        the raw token. This is stable per-token, non-reversible, short enough
        to log safely, and never exposed to clients.
        """
        if not self._api_key:
            return None
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        token = auth[7:].strip()
        if not token:
            return None
        return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        council_enabled: Optional[bool] = None,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.

        When ``council_enabled`` is not None, it overrides the council.enabled
        config setting for this single agent instance (honoring a per-request
        X-Hermes-Council header).
        """
        from run_agent import AIAgent
        from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model, _load_gateway_config
        from hermes_cli.tools_config import _get_platform_tools

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        model = _resolve_gateway_model()

        user_config = _load_gateway_config()
        enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            platform="api_server",
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            council_enabled=council_enabled,
        )
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: Request) -> Response:
        """GET /health — simple health check."""
        try:
            from agent.auxiliary_client import get_available_vision_backends

            vision_backends = get_available_vision_backends()
        except Exception:
            vision_backends = []

        return json_response({
            "status": "ok",
            "platform": "hermes-agent",
            "capabilities": {
                "supports_vision": bool(vision_backends),
                "vision_backends": vision_backends,
            },
        })

    async def _handle_models(self, request: Request) -> Response:
        """GET /v1/models — return hermes-agent as an available model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return json_response({
            "object": "list",
            "data": [
                {
                    "id": "hermes-agent",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hermes",
                    "permission": [],
                    "root": "hermes-agent",
                    "parent": None,
                }
            ],
        })

    async def _handle_chat_completions(self, request: Request) -> Response:
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return json_response(_openai_error("Invalid JSON in request body"), status=400)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        stream = body.get("stream", False)

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                # Accumulate system messages
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role in ("user", "assistant"):
                conversation_messages.append({"role": role, "content": content})

        # Extract the last user message as the primary input
        user_message = ""
        history = []
        if conversation_messages:
            user_message = conversation_messages[-1].get("content", "")
            history = conversation_messages[:-1]

        if not user_message:
            return json_response(
                {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
                status=400,
            )

        # Per-request Analyst Council override.  When absent, the agent uses
        # the council.enabled config default.  When "on"/"off", the header
        # forces council mode for this single request (inside or outside of
        # a continued session).
        council_override = _parse_council_header(request.headers.get("X-Hermes-Council", ""))

        # Allow caller to continue an existing session by passing X-Hermes-Session-Id.
        # When provided, history is loaded from state.db instead of from the request body.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        caller_fp = self._caller_fingerprint(request)  # F-003
        if provided_session_id:
            session_id = provided_session_id
            try:
                if self._session_db is None:
                    from hermes_state import SessionDB
                    self._session_db = SessionDB()
                # F-003: cross-owner session read guard. Legacy sessions (pre-v18)
                # have NULL owner_fingerprint — they remain readable only when the
                # caller is also anonymous (no API key configured server-side).
                # Otherwise owner must match exactly.
                stored_owner = self._session_db.get_session_owner(session_id)
                if stored_owner is None and caller_fp is None:
                    pass  # legacy/loopback anonymous — allow
                elif stored_owner == caller_fp and stored_owner is not None:
                    pass  # owner match
                elif stored_owner is None and caller_fp is not None:
                    # Legacy session being continued by an authenticated caller —
                    # upgrade the owner to bind it to this key from now on.
                    try:
                        self._session_db._execute_write(
                            lambda c: c.execute(
                                "UPDATE sessions SET owner_fingerprint = ? WHERE id = ? AND owner_fingerprint IS NULL",
                                (caller_fp, session_id),
                            )
                        )
                        logger.info(
                            "chat_completions: bound legacy session %s to caller fp=%s",
                            session_id, caller_fp,
                        )
                    except Exception as _bind_e:  # pragma: no cover
                        logger.warning("owner-bind failed for %s: %s", session_id, _bind_e)
                else:
                    logger.warning(
                        "chat_completions: session %s owner mismatch (stored=%s, caller=%s) — refusing",
                        session_id, stored_owner, caller_fp,
                    )
                    return json_response(
                        {"error": {"message": "Session not owned by caller", "type": "permission_denied", "code": "session_owner_mismatch"}},
                        status=403,
                    )
                history = self._session_db.get_messages_as_conversation(session_id)
            except Exception as e:
                logger.warning("Failed to load session history for %s: %s", session_id, e)
                history = []
        else:
            session_id = str(uuid.uuid4())
            logger.info(
                "chat_completions: no X-Hermes-Session-Id header; created new session %s "
                "(prior conversation context will not be restored)",
                session_id,
            )
            # history already set from request body above

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", "hermes-agent")
        created = int(time.time())

        if stream:
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # Filter out None — the agent fires stream_delta_callback(None)
                # to signal the CLI display to close its response box before
                # tool execution, but the SSE writer uses None as end-of-stream
                # sentinel.  Forwarding it would prematurely close the HTTP
                # response, causing Open WebUI (and similar frontends) to miss
                # the final answer after tool calls.  The SSE loop detects
                # completion via agent_task.done() instead.
                if delta is not None:
                    _stream_q.put(delta)

            def _on_tool_progress(name, preview, args):
                """Inject tool progress into the SSE stream for Open WebUI."""
                if name.startswith("_"):
                    return  # Skip internal events (_thinking)
                from agent.display import get_tool_emoji
                emoji = get_tool_emoji(name)
                label = preview or name
                _stream_q.put(f"\n`{emoji} {label}`\n")

            # Start agent in background.  agent_ref is a mutable container
            # so the SSE writer can interrupt the agent on client disconnect.
            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                agent_ref=agent_ref,
                council_enabled=council_override,
            ))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
            )

        # Non-streaming: run the agent (with optional Idempotency-Key)
        async def _compute_completion():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                council_enabled=council_override,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(body, keys=["model", "messages", "tools", "tool_choice", "stream"])
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_completion)
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_completion()
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        return json_response(response_data, headers={"X-Hermes-Session-Id": session_id})

    async def _write_sse_chat_completion(
        self, request: Request, completion_id: str, model: str,
        created: int, stream_q, agent_task, agent_ref=None, session_id: str = None,
    ) -> Response:
        """Write real streaming SSE from agent's stream_delta_callback queue."""
        import queue as _q

        sse_headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id

        async def event_generator():
            try:
                role_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"

                async def _delta_generator():
                    loop = asyncio.get_event_loop()
                    while True:
                        try:
                            delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                        except _q.Empty:
                            if agent_task.done():
                                while True:
                                    try:
                                        d = stream_q.get_nowait()
                                        if d is None: return
                                        yield d
                                    except _q.Empty:
                                        return
                                return
                            if await request.is_disconnected():
                                logger.info("Client disconnected during SSE stream")
                                if agent_ref: agent_ref.interrupt()
                                if not agent_task.done(): agent_task.cancel()
                                return
                            continue
                        if delta is None: return
                        yield delta

                import re
                buffer = ""
                intercept_mode = True
                tag_re = re.compile(r'<cognitive_state\s+realm="([^"]+)"\s+summary="([^"]*)"\s*/>')

                async for delta in _delta_generator():
                    if intercept_mode:
                        buffer += delta
                        if buffer.lstrip().startswith("<"):
                            if "/>" in buffer:
                                match = tag_re.search(buffer)
                                if match:
                                    realm, summary = match.groups()
                                    yield f'event: cognitive_state\ndata: {json.dumps({"realm": realm, "summary": summary})}\n\n'
                                    buffer = buffer[match.end():]
                                intercept_mode = False
                                if buffer.strip("\n \t"):
                                    content_chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"content": buffer.lstrip()}, "finish_reason": None}]}
                                    yield f"data: {json.dumps(content_chunk)}\n\n"
                            elif len(buffer) > 250:
                                intercept_mode = False
                                content_chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"content": buffer}, "finish_reason": None}]}
                                yield f"data: {json.dumps(content_chunk)}\n\n"
                            continue
                        else:
                            intercept_mode = False
                            content_chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"content": buffer}, "finish_reason": None}]}
                            yield f"data: {json.dumps(content_chunk)}\n\n"
                            continue

                    content_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"

                if intercept_mode and buffer:
                    content_chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"content": buffer}, "finish_reason": None}]}
                    yield f"data: {json.dumps(content_chunk)}\n\n"

                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                try:
                    result, agent_usage = await agent_task
                    usage = agent_usage or usage
                except Exception:
                    result = {}

                approval_handled = False
                try:
                    from tools.approval import pop_pending
                    pending = pop_pending(session_id) if self._companion_gate else None
                    if pending:
                        import threading as _threading
                        request_id = f"apr-{uuid.uuid4().hex[:12]}"
                        approval_event = _threading.Event()
                        self._pending_approvals[request_id] = {
                            "event": approval_event,
                            "decision": None,
                            "tool_name": "terminal",
                            "summary": (pending.get("command", ""))[:200],
                            "detail": pending.get("description", ""),
                            "session_id": session_id,
                            "command": pending.get("command", ""),
                        }
                        yield f"data: {json.dumps({'id': request_id, 'object': 'hermes.approval', 'type': 'approval_required', 'tool_name': 'terminal', 'summary': pending.get('description', '')})}\n\n"
                        await loop.run_in_executor(None, lambda: approval_event.wait(timeout=300))
                        if self._pending_approvals[request_id].get("decision") == "approved":
                            result["final_response"] = "\n[Approval granted. Command executed in background. See terminal for details.]\n"
                        else:
                            result["final_response"] = "\n[Approval denied.]\n"
                        del self._pending_approvals[request_id]
                except Exception as e:
                    logger.error(f"Approval gate failed: {e}", exc_info=True)

                if session_id and result and "state_snapshot" in result:
                    self._response_store.put(session_id, result["state_snapshot"])
                    self._response_store.set_conversation("default", session_id)

                final_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                if usage and ("total_tokens" in usage):
                    final_chunk["usage"] = usage
                
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error("SSE stream error: %s", e, exc_info=True)
                if agent_ref: agent_ref.interrupt()
                if not agent_task.done(): agent_task.cancel()
                try:
                    err_msg = f"\n\n[Error: {e}]"
                    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': err_msg}, 'finish_reason': 'error'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception:
                    pass

        return StreamingResponse(event_generator(), headers=sse_headers)


    async def _handle_responses(self, request: Request) -> Response:
        """POST /v1/responses — OpenAI Responses API format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Per-request Analyst Council override (same header as /v1/chat/completions).
        council_override = _parse_council_header(request.headers.get("X-Hermes-Council", ""))

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        raw_input = body.get("input")
        if raw_input is None:
            return json_response(_openai_error("Missing 'input' field"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = body.get("store", True)

        # conversation and previous_response_id are mutually exclusive
        if conversation and previous_response_id:
            return json_response(_openai_error("Cannot use both 'conversation' and 'previous_response_id'"), status=400)

        # Resolve conversation name to latest response_id
        if conversation:
            previous_response_id = self._response_store.get_conversation(conversation)
            # No error if conversation doesn't exist yet — it's a new conversation

        # Normalize input to message list
        input_messages: List[Dict[str, str]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    # Handle content that may be a list of content parts
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "input_text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and part.get("type") == "output_text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = "\n".join(text_parts)
                    input_messages.append({"role": role, "content": content})
        else:
            return json_response(_openai_error("'input' must be a string or array"), status=400)

        # Reconstruct conversation history from previous_response_id
        conversation_history: List[Dict[str, str]] = []
        if previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                return json_response(_openai_error(f"Previous response not found: {previous_response_id}"), status=404)
            conversation_history = list(stored.get("conversation_history", []))
            # If no instructions provided, carry forward from previous
            if instructions is None:
                instructions = stored.get("instructions")

        # Append new input messages to history (all but the last become history)
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        # Last input message is the user_message
        user_message = input_messages[-1].get("content", "") if input_messages else ""
        if not user_message:
            return json_response(_openai_error("No user message found in input"), status=400)

        # Truncation support
        if body.get("truncation") == "auto" and len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        # Run the agent (with Idempotency-Key support)
        session_id = str(uuid.uuid4())

        async def _compute_response():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                council_enabled=council_override,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(
                body,
                keys=["input", "instructions", "previous_response_id", "conversation", "model", "tools"],
            )
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_response)
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_response()
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        created_at = int(time.time())

        # Build the full conversation history for storage
        # (includes tool calls from the agent run)
        full_history = list(conversation_history)
        full_history.append({"role": "user", "content": user_message})
        # Add agent's internal messages if available
        agent_messages = result.get("messages", [])
        if agent_messages:
            full_history.extend(agent_messages)
        else:
            full_history.append({"role": "assistant", "content": final_response})

        # Build output items (includes tool calls + final message)
        output_items = self._extract_output_items(result)

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": body.get("model", "hermes-agent"),
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        # Store the complete response object for future chaining / GET retrieval
        if store:
            self._response_store.put(response_id, {
                "response": response_data,
                "conversation_history": full_history,
                "instructions": instructions,
            })
            # Update conversation mapping so the next request with the same
            # conversation name automatically chains to this response
            if conversation:
                self._response_store.set_conversation(conversation, response_id)

        return json_response(response_data)

    # ------------------------------------------------------------------
    # GET / DELETE response endpoints
    # ------------------------------------------------------------------

    async def _handle_get_response(self, request: Request) -> Response:
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.path_params["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return json_response(stored["response"])

    async def _handle_delete_response(self, request: Request) -> Response:
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.path_params["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Cron jobs API
    # ------------------------------------------------------------------

    # Check cron module availability once (not per-request)
    _CRON_AVAILABLE = False
    try:
        from cron.jobs import (
            list_jobs as _cron_list_fn,
            get_job as _cron_get_fn,
            create_job as _cron_create_fn,
            update_job as _cron_update_fn,
            remove_job as _cron_remove_fn,
            pause_job as _cron_pause_fn,
            resume_job as _cron_resume_fn,
            trigger_job as _cron_trigger_fn,
        )
        _cron_list = staticmethod(_cron_list_fn)
        _cron_get = staticmethod(_cron_get_fn)
        _cron_create = staticmethod(_cron_create_fn)
        _cron_update = staticmethod(_cron_update_fn)
        _cron_remove = staticmethod(_cron_remove_fn)
        _cron_pause = staticmethod(_cron_pause_fn)
        _cron_resume = staticmethod(_cron_resume_fn)
        _cron_trigger = staticmethod(_cron_trigger_fn)
        _CRON_AVAILABLE = True
    except ImportError:
        pass

    _JOB_ID_RE = __import__("re").compile(r"[a-f0-9]{12}")
    # Allowed fields for update — prevents clients injecting arbitrary keys
    _UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled", "labels", "metadata"}
    _MAX_NAME_LENGTH = 200
    _MAX_PROMPT_LENGTH = 5000

    def _check_jobs_available(self) -> Optional[Response]:
        """Return error response if cron module isn't available."""
        if not self._CRON_AVAILABLE:
            return json_response(
                {"error": "Cron module not available"}, status=501,
            )
        return None

    def _check_job_id(self, request: Request) -> tuple:
        """Validate and extract job_id. Returns (job_id, error_response)."""
        job_id = request.path_params["job_id"]
        if not self._JOB_ID_RE.fullmatch(job_id):
            return job_id, json_response(
                {"error": "Invalid job ID format"}, status=400,
            )
        return job_id, None

    async def _handle_list_jobs(self, request: Request) -> Response:
        """GET /api/jobs — list all cron jobs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            include_disabled = request.query_params.get("include_disabled", "").lower() in ("true", "1")
            jobs = self._cron_list(include_disabled=include_disabled)
            return json_response({"jobs": jobs})
        except Exception as e:
            return json_response({"error": str(e)}, status=500)

    async def _handle_create_job(self, request: Request) -> Response:
        """POST /api/jobs — create a new cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            body = await request.json()
            name = (body.get("name") or "").strip()
            schedule = (body.get("schedule") or "").strip()
            prompt = body.get("prompt", "")
            deliver = body.get("deliver", "local")
            skills = body.get("skills")
            labels = body.get("labels")
            metadata = body.get("metadata")
            repeat = body.get("repeat")

            if not name:
                return json_response({"error": "Name is required"}, status=400)
            if len(name) > self._MAX_NAME_LENGTH:
                return json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if not schedule:
                return json_response({"error": "Schedule is required"}, status=400)
            if len(prompt) > self._MAX_PROMPT_LENGTH:
                return json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
                return json_response({"error": "Repeat must be a positive integer"}, status=400)

            kwargs = {
                "prompt": prompt,
                "schedule": schedule,
                "name": name,
                "deliver": deliver,
            }
            if skills:
                kwargs["skills"] = skills
            if labels is not None:
                kwargs["labels"] = labels
            if metadata is not None:
                kwargs["metadata"] = metadata
            if repeat is not None:
                kwargs["repeat"] = repeat

            job = self._cron_create(**kwargs)
            return json_response({"job": job})
        except Exception as e:
            return json_response({"error": str(e)}, status=500)

    async def _handle_get_job(self, request: Request) -> Response:
        """GET /api/jobs/{job_id} — get a single cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_get(job_id)
            if not job:
                return json_response({"error": "Job not found"}, status=404)
            return json_response({"job": job})
        except Exception as e:
            return json_response({"error": str(e)}, status=500)

    async def _handle_update_job(self, request: Request) -> Response:
        """PATCH /api/jobs/{job_id} — update a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            body = await request.json()
            # Whitelist allowed fields to prevent arbitrary key injection
            sanitized = {k: v for k, v in body.items() if k in self._UPDATE_ALLOWED_FIELDS}
            if not sanitized:
                return json_response({"error": "No valid fields to update"}, status=400)
            # Validate lengths if present
            if "name" in sanitized and len(sanitized["name"]) > self._MAX_NAME_LENGTH:
                return json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if "prompt" in sanitized and len(sanitized["prompt"]) > self._MAX_PROMPT_LENGTH:
                return json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            job = self._cron_update(job_id, sanitized)
            if not job:
                return json_response({"error": "Job not found"}, status=404)
            return json_response({"job": job})
        except Exception as e:
            return json_response({"error": str(e)}, status=500)

    async def _handle_delete_job(self, request: Request) -> Response:
        """DELETE /api/jobs/{job_id} — delete a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            success = self._cron_remove(job_id)
            if not success:
                return json_response({"error": "Job not found"}, status=404)
            return json_response({"ok": True})
        except Exception as e:
            return json_response({"error": str(e)}, status=500)

    async def _handle_pause_job(self, request: Request) -> Response:
        """POST /api/jobs/{job_id}/pause — pause a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_pause(job_id)
            if not job:
                return json_response({"error": "Job not found"}, status=404)
            return json_response({"job": job})
        except Exception as e:
            return json_response({"error": str(e)}, status=500)

    async def _handle_resume_job(self, request: Request) -> Response:
        """POST /api/jobs/{job_id}/resume — resume a paused cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_resume(job_id)
            if not job:
                return json_response({"error": "Job not found"}, status=404)
            return json_response({"job": job})
        except Exception as e:
            return json_response({"error": str(e)}, status=500)

    async def _handle_run_job(self, request: Request) -> Response:
        """POST /api/jobs/{job_id}/run — trigger immediate execution."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_trigger(job_id)
            if not job:
                return json_response({"error": "Job not found"}, status=404)
            return json_response({"job": job})
        except Exception as e:
            return json_response({"error": str(e)}, status=500)

    async def _handle_macro_metrics(self, request: Request) -> Response:
        """GET /api/macro-metrics — fetch latest macroeconomic metrics from predictions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            if self._session_db is None:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            
            # Fetch all unresolved predictions
            predictions = self._session_db.get_unresolved_predictions()
            
            # Filter for macro_metric type
            macro_metrics = [
                p for p in predictions 
                if p.get("prediction_type") == "macro_metric"
            ]
            
            # Group by subject and keep the latest for each
            latest_metrics = {}
            for m in sorted(macro_metrics, key=lambda x: x.get("predicted_at", 0)):
                latest_metrics[m["subject"]] = {
                    "value": m["predicted_value"],
                    "timestamp": m["predicted_at"],
                    "confidence": m["confidence"]
                }
            
            return json_response({"metrics": latest_metrics})
        except Exception as e:
            logger.error("Error fetching macro metrics: %s", e)
            return json_response({"error": str(e)}, status=500)

    async def _handle_macro_history(self, request: Request) -> Response:
        """GET /api/macro-history — fetch historical time-series for a metric."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            metric_id = request.query_params.get("metric_id")
            days = int(request.query_params.get("days", "30"))
            if not metric_id:
                return json_response({"error": "metric_id is required"}, status=400)
                
            if self._session_db is None:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
                
            history = self._session_db.get_macro_history(metric_id, days=days)
            return json_response({"history": history})
        except Exception as e:
            logger.error("Error fetching macro history: %s", e)
            return json_response({"error": str(e)}, status=500)

    async def _handle_macro_correlation(self, request: Request) -> Response:
        """GET /api/macro-correlation — calculate correlation between two metrics."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            metric_a = request.query_params.get("metric_a")
            metric_b = request.query_params.get("metric_b")
            days = int(request.query_params.get("days", "90"))
            if not metric_a or not metric_b:
                return json_response({"error": "metric_a and metric_b are required"}, status=400)
                
            from agent.macro_correlation_engine import MacroCorrelationEngine
            engine = MacroCorrelationEngine(self._session_db)
            result = engine.get_correlation(metric_a, metric_b, days=days)
            return json_response(result)
        except Exception as e:
            logger.error("Error calculating macro correlation: %s", e)
            return json_response({"error": str(e)}, status=500)

    async def _handle_macro_narrative(self, request: Request) -> Response:
        """GET /api/macro-narrative — get truth-seeking insight for two metrics."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            metric_a = request.query_params.get("metric_a")
            metric_b = request.query_params.get("metric_b")
            days = int(request.query_params.get("days", "90"))
            if not metric_a or not metric_b:
                return json_response({"error": "metric_a and metric_b are required"}, status=400)
                
            from agent.macro_correlation_engine import MacroCorrelationEngine
            engine = MacroCorrelationEngine(self._session_db)
            result = engine.get_divergence_narrative(metric_a, metric_b, days=days)
            return json_response(result)
        except Exception as e:
            logger.error("Error generating macro narrative: %s", e)
            return json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # Approval gate handlers
    # ------------------------------------------------------------------

    async def _handle_approval_response(self, request: Request) -> Response:
        """POST /v1/approvals/{request_id} — companion sends approval decision."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        request_id = request.path_params.get("request_id", "")
        pending = self._pending_approvals.get(request_id)
        if not pending:
            return json_response({"error": "No pending approval with this ID"}, status=404)

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return json_response({"error": "Invalid JSON"}, status=400)

        decision = body.get("decision", "denied")
        event = pending.get("event")
        pending["decision"] = decision
        if event is not None:
            event.set()

        return json_response({"status": "ok", "decision": decision})

    async def _handle_list_approvals(self, request: Request) -> Response:
        """GET /v1/approvals — list pending approval requests (for polling clients)."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        import time as _t
        now = _t.time()
        pending_list = []
        expired_ids = []
        for req_id, info in self._pending_approvals.items():
            age = now - info.get("timestamp", now)
            if age > self._approval_timeout:
                expired_ids.append(req_id)
                continue
            pending_list.append({
                "request_id": req_id,
                "tool_name": info.get("tool_name", ""),
                "summary": info.get("summary", ""),
                "detail": info.get("detail", ""),
                "session_id": info.get("session_id", ""),
                "timeout_seconds": max(0, self._approval_timeout - int(age)),
            })
        for eid in expired_ids:
            exp = self._pending_approvals.pop(eid, None)
            if exp and exp.get("event"):
                exp["decision"] = "timeout"
                exp["event"].set()

        return json_response({"approvals": pending_list})

    def _emit_approval_sse(
        self,
        stream_q,
        request_id: str,
        tool_name: str,
        summary: str,
        detail: str,
        session_id: str,
    ) -> None:
        """Inject an approval_request event into the SSE stream queue."""
        event_data = json.dumps({
            "event": "approval_request",
            "request_id": request_id,
            "tool_name": tool_name,
            "summary": summary,
            "detail": detail,
            "session_id": session_id or "",
            "timeout_seconds": self._approval_timeout,
        })
        stream_q.put(f"\n\n[APPROVAL_REQUEST]{event_data}[/APPROVAL_REQUEST]\n\n")

    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_output_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the full output item array from the agent's messages.

        Walks *result["messages"]* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    items.append({
                        "type": "function_call",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        # Final assistant message
        final = result.get("final_response", "")
        if not final:
            final = result.get("error", "(No response generated)")

        items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": final,
                }
            ],
        })
        return items

    # ------------------------------------------------------------------
    # Audio Transcription
    # ------------------------------------------------------------------

    async def _handle_audio_transcriptions(self, request: Request) -> Response:
        """POST /v1/audio/transcriptions — transcribe an audio file using configured LLM API."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        
        try:
            form = await request.form()
            upload_file = form.get("file")
            if not upload_file:
                return json_response(_openai_error("Missing 'file' field"), status=400)
                
            audio_bytes = await upload_file.read()
            filename = upload_file.filename or "audio.wav"
            
            from gateway.run import _load_gateway_config, _resolve_runtime_agent_kwargs
            config = _resolve_runtime_agent_kwargs()
            base_url = config.get("base_url")
            api_key = config.get("api_key")
            
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
                if not api_key:
                    return json_response(_openai_error("No API key configured for audio transcription"), status=500)
            
            from openai import AsyncOpenAI
            
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            elif api_key and api_key.startswith("gsk_"):
                client_kwargs["base_url"] = "https://api.groq.com/openai/v1"
            
            client = AsyncOpenAI(**client_kwargs)
            model = "whisper-large-v3" if "groq" in client_kwargs.get("base_url", "") else "whisper-1"
            
            import io
            file_obj = io.BytesIO(audio_bytes)
            file_obj.name = filename
            
            transcription = await client.audio.transcriptions.create(
                model=model,
                file=file_obj,
            )
            
            return json_response({
                "text": transcription.text,
                "language": getattr(transcription, "language", "en")
            })
            
        except Exception as e:
            logger.error("Error in transcription: %s", e, exc_info=True)
            return json_response(
                _openai_error(f"Internal server error: {e}", err_type="server_error"),
                status=500,
            )

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        agent_ref: Optional[list] = None,
        council_enabled: Optional[bool] = None,
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.

        If *agent_ref* is a one-element list, the AIAgent instance is stored
        at ``agent_ref[0]`` before ``run_conversation`` begins.  This allows
        callers (e.g. the SSE writer) to call ``agent.interrupt()`` from
        another thread to stop in-progress LLM calls.

        ``council_enabled`` overrides the council.enabled config setting when
        not None (honoring the per-request X-Hermes-Council header).
        """
        loop = asyncio.get_event_loop()

        def _run():
            # Set env vars so approval.py knows this is a gateway session
            if session_id:
                os.environ["HERMES_SESSION_KEY"] = session_id
                os.environ["HERMES_SESSION_ID"] = session_id
            os.environ["HERMES_GATEWAY_SESSION"] = "1"

            agent = self._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
                tool_progress_callback=tool_progress_callback,
                council_enabled=council_enabled,
            )
            if agent_ref is not None:
                agent_ref[0] = agent
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
            )
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return result, usage

        return await loop.run_in_executor(None, _run)

    # ------------------------------------------------------------------
    # /v1/runs — structured event streaming
    # ------------------------------------------------------------------

    _MAX_CONCURRENT_RUNS = 10  # Prevent unbounded resource allocation
    _RUN_STREAM_TTL = 300  # seconds before orphaned runs are swept

    def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop"):
        """Return a tool_progress_callback that pushes structured events to the run's SSE queue."""
        def _push(event: Dict[str, Any]) -> None:
            q = self._run_streams.get(run_id)
            if q is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

        def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
            ts = time.time()
            if event_type == "tool.started":
                _push({
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "preview": preview,
                })
            elif event_type == "tool.completed":
                _push({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "duration": round(kwargs.get("duration", 0), 3),
                    "error": kwargs.get("is_error", False),
                })
            elif event_type == "reasoning.available":
                _push({
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": ts,
                    "text": preview or "",
                })
            # _thinking and subagent_progress are intentionally not forwarded

        return _callback

    async def _handle_runs(self, request: Request) -> Response:
        """POST /v1/runs — start an agent run, return run_id immediately."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Enforce concurrency limit
        if len(self._run_streams) >= self._MAX_CONCURRENT_RUNS:
            return json_response(
                _openai_error(f"Too many concurrent runs (max {self._MAX_CONCURRENT_RUNS})", code="rate_limit_exceeded"),
                status=429,
            )

        try:
            body = await request.json()
        except Exception:
            return json_response(_openai_error("Invalid JSON"), status=400)

        raw_input = body.get("input")
        if not raw_input:
            return json_response(_openai_error("Missing 'input' field"), status=400)

        user_message = raw_input if isinstance(raw_input, str) else (raw_input[-1].get("content", "") if isinstance(raw_input, list) else "")
        if not user_message:
            return json_response(_openai_error("No user message found in input"), status=400)

        run_id = f"run_{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Optional[Dict]]" = asyncio.Queue()
        self._run_streams[run_id] = q
        self._run_streams_created[run_id] = time.time()

        event_cb = self._make_run_event_callback(run_id, loop)

        # Also wire stream_delta_callback so message.delta events flow through
        def _text_cb(delta: Optional[str]) -> None:
            if delta is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "event": "message.delta",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "delta": delta,
                })
            except Exception:
                pass

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation_history: List[Dict[str, str]] = []
        if previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored:
                conversation_history = list(stored.get("conversation_history", []))
                if instructions is None:
                    instructions = stored.get("instructions")

        session_id = body.get("session_id") or run_id
        ephemeral_system_prompt = instructions

        async def _run_and_close():
            try:
                agent = self._create_agent(
                    ephemeral_system_prompt=ephemeral_system_prompt,
                    session_id=session_id,
                    stream_delta_callback=_text_cb,
                    tool_progress_callback=event_cb,
                )
                def _run_sync():
                    r = agent.run_conversation(
                        user_message=user_message,
                        conversation_history=conversation_history,
                    )
                    u = {
                        "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                        "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                        "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                    }
                    return r, u

                result, usage = await asyncio.get_running_loop().run_in_executor(None, _run_sync)
                final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                q.put_nowait({
                    "event": "run.completed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "output": final_response,
                    "usage": usage,
                })
            except Exception as exc:
                logger.exception("[api_server] run %s failed", run_id)
                try:
                    q.put_nowait({
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": str(exc),
                    })
                except Exception:
                    pass
            finally:
                # Sentinel: signal SSE stream to close
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

        task = asyncio.create_task(_run_and_close())
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        return json_response({"run_id": run_id, "status": "started"}, status=202)

    async def _handle_run_events(self, request: Request) -> Any:
        """GET /v1/runs/{run_id}/events — SSE stream of structured agent lifecycle events."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.path_params["run_id"]

        # Allow subscribing slightly before the run is registered (race condition window)
        for _ in range(20):
            if run_id in self._run_streams:
                break
            await asyncio.sleep(0.05)
        else:
            return json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

        q = self._run_streams[run_id]

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if event is None:
                    # Run finished — send final SSE comment and close
                    await response.write(b": stream closed\n\n")
                    break
                payload = f"data: {json.dumps(event)}\n\n"
                await response.write(payload.encode())
        except Exception as exc:
            logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
        finally:
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)

        return response

    async def _sweep_orphaned_runs(self) -> None:
        """Periodically clean up run streams that were never consumed."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            stale = [
                run_id
                for run_id, created_at in list(self._run_streams_created.items())
                if now - created_at > self._RUN_STREAM_TTL
            ]
            for run_id in stale:
                logger.debug("[api_server] sweeping orphaned run %s", run_id)
                self._run_streams.pop(run_id, None)
                self._run_streams_created.pop(run_id, None)

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the FastAPI web server."""
        try:
            @asynccontextmanager
            async def lifespan(app: FastAPI):
                yield
                # cleanup
            
            self._app = FastAPI(lifespan=lifespan)
            self._app.state.api_server_adapter = self
            
            # Strict, origin-aware CORS handling (see install_cors_middleware).
            self.install_cors_middleware(self._app)

            self._app.add_api_route("/health", self._handle_health, methods=["GET"])
            self._app.add_api_route("/v1/health", self._handle_health, methods=["GET"])
            self._app.add_api_route("/v1/models", self._handle_models, methods=["GET"])
            self._app.add_api_route("/v1/chat/completions", self._handle_chat_completions, methods=["POST"])
            self._app.add_api_route("/v1/audio/transcriptions", self._handle_audio_transcriptions, methods=["POST"])
            self._app.add_api_route("/v1/responses", self._handle_responses, methods=["POST"])
            self._app.add_api_route("/v1/responses/{response_id}", self._handle_get_response, methods=["GET"])
            self._app.add_api_route("/v1/responses/{response_id}", self._handle_delete_response, methods=["DELETE"])
            self._app.add_api_route("/api/jobs", self._handle_list_jobs, methods=["GET"])
            self._app.add_api_route("/api/jobs", self._handle_create_job, methods=["POST"])
            self._app.add_api_route("/api/jobs/{job_id}", self._handle_get_job, methods=["GET"])
            self._app.add_api_route("/api/jobs/{job_id}", self._handle_update_job, methods=["PATCH"])
            self._app.add_api_route("/api/jobs/{job_id}", self._handle_delete_job, methods=["DELETE"])
            self._app.add_api_route("/api/jobs/{job_id}/pause", self._handle_pause_job, methods=["POST"])
            self._app.add_api_route("/api/jobs/{job_id}/resume", self._handle_resume_job, methods=["POST"])
            self._app.add_api_route("/api/jobs/{job_id}/run", self._handle_run_job, methods=["POST"])
            self._app.add_api_route("/api/macro-metrics", self._handle_macro_metrics, methods=["GET"])
            self._app.add_api_route("/api/macro-history", self._handle_macro_history, methods=["GET"])
            self._app.add_api_route("/api/macro-correlation", self._handle_macro_correlation, methods=["GET"])
            self._app.add_api_route("/api/macro-narrative", self._handle_macro_narrative, methods=["GET"])
            self._app.add_api_route("/v1/approvals/{request_id}", self._handle_approval_response, methods=["POST"])
            self._app.add_api_route("/v1/approvals", self._handle_list_approvals, methods=["GET"])
            self._app.add_api_route("/v1/runs", self._handle_runs, methods=["POST"])
            self._app.add_api_route("/v1/runs/{run_id}/events", self._handle_run_events, methods=["GET"])

            # Start background sweep to clean up orphaned runs
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except AttributeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            import socket as _socket
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                    _s.settimeout(1)
                    _s.connect(('127.0.0.1', self._port))
                logger.error('[%s] Port %d already in use.', self.name, self._port)
                return False
            except (ConnectionRefusedError, OSError):
                pass  # port is free

            config = uvicorn.Config(self._app, host=self._host, port=self._port, log_level="warning")
            self._server = uvicorn.Server(config)
            
            # Instead of blocking connect, run serve in a task
            self._server_task = asyncio.create_task(self._server.serve())

            self._mark_connected()
            logger.info(
                "[%s] API server listening on http://%s:%d",
                self.name, self._host, self._port,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

        try:
            mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware, rate_limit_middleware) if mw is not None]
            self._app = web.Application(middlewares=mws)
            self._app["api_server_adapter"] = self
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/v1/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
            self._app.router.add_post("/v1/responses", self._handle_responses)
            self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
            self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)
            # Cron jobs management API
            self._app.router.add_get("/api/jobs", self._handle_list_jobs)
            self._app.router.add_post("/api/jobs", self._handle_create_job)
            self._app.router.add_get("/api/jobs/{job_id}", self._handle_get_job)
            self._app.router.add_patch("/api/jobs/{job_id}", self._handle_update_job)
            self._app.router.add_delete("/api/jobs/{job_id}", self._handle_delete_job)
            self._app.router.add_post("/api/jobs/{job_id}/pause", self._handle_pause_job)
            self._app.router.add_post("/api/jobs/{job_id}/resume", self._handle_resume_job)
            self._app.router.add_post("/api/jobs/{job_id}/run", self._handle_run_job)
            # Macro metrics API
            self._app.router.add_get("/api/macro-metrics", self._handle_macro_metrics)
            self._app.router.add_get("/api/macro-history", self._handle_macro_history)
            self._app.router.add_get("/api/macro-correlation", self._handle_macro_correlation)
            self._app.router.add_get("/api/macro-narrative", self._handle_macro_narrative)
            # Approval gate
            self._app.router.add_post("/v1/approvals/{request_id}", self._handle_approval_response)
            self._app.router.add_get("/v1/approvals", self._handle_list_approvals)
            # Structured event streaming
            self._app.router.add_post("/v1/runs", self._handle_runs)
            self._app.router.add_get("/v1/runs/{run_id}/events", self._handle_run_events)
            # Start background sweep to clean up orphaned (unconsumed) run streams
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            # Port conflict detection — fail fast if port is already in use
            import socket as _socket
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                    _s.settimeout(1)
                    _s.connect(('127.0.0.1', self._port))
                logger.error('[%s] Port %d already in use. Set a different port in config.yaml: platforms.api_server.port', self.name, self._port)
                return False
            except (ConnectionRefusedError, OSError):
                pass  # port is free

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            self._mark_connected()
            logger.info(
                "[%s] API server listening on http://%s:%d",
                self.name, self._host, self._port,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server."""
        self._mark_disconnected()
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Not used — HTTP request/response cycle handles delivery directly.
        """
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }
