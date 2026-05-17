"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless; opt-in session continuity via X-Hermes-Session-Id header; opt-in long-term memory scoping via X-Hermes-Session-Key header)
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id; X-Hermes-Session-Key supported)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent as an available model
- GET  /v1/capabilities            — machine-readable API capabilities for external UIs
- POST /v1/runs                    — start a run, returns run_id immediately (202)
- GET  /v1/runs/{run_id}           — retrieve current run status
- GET  /v1/runs/{run_id}/events    — SSE stream of structured lifecycle events
- POST /v1/runs/{run_id}/approval — resolve a pending run approval
- POST /v1/runs/{run_id}/stop       — interrupt a running agent
- GET  /health                     — health check
- GET  /health/detailed            — rich status for cross-container dashboard probing

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
import queue as _queue
import socket as _socket
import re
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    is_network_accessible,
)
from hermes_cli.config import load_config, save_config
from hermes_cli.models import curated_models_for_provider, list_available_providers
from hermes_state import SessionDB
from tools.memory_tool import MemoryStore
from tools.skills_tool import skill_view, skills_categories, skills_list

logger = logging.getLogger(__name__)

CHAT_STREAM_HEARTBEAT_SECONDS = 15.0
RUNTIME_GOVERNOR_POLL_SECONDS = 5.0
RUNTIME_GOVERNOR_UNAVAILABLE_MESSAGE = (
    "HermesOS runtime limits could not be verified. Try again in a moment."
)
RUNTIME_GOVERNOR_LIMIT_MESSAGE = (
    "HermesOS runtime limit reached. Upgrade or wait for your allowance to reset."
)


def _encode_dashboard_sse(event_name: str, payload: Dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = int(
    os.getenv("MAX_REQUEST_BYTES", "10000000")
)  # 10 MB default limit for POST bodies / long agent conversations
CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0
MAX_NORMALIZED_TEXT_LENGTH = 65_536  # 64 KB cap for normalized content parts
MAX_CONTENT_LIST_SIZE = 1_000  # Max items when content is an array


def _coerce_port(value: Any, default: int = DEFAULT_PORT) -> int:
    """Parse a listen port without letting malformed env/config values crash startup."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_TRUE_REQUEST_BOOL_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_REQUEST_BOOL_STRINGS = frozenset({"0", "false", "no", "off"})


def _coerce_request_bool(value: Any, default: bool = False) -> bool:
    """Normalize boolean-like API payload values.

    External clients should send real JSON booleans, but some OpenAI-compatible
    frontends and middleware serialize flags like ``stream`` as strings.  Using
    Python truthiness on those values misroutes requests because ``"false"`` is
    still truthy.  Treat only explicit bool-ish scalars as booleans; everything
    else falls back to the caller's default.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_REQUEST_BOOL_STRINGS:
            return True
        if normalized in _FALSE_REQUEST_BOOL_STRINGS:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _normalize_chat_content(
    content: Any, *, _max_depth: int = 10, _depth: int = 0,
) -> str:
    """Normalize OpenAI chat message content into a plain text string.

    Some clients (Open WebUI, LobeChat, etc.) send content as an array of
    typed parts instead of a plain string::

        [{"type": "text", "text": "hello"}, {"type": "input_text", "text": "..."}]

    This function flattens those into a single string so the agent pipeline
    (which expects strings) doesn't choke.

    Defensive limits prevent abuse: recursion depth, list size, and output
    length are all bounded.
    """
    if _depth > _max_depth:
        return ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content

    if isinstance(content, list):
        parts: List[str] = []
        items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
        for item in items:
            if isinstance(item, str):
                if item:
                    parts.append(item[:MAX_NORMALIZED_TEXT_LENGTH])
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text", "")
                    if text:
                        try:
                            parts.append(str(text)[:MAX_NORMALIZED_TEXT_LENGTH])
                        except Exception:
                            pass
                # Silently skip image_url / other non-text parts
            elif isinstance(item, list):
                nested = _normalize_chat_content(item, _max_depth=_max_depth, _depth=_depth + 1)
                if nested:
                    parts.append(nested)
            # Check accumulated size
            if sum(len(p) for p in parts) >= MAX_NORMALIZED_TEXT_LENGTH:
                break
        result = "\n".join(parts)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result

    # Fallback for unexpected types (int, float, bool, etc.)
    try:
        result = str(content)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result
    except Exception:
        return ""


# Content part type aliases used by the OpenAI Chat Completions and Responses
# APIs.  We accept both spellings on input and emit a single canonical internal
# shape (``{"type": "text", ...}`` / ``{"type": "image_url", ...}``) that the
# rest of the agent pipeline already understands.
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_FILE_PART_TYPES = frozenset({"file", "input_file"})


def _normalize_multimodal_content(content: Any) -> Any:
    """Validate and normalize multimodal content for the API server.

    Returns a plain string when the content is text-only, or a list of
    ``{"type": "text"|"image_url", ...}`` parts when images are present.
    The output shape is the native OpenAI Chat Completions vision format,
    which the agent pipeline accepts verbatim (OpenAI-wire providers) or
    converts (``_preprocess_anthropic_content`` for Anthropic).

    Raises ``ValueError`` with an OpenAI-style code on invalid input:
      * ``unsupported_content_type`` — file/input_file/file_id parts, or
        non-image ``data:`` URLs.
      * ``invalid_image_url`` — missing URL or unsupported scheme.
      * ``invalid_content_part`` — malformed text/image objects.

    Callers translate the ValueError into a 400 response.
    """
    # Scalar passthrough mirrors ``_normalize_chat_content``.
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content
    if not isinstance(content, list):
        # Mirror the legacy text-normalizer's fallback so callers that
        # pre-existed image support still get a string back.
        return _normalize_chat_content(content)

    items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
    normalized_parts: List[Dict[str, Any]] = []
    text_accum_len = 0

    for part in items:
        if isinstance(part, str):
            if part:
                trimmed = part[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
                text_accum_len += len(trimmed)
            continue

        if not isinstance(part, dict):
            # Ignore unknown scalars for forward compatibility with future
            # Responses API additions (e.g. ``refusal``).  The same policy
            # the text normalizer applies.
            continue

        raw_type = part.get("type")
        part_type = str(raw_type or "").strip().lower()

        if part_type in _TEXT_PART_TYPES:
            text = part.get("text")
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if text:
                trimmed = text[:MAX_NORMALIZED_TEXT_LENGTH]
                normalized_parts.append({"type": "text", "text": trimmed})
                text_accum_len += len(trimmed)
            continue

        if part_type in _IMAGE_PART_TYPES:
            detail = part.get("detail")
            image_ref = part.get("image_url")
            # OpenAI Responses sends ``input_image`` with a top-level
            # ``image_url`` string; Chat Completions sends ``image_url`` as
            # ``{"url": "...", "detail": "..."}``.  Support both.
            if isinstance(image_ref, dict):
                url_value = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url_value = image_ref
            if not isinstance(url_value, str) or not url_value.strip():
                raise ValueError("invalid_image_url:Image parts must include a non-empty image URL.")
            url_value = url_value.strip()
            lowered = url_value.lower()
            if lowered.startswith("data:"):
                if not lowered.startswith("data:image/") or "," not in url_value:
                    raise ValueError(
                        "unsupported_content_type:Only image data URLs are supported. "
                        "Non-image data payloads are not supported."
                    )
            elif not (lowered.startswith("http://") or lowered.startswith("https://")):
                raise ValueError(
                    "invalid_image_url:Image inputs must use http(s) URLs or data:image/... URLs."
                )
            image_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value}}
            if detail is not None:
                if not isinstance(detail, str) or not detail.strip():
                    raise ValueError("invalid_content_part:Image detail must be a non-empty string when provided.")
                image_part["image_url"]["detail"] = detail.strip()
            normalized_parts.append(image_part)
            continue

        if part_type in _FILE_PART_TYPES:
            raise ValueError(
                "unsupported_content_type:Inline image inputs are supported, "
                "but uploaded files and document inputs are not supported on this endpoint."
            )

        # Unknown part type — reject explicitly so clients get a clear error
        # instead of a silently dropped turn.
        raise ValueError(
            f"unsupported_content_type:Unsupported content part type {raw_type!r}. "
            "Only text and image_url/input_image parts are supported."
        )

    if not normalized_parts:
        return ""

    # Text-only: collapse to a plain string so downstream logging/trajectory
    # code sees the native shape and prompt caching on text-only turns is
    # unaffected.
    if all(p.get("type") == "text" for p in normalized_parts):
        return "\n".join(p["text"] for p in normalized_parts if p.get("text"))

    return normalized_parts


def _content_has_visible_payload(content: Any) -> bool:
    """True when content has any text or image attachment.  Used to reject empty turns."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "").strip().lower()
                if ptype in _TEXT_PART_TYPES and str(part.get("text") or "").strip():
                    return True
                if ptype in _IMAGE_PART_TYPES:
                    return True
    return False


def _multimodal_validation_error(exc: ValueError, *, param: str) -> "web.Response":
    """Translate a ``_normalize_multimodal_content`` ValueError into a 400 response."""
    raw = str(exc)
    code, _, message = raw.partition(":")
    if not message:
        code, message = "invalid_content_part", raw
    return web.json_response(
        _openai_error(message, code=code, param=param),
        status=400,
    )


def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


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
        # Use shared WAL-fallback helper so response_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same filesystem
        # issue addressed for state.db/kanban.db — see
        # hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="response_store.db")
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
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id),
        )
        self._conn.commit()
        return json.loads(row[0])

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()),
        )
        # Evict oldest entries beyond max_size
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            # Collect IDs that will be evicted
            evict_ids = [
                row[0]
                for row in self._conn.execute(
                    "SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?",
                    (count - self._max_size,),
                ).fetchall()
            ]
            if evict_ids:
                placeholders = ",".join("?" for _ in evict_ids)
                # Clear conversation mappings pointing to evicted responses
                self._conn.execute(
                    f"DELETE FROM conversations WHERE response_id IN ({placeholders})",
                    evict_ids,
                )
                # Delete evicted responses
                self._conn.execute(
                    f"DELETE FROM responses WHERE response_id IN ({placeholders})",
                    evict_ids,
                )
        self._conn.commit()

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        # Clear conversation mappings pointing to this response
        self._conn.execute(
            "DELETE FROM conversations WHERE response_id = ?", (response_id,)
        )
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


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers for explicitly allowed origins; handle OPTIONS preflight."""
        adapter = request.app.get("api_server_adapter")
        origin = request.headers.get("Origin", "")
        cors_headers = None
        if adapter is not None:
            if not adapter._origin_allowed(origin):
                return web.Response(status=403)
            cors_headers = adapter._cors_headers_for_origin(origin)

        if request.method == "OPTIONS":
            if cors_headers is None:
                return web.Response(status=403)
            return web.Response(status=200, headers=cors_headers)

        response = await handler(request)
        if cors_headers is not None:
            response.headers.update(cors_headers)
        return response
else:
    cors_middleware = None  # type: ignore[assignment]


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


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def body_limit_middleware(request, handler):
        """Reject overly large request bodies early based on Content-Length."""
        if request.method in {"POST", "PUT", "PATCH"}:
            cl = request.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > MAX_REQUEST_BYTES:
                        return web.json_response(_openai_error("Request body too large.", code="body_too_large"), status=413)
                except ValueError:
                    return web.json_response(_openai_error("Invalid Content-Length header.", code="invalid_content_length"), status=400)
        return await handler(request)
else:
    body_limit_middleware = None  # type: ignore[assignment]

_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "no-referrer",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def security_headers_middleware(request, handler):
        """Add security headers to all responses (including errors)."""
        response = await handler(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response
else:
    security_headers_middleware = None  # type: ignore[assignment]


class _IdempotencyCache:
    """In-memory idempotency cache with TTL and basic LRU semantics."""
    def __init__(self, max_items: int = 1000, ttl_seconds: int = 300):
        from collections import OrderedDict
        self._store = OrderedDict()
        self._inflight: Dict[tuple[str, str], "asyncio.Task[Any]"] = {}
        self._ttl = ttl_seconds
        self._max = max_items

    def _purge(self):
        now = time.time()
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

        inflight_key = (key, fingerprint)
        task = self._inflight.get(inflight_key)
        if task is None:
            async def _compute_and_store():
                resp = await compute_coro()
                import time as _t
                self._store[key] = {"resp": resp, "fp": fingerprint, "ts": _t.time()}
                self._purge()
                return resp

            task = asyncio.create_task(_compute_and_store())
            self._inflight[inflight_key] = task

            def _clear_inflight(done_task: "asyncio.Task[Any]") -> None:
                if self._inflight.get(inflight_key) is done_task:
                    self._inflight.pop(inflight_key, None)

            task.add_done_callback(_clear_inflight)

        return await asyncio.shield(task)


_idem_cache = _IdempotencyCache()


def _make_request_fingerprint(body: Dict[str, Any], keys: List[str]) -> str:
    from hashlib import sha256
    subset = {k: body.get(k) for k in keys}
    return sha256(repr(subset).encode("utf-8")).hexdigest()


def _derive_chat_session_id(
    system_prompt: Optional[str],
    first_user_message: str,
) -> str:
    """Derive a stable session ID from the conversation's first user message.

    OpenAI-compatible frontends (Open WebUI, LibreChat, etc.) send the full
    conversation history with every request.  The system prompt and first user
    message are constant across all turns of the same conversation, so hashing
    them produces a deterministic session ID that lets the API server reuse
    the same Hermes session (and therefore the same Docker container sandbox
    directory) across turns.
    """
    seed = f"{system_prompt or ''}\n{first_user_message}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"api-{digest}"


_CRON_AVAILABLE = False
try:
    from cron.jobs import (
        list_jobs as _cron_list,
        get_job as _cron_get,
        create_job as _cron_create,
        update_job as _cron_update,
        remove_job as _cron_remove,
        pause_job as _cron_pause,
        resume_job as _cron_resume,
        trigger_job as _cron_trigger,
    )
    _CRON_AVAILABLE = True
except ImportError:
    _cron_list = None
    _cron_get = None
    _cron_create = None
    _cron_update = None
    _cron_remove = None
    _cron_pause = None
    _cron_resume = None
    _cron_trigger = None


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
        raw_port = extra.get("port")
        if raw_port is None:
            raw_port = os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))
        self._port: int = _coerce_port(raw_port, DEFAULT_PORT)
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")),
        )
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("API_SERVER_MODEL_NAME", "")),
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        # Active run streams: run_id -> asyncio.Queue of SSE event dicts
        self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
        # Creation timestamps for orphaned-run TTL sweep
        self._run_streams_created: Dict[str, float] = {}
        # WebUI-style dashboard chat streams: stream_id -> backend-owned run state.
        self._chat_streams: Dict[str, Dict[str, Any]] = {}
        # Active run agent/task references for stop support
        self._active_run_agents: Dict[str, Any] = {}
        self._active_run_tasks: Dict[str, "asyncio.Task"] = {}
        # Pollable run status for dashboards and external control-plane UIs.
        self._run_statuses: Dict[str, Dict[str, Any]] = {}
        # Active approval session key for each run_id.  The approval core
        # resolves requests by session key, while API clients address the
        # in-flight run by run_id.
        self._run_approval_sessions: Dict[str, str] = {}
        self._session_db: Optional[SessionDB] = None
        self._memory_store: Optional[MemoryStore] = None

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

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        """Derive the advertised model name for /v1/models.

        Priority:
        1. Explicit override (config extra or API_SERVER_MODEL_NAME env var)
        2. Active profile name (so each profile advertises a distinct model)
        3. Fallback: "hermes-agent"
        """
        if explicit and explicit.strip():
            return explicit.strip()
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile and profile not in {"default", "custom"}:
                return profile
        except Exception:
            pass
        return "hermes-agent"

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

    @staticmethod
    def _runtime_limit_result(message: str, *, reason: str = "runtime_limit") -> Dict[str, Any]:
        text = message or RUNTIME_GOVERNOR_LIMIT_MESSAGE
        return {
            "final_response": text,
            "messages": [],
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "interrupted": True,
            "failed": True,
            "runtime_cutoff": True,
            "runtime_reason": reason,
        }

    @staticmethod
    def _zero_usage() -> Dict[str, int]:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _runtime_session_key(surface: str, session_id: Optional[str]) -> str:
        safe_surface = str(surface or "api").strip() or "api"
        safe_session = str(session_id or "default").strip() or "default"
        return f"api_server:{safe_surface}:{safe_session}"

    def _admit_runtime_governor(
        self,
        *,
        surface: str,
        session_id: Optional[str],
        source_message_id: str,
        message_preview: str,
    ) -> Dict[str, Any]:
        try:
            from gateway.runtime_governor import (
                DEFAULT_UNAVAILABLE_MESSAGE,
                RuntimeGovernorError,
                get_runtime_governor,
            )

            governor = get_runtime_governor()
            decision = governor.admit(
                platform="web" if surface.startswith("session") else "api_server",
                session_key=self._runtime_session_key(surface, session_id),
                source_message_id=source_message_id,
                message_preview=message_preview[:500],
            )
            if not decision.allowed:
                return {
                    "allowed": False,
                    "governor": governor,
                    "lease_id": None,
                    "reason": decision.reason or "denied",
                    "user_message": decision.user_message or DEFAULT_UNAVAILABLE_MESSAGE,
                }
            lease_id = decision.lease_id
            if lease_id:
                try:
                    governor.start(lease_id)
                except RuntimeGovernorError as exc:
                    return {
                        "allowed": False,
                        "governor": governor,
                        "lease_id": None,
                        "reason": "policy_unavailable",
                        "user_message": exc.user_message,
                    }
            return {
                "allowed": True,
                "governor": governor,
                "lease_id": lease_id,
                "reason": decision.reason or "allowed",
                "user_message": "",
            }
        except Exception as exc:
            logger.warning(
                "[api_server] runtime governor admission failed closed for %s/%s: %s",
                surface,
                session_id or "",
                exc.__class__.__name__,
            )
            return {
                "allowed": False,
                "governor": None,
                "lease_id": None,
                "reason": "policy_unavailable",
                "user_message": RUNTIME_GOVERNOR_UNAVAILABLE_MESSAGE,
            }

    @staticmethod
    def _runtime_heartbeat(
        *,
        governor: Any,
        lease_id: Optional[str],
        agent_ref: List[Any],
        cutoff_state: Dict[str, Any],
    ) -> None:
        if not governor or not lease_id or cutoff_state.get("cutoff"):
            return
        try:
            heartbeat = governor.heartbeat(lease_id)
            should_stop = bool(getattr(heartbeat, "should_stop", False))
            user_message = str(getattr(heartbeat, "user_message", "") or "")
            reason = str(getattr(heartbeat, "reason", "") or "")
        except Exception as exc:
            logger.warning("[api_server] runtime governor heartbeat failed closed: %s", exc.__class__.__name__)
            should_stop = True
            user_message = RUNTIME_GOVERNOR_UNAVAILABLE_MESSAGE
            reason = "policy_unavailable"
        if not should_stop:
            return
        cutoff_state["cutoff"] = True
        cutoff_state["message"] = user_message or RUNTIME_GOVERNOR_LIMIT_MESSAGE
        cutoff_state["reason"] = reason or "runtime_cutoff"
        agent = agent_ref[0] if agent_ref else None
        if agent is not None and hasattr(agent, "interrupt"):
            try:
                agent.interrupt(cutoff_state["message"])
            except Exception as exc:
                logger.debug("[api_server] runtime governor interrupt failed: %s", exc)

    def _install_runtime_step_callback(
        self,
        agent: Any,
        *,
        governor: Any,
        lease_id: Optional[str],
        agent_ref: List[Any],
        cutoff_state: Dict[str, Any],
    ) -> None:
        if not governor or not lease_id:
            return
        previous = getattr(agent, "step_callback", None)

        def _runtime_step_callback(iteration, prev_tools):
            self._runtime_heartbeat(
                governor=governor,
                lease_id=lease_id,
                agent_ref=agent_ref,
                cutoff_state=cutoff_state,
            )
            if callable(previous):
                return previous(iteration, prev_tools)
            return None

        try:
            agent.step_callback = _runtime_step_callback
        except Exception as exc:
            logger.debug("[api_server] failed to install runtime step callback: %s", exc)

    async def _await_runtime_guarded(
        self,
        future: "asyncio.Future",
        *,
        governor: Any,
        lease_id: Optional[str],
        agent_ref: List[Any],
        cutoff_state: Dict[str, Any],
        cutoff_factory: Optional[Any] = None,
    ) -> Any:
        if not governor or not lease_id:
            return await future

        while True:
            done, _ = await asyncio.wait({future}, timeout=RUNTIME_GOVERNOR_POLL_SECONDS)
            if done:
                return future.result()
            self._runtime_heartbeat(
                governor=governor,
                lease_id=lease_id,
                agent_ref=agent_ref,
                cutoff_state=cutoff_state,
            )
            if cutoff_state.get("cutoff"):
                cutoff_result = self._runtime_limit_result(
                    str(cutoff_state.get("message") or ""),
                    reason=str(cutoff_state.get("reason") or "runtime_cutoff"),
                )
                if cutoff_factory is not None:
                    return cutoff_factory(cutoff_result)
                return (cutoff_result, self._zero_usage())

    @staticmethod
    def _close_runtime_lease(governor: Any, lease_id: Optional[str], *, result: Optional[Dict[str, Any]], error: bool = False) -> None:
        if not governor or not lease_id:
            return
        try:
            if result and result.get("runtime_cutoff"):
                governor.fail(lease_id, reason=str(result.get("runtime_reason") or "runtime_cutoff"))
            elif error or (result and result.get("failed")):
                governor.fail(lease_id, reason="agent_error")
            else:
                governor.finish(lease_id, reason="agent_end")
        except Exception as exc:
            logger.warning("[api_server] runtime governor lease close failed: %s", exc.__class__.__name__)

    async def _run_agent_with_runtime(
        self,
        *,
        surface: str,
        runtime_session_id: Optional[str],
        source_message_id: str,
        message_preview: str,
        **run_kwargs,
    ) -> tuple:
        admission = self._admit_runtime_governor(
            surface=surface,
            session_id=runtime_session_id,
            source_message_id=source_message_id,
            message_preview=message_preview,
        )
        if not admission.get("allowed"):
            return (
                self._runtime_limit_result(
                    str(admission.get("user_message") or ""),
                    reason=str(admission.get("reason") or "denied"),
                ),
                self._zero_usage(),
            )

        governor = admission.get("governor")
        lease_id = admission.get("lease_id")
        result: Optional[Dict[str, Any]] = None
        try:
            result, usage = await self._run_agent(
                **run_kwargs,
                runtime_governor=governor,
                runtime_lease_id=lease_id,
            )
            return result, usage
        except Exception:
            self._close_runtime_lease(governor, lease_id, result=result, error=True)
            raise
        finally:
            if result is not None:
                self._close_runtime_lease(governor, lease_id, result=result)

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        If no API key is configured, all requests are allowed (only when API
        server is local).
        """
        if not self._api_key:
            return None  # No key configured — allow all (local-only use)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._api_key):
                return None  # Auth OK

        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    # ------------------------------------------------------------------
    # Session header helpers
    # ------------------------------------------------------------------

    # Soft length cap for session identifiers.  Headers are bounded in
    # aggregate by aiohttp (``client_max_size`` / default 8 KiB per
    # header), but we impose a tighter limit on the session headers so a
    # caller can't burn memory by passing a multi-kilobyte "session key".
    # 256 chars is well above any realistic stable channel identifier
    # (e.g. ``agent:main:webui:dm:user-42``) while staying small enough
    # that the sanitized form is safe to pass into Honcho / state.db.
    _MAX_SESSION_HEADER_LEN = 256

    def _parse_session_key_header(
        self, request: "web.Request"
    ) -> tuple[Optional[str], Optional["web.Response"]]:
        """Extract and validate the ``X-Hermes-Session-Key`` header.

        The session key is a stable per-channel identifier that scopes
        long-term memory (e.g. Honcho sessions) across transcripts.  It
        is independent of ``X-Hermes-Session-Id``: callers may send
        either, both, or neither.

        Returns ``(session_key, None)`` on success (with an empty/absent
        header yielding ``None`` for the key), or ``(None, error_response)``
        on validation failure.

        Security: like session continuation, accepting a caller-supplied
        memory scope requires API-key authentication so that an
        unauthenticated client on a local-only server can't inject itself
        into another user's long-term memory scope by guessing a key.
        """
        raw = request.headers.get("X-Hermes-Session-Key", "").strip()
        if not raw:
            return None, None

        if not self._api_key:
            logger.warning(
                "X-Hermes-Session-Key rejected: no API key configured. "
                "Set API_SERVER_KEY to enable long-term memory scoping."
            )
            return None, web.json_response(
                _openai_error(
                    "X-Hermes-Session-Key requires API key authentication. "
                    "Configure API_SERVER_KEY to enable this feature."
                ),
                status=403,
            )

        # Reject control characters that could enable header injection on
        # the echo path.
        if re.search(r'[\r\n\x00]', raw):
            return None, web.json_response(
                {"error": {"message": "Invalid session key", "type": "invalid_request_error"}},
                status=400,
            )

        if len(raw) > self._MAX_SESSION_HEADER_LEN:
            return None, web.json_response(
                {"error": {"message": "Session key too long", "type": "invalid_request_error"}},
                status=400,
            )

        return raw, None

    # ------------------------------------------------------------------
    # Session DB helper
    # ------------------------------------------------------------------

    def _get_session_db(self) -> SessionDB:
        """Create the session DB lazily."""
        if self._session_db is None:
            self._session_db = SessionDB()
        return self._session_db

    def _get_memory_store(self) -> MemoryStore:
        """Create the memory store lazily."""
        if self._memory_store is None:
            self._memory_store = MemoryStore()
            self._memory_store.load_from_disk()
        return self._memory_store

    @staticmethod
    def _normalize_session_record(session: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Parse serialized session fields into API-friendly JSON."""
        if session is None:
            return None
        normalized = dict(session)
        model_config = normalized.get("model_config")
        if model_config:
            try:
                normalized["model_config"] = json.loads(model_config)
            except (TypeError, json.JSONDecodeError):
                pass
        return normalized

    @staticmethod
    def _current_model_settings(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model/provider/base_url/api_mode from config.yaml."""
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            return {
                "model": str(model_cfg.get("default") or model_cfg.get("model") or "").strip(),
                "provider": str(model_cfg.get("provider") or "").strip(),
                "api_mode": str(model_cfg.get("api_mode") or "").strip(),
                "base_url": str(model_cfg.get("base_url") or "").strip(),
            }
        if isinstance(model_cfg, str):
            return {
                "model": model_cfg.strip(),
                "provider": "",
                "api_mode": "",
                "base_url": "",
            }
        return {"model": "", "provider": "", "api_mode": "", "base_url": ""}

    @staticmethod
    def _parse_int(value: Any, default: int, minimum: int = 0) -> int:
        """Parse an integer query parameter with bounds."""
        if value in (None, ""):
            return default
        parsed = int(value)
        if parsed < minimum:
            raise ValueError(f"Value must be >= {minimum}")
        return parsed

    @staticmethod
    def _build_user_content(
        text: str, attachments: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:
        """Build multimodal content from text + image attachments.

        Returns (user_content, persist_text) where user_content is either
        a plain string or a list of content parts for multimodal input.
        """
        if not attachments:
            return text, text

        image_parts: List[Dict[str, Any]] = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            mime = ""
            for key in ("contentType", "mimeType", "mediaType"):
                val = att.get(key)
                if isinstance(val, str) and val.strip():
                    mime = val.strip()
                    break
            if not mime.startswith("image/"):
                continue
            content = ""
            for key in ("content", "base64", "data"):
                val = att.get(key)
                if isinstance(val, str) and val.strip():
                    content = val.strip()
                    break
            if not content:
                data_url = att.get("dataUrl", "")
                if isinstance(data_url, str) and data_url.startswith("data:"):
                    content = data_url.split(",", 1)[-1] if "," in data_url else ""
            if not content:
                continue
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{content}"},
            })

        if not image_parts:
            return text, text

        content_parts: List[Dict[str, Any]] = []
        if text.strip():
            content_parts.append({"type": "text", "text": text})
        content_parts.extend(image_parts)
        return content_parts, text

    def _ensure_session_db(self):
        """Lazily initialise and return the shared SessionDB instance.

        Sessions are persisted to ``state.db`` so that ``hermes sessions list``
        shows API-server conversations alongside CLI and gateway ones.
        """
        if self._session_db is None:
            try:
                self._session_db = SessionDB()
            except Exception as e:
                logger.debug("SessionDB unavailable for API server: %s", e)
        return self._session_db

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        thinking_callback=None,
        reasoning_callback=None,
        status_callback=None,
        gateway_session_key: Optional[str] = None,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.

        ``gateway_session_key`` is a stable per-channel identifier supplied
        by the client (via ``X-Hermes-Session-Key``).  Unlike ``session_id``
        which scopes the short-term transcript and rotates on /new, this
        key is meant to persist across transcripts so long-term memory
        providers (e.g. Honcho) can scope their per-chat state correctly
        — matching the semantics of the native gateway's ``session_key``.
        """
        from run_agent import AIAgent
        from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model, _load_gateway_config, GatewayRunner
        from hermes_cli.tools_config import _get_platform_tools

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        reasoning_config = GatewayRunner._load_reasoning_config()
        model = _resolve_gateway_model()

        user_config = _load_gateway_config()
        enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

        # Load fallback provider chain so the API server platform has the
        # same fallback behaviour as Telegram/Discord/Slack (fixes #4954).
        fallback_model = GatewayRunner._load_fallback_model()

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
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            thinking_callback=thinking_callback,
            reasoning_callback=reasoning_callback,
            status_callback=status_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
            reasoning_config=reasoning_config,
            gateway_session_key=gateway_session_key,
        )
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dashboard REST API handlers
    # ------------------------------------------------------------------

    async def _handle_list_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions — list sessions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            limit = self._parse_int(request.query.get("limit"), 50)
            offset = self._parse_int(request.query.get("offset"), 0)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        source = (request.query.get("source") or "").strip() or None
        db = self._get_session_db()
        items = [
            self._normalize_session_record(item)
            for item in db.list_sessions_rich(source=source, limit=limit, offset=offset)
        ]
        total = db.session_count(source=source)
        return web.json_response({"items": items, "total": total})

    async def _handle_create_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions — create a new session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        title = body.get("title")
        source = str(body.get("source") or "api_server").strip() or "api_server"
        model = body.get("model")
        system_prompt = body.get("system_prompt")
        session_id = f"sess_{uuid.uuid4().hex}"
        db = self._get_session_db()

        try:
            db.create_session(
                session_id=session_id,
                source=source,
                model=model,
                system_prompt=system_prompt,
            )
            if title is not None:
                db.set_session_title(session_id, str(title))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        session = self._normalize_session_record(db.get_session(session_id))
        return web.json_response({"session": session})

    async def _handle_search_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/search — search messages across sessions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        query = (request.query.get("q") or "").strip()
        if not query:
            return web.json_response({"error": "Missing query parameter: q"}, status=400)
        try:
            limit = self._parse_int(request.query.get("limit"), 20)
            offset = self._parse_int(request.query.get("offset"), 0)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        results = self._get_session_db().search_messages(query=query, limit=limit, offset=offset)
        return web.json_response({"query": query, "count": len(results), "results": results})

    async def _handle_get_session(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id} — fetch one session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        session = self._normalize_session_record(self._get_session_db().get_session(session_id))
        if session is None:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"session": session})

    async def _handle_get_session_messages(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}/messages — fetch session messages."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        if db.get_session(session_id) is None:
            db.ensure_session(session_id, source="web")
        items = db.get_messages(session_id)
        return web.json_response({"items": items, "total": len(items)})

    async def _handle_update_session(self, request: "web.Request") -> "web.Response":
        """PATCH /api/sessions/{session_id} — update a session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        if db.get_session(session_id) is None:
            return web.json_response({"error": "Session not found"}, status=404)
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        try:
            if "title" in body:
                db.set_session_title(session_id, body.get("title"))
            if "system_prompt" in body:
                db.update_system_prompt(session_id, body.get("system_prompt"))
            if "end_reason" in body:
                db.end_session(session_id, str(body.get("end_reason") or "updated"))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        session = self._normalize_session_record(db.get_session(session_id))
        return web.json_response({"session": session})

    async def _handle_delete_session(self, request: "web.Request") -> "web.Response":
        """DELETE /api/sessions/{session_id} — delete a session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        deleted = self._get_session_db().delete_session(session_id)
        if not deleted:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"ok": True})

    async def _handle_fork_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/fork — clone a session and its messages."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        original = db.get_session(session_id)
        if original is None:
            return web.json_response({"error": "Session not found"}, status=404)

        forked_id = f"sess_{uuid.uuid4().hex}"
        try:
            db.create_session(
                session_id=forked_id,
                source=original.get("source") or "api_server",
                model=original.get("model"),
                system_prompt=original.get("system_prompt"),
                user_id=original.get("user_id"),
                parent_session_id=session_id,
            )
            messages = db.get_messages(session_id)
            for message in messages:
                db.append_message(
                    session_id=forked_id,
                    role=message.get("role"),
                    content=message.get("content"),
                    tool_name=message.get("tool_name"),
                    tool_calls=message.get("tool_calls"),
                    tool_call_id=message.get("tool_call_id"),
                    token_count=message.get("token_count"),
                    finish_reason=message.get("finish_reason"),
                    reasoning=message.get("reasoning"),
                )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        session = self._normalize_session_record(db.get_session(forked_id))
        return web.json_response({"session": session, "forked_from": session_id})

    async def _handle_session_chat(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/chat — run a session-aware chat turn."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        session = self._normalize_session_record(db.get_session(session_id))
        if session is None:
            db.ensure_session(session_id, source="web")
            session = self._normalize_session_record(db.get_session(session_id)) or {}

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        message = body.get("message")
        if not isinstance(message, str):
            return web.json_response({"error": "Missing or invalid 'message' field"}, status=400)

        raw_attachments_sync = body.get("attachments")
        if raw_attachments_sync:
            logger.debug("[chat] Received %d attachment(s): %s",
                         len(raw_attachments_sync),
                         [(a.get("name"), a.get("contentType"), len(a.get("content", "") or a.get("base64", "") or "")) for a in raw_attachments_sync if isinstance(a, dict)])
        user_content, persist_text = self._build_user_content(message, raw_attachments_sync)
        if isinstance(user_content, list):
            logger.debug("[chat] Built multimodal content with %d parts", len(user_content))

        model = body.get("model") or session.get("model") or "hermes-agent"
        system_message = body.get("system_message")
        history = db.get_messages_as_conversation(session_id)
        loop = asyncio.get_event_loop()
        run_id = f"run_{uuid.uuid4().hex}"
        admission = self._admit_runtime_governor(
            surface="session_chat",
            session_id=session_id,
            source_message_id=run_id,
            message_preview=message,
        )

        def _run():
            agent = self._create_agent(
                ephemeral_system_prompt=system_message,
                session_id=session_id,
            )
            agent._session_db = db
            agent_ref[0] = agent
            self._install_runtime_step_callback(
                agent,
                governor=governor,
                lease_id=lease_id,
                agent_ref=agent_ref,
                cutoff_state=cutoff_state,
            )
            result = agent.run_conversation(
                user_content,
                conversation_history=history,
                persist_user_message=persist_text,
            )
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return result, usage

        if not admission.get("allowed"):
            result = self._runtime_limit_result(
                str(admission.get("user_message") or ""),
                reason=str(admission.get("reason") or "denied"),
            )
            usage = self._zero_usage()
        else:
            governor = admission.get("governor")
            lease_id = admission.get("lease_id")
            agent_ref: List[Any] = [None]
            cutoff_state: Dict[str, Any] = {}
            result: Optional[Dict[str, Any]] = None
            try:
                future = asyncio.ensure_future(loop.run_in_executor(None, _run))
                result, usage = await self._await_runtime_guarded(
                    future,
                    governor=governor,
                    lease_id=lease_id,
                    agent_ref=agent_ref,
                    cutoff_state=cutoff_state,
                )
            except Exception as e:
                self._close_runtime_lease(governor, lease_id, result=result, error=True)
                logger.error("Error running session chat for %s: %s", session_id, e, exc_info=True)
                return web.json_response({"error": str(e)}, status=500)
            finally:
                if result is not None:
                    self._close_runtime_lease(governor, lease_id, result=result)

        return web.json_response({
            "session_id": session_id,
            "run_id": run_id,
            "model": model,
            "final_response": result.get("final_response"),
            "completed": result.get("completed", False),
            "partial": result.get("partial", False),
            "interrupted": result.get("interrupted", False),
            "api_calls": result.get("api_calls", 0),
            "messages": result.get("messages", []),
            "last_reasoning": result.get("last_reasoning"),
            "response_previewed": result.get("response_previewed", False),
            "usage": usage,
        })

    async def _handle_session_chat_stream(self, request: "web.Request") -> "web.StreamResponse":
        """POST /api/sessions/{session_id}/chat/stream — stream a session chat turn over SSE."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        session = self._normalize_session_record(db.get_session(session_id))
        if session is None:
            db.ensure_session(session_id, source="web")
            session = self._normalize_session_record(db.get_session(session_id)) or {}

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        message = body.get("message")
        if not isinstance(message, str):
            return web.json_response({"error": "Missing or invalid 'message' field"}, status=400)

        raw_attachments = body.get("attachments")
        if raw_attachments:
            logger.debug("[chat/stream] Received %d attachment(s): %s",
                         len(raw_attachments),
                         [(a.get("name"), a.get("contentType"), len(a.get("content", "") or a.get("base64", "") or "")) for a in raw_attachments if isinstance(a, dict)])
        user_content, persist_text = self._build_user_content(message, raw_attachments)
        if isinstance(user_content, list):
            logger.debug("[chat/stream] Built multimodal content with %d parts", len(user_content))

        system_message = body.get("system_message")
        history = db.get_messages_as_conversation(session_id)
        assistant_message_id = f"msg_asst_{uuid.uuid4().hex}"

        import queue as _q
        stream_q: _q.Queue = _q.Queue()

        def _encode_sse(event_name: str, payload: Dict[str, Any]) -> bytes:
            return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        def _queue_event(event_name: str, payload: Dict[str, Any]) -> None:
            stream_q.put(_encode_sse(event_name, payload))

        def _tool_map(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            mapping: Dict[str, Dict[str, Any]] = {}
            for item in messages:
                if item.get("role") != "assistant":
                    continue
                for index, tool_call in enumerate(item.get("tool_calls") or []):
                    tool_id = tool_call.get("id")
                    if not tool_id:
                        continue
                    fn = tool_call.get("function") or {}
                    raw_args = fn.get("arguments")
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
                    except json.JSONDecodeError:
                        parsed_args = raw_args
                    mapping[tool_id] = {
                        "tool_name": fn.get("name") or item.get("tool_name") or f"tool_{index + 1}",
                        "args": parsed_args,
                    }
            return mapping

        def _result_preview(content: Any, limit: int = 4000) -> str:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            return text[:limit] + ("..." if len(text) > limit else "")

        run_id = f"run_{uuid.uuid4().hex}"

        def _on_delta(delta):
            if delta:
                _queue_event(
                    "assistant.delta",
                    {"session_id": session_id, "run_id": run_id, "message_id": assistant_message_id, "delta": delta},
                )

        def _on_tool_progress(name, preview, args):
            if name == "_thinking":
                _queue_event(
                    "tool.progress",
                    {"session_id": session_id, "run_id": run_id, "message_id": assistant_message_id, "delta": preview},
                )
                return
            payload = {
                "session_id": session_id,
                "run_id": run_id,
                "tool_name": name,
                "preview": preview,
                "args": args,
            }
            _queue_event("tool.started", payload)

        def _on_approval_request(approval_data):
            if not isinstance(approval_data, dict):
                approval_data = {"description": str(approval_data)}
            payload = {
                "session_id": session_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "approval_id": str(approval_data.get("approval_id") or approval_data.get("id") or ""),
                "command": str(approval_data.get("command") or "")[:8000],
                "description": str(approval_data.get("description") or "Command approval required")[:1000],
                "pattern_key": str(approval_data.get("pattern_key") or ""),
                "pattern_keys": approval_data.get("pattern_keys") if isinstance(approval_data.get("pattern_keys"), list) else [],
            }
            _queue_event("approval", payload)

        agent_ref = [None]
        loop = asyncio.get_event_loop()

        async def _run_agent_task():
            admission = self._admit_runtime_governor(
                surface="session_chat_stream",
                session_id=session_id,
                source_message_id=run_id,
                message_preview=message,
            )
            if not admission.get("allowed"):
                return self._runtime_limit_result(
                    str(admission.get("user_message") or ""),
                    reason=str(admission.get("reason") or "denied"),
                )

            governor = admission.get("governor")
            lease_id = admission.get("lease_id")
            cutoff_state: Dict[str, Any] = {}
            result: Optional[Dict[str, Any]] = None

            def _run():
                approval_token = None
                reset_current_session_key = None
                unregister_gateway_notify = None
                try:
                    from tools.approval import (
                        register_gateway_notify,
                        reset_current_session_key as _reset_current_session_key,
                        set_current_session_key,
                        unregister_gateway_notify as _unregister_gateway_notify,
                    )
                    approval_token = set_current_session_key(session_id)
                    reset_current_session_key = _reset_current_session_key
                    unregister_gateway_notify = _unregister_gateway_notify
                    register_gateway_notify(session_id, _on_approval_request)
                except Exception as exc:
                    logger.debug("[chat/stream] Approval callbacks unavailable: %s", exc)

                try:
                    agent = self._create_agent(
                        ephemeral_system_prompt=system_message,
                        session_id=session_id,
                        stream_delta_callback=_on_delta,
                        tool_progress_callback=_on_tool_progress,
                    )
                    agent._session_db = db
                    agent_ref[0] = agent
                    self._install_runtime_step_callback(
                        agent,
                        governor=governor,
                        lease_id=lease_id,
                        agent_ref=agent_ref,
                        cutoff_state=cutoff_state,
                    )
                    return agent.run_conversation(
                        user_content,
                        conversation_history=history,
                        persist_user_message=persist_text,
                    )
                finally:
                    if unregister_gateway_notify is not None:
                        try:
                            unregister_gateway_notify(session_id)
                        except Exception as exc:
                            logger.debug("[chat/stream] Approval callback cleanup failed: %s", exc)
                    if approval_token is not None and reset_current_session_key is not None:
                        try:
                            reset_current_session_key(approval_token)
                        except Exception as exc:
                            logger.debug("[chat/stream] Approval session cleanup failed: %s", exc)

            try:
                future = asyncio.ensure_future(loop.run_in_executor(None, _run))
                result = await self._await_runtime_guarded(
                    future,
                    governor=governor,
                    lease_id=lease_id,
                    agent_ref=agent_ref,
                    cutoff_state=cutoff_state,
                    cutoff_factory=lambda cutoff_result: cutoff_result,
                )
                return result
            except Exception:
                self._close_runtime_lease(governor, lease_id, result=result, error=True)
                raise
            finally:
                if result is not None:
                    self._close_runtime_lease(governor, lease_id, result=result)

        agent_task = asyncio.ensure_future(_run_agent_task())

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)

        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        try:
            user_message_id = f"msg_user_{uuid.uuid4().hex}"
            await response.write(_encode_sse("session.created", {
                "session_id": session_id,
                "run_id": run_id,
                "title": session.get("title") or "New Chat",
            }))
            await response.write(_encode_sse("run.started", {
                "session_id": session_id,
                "run_id": run_id,
                "user_message": {
                    "id": user_message_id,
                    "role": "user",
                    "content": message,
                },
            }))
            await response.write(_encode_sse("message.started", {
                "session_id": session_id,
                "run_id": run_id,
                "message": {"id": assistant_message_id, "role": "assistant"},
            }))

            while True:
                try:
                    frame = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        while True:
                            try:
                                frame = stream_q.get_nowait()
                                if frame is None:
                                    break
                                await response.write(frame)
                            except _q.Empty:
                                break
                        break
                    continue

                if frame is None:
                    break

                await response.write(frame)

            result = await agent_task
            tools = _tool_map(result.get("messages") or [])
            for item in result.get("messages") or []:
                if item.get("role") != "tool":
                    continue
                tool_id = item.get("tool_call_id")
                tool_meta = tools.get(tool_id, {})
                await response.write(_encode_sse("tool.completed", {
                    "session_id": session_id,
                    "run_id": run_id,
                    "tool_call_id": tool_id,
                    "tool_name": tool_meta.get("tool_name") or item.get("tool_name") or "unknown",
                    "args": tool_meta.get("args"),
                    "result_preview": _result_preview(item.get("content")),
                }))

            await response.write(_encode_sse("assistant.completed", {
                "session_id": session_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "content": result.get("final_response") or "",
                "completed": result.get("completed", False),
                "partial": result.get("partial", False),
                "interrupted": result.get("interrupted", False),
            }))
            await response.write(_encode_sse("run.completed", {
                "session_id": session_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "completed": result.get("completed", False),
                "partial": result.get("partial", False),
                "interrupted": result.get("interrupted", False),
                "api_calls": result.get("api_calls"),
            }))
            await response.write(_encode_sse("done", {"session_id": session_id, "run_id": run_id, "state": "final"}))
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            agent = agent_ref[0]
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("Session SSE client disconnected; interrupted session %s", session_id)

        return response

    async def _handle_session_chat_start(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/chat/start — start a backend-owned WebUI-style stream."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        session = self._normalize_session_record(db.get_session(session_id))
        if session is None:
            db.ensure_session(session_id, source="web")
            session = self._normalize_session_record(db.get_session(session_id)) or {}

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        message = body.get("message")
        if not isinstance(message, str):
            return web.json_response({"error": "Missing or invalid 'message' field"}, status=400)

        raw_attachments = body.get("attachments")
        user_content, persist_text = self._build_user_content(message, raw_attachments)
        system_message = body.get("system_message")
        history = db.get_messages_as_conversation(session_id)

        import queue as _q
        stream_id = uuid.uuid4().hex
        run_id = f"run_{uuid.uuid4().hex}"
        assistant_message_id = f"msg_asst_{uuid.uuid4().hex}"
        user_message_id = f"msg_user_{uuid.uuid4().hex}"
        stream_q: _q.Queue = _q.Queue()
        agent_ref: List[Any] = [None]
        loop = asyncio.get_event_loop()
        stream_state: Dict[str, Any] = {}

        def _encode_sse(event_name: str, payload: Dict[str, Any]) -> bytes:
            return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        def _queue_event(event_name: str, payload: Dict[str, Any]) -> None:
            _record_runtime_event(event_name, payload)
            stream_q.put(_encode_sse(event_name, payload))

        def _record_runtime_event(event_name: str, payload: Dict[str, Any]) -> None:
            if not stream_state:
                return
            lock = stream_state.get("lock")

            def _record() -> None:
                stream_state["last_event_at"] = time.time()
                events = stream_state.setdefault("events", [])
                events.append({"event": event_name, "payload": payload})
                del events[:-200]

            if lock is not None:
                with lock:
                    _record()
            else:
                _record()

        def _snapshot_args(args: Any) -> Dict[str, str]:
            if not isinstance(args, dict):
                return {}
            args_snap: Dict[str, str] = {}
            for key, value in list(args.items())[:4]:
                value_text = str(value)
                args_snap[key] = value_text[:120] + ("..." if len(value_text) > 120 else "")
            return args_snap

        def _tool_context(name: Any, args: Any) -> str:
            try:
                from agent.display import build_tool_preview
                preview = build_tool_preview(str(name or "tool"), args if isinstance(args, dict) else {}, max_len=80)
                if preview:
                    return str(preview)
            except Exception as exc:
                logger.debug("[chat/start] Tool preview builder unavailable: %s", exc)
            if isinstance(args, dict):
                command = args.get("command")
                if command:
                    return str(command)
            return ""

        def _update_runtime(mutator) -> None:
            lock = stream_state.get("lock")
            if lock is not None:
                with lock:
                    mutator()
            else:
                mutator()

        def _tool_map(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            mapping: Dict[str, Dict[str, Any]] = {}
            for item in messages:
                if item.get("role") != "assistant":
                    continue
                for index, tool_call in enumerate(item.get("tool_calls") or []):
                    tool_id = tool_call.get("id")
                    if not tool_id:
                        continue
                    fn = tool_call.get("function") or {}
                    raw_args = fn.get("arguments")
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
                    except json.JSONDecodeError:
                        parsed_args = raw_args
                    mapping[tool_id] = {
                        "tool_name": fn.get("name") or item.get("tool_name") or f"tool_{index + 1}",
                        "args": parsed_args,
                    }
            return mapping

        def _result_preview(content: Any, limit: int = 4000) -> str:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            return text[:limit] + ("..." if len(text) > limit else "")

        def _on_delta(delta):
            if delta:
                def _mutate() -> None:
                    stream_state["assistant_content"] = str(stream_state.get("assistant_content") or "") + str(delta)

                _update_runtime(_mutate)
                _queue_event(
                    "token",
                    {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "message_id": assistant_message_id, "text": delta},
                )
                _queue_event(
                    "message.delta",
                    {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "message_id": assistant_message_id, "text": delta},
                )

        def _on_thinking(text):
            text = str(text or "")

            def _mutate() -> None:
                stream_state["thinking"] = text

            _update_runtime(_mutate)
            _queue_event(
                "thinking.delta",
                {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "message_id": assistant_message_id, "text": text},
            )

        def _on_reasoning(text):
            text = str(text or "")
            if not text:
                return

            def _mutate() -> None:
                stream_state["reasoning"] = str(stream_state.get("reasoning") or "") + text

            _update_runtime(_mutate)
            _queue_event(
                "reasoning.delta",
                {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "message_id": assistant_message_id, "text": text},
            )

        def _on_status(kind, text=None):
            body = str(text if text is not None else kind).strip()
            if not body:
                return
            payload = {
                "session_id": session_id,
                "stream_id": stream_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "kind": str(kind if text is not None else "status"),
                "text": body,
            }

            def _mutate() -> None:
                stream_state["status"] = {"kind": payload["kind"], "text": body}

            _update_runtime(_mutate)
            _queue_event("status.update", payload)

        def _on_tool_start(tool_call_id, name, args):
            tool_id = str(tool_call_id or f"tool_{uuid.uuid4().hex[:8]}")
            tool_name = str(name or "tool")
            context = _tool_context(tool_name, args)
            args_snap = _snapshot_args(args)
            tui_payload = {
                "session_id": session_id,
                "stream_id": stream_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "tool_id": tool_id,
                "name": tool_name,
                "context": context,
            }
            compat_payload = {
                "session_id": session_id,
                "stream_id": stream_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "tool_call_id": tool_id,
                "event_type": "tool.started",
                "name": tool_name,
                "preview": context,
                "args": args_snap,
            }

            def _mutate() -> None:
                active_tools = [
                    tool for tool in stream_state.setdefault("active_tools", [])
                    if tool.get("tool_id") != tool_id
                ]
                active_tools.append({"tool_id": tool_id, "name": tool_name, "context": context})
                stream_state["active_tools"] = active_tools

            _update_runtime(_mutate)
            _queue_event("tool.start", tui_payload)
            _queue_event("tool", compat_payload)

        def _on_tool_complete(tool_call_id, name, args, result):
            tool_id = str(tool_call_id or "")
            tool_name = str(name or "tool")
            context = _tool_context(tool_name, args)
            result_preview = _result_preview(result)
            args_snap = _snapshot_args(args)
            tui_payload = {
                "session_id": session_id,
                "stream_id": stream_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "tool_id": tool_id,
                "name": tool_name,
                "summary": result_preview,
            }
            compat_payload = {
                "session_id": session_id,
                "stream_id": stream_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "tool_call_id": tool_id,
                "tool_name": tool_name,
                "name": tool_name,
                "args": args_snap,
                "result_preview": result_preview,
                "preview": context,
            }

            def _mutate() -> None:
                stream_state["active_tools"] = [
                    tool for tool in stream_state.setdefault("active_tools", [])
                    if tool.get("tool_id") != tool_id
                ]

            _update_runtime(_mutate)
            _queue_event("tool.complete", tui_payload)
            _queue_event("tool.completed", compat_payload)

        def _on_tool_progress(*cb_args, **cb_kwargs):
            event_type = None
            name = None
            preview = None
            args = None
            if len(cb_args) >= 4:
                event_type, name, preview, args = cb_args[:4]
            elif len(cb_args) == 3:
                name, preview, args = cb_args
                event_type = "tool.started"
            elif len(cb_args) == 2:
                event_type, name = cb_args
            elif len(cb_args) == 1:
                name = cb_args[0]
                event_type = "tool.started"
            if event_type in ("reasoning.available", "_thinking"):
                reason_text = preview if event_type == "reasoning.available" else name
                if reason_text:
                    _queue_event("reasoning", {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "text": str(reason_text)})
                return
            args_snap = _snapshot_args(args)
            if event_type in (None, "tool.started"):
                _queue_event(
                    "tool",
                    {
                        "session_id": session_id,
                        "stream_id": stream_id,
                        "run_id": run_id,
                        "message_id": assistant_message_id,
                        "event_type": event_type or "tool.started",
                        "name": name,
                        "preview": preview,
                        "args": args_snap,
                    },
                )
            elif event_type == "tool.completed":
                _queue_event(
                    "tool_complete",
                    {
                        "session_id": session_id,
                        "stream_id": stream_id,
                        "run_id": run_id,
                        "message_id": assistant_message_id,
                        "event_type": event_type,
                        "name": name,
                        "preview": preview,
                        "args": args_snap,
                        "duration": cb_kwargs.get("duration"),
                        "is_error": bool(cb_kwargs.get("is_error", False)),
                    },
                )

        def _on_approval_request(approval_data):
            if not isinstance(approval_data, dict):
                approval_data = {"description": str(approval_data)}
            payload = {
                "session_id": session_id,
                "stream_id": stream_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "approval_id": str(approval_data.get("approval_id") or approval_data.get("id") or ""),
                "command": str(approval_data.get("command") or "")[:8000],
                "description": str(approval_data.get("description") or "Command approval required")[:1000],
                "pattern_key": str(approval_data.get("pattern_key") or ""),
                "pattern_keys": approval_data.get("pattern_keys") if isinstance(approval_data.get("pattern_keys"), list) else [],
            }

            def _mutate() -> None:
                stream_state["pending_approval"] = payload
                stream_state["status"] = {"kind": "approval", "text": payload["description"]}

            _update_runtime(_mutate)
            _queue_event("approval.request", payload)
            _queue_event("approval", payload)

        # Mirrors the TUI runtime state in memory so web refresh/reconnect can render active work.
        stream_state.update({
            "session_id": session_id,
            "stream_id": stream_id,
            "run_id": run_id,
            "message_id": assistant_message_id,
            "queue": stream_q,
            "agent_ref": agent_ref,
            "created_at": time.time(),
            "last_event_at": time.time(),
            "assistant_content": "",
            "reasoning": "",
            "thinking": "",
            "status": {"kind": "status", "text": "running"},
            "active_tools": [],
            "pending_approval": None,
            "events": [],
            "lock": threading.Lock(),
            "active": True,
        })
        self._chat_streams[stream_id] = stream_state

        _queue_event("session.created", {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "title": session.get("title") or "New Chat"})
        _queue_event("run.started", {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "user_message": {"id": user_message_id, "role": "user", "content": message}})
        _queue_event("message.started", {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "message": {"id": assistant_message_id, "role": "assistant"}})

        async def _run_agent_task():
            admission = self._admit_runtime_governor(
                surface="session_chat_start",
                session_id=session_id,
                source_message_id=run_id,
                message_preview=message,
            )
            if not admission.get("allowed"):
                return self._runtime_limit_result(
                    str(admission.get("user_message") or ""),
                    reason=str(admission.get("reason") or "denied"),
                )

            governor = admission.get("governor")
            lease_id = admission.get("lease_id")
            cutoff_state: Dict[str, Any] = {}
            result: Optional[Dict[str, Any]] = None

            def _run():
                approval_token = None
                reset_current_session_key = None
                unregister_gateway_notify = None
                try:
                    from tools.approval import (
                        register_gateway_notify,
                        reset_current_session_key as _reset_current_session_key,
                        set_current_session_key,
                        unregister_gateway_notify as _unregister_gateway_notify,
                    )
                    approval_token = set_current_session_key(session_id)
                    reset_current_session_key = _reset_current_session_key
                    unregister_gateway_notify = _unregister_gateway_notify
                    register_gateway_notify(session_id, _on_approval_request)
                except Exception as exc:
                    logger.debug("[chat/start] Approval callbacks unavailable: %s", exc)

                try:
                    agent = self._create_agent(
                        ephemeral_system_prompt=system_message,
                        session_id=session_id,
                        stream_delta_callback=_on_delta,
                        tool_progress_callback=_on_tool_progress,
                        tool_start_callback=_on_tool_start,
                        tool_complete_callback=_on_tool_complete,
                        thinking_callback=_on_thinking,
                        reasoning_callback=_on_reasoning,
                        status_callback=_on_status,
                    )
                    agent._session_db = db
                    agent_ref[0] = agent
                    self._install_runtime_step_callback(
                        agent,
                        governor=governor,
                        lease_id=lease_id,
                        agent_ref=agent_ref,
                        cutoff_state=cutoff_state,
                    )
                    return agent.run_conversation(
                        user_content,
                        conversation_history=history,
                        persist_user_message=persist_text,
                    )
                finally:
                    if unregister_gateway_notify is not None:
                        try:
                            unregister_gateway_notify(session_id)
                        except Exception as exc:
                            logger.debug("[chat/start] Approval callback cleanup failed: %s", exc)
                    if approval_token is not None and reset_current_session_key is not None:
                        try:
                            reset_current_session_key(approval_token)
                        except Exception as exc:
                            logger.debug("[chat/start] Approval session cleanup failed: %s", exc)

            try:
                future = asyncio.ensure_future(loop.run_in_executor(None, _run))
                result = await self._await_runtime_guarded(
                    future,
                    governor=governor,
                    lease_id=lease_id,
                    agent_ref=agent_ref,
                    cutoff_state=cutoff_state,
                    cutoff_factory=lambda cutoff_result: cutoff_result,
                )
                return result
            except Exception:
                self._close_runtime_lease(governor, lease_id, result=result, error=True)
                raise
            finally:
                if result is not None:
                    self._close_runtime_lease(governor, lease_id, result=result)

        async def _finish_stream():
            try:
                result = await _run_agent_task()
                tools = _tool_map(result.get("messages") or [])
                for item in result.get("messages") or []:
                    if item.get("role") != "tool":
                        continue
                    tool_id = item.get("tool_call_id")
                    tool_meta = tools.get(tool_id, {})
                    _queue_event("tool_complete", {
                        "session_id": session_id,
                        "stream_id": stream_id,
                        "run_id": run_id,
                        "tool_call_id": tool_id,
                        "tool_name": tool_meta.get("tool_name") or item.get("tool_name") or "unknown",
                        "args": tool_meta.get("args"),
                        "result_preview": _result_preview(item.get("content")),
                    })
                _queue_event("assistant.completed", {
                    "session_id": session_id,
                    "stream_id": stream_id,
                    "run_id": run_id,
                    "message_id": assistant_message_id,
                    "content": result.get("final_response") or "",
                    "completed": result.get("completed", False),
                    "partial": result.get("partial", False),
                    "interrupted": result.get("interrupted", False),
                })
                _queue_event("message.complete", {
                    "session_id": session_id,
                    "stream_id": stream_id,
                    "run_id": run_id,
                    "message_id": assistant_message_id,
                    "text": result.get("final_response") or "",
                    "status": "interrupted" if result.get("interrupted") else "complete",
                })
                _queue_event("run.completed", {
                    "session_id": session_id,
                    "stream_id": stream_id,
                    "run_id": run_id,
                    "message_id": assistant_message_id,
                    "completed": result.get("completed", False),
                    "partial": result.get("partial", False),
                    "interrupted": result.get("interrupted", False),
                    "api_calls": result.get("api_calls"),
                })
                _queue_event("done", {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "state": "final"})
            except Exception as exc:
                logger.error("Backend-owned session stream failed for %s: %s", session_id, exc, exc_info=True)
                _on_status("error", "stream failed")
                _queue_event("error", {"session_id": session_id, "stream_id": stream_id, "run_id": run_id, "error": str(exc)})
            finally:
                def _mutate() -> None:
                    stream_state["active"] = False
                    stream_state["active_tools"] = []
                    stream_state["pending_approval"] = None
                    stream_state["completed_at"] = time.time()

                _update_runtime(_mutate)
                stream_q.put(None)

        stream_state["task"] = asyncio.ensure_future(_finish_stream())

        return web.json_response({"stream_id": stream_id, "session_id": session_id, "run_id": run_id})

    async def _handle_chat_stream_status(self, request: "web.Request") -> "web.Response":
        """GET /api/chat/stream/status — report whether a backend-owned chat stream is alive."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        stream_id = request.query.get("stream_id", "")
        if not stream_id:
            return web.json_response({"error": "stream_id required"}, status=400)
        stream_state = self._chat_streams.get(stream_id)
        if not stream_state:
            return web.json_response({"active": False, "stream_id": stream_id})
        return web.json_response({
            "active": bool(stream_state.get("active")),
            "stream_id": stream_id,
            "session_id": stream_state.get("session_id"),
            "run_id": stream_state.get("run_id"),
        })

    async def _handle_chat_stream_snapshot(self, request: "web.Request") -> "web.Response":
        """GET /api/chat/stream/snapshot — return reconnect-safe live chat runtime state."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        stream_id = request.query.get("stream_id", "")
        if not stream_id:
            return web.json_response({"error": "stream_id required"}, status=400)
        if len(stream_id) > 128 or not re.fullmatch(r"[A-Za-z0-9_-]+", stream_id):
            return web.json_response({"error": "invalid stream_id"}, status=400)
        stream_state = self._chat_streams.get(stream_id)
        if not stream_state:
            return web.json_response({"error": "stream not found", "active": False, "stream_id": stream_id}, status=404)

        def _snapshot() -> Dict[str, Any]:
            return {
                "active": bool(stream_state.get("active")),
                "stream_id": stream_state.get("stream_id"),
                "session_id": stream_state.get("session_id"),
                "run_id": stream_state.get("run_id"),
                "message_id": stream_state.get("message_id"),
                "created_at": stream_state.get("created_at"),
                "last_event_at": stream_state.get("last_event_at"),
                "status": dict(stream_state.get("status") or {"kind": "status", "text": "running"}),
                "thinking": str(stream_state.get("thinking") or ""),
                "reasoning": str(stream_state.get("reasoning") or ""),
                "assistant_content": str(stream_state.get("assistant_content") or ""),
                "active_tools": [
                    {
                        "tool_id": str(tool.get("tool_id") or ""),
                        "name": str(tool.get("name") or "tool"),
                        "context": str(tool.get("context") or ""),
                    }
                    for tool in list(stream_state.get("active_tools") or [])
                    if isinstance(tool, dict)
                ],
                "pending_approval": stream_state.get("pending_approval"),
                "events": list(stream_state.get("events") or [])[-50:],
            }

        lock = stream_state.get("lock")
        if lock is not None:
            with lock:
                payload = _snapshot()
        else:
            payload = _snapshot()
        return web.json_response(payload)

    async def _handle_chat_stream_attach(self, request: "web.Request") -> "web.StreamResponse":
        """GET /api/chat/stream — attach to a backend-owned WebUI-style chat stream."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        stream_id = request.query.get("stream_id", "")
        if not stream_id:
            return web.json_response({"error": "stream_id required"}, status=400)
        stream_state = self._chat_streams.get(stream_id)
        if not stream_state:
            return web.json_response({"error": "stream not found"}, status=404)

        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            headers.update(cors)
        response = web.StreamResponse(status=200, headers=headers)
        await response.prepare(request)

        stream_q = stream_state["queue"]
        loop = asyncio.get_event_loop()
        try:
            while True:
                try:
                    frame = await loop.run_in_executor(
                        None,
                        lambda: stream_q.get(timeout=CHAT_STREAM_HEARTBEAT_SECONDS),
                    )
                except _queue.Empty:
                    if not stream_state.get("active"):
                        break
                    await response.write(_encode_dashboard_sse("heartbeat", {
                        "stream_id": stream_id,
                        "session_id": stream_state.get("session_id"),
                        "run_id": stream_state.get("run_id"),
                        "active": True,
                        "elapsed_seconds": int(time.time() - float(stream_state.get("created_at") or time.time())),
                    }))
                    continue
                if frame is None:
                    break
                await response.write(frame)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            logger.info("Chat stream client detached without cancelling stream %s", stream_id)
        return response

    async def _handle_chat_stream_cancel(self, request: "web.Request") -> "web.Response":
        """POST /api/chat/cancel — cancel a backend-owned chat stream."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        stream_id = request.query.get("stream_id", "")
        if not stream_id:
            return web.json_response({"error": "stream_id required"}, status=400)
        stream_state = self._chat_streams.get(stream_id)
        if not stream_state:
            return web.json_response({"ok": True, "cancelled": False, "stream_id": stream_id})
        agent_ref = stream_state.get("agent_ref") or []
        agent = agent_ref[0] if agent_ref else None
        if agent is not None:
            try:
                agent.interrupt("stream cancelled")
            except Exception as exc:
                logger.debug("Failed to interrupt chat stream %s: %s", stream_id, exc)
        stream_state["active"] = False
        return web.json_response({"ok": True, "cancelled": True, "stream_id": stream_id})

    async def _handle_approval_respond(self, request: "web.Request") -> "web.Response":
        """POST /api/approval/respond — resolve one blocking approval request."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            return web.json_response({"error": "session_id is required"}, status=400)
        session_id = session_id.strip()

        choice = body.get("choice", "deny")
        if choice not in ("once", "session", "always", "deny"):
            return web.json_response({"error": f"Invalid choice: {choice}"}, status=400)

        approval_id = body.get("approval_id")
        if approval_id is not None and not isinstance(approval_id, str):
            return web.json_response({"error": "approval_id must be a string"}, status=400)
        approval_id = approval_id.strip() if isinstance(approval_id, str) and approval_id.strip() else None

        try:
            from tools.approval import resolve_gateway_approval
        except ImportError:
            return web.json_response({"error": "Approval system unavailable"}, status=503)

        resolved = resolve_gateway_approval(session_id, choice, resolve_all=False, approval_id=approval_id)
        return web.json_response({"ok": True, "choice": choice, "resolved": resolved})

    async def _handle_get_memory(self, request: "web.Request") -> "web.Response":
        """GET /api/memory — read current memory state."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        target = (request.query.get("target") or "all").strip().lower()
        if target not in {"all", "memory", "user"}:
            return web.json_response({"error": "target must be one of: all, memory, user"}, status=400)

        store = self._get_memory_store()
        store.load_from_disk()
        targets = []
        if target in {"all", "memory"}:
            targets.append({
                "target": "memory",
                "entries": store.memory_entries,
                "entry_count": len(store.memory_entries),
            })
        if target in {"all", "user"}:
            targets.append({
                "target": "user",
                "entries": store.user_entries,
                "entry_count": len(store.user_entries),
            })
        return web.json_response({"targets": targets})

    async def _handle_add_memory(self, request: "web.Request") -> "web.Response":
        """POST /api/memory — add a memory entry."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        target = str(body.get("target") or "").strip().lower()
        content = str(body.get("content") or "")
        if target not in {"memory", "user"}:
            return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
        result = self._get_memory_store().add(target, content)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)

    async def _handle_replace_memory(self, request: "web.Request") -> "web.Response":
        """PATCH /api/memory — replace a memory entry."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        target = str(body.get("target") or "").strip().lower()
        old_text = str(body.get("old_text") or "")
        content = str(body.get("content") or "")
        if target not in {"memory", "user"}:
            return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
        result = self._get_memory_store().replace(target, old_text, content)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)

    async def _handle_delete_memory(self, request: "web.Request") -> "web.Response":
        """DELETE /api/memory — delete a memory entry."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        target = str(body.get("target") or "").strip().lower()
        old_text = str(body.get("old_text") or "")
        if target not in {"memory", "user"}:
            return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
        result = self._get_memory_store().remove(target, old_text)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)

    async def _handle_list_skills(self, request: "web.Request") -> "web.Response":
        """GET /api/skills — list skills."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        category = (request.query.get("category") or "").strip() or None
        return web.json_response(json.loads(skills_list(category=category)))

    async def _handle_skill_categories(self, request: "web.Request") -> "web.Response":
        """GET /api/skills/categories — list skill categories."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        return web.json_response(json.loads(skills_categories()))

    async def _handle_view_skill(self, request: "web.Request") -> "web.Response":
        """GET /api/skills/{name} — fetch skill details."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        file_path = (request.query.get("file_path") or "").strip() or None
        return web.json_response(json.loads(skill_view(name, file_path=file_path)))

    async def _handle_get_config(self, request: "web.Request") -> "web.Response":
        """GET /api/config — fetch the current config."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        config = load_config()
        current = self._current_model_settings(config)
        return web.json_response({
            "model": current["model"],
            "provider": current["provider"],
            "api_mode": current["api_mode"],
            "base_url": current["base_url"],
            "config": config,
        })

    async def _handle_update_config(self, request: "web.Request") -> "web.Response":
        """PATCH /api/config — update model/provider/base_url settings."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        config = load_config()
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            updated_model_cfg = dict(model_cfg)
        elif isinstance(model_cfg, str) and model_cfg.strip():
            updated_model_cfg = {"default": model_cfg.strip()}
        else:
            updated_model_cfg = {}

        if "model" in body:
            updated_model_cfg["default"] = str(body.get("model") or "").strip()
        if "provider" in body:
            updated_model_cfg["provider"] = str(body.get("provider") or "").strip()
        if "base_url" in body:
            updated_model_cfg["base_url"] = str(body.get("base_url") or "").strip()

        config["model"] = updated_model_cfg
        try:
            save_config(config)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        current = self._current_model_settings(config)
        return web.json_response({
            "ok": True,
            "model": current["model"],
            "provider": current["provider"],
            "base_url": current["base_url"],
        })

    async def _handle_available_models(self, request: "web.Request") -> "web.Response":
        """GET /api/available-models — list provider models and available providers."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        config = load_config()
        current = self._current_model_settings(config)
        provider = (request.query.get("provider") or current["provider"] or "openrouter").strip()
        models = [
            {"id": model_id, "description": description}
            for model_id, description in curated_models_for_provider(provider)
        ]
        providers = list_available_providers()
        return web.json_response({"provider": provider, "models": models, "providers": providers})

    # ------------------------------------------------------------------
    # Upstream HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "hermes-agent"})

    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed — rich status for cross-container dashboard probing.

        Returns gateway state, connected platforms, PID, and uptime so the
        dashboard can display full status without needing a shared PID file or
        /proc access.  No authentication required.
        """
        from gateway.status import read_runtime_status

        runtime = read_runtime_status() or {}
        return web.json_response({
            "status": "ok",
            "platform": "hermes-agent",
            "gateway_state": runtime.get("gateway_state"),
            "platforms": runtime.get("platforms", {}),
            "active_agents": runtime.get("active_agents", 0),
            "exit_reason": runtime.get("exit_reason"),
            "updated_at": runtime.get("updated_at"),
            "pid": os.getpid(),
        })

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — return hermes-agent as an available model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": self._model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hermes",
                    "permission": [],
                    "root": self._model_name,
                    "parent": None,
                }
            ],
        })

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """GET /v1/capabilities — advertise the stable API surface.

        External UIs and orchestrators use this endpoint to discover the API
        server's plugin-safe contract without scraping docs or assuming that
        every Hermes version exposes the same endpoints.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return web.json_response({
            "object": "hermes.api_server.capabilities",
            "platform": "hermes-agent",
            "model": self._model_name,
            "auth": {
                "type": "bearer",
                "required": bool(self._api_key),
            },
            "runtime": {
                "mode": "server_agent",
                "tool_execution": "server",
                "split_runtime": False,
                "description": (
                    "The API server creates a server-side Hermes AIAgent; "
                    "tools execute on the API-server host unless a future "
                    "explicit split-runtime mode is enabled."
                ),
            },
            "features": {
                "chat_completions": True,
                "chat_completions_streaming": True,
                "responses_api": True,
                "responses_streaming": True,
                "run_submission": True,
                "run_status": True,
                "run_events_sse": True,
                "run_stop": True,
                "run_approval_response": True,
                "tool_progress_events": True,
                "approval_events": True,
                "session_continuity_header": "X-Hermes-Session-Id",
                "session_key_header": "X-Hermes-Session-Key",
                "cors": bool(self._cors_origins),
            },
            "endpoints": {
                "health": {"method": "GET", "path": "/health"},
                "health_detailed": {"method": "GET", "path": "/health/detailed"},
                "models": {"method": "GET", "path": "/v1/models"},
                "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                "responses": {"method": "POST", "path": "/v1/responses"},
                "runs": {"method": "POST", "path": "/v1/runs"},
                "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
                "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
                "run_approval": {"method": "POST", "path": "/v1/runs/{run_id}/approval"},
                "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
            },
        })

    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        stream = _coerce_request_bool(body.get("stream"), default=False)

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, str]] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            if role == "system":
                # System messages don't support images (Anthropic rejects, OpenAI
                # text-model systems don't render them).  Flatten to text.
                content = _normalize_chat_content(raw_content)
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role in {"user", "assistant"}:
                try:
                    content = _normalize_multimodal_content(raw_content)
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"messages[{idx}].content")
                conversation_messages.append({"role": role, "content": content})

        # Extract the last user message as the primary input
        user_message: Any = ""
        history = []
        if conversation_messages:
            user_message = conversation_messages[-1].get("content", "")
            history = conversation_messages[:-1]

        if not _content_has_visible_payload(user_message):
            return web.json_response(
                {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
                status=400,
            )

        # Allow caller to scope long-term memory (e.g. Honcho) with a
        # stable per-channel identifier via X-Hermes-Session-Key.  This
        # is independent of X-Hermes-Session-Id: the key persists across
        # transcripts while the id rotates when the caller starts a new
        # transcript (i.e. /new semantics).  See _parse_session_key_header.
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        # Allow caller to continue an existing session by passing X-Hermes-Session-Id.
        # When provided, history is loaded from state.db instead of from the request body.
        #
        # Security: session continuation exposes conversation history, so it is
        # only allowed when the API key is configured and the request is
        # authenticated.  Without this gate, any unauthenticated client could
        # read arbitrary session history by guessing/enumerating session IDs.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if provided_session_id:
            if not self._api_key:
                logger.warning(
                    "Session continuation via X-Hermes-Session-Id rejected: "
                    "no API key configured.  Set API_SERVER_KEY to enable "
                    "session continuity."
                )
                return web.json_response(
                    _openai_error(
                        "Session continuation requires API key authentication. "
                        "Configure API_SERVER_KEY to enable this feature."
                    ),
                    status=403,
                )
            # Sanitize: reject control characters that could enable header injection.
            if re.search(r'[\r\n\x00]', provided_session_id):
                return web.json_response(
                    {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                    status=400,
                )
            session_id = provided_session_id
            try:
                db = self._ensure_session_db()
                if db is not None:
                    history = db.get_messages_as_conversation(session_id)
            except Exception as e:
                logger.warning("Failed to load session history for %s: %s", session_id, e)
                history = []
        else:
            # Derive a stable session ID from the conversation fingerprint so
            # that consecutive messages from the same Open WebUI (or similar)
            # conversation map to the same Hermes session.  The first user
            # message + system prompt are constant across all turns.
            first_user = ""
            for cm in conversation_messages:
                if cm.get("role") == "user":
                    first_user = cm.get("content", "")
                    break
            session_id = _derive_chat_session_id(system_prompt, first_user)
            # history already set from request body above

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", self._model_name)
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

            # Track which tool_call_ids we've emitted a "running" lifecycle
            # event for, so a "completed" event without a matching "running"
            # (e.g. internal/filtered tools) is silently dropped instead of
            # producing an orphaned event clients can't correlate.
            _started_tool_call_ids: set[str] = set()

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Emit ``hermes.tool.progress`` with ``status: running``.

                Replaces the old ``tool_progress_callback("tool.started",
                ...)`` emit so SSE consumers receive a single event per
                tool start, carrying both the legacy ``tool``/``emoji``/
                ``label`` payload (for #6972 frontends) and the new
                ``toolCallId``/``status`` correlation fields (#16588).

                Skips tools whose names start with ``_`` so internal
                events (``_thinking``, …) stay off the wire — matching
                the prior ``_on_tool_progress`` filter exactly.
                """
                if not tool_call_id or function_name.startswith("_"):
                    return
                _started_tool_call_ids.add(tool_call_id)
                from agent.display import build_tool_preview, get_tool_emoji
                label = build_tool_preview(function_name, function_args) or function_name
                _stream_q.put(("__tool_progress__", {
                    "tool": function_name,
                    "emoji": get_tool_emoji(function_name),
                    "label": label,
                    "toolCallId": tool_call_id,
                    "status": "running",
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Emit the matching ``status: completed`` event.

                Dropped if the start was filtered (internal tool, missing
                id, or never seen) so clients never get an orphaned
                ``completed`` they can't correlate to a prior ``running``.
                """
                if not tool_call_id or tool_call_id not in _started_tool_call_ids:
                    return
                _started_tool_call_ids.discard(tool_call_id)
                _stream_q.put(("__tool_progress__", {
                    "tool": function_name,
                    "toolCallId": tool_call_id,
                    "status": "completed",
                }))

            # Start agent in background.  agent_ref is a mutable container
            # so the SSE writer can interrupt the agent on client disconnect.
            #
            # ``tool_progress_callback`` is intentionally not wired here:
            # it would duplicate every emit because ``run_agent`` fires it
            # side-by-side with ``tool_start_callback``/``tool_complete_callback``.
            # The structured callbacks are strictly richer (they carry the
            # tool_call id), so they own the chat-completions SSE channel.
            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent_with_runtime(
                surface="chat_completions_stream",
                runtime_session_id=session_id,
                source_message_id=completion_id,
                message_preview=str(user_message),
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
                gateway_session_key=gateway_session_key,
            ))
            # Ensure SSE drain loops can terminate without relying on polling
            # agent_task.done(), which can race with queue timeout checks.
            agent_task.add_done_callback(lambda _fut: _stream_q.put(None))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
                gateway_session_key=gateway_session_key,
            )

        # Non-streaming: run the agent (with optional Idempotency-Key)
        async def _compute_completion():
            return await self._run_agent_with_runtime(
                surface="chat_completions",
                runtime_session_id=session_id,
                source_message_id=completion_id,
                message_preview=str(user_message),
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(body, keys=["model", "messages", "tools", "tool_choice", "stream"])
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_completion)
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_completion()
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        final_response = result.get("final_response") or ""
        is_partial = bool(result.get("partial"))
        is_failed = bool(result.get("failed"))
        completed = bool(result.get("completed", True))
        err_msg = result.get("error")

        # Decide finish_reason. OpenAI uses "length" for truncation, "stop"
        # for normal completion, and downstream SDKs accept "error" / custom
        # codes. See issue #22496.
        if is_partial and err_msg and "truncat" in err_msg.lower():
            finish_reason = "length"
        elif is_failed or (not completed and err_msg):
            finish_reason = "error"
        else:
            finish_reason = "stop"

        response_headers = {
            "X-Hermes-Session-Id": result.get("session_id", session_id),
        }
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key

        # Hard-fail path: no usable assistant text AND a real failure → 5xx
        # with OpenAI-style error envelope so SDK clients raise instead of
        # silently rendering the internal failure string as message.content.
        if not final_response and (is_failed or is_partial):
            err_body = _openai_error(
                err_msg or "Agent run did not produce a response.",
                err_type="server_error",
                code="agent_incomplete",
            )
            err_body["error"]["hermes"] = {
                "completed": completed,
                "partial": is_partial,
                "failed": is_failed,
            }
            response_headers["X-Hermes-Completed"] = "false"
            response_headers["X-Hermes-Partial"] = "true" if is_partial else "false"
            return web.json_response(err_body, status=502, headers=response_headers)

        # Soft-partial path: we have *some* text but the run did not complete
        # (e.g. truncation with partial buffered output). Still 200 but signal
        # truncation via finish_reason="length" + Hermes-specific extras.
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
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        if is_partial or is_failed or not completed:
            response_data["hermes"] = {
                "completed": completed,
                "partial": is_partial,
                "failed": is_failed,
                "error": err_msg,
                "error_code": "output_truncated" if finish_reason == "length" else "agent_error",
            }
            response_headers["X-Hermes-Completed"] = "false"
            response_headers["X-Hermes-Partial"] = "true" if is_partial else "false"
            if err_msg:
                response_headers["X-Hermes-Error"] = err_msg[:200]

        return web.json_response(response_data, headers=response_headers)

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task, agent_ref=None, session_id: str = None,
        gateway_session_key: str = None,
    ) -> "web.StreamResponse":
        """Write real streaming SSE from agent's stream_delta_callback queue.

        If the client disconnects mid-stream (network drop, browser tab close),
        the agent is interrupted via ``agent.interrupt()`` so it stops making
        LLM API calls, and the asyncio task wrapper is cancelled.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        # CORS middleware can't inject headers into StreamResponse after
        # prepare() flushes them, so resolve CORS headers up front.
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if gateway_session_key:
            sse_headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        try:
            last_activity = time.monotonic()

            # Role chunk
            role_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
            last_activity = time.monotonic()

            # Helper — route a queue item to the correct SSE event.
            async def _emit(item):
                """Write a single queue item to the SSE stream.

                Plain strings are sent as normal ``delta.content`` chunks.
                Tagged tuples ``("__tool_progress__", payload)`` are sent
                as a custom ``event: hermes.tool.progress`` SSE event so
                frontends can display them without storing the markers in
                conversation history.  See #6972 for the original event,
                #16588 for the ``toolCallId``/``status`` lifecycle fields.
                """
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "__tool_progress__":
                    event_data = json.dumps(item[1])
                    await response.write(
                        f"event: hermes.tool.progress\ndata: {event_data}\n\n".encode()
                    )
                else:
                    content_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                    }
                    await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
                return time.monotonic()

            # Stream content chunks as they arrive from the agent
            loop = asyncio.get_running_loop()
            while True:
                try:
                    delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain any remaining items
                        while True:
                            try:
                                delta = stream_q.get_nowait()
                                if delta is None:
                                    break
                                last_activity = await _emit(delta)
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if delta is None:  # End of stream sentinel
                    break

                last_activity = await _emit(delta)

            # Get usage from completed agent
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            err_msg = None
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                if isinstance(result, dict) and "error" in result:
                    err_msg = str(result["error"])
            except Exception as exc:
                err_msg = str(exc)
                logger.warning("Agent task %s failed, usage data lost: %s", completion_id, exc)

            if err_msg:
                # Provide a visual cue that the agent crashed natively in the chat
                error_delta = f"\n\n[Agent Error: {err_msg}]"
                await _emit(error_delta)

            # Finish chunk
            finish_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client disconnected mid-stream.  Interrupt the agent so it
            # stops making LLM API calls at the next loop iteration, then
            # cancel the asyncio task wrapper.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", completion_id)
        except Exception as _exc:
            # Agent crashed mid-stream.  Try to emit an error chunk
            # so the client gets a proper response instead of a
            # TransferEncodingError from incomplete chunked encoding.
            import traceback as _tb
            logger.error("Agent crashed mid-stream for %s: %s", completion_id, _tb.format_exc()[:300])
            try:
                error_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                }
                await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                await response.write(b"data: [DONE]\n\n")
            except Exception:
                pass

        return response

    async def _write_sse_responses(
        self,
        request: "web.Request",
        response_id: str,
        model: str,
        created_at: int,
        stream_q,
        agent_task,
        agent_ref,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        instructions: Optional[str],
        conversation: Optional[str],
        store: bool,
        session_id: str,
        gateway_session_key: Optional[str] = None,
    ) -> "web.StreamResponse":
        """Write an SSE stream for POST /v1/responses (OpenAI Responses API).

        Emits spec-compliant event types as the agent runs:

        - ``response.created`` — initial envelope (status=in_progress)
        - ``response.output_text.delta`` / ``response.output_text.done`` —
          streamed assistant text
        - ``response.output_item.added`` / ``response.output_item.done``
          with ``item.type == "function_call"`` — when the agent invokes a
          tool (both events fire; the ``done`` event carries the finalized
          ``arguments`` string)
        - ``response.output_item.added`` with
          ``item.type == "function_call_output"`` — tool result with
          ``{call_id, output, status}``
        - ``response.completed`` — terminal event carrying the full
          response object with all output items + usage (same payload
          shape as the non-streaming path for parity)
        - ``response.failed`` — terminal event on agent error

        If the client disconnects mid-stream, ``agent.interrupt()`` is
        called so the agent stops issuing upstream LLM calls, then the
        asyncio task is cancelled.  When ``store=True`` an initial
        ``in_progress`` snapshot is persisted immediately after
        ``response.created`` and disconnects update it to an
        ``incomplete`` snapshot so GET /v1/responses/{id} and
        ``previous_response_id`` chaining still have something to
        recover from.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if gateway_session_key:
            sse_headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        # State accumulated during the stream
        final_text_parts: List[str] = []
        # Track open function_call items by name so we can emit a matching
        # ``done`` event when the tool completes.  Order preserved.
        pending_tool_calls: List[Dict[str, Any]] = []
        # Output items we've emitted so far (used to build the terminal
        # response.completed payload).  Kept in the order they appeared.
        emitted_items: List[Dict[str, Any]] = []
        # Monotonic counter for output_index (spec requires it).
        output_index = 0
        # Monotonic counter for call_id generation if the agent doesn't
        # provide one (it doesn't, from tool_progress_callback).
        call_counter = 0
        # Canonical Responses SSE events include a monotonically increasing
        # sequence_number. Add it server-side for every emitted event so
        # clients that validate the OpenAI event schema can parse our stream.
        sequence_number = 0
        # Track the assistant message item id + content index for text
        # delta events — the spec ties deltas to a specific item.
        message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        message_output_index: Optional[int] = None
        message_opened = False

        async def _write_event(event_type: str, data: Dict[str, Any]) -> None:
            nonlocal sequence_number
            if "sequence_number" not in data:
                data["sequence_number"] = sequence_number
            sequence_number += 1
            payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            await response.write(payload.encode())

        def _envelope(status: str) -> Dict[str, Any]:
            env: Dict[str, Any] = {
                "id": response_id,
                "object": "response",
                "status": status,
                "created_at": created_at,
                "model": model,
            }
            return env

        final_response_text = ""
        agent_error: Optional[str] = None
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        terminal_snapshot_persisted = False

        def _persist_response_snapshot(
            response_env: Dict[str, Any],
            *,
            conversation_history_snapshot: Optional[List[Dict[str, Any]]] = None,
        ) -> None:
            if not store:
                return
            if conversation_history_snapshot is None:
                conversation_history_snapshot = list(conversation_history)
                conversation_history_snapshot.append({"role": "user", "content": user_message})
            self._response_store.put(response_id, {
                "response": response_env,
                "conversation_history": conversation_history_snapshot,
                "instructions": instructions,
                "session_id": session_id,
            })
            if conversation:
                self._response_store.set_conversation(conversation, response_id)

        def _persist_incomplete_if_needed() -> None:
            """Persist an ``incomplete`` snapshot if no terminal one was written.

            Called from both the client-disconnect (``ConnectionResetError``)
            and server-cancellation (``asyncio.CancelledError``) paths so
            GET /v1/responses/{id} and ``previous_response_id`` chaining keep
            working after abrupt stream termination.
            """
            if not store or terminal_snapshot_persisted:
                return
            incomplete_text = "".join(final_text_parts) or final_response_text
            incomplete_items: List[Dict[str, Any]] = list(emitted_items)
            if incomplete_text:
                incomplete_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": incomplete_text}],
                })
            incomplete_env = _envelope("incomplete")
            incomplete_env["output"] = incomplete_items
            incomplete_env["usage"] = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            incomplete_history = list(conversation_history)
            incomplete_history.append({"role": "user", "content": user_message})
            if incomplete_text:
                incomplete_history.append({"role": "assistant", "content": incomplete_text})
            _persist_response_snapshot(
                incomplete_env,
                conversation_history_snapshot=incomplete_history,
            )

        try:
            # response.created — initial envelope, status=in_progress
            created_env = _envelope("in_progress")
            created_env["output"] = []
            await _write_event("response.created", {
                "type": "response.created",
                "response": created_env,
            })
            _persist_response_snapshot(created_env)
            last_activity = time.monotonic()

            async def _open_message_item() -> None:
                """Emit response.output_item.added for the assistant message
                the first time any text delta arrives."""
                nonlocal message_opened, message_output_index, output_index
                if message_opened:
                    return
                message_opened = True
                message_output_index = output_index
                output_index += 1
                item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": message_output_index,
                    "item": item,
                })

            async def _emit_text_delta(delta_text: str) -> None:
                await _open_message_item()
                final_text_parts.append(delta_text)
                await _write_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": delta_text,
                    "logprobs": [],
                })

            async def _emit_tool_started(payload: Dict[str, Any]) -> str:
                """Emit response.output_item.added for a function_call.

                Returns the call_id so the matching completion event can
                reference it.  Prefer the real ``tool_call_id`` from the
                agent when available; fall back to a generated call id for
                safety in tests or older code paths.
                """
                nonlocal output_index, call_counter
                call_counter += 1
                call_id = payload.get("tool_call_id") or f"call_{response_id[5:]}_{call_counter}"
                args = payload.get("arguments", {})
                if isinstance(args, dict):
                    arguments_str = json.dumps(args)
                else:
                    arguments_str = str(args)
                item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "in_progress",
                    "name": payload.get("name", ""),
                    "call_id": call_id,
                    "arguments": arguments_str,
                }
                idx = output_index
                output_index += 1
                pending_tool_calls.append({
                    "call_id": call_id,
                    "name": payload.get("name", ""),
                    "arguments": arguments_str,
                    "item_id": item["id"],
                    "output_index": idx,
                })
                emitted_items.append({
                    "type": "function_call",
                    "name": payload.get("name", ""),
                    "arguments": arguments_str,
                    "call_id": call_id,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": item,
                })
                return call_id

            async def _emit_tool_completed(payload: Dict[str, Any]) -> None:
                """Emit response.output_item.done (function_call) followed
                by response.output_item.added (function_call_output)."""
                nonlocal output_index
                call_id = payload.get("tool_call_id")
                result = payload.get("result", "")
                pending = None
                if call_id:
                    for i, p in enumerate(pending_tool_calls):
                        if p["call_id"] == call_id:
                            pending = pending_tool_calls.pop(i)
                            break
                if pending is None:
                    # Completion without a matching start — skip to avoid
                    # emitting orphaned done events.
                    return

                # function_call done
                done_item = {
                    "id": pending["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "name": pending["name"],
                    "call_id": pending["call_id"],
                    "arguments": pending["arguments"],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": pending["output_index"],
                    "item": done_item,
                })

                # function_call_output added (result)
                result_str = result if isinstance(result, str) else json.dumps(result)
                output_parts = [{"type": "input_text", "text": result_str}]
                output_item = {
                    "id": f"fco_{uuid.uuid4().hex[:24]}",
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                    "status": "completed",
                }
                idx = output_index
                output_index += 1
                emitted_items.append({
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": output_item,
                })
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": output_item,
                })

            # Main drain loop — thread-safe queue fed by agent callbacks.
            async def _dispatch(it) -> None:
                """Route a queue item to the correct SSE emitter.

                Plain strings are text deltas — they are batched (50ms)
                to reduce Open WebUI re-render storms.  Tagged tuples
                with ``__tool_started__`` / ``__tool_completed__``
                prefixes are tool lifecycle events and flush the buffer
                before emitting.
                """
                nonlocal _batch_timer
                if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
                    tag, payload = it
                    # Flush batched text before tool events
                    if _batch_buf:
                        await _flush_batch()
                    if tag == "__tool_started__":
                        await _emit_tool_started(payload)
                    elif tag == "__tool_completed__":
                        await _emit_tool_completed(payload)
                elif isinstance(it, str):
                    # Batch text deltas — append to buffer, flush on timer
                    _batch_buf.append(it)
                    if _batch_timer is None:
                        _batch_timer = asyncio.create_task(_batch_flush_after(0.05))
                # Other types are silently dropped.

            # ── Batching state ──
            _batch_buf: List[str] = []
            _batch_timer: Optional[asyncio.Task] = None
            _batch_lock = asyncio.Lock()

            async def _batch_flush_after(delay: float) -> None:
                """Wait delay seconds, then flush accumulated text deltas."""
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
                # Clear timer reference BEFORE flush so new deltas
                # can start a fresh timer while we emit
                nonlocal _batch_buf, _batch_timer
                _batch_timer = None
                await _flush_batch()

            async def _flush_batch() -> None:
                """Emit a single SSE delta for all accumulated text."""
                nonlocal _batch_buf
                async with _batch_lock:
                    if _batch_buf:
                        combined = "".join(_batch_buf)
                        _batch_buf = []
                        await _emit_text_delta(combined)

            loop = asyncio.get_running_loop()
            while True:
                try:
                    item = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain remaining
                        while True:
                            try:
                                item = stream_q.get_nowait()
                                if item is None:
                                    break
                                await _dispatch(item)
                                last_activity = time.monotonic()
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if item is None:  # EOS sentinel
                    # Cancel pending timer and flush remaining batched text
                    if _batch_timer and not _batch_timer.done():
                        _batch_timer.cancel()
                        _batch_timer = None
                    if _batch_buf:
                        await _flush_batch()
                    break

                await _dispatch(item)
                last_activity = time.monotonic()

            # Flush any final batched text before processing result
            if _batch_buf:
                await _flush_batch()

            # Pick up agent result + usage from the completed task
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                # If the agent produced a final_response but no text
                # deltas were streamed (e.g. some providers only emit
                # the full response at the end), emit a single fallback
                # delta so Responses clients still receive a live text part.
                agent_final = result.get("final_response", "") if isinstance(result, dict) else ""
                if agent_final and not final_text_parts:
                    await _emit_text_delta(agent_final)
                if agent_final and not final_response_text:
                    final_response_text = agent_final
                if isinstance(result, dict) and result.get("error") and not final_response_text:
                    agent_error = result["error"]
            except Exception as e:  # noqa: BLE001
                logger.error("Error running agent for streaming responses: %s", e, exc_info=True)
                agent_error = str(e)

            # Close the message item if it was opened
            final_response_text = "".join(final_text_parts) or final_response_text
            if message_opened:
                await _write_event("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": final_response_text,
                    "logprobs": [],
                })
                msg_done_item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": final_response_text}
                    ],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": message_output_index,
                    "item": msg_done_item,
                })

            # Always append a final message item in the completed
            # response envelope so clients that only parse the terminal
            # payload still see the assistant text.  This mirrors the
            # shape produced by _extract_output_items in the batch path.
            final_items: List[Dict[str, Any]] = list(emitted_items)

            # Trim large content from tool call arguments to keep the
            # response.completed event under ~100KB.  Clients already
            # received full details via incremental events.
            for _item in final_items:
                if _item.get("type") == "function_call":
                    try:
                        _args = json.loads(_item.get("arguments", "{}")) if isinstance(_item.get("arguments"), str) else _item.get("arguments", {})
                        if isinstance(_args, dict):
                            for _k in ("content", "query", "pattern", "old_string", "new_string"):
                                if isinstance(_args.get(_k), str) and len(_args[_k]) > 500:
                                    _args[_k] = "[" + str(len(_args[_k])) + " chars — truncated for response.completed]"
                            _item["arguments"] = json.dumps(_args)
                    except Exception:
                        pass
                elif _item.get("type") == "function_call_output":
                    _output = _item.get("output", [])
                    if isinstance(_output, list) and _output:
                        _first = _output[0]
                        if isinstance(_first, dict) and _first.get("type") == "input_text":
                            _text = _first.get("text", "")
                            if len(_text) > 1000:
                                _first["text"] = _text[:500] + "...[" + str(len(_text) - 500) + " more chars]"
                                _item["output"] = [_first]

            final_items.append({
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": final_response_text or (agent_error or "")}
                ],
            })

            if agent_error:
                failed_env = _envelope("failed")
                failed_env["output"] = final_items
                failed_env["error"] = {"message": agent_error, "type": "server_error"}
                failed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                _failed_history = list(conversation_history)
                _failed_history.append({"role": "user", "content": user_message})
                if final_response_text or agent_error:
                    _failed_history.append({
                        "role": "assistant",
                        "content": final_response_text or agent_error,
                    })
                _persist_response_snapshot(
                    failed_env,
                    conversation_history_snapshot=_failed_history,
                )
                terminal_snapshot_persisted = True
                await _write_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_env,
                })
            else:
                completed_env = _envelope("completed")
                completed_env["output"] = final_items
                completed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                full_history = self._build_response_conversation_history(
                    conversation_history,
                    user_message,
                    result,
                    final_response_text,
                )
                _persist_response_snapshot(
                    completed_env,
                    conversation_history_snapshot=full_history,
                )
                terminal_snapshot_persisted = True
                await _write_event("response.completed", {
                    "type": "response.completed",
                    "response": completed_env,
                })

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            _persist_incomplete_if_needed()
            # Client disconnected — interrupt the agent so it stops
            # making upstream LLM calls, then cancel the task.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", response_id)
        except asyncio.CancelledError:
            # Server-side cancellation (e.g. shutdown, request timeout) —
            # persist an incomplete snapshot so GET /v1/responses/{id} and
            # previous_response_id chaining still work, then re-raise so the
            # runtime's cancellation semantics are respected.
            _persist_incomplete_if_needed()
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE task cancelled")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
            logger.info("SSE task cancelled; persisted incomplete snapshot for %s", response_id)
            raise
        except Exception as _exc:
            # Agent crashed with an unhandled error (e.g. model API error like
            # BadRequestError, AuthenticationError).  Emit a response.failed
            # event and properly terminate the SSE stream so the client doesn't
            # get a TransferEncodingError from incomplete chunked encoding.
            import traceback as _tb
            _persist_incomplete_if_needed()
            agent_error = _tb.format_exc()
            try:
                failed_env = _envelope("failed")
                failed_env["output"] = list(emitted_items)
                failed_env["error"] = {"message": str(_exc)[:500], "type": "server_error"}
                failed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                await _write_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_env,
                })
            except Exception:
                pass
            logger.error("Agent crashed mid-stream for %s: %s", response_id, str(agent_error)[:300])

        return response

    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses — OpenAI Responses API format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Long-term memory scope header (see chat_completions for details).
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        raw_input = body.get("input")
        if raw_input is None:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = _coerce_request_bool(body.get("store"), default=True)

        # conversation and previous_response_id are mutually exclusive
        if conversation and previous_response_id:
            return web.json_response(_openai_error("Cannot use both 'conversation' and 'previous_response_id'"), status=400)

        # Resolve conversation name to latest response_id
        if conversation:
            previous_response_id = self._response_store.get_conversation(conversation)
            # No error if conversation doesn't exist yet — it's a new conversation

        # Normalize input to message list
        input_messages: List[Dict[str, Any]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for idx, item in enumerate(raw_input):
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    try:
                        content = _normalize_multimodal_content(item.get("content", ""))
                    except ValueError as exc:
                        return _multimodal_validation_error(exc, param=f"input[{idx}].content")
                    input_messages.append({"role": role, "content": content})
        else:
            return web.json_response(_openai_error("'input' must be a string or array"), status=400)

        # Accept explicit conversation_history from the request body.
        # This lets stateless clients supply their own history instead of
        # relying on server-side response chaining via previous_response_id.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, Any]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                try:
                    entry_content = _normalize_multimodal_content(entry["content"])
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"conversation_history[{i}].content")
                conversation_history.append({"role": str(entry["role"]), "content": entry_content})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                return web.json_response(_openai_error(f"Previous response not found: {previous_response_id}"), status=404)
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            # If no instructions provided, carry forward from previous
            if instructions is None:
                instructions = stored.get("instructions")

        # Append new input messages to history (all but the last become history)
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        # Last input message is the user_message
        user_message: Any = input_messages[-1].get("content", "") if input_messages else ""
        if not _content_has_visible_payload(user_message):
            return web.json_response(_openai_error("No user message found in input"), status=400)

        # Truncation support
        if body.get("truncation") == "auto" and len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        # Reuse session from previous_response_id chain so the dashboard
        # groups the entire conversation under one session entry.
        session_id = stored_session_id or str(uuid.uuid4())

        stream = _coerce_request_bool(body.get("stream"), default=False)
        if stream:
            # Streaming branch — emit OpenAI Responses SSE events as the
            # agent runs so frontends can render text deltas and tool
            # calls in real time.  See _write_sse_responses for details.
            import queue as _q
            response_id = f"resp_{uuid.uuid4().hex[:28]}"
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # None from the agent is a CLI box-close signal, not EOS.
                # Forwarding would kill the SSE stream prematurely; the
                # SSE writer detects completion via agent_task.done().
                if delta is not None:
                    _stream_q.put(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                """Queue non-start tool progress events if needed in future.

                The structured Responses stream uses ``tool_start_callback``
                and ``tool_complete_callback`` for exact call-id correlation,
                so progress events are currently ignored here.
                """
                return

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Queue a started tool for live function_call streaming."""
                _stream_q.put(("__tool_started__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Queue a completed tool result for live function_call_output streaming."""
                _stream_q.put(("__tool_completed__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                    "result": function_result,
                }))

            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent_with_runtime(
                surface="responses_stream",
                runtime_session_id=session_id,
                source_message_id=response_id,
                message_preview=str(user_message),
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
                gateway_session_key=gateway_session_key,
            ))
            # Ensure SSE drain loops can terminate without relying on polling
            # agent_task.done(), which can race with queue timeout checks.
            agent_task.add_done_callback(lambda _fut: _stream_q.put(None))

            model_name = body.get("model", self._model_name)
            created_at = int(time.time())

            return await self._write_sse_responses(
                request=request,
                response_id=response_id,
                model=model_name,
                created_at=created_at,
                stream_q=_stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=conversation_history,
                user_message=user_message,
                instructions=instructions,
                conversation=conversation,
                store=store,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
            )

        async def _compute_response():
            return await self._run_agent_with_runtime(
                surface="responses",
                runtime_session_id=session_id,
                source_message_id=f"resp_{uuid.uuid4().hex[:28]}",
                message_preview=str(user_message),
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
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
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_response()
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return web.json_response(
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
        full_history = self._build_response_conversation_history(
            conversation_history,
            user_message,
            result,
            final_response,
        )

        # Build output items from the current turn only.  AIAgent returns a
        # full transcript in result["messages"], while older/mocked paths may
        # return only the current turn suffix.
        output_start_index = self._response_messages_turn_start_index(
            conversation_history,
            user_message,
            result,
        )
        output_items = self._extract_output_items(result, start_index=output_start_index)

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": body.get("model", self._model_name),
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
                "session_id": session_id,
            })
            # Update conversation mapping so the next request with the same
            # conversation name automatically chains to this response
            if conversation:
                self._response_store.set_conversation(conversation, response_id)

        response_headers = {"X-Hermes-Session-Id": session_id}
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key
        return web.json_response(response_data, headers=response_headers)

    # ------------------------------------------------------------------
    # GET / DELETE response endpoints
    # ------------------------------------------------------------------

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Cron jobs API
    # ------------------------------------------------------------------

    _JOB_ID_RE = __import__("re").compile(r"[a-f0-9]{12}")
    # Allowed fields for update — prevents clients injecting arbitrary keys
    _UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled"}
    _MAX_NAME_LENGTH = 200
    _MAX_PROMPT_LENGTH = 5000

    @staticmethod
    def _check_jobs_available() -> Optional["web.Response"]:
        """Return error response if cron module isn't available."""
        if not _CRON_AVAILABLE:
            return web.json_response(
                {"error": "Cron module not available"}, status=501,
            )
        return None

    def _check_job_id(self, request: "web.Request") -> tuple:
        """Validate and extract job_id. Returns (job_id, error_response)."""
        job_id = request.match_info["job_id"]
        if not self._JOB_ID_RE.fullmatch(job_id):
            return job_id, web.json_response(
                {"error": "Invalid job ID format"}, status=400,
            )
        return job_id, None

    async def _handle_list_jobs(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs — list all cron jobs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            include_disabled = request.query.get("include_disabled", "").lower() in {"true", "1"}
            jobs = _cron_list(include_disabled=include_disabled)
            return web.json_response({"jobs": jobs})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_create_job(self, request: "web.Request") -> "web.Response":
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
            repeat = body.get("repeat")

            if not name:
                return web.json_response({"error": "Name is required"}, status=400)
            if len(name) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if not schedule:
                return web.json_response({"error": "Schedule is required"}, status=400)
            if len(prompt) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
                return web.json_response({"error": "Repeat must be a positive integer"}, status=400)

            kwargs = {
                "prompt": prompt,
                "schedule": schedule,
                "name": name,
                "deliver": deliver,
            }
            if skills:
                kwargs["skills"] = skills
            if repeat is not None:
                kwargs["repeat"] = repeat

            job = _cron_create(**kwargs)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_job(self, request: "web.Request") -> "web.Response":
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
            job = _cron_get(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_update_job(self, request: "web.Request") -> "web.Response":
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
                return web.json_response({"error": "No valid fields to update"}, status=400)
            # Validate lengths if present
            if "name" in sanitized and len(sanitized["name"]) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if "prompt" in sanitized and len(sanitized["prompt"]) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            job = _cron_update(job_id, sanitized)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_delete_job(self, request: "web.Request") -> "web.Response":
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
            success = _cron_remove(job_id)
            if not success:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_pause_job(self, request: "web.Request") -> "web.Response":
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
            job = _cron_pause(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_resume_job(self, request: "web.Request") -> "web.Response":
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
            job = _cron_resume(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_run_job(self, request: "web.Request") -> "web.Response":
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
            job = _cron_trigger(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response_conversation_history(
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
        final_response: Any,
    ) -> List[Dict[str, Any]]:
        """Build the stored Responses transcript without duplicating history."""
        prior = list(conversation_history)
        current_user = {"role": "user", "content": user_message}
        agent_messages = result.get("messages") if isinstance(result, dict) else None

        if isinstance(agent_messages, list) and agent_messages:
            turn_start = APIServerAdapter._response_messages_turn_start_index(
                conversation_history,
                user_message,
                result,
            )
            if turn_start:
                return list(agent_messages)

            full_history = prior
            full_history.append(current_user)
            full_history.extend(agent_messages)
            return full_history

        full_history = prior
        full_history.append(current_user)
        full_history.append({"role": "assistant", "content": final_response})
        return full_history

    @staticmethod
    def _response_messages_turn_start_index(
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
    ) -> int:
        """Detect transcript-shaped result["messages"] and return turn start."""
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(agent_messages, list) or not agent_messages:
            return 0

        prior = list(conversation_history)
        current_user = {"role": "user", "content": user_message}
        expected_prefix = prior + [current_user]
        if agent_messages[:len(expected_prefix)] == expected_prefix:
            return len(expected_prefix)
        if prior and agent_messages[:len(prior)] == prior:
            return len(prior)
        return 0

    @staticmethod
    def _extract_output_items(result: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """
        Build the output item array from the agent's messages.

        Walks *result["messages"]* starting at *start_index* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])
        if start_index > 0:
            messages = messages[start_index:]

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
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
        runtime_governor=None,
        runtime_lease_id: Optional[str] = None,
        gateway_session_key: Optional[str] = None,
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.

        If *agent_ref* is a one-element list, the AIAgent instance is stored
        at ``agent_ref[0]`` before ``run_conversation`` begins.  This allows
        callers (e.g. the SSE writer) to call ``agent.interrupt()`` from
        another thread to stop in-progress LLM calls.
        """
        loop = asyncio.get_running_loop()
        effective_agent_ref = agent_ref if agent_ref is not None else [None]
        cutoff_state: Dict[str, Any] = {"cutoff": False, "message": "", "reason": ""}

        def _run():
            agent = self._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
                tool_progress_callback=tool_progress_callback,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                gateway_session_key=gateway_session_key,
            )
            effective_agent_ref[0] = agent
            self._install_runtime_step_callback(
                agent,
                governor=runtime_governor,
                lease_id=runtime_lease_id,
                agent_ref=effective_agent_ref,
                cutoff_state=cutoff_state,
            )
            effective_task_id = session_id or str(uuid.uuid4())
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
                task_id=effective_task_id,
            )
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            # Include the effective session ID in the result so callers
            # (e.g. X-Hermes-Session-Id header) can track compression-
            # triggered session rotations. (#16938)
            _eff_sid = getattr(agent, "session_id", session_id)
            if isinstance(_eff_sid, str) and _eff_sid:
                result["session_id"] = _eff_sid
            return result, usage

        future = asyncio.ensure_future(loop.run_in_executor(None, _run))
        return await self._await_runtime_guarded(
            future,
            governor=runtime_governor,
            lease_id=runtime_lease_id,
            agent_ref=effective_agent_ref,
            cutoff_state=cutoff_state,
        )

    # ------------------------------------------------------------------
    # /v1/runs — structured event streaming
    # ------------------------------------------------------------------

    _MAX_CONCURRENT_RUNS = 10  # Prevent unbounded resource allocation
    _RUN_STREAM_TTL = 300  # seconds before orphaned runs are swept
    _RUN_STATUS_TTL = 3600  # seconds to retain terminal run status for polling

    def _set_run_status(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        """Update pollable run status without exposing private agent objects."""
        now = time.time()
        current = self._run_statuses.get(run_id, {})
        current.update({
            "object": "hermes.run",
            "run_id": run_id,
            "status": status,
            "updated_at": now,
        })
        current.setdefault("created_at", fields.pop("created_at", now))
        current.update(fields)
        self._run_statuses[run_id] = current
        return current

    def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop"):
        """Return a tool_progress_callback that pushes structured events to the run's SSE queue."""
        def _push(event: Dict[str, Any]) -> None:
            self._set_run_status(
                run_id,
                self._run_statuses.get(run_id, {}).get("status", "running"),
                last_event=event.get("event"),
            )
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

    async def _handle_runs(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs — start an agent run, return run_id immediately."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Long-term memory scope header (see chat_completions for details).
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        # Enforce concurrency limit
        if len(self._run_streams) >= self._MAX_CONCURRENT_RUNS:
            return web.json_response(
                _openai_error(f"Too many concurrent runs (max {self._MAX_CONCURRENT_RUNS})", code="rate_limit_exceeded"),
                status=429,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        raw_input = body.get("input")
        if not raw_input:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        user_message = raw_input if isinstance(raw_input, str) else (raw_input[-1].get("content", "") if isinstance(raw_input, list) else "")
        if not user_message:
            return web.json_response(_openai_error("No user message found in input"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")

        # Accept explicit conversation_history from the request body.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, str]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored:
                conversation_history = list(stored.get("conversation_history", []))
                stored_session_id = stored.get("session_id")
                if instructions is None:
                    instructions = stored.get("instructions")

        # When input is a multi-message array, extract all but the last
        # message as conversation history (the last becomes user_message).
        # Only fires when no explicit history was provided.
        if not conversation_history and isinstance(raw_input, list) and len(raw_input) > 1:
            for msg in raw_input[:-1]:
                if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                    content = msg["content"]
                    if isinstance(content, list):
                        # Flatten multi-part content blocks to text
                        content = " ".join(
                            part.get("text", "") for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        )
                    conversation_history.append({"role": msg["role"], "content": str(content)})

        run_id = f"run_{uuid.uuid4().hex}"
        session_id = body.get("session_id") or stored_session_id or run_id
        approval_session_key = gateway_session_key or session_id or run_id
        ephemeral_system_prompt = instructions
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Optional[Dict]]" = asyncio.Queue()
        created_at = time.time()
        self._run_streams[run_id] = q
        self._run_streams_created[run_id] = created_at
        self._run_approval_sessions[run_id] = approval_session_key

        event_cb = self._make_run_event_callback(run_id, loop)

        # Also wire stream_delta_callback so message.delta events flow through.
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

        self._set_run_status(
            run_id,
            "queued",
            created_at=created_at,
            session_id=session_id,
            model=body.get("model", self._model_name),
        )

        async def _run_and_close():
            try:
                self._set_run_status(run_id, "running")
                agent = self._create_agent(
                    ephemeral_system_prompt=ephemeral_system_prompt,
                    session_id=session_id,
                    stream_delta_callback=_text_cb,
                    tool_progress_callback=event_cb,
                    gateway_session_key=gateway_session_key,
                )
                self._active_run_agents[run_id] = agent

                def _approval_notify(approval_data: Dict[str, Any]) -> None:
                    event = dict(approval_data or {})
                    event.update({
                        "event": "approval.request",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "choices": ["once", "session", "always", "deny"],
                    })
                    self._set_run_status(
                        run_id,
                        "waiting_for_approval",
                        last_event="approval.request",
                    )
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, event)
                    except Exception:
                        pass

                def _run_sync():
                    from gateway.session_context import clear_session_vars, set_session_vars
                    from tools.approval import (
                        register_gateway_notify,
                        reset_current_session_key,
                        set_current_session_key,
                        unregister_gateway_notify,
                    )

                    effective_task_id = session_id or run_id
                    approval_token = None
                    session_tokens = []
                    try:
                        # Bind approval/session identity for this API run via
                        # contextvars so concurrent runs do not share process
                        # environment state.
                        approval_token = set_current_session_key(approval_session_key)
                        session_tokens = set_session_vars(
                            platform="api_server",
                            session_key=approval_session_key,
                        )
                        register_gateway_notify(approval_session_key, _approval_notify)
                        r = agent.run_conversation(
                            user_message=user_message,
                            conversation_history=conversation_history,
                            task_id=effective_task_id,
                        )
                    finally:
                        try:
                            unregister_gateway_notify(approval_session_key)
                        finally:
                            if approval_token is not None:
                                try:
                                    reset_current_session_key(approval_token)
                                except Exception:
                                    pass
                            if session_tokens:
                                try:
                                    clear_session_vars(session_tokens)
                                except Exception:
                                    pass
                    u = {
                        "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                        "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                        "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                    }
                    return r, u

                result, usage = await asyncio.get_running_loop().run_in_executor(None, _run_sync)
                # Check for structured failure (non-retryable client errors like
                # 401/400 return failed=True instead of raising, so the except
                # block below never fires — issue #15561).
                if isinstance(result, dict) and result.get("failed"):
                    error_msg = result.get("error") or "agent run failed"
                    q.put_nowait({
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": error_msg,
                    })
                    self._set_run_status(
                        run_id,
                        "failed",
                        error=error_msg,
                        last_event="run.failed",
                    )
                else:
                    final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                    q.put_nowait({
                        "event": "run.completed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "output": final_response,
                        "usage": usage,
                    })
                    self._set_run_status(
                        run_id,
                        "completed",
                        output=final_response,
                        usage=usage,
                        last_event="run.completed",
                    )
            except asyncio.CancelledError:
                self._set_run_status(
                    run_id,
                    "cancelled",
                    last_event="run.cancelled",
                )
                try:
                    q.put_nowait({
                        "event": "run.cancelled",
                        "run_id": run_id,
                        "timestamp": time.time(),
                    })
                except Exception:
                    pass
                raise
            except Exception as exc:
                logger.exception("[api_server] run %s failed", run_id)
                self._set_run_status(
                    run_id,
                    "failed",
                    error=str(exc),
                    last_event="run.failed",
                )
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
                # If the asyncio wrapper is cancelled (for example via
                # /stop), the executor thread can still be blocked waiting
                # on an approval Event.  Unregistering here releases those
                # waits immediately; the in-thread unregister is harmlessly
                # idempotent on normal completion.
                try:
                    from tools.approval import unregister_gateway_notify

                    unregister_gateway_notify(approval_session_key)
                except Exception:
                    pass
                # Sentinel: signal SSE stream to close
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
                self._active_run_agents.pop(run_id, None)
                self._active_run_tasks.pop(run_id, None)
                self._run_approval_sessions.pop(run_id, None)

        task = asyncio.create_task(_run_and_close())
        self._active_run_tasks[run_id] = task
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        response_headers = (
            {"X-Hermes-Session-Key": gateway_session_key} if gateway_session_key else {}
        )
        return web.json_response(
            {"run_id": run_id, "status": "started"},
            status=202,
            headers=response_headers,
        )

    async def _handle_get_run(self, request: "web.Request") -> "web.Response":
        """GET /v1/runs/{run_id} — return pollable run status for external UIs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        status = self._run_statuses.get(run_id)
        if status is None:
            return web.json_response(
                _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                status=404,
            )
        return web.json_response(status)

    async def _handle_run_events(self, request: "web.Request") -> "web.StreamResponse":
        """GET /v1/runs/{run_id}/events — SSE stream of structured agent lifecycle events."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]

        # Allow subscribing slightly before the run is registered (race condition window)
        for _ in range(20):
            if run_id in self._run_streams:
                break
            await asyncio.sleep(0.05)
        else:
            return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

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


    async def _handle_run_approval(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/approval — resolve a pending run approval."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        status = self._run_statuses.get(run_id)
        if status is None:
            return web.json_response(
                _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                status=404,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        raw_choice = str(body.get("choice", "")).strip().lower()
        aliases = {"approve": "once", "approved": "once", "allow": "once"}
        choice = aliases.get(raw_choice, raw_choice)
        allowed = {"once", "session", "always", "deny"}
        if choice not in allowed:
            return web.json_response(
                _openai_error(
                    "Invalid approval choice; expected one of: once, session, always, deny",
                    code="invalid_approval_choice",
                ),
                status=400,
            )

        approval_session_key = self._run_approval_sessions.get(run_id)
        if not approval_session_key:
            return web.json_response(
                _openai_error(
                    f"Run has no active approval session: {run_id}",
                    code="approval_not_active",
                ),
                status=409,
            )

        resolve_all = (
            _coerce_request_bool(body.get("all"), default=False)
            or _coerce_request_bool(body.get("resolve_all"), default=False)
        )
        try:
            from tools.approval import resolve_gateway_approval

            resolved = resolve_gateway_approval(
                approval_session_key,
                choice,
                resolve_all=resolve_all,
            )
        except Exception as exc:
            logger.exception("[api_server] approval resolution failed for run %s", run_id)
            return web.json_response(_openai_error(str(exc)), status=500)

        if resolved <= 0:
            return web.json_response(
                _openai_error(
                    f"Run has no pending approval: {run_id}",
                    code="approval_not_pending",
                ),
                status=409,
            )

        self._set_run_status(run_id, "running", last_event="approval.responded")
        q = self._run_streams.get(run_id)
        if q is not None:
            try:
                q.put_nowait({
                    "event": "approval.responded",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "choice": choice,
                    "resolved": resolved,
                })
            except Exception:
                pass

        return web.json_response({
            "object": "hermes.run.approval_response",
            "run_id": run_id,
            "choice": choice,
            "resolved": resolved,
        })

    async def _handle_stop_run(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/stop — interrupt a running agent."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        agent = self._active_run_agents.get(run_id)
        task = self._active_run_tasks.get(run_id)

        if agent is None and task is None:
            return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

        self._set_run_status(run_id, "stopping", last_event="run.stopping")

        if agent is not None:
            try:
                agent.interrupt("Stop requested via API")
            except Exception:
                pass

        if task is not None and not task.done():
            task.cancel()
            # Bounded wait: run_conversation() executes in the default
            # executor thread which task.cancel() cannot preempt — we rely on
            # agent.interrupt() above to break the loop. Cap the wait so a
            # slow/unresponsive interrupt can't hang this handler.
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "[api_server] stop for run %s timed out after 5s; "
                    "agent may still be finishing the current step",
                    run_id,
                )
            except (asyncio.CancelledError, Exception):
                pass

        return web.json_response({"run_id": run_id, "status": "stopping"})

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
                try:
                    from tools.approval import unregister_gateway_notify

                    approval_session_key = self._run_approval_sessions.get(run_id)
                    if approval_session_key:
                        unregister_gateway_notify(approval_session_key)
                except Exception:
                    pass
                self._run_streams.pop(run_id, None)
                self._run_streams_created.pop(run_id, None)
                self._active_run_agents.pop(run_id, None)
                self._active_run_tasks.pop(run_id, None)
                self._run_approval_sessions.pop(run_id, None)

            stale_statuses = [
                run_id
                for run_id, status in list(self._run_statuses.items())
                if status.get("status") in {"completed", "failed", "cancelled"}
                and now - float(status.get("updated_at", 0) or 0) > self._RUN_STATUS_TTL
            ]
            for run_id in stale_statuses:
                self._run_statuses.pop(run_id, None)

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        try:
            mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware) if mw is not None]
            self._app = web.Application(middlewares=mws, client_max_size=MAX_REQUEST_BYTES)
            self._app["api_server_adapter"] = self
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/health/detailed", self._handle_health_detailed)
            self._app.router.add_get("/v1/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_get("/v1/capabilities", self._handle_capabilities)
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
            # Structured event streaming
            self._app.router.add_post("/v1/runs", self._handle_runs)
            self._app.router.add_get("/v1/runs/{run_id}", self._handle_get_run)
            self._app.router.add_get("/v1/runs/{run_id}/events", self._handle_run_events)
            self._app.router.add_post("/v1/runs/{run_id}/approval", self._handle_run_approval)
            self._app.router.add_post("/v1/runs/{run_id}/stop", self._handle_stop_run)
            # Dashboard REST API — sessions, memory, skills, config
            self._app.router.add_get("/api/sessions", self._handle_list_sessions)
            self._app.router.add_post("/api/sessions", self._handle_create_session)
            self._app.router.add_get("/api/sessions/search", self._handle_search_sessions)
            self._app.router.add_get("/api/sessions/{session_id}", self._handle_get_session)
            self._app.router.add_get("/api/sessions/{session_id}/messages", self._handle_get_session_messages)
            self._app.router.add_patch("/api/sessions/{session_id}", self._handle_update_session)
            self._app.router.add_delete("/api/sessions/{session_id}", self._handle_delete_session)
            self._app.router.add_post("/api/sessions/{session_id}/fork", self._handle_fork_session)
            self._app.router.add_post("/api/sessions/{session_id}/chat", self._handle_session_chat)
            self._app.router.add_post("/api/sessions/{session_id}/chat/stream", self._handle_session_chat_stream)
            self._app.router.add_post("/api/sessions/{session_id}/chat/start", self._handle_session_chat_start)
            self._app.router.add_get("/api/chat/stream", self._handle_chat_stream_attach)
            self._app.router.add_get("/api/chat/stream/snapshot", self._handle_chat_stream_snapshot)
            self._app.router.add_get("/api/chat/stream/status", self._handle_chat_stream_status)
            self._app.router.add_post("/api/chat/cancel", self._handle_chat_stream_cancel)
            self._app.router.add_post("/api/approval/respond", self._handle_approval_respond)
            self._app.router.add_get("/api/memory", self._handle_get_memory)
            self._app.router.add_post("/api/memory", self._handle_add_memory)
            self._app.router.add_patch("/api/memory", self._handle_replace_memory)
            self._app.router.add_delete("/api/memory", self._handle_delete_memory)
            self._app.router.add_get("/api/skills", self._handle_list_skills)
            self._app.router.add_get("/api/skills/categories", self._handle_skill_categories)
            self._app.router.add_get("/api/skills/{name}", self._handle_view_skill)
            self._app.router.add_get("/api/config", self._handle_get_config)
            self._app.router.add_patch("/api/config", self._handle_update_config)
            self._app.router.add_get("/api/available-models", self._handle_available_models)
            # Start background sweep to clean up orphaned (unconsumed) run streams
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            # Refuse to start network-accessible without authentication
            if is_network_accessible(self._host) and not self._api_key:
                logger.error(
                    "[%s] Refusing to start: binding to %s requires API_SERVER_KEY. "
                    "Set API_SERVER_KEY or use the default 127.0.0.1.",
                    self.name, self._host,
                )
                return False

            # Refuse to start network-accessible with a placeholder key.
            # Ported from openclaw/openclaw#64586.
            if is_network_accessible(self._host) and self._api_key:
                try:
                    from hermes_cli.auth import has_usable_secret
                    if not has_usable_secret(self._api_key, min_length=8):
                        logger.error(
                            "[%s] Refusing to start: API_SERVER_KEY is set to a "
                            "placeholder value. Generate a real secret "
                            "(e.g. `openssl rand -hex 32`) and set API_SERVER_KEY "
                            "before exposing the API server on %s.",
                            self.name, self._host,
                        )
                        return False
                except ImportError:
                    pass

            # Port conflict detection — fail fast if port is already in use
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
            if not self._api_key:
                logger.warning(
                    "[%s] ⚠️  No API key configured (API_SERVER_KEY / platforms.api_server.key). "
                    "All requests will be accepted without authentication. "
                    "Set an API key for production deployments to prevent "
                    "unauthorized access to sessions, responses, and cron jobs.",
                    self.name,
                )
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                self.name, self._host, self._port, self._model_name,
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
