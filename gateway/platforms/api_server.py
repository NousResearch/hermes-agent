"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless; opt-in session continuity via X-Hermes-Session-Id header; opt-in long-term memory scoping via X-Hermes-Session-Key header)
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id; X-Hermes-Session-Key supported)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent as an available model
- GET/POST /v1/capabilities        — machine-readable API capabilities for external UIs
- GET  /v1/sessions                — list client-visible Hermes sessions
- GET  /v1/sessions/{session_id}   — read a session summary
- GET  /v1/sessions/{session_id}/messages — read session message history
- GET  /api/sessions               — list client-visible Hermes sessions
- POST /api/sessions               — create an empty Hermes session
- GET/PATCH/DELETE /api/sessions/{session_id} — read/update/delete a session
- GET  /api/sessions/{session_id}/messages — read session message history
- POST /api/sessions/{session_id}/fork — branch a session using SessionDB lineage
- POST /api/sessions/{session_id}/chat[/stream] — chat with a persisted session
- POST /v1/runs                    — start a run, returns run_id immediately (202)
- GET  /v1/runs/{run_id}           — retrieve current run status
- GET  /v1/runs/{run_id}/events    — SSE stream of structured lifecycle events
- POST /v1/runs/{run_id}/approval — resolve a pending run approval
- POST /v1/runs/{run_id}/stop       — interrupt a running agent
- GET  /health                     — health check
- GET  /health/detailed            — rich status for cross-container dashboard probing

Any OpenAI-compatible frontend (Open WebUI, LobeChat, LibreChat,
AnythingLLM, NextChat, ChatBox, etc.) can connect to hermes-agent
through this adapter by pointing at http://localhost:8642/v1 and
authenticating with API_SERVER_KEY.

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import hashlib
import hmac
import inspect
import json
import logging
import os
import socket as _socket
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.dev_control.read_models import (
    build_agent_board_response,
    build_agent_board_rows,
)
from gateway.dev_control.ci_status import fetch_ci_status
from gateway.dev_control.project_scope import resolve_project_id
from gateway.dev_control.chat_project_context import build_chat_project_context_overlay
from gateway.dev_control.routes import (
    DevControlRouteMixin,
    dev_control_capabilities,
    register_dev_control_routes,
)
from gateway.ao_snapshot_cache import AOSnapshotCache
from gateway.platforms.kanban_api_routes import register_kanban_routes
from gateway.read_model_cache import ReadModelCache, read_model_etag, read_model_metric_headers, request_fingerprint
from gateway.subagent_events import SubagentEventStore, events_response
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    is_network_accessible,
)

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = 10_000_000  # 10 MB — accommodates long agent conversations with tool calls
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


def _session_chat_user_message(body: Dict[str, Any], *, param: str = "message") -> tuple[Any, Optional["web.Response"]]:
    """Parse and normalize session chat ``message`` / ``input`` like chat completions."""
    user_message = body.get("message") or body.get("input")
    if not _content_has_visible_payload(user_message):
        return None, web.json_response(
            _openai_error("Missing 'message' field", code="missing_message"),
            status=400,
        )
    try:
        return _normalize_multimodal_content(user_message), None
    except ValueError as exc:
        return None, _multimodal_validation_error(exc, param=param)


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
        self._db_path: Optional[str] = db_path if db_path != ":memory:" else None
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
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
        # response_store.db contains conversation history (tool payloads,
        # prompts, results). Tighten to owner-only after creation so other
        # local users on a shared box can't read it. Run once at __init__
        # rather than after every commit — chmod-on-every-write is wasted
        # syscalls on a hot path.
        self._tighten_file_permissions()

    def _tighten_file_permissions(self) -> None:
        """Force owner-only permissions on the DB and SQLite sidecars."""
        if not self._db_path:
            return
        for candidate in (
            Path(self._db_path),
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
        ):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug(
                    "Failed to restrict response store permissions for %s",
                    candidate,
                    exc_info=True,
                )

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


# Slash commands serviceable via POST /v1/slash on the API server.
API_SERVER_SLASH_COMMANDS = frozenset({
    "help", "commands", "status", "profile", "yolo", "restart",
    "new", "reset", "clear", "stop", "steer", "queue", "background",
    "codex-runtime",
})

# Commands the Oryn Workspace app may handle natively (still listed in catalog).
NATIVE_APP_SLASH_COMMANDS = frozenset({"model"})


def _slash_execution_surface(canonical: str) -> tuple[bool, str]:
    """Return (api_executable, execution_surface) for a canonical command name."""
    if canonical in NATIVE_APP_SLASH_COMMANDS:
        return True, "native_app"
    if canonical in API_SERVER_SLASH_COMMANDS:
        return True, "api_server"
    return False, "unavailable"


def build_slash_completion_payload(text: str) -> dict:
    """Build slash completion items for POST /v1/complete/slash."""
    if not text or not str(text).startswith("/"):
        return {"items": [], "replace_from": 0}

    text = str(text)
    try:
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import to_plain_text

        from agent.skill_commands import get_skill_commands
        from agent.skill_bundles import get_skill_bundles

        completer = SlashCommandCompleter(
            skill_commands_provider=lambda: get_skill_commands(),
            skill_bundles_provider=lambda: get_skill_bundles(),
        )
        doc = Document(text, len(text))
        items = [
            {
                "text": c.text,
                "display": c.display or c.text,
                "meta": to_plain_text(c.display_meta) if c.display_meta else "",
            }
            for c in completer.get_completions(doc, None)
        ][:30]
        replace_from = text.rfind(" ") + 1 if " " in text else 1
        return {"items": items, "replace_from": replace_from}
    except Exception as exc:  # noqa: BLE001
        logger.debug("slash completion failed: %s", exc)
        return {"items": [], "replace_from": 1}


def build_skill_completion_payload(text: str) -> dict:
    """Build inline skill picker items for POST /v1/complete/skills."""
    text = str(text or "")
    marker = "//"
    idx = text.rfind(marker)
    if idx < 0:
        return {"items": [], "replace_from": 0}

    query = text[idx + len(marker):]
    if any(ch.isspace() for ch in query):
        return {"items": [], "replace_from": 0}

    try:
        from agent.skill_commands import get_skill_commands

        query_lower = query.lower()
        items = []
        for cmd, info in sorted(get_skill_commands().items()):
            cmd_name = cmd.lstrip("/")
            if query_lower and not (
                cmd_name.lower().startswith(query_lower)
                or cmd.lower().startswith(f"/{query_lower}")
            ):
                continue
            description = str(info.get("description") or f"Invoke the {info.get('name', cmd_name)} skill")
            items.append(
                {
                    "text": cmd,
                    "display": cmd,
                    "meta": description,
                }
            )
            if len(items) >= 30:
                break

        return {"items": items, "replace_from": idx}
    except Exception as exc:  # noqa: BLE001
        logger.debug("skill completion failed: %s", exc)
        return {"items": [], "replace_from": idx}


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


class APIServerAdapter(DevControlRouteMixin, BasePlatformAdapter):
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
        # Active run agent/task references for stop support
        self._active_run_agents: Dict[str, Any] = {}
        self._active_run_tasks: Dict[str, "asyncio.Task"] = {}
        # Pollable run status for dashboards and external control-plane UIs.
        self._run_statuses: Dict[str, Dict[str, Any]] = {}
        # Active approval session key for each run_id.  The approval core
        # resolves requests by session key, while API clients address the
        # in-flight run by run_id.
        self._run_approval_sessions: Dict[str, str] = {}
        self._session_db: Optional[Any] = None  # Lazy-init SessionDB for session continuity
        self._subagent_event_store: Optional[SubagentEventStore] = None
        self._dev_execution_store: Optional[Any] = None
        self._dev_clarification_store: Optional[Any] = None
        self._dev_plan_artifact_store: Optional[Any] = None
        self._dev_project_goal_store: Optional[Any] = None
        self._dev_verification_store: Optional[Any] = None
        self._dev_signal_store: Optional[Any] = None
        self._dev_product_event_store: Optional[Any] = None
        self._dev_incident_store: Optional[Any] = None
        self._dev_scm_store: Optional[Any] = None
        self._dev_reliability_store: Optional[Any] = None
        self._dev_supervisor_loop_task: Optional["asyncio.Task"] = None
        self._session_model_overrides: Dict[str, Dict[str, Any]] = {}
        self._session_reasoning_overrides: Dict[str, Dict[str, Any]] = {}
        self._read_model_cache = ReadModelCache()
        self._ao_snapshot_cache = AOSnapshotCache()

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
    def _clean_log_value(value: Any, *, max_len: int = 200) -> str:
        """Sanitize request metadata before it reaches security logs."""
        if value is None:
            return ""
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        return text[:max_len]

    def _request_audit_context(self, request: "web.Request") -> Dict[str, str]:
        """Return non-secret source metadata for security/audit warnings."""
        peer_ip = ""
        try:
            peer = request.transport.get_extra_info("peername") if request.transport else None
            if isinstance(peer, (tuple, list)) and peer:
                peer_ip = str(peer[0])
        except Exception:
            peer_ip = ""

        return {
            "remote": self._clean_log_value(getattr(request, "remote", "") or peer_ip),
            "peer_ip": self._clean_log_value(peer_ip),
            "forwarded_for": self._clean_log_value(request.headers.get("X-Forwarded-For", "")),
            "real_ip": self._clean_log_value(request.headers.get("X-Real-IP", "")),
            "method": self._clean_log_value(request.method, max_len=16),
            "path": self._clean_log_value(request.path_qs, max_len=500),
            "user_agent": self._clean_log_value(request.headers.get("User-Agent", ""), max_len=300),
        }

    def _request_audit_log_suffix(self, request: "web.Request") -> str:
        ctx = self._request_audit_context(request)
        fields = [f"{key}={value!r}" for key, value in ctx.items() if value]
        return " ".join(fields) if fields else "source='unknown'"

    def _cron_origin_from_request(self, request: "web.Request") -> Dict[str, str]:
        """Persist safe API source metadata on cron jobs created over HTTP."""
        ctx = self._request_audit_context(request)
        origin = {
            "platform": "api_server",
            "chat_id": "api",
        }
        if ctx.get("remote"):
            origin["source_ip"] = ctx["remote"]
        if ctx.get("peer_ip"):
            origin["peer_ip"] = ctx["peer_ip"]
        if ctx.get("forwarded_for"):
            origin["forwarded_for"] = ctx["forwarded_for"]
        if ctx.get("real_ip"):
            origin["real_ip"] = ctx["real_ip"]
        if ctx.get("user_agent"):
            origin["user_agent"] = ctx["user_agent"]
        return origin

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        connect() refuses to start the API server without API_SERVER_KEY, so
        the no-key branch only exists for tests or unsupported manual wiring.
        """
        if not self._api_key:
            return None

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._api_key):
                return None  # Auth OK

        logger.warning(
            "API server rejected invalid API key: %s",
            self._request_audit_log_suffix(request),
        )
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

    def _ensure_session_db(self):
        """Lazily initialise and return the shared SessionDB instance.

        Sessions are persisted to ``state.db`` so that ``hermes sessions list``
        shows API-server conversations alongside CLI and gateway ones.
        """
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.debug("SessionDB unavailable for API server: %s", e)
        return self._session_db

    def _cached_read_model_response(
        self,
        request: "web.Request",
        payload: Dict[str, Any],
        fingerprint: str,
        *,
        cached: Any = None,
        model_name: str = "unknown",
        status: int = 200,
    ) -> "web.Response":
        etag = read_model_etag(fingerprint)
        if request_fingerprint(request) == fingerprint:
            headers = {"ETag": etag}
            if cached is not None:
                headers.update(read_model_metric_headers(cached, payload_size_bytes=0))
                logger.info(
                    "oryn read-model %s status=304 cache=%s total_ms=%.2f compute_ms=%.2f bytes=0",
                    model_name,
                    cached.cache_status,
                    cached.total_ms,
                    cached.compute_ms,
                )
            return web.Response(status=304, headers=headers)
        response_payload = dict(payload)
        response_payload.setdefault("fingerprint", fingerprint)
        payload_size = len(json.dumps(response_payload, ensure_ascii=False, default=str).encode("utf-8"))
        response = web.json_response(response_payload, status=status)
        response.headers["ETag"] = etag
        if cached is not None:
            response.headers.update(read_model_metric_headers(cached, payload_size_bytes=payload_size))
            logger.info(
                "oryn read-model %s status=%s cache=%s total_ms=%.2f compute_ms=%.2f cache_read_ms=%.2f store_ms=%.2f bytes=%s",
                model_name,
                status,
                cached.cache_status,
                cached.total_ms,
                cached.compute_ms,
                cached.cache_read_ms,
                cached.store_ms,
                payload_size,
            )
        return response

    @staticmethod
    def _session_list_fingerprint(db: Any, *, limit: int, offset: int) -> str:
        try:
            with db._lock:
                row = db._conn.execute(
                    """
                    SELECT
                        COUNT(*) AS session_count,
                        COALESCE(MAX(started_at), 0) AS max_started_at,
                        COALESCE(MAX(ended_at), 0) AS max_ended_at,
                        COALESCE(SUM(message_count), 0) AS total_messages,
                        COALESCE(MAX(title), '') AS max_title,
                        (SELECT COALESCE(MAX(timestamp), 0) FROM messages) AS max_message_at
                    FROM sessions
                    """
                ).fetchone()
            parts = [
                "sessions",
                str(limit),
                str(offset),
                str(row["session_count"]),
                str(row["max_started_at"]),
                str(row["max_ended_at"]),
                str(row["total_messages"]),
                str(row["max_title"]),
                str(row["max_message_at"]),
            ]
            return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha256(f"sessions|{limit}|{offset}|{time.time()}".encode("utf-8")).hexdigest()

    @staticmethod
    def _session_messages_fingerprint(db: Any, session_id: str) -> str:
        try:
            with db._lock:
                session = db._conn.execute(
                    "SELECT message_count, title FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                row = db._conn.execute(
                    """
                    SELECT COUNT(*) AS message_count, COALESCE(MAX(timestamp), 0) AS max_message_at
                    FROM messages
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
            parts = [
                "messages",
                session_id,
                str((session["message_count"] if session else None) or row["message_count"]),
                str(row["max_message_at"]),
                str((session["title"] if session else None) or ""),
            ]
            return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha256(f"messages|{session_id}|{time.time()}".encode("utf-8")).hexdigest()

    @staticmethod
    def _subagent_board_fingerprint(store: Any, params: Any) -> str:
        try:
            row = store._conn.execute("SELECT COALESCE(MAX(event_id), 0) AS max_event_id FROM subagent_events").fetchone()
            max_event_id = row["max_event_id"] if row else 0
        except Exception:
            max_event_id = int(time.time())
        # AO bridge state is live process state; a short time bucket keeps it fresh
        # while still allowing repeated UI reloads to coalesce.
        try:
            bucket_seconds = max(5, int(os.getenv("ORYN_SUBAGENT_BOARD_LIVE_BUCKET_SECONDS", "60")))
        except ValueError:
            bucket_seconds = 60
        ao_bucket = int(time.time() / bucket_seconds)
        query = "&".join(f"{key}={params.get(key)}" for key in sorted(params.keys()))
        return hashlib.sha256(f"subagents|{max_event_id}|{ao_bucket}|{query}".encode("utf-8")).hexdigest()

    @staticmethod
    def _table_max_timestamp(store: Any, table: str, column: str = "updated_at") -> float:
        try:
            row = store._conn.execute(f"SELECT COALESCE(MAX({column}), 0) AS value FROM {table}").fetchone()
            return float(row["value"] if row else 0)
        except Exception:
            return 0.0

    def _project_dashboard_fingerprint(
        self,
        project_id: Optional[str],
        *,
        plan_limit: int = 12,
        derive_plans: bool = False,
    ) -> str:
        parts = ["project-dashboard", project_id or "", str(plan_limit), str(derive_plans)]
        for store, table, column in (
            (self._dev_clarification_store, "dev_clarification_sessions", "updated_at"),
            (self._dev_plan_artifact_store, "dev_plan_artifacts", "updated_at"),
            (self._dev_plan_artifact_store, "dev_plan_artifact_builds", "created_at"),
            (self._dev_execution_store, "dev_execution_plans", "updated_at"),
            (self._dev_execution_store, "dev_execution_plan_tasks", "updated_at"),
            (self._dev_verification_store, "dev_verification_runs", "updated_at"),
            (self._dev_signal_store, "dev_signal_reports", "updated_at"),
            (self._dev_signal_store, "dev_backlog_proposals", "updated_at"),
            (self._dev_incident_store, "dev_incidents", "updated_at"),
            (self._dev_scm_store, "dev_merge_readiness", "created_at"),
            (self._dev_scm_store, "dev_merge_approvals", "created_at"),
            (self._dev_reliability_store, "dev_reliability_outcomes", "updated_at"),
            (self._dev_reliability_store, "dev_reliability_improvements", "measured_at"),
            (self._subagent_event_store, "subagent_events", "event_id"),
        ):
            if store is None:
                parts.append("0")
                continue
            parts.append(str(self._table_max_timestamp(store, table, column=column)))
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _invalidate_ao_read_models(self) -> None:
        try:
            self._ao_snapshot_cache.invalidate()
        except Exception as exc:
            logger.debug("AO snapshot cache invalidation failed: %s", exc)
        try:
            self._read_model_cache.invalidate_prefix("subagents:board")
            self._read_model_cache.invalidate_prefix("project-dashboard")
        except Exception as exc:
            logger.debug("AO read-model cache invalidation failed: %s", exc)

    def _ensure_subagent_event_store(self) -> Optional[SubagentEventStore]:
        """Lazily initialise persistent subagent event history."""
        if self._subagent_event_store is None:
            try:
                self._subagent_event_store = SubagentEventStore()
            except Exception as exc:
                logger.warning("Subagent event store unavailable: %s", exc)
        return self._subagent_event_store

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        reasoning_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        context_usage_callback=None,
        gateway_session_key: Optional[str] = None,
        model_override: Optional[str] = None,
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
        if session_id:
            reasoning_override = self._session_reasoning_overrides.get(session_id)
            if reasoning_override is not None:
                reasoning_config = reasoning_override
        model = self._request_model_override(model_override) or _resolve_gateway_model()
        model, runtime_kwargs = self._apply_session_model_override(session_id, model, runtime_kwargs)

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
            reasoning_callback=reasoning_callback,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            context_usage_callback=context_usage_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
            reasoning_config=reasoning_config,
            gateway_session_key=gateway_session_key,
        )
        return agent

    @staticmethod
    def _cleanup_agent_resources(agent: Any) -> None:
        """Best-effort teardown for one-shot API-server agents."""
        if agent is None:
            return
        try:
            if hasattr(agent, "shutdown_memory_provider"):
                session_messages = getattr(agent, "_session_messages", None)
                if isinstance(session_messages, list):
                    agent.shutdown_memory_provider(session_messages)
                else:
                    agent.shutdown_memory_provider()
        except Exception:
            pass
        try:
            if hasattr(agent, "close"):
                agent.close()
        except Exception:
            pass
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception:
            pass

    def _request_model_override(self, requested_model: Optional[str]) -> Optional[str]:
        """Treat the advertised profile model as an alias for the configured LLM."""
        if not requested_model:
            return None
        model = str(requested_model).strip()
        if not model or model == self._model_name:
            return None
        return model

    def _apply_session_model_override(
        self,
        session_id: Optional[str],
        model: str,
        runtime_kwargs: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Apply API-server session-scoped model routing overrides."""
        if not session_id:
            return model, runtime_kwargs

        override = self._session_model_overrides.get(session_id)
        if not override:
            return model, runtime_kwargs

        next_kwargs = dict(runtime_kwargs)
        model = override.get("model", model)
        for key in ("provider", "api_key", "base_url", "api_mode"):
            if key in override and override[key] is not None:
                next_kwargs[key] = override[key]
        return model, next_kwargs

    @staticmethod
    def _serialize_reasoning_effort(reasoning_config: Optional[Dict[str, Any]]) -> Optional[str]:
        if not reasoning_config:
            return None
        if not reasoning_config.get("enabled", True):
            return "none"
        effort = reasoning_config.get("effort")
        if isinstance(effort, str) and effort.strip():
            return effort.strip().lower()
        return None

    # ------------------------------------------------------------------
    # HTTP Handlers
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
            "restart_requested": bool(runtime.get("restart_requested", False)),
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

    async def _handle_current_model(self, request: "web.Request") -> "web.Response":
        """GET /v1/model/current — return the configured gateway model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from hermes_cli.inventory import load_picker_context

            ctx = load_picker_context()
            return web.json_response({
                "object": "hermes.model.current",
                "provider": ctx.current_provider,
                "model": ctx.current_model,
                "base_url": ctx.current_base_url,
            })
        except Exception as exc:
            logger.warning("Failed to resolve current API-server model: %s", exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_openrouter_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/providers/openrouter/models — browse tool-capable OpenRouter models."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        query = str(request.rel_url.query.get("q", "")).strip()
        limit_raw = request.rel_url.query.get("limit")
        offset_raw = request.rel_url.query.get("offset", "0")
        force_refresh = str(request.rel_url.query.get("force_refresh", "")).lower() in {
            "1",
            "true",
            "yes",
        }

        try:
            limit = int(limit_raw) if limit_raw not in (None, "") else None
        except (TypeError, ValueError):
            limit = None

        try:
            offset = int(offset_raw)
        except (TypeError, ValueError):
            offset = 0

        try:
            from hermes_cli.models import list_openrouter_picker_models

            models = list_openrouter_picker_models(
                query=query,
                limit=limit,
                offset=offset,
                force_refresh=force_refresh,
            )
            return web.json_response(
                {
                    "object": "hermes.provider.models",
                    "provider": "openrouter",
                    "data": models,
                }
            )
        except Exception as exc:
            logger.warning("Failed to list OpenRouter picker models: %s", exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_codex_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/providers/codex/models — browse Codex CLI catalog models."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        query = str(request.rel_url.query.get("q", "")).strip()
        limit_raw = request.rel_url.query.get("limit")
        try:
            limit = int(limit_raw) if limit_raw not in (None, "") else None
        except (TypeError, ValueError):
            limit = None

        access_token = None
        authenticated = False
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
            token = creds.get("api_key") if isinstance(creds, dict) else None
            if isinstance(token, str) and token.strip():
                access_token = token.strip()
                authenticated = True
        except Exception:
            access_token = None

        try:
            from hermes_cli.codex_models import list_codex_picker_models

            models = list_codex_picker_models(
                access_token=access_token,
                query=query,
                limit=limit,
            )
            return web.json_response(
                {
                    "object": "hermes.provider.models",
                    "provider": "codex",
                    "authenticated": authenticated,
                    "data": models,
                }
            )
        except Exception as exc:
            logger.warning("Failed to list Codex picker models: %s", exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_get_session_model(self, request: "web.Request") -> "web.Response":
        """GET /v1/sessions/{session_id}/model — read session model override."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id or len(session_id) > self._MAX_SESSION_HEADER_LEN or re.search(r'[\r\n\x00]', session_id):
            return web.json_response(
                {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                status=400,
            )

        override = self._session_model_overrides.get(session_id, {})
        reasoning_config = self._session_reasoning_overrides.get(session_id)
        return web.json_response(
            {
                "object": "hermes.session.model",
                "session_id": session_id,
                "provider": override.get("provider"),
                "model": override.get("model"),
                "base_url": override.get("base_url"),
                "api_mode": override.get("api_mode"),
                "reasoning_effort": self._serialize_reasoning_effort(reasoning_config),
            }
        )

    async def _handle_set_session_model(self, request: "web.Request") -> "web.Response":
        """POST /v1/sessions/{session_id}/model — switch one API session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id or len(session_id) > self._MAX_SESSION_HEADER_LEN or re.search(r'[\r\n\x00]', session_id):
            return web.json_response(
                {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                status=400,
            )

        try:
            maybe_body = request.json()
            body = await maybe_body if inspect.isawaitable(maybe_body) else maybe_body
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        model_input = str(body.get("model") or "").strip()
        explicit_provider = str(body.get("provider") or "").strip()
        reasoning_effort_raw = body.get("reasoning_effort")
        reasoning_effort = (
            str(reasoning_effort_raw).strip()
            if reasoning_effort_raw is not None
            else None
        )
        if not model_input or not explicit_provider:
            return web.json_response(
                _openai_error("'provider' and 'model' are required"),
                status=400,
            )

        parsed_reasoning = None
        if reasoning_effort is not None:
            from hermes_constants import parse_reasoning_effort

            if not reasoning_effort:
                self._session_reasoning_overrides.pop(session_id, None)
            else:
                parsed_reasoning = parse_reasoning_effort(reasoning_effort)
                if parsed_reasoning is None:
                    return web.json_response(
                        _openai_error(
                            "Invalid reasoning_effort. Valid: none, minimal, low, medium, high, xhigh."
                        ),
                        status=400,
                    )

        try:
            from gateway.run import _resolve_gateway_model
            from hermes_cli.config import get_compatible_custom_providers, load_config
            from hermes_cli.model_switch import switch_model
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime = resolve_runtime_provider(requested=None)
            current_api_key = runtime.get("api_key", "")
            if not callable(current_api_key):
                current_api_key = str(current_api_key or "")

            cfg = load_config()
            result = switch_model(
                raw_input=model_input,
                current_provider=str(runtime.get("provider", "") or ""),
                current_model=_resolve_gateway_model(),
                current_base_url=str(runtime.get("base_url", "") or ""),
                current_api_key=current_api_key,
                is_global=False,
                explicit_provider=explicit_provider,
                user_providers=cfg.get("providers") if isinstance(cfg.get("providers"), dict) else None,
                custom_providers=get_compatible_custom_providers(cfg),
            )
            if not result.success:
                return web.json_response(
                    _openai_error(result.error_message or "model switch failed"),
                    status=400,
                )

            self._session_model_overrides[session_id] = {
                "model": result.new_model,
                "provider": result.target_provider,
                "api_key": result.api_key,
                "base_url": result.base_url,
                "api_mode": result.api_mode,
            }
            if parsed_reasoning is not None:
                self._session_reasoning_overrides[session_id] = parsed_reasoning
            return web.json_response({
                "object": "hermes.session.model",
                "session_id": session_id,
                "model": result.new_model,
                "provider": result.target_provider,
                "base_url": result.base_url,
                "api_mode": result.api_mode,
                "reasoning_effort": self._serialize_reasoning_effort(
                    self._session_reasoning_overrides.get(session_id)
                ),
                "warning": result.warning_message or "",
            })
        except Exception as exc:
            logger.warning("Session model switch failed for %s: %s", session_id, exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    # ── Session goals (/goal Ralph loop) ────────────────────────────────

    def _goal_max_turns_from_config(self) -> int:
        try:
            from hermes_cli.config import load_config

            goals_cfg = (load_config() or {}).get("goals") or {}
            return int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            return 20

    def _goal_manager(self, session_id: str):
        from hermes_cli.goals import GoalManager

        return GoalManager(
            session_id=session_id,
            default_max_turns=self._goal_max_turns_from_config(),
        )

    @staticmethod
    def _serialize_goal_state(state) -> Optional[Dict[str, Any]]:
        if state is None or state.status == "cleared":
            return None
        return {
            "goal": state.goal,
            "status": state.status,
            "turns_used": state.turns_used,
            "max_turns": state.max_turns,
            "subgoals": list(state.subgoals or []),
            "last_verdict": state.last_verdict,
            "last_reason": state.last_reason,
            "paused_reason": state.paused_reason,
            "created_at": state.created_at,
            "last_turn_at": state.last_turn_at,
        }

    def _goal_status_payload(
        self,
        session_id: str,
        decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mgr = self._goal_manager(session_id)
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "goal": self._serialize_goal_state(mgr.state),
        }
        if decision:
            payload.update({
                "message": decision.get("message") or "",
                "should_continue": bool(decision.get("should_continue")),
                "continuation_prompt": decision.get("continuation_prompt"),
                "verdict": decision.get("verdict"),
                "reason": decision.get("reason"),
            })
        return payload

    def _evaluate_session_goal_after_turn(
        self,
        session_id: str,
        final_response: str,
        *,
        user_initiated: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        try:
            mgr = self._goal_manager(session_id)
            if not mgr.is_active():
                return None
            decision = mgr.evaluate_after_turn(
                final_response or "",
                user_initiated=user_initiated,
            )
            return self._goal_status_payload(session_id, decision)
        except Exception as exc:
            logger.debug("goal evaluation failed for %s: %s", session_id, exc)
            return None

    async def _handle_get_session_goal(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return web.json_response(_openai_error("Session ID required"), status=400)

        try:
            payload = self._goal_status_payload(session_id)
            return web.json_response({
                "object": "hermes.session.goal",
                **payload,
            })
        except Exception as exc:
            logger.warning("GET session goal failed for %s: %s", session_id, exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_post_session_goal(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return web.json_response(_openai_error("Session ID required"), status=400)

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        action = str(body.get("action") or "").strip().lower()
        if not action:
            return web.json_response(_openai_error("'action' is required"), status=400)

        try:
            mgr = self._goal_manager(session_id)
            message = ""
            kickoff_message: Optional[str] = None

            if action == "set":
                goal_text = str(body.get("goal") or "").strip()
                if not goal_text:
                    return web.json_response(_openai_error("goal text is required"), status=400)
                if self._find_active_run_id_for_session(session_id):
                    return web.json_response(
                        _openai_error(
                            "Agent is running — use pause/clear mid-run, or stop before setting a new goal."
                        ),
                        status=409,
                    )
                raw_max_turns = body.get("max_turns")
                max_turns = None
                if raw_max_turns is not None:
                    try:
                        max_turns = int(raw_max_turns)
                    except (TypeError, ValueError):
                        return web.json_response(
                            _openai_error("max_turns must be an integer"), status=400
                        )
                try:
                    state = mgr.set(goal_text, max_turns=max_turns)
                except ValueError as exc:
                    return web.json_response(_openai_error(str(exc)), status=400)
                message = f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}"
                kickoff_message = state.goal
            elif action == "pause":
                state = mgr.pause(reason="user-paused")
                if state is None:
                    return web.json_response(_openai_error("No active goal."), status=404)
                message = f"⏸ Goal paused: {state.goal}"
            elif action == "resume":
                state = mgr.resume()
                if state is None:
                    return web.json_response(_openai_error("No goal to resume."), status=404)
                message = f"⊙ Goal resumed ({state.max_turns}-turn budget): {state.goal}"
            elif action == "clear":
                had = mgr.has_goal()
                mgr.clear()
                message = "Goal cleared." if had else "No active goal."
            elif action == "add_subgoal":
                subgoal = str(body.get("text") or body.get("subgoal") or "").strip()
                if not subgoal:
                    return web.json_response(_openai_error("subgoal text is required"), status=400)
                try:
                    added = mgr.add_subgoal(subgoal)
                except (ValueError, RuntimeError) as exc:
                    return web.json_response(_openai_error(str(exc)), status=400)
                idx = len(mgr.state.subgoals) if mgr.state else 0
                message = f"✓ Added subgoal {idx}: {added}"
            elif action == "remove_subgoal":
                try:
                    index = int(body.get("index"))
                except (TypeError, ValueError):
                    return web.json_response(_openai_error("index must be an integer"), status=400)
                try:
                    removed = mgr.remove_subgoal(index)
                except (IndexError, RuntimeError) as exc:
                    return web.json_response(_openai_error(str(exc)), status=400)
                message = f"✓ Removed subgoal {index}: {removed}"
            elif action == "clear_subgoals":
                try:
                    prev = mgr.clear_subgoals()
                except RuntimeError as exc:
                    return web.json_response(_openai_error(str(exc)), status=400)
                message = (
                    f"✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}."
                    if prev
                    else "No subgoals to clear."
                )
            elif action == "set_max_turns":
                try:
                    max_turns = int(body.get("max_turns"))
                except (TypeError, ValueError):
                    return web.json_response(
                        _openai_error("max_turns must be an integer"), status=400
                    )
                try:
                    state = mgr.set_max_turns(max_turns)
                except (ValueError, RuntimeError) as exc:
                    return web.json_response(_openai_error(str(exc)), status=400)
                message = (
                    f"Turn budget updated to {state.max_turns} "
                    f"({state.turns_used} used so far)."
                )
            else:
                return web.json_response(_openai_error(f"Unknown action '{action}'"), status=400)

            payload = self._goal_status_payload(session_id)
            payload["message"] = message
            if kickoff_message:
                payload["kickoff_message"] = kickoff_message
            return web.json_response({
                "object": "hermes.session.goal",
                **payload,
            })
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc)), status=400)
        except Exception as exc:
            logger.warning("POST session goal failed for %s: %s", session_id, exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    def _dispatch_goal_slash(self, session_id: str, cmd_arg: str) -> dict:
        args = (cmd_arg or "").strip()
        lower = args.lower()
        mgr = self._goal_manager(session_id)

        if not args or lower == "status":
            return {"type": "text", "content": mgr.status_line()}

        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            if state is None:
                return {"type": "error", "message": "No active goal."}
            return {"type": "text", "content": f"⏸ Goal paused: {state.goal}"}

        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return {"type": "error", "message": "No goal to resume."}
            return {"type": "text", "content": f"⊙ Goal resumed ({state.max_turns}-turn budget): {state.goal}"}

        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            return {
                "type": "text",
                "content": "Goal cleared." if had else "No active goal.",
            }

        if self._find_active_run_id_for_session(session_id):
            return {
                "type": "error",
                "message": (
                    "Agent is running — use /goal status, pause, or clear mid-run, "
                    "or /stop before setting a new goal."
                ),
            }

        try:
            state = mgr.set(args)
        except ValueError as exc:
            return {"type": "error", "message": str(exc)}

        return {
            "type": "send",
            "message": state.goal,
        }

    def _dispatch_subgoal_slash(self, session_id: str, cmd_arg: str) -> dict:
        args = (cmd_arg or "").strip()
        mgr = self._goal_manager(session_id)
        if not mgr.has_goal():
            return {"type": "error", "message": "No active goal. Set one with /goal <text>."}

        if not args:
            return {
                "type": "text",
                "content": f"{mgr.status_line()}\n{mgr.render_subgoals()}",
            }

        tokens = args.split(None, 1)
        verb = tokens[0].lower()
        rest = tokens[1].strip() if len(tokens) > 1 else ""

        if verb == "remove":
            if not rest:
                return {"type": "error", "message": "usage: /subgoal remove <n>"}
            try:
                idx = int(rest.split()[0])
            except ValueError:
                return {"type": "error", "message": "/subgoal remove: <n> must be an integer."}
            try:
                removed = mgr.remove_subgoal(idx)
            except (IndexError, RuntimeError) as exc:
                return {"type": "error", "message": f"/subgoal remove: {exc}"}
            return {"type": "text", "content": f"✓ Removed subgoal {idx}: {removed}"}

        if verb == "clear":
            try:
                prev = mgr.clear_subgoals()
            except RuntimeError as exc:
                return {"type": "error", "message": f"/subgoal clear: {exc}"}
            if prev:
                return {
                    "type": "text",
                    "content": f"✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}.",
                }
            return {"type": "text", "content": "No subgoals to clear."}

        try:
            text = mgr.add_subgoal(args)
        except (ValueError, RuntimeError) as exc:
            return {"type": "error", "message": f"/subgoal: {exc}"}
        idx = len(mgr.state.subgoals) if mgr.state else 0
        return {"type": "text", "content": f"✓ Added subgoal {idx}: {text}"}

    @staticmethod
    def _serialize_message_content(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)

    async def _handle_v1_list_sessions(self, request: "web.Request") -> "web.Response":
        """GET /v1/sessions — paginated session list for remote clients."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            limit = int(request.rel_url.query.get("limit", "50"))
            offset = int(request.rel_url.query.get("offset", "0"))
        except (TypeError, ValueError):
            return web.json_response(_openai_error("Invalid limit or offset"), status=400)

        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        try:
            db = self._ensure_session_db()
            if db is None:
                return web.json_response(_openai_error("Session database unavailable"), status=503)
            fingerprint = self._session_list_fingerprint(db, limit=limit, offset=offset)

            def _compute_payload() -> Dict[str, Any]:
                sessions = db.list_sessions_rich(
                    limit=limit,
                    offset=offset,
                    order_by_last_active=True,
                )
                total = db.session_count()
                payload = []
                for session in sessions:
                    payload.append({
                        "id": session.get("id"),
                        "title": session.get("title"),
                        "model": session.get("model"),
                        "preview": session.get("preview"),
                        "started_at": session.get("started_at"),
                        "last_active": session.get("last_active"),
                        "message_count": session.get("message_count"),
                        "source": session.get("source"),
                    })
                return {
                    "object": "list",
                    "data": payload,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }

            cached = await self._read_model_cache.get_or_compute(
                key=f"sessions:list:{limit}:{offset}",
                fingerprint=fingerprint,
                compute=_compute_payload,
            )
            return self._cached_read_model_response(
                request,
                cached.payload,
                cached.fingerprint,
                cached=cached,
                model_name="sessions.list",
            )
        except Exception as exc:
            logger.warning("GET /v1/sessions failed: %s", exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_v1_get_session(self, request: "web.Request") -> "web.Response":
        """GET /v1/sessions/{session_id} — session metadata."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return web.json_response(_openai_error("Session ID required"), status=400)

        try:
            from hermes_state import SessionDB

            db = SessionDB()
            try:
                resolved = db.resolve_session_id(session_id)
                if not resolved:
                    return web.json_response(_openai_error("Session not found"), status=404)
                sessions = db.list_sessions_rich(limit=10000, offset=0, include_children=True)
                match = next((s for s in sessions if s.get("id") == resolved), None)
                if not match:
                    return web.json_response(_openai_error("Session not found"), status=404)
                return web.json_response({
                    "object": "hermes.session",
                    "session_id": resolved,
                    "title": match.get("title"),
                    "model": match.get("model"),
                    "preview": match.get("preview"),
                    "started_at": match.get("started_at"),
                    "last_active": match.get("last_active"),
                    "message_count": match.get("message_count"),
                    "source": match.get("source"),
                })
            finally:
                db.close()
        except Exception as exc:
            logger.warning("GET /v1/sessions/{id} failed: %s", exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_v1_get_session_messages(self, request: "web.Request") -> "web.Response":
        """GET /v1/sessions/{session_id}/messages — message history for resume/display."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return web.json_response(_openai_error("Session ID required"), status=400)

        try:
            db = self._ensure_session_db()
            if db is None:
                return web.json_response(_openai_error("Session database unavailable"), status=503)
            resolved = db.resolve_session_id(session_id)
            if not resolved:
                return web.json_response(_openai_error("Session not found"), status=404)
            fingerprint = self._session_messages_fingerprint(db, resolved)

            def _compute_payload() -> Dict[str, Any]:
                raw_messages = db.get_messages(resolved)
                messages = []
                for msg in raw_messages:
                    role = msg.get("role")
                    if role not in ("user", "assistant", "tool"):
                        continue
                    item = {
                        "role": role,
                        "content": self._serialize_message_content(msg.get("content")),
                        "timestamp": msg.get("timestamp"),
                    }
                    if role == "assistant":
                        reasoning = msg.get("reasoning")
                        if reasoning:
                            item["reasoning"] = reasoning
                        reasoning_content = msg.get("reasoning_content")
                        if reasoning_content:
                            item["reasoning_content"] = reasoning_content
                        tool_calls = msg.get("tool_calls")
                        if tool_calls:
                            item["tool_calls"] = tool_calls
                    elif role == "tool":
                        tool_call_id = msg.get("tool_call_id")
                        if tool_call_id:
                            item["tool_call_id"] = tool_call_id
                        tool_name = msg.get("tool_name")
                        if tool_name:
                            item["tool_name"] = tool_name
                    messages.append(item)
                return {
                    "object": "hermes.session.messages",
                    "session_id": resolved,
                    "messages": messages,
                }

            cached = await self._read_model_cache.get_or_compute(
                key=f"sessions:messages:{resolved}",
                fingerprint=fingerprint,
                compute=_compute_payload,
            )
            return self._cached_read_model_response(
                request,
                cached.payload,
                cached.fingerprint,
                cached=cached,
                model_name="sessions.messages",
            )
        except Exception as exc:
            logger.warning("GET /v1/sessions/{id}/messages failed: %s", exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_v1_delete_session(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/sessions/{session_id} — remove a stored session and its messages."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return web.json_response(_openai_error("Session ID required"), status=400)

        try:
            from hermes_constants import get_hermes_home
            from hermes_state import SessionDB

            db = SessionDB()
            try:
                resolved = db.resolve_session_id(session_id)
                if not resolved:
                    return web.json_response(_openai_error("Session not found"), status=404)
                sessions_dir = get_hermes_home() / "sessions"
                if not db.delete_session(resolved, sessions_dir=sessions_dir):
                    return web.json_response(_openai_error("Session not found"), status=404)
                return web.json_response({
                    "object": "hermes.session.deleted",
                    "session_id": resolved,
                    "deleted": True,
                })
            finally:
                db.close()
        except Exception as exc:
            logger.warning("DELETE /v1/sessions/{id} failed: %s", exc)
            return web.json_response(_openai_error(str(exc)), status=500)

    async def _handle_commands(self, request: "web.Request") -> "web.Response":
        """GET /v1/commands — Return a dynamic list of active API-callable slash commands."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        from hermes_cli.commands import COMMAND_REGISTRY, resolve_command
        
        # Emitted command collection mapping
        data = []
        emoji_map = {
            "help": "✨", "commands": "✨",
            "status": "ℹ️",
            "profile": "👤",
            "yolo": "🚀",
            "clear": "🗑️", "new": "🗑️", "reset": "🗑️",
            "usage": "📊",
            "agents": "🤖", "tasks": "🤖",
            "undo": "⏮️",
            "fast": "⚡️",
            "compress": "🗜️",
            "reload-mcp": "🔌", "reload_mcp": "🔌",
            "reload-skills": "🎓", "reload_skills": "🎓",
            "voice": "🎙️",
            "restart": "🔄",
            "stop": "🛑",
            "undo": "⏮️"
        }

        for cmd in COMMAND_REGISTRY:
            # Filter out commands that are exclusive to the CLI loop
            if cmd.cli_only:
                continue

            emoji = emoji_map.get(cmd.name, "⚙️")
            api_executable, execution_surface = _slash_execution_surface(cmd.name)
            data.append({
                "name": f"/{cmd.name}",
                "description": cmd.description,
                "emoji": emoji,
                "argsHint": cmd.args_hint,
                "category": cmd.category,
                "apiExecutable": api_executable,
                "executionSurface": execution_surface,
            })

        return web.json_response({
            "object": "list",
            "data": data
        })

    async def _handle_complete_slash(self, request: "web.Request") -> "web.Response":
        """POST /v1/complete/slash — live slash-command completion."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        text = str(body.get("text") or "")
        payload = build_slash_completion_payload(text)
        return web.json_response({
            "object": "hermes.slash.completion",
            **payload,
        })

    async def _handle_complete_skills(self, request: "web.Request") -> "web.Response":
        """POST /v1/complete/skills — inline skill picker completion for // triggers."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        text = str(body.get("text") or "")
        payload = build_skill_completion_payload(text)
        return web.json_response({
            "object": "hermes.skill.completion",
            **payload,
        })

    async def _handle_slash(self, request: "web.Request") -> "web.Response":
        """POST /v1/slash — structured slash-command dispatch."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        command = str(body.get("command") or "").strip()
        if not command:
            return web.json_response(_openai_error("'command' is required"), status=400)

        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if not session_id or len(session_id) > self._MAX_SESSION_HEADER_LEN or re.search(r'[\r\n\x00]', session_id):
            return web.json_response(
                _openai_error("Valid session_id is required"),
                status=400,
            )

        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err:
            return key_err

        result = self._dispatch_slash_command(
            command, session_id, gateway_session_key=gateway_session_key,
        )
        return web.json_response({
            "object": "hermes.slash.result",
            **result,
        })


    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """GET/POST /v1/capabilities — advertise the stable API surface.

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
                "context_usage_events": True,
                "approval_events": True,
                "session_resources": True,
                "session_chat": True,
                "session_chat_streaming": True,
                "session_fork": True,
                "ao_session_controls": True,
                "session_model_selection": True,
                "admin_config_rw": False,
                "jobs_admin": False,
                "memory_write_api": False,
                "skills_api": True,
                "audio_api": False,
                "realtime_voice": False,
                "session_continuity_header": "X-Hermes-Session-Id",
                "session_key_header": "X-Hermes-Session-Key",
                "slash_commands": True,
                "slash_completion": True,
                "skill_completion": True,
                "slash_dispatch": True,
                "slash_command_catalog": True,
                "slash_chat_interception": True,
                "kanban": True,
                "slash_execution": "full",
                "cors": bool(self._cors_origins),
            },
            "endpoints": {
                "health": {"method": "GET", "path": "/health"},
                "health_detailed": {"method": "GET", "path": "/health/detailed"},
                "models": {"method": "GET", "path": "/v1/models"},
                "current_model": {"method": "GET", "path": "/v1/model/current"},
                "openrouter_models": {
                    "method": "GET",
                    "path": "/v1/providers/openrouter/models",
                },
                "session_model": {"method": "POST", "path": "/v1/sessions/{session_id}/model"},
                "commands": {"method": "GET", "path": "/v1/commands"},
                "skills": {"method": "GET", "path": "/v1/skills"},
                "complete_slash": {"method": "POST", "path": "/v1/complete/slash"},
                "complete_skills": {"method": "POST", "path": "/v1/complete/skills"},
                "slash": {"method": "POST", "path": "/v1/slash"},
                "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                "responses": {"method": "POST", "path": "/v1/responses"},
                "runs": {"method": "POST", "path": "/v1/runs"},
                "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
                "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
                "run_subagent_events": {"method": "GET", "path": "/v1/runs/{run_id}/subagents/events"},
                "run_approval": {"method": "POST", "path": "/v1/runs/{run_id}/approval"},
                "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
                "run_steer": {"method": "POST", "path": "/v1/runs/{run_id}/steer"},
                "background_task_follow_up": {
                    "method": "POST",
                    "path": "/v1/background/tasks/{task_id}/follow-up",
                },
                "toolsets": {"method": "GET", "path": "/v1/toolsets"},
                "sessions": {"method": "GET", "path": "/api/sessions"},
                "session_create": {"method": "POST", "path": "/api/sessions"},
                "session": {"method": "GET", "path": "/api/sessions/{session_id}"},
                "session_update": {"method": "PATCH", "path": "/api/sessions/{session_id}"},
                "session_delete": {"method": "DELETE", "path": "/api/sessions/{session_id}"},
                "session_messages": {"method": "GET", "path": "/api/sessions/{session_id}/messages"},
                "session_fork": {"method": "POST", "path": "/api/sessions/{session_id}/fork"},
                "session_chat": {"method": "POST", "path": "/api/sessions/{session_id}/chat"},
                "session_chat_stream": {"method": "POST", "path": "/api/sessions/{session_id}/chat/stream"},
                "ao_sessions": {"method": "GET", "path": "/v1/ao/sessions"},
                "ao_session_detail": {"method": "GET", "path": "/v1/ao/sessions/{session_id}"},
                "ao_session_stop": {"method": "POST", "path": "/v1/ao/sessions/{session_id}/stop"},
                "ao_session_open": {"method": "POST", "path": "/v1/ao/sessions/{session_id}/open"},
                "ao_session_follow_up": {"method": "POST", "path": "/v1/ao/sessions/{session_id}/follow-up"},
                "ao_session_retry": {"method": "POST", "path": "/v1/ao/sessions/{session_id}/retry"},
                "ao_session_reassign": {"method": "POST", "path": "/v1/ao/sessions/{session_id}/reassign"},
                "session_subagent_events": {"method": "GET", "path": "/v1/sessions/{session_id}/subagents/events"},
                "run_subagent_events": {"method": "GET", "path": "/v1/runs/{run_id}/subagents/events"},
                "subagent_board": {"method": "GET", "path": "/v1/subagents/board"},
                "subagent_events": {"method": "GET", "path": "/v1/subagents/events"},
                **dev_control_capabilities(),
                "kanban_board": {"method": "GET", "path": "/v1/kanban/board"},
                "kanban_events": {"method": "GET", "path": "/v1/kanban/events"},
            },
        })

    async def _handle_skills(self, request: "web.Request") -> "web.Response":
        """GET /v1/skills — list installed skills visible to the API-server agent.

        Read-only listing intended for external clients that need to know
        which skills are available without sending a chat message and asking
        the model. Mirrors what the gateway/CLI surfaces through
        ``/skills list``, but as a deterministic JSON payload.

        Returns the same skill metadata (name, description, category) the
        skills hub uses internally. Disabled skills are excluded so the
        listing matches what the agent actually loads.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from tools.skills_tool import _find_all_skills, _sort_skills
            skills = _sort_skills(_find_all_skills(skip_disabled=False))
        except Exception:
            logger.exception("GET /v1/skills failed")
            return web.json_response(
                _openai_error("Failed to enumerate skills", err_type="server_error"),
                status=500,
            )

        return web.json_response({
            "object": "list",
            "data": skills,
        })

    async def _handle_toolsets(self, request: "web.Request") -> "web.Response":
        """GET /v1/toolsets — list toolsets and their resolved tools.

        Returns the toolset surface the api_server platform actually exposes
        to its agent: each toolset's enabled/configured state plus the
        concrete tool names it expands to. This is the deterministic
        equivalent of what a client would otherwise have to recover by
        asking the model what tools it can call.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from hermes_cli.config import load_config
            from hermes_cli.tools_config import (
                _get_effective_configurable_toolsets,
                _get_platform_tools,
                _toolset_has_keys,
            )
            from toolsets import resolve_toolset

            config = load_config()
            enabled_toolsets = _get_platform_tools(
                config,
                "api_server",
                include_default_mcp_servers=False,
            )
            data: List[Dict[str, Any]] = []
            for name, label, desc in _get_effective_configurable_toolsets():
                try:
                    tools = sorted(set(resolve_toolset(name)))
                except Exception:
                    tools = []
                is_enabled = name in enabled_toolsets
                data.append({
                    "name": name,
                    "label": label,
                    "description": desc,
                    "enabled": is_enabled,
                    "configured": _toolset_has_keys(name, config),
                    "tools": tools,
                })
        except Exception:
            logger.exception("GET /v1/toolsets failed")
            return web.json_response(
                _openai_error("Failed to enumerate toolsets", err_type="server_error"),
                status=500,
            )

        return web.json_response({
            "object": "list",
            "platform": "api_server",
            "data": data,
        })

    # ------------------------------------------------------------------
    # /api/sessions — thin client/session resource API
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_nonnegative_int(value: Any, default: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if parsed < 0:
            return default
        return min(parsed, maximum)

    @staticmethod
    def _session_response(session: Dict[str, Any]) -> Dict[str, Any]:
        """Return a stable, client-safe session representation."""
        safe_keys = (
            "id", "source", "user_id", "model", "title", "started_at", "ended_at",
            "end_reason", "message_count", "tool_call_count", "input_tokens",
            "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd",
            "api_call_count", "parent_session_id", "last_active", "preview",
            "_lineage_root_id",
        )
        payload = {key: session.get(key) for key in safe_keys if key in session}
        # Avoid exposing full system prompts/model_config through the client API;
        # callers only need to know whether those snapshots exist.
        payload["has_system_prompt"] = bool(session.get("system_prompt"))
        payload["has_model_config"] = bool(session.get("model_config"))
        return payload

    @staticmethod
    def _message_response(message: Dict[str, Any]) -> Dict[str, Any]:
        safe_keys = (
            "id", "session_id", "role", "content", "tool_call_id", "tool_calls",
            "tool_name", "timestamp", "token_count", "finish_reason", "reasoning",
            "reasoning_content",
        )
        return {key: message.get(key) for key in safe_keys if key in message}

    async def _read_json_body(self, request: "web.Request") -> tuple[Dict[str, Any], Optional["web.Response"]]:
        try:
            body = await request.json()
        except Exception:
            return {}, web.json_response(_openai_error("Invalid JSON in request body"), status=400)
        if not isinstance(body, dict):
            return {}, web.json_response(_openai_error("Request body must be a JSON object"), status=400)
        return body, None

    def _get_existing_session_or_404(self, session_id: str) -> tuple[Optional[Dict[str, Any]], Optional["web.Response"]]:
        db = self._ensure_session_db()
        if db is None:
            return None, web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)
        session = db.get_session(session_id)
        if not session:
            return None, web.json_response(_openai_error(f"Session not found: {session_id}", code="session_not_found"), status=404)
        return session, None

    def _conversation_history_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        db = self._ensure_session_db()
        if db is None:
            return []
        try:
            return db.get_messages_as_conversation(session_id)
        except Exception as exc:
            logger.warning("Failed to load session history for %s: %s", session_id, exc)
            return []

    async def _handle_list_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions — list persisted Hermes sessions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        db = self._ensure_session_db()
        if db is None:
            return web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)

        limit = self._parse_nonnegative_int(request.query.get("limit"), default=50, maximum=200)
        offset = self._parse_nonnegative_int(request.query.get("offset"), default=0, maximum=1_000_000)
        source = request.query.get("source") or None
        include_children = _coerce_request_bool(request.query.get("include_children"), default=False)
        sessions = db.list_sessions_rich(
            source=source,
            limit=limit,
            offset=offset,
            include_children=include_children,
            order_by_last_active=True,
        )
        return web.json_response({
            "object": "list",
            "data": [self._session_response(s) for s in sessions],
            "limit": limit,
            "offset": offset,
            "has_more": len(sessions) == limit,
        })

    async def _handle_create_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions — create an empty Hermes session row."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        body, err = await self._read_json_body(request)
        if err:
            return err

        db = self._ensure_session_db()
        if db is None:
            return web.json_response(_openai_error("Session database unavailable", code="session_db_unavailable"), status=503)

        raw_id = body.get("id") or body.get("session_id")
        session_id = str(raw_id).strip() if raw_id else f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if not session_id or re.search(r'[\r\n\x00]', session_id):
            return web.json_response(_openai_error("Invalid session ID", code="invalid_session_id"), status=400)
        if len(session_id) > self._MAX_SESSION_HEADER_LEN:
            return web.json_response(_openai_error("Session ID too long", code="invalid_session_id"), status=400)
        if db.get_session(session_id):
            return web.json_response(_openai_error(f"Session already exists: {session_id}", code="session_exists"), status=409)

        model = body.get("model") or self._model_name
        system_prompt = body.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_prompt must be a string", code="invalid_system_prompt"), status=400)
        db.create_session(session_id, "api_server", model=str(model) if model else None, system_prompt=system_prompt)
        title = body.get("title")
        if title is not None:
            try:
                db.set_session_title(session_id, str(title))
            except ValueError as exc:
                db.delete_session(session_id)
                return web.json_response(_openai_error(str(exc), code="invalid_title"), status=400)
        session = db.get_session(session_id) or {"id": session_id, "source": "api_server", "model": model, "title": title}
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)}, status=201)

    async def _handle_get_session(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session, err = self._get_existing_session_or_404(request.match_info["session_id"])
        if err:
            return err
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)})

    async def _handle_patch_session(self, request: "web.Request") -> "web.Response":
        """PATCH /api/sessions/{session_id} — update client-safe session metadata."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        session, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        allowed = {"title", "end_reason"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            return web.json_response(_openai_error(f"Unsupported session fields: {', '.join(unknown)}", code="unsupported_session_field"), status=400)

        db = self._ensure_session_db()
        if "title" in body:
            try:
                db.set_session_title(session_id, "" if body["title"] is None else str(body["title"]))
            except ValueError as exc:
                return web.json_response(_openai_error(str(exc), code="invalid_title"), status=400)
        if body.get("end_reason"):
            db.end_session(session_id, str(body["end_reason"]))
        session = db.get_session(session_id) or session
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)})

    async def _handle_delete_session(self, request: "web.Request") -> "web.Response":
        """DELETE /api/sessions/{session_id}."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        session, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = self._ensure_session_db()
        deleted = db.delete_session(session_id)
        return web.json_response({"object": "hermes.session.deleted", "id": session_id, "deleted": bool(deleted)})

    async def _handle_session_messages(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}/messages."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        _, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = self._ensure_session_db()
        messages = db.get_messages(session_id)
        return web.json_response({
            "object": "list",
            "session_id": session_id,
            "data": [self._message_response(m) for m in messages],
        })

    async def _handle_fork_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/fork — branch via current SessionDB primitives."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        source_id = request.match_info["session_id"]
        source, err = self._get_existing_session_or_404(source_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        db = self._ensure_session_db()
        fork_id = str(body.get("id") or body.get("session_id") or f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}").strip()
        if not fork_id or re.search(r'[\r\n\x00]', fork_id):
            return web.json_response(_openai_error("Invalid session ID", code="invalid_session_id"), status=400)
        if db.get_session(fork_id):
            return web.json_response(_openai_error(f"Session already exists: {fork_id}", code="session_exists"), status=409)

        # Match the CLI /branch semantics: mark the original as branched, then
        # create a child session that carries the transcript forward. This uses
        # SessionDB's native parent_session_id/end_reason visibility model rather
        # than inventing a parallel fork store.
        db.end_session(source_id, "branched")
        db.create_session(
            fork_id,
            "api_server",
            model=source.get("model"),
            system_prompt=source.get("system_prompt"),
            parent_session_id=source_id,
        )
        messages = db.get_messages(source_id)
        db.replace_messages(fork_id, messages)
        title = body.get("title")
        if title is None:
            base = source.get("title") or "fork"
            try:
                title = db.get_next_title_in_lineage(base)
            except Exception:
                title = f"{base} fork"
        try:
            db.set_session_title(fork_id, str(title))
        except ValueError as exc:
            return web.json_response(_openai_error(str(exc), code="invalid_title"), status=400)
        fork = db.get_session(fork_id) or {"id": fork_id, "parent_session_id": source_id}
        return web.json_response({"object": "hermes.session", "session": self._session_response(fork)}, status=201)

    async def _handle_session_chat(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/chat — one synchronous agent turn."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err
        session_id = request.match_info["session_id"]
        _, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        user_message, err = _session_chat_user_message(body)
        if err is not None:
            return err
        system_prompt = body.get("system_message") or body.get("instructions")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_message must be a string", code="invalid_system_message"), status=400)
        history = self._conversation_history_for_session(session_id)
        result, usage = await self._run_agent(
            user_message=user_message,
            conversation_history=history,
            ephemeral_system_prompt=system_prompt,
            session_id=session_id,
            gateway_session_key=gateway_session_key,
        )
        effective_session_id = result.get("session_id") if isinstance(result, dict) else session_id
        final_response = result.get("final_response", "") if isinstance(result, dict) else ""
        headers = {"X-Hermes-Session-Id": effective_session_id or session_id}
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        return web.json_response(
            {
                "object": "hermes.session.chat.completion",
                "session_id": effective_session_id or session_id,
                "message": {"role": "assistant", "content": final_response},
                "usage": usage,
            },
            headers=headers,
        )

    async def _handle_session_chat_stream(self, request: "web.Request") -> "web.StreamResponse":
        """POST /api/sessions/{session_id}/chat/stream — SSE wrapper over _run_agent."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err
        session_id = request.match_info["session_id"]
        _, err = self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        user_message, err = _session_chat_user_message(body)
        if err is not None:
            return err
        system_prompt = body.get("system_message") or body.get("instructions")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return web.json_response(_openai_error("system_message must be a string", code="invalid_system_message"), status=400)

        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Optional[tuple[str, Dict[str, Any]]]]" = asyncio.Queue()
        message_id = f"msg_{uuid.uuid4().hex}"
        run_id = f"run_{uuid.uuid4().hex}"
        seq = 0

        def _event_payload(name: str, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
            nonlocal seq
            seq += 1
            payload.setdefault("session_id", session_id)
            payload.setdefault("run_id", run_id)
            payload.setdefault("seq", seq)
            payload.setdefault("ts", time.time())
            return name, payload

        def _enqueue(name: str, payload: Dict[str, Any]) -> None:
            event = _event_payload(name, payload)
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            try:
                if running_loop is loop:
                    queue.put_nowait(event)
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                pass

        def _delta(delta: str) -> None:
            if delta:
                _enqueue("assistant.delta", {"message_id": message_id, "delta": delta})

        def _tool_progress(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs) -> None:
            if event_type == "reasoning.available":
                _enqueue("tool.progress", {"message_id": message_id, "tool_name": tool_name or "_thinking", "delta": preview or ""})
            elif event_type in {"tool.started", "tool.completed", "tool.failed"}:
                event_name = event_type.replace("tool.", "tool.")
                _enqueue(event_name, {"message_id": message_id, "tool_name": tool_name, "preview": preview, "args": args})

        async def _run_and_signal() -> None:
            try:
                await queue.put(_event_payload("run.started", {"user_message": {"role": "user", "content": user_message}}))
                await queue.put(_event_payload("message.started", {"message": {"id": message_id, "role": "assistant"}}))
                history = self._conversation_history_for_session(session_id)
                result, usage = await self._run_agent(
                    user_message=user_message,
                    conversation_history=history,
                    ephemeral_system_prompt=system_prompt,
                    session_id=session_id,
                    stream_delta_callback=_delta,
                    tool_progress_callback=_tool_progress,
                    gateway_session_key=gateway_session_key,
                )
                final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                effective_session_id = result.get("session_id", session_id) if isinstance(result, dict) else session_id
                turn_messages = self._turn_transcript_messages(history, user_message, result) if isinstance(result, dict) else []
                await queue.put(_event_payload("assistant.completed", {
                    "session_id": effective_session_id,
                    "message_id": message_id,
                    "content": final_response,
                    "completed": True,
                    "partial": False,
                    "interrupted": False,
                }))
                await queue.put(_event_payload("run.completed", {
                    "session_id": effective_session_id,
                    "message_id": message_id,
                    "completed": True,
                    "messages": turn_messages,
                    "usage": usage,
                }))
            except Exception as exc:
                logger.exception("[api_server] session chat stream failed")
                await queue.put(_event_payload("error", {"message": str(exc)}))
            finally:
                await queue.put(_event_payload("done", {}))
                await queue.put(None)

        task = asyncio.create_task(_run_and_signal())
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Hermes-Session-Id": session_id,
        }
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        response = web.StreamResponse(status=200, headers=headers)
        await response.prepare(request)
        last_write = time.monotonic()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    last_write = time.monotonic()
                    continue
                if item is None:
                    break
                name, payload = item
                data = json.dumps(payload, ensure_ascii=False)
                await response.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
                last_write = time.monotonic()
        except (asyncio.CancelledError, ConnectionResetError):
            task.cancel()
            raise
        except Exception as exc:
            logger.debug("[api_server] session SSE stream error: %s", exc)
        return response

     # ------------------------------------------------------------------
     # Server-side slash commands
     # ------------------------------------------------------------------

    def _is_api_slash_command(self, text: Any) -> bool:
        """Return True when ``text`` is a recognised gateway slash command.

        The chat-completions / responses handlers use this to intercept
        commands such as ``/help`` or ``/new`` and answer them
        server-side instead of forwarding them to the agent loop.
        """
        raw_text = ""
        if isinstance(text, str):
            raw_text = text
        elif isinstance(text, list):
            for part in text:
                if isinstance(part, dict) and part.get("type") == "text":
                    raw_text = part.get("text", "")
                    break
        
        raw_text = raw_text.strip()
        if not raw_text or not raw_text.startswith("/"):
            return False
            
        parts = raw_text.split(maxsplit=1)
        cmd_name = parts[0][1:].lower()
        from hermes_cli.commands import resolve_command
        return resolve_command(cmd_name) is not None

    def _find_active_run_id_for_session(self, session_id: str) -> Optional[str]:
        """Return the run/completion id for an in-flight agent turn on a session."""
        if not session_id:
            return None
        normalized = session_id.casefold()
        for run_id, status in self._run_statuses.items():
            active_session = str(status.get("session_id") or "").casefold()
            if active_session != normalized:
                continue
            if status.get("status") in {"running", "waiting_for_approval", "stopping"}:
                return run_id
        return None

    def _interrupt_run(self, run_id: str) -> str:
        """Stop an active run by id; returns user-facing markdown."""
        agent = self._active_run_agents.get(run_id)
        task = self._active_run_tasks.get(run_id)

        if agent is None and task is None:
            return f"No active run found for `{run_id}`."

        self._set_run_status(run_id, "stopping", last_event="run.stopping")

        if agent is not None:
            try:
                agent.interrupt("Stop requested via slash command")
            except Exception:  # noqa: BLE001
                pass

        if task is not None and not task.done():
            task.cancel()

        return f"Stop requested for run `{run_id}`."

    def _apply_session_model_from_slash(
        self, session_id: str, model_arg: str,
    ) -> tuple[bool, str]:
        """Apply a /model slash argument to a session override."""
        model_arg = (model_arg or "").strip()
        if not model_arg:
            return False, "usage: /model <provider/model> or /model <model-name>"

        try:
            from gateway.run import _resolve_gateway_model
            from hermes_cli.config import get_compatible_custom_providers, load_config
            from hermes_cli.model_switch import switch_model
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime = resolve_runtime_provider(requested=None)
            current_api_key = runtime.get("api_key", "")
            if not callable(current_api_key):
                current_api_key = str(current_api_key or "")

            cfg = load_config()
            explicit_provider = ""
            model_input = model_arg
            if "/" in model_arg and not model_arg.startswith("/"):
                explicit_provider, _, model_input = model_arg.partition("/")
                explicit_provider = explicit_provider.strip()
                model_input = model_input.strip()

            result = switch_model(
                raw_input=model_input,
                current_provider=str(runtime.get("provider", "") or ""),
                current_model=_resolve_gateway_model(),
                current_base_url=str(runtime.get("base_url", "") or ""),
                current_api_key=current_api_key,
                is_global=False,
                explicit_provider=explicit_provider,
                user_providers=cfg.get("providers") if isinstance(cfg.get("providers"), dict) else None,
                custom_providers=get_compatible_custom_providers(cfg),
            )
            if not result.success:
                return False, result.error_message or "model switch failed"

            self._session_model_overrides[session_id] = {
                "model": result.new_model,
                "provider": result.target_provider,
                "api_key": result.api_key,
                "base_url": result.base_url,
                "api_mode": result.api_mode,
            }
            return True, (
                f"Session model set to **{result.target_provider}/{result.new_model}** "
                f"for session `{session_id}`."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("slash /model failed for %s: %s", session_id, exc)
            return False, f"Could not switch model: {exc}"

    def _dispatch_codex_runtime_slash(self, cmd_arg: str, session_id: str) -> dict:
        """Toggle or query the optional codex app-server runtime."""
        if self._find_active_run_id_for_session(session_id):
            return {
                "type": "error",
                "message": (
                    "Agent is running — wait or /stop first, then change runtime."
                ),
            }

        from hermes_cli import codex_runtime_switch as crs

        new_value, errors = crs.parse_args(cmd_arg)
        if errors:
            return {"type": "error", "message": "❌ " + "\n❌ ".join(errors)}

        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:  # noqa: BLE001
            return {"type": "error", "message": f"Could not load config: {exc}"}

        cfg = load_config()
        result = crs.apply(
            cfg,
            new_value,
            persist_callback=(save_config if new_value is not None else None),
        )
        prefix = "✓" if result.success else "✗"
        if result.success:
            return {"type": "text", "content": f"{prefix} {result.message}"}
        return {"type": "error", "message": f"{prefix} {result.message}"}

    def _dispatch_slash_command(
        self,
        command_text: str,
        session_id: str,
        gateway_session_key: str = None,
    ) -> dict:
        """Dispatch a slash command and return a structured result dict."""
        command_text = (command_text or "").strip()
        if not command_text.startswith("/"):
            return {"type": "error", "message": "Not a slash command."}

        parts = command_text.split(maxsplit=1)
        cmd_name = parts[0][1:].lower()
        cmd_arg = parts[1] if len(parts) > 1 else ""

        # quick_commands
        try:
            from hermes_cli.config import load_config

            qcmds = load_config().get("quick_commands", {}) or {}
            if isinstance(qcmds, dict) and cmd_name in qcmds:
                qc = qcmds[cmd_name]
                if isinstance(qc, dict) and qc.get("type") == "alias":
                    return {"type": "alias", "target": str(qc.get("target", ""))}
                if isinstance(qc, dict) and qc.get("type") == "exec":
                    import subprocess

                    proc = subprocess.run(
                        str(qc.get("command", "")),
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    output = (
                        (proc.stdout or "")
                        + ("\n" if proc.stdout and proc.stderr else "")
                        + (proc.stderr or "")
                    ).strip()[:4000]
                    if proc.returncode != 0:
                        return {
                            "type": "error",
                            "message": output or f"quick command failed with exit code {proc.returncode}",
                        }
                    return {"type": "text", "content": output or "(no output)"}
        except Exception as exc:  # noqa: BLE001
            logger.debug("quick_commands dispatch failed: %s", exc)

        # plugin commands
        try:
            from hermes_cli.plugins import (
                get_plugin_command_handler,
                resolve_plugin_command_result,
            )

            handler = get_plugin_command_handler(cmd_name)
            if handler:
                result = resolve_plugin_command_result(handler(cmd_arg))
                return {"type": "text", "content": str(result or "(no output)")}
        except Exception:  # noqa: BLE001
            pass

        # skill commands
        try:
            from agent.skill_commands import (
                build_skill_invocation_message,
                scan_skill_commands,
            )

            skill_key = f"/{cmd_name}"
            skill_cmds = scan_skill_commands()
            if skill_key in skill_cmds:
                msg = build_skill_invocation_message(
                    skill_key, cmd_arg, task_id=session_id or "",
                )
                if msg:
                    return {
                        "type": "skill",
                        "message": msg,
                        "name": skill_cmds[skill_key].get("name", cmd_name),
                    }
        except Exception:  # noqa: BLE001
            pass

        from hermes_cli.commands import resolve_command

        cmd = resolve_command(cmd_name)
        canonical = cmd.name if cmd is not None else cmd_name

        if cmd is None and cmd_name not in NATIVE_APP_SLASH_COMMANDS:
            return {
                "type": "error",
                "message": f"Unknown command `/{cmd_name}`. Send `/help` for available commands.",
            }

        if canonical == "queue" or cmd_name in {"queue", "q"}:
            if not cmd_arg:
                return {"type": "error", "message": "usage: /queue <prompt>"}
            return {"type": "send", "message": cmd_arg}

        if canonical == "steer":
            if not cmd_arg:
                return {"type": "error", "message": "usage: /steer <prompt>"}
            run_id = self._find_active_run_id_for_session(session_id)
            agent = self._active_run_agents.get(run_id) if run_id else None
            if agent is not None and hasattr(agent, "steer"):
                try:
                    if agent.steer(cmd_arg):
                        preview = cmd_arg[:80] + ("..." if len(cmd_arg) > 80 else "")
                        return {
                            "type": "text",
                            "content": (
                                f"Steer queued — arrives after the next tool call: {preview}"
                            ),
                        }
                except Exception as exc:  # noqa: BLE001
                    logger.debug("steer failed: %s", exc)
            return {"type": "send", "message": cmd_arg}

        if canonical == "background" or cmd_name in {"bg", "btw"}:
            if not cmd_arg.strip():
                return {"type": "error", "message": "usage: /background <prompt>"}
            return self._dispatch_api_background_command(
                session_id,
                cmd_arg.strip(),
                gateway_session_key=gateway_session_key,
            )

        if canonical == "stop":
            run_id = self._find_active_run_id_for_session(session_id)
            if not run_id:
                return {"type": "error", "message": "No active agent run to stop for this session."}
            content = self._interrupt_run(run_id)
            return {"type": "text", "content": content}

        if canonical == "model":
            ok, message = self._apply_session_model_from_slash(session_id, cmd_arg)
            if ok:
                return {"type": "text", "content": message}
            return {"type": "error", "message": message}

        if canonical in ("help", "commands"):
            from hermes_cli.commands import gateway_help_lines

            lines = gateway_help_lines()
            body = "\n".join(f"- {ln}" for ln in lines) or "_(no commands available)_"
            return {"type": "text", "content": "**Available commands**\n\n" + body}

        if canonical == "status":
            title = None
            try:
                db = self._ensure_session_db()
                if db is not None:
                    title = db.get_session_title(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("status command: could not load session title: %s", exc)
            content = "**Session status**\n\n" + "\n".join([
                f"- **Session ID:** `{session_id}`",
                f"- **Title:** {title or '(untitled)'}",
                "- **Platform:** API Server",
            ])
            return {"type": "text", "content": content}

        if canonical == "profile":
            try:
                from hermes_cli.profiles import get_active_profile_name

                profile_name = get_active_profile_name()
            except Exception as exc:  # noqa: BLE001
                logger.debug("profile command: could not resolve profile: %s", exc)
                profile_name = "unknown"
            try:
                from hermes_constants import get_hermes_home

                home = str(get_hermes_home())
            except Exception:  # noqa: BLE001
                home = os.path.expanduser("~/.hermes")
            return {
                "type": "text",
                "content": (
                    "**Active profile**\n\n"
                    f"- **Profile:** `{profile_name}`\n"
                    f"- **Home:** `{home}`"
                ),
            }

        if canonical == "yolo":
            key = gateway_session_key or session_id
            try:
                from tools import approval

                if approval.is_session_yolo_enabled(key):
                    approval.disable_session_yolo(key)
                    return {"type": "text", "content": "YOLO mode **disabled** for this session."}
                approval.enable_session_yolo(key)
                return {
                    "type": "text",
                    "content": (
                        "YOLO mode **enabled** for this session — tool "
                        "approvals will be auto-accepted."
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("yolo command failed: %s", exc)
                return {"type": "error", "message": f"Could not toggle YOLO mode: {exc}"}

        if canonical == "codex-runtime":
            return self._dispatch_codex_runtime_slash(cmd_arg, session_id)

        if canonical == "goal":
            return self._dispatch_goal_slash(session_id, cmd_arg)

        if canonical == "subgoal":
            return self._dispatch_subgoal_slash(session_id, cmd_arg)

        if canonical == "restart":
            try:
                import weakref

                from gateway.run import _gateway_runner_ref

                runner = _gateway_runner_ref()
                if runner is not None:
                    runner.request_restart(detached=True)
                    return {
                        "type": "text",
                        "content": "Restart requested! Reloading the Hermes Gateway proxy...",
                    }
                import subprocess

                subprocess.Popen(
                    "hermes gateway restart",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return {
                    "type": "text",
                    "content": "Restart requested! Relaunching gateway via CLI helper...",
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("api_server restart slash command failed: %s", exc)
                return {"type": "error", "message": f"Could not request restart: {exc}"}

        if canonical in ("new", "reset", "clear"):
            try:
                db = self._ensure_session_db()
                if db is None:
                    return {
                        "type": "text",
                        "content": "Conversation cleared (no session database configured).",
                    }
                db.clear_messages(session_id)
                return {
                    "type": "text",
                    "content": (
                        f"Conversation cleared. Session `{session_id}` now has "
                        "no messages."
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("clear command failed for %s: %s", session_id, exc)
                return {"type": "error", "message": f"Could not clear the conversation: {exc}"}

        return {
            "type": "error",
            "message": (
                f"The command `/{cmd_name}` is recognised but is not available "
                "over the API server. Send `/help` for the list of supported "
                "commands."
            ),
        }

    def _dispatch_api_background_command(
        self,
        parent_session_id: str,
        prompt: str,
        gateway_session_key: str = None,
    ) -> dict:
        """Start a Hermes-native /background task for an API session."""
        from datetime import datetime

        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"
        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")

        self._persist_subagent_event(
            {
                "event": "subagent.start",
                "subagent_id": task_id,
                "run_id": task_id,
                "depth": 0,
                "goal": prompt,
                "status": "running",
                "runtime": "background",
                "preview": preview,
                "timestamp": time.time(),
            },
            session_id=parent_session_id,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return {
                "type": "error",
                "message": "Background tasks require an active API server loop.",
            }

        task = loop.create_task(
            self._run_api_background_task(
                parent_session_id=parent_session_id,
                prompt=prompt,
                task_id=task_id,
                gateway_session_key=gateway_session_key,
            )
        )
        try:
            background_tasks = getattr(self, "_background_tasks", None)
            if background_tasks is not None:
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
        except Exception:  # noqa: BLE001
            pass

        content = (
            f'🔄 Background task started: "{preview}"\n'
            f"Task ID: {task_id}\n"
            f"You can keep chatting — results will appear when done."
        )
        return {"type": "text", "content": content}

    async def _run_api_background_task(
        self,
        *,
        parent_session_id: str,
        prompt: str,
        task_id: str,
        gateway_session_key: str = None,
        is_follow_up: bool = False,
    ) -> None:
        """Execute a background prompt in an isolated session and mirror progress to the parent."""
        started = time.time()
        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")

        def _persist(event: Dict[str, Any]) -> None:
            payload = dict(event)
            payload.setdefault("subagent_id", task_id)
            payload.setdefault("run_id", task_id)
            payload.setdefault("goal", prompt)
            payload.setdefault("runtime", "background")
            payload.setdefault("depth", 0)
            payload.setdefault("timestamp", time.time())
            self._persist_subagent_event(payload, session_id=parent_session_id)

        if is_follow_up:
            _persist({
                "event": "subagent.progress",
                "subagent_id": task_id,
                "status": "running",
                "message": "Follow-up sent",
                "preview": preview,
            })

        def _tool_progress(event_type, tool_name=None, preview=None, args=None, **kwargs):
            if str(event_type).startswith("subagent."):
                event = self._subagent_event_payload(
                    event_type,
                    task_id,
                    tool_name=tool_name,
                    preview=preview,
                    args=args,
                    extra={
                        **kwargs,
                        "subagent_id": task_id,
                        "goal": prompt,
                        "runtime": "background",
                        "depth": 0,
                    },
                )
                event["subagent_id"] = task_id
                _persist(event)
                return
            if not tool_name or str(tool_name).startswith("_"):
                return
            if event_type == "tool.started":
                _persist({
                    "event": "subagent.tool",
                    "subagent_id": task_id,
                    "tool": tool_name,
                    "tool_name": tool_name,
                    "preview": preview,
                    "status": "running",
                })
            elif event_type == "tool.completed":
                _persist({
                    "event": "subagent.tool",
                    "subagent_id": task_id,
                    "tool": tool_name,
                    "tool_name": tool_name,
                    "preview": preview,
                    "status": "failed" if kwargs.get("is_error") else "completed",
                })

        status = "completed"
        summary = ""
        usage: Dict[str, Any] = {}
        try:
            result, usage = await self._run_agent(
                user_message=prompt,
                conversation_history=[],
                session_id=task_id,
                tool_progress_callback=_tool_progress,
                gateway_session_key=gateway_session_key,
                run_id=task_id,
            )
            if isinstance(result, dict) and result.get("failed"):
                status = "failed"
                summary = str(result.get("error") or "Background task failed.")
            else:
                summary = (
                    str(result.get("final_response") or "")
                    if isinstance(result, dict)
                    else str(result or "")
                )
                if not summary.strip():
                    summary = "Background task completed."
        except Exception as exc:  # noqa: BLE001
            logger.warning("api_server background task %s failed: %s", task_id, exc)
            status = "failed"
            summary = str(exc)
        finally:
            self._active_run_agents.pop(task_id, None)

        complete_payload: Dict[str, Any] = {
            "event": "subagent.complete",
            "subagent_id": task_id,
            "status": status,
            "summary": summary[:4000],
            "preview": preview,
            "duration_seconds": round(time.time() - started, 2),
        }
        if status == "completed" and usage:
            complete_payload["input_tokens"] = usage.get("input_tokens")
            complete_payload["output_tokens"] = usage.get("output_tokens")
        _persist(complete_payload)

    async def _handle_background_task_follow_up(self, request: "web.Request") -> "web.Response":
        """POST /v1/background/tasks/{task_id}/follow-up — reply to a background agent."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        task_id = str(request.match_info.get("task_id") or "").strip()
        if not task_id.startswith("bg_"):
            return web.json_response(_openai_error("Invalid background task id"), status=400)

        try:
            body = await request.json()
        except Exception:
            body = {}

        message = str(body.get("message") or "").strip()
        if not message:
            return web.json_response(_openai_error("Follow-up message required"), status=400)

        parent_session_id = str(body.get("session_id") or "").strip()
        if not parent_session_id:
            parent_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if not parent_session_id:
            return web.json_response(_openai_error("session_id is required"), status=400)

        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return key_err

        preview = message[:80] + ("..." if len(message) > 80 else "")
        agent = self._active_run_agents.get(task_id)
        if agent is not None and hasattr(agent, "steer"):
            try:
                if agent.steer(message):
                    event = self._persist_subagent_event(
                        {
                            "event": "subagent.progress",
                            "subagent_id": task_id,
                            "run_id": task_id,
                            "runtime": "background",
                            "status": "running",
                            "message": "Follow-up steered",
                            "preview": preview,
                            "timestamp": time.time(),
                        },
                        session_id=parent_session_id,
                    )
                    return web.json_response({
                        "ok": True,
                        "task_id": task_id,
                        "mode": "steer",
                        "message": "Follow-up queued for the running background agent.",
                        "event": event,
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("background follow-up steer failed for %s: %s", task_id, exc)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return web.json_response(
                _openai_error("Background follow-up requires an active API server loop."),
                status=503,
            )

        task = loop.create_task(
            self._run_api_background_task(
                parent_session_id=parent_session_id,
                prompt=message,
                task_id=task_id,
                gateway_session_key=gateway_session_key,
                is_follow_up=True,
            )
        )
        try:
            background_tasks = getattr(self, "_background_tasks", None)
            if background_tasks is not None:
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
        except Exception:  # noqa: BLE001
            pass

        return web.json_response({
            "ok": True,
            "task_id": task_id,
            "mode": "continue",
            "message": "Follow-up sent to the background agent.",
        })

    def _execute_api_slash_command(
        self, command_text: str, session_id: str, gateway_session_key: str = None
    ) -> str:
        """Execute a recognised gateway slash command server-side.

        Returns a markdown string suitable for handing straight back to an
        OpenAI-compatible client as assistant message content.
        """
        result = self._dispatch_slash_command(
            command_text, session_id, gateway_session_key=gateway_session_key,
        )
        result_type = result.get("type")
        if result_type == "text":
            return str(result.get("content") or "")
        if result_type == "error":
            return str(result.get("message") or "Command failed.")
        if result_type == "send":
            return f"Queued message for next turn: {result.get('message', '')}"
        if result_type == "skill":
            return str(result.get("message") or "")
        if result_type == "alias":
            target = str(result.get("target") or "").strip()
            if target:
                return self._execute_api_slash_command(
                    target if target.startswith("/") else f"/{target}",
                    session_id,
                    gateway_session_key=gateway_session_key,
                )
        return str(result.get("message") or result.get("content") or "Command failed.")

    async def _api_slash_command_chat_response(
        self,
        request: "web.Request",
        command_feedback: str,
        *,
        stream: bool,
        completion_id: str,
        model_name: str,
        created: int,
        session_id: str,
        gateway_session_key: str = None,
    ) -> "web.Response":
        """Pack a slash-command result into a Chat Completions response.

        Non-streaming returns an immediate OpenAI ``chat.completion``
        JSON body.  Streaming feeds the markdown through a one-shot queue
        and an already-resolved ``agent_task`` future so the existing SSE
        writer streams it without ever spawning the agent.
        """
        if stream:
            import queue as _q
            cmd_q: "_q.Queue" = _q.Queue()
            cmd_q.put(command_feedback)
            cmd_q.put(None)  # EOS sentinel
            done_task: "asyncio.Future" = asyncio.get_running_loop().create_future()
            # _write_sse_chat_completion does `result, usage = await agent_task`.
            done_task.set_result((
                {"final_response": command_feedback},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ))
            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, cmd_q,
                done_task, None, session_id=session_id,
                gateway_session_key=gateway_session_key,
            )

        response_headers = {"X-Hermes-Session-Id": session_id}
        if gateway_session_key:
            response_headers["X-Hermes-Session-Key"] = gateway_session_key
        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": command_feedback},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return web.json_response(response_data, headers=response_headers)

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

        project_overlay = build_chat_project_context_overlay(body)
        if project_overlay:
            system_prompt = (
                f"{system_prompt}\n\n{project_overlay}"
                if system_prompt
                else project_overlay
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

        # ── Server-side slash command interception ──
        # Recognised gateway commands (/help, /status, /new, …) are
        # answered by the API server directly and never reach the agent.
        if self._is_api_slash_command(user_message):
            command_feedback = self._execute_api_slash_command(
                user_message, session_id, gateway_session_key=gateway_session_key,
            )
            return await self._api_slash_command_chat_response(
                request, command_feedback,
                stream=stream, completion_id=completion_id,
                model_name=model_name, created=created,
                session_id=session_id, gateway_session_key=gateway_session_key,
            )

        if stream:
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()
            approval_session_key = gateway_session_key or session_id or completion_id
            self._run_approval_sessions[completion_id] = approval_session_key
            self._set_run_status(
                completion_id,
                "running",
                created_at=time.time(),
                session_id=session_id,
                model=model_name,
                gateway_session_key=gateway_session_key,
            )

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

            def _on_reasoning(delta):
                if delta:
                    _stream_q.put(("__reasoning_delta__", {"text": delta}))

            # Track which tool_call_ids we've emitted a "running" lifecycle
            # event for, so a "completed" event without a matching "running"
            # (e.g. internal/filtered tools) is silently dropped instead of
            # producing an orphaned event clients can't correlate.
            _started_tool_call_ids: set[str] = set()
            _legacy_started_tools: set[str] = set()
            def _on_tool_progress(event_type, tool_name=None, preview=None, args=None, **kwargs):
                """Bridge non-tool progress into chat-completions SSE.

                Tool start/complete use structured callbacks so clients receive
                correlated ``toolCallId`` payloads (and todo arguments/results).
                Legacy ``tool.started`` / ``tool.completed`` events must not be
                mirrored here — they previously suppressed the structured pair.
                """
                if str(event_type).startswith("subagent."):
                    event = self._subagent_event_payload(
                        event_type,
                        completion_id,
                        tool_name=tool_name,
                        preview=preview,
                        args=args,
                        extra=kwargs,
                    )
                    event = self._persist_subagent_event(event, session_id=session_id)
                    _stream_q.put(("__subagent_event__", event))
                    return
                if event_type == "reasoning.available" and preview:
                    _stream_q.put(("__reasoning_delta__", {"text": preview}))
                    return

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
                if function_name in _legacy_started_tools:
                    return
                from agent.display import build_tool_preview, get_tool_emoji
                label = build_tool_preview(function_name, function_args) or function_name
                progress_payload = {
                    "tool": function_name,
                    "emoji": get_tool_emoji(function_name),
                    "label": label,
                    "toolCallId": tool_call_id,
                    "status": "running",
                }
                if function_name == "todo" and function_args:
                    try:
                        progress_payload["arguments"] = json.dumps(function_args)
                    except Exception:
                        pass
                _stream_q.put(("__tool_progress__", progress_payload))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Emit the matching ``status: completed`` event.

                Dropped if the start was filtered (internal tool, missing
                id, or never seen) so clients never get an orphaned
                ``completed`` they can't correlate to a prior ``running``.
                """
                if not tool_call_id or tool_call_id not in _started_tool_call_ids:
                    return
                _started_tool_call_ids.discard(tool_call_id)
                progress_payload = {
                    "tool": function_name,
                    "toolCallId": tool_call_id,
                    "status": "completed",
                }
                if function_name == "todo" and function_result:
                    progress_payload["result"] = function_result
                _stream_q.put(("__tool_progress__", progress_payload))

            def _on_context_usage(payload: Dict[str, Any]) -> None:
                if payload:
                    _stream_q.put(("__context_usage__", payload))

            def _on_approval_request(approval_data: Dict[str, Any]) -> None:
                event = dict(approval_data or {})
                event.update({
                    "run_id": completion_id,
                    "session_id": session_id,
                    "gateway_session_key": gateway_session_key,
                    "timestamp": time.time(),
                    "choices": ["once", "session", "always", "deny"],
                })
                self._set_run_status(
                    completion_id,
                    "waiting_for_approval",
                    last_event="approval.request",
                )
                _stream_q.put(("__approval_request__", event))

            def _on_approval_request(approval_data: Dict[str, Any]) -> None:
                event = dict(approval_data or {})
                event.update({
                    "run_id": completion_id,
                    "session_id": session_id,
                    "gateway_session_key": gateway_session_key,
                    "timestamp": time.time(),
                    "choices": ["once", "session", "always", "deny"],
                })
                self._set_run_status(
                    completion_id,
                    "waiting_for_approval",
                    last_event="approval.request",
                )
                _stream_q.put(("__approval_request__", event))

            # Start agent in background.  agent_ref is a mutable container
            # so the SSE writer can interrupt the agent on client disconnect.
            #
            # ``tool_progress_callback`` is intentionally not wired here:
            # it would duplicate every emit because ``run_agent`` fires it
            # side-by-side with ``tool_start_callback``/``tool_complete_callback``.
            # The structured callbacks are strictly richer (they carry the
            # tool_call id), so they own the chat-completions SSE channel.
            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                reasoning_callback=_on_reasoning,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                context_usage_callback=_on_context_usage,
                agent_ref=agent_ref,
                gateway_session_key=gateway_session_key,
                model_override=model_name,
                approval_notify_callback=_on_approval_request,
                run_id=completion_id,
            ))
            self._active_run_tasks[completion_id] = agent_task

            def _cleanup_chat_run(_fut):
                _stream_q.put(None)
                self._active_run_tasks.pop(completion_id, None)
                self._active_run_agents.pop(completion_id, None)
                self._set_run_status(completion_id, "completed", last_event="run.completed")

            agent_task.add_done_callback(_cleanup_chat_run)

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
                gateway_session_key=gateway_session_key,
            )

        # Non-streaming: run the agent (with optional Idempotency-Key)
        async def _compute_completion():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
                model_override=model_name,
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

            # Tracks whether any assistant *text* content was streamed as
            # ``delta.content`` chunks.  Reasoning deltas, tool progress, and
            # other tagged events do NOT count.  When this stays False for the
            # whole turn (e.g. the Codex Responses/app-server runtimes deliver
            # the answer via a completed message item rather than
            # ``output_text.delta`` events), we fall back to emitting the
            # agent's ``final_response`` so the client renders something
            # instead of an empty assistant turn.  See the non-streaming path
            # which already returns ``final_response`` directly.
            streamed_text_content = {"value": False}

            # Helper — route a queue item to the correct SSE event.
            async def _emit(item):
                """Write a single queue item to the SSE stream.

                Plain strings are sent as normal ``delta.content`` chunks.
                Tagged tuples are sent as custom SSE events so frontends can
                display auxiliary agent state without storing markers in
                conversation history. Tool progress uses
                ``hermes.tool.progress`` (see #6972 / #16588), and reasoning
                deltas use ``hermes.reasoning.delta`` for live thought panels.
                """
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "__tool_progress__":
                    event_data = json.dumps(item[1])
                    await response.write(
                        f"event: hermes.tool.progress\ndata: {event_data}\n\n".encode()
                    )
                elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__reasoning_delta__":
                    event_data = json.dumps(item[1])
                    await response.write(
                        f"event: hermes.reasoning.delta\ndata: {event_data}\n\n".encode()
                    )
                elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__approval_request__":
                    event_data = json.dumps(item[1])
                    await response.write(
                        f"event: approval.request\ndata: {event_data}\n\n".encode()
                    )
                elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__context_usage__":
                    event_data = json.dumps(item[1])
                    await response.write(
                        f"event: hermes.context.usage\ndata: {event_data}\n\n".encode()
                    )
                elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__subagent_event__":
                    event_name = item[1].get("event", "subagent.progress")
                    event_data = json.dumps(item[1])
                    await response.write(
                        f"event: {event_name}\ndata: {event_data}\n\n".encode()
                    )
                else:
                    if item:
                        streamed_text_content["value"] = True
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
            finish_reason = "stop"
            goal_status_payload = None
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                if isinstance(result, dict) and result.get("failed"):
                    finish_reason = "error"
                    err_msg = result.get("error") or "agent run failed"
                    self._set_run_status(
                        completion_id,
                        "failed",
                        error=err_msg,
                        last_event="run.failed",
                    )
                    error_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": f"\n\nHermes run failed: {err_msg}"
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                else:
                    # Fallback: some runtimes (Codex Responses / app-server)
                    # deliver the assistant answer as a completed message item
                    # rather than ``output_text.delta`` events, so nothing was
                    # streamed as ``delta.content`` even though the turn
                    # succeeded.  Emit ``final_response`` now so the client
                    # renders the answer instead of an empty assistant turn
                    # (which makes the live reasoning panel vanish with no
                    # replacement).
                    final_response_text = (
                        result.get("final_response", "") if isinstance(result, dict) else ""
                    )
                    if final_response_text and not streamed_text_content["value"]:
                        fallback_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": final_response_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        await response.write(
                            f"data: {json.dumps(fallback_chunk)}\n\n".encode()
                        )
                        streamed_text_content["value"] = True
                    self._set_run_status(
                        completion_id,
                        "completed",
                        output=final_response_text,
                        usage=usage,
                        last_event="run.completed",
                    )
                    if session_id and isinstance(result, dict) and not result.get("failed"):
                        final_for_judge = (
                            final_response_text
                            or result.get("final_response")
                            or ""
                        )
                        goal_status_payload = self._evaluate_session_goal_after_turn(
                            session_id,
                            final_for_judge,
                        )
            except Exception as exc:
                logger.warning("Agent task %s failed, usage data lost: %s", completion_id, exc)
                finish_reason = "error"
                err_msg = str(exc)
                self._set_run_status(
                    completion_id,
                    "failed",
                    error=err_msg,
                    last_event="run.failed",
                )
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"\n\nHermes run failed: {err_msg}"
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())

            if goal_status_payload:
                try:
                    await response.write(
                        f"event: hermes.goal.status\ndata: {json.dumps(goal_status_payload)}\n\n".encode()
                    )
                except Exception:
                    logger.debug("Failed to emit goal status SSE event", exc_info=True)

            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    from agent.context_usage import build_context_usage_payload

                    final_context_payload = build_context_usage_payload(agent)
                    await response.write(
                        f"event: hermes.context.usage\ndata: {json.dumps(final_context_payload)}\n\n".encode()
                    )
                except Exception:
                    logger.debug("Failed to emit final context usage event", exc_info=True)

            # Finish chunk
            finish_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
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

        finally:
            self._run_approval_sessions.pop(completion_id, None)

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

        # ── Server-side slash command interception ──
        # Mirrors _handle_chat_completions: recognised gateway commands are
        # answered by the API server directly, packed into the Responses
        # API payload shape, and never reach the agent.
        if self._is_api_slash_command(user_message):
            command_feedback = self._execute_api_slash_command(
                user_message, session_id, gateway_session_key=gateway_session_key,
            )
            response_id = f"resp_{uuid.uuid4().hex[:28]}"
            created_at = int(time.time())
            model_name = body.get("model", self._model_name)
            if stream:
                import queue as _q
                cmd_q: "_q.Queue" = _q.Queue()
                cmd_q.put(command_feedback)
                cmd_q.put(None)  # EOS sentinel
                done_task: "asyncio.Future" = asyncio.get_running_loop().create_future()
                # _write_sse_responses does `result, usage = await agent_task`.
                done_task.set_result((
                    {"final_response": command_feedback},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                ))
                return await self._write_sse_responses(
                    request=request,
                    response_id=response_id,
                    model=model_name,
                    created_at=created_at,
                    stream_q=cmd_q,
                    agent_task=done_task,
                    agent_ref=[None],
                    conversation_history=conversation_history,
                    user_message=user_message,
                    instructions=instructions,
                    conversation=conversation,
                    store=store,
                    session_id=session_id,
                    gateway_session_key=gateway_session_key,
                )
            response_data = {
                "id": response_id,
                "object": "response",
                "status": "completed",
                "created_at": created_at,
                "model": model_name,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": command_feedback}],
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
            response_headers = {"X-Hermes-Session-Id": session_id}
            if gateway_session_key:
                response_headers["X-Hermes-Session-Key"] = gateway_session_key
            return web.json_response(response_data, headers=response_headers)

        if stream:
            # Streaming branch — emit OpenAI Responses SSE events as the
            # agent runs so frontends can render text deltas and tool
            # calls in real time.  See _write_sse_responses for details.
            import queue as _q
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
            agent_task = asyncio.ensure_future(self._run_agent(
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

            response_id = f"resp_{uuid.uuid4().hex[:28]}"
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
            return await self._run_agent(
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
            logger.warning(
                "Cron jobs API rejected invalid job_id %r: %s",
                job_id,
                self._request_audit_log_suffix(request),
            )
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
                "origin": self._cron_origin_from_request(request),
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

    @classmethod
    def _turn_transcript_messages(
        cls,
        conversation_history: List[Dict[str, Any]],
        user_message: Any,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return this turn's assistant/tool messages in client-safe shape.

        The streaming SSE contract delivers all assistant text as
        ``assistant.delta`` events under one ``message_id`` interleaved with
        ``tool.*`` events, and a single ``assistant.completed`` carrying only
        the final reply.  A client that accumulates deltas into one buffer
        cannot reconstruct *intermediate* assistant text segments that preceded
        tool calls — so when the page is re-opened mid/post-stream those
        segments appear lost, even though state.db persisted them correctly.

        Emitting the authoritative per-turn transcript on ``run.completed`` lets
        any SSE consumer reconcile its live view against ground truth without a
        separate ``GET /messages`` round-trip.  Purely additive: clients that
        ignore the field are unaffected.  Refs #34703.
        """
        agent_messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(agent_messages, list) or not agent_messages:
            return []
        start = cls._response_messages_turn_start_index(
            conversation_history, user_message, result
        )
        turn = agent_messages[start:]
        out: List[Dict[str, Any]] = []
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in {"assistant", "tool"}:
                continue
            out.append(cls._message_response(msg))
        return out

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
        reasoning_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        context_usage_callback=None,
        agent_ref: Optional[list] = None,
        gateway_session_key: Optional[str] = None,
        model_override: Optional[str] = None,
        approval_notify_callback=None,
        run_id: Optional[str] = None,
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

        def _run():
            approval_session_key = gateway_session_key or session_id or ""
            approval_token = None
            session_tokens = []
            agent = None
            agent = self._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
                reasoning_callback=reasoning_callback,
                tool_progress_callback=tool_progress_callback,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                context_usage_callback=context_usage_callback,
                gateway_session_key=gateway_session_key,
                model_override=model_override,
            )
            if agent_ref is not None:
                agent_ref[0] = agent
            if run_id:
                self._active_run_agents[run_id] = agent
            effective_task_id = session_id or str(uuid.uuid4())
            try:
                if approval_session_key:
                    try:
                        from gateway.session_context import clear_session_vars, set_session_vars
                        from tools.approval import (
                            register_gateway_notify,
                            reset_current_session_key,
                            set_current_session_key,
                            unregister_gateway_notify,
                        )

                        approval_token = set_current_session_key(approval_session_key)
                        session_tokens = set_session_vars(
                            platform="api_server",
                            session_key=approval_session_key,
                        )
                        if approval_notify_callback is not None:
                            register_gateway_notify(approval_session_key, approval_notify_callback)
                    except Exception:
                        logger.debug("Failed to bind API server approval context", exc_info=True)

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
            finally:
                if run_id:
                    self._active_run_agents.pop(run_id, None)
                if approval_session_key:
                    try:
                        from gateway.session_context import clear_session_vars
                        from tools.approval import reset_current_session_key, unregister_gateway_notify

                        unregister_gateway_notify(approval_session_key)
                        if approval_token is not None:
                            reset_current_session_key(approval_token)
                        if session_tokens:
                            clear_session_vars(session_tokens)
                    except Exception:
                        pass
                self._cleanup_agent_resources(agent)

        return await loop.run_in_executor(None, _run)

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

    def _resolve_active_approval_session_key(self, run_id: str) -> Optional[str]:
        """Find the approval session key for a run/completion/session identifier."""
        direct = self._run_approval_sessions.get(run_id)
        if direct:
            return direct

        for active_id, session_key in list(self._run_approval_sessions.items()):
            if run_id == session_key:
                return session_key
            status = self._run_statuses.get(active_id) or {}
            if run_id == status.get("session_id") or run_id == status.get("gateway_session_key"):
                return session_key
        return None

    def _subagent_event_payload(
        self,
        event_type: str,
        run_id: str,
        tool_name: str = None,
        preview: str = None,
        args=None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Normalize a Hermes-native subagent progress callback for SSE."""
        payload: Dict[str, Any] = {
            "event": event_type,
            "run_id": run_id,
            "timestamp": time.time(),
        }
        for key, value in (extra or {}).items():
            if value is not None:
                payload[key] = value
        if tool_name is not None:
            payload.setdefault("tool", tool_name)
            payload.setdefault("tool_name", tool_name)
        if preview is not None:
            payload.setdefault("preview", preview)
            if event_type == "subagent.thinking":
                payload.setdefault("text", preview)
            elif event_type == "subagent.tool":
                payload.setdefault("tool_preview", preview)
            elif event_type == "subagent.progress":
                payload.setdefault("message", preview)
        if args is not None:
            payload.setdefault("args", args)
        return payload

    def _persist_subagent_event(self, payload: Dict[str, Any], *, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Append a subagent event to state.db and return the public payload."""
        event = dict(payload)
        prompt_meta = event.pop("_ao_prompt_metadata", None)
        store = self._ensure_subagent_event_store()
        if store is None:
            return event
        try:
            if prompt_meta and event.get("ao_session_id"):
                store.upsert_ao_prompt(
                    ao_session_id=str(event.get("ao_session_id")),
                    project_id=prompt_meta.get("project_id") or event.get("ao_project_id"),
                    prompt=prompt_meta.get("prompt") or "",
                    goal=prompt_meta.get("goal") or event.get("goal"),
                    issue_id=prompt_meta.get("issue_id") or event.get("issue_id"),
                    branch=prompt_meta.get("branch") or event.get("branch"),
                    agent=prompt_meta.get("agent") or event.get("agent"),
                    model=prompt_meta.get("model") or event.get("model"),
                    reasoning_effort=prompt_meta.get("reasoning_effort") or event.get("reasoning_effort"),
                    launch_profile_id=prompt_meta.get("launch_profile_id") or event.get("launch_profile_id"),
                    launch_plan_id=prompt_meta.get("launch_plan_id") or event.get("launch_plan_id"),
                    launch_task_id=prompt_meta.get("launch_task_id") or event.get("launch_task_id"),
                    permissions=prompt_meta.get("permissions") or event.get("permissions"),
                    acceptance_criteria=prompt_meta.get("acceptance_criteria") or event.get("acceptance_criteria"),
                )
            return store.append_event(event, session_id=session_id)
        except Exception as exc:
            logger.debug("Failed to persist subagent event: %s", exc)
            return event

    def _persist_ao_action_event(
        self,
        *,
        action: str,
        source_session_id: str,
        status: str,
        message: str,
        session: Any = None,
        target_session_id: Optional[str] = None,
        base_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist an AO operator action without treating Open as history."""
        event_payload: Dict[str, Any] = dict(base_event or {})
        if session is not None:
            event_payload.update(session.event_fields())
        event_payload.update({
            "event": "subagent.action",
            "subagent_id": event_payload.get("subagent_id") or f"ao:{source_session_id}",
            "runtime": "ao",
            "ao_session_id": source_session_id,
            "action": action,
            "action_status": status,
            "status": status if action == "stop" else event_payload.get("status") or status,
            "source_ao_session_id": source_session_id,
            "target_ao_session_id": target_session_id,
            "message": message,
            "preview": message,
            "timestamp": time.time(),
        })
        event_payload.pop("event_id", None)
        event_payload.pop("created_at", None)
        return self._persist_subagent_event(event_payload)

    @staticmethod
    def _ao_action_response(
        *,
        action: str,
        source_session_id: str,
        status: str,
        message: str,
        session_payload: Optional[Dict[str, Any]] = None,
        event: Optional[Dict[str, Any]] = None,
        target_session_id: Optional[str] = None,
        ok: bool = True,
    ) -> Dict[str, Any]:
        return {
            "ok": ok,
            "mode": action,
            "action": action,
            "source_session_id": source_session_id,
            "source_ao_session_id": source_session_id,
            "target_ao_session_id": target_session_id,
            "status": status,
            "message": message,
            "session": session_payload,
            "event": event,
            "action_event": event,
        }

    @staticmethod
    def _ao_stop_event_is_terminal(event: Optional[Dict[str, Any]]) -> bool:
        if not event:
            return False
        status = str(event.get("status") or event.get("action_status") or "").lower()
        action = str(event.get("action") or "").lower()
        event_type = str(event.get("event") or "").lower()
        if action == "stop" and status in {"killed", "terminated", "cancelled", "canceled", "already_stopped"}:
            return True
        if event_type == "subagent.complete" and status in {"killed", "terminated", "cancelled", "canceled", "failed"}:
            return True
        return False

    def _make_run_event_callback(
        self,
        run_id: str,
        loop: "asyncio.AbstractEventLoop",
        session_id: Optional[str] = None,
    ):
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
            if str(event_type).startswith("subagent."):
                event = self._subagent_event_payload(
                    event_type,
                    run_id,
                    tool_name=tool_name,
                    preview=preview,
                    args=args,
                    extra=kwargs,
                )
                _push(self._persist_subagent_event(event, session_id=session_id))
            elif event_type == "tool.started":
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
            # _thinking and legacy subagent_progress remain internal-only.

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

        event_cb = self._make_run_event_callback(run_id, loop, session_id=session_id)

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
            agent = None
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
                        u = {
                            "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                            "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                            "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                        }
                        return r, u
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
                        self._cleanup_agent_resources(agent)

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
        approval_session_key = self._resolve_active_approval_session_key(run_id)
        if status is None and not approval_session_key:
            return web.json_response(
                _openai_error(f"Run not found: {run_id}", code="run_not_found"),
                status=404,
            )
        if status is None:
            status = {
                "object": "hermes.run",
                "run_id": run_id,
                "status": "waiting_for_approval",
                "updated_at": time.time(),
            }
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
                event_name = event.get("event") if isinstance(event, dict) else None
                if isinstance(event_name, str) and event_name.startswith("subagent."):
                    payload = f"event: {event_name}\ndata: {json.dumps(event)}\n\n"
                else:
                    payload = f"data: {json.dumps(event)}\n\n"
                await response.write(payload.encode())
        except Exception as exc:
            logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
        finally:
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)

        return response

    async def _handle_session_subagent_events(self, request: "web.Request") -> "web.Response":
        """GET /v1/sessions/{session_id}/subagents/events — replay subagent events."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return web.json_response(_openai_error("Session ID required"), status=400)
        limit = self._bounded_query_limit(request, default=500)
        store = self._ensure_subagent_event_store()
        if store is None:
            return web.json_response(_openai_error("Subagent event store unavailable"), status=503)
        return web.json_response(events_response(
            store.list_events(session_id=session_id, limit=limit),
            session_id=session_id,
        ))

    async def _handle_run_subagent_events(self, request: "web.Request") -> "web.Response":
        """GET /v1/runs/{run_id}/subagents/events — replay subagent events for a run."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = str(request.match_info.get("run_id", "")).strip()
        if not run_id:
            return web.json_response(_openai_error("Run ID required"), status=400)
        limit = self._bounded_query_limit(request, default=500)
        store = self._ensure_subagent_event_store()
        if store is None:
            return web.json_response(_openai_error("Subagent event store unavailable"), status=503)
        return web.json_response(events_response(
            store.list_events(run_id=run_id, limit=limit),
            run_id=run_id,
        ))

    async def _handle_subagent_events(self, request: "web.Request") -> "web.Response":
        """GET /v1/subagents/events — replay persisted subagent events with filters."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        limit = self._bounded_query_limit(request, default=500)
        store = self._ensure_subagent_event_store()
        if store is None:
            return web.json_response(_openai_error("Subagent event store unavailable"), status=503)
        params = request.rel_url.query
        return web.json_response(events_response(
            store.list_events(
                session_id=params.get("session_id") or None,
                run_id=params.get("run_id") or None,
                subagent_id=params.get("subagent_id") or None,
                ao_session_id=params.get("ao_session_id") or None,
                runtime=params.get("runtime") or None,
                status=params.get("status") or None,
                limit=limit,
            ),
        ))

    async def _handle_subagent_board(self, request: "web.Request") -> "web.Response":
        """GET /v1/subagents/board — latest subagent/AO worker rows for Oryn."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        params = request.rel_url.query
        limit = self._bounded_query_limit(request, default=250)
        store = self._ensure_subagent_event_store()
        if store is None:
            return web.json_response(_openai_error("Subagent event store unavailable"), status=503)

        fingerprint = self._subagent_board_fingerprint(store, params)

        def _compute_payload() -> Dict[str, Any]:
            rows = build_agent_board_rows(
                store=store,
                params=params,
                limit=limit,
                ao_snapshot_cache=self._ao_snapshot_cache,
            )
            return build_agent_board_response(rows)

        key = "subagents:board:" + "&".join(f"{name}={params.get(name)}" for name in sorted(params.keys())) + f":{limit}"
        cached = await self._read_model_cache.get_or_compute(
            key=key,
            fingerprint=fingerprint,
            compute=_compute_payload,
        )
        return self._cached_read_model_response(
            request,
            cached.payload,
            cached.fingerprint,
            cached=cached,
            model_name="subagents.board",
        )

    def _bounded_query_limit(self, request: "web.Request", *, default: int = 100) -> int:
        try:
            return max(1, min(int(request.rel_url.query.get("limit", str(default))), 2000))
        except Exception:
            return default

    @staticmethod
    def _subagent_runtime(payload: Dict[str, Any]) -> str:
        runtime = str(payload.get("runtime") or "").strip().lower()
        if runtime:
            return runtime
        return "ao" if payload.get("ao_session_id") else "hermes"

    @staticmethod
    def _subagent_lane(status: Optional[str]) -> str:
        raw = str(status or "").strip().lower()
        if raw in {"queued", "pending", "created", "scheduled"}:
            return "queued"
        if raw in {"needs_input", "input_required", "waiting_for_input", "blocked", "paused", "approval_required"}:
            return "needs_input"
        if raw in {"failed", "fail", "error", "errored", "killed", "terminated", "timed_out", "timeout", "cancelled", "canceled"}:
            return "failed"
        if raw in {"completed", "complete", "done", "success", "succeeded", "merged"}:
            return "completed"
        if raw in {"spawned", "running", "thinking", "progress", "working", "active", ""}:
            return "running"
        return "running"

    @staticmethod
    def _subagent_lane_reason(status: Optional[str], lane: str, runtime: str) -> str:
        raw = str(status or "").strip().lower()
        if lane == "queued":
            return "Worker is queued and has not started active work yet."
        if lane == "running":
            return "Worker is active and reporting progress."
        if lane == "needs_input":
            return "Worker is waiting for input or approval."
        if lane == "failed":
            if raw in {"killed", "terminated", "cancelled", "canceled"}:
                return "Worker was stopped before completing."
            if raw in {"timed_out", "timeout"}:
                return "Worker timed out before completing."
            return "Worker ended with a failed status."
        if lane == "completed":
            return "Worker reached a terminal completed state."
        return f"{runtime.title()} worker status is {raw or 'unknown'}."

    @staticmethod
    def _subagent_attention_level(lane: str) -> str:
        if lane == "failed":
            return "high"
        if lane == "needs_input":
            return "medium"
        return "none"

    @staticmethod
    def _subagent_group_fields(payload: Dict[str, Any], runtime: str) -> Dict[str, str]:
        project_id = str(payload.get("ao_project_id") or "").strip()
        if project_id:
            return {
                "group_key": f"project:{project_id}",
                "group_label": project_id,
                "group_kind": "project",
            }
        session_id = str(payload.get("session_id") or "").strip()
        if session_id:
            short_session = session_id[:8]
            return {
                "group_key": f"session:{session_id}",
                "group_label": f"Session {short_session}",
                "group_kind": "session",
            }
        label = "AO" if runtime == "ao" else "Hermes Native"
        return {
            "group_key": f"runtime:{runtime}",
            "group_label": label,
            "group_kind": "runtime",
        }

    @staticmethod
    def _subagent_numeric(payload: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _subagent_token_total(payload: Dict[str, Any]) -> Optional[int]:
        direct = payload.get("token_total") or payload.get("total_tokens")
        if direct is not None:
            try:
                return int(direct)
            except (TypeError, ValueError):
                pass
        total = 0
        seen = False
        for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                total += int(value)
                seen = True
            except (TypeError, ValueError):
                pass
        return total if seen else None

    @staticmethod
    def _subagent_current_activity(payload: Dict[str, Any]) -> Optional[str]:
        for key in ("message", "text", "preview", "activity", "summary", "tool_name", "tool"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _apply_recent_action_fields(item: Dict[str, Any], event: Dict[str, Any]) -> None:
        item["recent_action"] = event.get("action")
        item["recent_action_status"] = event.get("action_status") or event.get("status")
        item["recent_action_message"] = event.get("message") or event.get("preview")
        item["recent_action_at"] = event.get("created_at") or event.get("timestamp")

    @staticmethod
    def _subagent_event_order_value(event: Dict[str, Any]) -> float:
        for key in ("created_at", "timestamp", "event_id"):
            value = event.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    @classmethod
    def _subagent_action_belongs_to_current_lifecycle(
        cls,
        action_event: Dict[str, Any],
        latest_lifecycle_at: Optional[float],
    ) -> bool:
        if latest_lifecycle_at is None:
            return True
        return cls._subagent_event_order_value(action_event) >= latest_lifecycle_at

    @staticmethod
    def _apply_summary_quality_fields(item: Dict[str, Any]) -> None:
        summary = str(item.get("summary") or "").strip()
        status = str(item.get("status") or "").lower()
        goal = str(item.get("goal") or "")
        current = str(item.get("current_activity") or "")
        text = summary or current
        warning: Optional[str] = None

        if status in {"completed", "complete", "done", "success", "succeeded"}:
            if not summary:
                warning = "Worker completed without a final summary."
            elif len(summary) < 24:
                warning = "Worker summary is very short."

        expected_prefixes = sorted(set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}_DONE\b", f"{goal}\n{text}")))
        for prefix in expected_prefixes:
            if prefix not in summary:
                warning = f"Worker summary is missing expected completion marker {prefix}."
                break

        weak_patterns = (
            "did not produce a clear",
            "was still searching",
            "no definitive answer",
            "cannot confirm",
            "verification gap",
            "partial exploration",
            "focused on project activation",
        )
        lower_summary = summary.lower()
        if summary and any(pattern in lower_summary for pattern in weak_patterns):
            warning = "Worker summary looks incomplete or inconclusive."

        tool_log_lines = sum(
            1 for line in summary.splitlines()
            if re.match(r"^\s*(ran|read|searched|grep|rg|sed|cat|terminal|mcp_|activated)\b", line.lower())
        )
        if summary and tool_log_lines >= max(2, len([line for line in summary.splitlines() if line.strip()]) // 2):
            warning = "Worker summary mostly describes tool activity instead of a conclusion."

        item["summary_quality"] = "warning" if warning else "ok"
        item["summary_warning"] = warning

    def _subagent_board_item_from_event(self, event: Dict[str, Any], event_count: int) -> Dict[str, Any]:
        status = event.get("status")
        runtime = self._subagent_runtime(event)
        row_id = str(event.get("subagent_id") or event.get("ao_session_id"))
        created_at = event.get("created_at")
        lane = self._subagent_lane(status)
        group_fields = self._subagent_group_fields(event, runtime)
        has_prompt_meta = False
        return {
            "id": row_id,
            "subagent_id": event.get("subagent_id"),
            "parent_id": event.get("parent_id"),
            "session_id": event.get("session_id"),
            "run_id": event.get("run_id"),
            "runtime": runtime,
            "runtime_session_id": event.get("runtime_session_id") or event.get("ao_session_id"),
            "runtime_project_id": event.get("runtime_project_id") or event.get("ao_project_id"),
            "runtime_selection": event.get("runtime_selection"),
            "selected_runtime": event.get("selected_runtime") or runtime,
            "runtime_selection_reason": event.get("runtime_selection_reason"),
            "runtime_fallback_reason": event.get("runtime_fallback_reason"),
            "status": status,
            "lane": lane,
            "lane_reason": self._subagent_lane_reason(status, lane, runtime),
            "attention_level": self._subagent_attention_level(lane),
            "goal": event.get("goal"),
            "summary": event.get("summary"),
            "current_activity": self._subagent_current_activity(event),
            "created_at": created_at,
            "updated_at": created_at,
            "last_activity_at": created_at,
            "event_count": event_count,
            "ao_session_id": event.get("ao_session_id"),
            "ao_project_id": event.get("ao_project_id"),
            "workspace_path": event.get("workspace_path"),
            "branch": event.get("branch"),
            "issue_id": event.get("issue_id"),
            "tmux_name": event.get("tmux_name"),
            "open_url": event.get("open_url"),
            "open_command": event.get("open_command"),
            "agent": event.get("agent"),
            "model": event.get("model"),
            "reasoning_effort": event.get("reasoning_effort"),
            "launch_profile_id": event.get("launch_profile_id"),
            "launch_plan_id": event.get("launch_plan_id"),
            "launch_task_id": event.get("launch_task_id"),
            "permissions": event.get("permissions"),
            "acceptance_criteria": event.get("acceptance_criteria") or [],
            "duration_seconds": self._subagent_numeric(event, "duration_seconds"),
            "token_total": self._subagent_token_total(event),
            "cost_usd": self._subagent_numeric(event, "cost_usd"),
            "files_read": event.get("files_read") or [],
            "files_written": event.get("files_written") or [],
            "output_tail": event.get("output_tail") or [],
            "recent_action": event.get("action") if event.get("event") == "subagent.action" else None,
            "recent_action_status": event.get("action_status") if event.get("event") == "subagent.action" else None,
            "recent_action_message": event.get("message") if event.get("event") == "subagent.action" else None,
            "recent_action_at": event.get("created_at") or event.get("timestamp") if event.get("event") == "subagent.action" else None,
            "summary_quality": "ok",
            "summary_warning": None,
            "runtime_health": None,
            "runtime_warning": None,
            "diagnostic_status": None,
            "diagnostic_message": None,
            "recovery_recommendation": None,
            "transcript_available": False,
            "transcript_tail": None,
            "transcript_captured_at": None,
            "has_prompt_metadata": has_prompt_meta,
            "action_unavailable_reason": "AO controls are only available for AO-backed workers." if runtime != "ao" else None,
            **group_fields,
            "can_open": runtime == "ao" and bool(event.get("ao_session_id")),
            "can_stop": False,
            "can_follow_up": False,
            "can_retry": False,
            "can_reassign": False,
        }

    def _merge_ao_session_into_board_item(
        self,
        item: Dict[str, Any],
        session: Any,
        store: SubagentEventStore,
        *,
        runtime_health: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row = dict(item)
        row_id = f"ao:{session.id}"
        status = row.get("status") or session.display_status
        runtime_health = runtime_health or {"runtime_health": "ok", "runtime_warning": None}
        is_stale = runtime_health.get("runtime_health") == "stale"
        if is_stale:
            status = "terminated"
        lane = self._subagent_lane(status)
        if lane == "running":
            status = session.display_status
            lane = self._subagent_lane(status)
            if is_stale:
                status = "terminated"
                lane = "failed"
        prompt_meta = store.get_ao_prompt(session.id)
        group_fields = self._subagent_group_fields({
            **row,
            "ao_project_id": session.project_id or row.get("ao_project_id"),
        }, "ao")
        action_unavailable_reason = None
        if lane == "running":
            action_unavailable_reason = "Retry and reassign are available after the worker reaches a terminal state."
        elif not prompt_meta:
            action_unavailable_reason = "Original AO prompt metadata is unavailable for retry or reassign."
        prompt_agent = (prompt_meta or {}).get("agent")
        prompt_model = (prompt_meta or {}).get("model")
        prompt_reasoning_effort = (prompt_meta or {}).get("reasoning_effort")
        prompt_launch_profile_id = (prompt_meta or {}).get("launch_profile_id")
        prompt_launch_plan_id = (prompt_meta or {}).get("launch_plan_id")
        prompt_launch_task_id = (prompt_meta or {}).get("launch_task_id")
        row.update({
            "id": row.get("id") or row_id,
            "subagent_id": row.get("subagent_id") or row_id,
            "runtime": "ao",
            "runtime_session_id": session.id,
            "runtime_project_id": session.project_id or row.get("runtime_project_id") or row.get("ao_project_id"),
            "status": status,
            "lane": lane,
            "lane_reason": runtime_health.get("runtime_warning") or self._subagent_lane_reason(status, lane, "ao"),
            "attention_level": self._subagent_attention_level(lane),
            "goal": row.get("goal") or (prompt_meta or {}).get("goal") or session.activity or f"AO session {session.id}",
            "summary": row.get("summary") or session.summary,
            "current_activity": session.activity or row.get("current_activity"),
            "updated_at": row.get("updated_at") or time.time(),
            "last_activity_at": row.get("last_activity_at") or row.get("updated_at") or time.time(),
            "event_count": int(row.get("event_count") or 0),
            "ao_session_id": session.id,
            "ao_project_id": session.project_id or row.get("ao_project_id"),
            "workspace_path": session.workspace_path or row.get("workspace_path"),
            "branch": session.branch or row.get("branch"),
            "issue_id": session.issue_id or row.get("issue_id"),
            "tmux_name": session.tmux_name or row.get("tmux_name"),
            "open_command": session.open_command or row.get("open_command"),
            "agent": row.get("agent") or prompt_agent or session.agent,
            "model": row.get("model") or prompt_model or session.model,
            "reasoning_effort": row.get("reasoning_effort") or prompt_reasoning_effort or session.reasoning_effort,
            "launch_profile_id": row.get("launch_profile_id") or prompt_launch_profile_id,
            "launch_plan_id": row.get("launch_plan_id") or prompt_launch_plan_id,
            "launch_task_id": row.get("launch_task_id") or prompt_launch_task_id,
            "permissions": row.get("permissions") or (prompt_meta or {}).get("permissions"),
            "acceptance_criteria": row.get("acceptance_criteria") or (prompt_meta or {}).get("acceptance_criteria") or [],
            "duration_seconds": row.get("duration_seconds"),
            "token_total": row.get("token_total"),
            "cost_usd": row.get("cost_usd"),
            "files_read": row.get("files_read") or [],
            "files_written": row.get("files_written") or [],
            "output_tail": row.get("output_tail") or [],
            "recent_action": row.get("recent_action"),
            "recent_action_status": row.get("recent_action_status"),
            "recent_action_message": row.get("recent_action_message"),
            "recent_action_at": row.get("recent_action_at"),
            "summary_quality": row.get("summary_quality") or "ok",
            "summary_warning": row.get("summary_warning") or runtime_health.get("runtime_warning"),
            "runtime_health": runtime_health.get("runtime_health"),
            "runtime_warning": runtime_health.get("runtime_warning"),
            "diagnostic_status": "stale" if is_stale else lane,
            "diagnostic_message": runtime_health.get("runtime_warning") or self._subagent_lane_reason(status, lane, "ao"),
            "recovery_recommendation": self._ao_recovery_recommendation(status=status, lane=lane, is_stale=is_stale),
            "transcript_available": bool(session.tmux_name) and not is_stale,
            "transcript_tail": row.get("transcript_tail"),
            "transcript_captured_at": row.get("transcript_captured_at"),
            "has_prompt_metadata": bool(prompt_meta),
            "action_unavailable_reason": action_unavailable_reason,
            **group_fields,
            "can_open": True,
            "can_stop": (not is_stale) and lane in {"queued", "running", "needs_input"},
            "can_follow_up": (not is_stale) and lane == "running",
            "can_retry": lane in {"failed", "completed"} and bool(prompt_meta),
            "can_reassign": lane in {"failed", "completed"} and bool(prompt_meta),
        })
        return row

    @staticmethod
    def _ao_recovery_recommendation(*, status: Optional[str], lane: str, is_stale: bool = False) -> str:
        if is_stale:
            return "Runtime is gone. Use Repair Retry to spawn a replacement from the original task context, or Open to inspect the worktree."
        if lane == "running":
            return "Worker is running. Use Resume or Follow-up to steer it, or Stop if it is no longer useful."
        if lane == "needs_input":
            return "Worker needs input. Send a follow-up or open the worker terminal/worktree for more detail."
        if lane == "failed":
            raw = str(status or "").lower()
            if raw in {"killed", "terminated", "cancelled", "canceled"}:
                return "Worker was stopped or terminated. Use Repair Retry to spawn a replacement with the latest diagnostics."
            return "Worker failed. Use transcript tail and action history to diagnose, then Repair Retry if the original prompt is available."
        if lane == "completed":
            return "Worker completed. Review summary and transcript tail; Retry or Reassign if the result needs another pass."
        return "Review diagnostics and choose a recovery action if needed."

    @staticmethod
    def _bounded_transcript_lines(request: "web.Request", *, default: int = 120) -> int:
        try:
            return max(20, min(int(request.rel_url.query.get("lines", str(default))), 240))
        except Exception:
            return default

    def _subagent_board_item_matches(self, item: Dict[str, Any], params: Any) -> bool:
        runtime = params.get("runtime")
        if runtime:
            wanted = runtime.lower()
            actual = str(item.get("runtime") or "").lower()
            if wanted == "native":
                wanted = "hermes"
            if actual != wanted:
                return False
        status = params.get("status")
        if status:
            wanted_status = status.lower()
            if wanted_status == "needs_attention":
                wanted_status = "needs_input"
            actual_status = str(item.get("status") or "").lower()
            actual_lane = str(item.get("lane") or "").lower()
            if wanted_status not in {actual_status, actual_lane}:
                return False
        lane = params.get("lane")
        if lane and str(item.get("lane") or "").lower() != lane.lower():
            return False
        include_completed = str(params.get("include_completed") or "").lower()
        if include_completed in {"0", "false", "no"} and item.get("lane") == "completed":
            return False
        group_kind = params.get("group_kind")
        if group_kind and item.get("group_kind") != group_kind:
            return False
        group_key = params.get("group_key")
        if group_key and item.get("group_key") != group_key:
            return False
        updated_after = params.get("updated_after")
        if updated_after:
            try:
                if float(item.get("updated_at") or 0) <= float(updated_after):
                    return False
            except (TypeError, ValueError):
                return False
        project_id = params.get("project_id")
        if project_id and project_id not in {item.get("ao_project_id"), item.get("runtime_project_id")}:
            return False
        session_id = params.get("session_id")
        if session_id and item.get("session_id") != session_id:
            return False
        ao_session_id = params.get("ao_session_id")
        if ao_session_id and item.get("ao_session_id") != ao_session_id:
            return False
        needle = str(params.get("q") or params.get("search") or "").strip().lower()
        if needle:
            haystack = " ".join(str(item.get(key) or "") for key in (
                "goal", "summary", "current_activity", "branch", "ao_project_id", "ao_session_id"
            )).lower()
            if needle not in haystack:
                return False
        return True

    def _ao_session_payload(self, session: Any, *, include_events: bool = False) -> Dict[str, Any]:
        def _scalar(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return None

        event_fields = {}
        raw_event_fields = getattr(session, "event_fields", None)
        if callable(raw_event_fields):
            try:
                candidate = raw_event_fields()
            except Exception:
                candidate = {}
            if isinstance(candidate, dict):
                event_fields = candidate

        session_id = _scalar(getattr(session, "id", None))
        if not session_id:
            session_id = _scalar(getattr(session, "ao_session_id", None))

        payload = dict(event_fields)
        payload.update({
            "id": session_id,
            "status": _scalar(getattr(session, "display_status", None)),
            "activity": _scalar(getattr(session, "activity", None)),
            "agent": _scalar(getattr(session, "agent", None)),
            "pr": _scalar(getattr(session, "pr", None)),
            "summary": _scalar(getattr(session, "summary", None)),
        })
        store = self._ensure_subagent_event_store()
        prompt_meta = store.get_ao_prompt(session_id) if store and isinstance(session_id, str) else None
        status_value = _scalar(getattr(session, "display_status", None))
        if prompt_meta:
            payload["agent"] = prompt_meta.get("agent") or payload.get("agent")
            payload["model"] = prompt_meta.get("model") or payload.get("model")
            payload["reasoning_effort"] = prompt_meta.get("reasoning_effort") or payload.get("reasoning_effort")
            payload["runtime_selection"] = prompt_meta.get("runtime_selection") or payload.get("runtime_selection")
            payload["selected_runtime"] = prompt_meta.get("selected_runtime") or payload.get("selected_runtime") or payload.get("runtime")
            payload["runtime_selection_reason"] = prompt_meta.get("runtime_selection_reason") or payload.get("runtime_selection_reason")
            payload["runtime_fallback_reason"] = prompt_meta.get("runtime_fallback_reason") or payload.get("runtime_fallback_reason")
            payload["launch_profile_id"] = prompt_meta.get("launch_profile_id")
            payload["launch_plan_id"] = prompt_meta.get("launch_plan_id")
            payload["launch_task_id"] = prompt_meta.get("launch_task_id")
            payload["permissions"] = prompt_meta.get("permissions")
            payload["acceptance_criteria"] = prompt_meta.get("acceptance_criteria") or []
        lane = self._subagent_lane(status_value)
        payload["lane"] = lane
        payload["lane_reason"] = self._subagent_lane_reason(status_value, lane, "ao")
        payload["attention_level"] = self._subagent_attention_level(lane)
        payload["can_retry"] = bool(prompt_meta)
        payload["can_reassign"] = bool(prompt_meta)
        payload["has_prompt_metadata"] = bool(prompt_meta)
        payload["action_unavailable_reason"] = None if prompt_meta else "Original AO prompt metadata is unavailable for retry or reassign."
        if prompt_meta:
            payload["goal"] = prompt_meta.get("goal")
            payload["prompt_available"] = True
        else:
            payload["prompt_available"] = False
        if include_events and store and isinstance(session_id, str):
            payload["events"] = store.list_events(ao_session_id=session_id)
        return payload

    async def _handle_ao_sessions(self, request: "web.Request") -> "web.Response":
        """GET /v1/ao/sessions — list AO sessions known to AO/Hermes."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        project_id = request.rel_url.query.get("project_id")
        try:
            from tools.ao_bridge import AOBridge

            sessions = AOBridge().list(project_id=project_id)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)
        return web.json_response({
            "object": "list",
            "data": [self._ao_session_payload(session) for session in sessions],
            "total": len(sessions),
        })

    async def _handle_ao_session_detail(self, request: "web.Request") -> "web.Response":
        """GET /v1/ao/sessions/{session_id} — AO session detail with Hermes event history."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        try:
            from tools.ao_bridge import AOBridge

            session = AOBridge().status(session_id)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        if not session:
            return web.json_response(_openai_error("AO session not found"), status=404)
        return web.json_response({
            "object": "hermes.ao_session",
            "session": self._ao_session_payload(session, include_events=True),
        })

    def _ao_diagnostics_payload(self, *, session: Any, bridge: Any, lines: int) -> Dict[str, Any]:
        store = self._ensure_subagent_event_store()
        latest = store.latest_event_for_ao_session(session.id) if store else None
        events = store.list_events(ao_session_id=session.id, limit=500) if store else []
        latest_lifecycle_at = None
        for event in events:
            if event.get("event") != "subagent.action":
                latest_lifecycle_at = self._subagent_event_order_value(event)
        last_action = next(
            (
                event for event in reversed(events)
                if event.get("event") == "subagent.action"
                and self._subagent_action_belongs_to_current_lifecycle(event, latest_lifecycle_at)
            ),
            None,
        )
        runtime_health = bridge.runtime_health(session)
        is_stale = runtime_health.get("runtime_health") == "stale"
        status = "terminated" if is_stale else session.display_status
        lane = "failed" if is_stale else self._subagent_lane(status)
        transcript_tail = bridge.capture_output(session, lines=lines) if session.tmux_name else ""
        diagnostic_message = (
            runtime_health.get("runtime_warning")
            or (latest or {}).get("summary")
            or (latest or {}).get("message")
            or session.activity
            or self._subagent_lane_reason(status, lane, "ao")
        )
        return {
            "object": "hermes.ao_session_diagnostics",
            "session_id": session.id,
            "session": self._ao_session_payload(session, include_events=False),
            "runtime_health": runtime_health.get("runtime_health"),
            "runtime_warning": runtime_health.get("runtime_warning"),
            "tmux_alive": runtime_health.get("tmux_alive"),
            "process_alive": runtime_health.get("process_alive"),
            "diagnostic_status": "stale" if is_stale else lane,
            "diagnostic_message": diagnostic_message,
            "recovery_recommendation": self._ao_recovery_recommendation(
                status=status,
                lane=lane,
                is_stale=is_stale,
            ),
            "last_action": last_action,
            "last_event": latest,
            "transcript_available": bool(transcript_tail),
            "transcript_tail": transcript_tail,
            "transcript_lines": lines,
            "transcript_captured_at": time.time(),
            "can_resume": (not is_stale) and lane == "running",
            "can_repair_retry": bool(store.get_ao_prompt(session.id) if store else None),
            "can_stop": (not is_stale) and lane in {"queued", "running", "needs_input"},
            "can_open": True,
        }

    async def _handle_ao_session_diagnostics(self, request: "web.Request") -> "web.Response":
        """GET /v1/ao/sessions/{session_id}/diagnostics — recovery-oriented AO detail."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        lines = self._bounded_transcript_lines(request)
        try:
            from tools.ao_bridge import AOBridge

            bridge = AOBridge()
            session = bridge.status(session_id)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        if not session:
            return web.json_response(_openai_error("AO session not found"), status=404)
        return web.json_response(self._ao_diagnostics_payload(session=session, bridge=bridge, lines=lines))

    async def _handle_ao_session_follow_up(self, request: "web.Request") -> "web.Response":
        """POST /v1/ao/sessions/{session_id}/follow-up — send a message to a running AO worker."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        try:
            body = await request.json()
        except Exception:
            body = {}
        message = str(body.get("message") or "").strip()
        if not message:
            return web.json_response(_openai_error("Follow-up message required"), status=400)
        try:
            from tools.ao_bridge import AOBridge

            bridge = AOBridge()
            session = bridge.send(session_id, message) or bridge.status(session_id)
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=404)
        progress_event = None
        action_event = None
        if session:
            self._persist_subagent_event({
                "event": "subagent.progress",
                "subagent_id": f"ao:{session.id}",
                "runtime": "ao",
                "ao_session_id": session.id,
                "ao_project_id": session.project_id,
                "workspace_path": session.workspace_path,
                "branch": session.branch,
                "tmux_name": session.tmux_name,
                "open_command": session.open_command,
                "status": session.display_status,
                "message": "Follow-up sent",
                "preview": "Follow-up sent",
                "timestamp": time.time(),
            })
            progress_event = self._ensure_subagent_event_store().latest_event_for_ao_session(session.id) if self._ensure_subagent_event_store() else None
        action_event = self._persist_ao_action_event(
            action="follow-up",
            source_session_id=session_id,
            status="succeeded",
            message="Follow-up sent",
            session=session,
            base_event=progress_event,
        )
        self._invalidate_ao_read_models()
        return web.json_response(self._ao_action_response(
            action="follow-up",
            source_session_id=session_id,
            status=session.display_status if session else "unknown",
            message="Follow-up sent",
            session_payload=self._ao_session_payload(session) if session else None,
            event=action_event,
        ))

    async def _handle_ao_session_resume(self, request: "web.Request") -> "web.Response":
        """POST /v1/ao/sessions/{session_id}/resume — steer a running AO worker from diagnostics."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        try:
            body = await request.json()
        except Exception:
            body = {}
        message = str(body.get("message") or "").strip() or (
            "Resume from your latest state. Report current blockers, continue the original task, "
            "and produce a concise status update."
        )
        try:
            from tools.ao_bridge import AOBridge

            bridge = AOBridge()
            before = bridge.status(session_id)
            if not before:
                return web.json_response(_openai_error("AO session not found"), status=404)
            runtime_health = bridge.runtime_health(before)
            if runtime_health.get("runtime_health") == "stale" or self._subagent_lane(before.display_status) != "running":
                return web.json_response(
                    _openai_error("AO session is not running; use Repair Retry instead."),
                    status=409,
                )
            session = bridge.send(session_id, message) or bridge.status(session_id) or before
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=404)

        progress_event = self._persist_subagent_event({
            "event": "subagent.progress",
            "subagent_id": f"ao:{session.id}",
            "runtime": "ao",
            "ao_session_id": session.id,
            "ao_project_id": session.project_id,
            "workspace_path": session.workspace_path,
            "branch": session.branch,
            "tmux_name": session.tmux_name,
            "open_command": session.open_command,
            "status": session.display_status,
            "message": "Resume sent",
            "preview": "Resume sent",
            "timestamp": time.time(),
        })
        action_event = self._persist_ao_action_event(
            action="resume",
            source_session_id=session_id,
            status="succeeded",
            message="Resume sent",
            session=session,
            base_event=progress_event,
        )
        self._invalidate_ao_read_models()
        return web.json_response(self._ao_action_response(
            action="resume",
            source_session_id=session_id,
            status=session.display_status,
            message="Resume sent",
            session_payload=self._ao_session_payload(session),
            event=action_event,
        ))

    async def _handle_ao_session_retry(self, request: "web.Request") -> "web.Response":
        """POST /v1/ao/sessions/{session_id}/retry — spawn a replacement AO worker."""
        return await self._spawn_related_ao_session(request, mode="retry")

    async def _handle_ao_session_repair_retry(self, request: "web.Request") -> "web.Response":
        """POST /v1/ao/sessions/{session_id}/repair-retry — spawn a diagnostics-aware replacement."""
        return await self._spawn_related_ao_session(request, mode="repair-retry")

    async def _handle_ao_session_reassign(self, request: "web.Request") -> "web.Response":
        """POST /v1/ao/sessions/{session_id}/reassign — spawn a replacement AO worker with optional overrides."""
        return await self._spawn_related_ao_session(request, mode="reassign")

    async def _spawn_related_ao_session(self, request: "web.Request", *, mode: str) -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        source_session_id = request.match_info.get("session_id", "")
        try:
            body = await request.json()
        except Exception:
            body = {}
        store = self._ensure_subagent_event_store()
        prompt_meta = store.get_ao_prompt(source_session_id) if store else None
        if not prompt_meta:
            return web.json_response(
                _openai_error("Original AO prompt metadata is unavailable for this session"),
                status=409,
            )
        latest = store.latest_event_for_ao_session(source_session_id) if store else None
        prior_summary = (latest or {}).get("summary") or (latest or {}).get("message") or (latest or {}).get("preview")
        instruction = str(body.get("instruction") or "").strip()
        project_id = resolve_project_id(body.get("project_id"), prompt_meta.get("project_id"))
        agent = body.get("agent") or prompt_meta.get("agent")
        model = body.get("model") or prompt_meta.get("model")
        reasoning_effort = body.get("reasoning_effort") or prompt_meta.get("reasoning_effort")
        goal = prompt_meta.get("goal") or f"{mode.title()} AO worker"
        diagnostic_context = ""
        if mode == "repair-retry":
            try:
                from tools.ao_bridge import AOBridge as _DiagnosticsAOBridge

                diagnostics_bridge = _DiagnosticsAOBridge()
                source_session = diagnostics_bridge.status(source_session_id)
                if source_session:
                    health = diagnostics_bridge.runtime_health(source_session)
                    tail = diagnostics_bridge.capture_output(source_session, lines=120)
                    diagnostic_context = "\n\nRecovery diagnostics:\n"
                    diagnostic_context += f"Runtime health: {health.get('runtime_health') or 'unknown'}\n"
                    if health.get("runtime_warning"):
                        diagnostic_context += f"Runtime warning: {health.get('runtime_warning')}\n"
                    if tail:
                        diagnostic_context += "Recent transcript tail:\n" + tail[-8000:]
            except Exception as exc:
                logger.debug("Failed to collect AO repair-retry diagnostics for %s: %s", source_session_id, exc)
        prompt = self._related_ao_prompt(
            mode=mode,
            original_prompt=prompt_meta.get("prompt") or "",
            prior_summary=prior_summary,
            instruction=(instruction + diagnostic_context).strip(),
            model=model,
        )
        try:
            from tools.ao_bridge import AOBridge
            from tools.ao_delegate_tool import build_ao_worker_prompt

            bridge = AOBridge()
            launch_prompt = build_ao_worker_prompt(prompt, goal=f"{mode.title()}: {goal}")
            session = bridge.spawn(
                project_id=project_id,
                prompt=launch_prompt,
                issue_id=prompt_meta.get("issue_id"),
                agent=agent,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            return web.json_response(_openai_error(str(exc)), status=500)

        if store:
            store.upsert_ao_prompt(
                ao_session_id=session.id,
                project_id=project_id,
                prompt=prompt,
                goal=f"{mode.title()}: {goal}",
                issue_id=prompt_meta.get("issue_id"),
                branch=session.branch,
                agent=session.agent or agent,
                model=session.model or model,
                reasoning_effort=session.reasoning_effort or reasoning_effort,
                launch_profile_id=prompt_meta.get("launch_profile_id"),
                launch_plan_id=prompt_meta.get("launch_plan_id"),
                launch_task_id=prompt_meta.get("launch_task_id"),
                permissions=prompt_meta.get("permissions"),
                acceptance_criteria=prompt_meta.get("acceptance_criteria") or [],
            )
        start_event = self._persist_subagent_event({
            "event": "subagent.start",
            "subagent_id": f"ao:{session.id}",
            "parent_id": (latest or {}).get("subagent_id"),
            "depth": int((latest or {}).get("depth") or 0),
            "goal": f"{mode.title()}: {goal}",
            "runtime": "ao",
            "ao_session_id": session.id,
            "ao_project_id": session.project_id,
            "workspace_path": session.workspace_path,
            "branch": session.branch,
            "issue_id": session.issue_id,
            "tmux_name": session.tmux_name,
            "open_command": session.open_command,
            "agent": session.agent,
            "model": session.model,
            "reasoning_effort": session.reasoning_effort,
            "launch_profile_id": prompt_meta.get("launch_profile_id"),
            "launch_plan_id": prompt_meta.get("launch_plan_id"),
            "launch_task_id": prompt_meta.get("launch_task_id"),
            "permissions": prompt_meta.get("permissions"),
            "acceptance_criteria": prompt_meta.get("acceptance_criteria") or [],
            "status": session.display_status,
            "message": f"AO {mode} session spawned from {source_session_id}",
            "timestamp": time.time(),
        })
        action_event = self._persist_ao_action_event(
            action=mode,
            source_session_id=source_session_id,
            target_session_id=session.id,
            status="succeeded",
            message=f"AO {mode} session spawned",
            session=session,
            base_event=latest,
        )
        self._invalidate_ao_read_models()
        return web.json_response(self._ao_action_response(
            action=mode,
            source_session_id=source_session_id,
            target_session_id=session.id,
            status=session.display_status,
            message=f"AO {mode} session spawned",
            session_payload=self._ao_session_payload(session),
            event=action_event,
        ))

    @staticmethod
    def _related_ao_prompt(
        *,
        mode: str,
        original_prompt: str,
        prior_summary: Optional[str],
        instruction: str,
        model: Optional[str],
    ) -> str:
        parts = [
            f"You are a {mode} AO worker for a previous Agent Orchestrator session.",
            "",
            "Original task:",
            original_prompt.strip(),
        ]
        if prior_summary:
            parts.extend(["", "Previous session summary/status:", prior_summary.strip()])
        if model:
            parts.extend(["", f"Requested model/agent preference: {model}"])
        if instruction:
            parts.extend(["", "Additional instruction:", instruction])
        return "\n".join(part for part in parts if part is not None)

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

        approval_session_key = self._resolve_active_approval_session_key(run_id)
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

        if status is not None:
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

    async def _handle_steer_run(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/steer — inject guidance into a live run."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            body = {}

        text = str(body.get("text") or "").strip()
        if not text:
            return web.json_response(_openai_error("'text' is required"), status=400)

        agent = self._active_run_agents.get(run_id)
        if agent is None:
            return web.json_response(
                _openai_error(
                    f"Run not found or not steerable: {run_id}",
                    code="run_not_found",
                ),
                status=404,
            )

        if not hasattr(agent, "steer"):
            return web.json_response(
                _openai_error(
                    "Active run does not support steer",
                    code="steer_unsupported",
                ),
                status=409,
            )

        try:
            accepted = bool(agent.steer(text))
        except Exception as exc:  # noqa: BLE001
            logger.debug("steer failed for run %s: %s", run_id, exc)
            return web.json_response(
                _openai_error(f"Steer failed: {exc}", code="steer_failed"),
                status=500,
            )

        preview = text[:80] + ("..." if len(text) > 80 else "")
        return web.json_response({
            "object": "hermes.run.steer_result",
            "run_id": run_id,
            "status": "queued" if accepted else "rejected",
            "text": text,
            "message": (
                f"Steer queued — arrives after the next tool call: {preview}"
                if accepted
                else "Steer rejected"
            ),
        })

    async def _handle_ao_session_stop(self, request: "web.Request") -> "web.Response":
        """POST /v1/ao/sessions/{session_id}/stop — stop an AO worker session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        store = self._ensure_subagent_event_store()
        latest = store.latest_event_for_ao_session(session_id) if store else None
        if self._ao_stop_event_is_terminal(latest):
            response = self._ao_action_response(
                action="stop",
                source_session_id=session_id,
                status="already_stopped",
                message="AO worker is already stopped.",
                session_payload=None,
                event=latest,
            )
            response["session_id"] = session_id
            return web.json_response(response)

        try:
            from tools.ao_bridge import AOBridge

            bridge = AOBridge()
            before_session = bridge.status(session_id)
            bridge.kill(session_id, session=before_session)
            try:
                session = bridge.status(session_id) or before_session
            except Exception:
                session = before_session
        except Exception as exc:
            latest = store.latest_event_for_ao_session(session_id) if store else None
            if self._ao_stop_event_is_terminal(latest):
                response = self._ao_action_response(
                    action="stop",
                    source_session_id=session_id,
                    status="already_stopped",
                    message="AO worker is already stopped.",
                    session_payload=None,
                    event=latest,
                )
                response["session_id"] = session_id
                return web.json_response(response)
            return web.json_response({"error": str(exc)}, status=404)
        if session is not None and not isinstance(getattr(session, "id", None), str):
            session = None

        if store:
            latest = store.latest_event_for_ao_session(session_id)
        event_payload: Dict[str, Any] = dict(latest or {})
        if session is not None:
            event_payload.update(session.event_fields())
        event_payload.update({
            "event": "subagent.complete",
            "subagent_id": event_payload.get("subagent_id") or f"ao:{session_id}",
            "runtime": "ao",
            "ao_session_id": session_id,
            "status": "killed",
            "summary": "AO worker stopped by user.",
            "message": "AO worker stopped by user.",
            "preview": "AO worker stopped by user.",
            "timestamp": time.time(),
        })
        event_payload.pop("event_id", None)
        event_payload.pop("created_at", None)
        event = self._persist_subagent_event(event_payload)
        action_event = self._persist_ao_action_event(
            action="stop",
            source_session_id=session_id,
            status="killed",
            message="AO worker stopped by user.",
            session=session,
            base_event=event,
        )
        response = self._ao_action_response(
            action="stop",
            source_session_id=session_id,
            status="killed",
            message="AO worker stopped by user.",
            session_payload=self._ao_session_payload(session) if session else None,
            event=action_event,
        )
        response["session_id"] = session_id
        self._invalidate_ao_read_models()
        return web.json_response(response)

    async def _handle_ao_session_open(self, request: "web.Request") -> "web.Response":
        """POST /v1/ao/sessions/{session_id}/open — open AO worktree and return attach info."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        try:
            from tools.ao_bridge import AOBridge

            result = AOBridge().open_session(session_id)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=404)
        return web.json_response(result)

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
            assert self._app is not None
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/health/detailed", self._handle_health_detailed)
            self._app.router.add_get("/v1/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_get("/v1/model/current", self._handle_current_model)
            self._app.router.add_get(
                "/v1/providers/openrouter/models",
                self._handle_openrouter_models,
            )
            self._app.router.add_get(
                "/v1/providers/codex/models",
                self._handle_codex_models,
            )
            self._app.router.add_get("/v1/sessions/{session_id}/model", self._handle_get_session_model)
            self._app.router.add_post("/v1/sessions/{session_id}/model", self._handle_set_session_model)
            self._app.router.add_get("/v1/sessions/{session_id}/goal", self._handle_get_session_goal)
            self._app.router.add_post("/v1/sessions/{session_id}/goal", self._handle_post_session_goal)
            self._app.router.add_get("/v1/sessions", self._handle_v1_list_sessions)
            self._app.router.add_get("/v1/sessions/{session_id}", self._handle_v1_get_session)
            self._app.router.add_get("/v1/sessions/{session_id}/messages", self._handle_v1_get_session_messages)
            self._app.router.add_get("/v1/sessions/{session_id}/subagents/events", self._handle_session_subagent_events)
            self._app.router.add_delete("/v1/sessions/{session_id}", self._handle_v1_delete_session)
            self._app.router.add_get("/v1/capabilities", self._handle_capabilities)
            self._app.router.add_post("/v1/capabilities", self._handle_capabilities)
            self._app.router.add_get("/v1/commands", self._handle_commands)
            self._app.router.add_get("/v1/skills", self._handle_skills)
            self._app.router.add_get("/v1/toolsets", self._handle_toolsets)
            self._app.router.add_post("/v1/complete/slash", self._handle_complete_slash)
            self._app.router.add_post("/v1/complete/skills", self._handle_complete_skills)
            self._app.router.add_post("/v1/slash", self._handle_slash)
            # Session/client control surface (thin wrappers over SessionDB + _run_agent)
            self._app.router.add_get("/api/sessions", self._handle_list_sessions)
            self._app.router.add_post("/api/sessions", self._handle_create_session)
            self._app.router.add_get("/api/sessions/{session_id}", self._handle_get_session)
            self._app.router.add_patch("/api/sessions/{session_id}", self._handle_patch_session)
            self._app.router.add_delete("/api/sessions/{session_id}", self._handle_delete_session)
            self._app.router.add_get("/api/sessions/{session_id}/messages", self._handle_session_messages)
            self._app.router.add_post("/api/sessions/{session_id}/fork", self._handle_fork_session)
            self._app.router.add_post("/api/sessions/{session_id}/chat", self._handle_session_chat)
            self._app.router.add_post("/api/sessions/{session_id}/chat/stream", self._handle_session_chat_stream)
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
            self._app.router.add_get("/v1/runs/{run_id}/subagents/events", self._handle_run_subagent_events)
            self._app.router.add_get("/v1/subagents/board", self._handle_subagent_board)
            self._app.router.add_get("/v1/subagents/events", self._handle_subagent_events)
            register_dev_control_routes(self._app, self)
            self._app.router.add_post("/v1/runs/{run_id}/approval", self._handle_run_approval)
            self._app.router.add_post("/v1/runs/{run_id}/stop", self._handle_stop_run)
            self._app.router.add_post("/v1/runs/{run_id}/steer", self._handle_steer_run)
            self._app.router.add_post(
                "/v1/background/tasks/{task_id}/follow-up",
                self._handle_background_task_follow_up,
            )
            self._app.router.add_get("/v1/ao/sessions", self._handle_ao_sessions)
            self._app.router.add_get("/v1/ao/sessions/{session_id}", self._handle_ao_session_detail)
            # Store the adapter after native routes are registered. Local Hermes-Relay
            # bootstrap shims use this key as a feature-detection hook; registering
            # native routes first lets those shims no-op instead of shadowing the
            # upstream session-control handlers.
            self._app["api_server_adapter"] = self
            self._app.router.add_post("/v1/ao/sessions/{session_id}/stop", self._handle_ao_session_stop)
            self._app.router.add_post("/v1/ao/sessions/{session_id}/open", self._handle_ao_session_open)
            self._app.router.add_get("/v1/ao/sessions/{session_id}/diagnostics", self._handle_ao_session_diagnostics)
            self._app.router.add_post("/v1/ao/sessions/{session_id}/follow-up", self._handle_ao_session_follow_up)
            self._app.router.add_post("/v1/ao/sessions/{session_id}/resume", self._handle_ao_session_resume)
            self._app.router.add_post("/v1/ao/sessions/{session_id}/retry", self._handle_ao_session_retry)
            self._app.router.add_post("/v1/ao/sessions/{session_id}/repair-retry", self._handle_ao_session_repair_retry)
            self._app.router.add_post("/v1/ao/sessions/{session_id}/reassign", self._handle_ao_session_reassign)
            register_kanban_routes(self._app, self)
            # Start background sweep to clean up orphaned (unconsumed) run streams
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            self._dev_supervisor_loop_task = asyncio.create_task(self._run_dev_supervisor_loop())
            try:
                self._background_tasks.add(self._dev_supervisor_loop_task)
            except TypeError:
                pass
            if hasattr(self._dev_supervisor_loop_task, "add_done_callback"):
                self._dev_supervisor_loop_task.add_done_callback(self._background_tasks.discard)

            # Refuse to start without authentication. The API server can
            # dispatch terminal-capable agent work, so every deployment needs
            # an explicit API_SERVER_KEY regardless of bind address.
            if not self._api_key:
                logger.error(
                    "[%s] Refusing to start: API_SERVER_KEY is required for the API server, "
                    "including loopback-only binds on %s.",
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
        if self._dev_supervisor_loop_task:
            self._dev_supervisor_loop_task.cancel()
            self._dev_supervisor_loop_task = None
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
