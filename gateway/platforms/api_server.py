Warning: truncated output (original token count: 64106)
Total output lines: 4615

"""OpenAI-compatible API server platform adapter (aiohttp).

Serves /v1/chat/completions, /v1/responses, /v1/models, /v1/capabilities, /api/sessions,
/v1/runs, /api/jobs and /health* (full table: ``APIServerAdapter._http_route_table``); any
OpenAI-compatible frontend connects at http://localhost:8642/v1 with API_SERVER_KEY. Under
``gateway.multiplex_profiles`` secondary profiles live at ``/p/<profile>/...``.
"""

import asyncio
import concurrent.futures
import errno
import hashlib
import hmac
import itertools
import json
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar, copy_context
from functools import wraps
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# _resolve_request_profile result for a /p/<profile>/ prefix this gateway does not serve (-> 404);
# distinct from None (no prefix / multiplexing off -> default profile).
_PROFILE_REJECTED = object()


def _prefix_names_served_profile(profile: str) -> bool:
    """True when a /p/<profile>/ prefix names the profile this gateway serves. Fail closed: a
    single-profile gateway answering /p/<x>/ served the owner's toolsets under another URL."""
    try:
        from hermes_cli.profiles import profile_matches_home
        return profile_matches_home(profile)
    except Exception:
        return False


# Per-request /p/<profile>/ selection: set by the profile-prefix middleware, read by handlers.
_api_request_profile: ContextVar[Optional[str]] = ContextVar(
    "api_server_request_profile", default=None)
_api_request_browser_control_principal: ContextVar[str] = ContextVar(
    "api_server_browser_control_principal", default="")
_api_request_browser_control_transport_family: ContextVar[str] = ContextVar(
    "api_server_browser_control_transport_family", default="")

class _ArtifactScopeFacade:
    """Minimal scope for ``artifact_scope_key``: server-derived principal + session + transport family."""
    __slots__ = ("principal_id", "session_id", "transport_family")

    def __init__(self, principal_id: str, *, session_id: str = "", transport_family: str = ""):
        self.principal_id = principal_id
        self.session_id = session_id
        self.transport_family = transport_family


# Advertised in capabilities and echoed in registration responses; validated by the broker.
_BROWSER_CONTROL_PROTOCOL_VERSION = 1

# /v1/capabilities static feature flags (order is part of the JSON shape).
_STATIC_FEATURE_FLAGS = {
    "run_status": True, "run_events_sse": True, "run_stop": True, "run_steer": True,
    "run_approval_response": True, "tool_progress_events": True, "approval_events": True,
    "session_resources": True, "model_options": True, "session_chat": True,
    "session_chat_streaming": True, "session_fork": True, "session_model_lock": True,
    "reasoning_streaming": True,
    "admin_config_rw": False, "jobs_admin": False, "memory_write_api": False,
    "skills_api": True, "audio_api": False, "realtime_voice": False,
    "session_continuity_header": "X-Hermes-Session-Id",
    "session_key_header": "X-Hermes-Session-Key"}
# /v1/capabilities "endpoints" table: name -> (method, path).
_CAPABILITY_ENDPOINTS = (
    ("health", ("GET", "/health")), ("health_detailed", ("GET", "/health/detailed")),
    ("models", ("GET", "/v1/models")), ("model_options", ("GET", "/api/model/options")),
    ("chat_completions", ("POST", "/v1/chat/completions")),
    ("responses", ("POST", "/v1/responses")), ("runs", ("POST", "/v1/runs")),
    ("run_status", ("GET", "/v1/runs/{run_id}")),
    ("run_events", ("GET", "/v1/runs/{run_id}/events")),
    ("run_approval", ("POST", "/v1/runs/{run_id}/approval")),
    ("run_steer", ("POST", "/v1/runs/{run_id}/steer")),
    ("run_stop", ("POST", "/v1/runs/{run_id}/stop")), ("skills", ("GET", "/v1/skills")),
    ("toolsets", ("GET", "/v1/toolsets")), ("sessions", ("GET", "/api/sessions")),
    ("session_create", ("POST", "/api/sessions")),
    ("session", ("GET", "/api/sessions/{session_id}")),
    ("session_update", ("PATCH", "/api/sessions/{session_id}")),
    ("session_delete", ("DELETE", "/api/sessions/{session_id}")),
    ("session_messages", ("GET", "/api/sessions/{session_id}/messages")),
    ("session_fork", ("POST", "/api/sessions/{session_id}/fork")),
    ("session_chat", ("POST", "/api/sessions/{session_id}/chat")),
    ("session_chat_stream", ("POST", "/api/sessions/{session_id}/chat/stream")),
    ("session_model_lock", ("POST", "/api/sessions/{session_id}/model")),
    ("browser_control_register", ("POST", "/v1/browser-control/register")),
    ("browser_control_ws", ("GET", "/v1/browser-control/ws")),
    ("artifact_upload", ("POST", "/v1/artifacts/upload")),
    ("artifact_download", ("GET", "/v1/artifacts/download/{artifact_id}")))
_BROWSER_CONTROL_WS_PROTOCOL = "hermes-browser-control-v1"
_BROWSER_CONTROL_TICKET_PROTOCOL_PREFIX = "hermes-browser-control-ticket."


def _approval_event_choices(*, smart_denied: bool, allow_session: bool, allow_permanent: bool) -> list[str]:
    if smart_denied or not allow_session:
        return ["once", "deny"]
    return ["once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]


def _approval_request_event(run_id: str, approval_data: Optional[Dict[str, Any]], **fields: Any) -> Dict[str, Any]:
    """The ``approval.request`` payload every approval surface emits (runs bridge, session stream,
    chat completions): the flagged command redacted before egress (#48456), the ``_run_event``
    envelope, and the ``choices`` the client may send back to ``POST /v1/runs/{id}/approval``."""
    from gateway.platforms.api_server_runs import _run_event
    event = dict(approval_data or {})
    if "command" in event:
        from gateway.run import _redact_approval_command
        event["command"] = _redact_approval_command(event.get("command"))
    event.update(_run_event(run_id, "approval.request", **fields, choices=_approval_event_choices(
        smart_denied=bool(event.get("smart_denied")),
        allow_session=event.get("allow_session") is not False,
        allow_permanent=event.get("allow_permanent") is not False)))
    return event


try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.display_config import resolve_display_setting
from gateway.platforms import api_server_room_dispatch as _room_dispatch
from gateway.platforms import api_server_room_grants as _room_grants
from gateway.platforms import api_server_runs as _api_runs
from gateway.platforms.api_server_openai_routes import OpenAICompatRoutesMixin
from gateway.platforms.api_server_memory_sessions import ApiServerMemorySessions
from gateway.platforms.base import (
    MEDIA_TAG_CLEANUP_RE, BasePlatformAdapter, SendResult, _terminal_sentinel_start, is_network_accessible,
    validate_media_delivery_path)
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from agent.redact import redact_sensitive_text
from agent.interrupt_compat import request_hard_interrupt
from gateway.readiness import collect_runtime_readiness
from gateway.browser_control_artifacts import (
    ArtifactError, ArtifactRateLimiter, ArtifactStore, ArtifactTooLarge, DEFAULT_ALLOWED_MIME_TYPES,
    DEFAULT_MAX_ARTIFACT_BYTES, DEFAULT_ARTIFACT_TTL_SECONDS)
from gateway.browser_control_broker import (
    BROWSER_CONTROL_ARTIFACT_CAPABILITIES, BROWSER_CONTROL_CAPABILITIES, BROWSER_CONTROL_DEVELOPER_CAPABILITIES,
    ControllerScope, ControllerTicketInvalid, browser_control_developer_mode,
    browser_control_protocol_supported, filter_browser_control_capabilities, get_browser_control_broker)

from gateway.platforms._shared import coerce_port as _coerce_port
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from gateway.platforms.tcp_site import start_tcp_site
from hermes_state_errors import SessionActiveWriteGuardError


logger = logging.getLogger(__name__)


def _browser_controller_ws_sender(ws, loop, *, wait_timeout: float = 10.0):
    """Return a loop-aware broker sender for one aiohttp controller socket.

    A wait timeout means the coroutine is still in flight, not that the frame was rejected:
    keep the broker command pending (its own deadline decides); a real send error propagates.
    """

    def send(frame: dict) -> None:
        if ws.closed:
            raise ConnectionError("browser-control websocket is closed")
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            loop.create_task(ws.send_json(frame))
            return
        future = asyncio.run_coroutine_threadsafe(ws.send_json(frame), loop)
        try:
            future.result(timeout=wait_timeout)
        except concurrent.futures.TimeoutError:
            if future.done():
                raise

            def observe_late_send(completed):
                try:
                    completed.result()
                except Exception:
                    logger.exception("browser-controller websocket send failed after wait timeout")
            future.add_done_callback(observe_late_send)
    return send


async def _call_verifier(verifier, *args, **kwargs):
    """Await a sync-or-async verifier; sync ones may do blocking network I/O (signing-cert / JWKS
    fetches), so they run off the loop."""
    if asyncio.iscoroutinefunction(verifier):
        return await verifier(*args, **kwargs)
    return await asyncio.to_thread(verifier, *args, **kwargs)


def _hermes_version() -> str:
    """Canonical base version for API protocol and compatibility payloads."""
    from hermes_cli.version_info import get_version_info
    return get_version_info().base_version


# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
_BIND_ATTEMPTS = 5  # EADDRINUSE retries while a restart's predecessor releases the port (#91547)


def listen_address(extra: Dict[str, Any]) -> tuple[str, int]:
    """Host/port the adapter binds: config.yaml ``platforms.api_server`` wins over the env fallbacks.

    Shared with the CLI restart path, which must wait on the SAME address the replacement will
    bind — an env-only reading missed every config.yaml port (#91547).
    """
    host = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
    raw_port = extra.get("port")
    if raw_port is None:
        raw_port = os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))
    return host, _coerce_port(raw_port, DEFAULT_PORT)
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = 10_000_000  # 10 MB — accommodates long agent conversations with tool calls
# Send a comment before remote API clients' common 20-second idle deadline.
# This constant is shared by OpenAI chat/Responses and native session SSE.
CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 10.0
API_SERVER_HEARTBEAT_SECONDS = 30.0
API_SERVER_LATENCY_SAMPLE_LIMIT = 512
MAX_NORMALIZED_TEXT_LENGTH = 65_536  # 64 KB cap for normalized content parts
MAX_CONTENT_LIST_SIZE = 1_000  # Max items when content is an array
RESPONSES_AUTO_TRUNCATION_HISTORY_LIMIT = 100


class ThreadSafeAsyncQueue(asyncio.Queue):
    """``asyncio.Queue`` a non-loop thread (run_conversation's executor) can push into via
    ``put_threadsafe``; the SSE consumer's ``await get()`` is woken by ``call_soon_threadsafe``."""

    def put_threadsafe(self, item, *, loop: asyncio.AbstractEventLoop = None) -> None:
        (loop or self._loop_ref).call_soon_threadsafe(self.put_nowait, item)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Always constructed inside a running async handler (the SSE
        # request handlers below), so get_running_loop() is safe here.
        self._loop_ref = asyncio.get_running_loop()


def _sse_frame(
    data: Any, *, event: str = None, ensure_ascii: bool = True, id: Optional[int] = None
) -> bytes:
    """Encode one SSE frame (``id:``/``event:`` lines if given, then ``data: <json>\n\n``) for
    every SSE writer. ``ensure_ascii=False`` keeps raw non-ASCII on the wire."""
    prefix = (f"id: {id}\n" if id is not None else "") + (f"event: {event}\n" if event else "")
    return f"{prefix}data: {json.dumps(data, ensure_ascii=ensure_ascii)}\n\n".encode()


_TRUE_REQUEST_BOOL_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_REQUEST_BOOL_STRINGS = frozenset({"0", "false", "no", "off"})


def _coerce_request_bool(value: Any, default: bool = False) -> bool:
    """Normalize boolean-like payload values; only explicit bool-ish scalars count (some
    frontends send ``"false"`` for ``stream``, which is truthy), else ``default``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_REQUEST_BOOL_STRINGS:
            return True
        return False if normalized in _FALSE_REQUEST_BOOL_STRINGS else default
    return bool(value) if isinstance(value, (int, float)) else default


_REQUEST_OPTION_MISSING = object()
# Full internal ladder + "none" (what /reasoning and config.yaml accept); provider
# vocabulary clamping happens downstream in agent.reasoning_effort.
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})
_RUNTIME_AGENT_OVERRIDE_KEYS = (
    "api_key", "base_url", "provider", "api_mode", "command", "args", "credential_pool")


def _clean_request_string(value: Any) -> Optional[str]:
    """Return a stripped request string, or None for absent/non-string values."""
    return (value.strip() or None) if isinstance(value, str) else None


def _request_reasoning_config(model_options: Any) -> Optional[Dict[str, Any]]:
    """Translate model_options (structured ``reasoning`` or legacy ``reasoning_effort``) into
    AIAgent reasoning_config; unknown effort values are ignored, never raised."""
    if not isinstance(model_options, dict):
        return None
    reasoning = model_options.get("reasoning")
    enabled: Any = None
    effort: Any = model_options.get("reasoning_effort")
    if isinstance(reasoning, dict):
        enabled = reasoning.get("enabled")
        effort = reasoning.get("effort", effort)
    effort_norm = str(effort).strip().lower() if effort is not None else ""
    if enabled is False or effort_norm == "none":
        return {"enabled": False}
    if effort_norm in _REASONING_EFFORTS and effort_norm != "none":
        return {"enabled": True, "effort": effort_norm}
    if enabled is True:
        return {"enabled": True}
    return None


def _request_service_tier(model_options: Any) -> Any:
    """Return a per-request service_tier override or _REQUEST_OPTION_MISSING."""
    if not isinstance(model_options, dict):
        return _REQUEST_OPTION_MISSING
    if "service_tier" in model_options:
        raw_tier = model_options.get("service_tier")
        return _clean_request_string(raw_tier) if isinstance(raw_tier, str) else raw_tier
    if "fast" in model_options:
        return "priority" if _coerce_request_bool(model_options.get("fast"), default=False) else None
    return _REQUEST_OPTION_MISSING


def _apply_runtime_agent_overrides(
    runtime_kwargs: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge resolved provider/runtime fields into ``runtime_kwargs`` in place."""
    if not isinstance(overrides, dict):
        return runtime_kwargs
    for key in _RUNTIME_AGENT_OVERRIDE_KEYS:
        value = overrides.get(key)
        if value is None:
            continue
        runtime_kwargs[key] = list(value) if key == "args" and isinstance(value, (list, tuple)) else value
    return runtime_kwargs


def _resolve_request_runtime_agent_kwargs(provider: str, target_model: Optional[str] = None) -> Dict[str, Any]:
    """gateway.run._resolve_runtime_agent_kwargs() for an explicit provider/model, so an API
    caller uses the same authenticated provider catalog without mutating config.yaml."""
    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error, _get_model_config
    try:
        runtime = resolve_runtime_provider(requested=provider, target_model=target_model)
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    return {
        **{k: runtime.get(k) for k in ("api_key", "base_url", "provider", "api_mode", "command")},
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool")}


def _request_agent_overrides(
    body: Any, *, virtual_model: Optional[str] = None, allow_bare_model: bool = True
) -> Dict[str, Any]:
    """Extract per-request model/provider/options for _run_agent.

    The virtual model (``hermes-agent``) means "gateway default". A bare ``model`` without
    ``provider`` is honored only when ``allow_bare_model`` (generic clients hardcode "gpt-4o";
    OpenAI-compatible handlers pass the ``direct_model_requests`` opt-in, Hermes-native
    endpoints always allow it). An explicit ``provider`` is always honored.
    """
    if not isinstance(body, dict):
        return {}
    overrides: Dict[str, Any] = {}
    provider = _clean_request_string(body.get("provider"))
    if provider:
        overrides["requested_provider"] = provider
    model = _clean_request_string(body.get("model"))
    if model and model != virtual_model and (provider or allow_bare_model):
        overrides["requested_model"] = model
    model_options = body.get("model_options")
    if isinstance(model_options, dict):
        overrides["model_options"] = dict(model_options)
    return overrides


def _request_relay_metadata(body: Any) -> Dict[str, Any]:
    """Extract Relay metadata from an OpenAI request body."""
    if not isinstance(body, dict):
        return {}
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    return dict(metadata)


def _is_compressed_summary_message(message: Any) -> bool:
    """Recognize every compaction carrier shape via the compressor's own classifier
    (SessionDB drops the in-process marker; a prefix scan misses merge-into-tail carriers)."""
    if not isinstance(message, dict):
        return False
    from agent.context_compressor import is_compaction_summary_message
    return is_compaction_summary_message(message)


def _project_client_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Strip compaction scaffolding: standalone handoffs become hidden empty rows (stable
    ids), merged handoffs keep only the real prior-tail content; inherited tool calls dropped."""
    from agent.compaction_display import (
        _COMPACTION_INTERNAL_FIELDS, project_compaction_message_for_display)
    if (message.get("display_kind") == "hidden"
            and (message.get("display_metadata") or {}).get("notification_category") == "diagnostic"):
        # Retain row identity and execution evidence in storage, not in the notification UI.
        return {k: v for k, v in message.items() if k in {
            "id", "session_id", "role", "timestamp", "display_kind", "platform_message_id",
        }} | {"content": ""}
    projected = project_compaction_message_for_display(message)
    if projected is None:
        projected = {k: v for k, v in message.items() if k not in _COMPACTION_INTERNAL_FIELDS}
        projected["content"] = ""
        projected["display_kind"] = "hidden"
    return projected


def _auto_truncate_response_history(
    conversation_history: List[Dict[str, Any]],
    *,
    limit: int = RESPONSES_AUTO_TRUNCATION_HISTORY_LIMIT) -> List[Dict[str, Any]]:
    """Keep the most recent ``limit`` messages, always preserving compaction summaries
    wherever they sit (the /compress path can leave them after a retained system head)."""
    if limit <= 0 or len(conversation_history) <= limit:
        return conversation_history
    summary_indices = [i for i, m in enumerate(conversation_history) if _is_compressed_summary_message(m)]
    if not summary_indices:
        return conversation_history[-limit:]
    kept_indices = set(summary_indices[:limit])
    remaining = limit - len(kept_indices)
    if remaining > 0:
        summary_index_set = set(summary_indices)
        for index in range(len(conversation_history) - 1, -1, -1):
            if index in summary_index_set:
                continue
            kept_indices.add(index)
            remaining -= 1
            if remaining <= 0:
                break
    return [conversation_history[index] for index in sorted(kept_indices)]


def _cap_text(text: str) -> str:
    return text[:MAX_NORMALIZED_TEXT_LENGTH] if len(text) > MAX_NORMALIZED_TEXT_LENGTH else text


def _cap_list(items: list) -> list:
    return items[:MAX_CONTENT_LIST_SIZE] if len(items) > MAX_CONTENT_LIST_SIZE else items


def _normalize_chat_content(content: Any, *, _max_depth: int = 10, _depth: int = 0) -> str:
    """Flatten OpenAI chat content (string or typed-part array) into one plain string; non-text
    parts are skipped, recursion depth / list size / output length are bounded."""
    if _depth > _max_depth or content is None:
        return ""
    if isinstance(content, str):
        return _cap_text(content)
    if isinstance(content, list):
        parts: List[str] = []
        total_len = 0
        for item in _cap_list(content):
            part = ""
            if isinstance(item, str):
                part = item
            elif isinstance(item, dict):
                if str(item.get("type") or "").strip().lower() in _TEXT_PART_TYPES:
                    text = item.get("text", "")
                    if text:
                        with suppress(Exception):
                            part = str(text)
            elif isinstance(item, list):
                part = _normalize_chat_content(item, _max_depth=_max_depth, _depth=_depth + 1)
            if part:
                part = _cap_text(part)
                parts.append(part)
                total_len += len(part)
            if total_len >= MAX_NORMALIZED_TEXT_LENGTH:
                break
        return _cap_text("\n".join(parts))
    try:
        return _cap_text(str(content))
    except Exception:
        return ""


# Chat Completions / Responses part-type spellings; emitted shape is always the canonical
# ``{"type": "text", ...}`` / ``{"type": "image_url", ...}`` the agent pipeline understands.
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})
_FILE_PART_TYPES = frozenset({"file", "input_file"})


def _normalize_image_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one image part (Responses top-level ``image_url`` string or Chat Completions
    ``{"url", "detail"}`` dict) into the canonical vision shape; raises ValueError."""
    detail = part.get("detail")
    image_ref = part.get("image_url")
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
                "Non-image data payloads are not supported.")
    elif not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError(
            "invalid_image_url:Image inputs must use http(s) URLs or data:image/... URLs.")
    image_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url_value}}
    if detail is not None:
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("invalid_content_part:Image detail must be a non-empty string when provided.")
        image_part["image_url"]["detail"] = detail.strip()
    return image_part


def _normalize_multimodal_content(content: Any) -> Any:
    """Validate multimodal content: a plain string when text-only, else canonical ``text`` /
    ``image_url`` parts (native OpenAI vision shape; Anthropic conversion happens downstream).

    Raises ``ValueError("<code>:<message>")`` with codes ``unsupported_content_type`` (file
    parts, non-image data URLs, unknown types), ``invalid_image_url``, ``invalid_content_part``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return _cap_text(content)
    if not isinstance(content, list):
        return _normalize_chat_content(content)
    normalized_parts: List[Dict[str, Any]] = []
    for part in _cap_list(content):
        if isinstance(part, str):
            if part:
                normalized_parts.append({"type": "text", "text": _cap_text(part)})
            continue
        if not isinstance(part, dict):
            continue  # unknown scalars are ignored for forward compatibility (e.g. ``refusal``)
        raw_type = part.get("type")
        part_type = str(raw_type or "").strip().lower()
        if part_type in _TEXT_PART_TYPES:
            text = part.get("text")
            if text is not None and str(text):
                normalized_parts.append({"type": "text", "text": _cap_text(str(text))})
        elif part_type in _IMAGE_PART_TYPES:
            normalized_parts.append(_normalize_image_part(part))
        elif part_type in _FILE_PART_TYPES:
            raise ValueError(
                "unsupported_content_type:Inline image inputs are supported, "
                "but uploaded files and document inputs are not supported on this endpoint.")
        else:
            raise ValueError(
                f"unsupported_content_type:Unsupported content part type {raw_type!r}. "
                "Only text and image_url/input_image parts are supported.")
    if not normalized_parts:
        return ""
    # Text-only collapses to a plain string so trajectory logging and prompt caching see
    # the native shape.
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
                if ptype in _IMAGE_PART_TYPES or (
                        ptype in _TEXT_PART_TYPES and str(part.get("text") or "").strip()):
                    return True
    return False


def _multimodal_validation_error(exc: ValueError, *, param: str) -> "web.Response":
    """Translate a ``_normalize_multimodal_content`` ValueError into a 400 response."""
    raw = str(exc)
    code, _, message = raw.partition(":")
    if not message:
        code, message = "invalid_content_part", raw
    return _error_response(message, 400, code=code, param=param)


def _reap_disconnected_agent_processes(
    agent: Any, *, source: str = "api_server_sse_disconnect") -> None:
    """Reap background processes an abandoned API-server turn created (these turns bypass
    ``TurnRunner``). Daemon-thread fire-and-forget; epoch-gated so a stale reaper never kills
    a newer run's process on a shared task_id.

    Mirrors the gateway-turn cleanup in ``gateway/run.py`` (#76115) for this API-server surface, which runs
    its own agent lifecycle via ``_run_agent`` and never passes through ``TurnRunner`` — so it needs its own
    trigger for the same baseline-diff reap.
    """
    process_task_id = getattr(agent, "_gateway_turn_process_task_id", "")
    process_baseline = getattr(agent, "_gateway_turn_process_baseline", None)
    if not process_task_id or process_baseline is None:
        return
    epoch = getattr(agent, "_gateway_turn_process_epoch", None)
    is_still_current: Optional[Any] = None
    if epoch is not None:
        def _epoch_still_current(_task_id=process_task_id, _epoch=epoch):
            # Skip only when a NEWER run claimed this task_id. A missing entry means
            # our own clear pruned it — no newer claimant, so the reap must proceed.
            with _TURN_PROCESS_EPOCH_LOCK:
                current = _TURN_PROCESS_EPOCHS.get(_task_id)
            return current is None or current == _epoch
        is_still_current = _epoch_still_current
    from gateway.run import _reap_gateway_turn_processes
    threading.Thread(
        target=copy_context().run, args=(_reap_gateway_turn_processes, process_task_id, process_baseline),
        kwargs={"source": source, "is_still_current": is_still_current},
        name=f"api-turn-reaper-{process_task_id[:12]}", daemon=True).start()


# Per-task-id run epochs for the reap gate: monotonic counter (never reused),
# pruned on clear while still current, so the dict is bounded to in-flight runs.
_TURN_PROCESS_EPOCHS: Dict[str, int] = {}
_TURN_PROCESS_EPOCH_LOCK = threading.Lock()
_TURN_PROCESS_EPOCH_COUNTER = itertools.count(1)


def _publish_turn_process_ownership(agent: Any, task_id: str) -> None:
    """Snapshot the process baseline and claim the task_id's epoch — the single place every
    API-server agent lifecycle records turn ownership (marker names cannot drift)."""
    from tools.process_registry import process_registry
    with _TURN_PROCESS_EPOCH_LOCK:
        epoch = next(_TURN_PROCESS_EPOCH_COUNTER)
        _TURN_PROCESS_EPOCHS[task_id] = epoch
    agent._gateway_turn_process_task_id = task_id
    agent._gateway_turn_process_baseline = process_registry.snapshot_running_ids(task_id)
    agent._gateway_turn_process_epoch = epoch


def _clear_turn_process_ownership(agent: Any) -> None:
    """Clear turn ownership as soon as the turn ends: a later disconnect/cancel must not reap
    background work the turn deliberately left running (same guard as gateway/run.py)."""
    task_id = getattr(agent, "_gateway_turn_process_task_id", "")
    epoch = getattr(agent, "_gateway_turn_process_epoch", None)
    if task_id and epoch is not None:
        with _TURN_PROCESS_EPOCH_LOCK:
            # Prune only when this run is still the current claimant; a
            # newer concurrent run owns the entry otherwise.
            if _TURN_PROCESS_EPOCHS.get(task_id) == epoch:
                del _TURN_PROCESS_EPOCHS[task_id]
    agent._gateway_turn_process_task_id = ""
    agent._gateway_turn_process_baseline = frozenset()
    agent._gateway_turn_process_epoch = None


def _session_chat_user_message(body: Dict[str, Any], *, param: str = "message") -> tuple[Any, Optional["web.Response"]]:
    """Parse and normalize session chat ``message`` / ``input`` like chat completions."""
    user_message = body.get("message") or body.get("input")
    if not _content_has_visible_payload(user_message):
        return None, _error_response("Missing 'message' field", 400, code="missing_message")
    try:
        return _normalize_multimodal_content(user_message), None
    except ValueError as exc:
        return None, _multimodal_validation_error(exc, param=param)


def _request_turn_author(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalized body ``author``, None when absent or null, ValueError when not an object. It only labels memory."""
    raw = body.get("author")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("author must be an object")
    from agent.turn_author import parse_turn_author
    return parse_turn_author(raw)


_USAGE_TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def _chat_usage_payload(usage: Dict[str, Any]) -> Dict[str, int]:
    """OpenAI Chat Completions ``usage`` block (prompt/completion/total) from the agent's usage."""
    values = (usage.get(key, 0) for key in _USAGE_TOKEN_KEYS)
    return dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), values))


def _responses_usage_payload(usage: Dict[str, Any]) -> Dict[str, int]:
    """OpenAI Responses ``usage`` block from the agent's usage dict."""
    return {key: usage.get(key, 0) for key in _USAGE_TOKEN_KEYS}


async def _abandon_agent_task(
    agent_ref, agent_task, reason: str, *,
    reap_source: str = "api_server_sse_disconnect", await_cancel: bool = True) -> None:
    """Interrupt + reap an abandoned SSE agent run, then cancel its task wrapper.
    ``await_cancel=False`` on the CancelledError path, which must not await in the handler."""
    agent = agent_ref[0] if agent_ref else None
    if agent is not None:
        with suppress(Exception):
            # The abandoning client/server is the issuer, not the user (#112647).
            request_hard_interrupt(agent, reason, tool_reason=reason.lower())
        _reap_disconnected_agent_processes(agent, source=reap_source)
    if not agent_task.done():
        agent_task.cancel()
        if await_cancel:
            with suppress(asyncio.CancelledError, Exception):
                await agent_task


def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


class ResponseStore:
    """SQLite-backed LRU store for Responses API state (full conversation history per response
    for ``previous_response_id`` chaining). Persists across restarts; in-memory fallback."""

    def __init__(self, max_size: int = MAX_STORED_RESPONSES, db_path: str = None):
        self._max_size = max_size
        if db_path is None:
            db_path = ":memory:"
            with suppress(Exception):
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
        self._db_path: Optional[str] = db_path if db_path != ":memory:" else None
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        # Shared WAL-fallback so response_store.db degrades gracefully on NFS/SMB/FUSE homes.
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="response_store.db")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "response_id TEXT PRIMARY KEY, data TEXT NOT NULL, accessed_at REAL NOT NULL)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations (name TEXT PRIMARY KEY, response_id TEXT NOT NULL)")
        self._conn.commit()
        # Conversation history lives here: owner-only perms, once at init (not per commit).
        self._tighten_file_permissions()

    def _tighten_file_permissions(self) -> None:
        """Force owner-only permissions on the DB and SQLite sidecars."""
        if not self._db_path:
            return
        for candidate in (Path(self._db_path), Path(f"{self._db_path}-wal"), Path(f"{self._db_path}-shm")):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug("Failed to restrict response store permissions for %s", candidate, exc_info=True)

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", (response_id,)).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id))
        self._conn.commit()
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupted JSON in response store for id=%s, evicting entry", response_id)
            self._conn.execute("DELETE FROM responses WHERE response_id = ?", (response_id,))
            self._conn.commit()
            return None

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()))
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            evict_ids = [row[0] for row in self._conn.execute(
                "SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?",
                (count - self._max_size,)).fetchall()]
            if evict_ids:
                placeholders = ",".join("?" for _ in evict_ids)
                # Conversation mappings pointing at evicted responses go too.
                self._conn.execute(f"DELETE FROM conversations WHERE response_id IN ({placeholders})", evict_ids)
                self._conn.execute(f"DELETE FROM responses WHERE response_id IN ({placeholders})", evict_ids)
        self._conn.commit()

    def delete(self, response_id: str) -> bool:
        """Remove a response (and conversation mappings to it). True if found and deleted."""
        self._conn.execute("DELETE FROM conversations WHERE response_id = ?", (response_id,))
        cursor = self._conn.execute("DELETE FROM responses WHERE response_id = ?", (response_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        row = self._conn.execute("SELECT response_id FROM conversations WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        self._conn.execute("INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)", (name, response_id))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with suppress(Exception):
            self._conn.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0


_CORS_HEADERS = {
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, Idempotency-Key, X-Hermes-Session-Id"}
_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "no-referrer"}

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
      …44106 tokens truncated…s)
        except Exception:
            # Fail closed: a crashing verifier must never admit a fire.
            logger.exception("cron fire: verifier crashed; rejecting token")
            claims = None
        if claims is None:
            logger.warning("cron fire: rejected invalid token: %s", self._request_audit_log_suffix(request))
            return web.json_response({"error": "invalid fire token"}, status=401)
        draining = self._draining_response()
        if draining is not None:
            return draining
        with _reserve_pending_api_work(self) as reservation:
            body = {}
            with suppress(Exception):
                body = await request.json()
            job_id = (body or {}).get("job_id")
            if not job_id:
                return web.json_response({"error": "missing job_id"}, status=400)
            # `hermes pause` ESTOP: refuse the fire and ask NAS to retry later.
            # Placed after JWT verify (don't leak pause state to unauth callers)
            # and after the drain check (drain is transient shutdown, ESTOP is
            # operator override). 503 + Retry-After reschedules the job via NAS
            # retry or the misfire backstop rather than silently dropping it —
            # matches _CRON_FIRE_RETRY_AFTER_SECONDS in web_routers/cron.py.
            with suppress(ImportError):
                from agent.estop import check_paused as _estop_check_paused
                if _estop_check_paused("cron-webhook", logger):
                    return web.json_response(
                        {"error": "hermes is paused (ESTOP)", "job_id": job_id},
                        status=503,
                        headers={"Retry-After": str(60)},
                    )
            from cron.scheduler_provider import provider_supports_split_fire, resolve_cron_scheduler
            provider = resolve_cron_scheduler()
            loop = asyncio.get_running_loop()
            # Live adapters (parity with the built-in ticker): E2EE / relay-fronted platforms
            # have no native credential, so without them delivery fails.
            runner = self.gateway_runner or request.app.get("gateway_runner")
            if runner is None:
                with suppress(Exception):
                    from gateway.run import _gateway_runner_ref
                    runner = _gateway_runner_ref()
            adapters = getattr(runner, "adapters", None) or None

            def _detach_fire(fire_fn, *fire_args) -> "web.Response":
                # The done callback owns the reservation once the task is detached.
                task = asyncio.create_task(asyncio.to_thread(fire_fn, *fire_args, adapters=adapters, loop=loop))
                reservation["detached"] = True
                task.add_done_callback(lambda _task: _release_pending_api_work(self, reservation))
                self._track_background_task(task, tolerate_missing=True)
                return web.json_response({"status": "accepted", "job_id": job_id}, status=202)

            if not provider_supports_split_fire(provider):
                # A legacy single-phase provider overrides ``fire_due`` but inherits the base
                # ``claim_fire``; the split path would silently bypass that override.
                return _detach_fire(provider.fire_due, job_id)
            # Persist the attempt + exact store owner before acknowledging NAS; a failure here
            # is retryable and the reservation remains attached.
            try:
                claimed_job = await asyncio.to_thread(provider.claim_fire, job_id)
            except Exception as exc:
                logger.error("cron fire admission failed for %s: %s", job_id, exc)
                return web.json_response({"error": "cron fire admission failed", "job_id": job_id}, status=503)
            if claimed_job is None:
                return web.json_response({"status": "duplicate", "job_id": job_id}, status=200)
            return _detach_fire(provider.fire_claimed, claimed_job)

    # -- Agent execution --------------------------------------------------------------

    def _track_background_task(self, task, *, tolerate_missing: bool = False) -> None:
        """Register a task in ``_background_tasks`` (tolerates test doubles) with auto-discard.
        ``tolerate_missing`` (cron fire paths) also swallows AttributeError from the whole
        registration; the run/sweep paths only tolerate an unhashable task."""
        if tolerate_missing:
            with suppress(TypeError, AttributeError):
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return
        with suppress(TypeError):
            self._background_tasks.add(task)
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

    def _concurrency_limited_response(self) -> Optional["web.Response"]:
        """429 when the concurrent-run cap is reached (0 disables), else None. Uses the same
        adapter-owned work count as shutdown draining (admitted requests included)."""
        limit = self._max_concurrent_runs
        if limit <= 0:
            return None
        inflight = self.active_agent_work_count()
        # The current request's own reservation must not consume its last available slot.
        reservation = _api_agent_request_reservation.get()
        if reservation and reservation["active"]:
            inflight -= 1
        if inflight >= limit:
            return _error_response(
                f"Too many concurrent runs (max {limit})", 429, err_type="rate_limit_error",
                code="rate_limit_exceeded", headers={"Retry-After": "1"})
        return None

    @staticmethod
    def _bind_api_server_session(
        *, chat_id: str = "", session_key: str = "", session_id: str = "", profile: str = "",
        browser_control_principal: str = "", browser_control_transport_family: str = "",
        session_history_delivery: str = "") -> list:
        """Bind an API turn with push disabled and history delivery default-denied.

        Only routes whose continuation reads SessionDB may pass "1". An omitted
        declaration or fingerprint-derived identity keeps delegation synchronous.

        ``profile`` is the ``/p/<profile>/`` prefix serving the request (``""`` = default). It must
        reach ``HERMES_SESSION_PROFILE``: the persistent-Docker container key is derived from it, so an
        unbound profile collapses every profile's turns onto the default sandbox (#96370)."""
        from gateway.session_context import set_session_vars
        return set_session_vars(
            platform="api_server", chat_id=chat_id, session_key=session_key, session_id=session_id,
            profile=profile, browser_control_principal=browser_control_principal,
            browser_control_transport_family=browser_control_transport_family,
            async_delivery=False, cron_session="", session_history_delivery=session_history_delivery)

    def _turn_runtime_metadata(
        self, agent: Any, *, route: Optional[Dict[str, Any]], requested_runtime: Optional[Dict[str, Any]],
        route_source: str, confirmed_runtime_lock: bool) -> Dict[str, Any]:
        """Sanitized actual-vs-requested runtime for a finished turn; raises RuntimeError when a
        confirmed model lock's provider/model differs from what the agent actually ran with."""
        runtime = dict(getattr(agent, "_hermes_api_runtime", {}) or {})
        raw_provider = getattr(agent, "provider", "")
        raw_model = getattr(agent, "model", "")
        actual_provider = self._clean_runtime_id(raw_provider, max_len=80) if isinstance(raw_provider, str) else ""
        actual_model = self._clean_runtime_id(raw_model) if isinstance(raw_model, str) else ""
        resolved_provider = self._clean_runtime_id(runtime.get("provider"), max_len=80)
        for key, actual in (("provider", actual_provider), ("model", actual_model)):
            if actual:
                runtime[key] = actual
            else:
                runtime.setdefault(key, "")
        route = route or {}
        requested_runtime = requested_runtime or {}
        if confirmed_runtime_lock:
            requested_provider = self._clean_runtime_id(
                route.get("provider") or requested_runtime.get("provider"), max_len=80)
            # _create_agent records the provider after resolving the request through the
            # provider catalog. Compare that identity with the agent's actual runtime so
            # aliases and named custom providers do not fail a literal-string check.
            expected_provider = self._clean_runtime_id(
                resolved_provider or requested_provider, max_len=80)
            expected_model = self._clean_runtime_id(route.get("model") or requested_runtime.get("model"))
            if (expected_provider and actual_provider != expected_provider) or (
                expected_model and actual_model != expected_model):
                raise RuntimeError(
                    "confirmed model lock runtime mismatch: "
                    f"expected provider={requested_provider or expected_provider or '<unspecified>'} "
                    f"model={expected_model or '<unspecified>'}; "
                    f"actual provider={actual_provider or '<unknown>'} "
                    f"model={actual_model or '<unknown>'}")
        if requested_runtime:
            model, provider = self._requested_ids(requested_runtime)
            runtime["requested"] = {"provider": provider, "model": model}
        runtime["route_source"] = route_source or runtime.get("route_source") or "global"
        return self._sanitize_runtime_metadata(
            runtime=runtime, requested_runtime=requested_runtime or None, route_source=route_source or "global",
            model_lock=("confirmed" if confirmed_runtime_lock else ""))

    def _finish_turn_result(
        self, agent: Any, result: Any, session_id: Optional[str], *, route, requested_runtime, route_source,
        confirmed_runtime_lock: bool) -> tuple:
        """Attach usage, effective session id, ``_compressed`` and runtime metadata to a finished turn."""
        usage = {"input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                 "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                 "total_tokens": getattr(agent, "session_total_tokens", 0) or 0}
        # Effective session id lets callers track compression-triggered rotations.
        # (#16938)
        _eff_sid = getattr(agent, "session_id", session_id)
        if isinstance(_eff_sid, str) and _eff_sid:
            result["session_id"] = _eff_sid
        # _compressed tells _build_response_conversation_history to store the compacted
        # transcript as-is (rotation changes session_id; in-place compaction sets a flag).
        _session_rotated = isinstance(_eff_sid, str) and isinstance(session_id, str) and _eff_sid != session_id
        if getattr(agent, "_last_compaction_in_place", False) or _session_rotated:
            result["_compressed"] = True
        if requested_runtime or route or confirmed_runtime_lock or (route_source and route_source != "global"):
            runtime = self._turn_runtime_metadata(
                agent, route=route, requested_runtime=requested_runtime,
                route_source=route_source, confirmed_runtime_lock=confirmed_runtime_lock)
            if isinstance(result, dict):
                result["runtime"] = runtime
            usage["runtime"] = runtime
        return result, usage

    async def _run_agent(
        self, user_message: str, conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None, session_id: Optional[str] = None,
        stream_delta_callback=None, tool_progress_callback=None, tool_start_callback=None,
        tool_complete_callback=None, interim_assistant_callback=None, reasoning_callback=None,
        status_callback=None, agent_ref: Optional[list] = None, active_run_id: Optional[str] = None,
        gateway_session_key: Optional[str] = None, requested_model: Optional[str] = None,
        requested_provider: Optional[str] = None, model_options: Optional[Dict[str, Any]] = None,
        route: Optional[Dict[str, Any]] = None, session_model: Optional[str] = None,
        requested_runtime: Optional[Dict[str, Any]] = None, route_source: str = "global",
        confirmed_runtime_lock: bool = False, bind_declared_conversation: bool = False,
        session_history_delivery: str = "", turn_author: Optional[Dict[str, Any]] = None,
        relay_metadata: Optional[Dict[str, Any]] = None, notification_category: str = "result",
        resume_unanswered_turn: bool = False, approval_notify_callback=None,
        approval_session_key: Optional[str] = None) -> tuple:
        """Create an agent and run one turn in a thread executor -> ``(result, usage)``.
        ``approval_notify_callback`` (with ``approval_session_key``) routes dangerous-command
        approval requests to the caller's stream, keyed like ``/v1/runs`` approvals (#51871).
        ``agent_ref[0]`` receives the agent so SSE writers can interrupt it; ``active_run_id``
        registers it in ``_active_run_agents``. Under a confirmed model lock the actual
        provider/model must match or the turn fails; ``runtime`` metadata is attached.
        ``session_history_delivery`` declares #98619 session-id provenance and default-denies: only audited
        producers whose client can address the id again pass "1" (see
        ``_bind_api_server_session``).
        ``turn_author`` only labels the turn for memory attribution. It grants nothing.
        ``resume_unanswered_turn`` marks a policy-gated re-run of a turn whose user row the failed attempt
        already persisted: the transcript's unanswered tail row is adopted from ``conversation_history``
        as THIS turn's user message instead of being appended a second time
        (``agent.session_persistence.adopt_unanswered_turn``; #115325)."""
        loop = asyncio.get_running_loop()
        # ContextVars do not follow run_in_executor threads: capture here, re-enter in _run().
        request_profile = _api_request_profile.get()
        request_browser_control_principal = _api_request_browser_control_principal.get()
        request_browser_control_transport_family = _api_request_browser_control_transport_family.get()

        def _run():
            from gateway.session_context import clear_session_vars
            with self._profile_scope(request_profile):
                tokens = self._bind_api_server_session(
                    chat_id=session_id or "", session_key=gateway_session_key or session_id or "",
                    session_id=session_id or "", profile=request_profile or "",
                    browser_control_principal=request_browser_control_principal,
                    browser_control_transport_family=request_browser_control_transport_family,
                    session_history_delivery=session_history_delivery)
                agent = None
                from agent.notification_presentation import notification_turn
                from gateway.warning_notifications import diagnostic_turn_muted
                muted = diagnostic_turn_muted({"notification_category": notification_category}, "api_server")
                try:
                    agent = self._create_agent(
                        ephemeral_system_prompt=ephemeral_system_prompt, session_id=session_id,
                        stream_delta_callback=stream_delta_callback, tool_progress_callback=tool_progress_callback,
                        tool_start_callback=tool_start_callback, tool_complete_callback=tool_complete_callback,
                        interim_assistant_callback=interim_assistant_callback,
                        reasoning_callback=reasoning_callback, status_callback=status_callback,
                        gateway_session_key=gateway_session_key, requested_model=requested_model,
                        requested_provider=requested_provider, model_options=model_options, route=route,
                        session_model=session_model, confirmed_runtime_lock=confirmed_runtime_lock)
                    if agent_ref is not None:
                        agent_ref[0] = agent
                    if resume_unanswered_turn:
                        # A dispatcher's re-run of a failed delivery turn: the DM's own row is already
                        # in the store (the failed attempt persisted it at turn start), so continue THAT
                        # row instead of appending a second copy of the same text (#115325).
                        from agent.session_persistence import adopt_unanswered_turn

                        adopt_unanswered_turn(conversation_history, user_message, agent)
                    if active_run_id:
                        self._active_run_agents[active_run_id] = agent
                    effective_task_id = session_id or str(uuid.uuid4())
                    # Process baseline for disconnect reaping (this surface bypasses TurnRunner)
                    # + shutdown-interrupt registration, once for every caller.
                    # Baseline for selective background-process reaping on SSE client disconnect — mirrors
                    # gateway/run.py's gateway-turn cleanup (#76115); this API-server surface runs its own
                    # agent lifecycle and doesn't go through TurnRunner, so it needs its own baseline.
                    # /v1/runs runs its own agent lifecycle (no TurnRunner, no _run_agent) — record turn
                    # process ownership so stop/cancel can reap only the background processes this run
                    # created (#76115).
                    _publish_turn_process_ownership(agent, effective_task_id)
                    # Registering here, once, covers every _run_agent() caller — the same reason the
                    # _ProviderAuthResolutionError handler below lives here rather than in each route. Only
                    # two callers pass ``agent_ref``, and only /v1/runs has a run_id, so neither is a usable
                    # hook for the rest. See #63529.
                    self._shutdown_interruptible_agents[id(agent)] = agent
                    # Passed only when set: a human turn keeps today's call shape.
                    author_kwargs = {"turn_author": turn_author} if turn_author is not None else {}
                    conversation_kwargs = dict(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        task_id=effective_task_id,
                        **author_kwargs,
                    )
                    if relay_metadata:
                        conversation_kwargs["relay_metadata"] = relay_metadata
                    approval_token = None
                    if approval_notify_callback is not None and approval_session_key:
                        # Same machinery as /v1/runs (_run_agent_sync): the contextvar scopes
                        # this turn's approvals to the key the resolve endpoint looks up.
                        from tools.approval import register_gateway_notify
                        from tools.approval_context import set_current_session_key
                        approval_token = set_current_session_key(approval_session_key)
                        register_gateway_notify(approval_session_key, approval_notify_callback)
                    try:
                        with notification_turn(agent, muted=muted, session_id=session_id or ""):
                            result = agent.run_conversation(**conversation_kwargs)
                    finally:
                        if approval_token is not None:
                            from tools.approval_context import reset_current_session_key
                            _api_runs._unregister_approval_notify(approval_session_key)
                            with suppress(Exception):
                                reset_current_session_key(approval_token)
                    result, usage = self._finish_turn_result(
                        agent, result, session_id, route=route, requested_runtime=requested_runtime,
                        route_source=route_source, confirmed_runtime_lock=confirmed_runtime_lock)
                    if muted and isinstance(result, dict):
                        # Project presentation only after finishing the source outcome. Keep
                        # the agent's result, transcript, failure flags and usage intact.
                        result = {**result, "_notification_presentation_suppressed": True}
                    return result, usage
                except _ProviderAuthResolutionError as exc:
                    # Typed provider-auth failure only, handled once for every caller in
                    # run.py's response shape (text, no HTTP error).
                    logger.warning("Provider resolution failed for session=%s: %s",
                                   session_id or "", exc)
                    return (
                        {"final_response": exc.user_text(), "messages": [],
                         "api_calls": 0, "tools": [],
                         **({"_notification_presentation_suppressed": True} if muted else {})},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                except Exception as exc:
                    if muted:
                        # Keep the original exception/traceback for logs and failure
                        # handling; the HTTP/SSE boundary suppresses its presentation.
                        setattr(exc, "_notification_presentation_suppressed", True)
                    raise
                finally:
                    # Turn over (any outcome): clear ownership so a late disconnect can't reap
                    # background work this turn deliberately left running.
                    if active_run_id:
                        self._active_run_agents.pop(active_run_id, None)
                    if agent is not None:
                        _clear_turn_process_ownership(agent)
                        self._shutdown_interruptible_agents.pop(id(agent), None)
                        self._memory_sessions.checkin(agent)
                        # Bind the declared key to the row the turn actually ended on
                        # (agent.session_id carries a mid-turn rotation). Opt-in per route.
                        # Record the declared conversation on the row the turn actually ended on —
                        # ``agent.session_id`` already carries a mid-turn compression rotation (#16938), so
                        # the next reply resolves the live transcript rather than its retired parent.
                        # Opt-in: only the routes that resolve their session id from the declared key
                        # (/v1/responses, /v1/runs) record one, so no other caller's rows change shape.
                        if bind_declared_conversation:
                            self._bind_declared_conversation(
                                getattr(agent, "session_id", None) or session_id, gateway_session_key)
                    clear_session_vars(tokens)
        self._activate_admitted_request()
        self._inflight_agent_runs += 1
        started_at = time.perf_counter()
        usage: Optional[Dict[str, Any]] = None
        try:
# Worker-scoped count rides along so the shutdown close gate still sees the thread
            # after this handler task is cancelled (#116535); released in the worker's finally.
            result, usage = await _api_runs._submit_api_worker(loop, _run)
            return result, usage
        finally:
            self._inflight_agent_runs -= 1
            if usage is not None:
                self._record_api_metrics(usage, time.perf_counter() - started_at)

    # -- /v1/runs, room grants, room dispatch: thin delegators (real methods: tests assert
    # __dict__ membership and patch the module-level implementations) ---------------------

    _RUN_STREAM_TTL = 300  # seconds before orphaned runs are swept
    _RUN_STATUS_TTL = 3600  # seconds to retain terminal run status for polling

    def _set_run_status(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        return _api_runs._set_run_status(self, run_id, status, **fields)

    def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop"):
        return _api_runs._make_run_event_callback(self, run_id, loop, _api_server=sys.modules[__name__])

    def _run_idempotency_scope(self, request: "web.Request") -> str:
        return _api_runs._run_idempotency_scope(self, request, _api_server=sys.modules[__name__])

    @staticmethod
    def _room_grant_token(request: "web.Request") -> str:
        return _room_grants._room_grant_token(request)

    def _room_grant_secret(self) -> bytes:
        return _room_grants._room_grant_secret(self)

    def _room_grant_claims(self, request: "web.Request", *, permission: str) -> dict[str, Any]:
        return _room_grants._room_grant_claims(self, request, permission=permission)

    def _check_run_auth(self, request: "web.Request", *, permission: str) -> "web.Response | None":
        return _api_runs._check_run_auth(self, request, permission=permission, _api_server=sys.modules[__name__])

    async def _ensure_hosted_member_session(self, dispatch: Any) -> str:
        return await _room_dispatch._ensure_hosted_member_session(self, dispatch)

    async def _normalize_room_dispatch(self, request: "web.Request", body: Any) -> tuple[Any, "web.Response | None"]:
        return await _room_dispatch._normalize_room_dispatch(self, request, body, _api_server=sys.modules[__name__])

    _handle_room_member_invitation = _room_grant_delegate("_handle_room_member_invitation")
    _handle_room_member_capabilities = _room_grant_delegate("_handle_room_member_capabilities")
    _handle_room_member_grant_refresh = _room_grant_delegate("_handle_room_member_grant_refresh")
    _handle_room_member_grant_revoke = _room_grant_delegate("_handle_room_member_grant_revoke")

    def _durable_run_status(self, request: "web.Request", run_id: str) -> Dict[str, Any] | None:
        return _api_runs._durable_run_status(self, request, run_id)

    @_admit_api_agent_request
    async def _handle_runs(self, request: "web.Request") -> "web.Response":
        return await _api_runs._handle_runs(self, request, _api_server=sys.modules[__name__])

    def _request_owns_run(self, request: "web.Request", run_id: str) -> bool:
        return _api_runs._request_owns_run(self, request, run_id)

    def _release_run_owner_if_forgotten(self, run_id: str) -> None:
        _api_runs._release_run_owner_if_forgotten(self, run_id)

    _handle_get_run = _run_route_delegate("_handle_get_run")
    _handle_run_events = _run_route_delegate("_handle_run_events")
    _handle_run_approval = _run_route_delegate("_handle_run_approval")
    _handle_steer_run = _run_route_delegate("_handle_steer_run")
    _handle_stop_run = _run_route_delegate("_handle_stop_run")

    async def _sweep_orphaned_runs(self) -> None:
        return await _api_runs._sweep_orphaned_runs(self)

    def _sweep_orphaned_runs_once(self, now: Optional[float] = None) -> None:
        return _api_runs._sweep_orphaned_runs_once(self, now)

    # -- BasePlatformAdapter interface ------------------------------------------------

    def _api_key_passes_startup_guard(self) -> bool:
        """Return True when API_SERVER_KEY is present and strong enough to start."""
        if not self._api_key:
            logger.error(
                "[%s] Refusing to start: API_SERVER_KEY is required for the API server, "
                "including loopback-only binds on %s.",
                self.name, self._host)
            return False
        try:
            from hermes_cli.auth import has_usable_secret
        except Exception as exc:
            # Fail CLOSED: "could not check" must not mean "start" on a terminal-capable endpoint.
            logger.error(
                "[%s] Refusing to start: API_SERVER_KEY strength could not be "
                "verified (%s: %s), and this endpoint dispatches "
                "terminal-capable agent work. Repair the installation before "
                "starting the API server on %s.",
                self.name, type(exc).__name__, exc, self._host)
            return False
        if not has_usable_secret(self._api_key, min_length=16):
            logger.error(
                "[%s] Refusing to start: API_SERVER_KEY is a "
                "placeholder or too short (<16 chars). This endpoint "
                "dispatches terminal-capable agent work — a guessable "
                "key is remote code execution. Generate a strong secret "
                "(e.g. `openssl rand -hex 32`) and set API_SERVER_KEY "
                "before starting the API server on %s.",
                self.name, self._host)
            return False
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False
        with self._session_db_cache_lock:
            self._session_db_cache_closed = False
        if not self._api_key_passes_startup_guard():
            # Config error, not transient: a bare ``return False`` would make the reconnect watcher
            # re-instantiate the adapter (+ sqlite connection) until EMFILE.
            self._set_fatal_error(
                # A rejected API_SERVER_KEY is a configuration error, not a transient blip — the key will
                # not become valid on its own. A bare ``return False`` makes the reconnect watcher in
                # gateway.run treat it as retryable and loop forever at the backoff cap, re-instantiating
                # the adapter (and its ResponseStore sqlite connection) every retry (#38803: ~501 leaked
                # connections / 1002 fds over 2.5 days until EMFILE took the whole gateway down).
                # Non-retryable drops it from the reconnect queue — same treatment as the port-conflict
                # guard (api_server_port_in_use). The guard already logged the specific rejection reason
                # just above.
                "api_server_key_invalid",
                "API_SERVER_KEY was rejected by the startup guard (missing, "
                "placeholder/too short, or strength unverifiable — see the "
                "error logged above). Generate a strong secret (e.g. "
                "`openssl rand -hex 32`), set API_SERVER_KEY, then "
                "`/platform resume api_server`.",
                retryable=False)
            return False
        try:
            mws = [mw for mw in (
                self._make_profile_prefix_middleware(), cors_middleware, body_limit_middleware,
                security_headers_middleware) if mw is not None]
            self._app = web.Application(middlewares=mws, client_max_size=MAX_REQUEST_BYTES)
            assert self._app is not None
            # Native routes + multiplex /p/<profile>/ mirrors (the prefix middleware validates and
            # scopes config/credentials when multiplexing is on).
            for method, path, handler in self._http_route_table():
                self._app.router.add_route(method, path, handler)
                self._app.router.add_route(method, f"/p/{{profile}}{path}", handler)
            # Registered LAST so every native mirror above wins: anything else under /p/<profile>/ is a
            # secondary profile's inbound-port platform (Twilio, LINE, Teams, ...) served on this listener.
            self._app.router.add_route("*", "/p/{profile}/{tail:.*}", self._handle_profile_ingress)
            # After native routes: Relay bootstrap shims feature-detect on this key and must
            # no-op rather than shadow the native session-control handlers.
            self._app["api_server_adapter"] = self
            if self.gateway_runner is not None:
                self._app["gateway_runner"] = self.gateway_runner
            self._track_background_task(asyncio.create_task(self._sweep_orphaned_runs()))
            # Network-accessible + unsandboxed local terminal backend = host-user RCE surface;
            # warn, don't refuse (the operator may have a firewall / strong key).
            if is_network_accessible(self._host):
                _backend = "local"
                with suppress(Exception):
                    from hermes_cli.config import load_config as _load_cfg
                    _backend = ((_load_cfg() or {}).get("terminal") or {}).get("backend", "local")
                if str(_backend).lower() == "local":
                    logger.warning(
                        "[%s] API server is network-accessible (%s) AND the "
                        "terminal backend is 'local' (unsandboxed). Agent work "
                        "dispatched through this endpoint runs as the host user "
                        "with full terminal/file access. Strongly consider a "
                        "sandboxed backend (terminal.backend: docker) and "
                        "firewalling this port to trusted networks only.",
                        self.name, self._host)

            # Plugin-registered native handlers, wired before AppRunner.setup() freezes the router.
            self._wire_plugin_handlers(self._app)
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
# Bind directly instead of probing 127.0.0.1 first — the old single-family pre-probe raced the
            # real bind and reported a TIME_WAIT socket as "in use" (#10297), failing gateway restarts for
            # up to ~60s. Platform-dependent SO_REUSEADDR and the macOS TIME_WAIT rebind live in
            # start_tcp_site; the loop below covers a predecessor still holding the port for a moment.
            try:
                # aiohttp registers a site with its runner before binding, so a failed start leaves the
                # site registered: rebuild the runner per attempt rather than reach into its internals.
                for attempt in range(_BIND_ATTEMPTS):
                    try:
                        self._site = await start_tcp_site(self._runner, self._host, self._port, log_tag=self.name)
                        break
                    except OSError as exc:
                        if exc.errno != errno.EADDRINUSE or attempt == _BIND_ATTEMPTS - 1:
                            raise
                        await self._runner.cleanup()
                        self._runner = web.AppRunner(self._app)
                        await self._runner.setup()
                        await asyncio.sleep(0.2 * (attempt + 1))
            except OSError as exc:
                await self._runner.cleanup()
                self._runner = None
                self._site = None
                if getattr(exc, "errno", None) == errno.EADDRINUSE:
                    # Config error: non-retryable, or the reconnect watcher leaks fds forever.
                    self._set_fatal_error(
                        # A port conflict is a configuration error, not a transient blip — another process
                        # holds the port for its lifetime. A bare ``return False`` makes the reconnect
                        # watcher in gateway.run treat it as retryable and loop forever at the backoff cap
                        # (observed: 1568+ retries over 5 days across multi-profile setups all defaulting
                        # to the same port, #52132), filling errors.log and leaking the adapter's ResponseStore
                        # fds each retry. Non-retryable drops it from the reconnect queue; the operator
                        # recovers with ``/platform resume api_server`` after changing the port.
                        "api_server_port_in_use",
                        f"Port {self._port} already in use. Set "
                        f"platforms.api_server.port in config.yaml to a "
                        f"different value, then `/platform resume api_server`.",
                        retryable=False)
                logger.error(
                    "[%s] Could not bind %s:%d: %s. Set a different port in "
                    "config.yaml: platforms.api_server.port",
                    self.name, self._host, self._port, exc)
                return False
            from gateway.platforms.shared_ingress import bound_listener_port, bound_site_endpoints, listener_base_url
            self._bound_listener_endpoints = bound_site_endpoints(self._site, self._host, self._port)
            listener_port = bound_listener_port(self._bound_listener_endpoints, self._port)
            self._mark_connected(listener_base=listener_base_url(self._host, listener_port))
            # Publish a metrics-bearing snapshot at bind and keep it fresh: the
            # heartbeat loop updates last_heartbeat/metrics_today (#52323).
            self._publish_runtime_status()
            self._track_background_task(asyncio.create_task(self._heartbeat_loop()))
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                self.name, self._host, listener_port, self._model_name)
            return True
        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp server and release every owned resource, including the ResponseStore
        connection (the reconnect loop builds a fresh adapter per retry; leaked fds hit EMFILE).

        Without this, every adapter instance leaks 2 file descriptors (the database file and its WAL
        sidecar) — the reconnect loop in ``gateway.run`` constructs a fresh adapter on every retry, so 2
        fds/retry × 300s backoff cap ≈ 12 fds/hour, which exhausts the default 2560 fd limit after ~12h of
        failed reconnects and turns the whole gateway into a zombie (OSError: [Errno 24] Too many open
        files, #37011).
        """
        self._mark_disconnected()
        # getattr: disconnect() tolerates bare __new__ fixtures (pinned in test_api_server_run_idempotency).
        routed = getattr(self, "_response_stores", {})
        stores = [s for s in (getattr(self, "_response_store", None), *list(routed.values())) if s is not None]
        routed.clear()
        for store in stores:
            try:
                store.close()
            except Exception:
                logger.debug("Failed to close response store for %s", self.name, exc_info=True)
        _api_runs._close_run_state(self)
        with suppress(Exception):
            await asyncio.to_thread(self._memory_sessions.close_all)
        try:
            if self._site:
                await self._site.stop()
                self._site = None
            if self._runner:
                await self._runner.cleanup()
                self._runner = None
        finally:
            self._close_cached_session_dbs()
            self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Not used — the HTTP request/response cycle handles delivery directly."""
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {"name": "API Server", "type": "api", "host": self._host, "port": self._port}
