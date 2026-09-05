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
from contextvars import ContextVar
from functools import wraps
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    "admin_config_rw": False, "jobs_admin": False, "memory_write_api": False,
    "skills_api": True, "audio_api": False, "realtime_voice": True,
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
    ("artifact_download", ("GET", "/v1/artifacts/download/{artifact_id}")),
    ("realtime_voice", ("GET", "/v1/audio/converse")))
_BROWSER_CONTROL_WS_PROTOCOL = "hermes-browser-control-v1"
_BROWSER_CONTROL_TICKET_PROTOCOL_PREFIX = "hermes-browser-control-ticket."


def _approval_event_choices(*, smart_denied: bool, allow_session: bool, allow_permanent: bool) -> list[str]:
    if smart_denied or not allow_session:
        return ["once", "deny"]
    return ["once", "session", "always", "deny"] if allow_permanent else ["once", "session", "deny"]


try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms import api_server_room_dispatch as _room_dispatch
from gateway.platforms import api_server_room_grants as _room_grants
from gateway.platforms import api_server_converse as _api_server_converse
from gateway.platforms import api_server_runs as _api_runs
from gateway.platforms.api_server_openai_routes import OpenAICompatRoutesMixin
from gateway.platforms.base import (
    MEDIA_TAG_CLEANUP_RE, BasePlatformAdapter, SendResult, is_network_accessible, validate_media_delivery_path)
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
    """Canonical Hermes version: ``hermes_cli.__version__`` (dist-info can be stale on
    source checkouts), then distribution metadata, then "dev". Never raises."""
    with suppress(Exception):
        from hermes_cli import __version__
        return __version__
    try:
        from importlib.metadata import version
        return version("hermes-agent")
    except Exception:
        return "dev"


# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = 10_000_000  # 10 MB — accommodates long agent conversations with tool calls
CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0
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


def _sse_frame(data: Any, *, event: str = None, ensure_ascii: bool = True) -> bytes:
    """Encode one SSE frame (``event:`` line if given, then ``data: <json>\n\n``) for every
    SSE writer. ``ensure_ascii=False`` keeps raw non-ASCII on the wire."""
    prefix = f"event: {event}\n" if event else ""
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
    "api_key", "base_url", "provider", "api_mode", "command", "args", "credential_pool", "max_tokens")


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
    model_cfg = _get_model_config()
    max_tokens = None
    env_max_tokens = os.environ.get("HERMES_MAX_TOKENS")
    if env_max_tokens:
        with suppress(ValueError, TypeError):
            max_tokens = int(env_max_tokens)
    elif isinstance(model_cfg, dict):
        cfg_max_tokens = model_cfg.get("max_tokens")
        if isinstance(cfg_max_tokens, int):
            max_tokens = cfg_max_tokens
    if max_tokens is None:
        runtime_max_tokens = runtime.get("max_output_tokens")
        if isinstance(runtime_max_tokens, int) and runtime_max_tokens > 0:
            max_tokens = runtime_max_tokens
    return {
        **{k: runtime.get(k) for k in ("api_key", "base_url", "provider", "api_mode", "command")},
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"), "max_tokens": max_tokens}


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
        target=_reap_gateway_turn_processes, args=(process_task_id, process_baseline),
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
            request_hard_interrupt(agent, reason)
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
    "Access-Control-Allow-Headers": "Authorization, Content-Type, Idempotency-Key"}
_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "no-referrer"}

if AIOHTTP_AVAILABLE:
    def _is_voice_key_ws_upgrade(request) -> bool:
        """True for a ``/v1/audio/converse`` WebSocket upgrade carrying a
        ``hermes-key.<KEY>`` subprotocol.

        Such handshakes authenticate with an explicit bearer key in
        ``Sec-WebSocket-Protocol`` (validated constant-time by the handler), not an
        ambient cookie. The Origin/CSRF guard exists to protect credentials the
        browser attaches automatically — a hostile page can set any subprotocol but
        cannot produce a key it lacks — so it adds nothing on top of the key here,
        while it *does* block legitimate browser-context clients (Electron renderers
        send an unsuppressible ``Origin: null``). The converse route accepts no
        cookie auth, so relaxing Origin for key-bearing upgrades is safe.

        PRESENCE, NOT VALIDITY (on purpose): the exemption keys off the mere presence
        of a ``hermes-key.`` subprotocol, not whether the key is correct. Validating
        the key here would duplicate auth into the wrong layer — two constant-time
        paths to keep in lockstep, and a wrong key would 403-as-origin instead of
        401-as-auth. The only cost of presence-based matching is that a hostile page
        can reach the handler's constant-time key check and burn a 401 — but any
        non-browser client can already do that by omitting Origin entirely, so it
        concedes nothing.

        INVARIANT: this holds ONLY while ``/v1/audio/converse`` authenticates with
        the explicit key alone (no cookie/session fallback). If ambient auth is ever
        added to that handler, this exemption silently becomes a Cross-Site
        WebSocket Hijacking hole — ``test_cookie_is_not_a_credential`` guards it."""
        if request.headers.get("Upgrade", "").lower() != "websocket":
            return False
        if "upgrade" not in request.headers.get("Connection", "").lower():
            return False
        protocols = request.headers.get("Sec-WebSocket-Protocol", "")
        return any(p.strip().startswith("hermes-key.") for p in protocols.split(","))

    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers for explicitly allowed origins; handle OPTIONS preflight."""
        adapter = request.app.get("api_server_adapter")
        origin = request.headers.get("Origin", "")
        cors_headers = None
        if adapter is not None:
            if not adapter._origin_allowed(origin) and not _is_voice_key_ws_upgrade(request):
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

    @web.middleware
    async def body_limit_middleware(request, handler):
        """Reject overly large request bodies early based on Content-Length."""
        if request.method in {"POST", "PUT", "PATCH"}:
            cl = request.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > MAX_REQUEST_BYTES:
                        return _error_response("Request body too large.", 413, code="body_too_large")
                except ValueError:
                    return _error_response("Invalid Content-Length header.", 400, code="invalid_content_length")
        try:
            return await handler(request)
        except web.HTTPRequestEntityTooLarge:
            # client_max_size tripped mid-read (chunked bodies carry no Content-Length): a
            # proper 413, not the handler's 400 "Invalid JSON".
            return _error_response("Request body too large.", 413, code="body_too_large")

    @web.middleware
    async def security_headers_middleware(request, handler):
        """Add security headers to all responses (including errors)."""
        response = await handler(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response
else:
    cors_middleware = body_limit_middleware = security_headers_middleware = None  # type: ignore

_MEDIA_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp", ".bmp": "image/bmp"}
_MEDIA_IMG_EXT = set(_MEDIA_MIME)
_MEDIA_DATA_URL_MAX_BYTES = 5 * 1024 * 1024  # skip images larger than 5MB


def _resolve_media_to_data_urls(text: str) -> str:
    """Replace ``MEDIA:<path>`` tags with inline base64 data URLs (remote frontends can't read
    server paths); non-image/unreadable paths stay untouched. Security: the shared
    ``MEDIA_TAG_CLEANUP_RE`` anchor + ``validate_media_delivery_path`` denylist — a bare-token
    match would let a traversal path in the reply exfiltrate any readable image."""
    if not text or "MEDIA:" not in text:
        return text
    import base64

    def _to_data_url(path_str: str) -> Optional[str]:
        # validate_media_delivery_path() strips wrapping quotes/trailing punctuation itself.
        safe_path = validate_media_delivery_path(path_str)
        p = Path(safe_path) if safe_path else None
        suffix = p.suffix.lower() if p else ""
        if suffix not in _MEDIA_IMG_EXT:
            return None
        try:
            if p.stat().st_size > _MEDIA_DATA_URL_MAX_BYTES:
                return None
            b64 = base64.b64encode(p.read_bytes()).decode()
        except OSError:
            return None
        return f"![image](data:{_MEDIA_MIME[suffix]};base64,{b64})"

    def _repl(m: "re.Match[str]") -> str:
        return _to_data_url(m.group("path")) or m.group(0)
    try:
        return MEDIA_TAG_CLEANUP_RE.sub(_repl, text)
    except Exception:
        return text


def _redact_api_error_text(value: Any, *, limit: int | None = None) -> str:
    """Redact API-bound error text before it crosses the HTTP boundary."""
    redacted = redact_sensitive_text(str(value), force=True)
    return redacted[:limit] if limit is not None else redacted


def _openai_error(message: str, err_type: str = "invalid_request_error", param: str = None, code: str = None) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {"error": {
        "message": _redact_api_error_text(message), "type": err_type, "param": param, "code": code}}


def _error_response(
    message: str, status: int, *, err_type: str = "invalid_request_error",
    param: str = None, code: str = None, headers: Optional[Dict[str, str]] = None,
) -> "web.Response":
    """``web.json_response(_openai_error(...), status=...)`` in one call."""
    return web.json_response(_openai_error(message, err_type, param, code), status=status, headers=headers)


def _invalid_request(message: str) -> "web.Response":
    """400 with the bare ``{message, type}`` envelope the OpenAI-compatible validators use."""
    return web.json_response({"error": {"message": message, "type": "invalid_request_error"}}, status=400)


_api_agent_request_reservation: ContextVar[Optional[dict[str, bool]]] = ContextVar(
    "api_agent_request_reservation", default=None)


def _admit_api_agent_request(handler):
    """Reserve an authenticated API turn before its handler first awaits: drain check +
    reservation in one non-awaiting block so a request admitted just before shutdown can't go
    invisible while parsing its body. The mutable reservation releases the slot exactly once."""
    @wraps(handler)
    async def _wrapped(self, request, *args, **kwargs):
        auth_err = (
            self._check_run_auth(request, permission="dispatch")
            if _api_runs._uses_room_run_auth(self, request)
            else self._check_auth(request))
        if auth_err:
            return auth_err
        draining = self._draining_response()
        if draining is not None:
            return draining
        reservation = {"active": True}
        token = _api_agent_request_reservation.set(reservation)
        self._pending_agent_requests += 1
        try:
            return await handler(self, request, *args, **kwargs)
        finally:
            _release_pending_api_work(self, reservation)
            _api_agent_request_reservation.reset(token)
    return _wrapped


def _release_pending_api_work(adapter, reservation: dict[str, bool]) -> None:
    """Release a pending-work reservation exactly once."""
    if reservation["active"]:
        reservation["active"] = False
        adapter._pending_agent_requests = max(0, adapter._pending_agent_requests - 1)


def _require_auth(handler):
    """Run ``self._check_auth`` first and return its 401 instead of the handler."""
    @wraps(handler)
    async def _wrapped(self, request, *args, **kwargs):
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        return await handler(self, request, *args, **kwargs)
    return _wrapped


@contextmanager
def _reserve_pending_api_work(adapter):
    """Keep externally-triggered background work visible across awaits; a handler may detach
    the reservation to a task whose done callback then owns release."""
    reservation = {"active": True, "detached": False}
    adapter._pending_agent_requests += 1
    try:
        yield reservation
    finally:
        if not reservation["detached"]:
            _release_pending_api_work(adapter, reservation)


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
                self._store[key] = {"resp": resp, "fp": fingerprint, "ts": time.time()}
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
    subset = {k: body.get(k) for k in keys}
    return hashlib.sha256(repr(subset).encode("utf-8")).hexdigest()


def _derive_chat_session_id(system_prompt: Optional[str], first_user_message: str) -> str:
    """Stable session id from the system prompt + first user message (constant across all
    turns of an Open WebUI-style conversation), so one Hermes session/sandbox is reused."""
    seed = f"{system_prompt or ''}\n{first_user_message}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"api-{digest}"


_CRON_AVAILABLE = False
try:
    from cron.jobs import (
        list_jobs as _cron_list, get_job as _cron_get, update_job as _cron_update,
        remove_job as _cron_remove, pause_job as _cron_pause, resume_job as _cron_resume,
        trigger_job as _cron_trigger)
    from cron.scheduler import (
        CronSchedulerRegistrationError as _CronSchedulerRegistrationError,
        create_job_with_scheduler_registration as _cron_create)
    _CRON_AVAILABLE = True
except ImportError:
    _cron_list = _cron_get = _cron_create = _cron_update = None
    _cron_remove = _cron_pause = _cron_resume = _cron_trigger = None

    class _CronSchedulerRegistrationError(RuntimeError):
        pass


def _notify_cron_provider_jobs_changed() -> None:
    """Best-effort notify of the active cron provider after a REST mutation (built-in: no-op)."""
    with suppress(Exception):
        from cron.scheduler import _notify_provider_jobs_changed
        _notify_provider_jobs_changed()


# Defense-in-depth parity with the cronjob tool's prompt injection scan (the REST
# endpoints are authenticated, so this is not the trust boundary). Optional import:
# a missing scanner must not disable the cron REST API.
try:
    from tools.cronjob_tools import _scan_cron_prompt as _scan_cron_prompt
except Exception:  # pragma: no cover - scanner is optional hardening
    _scan_cron_prompt = None


class _ProviderAuthResolutionError(RuntimeError):
    """Provider credential resolution failed. Typed so callers never mislabel other
    RuntimeErrors from run_conversation() (e.g. a closed OpenAI client) as auth failures."""


class _SessionEventQueue:
    """Ordered SSE event queue for one /api/sessions/{id}/chat/stream run. ``payload`` stamps
    session_id/run_id/seq/ts; ``enqueue`` is executor-thread safe (hops onto the owning loop)."""

    def __init__(self, session_id: str, run_id: str):
        self.loop = asyncio.get_running_loop()
        self.queue: "asyncio.Queue[Optional[tuple[str, Dict[str, Any]]]]" = asyncio.Queue()
        self.session_id = session_id
        self.run_id = run_id
        self.seq = 0

    def payload(self, name: str, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        self.seq += 1
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("seq", self.seq)
        payload.setdefault("ts", time.time())
        return name, payload

    def enqueue(self, name: str, payload: Dict[str, Any]) -> None:
        event = self.payload(name, payload)
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        with suppress(RuntimeError):
            if running_loop is self.loop:
                self.queue.put_nowait(event)
            else:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, event)


def _room_grant_delegate(name: str):
    """Adapter method forwarding to ``api_server_room_grants.<name>`` (looked up at call time so
    test patches on that module take effect) with this module's error/profile bindings."""
    async def _handler(self, request: "web.Request") -> "web.Response":
        return await getattr(_room_grants, name)(
            self, request, _openai_error=_openai_error, _api_request_profile=_api_request_profile)
    _handler.__name__ = name
    return _handler


def _run_route_delegate(name: str):
    """Adapter method forwarding to ``api_server_runs.<name>`` (call-time lookup) with this
    module's namespace as ``_api_server``."""
    async def _handler(self, request: "web.Request") -> "web.StreamResponse":
        return await getattr(_api_runs, name)(self, request, _api_server=sys.modules[__name__])
    _handler.__name__ = name
    return _handler


class APIServerAdapter(OpenAICompatRoutesMixin, BasePlatformAdapter):
    """aiohttp server routing OpenAI-format requests through hermes-agent's AIAgent."""

    # Stateless request/response (``send()`` is a stub): async-delivery tools must not promise
    # delivery here, and a resumed turn completes the work rather than asking.
    supports_async_delivery: bool = False
    # Same statelessness applies to the startup auto-resume prompt: no client is waiting to answer "session
    # restored — what next?", so a resumed turn should complete the interrupted work rather than acknowledge
    # (#57056).
    interactive_resume: bool = False

    # Admission-gated OpenAI-compatible entry points (bodies live in the mixin).
    _handle_chat_completions = _admit_api_agent_request(OpenAICompatRoutesMixin._handle_chat_completions)
    _handle_responses = _admit_api_agent_request(OpenAICompatRoutesMixin._handle_responses)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        raw_port = extra.get("port")
        if raw_port is None:
            raw_port = os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))
        self._port: int = _coerce_port(raw_port, DEFAULT_PORT)
        self._api_key: str = extra.get("key", _get_scoped_secret("API_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")))
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("API_SERVER_MODEL_NAME", "")))
        # alias (client "model") -> {model, provider?, api_key? (UPSTREAM, never logged), base_url?}
        self._model_routes: Dict[str, Dict[str, Any]] = self._parse_model_routes(extra.get("model_routes"))
        # Opt-in bare ``model`` passthrough on OpenAI-compatible surfaces (generic clients
        # hardcode "gpt-4o" etc., hence off by default).
        # Off by default: generic OpenAI clients routinely hardcode model names ("gpt-4o", ...), and
        # existing deployments rely on those falling back to the gateway default rather than switching the
        # executing model. Requests that send an explicit ``provider`` — and the Hermes-native session-chat
        # and /v1/runs endpoints — are always honored regardless of this flag. (Idea credit: PR #22825 by
        # @mssteuer.)
        self._direct_model_requests: bool = _coerce_request_bool(
            extra.get("direct_model_requests"), default=False)
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        _api_runs._initialize_run_state(self, store_factory=RunIdempotencyStore)
        self._session_db: Optional[Any] = None  # explicit override (tests/manual wiring)
        self._session_dbs: Dict[str, Any] = {}  # per-profile-home SessionDB cache
        self._session_db_cache_lock = threading.Lock()
        self._session_db_cache_closed = False
        # Last-known-good model per gateway_session_key ("*" = process-wide; never session_id,
        # which is per request -> unbounded). Recovers a transient empty model resolution.
        self._last_resolved_model: Dict[str, str] = {}
        self._session_db_lock: Optional[asyncio.Lock] = None  # single-flight for lazy init
        self._max_concurrent_runs: int = self._resolve_max_concurrent_runs()  # 0 disables
        # In-flight _run_agent() turns (/v1/runs tracks its own via _active_run_tasks).
        # Concurrency cap shared across all agent-serving endpoints (/v1/chat/completions, /v1/responses,
        # /v1/runs). Read from config.yaml gateway.api_server.max_concurrent_runs; 0 disables the cap.
        # Bounds CPU / memory / upstream-LLM-quota exhaustion from a request flood (#7483).
        self._inflight_agent_runs: int = 0
        # Every agent inside _run_agent() for shutdown interrupt, keyed by id() (the strong ref
        # keeps the id() from recycling); distinct from the run_id-keyed _active_run_agents.
        self._shutdown_interruptible_agents: Dict[int, Any] = {}
        self.gateway_runner: Optional[Any] = None  # set by gateway/run.py
        # Admitted requests not yet in agent bookkeeping, so shutdown drain counts them.
        self._pending_agent_requests: int = 0
        # Shared broker; this adapter maps HTTP registration + controller WS onto it.
        self._browser_control_broker = get_browser_control_broker()
        # One-shot artifact transport: lazy per-profile stores + limiter (tests inject).
        self._browser_control_artifacts: Dict[str, ArtifactStore] = {}
        self._browser_control_artifact_limiter: Optional[ArtifactRateLimiter] = None

    def active_agent_work_count(self) -> int:
        """All live agent work: pending admissions + in-flight turns + live /v1/runs tasks
        (task-based, since ``_active_run_agents`` has a queued-before-agent gap)."""
        try:
            return (int(getattr(self, "_pending_agent_requests", 0))
                    + int(self._inflight_agent_runs)
                    + sum(not task.done() for task in self._active_run_tasks.values()))
        except Exception:
            return 0

    def interrupt_active_runs(self, reason: str) -> int:
        """Interrupt every adapter-owned agent during shutdown (they are not in
        ``GatewayRunner._running_agents``): exactly the set the drain waits on. Returns count."""
        # Dedupe by identity: an agent in both registries must be interrupted once.
        agents = {id(agent): agent for agent in (
            *self._active_run_agents.values(), *self._shutdown_interruptible_agents.values())
            if agent is not None}
        interrupted = 0
        for agent in agents.values():
            try:
                if request_hard_interrupt(agent, reason):
                    interrupted += 1
            except Exception as exc:
                logger.debug("[api_server] failed interrupting active agent: %s", exc)
        return interrupted

    @staticmethod
    def _gateway_is_draining() -> bool:
        """Whether the owning gateway currently refuses new agent turns."""
        try:
            from gateway.run import _gateway_runner_ref
            runner = _gateway_runner_ref()
            return bool(runner and (getattr(runner, "_draining", False)
                                    or getattr(runner, "_external_drain_active", False)))
        except Exception:
            return False

    def _draining_response(self) -> Optional["web.Response"]:
        """Return a retryable response while the gateway drains existing work."""
        if not self._gateway_is_draining():
            return None
        return _error_response(
            "Gateway is draining existing work; retry shortly.", 503, code="gateway_draining",
            headers={"Retry-After": "1"})

    def _activate_admitted_request(self) -> None:
        """Transfer this request's drain reservation to agent bookkeeping."""
        reservation = _api_agent_request_reservation.get()
        if reservation:
            _release_pending_api_work(self, reservation)

    def _readiness_work_counts(self) -> tuple[int, int, int]:
        """Return bounded work counts from each subsystem's public state."""
        # "stopping" is not terminal: executor work continues until the agent notices.
        active_api_runs = sum(
            1 for status in self._run_statuses.values()
            if status.get("status") in {"queued", "running", "waiting_for_approval", "stopping"})
        process_depth = 0
        active_delegations = 0
        with suppress(Exception):
            from tools.process_registry import process_registry
            process_depth = process_registry.completion_queue.qsize()
        with suppress(Exception):
            from tools.async_delegation import active_count
            active_delegations = active_count()
        return active_api_runs, process_depth, active_delegations

    @staticmethod
    def _parse_cors_origins(value: Any) -> tuple[str, ...]:
        """Normalize configured CORS origins into a stable tuple."""
        if not value:
            return ()
        items = (
            value.split(",") if isinstance(value, str)
            else value if isinstance(value, (list, tuple, set)) else [str(value)])
        return tuple(str(item).strip() for item in items if str(item).strip())

    @staticmethod
    def _resolve_max_concurrent_runs() -> int:
        """gateway.api_server.max_concurrent_runs (0 disables; default 10; negatives -> 0)."""
        default = 10
        try:
            from hermes_cli.config import cfg_get, load_config
            raw = cfg_get(
                load_config(), "gateway", "api_server", "max_concurrent_runs", default=default)
            value = int(raw)
        except Exception:
            return default
        return max(0, value)

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        """Advertised /v1/models name: explicit override > active profile name > "hermes-agent"
        (precedence owned by ``hermes_cli.model_switch.resolve_effective_model``)."""
        from hermes_cli.model_switch import resolve_effective_model
        profile_name = ""
        with suppress(Exception):
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile and profile not in {"default", "custom"}:
                profile_name = profile
        return resolve_effective_model(explicit, profile_name, "hermes-agent")

    def _cors_headers_for_origin(self, origin: str) -> Optional[Dict[str, str]]:
        """Return CORS headers for an allowed browser origin."""
        if not origin or not self._cors_origins:
            return None
        if "*" in self._cors_origins:
            return {**_CORS_HEADERS, "Access-Control-Allow-Origin": "*", "Access-Control-Max-Age": "600"}
        if origin not in self._cors_origins:
            return None
        return {**_CORS_HEADERS, "Access-Control-Allow-Origin": origin, "Vary": "Origin",
                "Access-Control-Max-Age": "600"}

    def _origin_allowed(self, origin: str) -> bool:
        """Allow non-browser clients and explicitly configured browser origins."""
        return not origin or "*" in self._cors_origins or origin in self._cors_origins

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
        with suppress(Exception):
            peer = request.transport.get_extra_info("peername") if request.transport else None
            if isinstance(peer, (tuple, list)) and peer:
                peer_ip = str(peer[0])
        return {
            "remote": self._clean_log_value(getattr(request, "remote", "") or peer_ip),
            "peer_ip": self._clean_log_value(peer_ip),
            "forwarded_for": self._clean_log_value(request.headers.get("X-Forwarded-For", "")),
            "real_ip": self._clean_log_value(request.headers.get("X-Real-IP", "")),
            "method": self._clean_log_value(request.method, max_len=16),
            "path": self._clean_log_value(request.path_qs, max_len=500),
            "user_agent": self._clean_log_value(request.headers.get("User-Agent", ""), max_len=300)}

    def _request_audit_log_suffix(self, request: "web.Request") -> str:
        ctx = self._request_audit_context(request)
        fields = [f"{key}={value!r}" for key, value in ctx.items() if value]
        return " ".join(fields) if fields else "source='unknown'"

    def _cron_origin_from_request(self, request: "web.Request") -> Dict[str, str]:
        """Persist safe API source metadata on cron jobs created over HTTP."""
        ctx = self._request_audit_context(request)
        origin = {"platform": "api_server", "chat_id": "api"}
        for ctx_key, origin_key in (("remote", "source_ip"), ("peer_ip", "peer_ip"),
                                    ("forwarded_for", "forwarded_for"), ("real_ip", "real_ip"),
                                    ("user_agent", "user_agent")):
            if ctx.get(ctx_key):
                origin[origin_key] = ctx[ctx_key]
        return origin

    def _expected_api_key(self) -> str:
        """Return the API key authorized for the URL-selected profile."""
        profile = _api_request_profile.get()
        if not profile or profile == "default":
            return self._api_key
        try:
            from agent.secret_scope import get_secret
            from hermes_cli.auth import has_usable_secret
            key = get_secret("API_SERVER_KEY", "") or ""
            return key if has_usable_secret(key, min_length=16) else ""
        except Exception as exc:
            # Fail closed; never log the key or exception text.
            logger.warning(
                "Failed to resolve a usable profile-scoped API_SERVER_KEY for %r: %s",
                profile, type(exc).__name__)
            return ""

    @staticmethod
    def _auth_failed_response() -> "web.Response":
        return web.json_response(
            {"error": {"message": "Invalid gateway API key (API_SERVER_KEY)", "type": "gateway_auth_error",
                       "code": "gateway_auth_failed"}},
            status=401)

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """Validate the Bearer token; None when OK, else a 401. The no-key branch (connect()
        refuses to start without API_SERVER_KEY) exists for tests/manual wiring on the default
        listener only; named profiles fail closed rather than inherit the owner's key."""
        profile = _api_request_profile.get()
        expected_key = self._expected_api_key()
        if not expected_key:
            if not (profile and profile != "default"):
                return None
            logger.warning(
                "API server rejected request for profile %r: no profile-scoped "
                "API_SERVER_KEY is configured; %s",
                profile, self._request_audit_log_suffix(request))
            return self._auth_failed_response()
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            # Compare as bytes: compare_digest raises TypeError on non-ASCII str, and the
            # token is raw client input — a stray byte must 401, not 500.
            if hmac.compare_digest(token.encode(), expected_key.encode()):
                return None
        logger.warning("API server rejected invalid API key: %s", self._request_audit_log_suffix(request))
        return self._auth_failed_response()

    @staticmethod
    def _normalize_callback_platform(value: str) -> str:
        normalized = (value or "").strip().lower().replace("-", "_")
        return normalized if re.fullmatch(r"[a-z0-9_]+", normalized) else ""

    def _get_platform_callback_adapter(
        self, request: "web.Request", platform_name: str) -> Optional[Any]:
        injected = request.app.get("platform_event_adapters")
        adapter = injected.get(platform_name) if isinstance(injected, dict) else None
        if adapter is None:
            adapter = request.app.get(f"{platform_name}_adapter")
        if adapter is not None:
            return adapter
        runner = self.gateway_runner or request.app.get("gateway_runner")
        adapters = getattr(runner, "adapters", None)
        if not adapters:
            return None
        try:
            return adapters.get(Platform(platform_name))
        except Exception:
            for platform, candidate in adapters.items():
                if getattr(platform, "value", platform) == platform_name:
                    return candidate
        return None

    async def _handle_platform_event_callback(self, request: "web.Request") -> "web.Response":
        platform_name = self._normalize_callback_platform(request.match_info.get("platform", ""))
        if not platform_name:
            return _error_response("Invalid platform name", 400, code="invalid_platform")
        adapter = self._get_platform_callback_adapter(request, platform_name)
        if adapter is None:
            return _error_response("Platform adapter is not connected", 503, code="platform_unavailable")
        verifier = getattr(adapter, "verify_http_event_request", None)
        dispatcher = getattr(adapter, "dispatch_http_event", None)
        if verifier is None or dispatcher is None:
            return _error_response(
                "Platform adapter does not support HTTP events", 503, code="platform_http_events_unsupported")
        auth_header = request.headers.get("Authorization", "")
        try:
            ok, code = await _call_verifier(verifier, auth_header)
        except Exception:
            # Fail closed: a crashing verifier must never admit the event.
            logger.exception("Platform HTTP event verifier failed for %s", platform_name)
            ok, code = False, "platform_event_verifier_error"
        if not ok:
            return _error_response(
                "Invalid platform event authorization", 401, code=code or "invalid_platform_event_authorization")
        try:
            payload = await request.json()
        except Exception:
            return _error_response("Invalid JSON in platform event", 400, code="invalid_json")
        if not isinstance(payload, dict):
            return _error_response("Platform event must be a JSON object", 400, code="invalid_request")
        try:
            result = await dispatcher(payload)
        except Exception:
            logger.exception("Platform HTTP event dispatch failed for %s", platform_name)
            return _error_response(
                "Platform event dispatch failed", 500, err_type="server_error",
                code="platform_event_dispatch_failed")
        return web.json_response(result if isinstance(result, dict) else {})

    # -- Multi-profile multiplexing (/p/<profile>/...) --------------------------------

    def _resolve_request_profile(self, request: "web.Request"):
        """Resolve + validate the /p/<profile>/ prefix: ``None`` (no prefix, or multiplexing
        off and the prefix names this gateway's own profile), the served profile name, or
        ``_PROFILE_REJECTED`` (-> 404). Fail closed: a foreign prefix must never be ignored."""
        profile = (request.match_info.get("profile") or "").strip()
        if not profile:
            return None
        cfg = getattr(self.gateway_runner, "config", None)
        if not getattr(cfg, "multiplex_profiles", False):
            return None if _prefix_names_served_profile(profile) else _PROFILE_REJECTED
        try:
            from hermes_cli.profiles import profiles_to_serve
            served = {
                name for name, _ in profiles_to_serve(
                    multiplex=True, profile_allowlist=getattr(cfg, "multiplex_profile_allowlist", None))}
        except Exception:
            return _PROFILE_REJECTED
        return profile if profile in served else _PROFILE_REJECTED

    @staticmethod
    def _profile_scope(profile: Optional[str]):
        """Enter the multiplex profile runtime scope, or a no-op when unset. No prefix AND
        multiplexing active enters the DEFAULT profile's scope (an unscoped run would raise
        UnscopedSecretError on its first credential read); single-profile gateways no-op.

        Single-profile gateways keep the no-op — ``get_secret`` falls through to ``os.environ`` there,
        unchanged. See #61276.
        """
        if not profile:
            with suppress(Exception):
                from agent.secret_scope import is_multiplex_active
                if is_multiplex_active():
                    from gateway.run import _profile_runtime_scope
                    from hermes_constants import get_hermes_home
                    return _profile_runtime_scope(get_hermes_home())
            return nullcontext()
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir
        return _profile_runtime_scope(get_profile_dir(profile))

    def _make_profile_prefix_middleware(self):
        """Reject unknown /p/<profile>/ prefixes and scope the request home."""

        @web.middleware
        async def profile_prefix_middleware(request: "web.Request", handler):
            profile = self._resolve_request_profile(request)
            if profile is _PROFILE_REJECTED:
                return web.json_response({"error": "Unknown or unconfigured profile"}, status=404)
            token = _api_request_profile.set(profile)
            try:
                with self._profile_scope(profile):
                    resolved_profile = profile or "default"
                    principal_token = _api_request_browser_control_principal.set(
                        self._derive_browser_control_principal(resolved_profile))
                    family_token = _api_request_browser_control_transport_family.set(
                        self._browser_control_transport_family(request))
                    try:
                        return await handler(request)
                    finally:
                        _api_request_browser_control_transport_family.reset(family_token)
                        _api_request_browser_control_principal.reset(principal_token)
            finally:
                _api_request_profile.reset(token)
        return profile_prefix_middleware

    def _http_route_table(self) -> List[tuple]:
        """(method, path, handler) rows registered by ``connect()`` (a method so multiplex tests
        can assert the /p/<profile>/ mirrors without a listener)."""
        routes: List[tuple] = [
            ("GET", "/health", self._handle_health),
            ("GET", "/health/detailed", self._handle_health_detailed),
            ("GET", "/v1/health", self._handle_health),
            ("GET", "/v1/models", self._handle_models),
            ("GET", "/api/model/options", self._handle_model_options),
            ("GET", "/v1/capabilities", self._handle_capabilities),
            # Browser-control (gated on browser.extension_control.enabled + API key): POST
            # mints a short-lived ticket, WS consumes it; artifacts are bounded + scope-bound.
            ("POST", "/v1/browser-control/register", self._handle_browser_control_register),
            ("GET", "/v1/browser-control/ws", self._handle_browser_control_ws),
            ("POST", "/v1/artifacts/upload", self._handle_artifact_upload),
            ("GET", "/v1/artifacts/download/{artifact_id}", self._handle_artifact_download),
            ("GET", "/v1/skills", self._handle_skills),
            ("GET", "/v1/toolsets", self._handle_toolsets),
            ("GET", "/api/sessions", self._handle_list_sessions),
            ("POST", "/api/sessions", self._handle_create_session),
            ("GET", "/api/sessions/{session_id}", self._handle_get_session),
            ("PATCH", "/api/sessions/{session_id}", self._handle_patch_session),
            ("DELETE", "/api/sessions/{session_id}", self._handle_delete_session),
            ("GET", "/api/sessions/{session_id}/messages", self._handle_session_messages),
            ("POST", "/api/sessions/{session_id}/fork", self._handle_fork_session),
            ("POST", "/api/sessions/{session_id}/chat", self._handle_session_chat),
            ("POST", "/api/sessions/{session_id}/chat/stream", self._handle_session_chat_stream),
            ("POST", "/api/sessions/{session_id}/model", self._handle_session_model_lock),
            ("POST", "/v1/chat/completions", self._handle_chat_completions),
            ("POST", "/v1/responses", self._handle_responses),
            ("GET", "/v1/responses/{response_id}", self._handle_get_response),
            ("DELETE", "/v1/responses/{response_id}", self._handle_delete_response),
            # Platform event ingress: authenticated by the target adapter's own verifier,
            # NOT API_SERVER_KEY (external platforms hold no API server key).
            ("POST", "/api/platforms/{platform}/events", self._handle_platform_event_callback),
            ("GET", "/api/jobs", self._handle_list_jobs),
            ("POST", "/api/jobs", self._handle_create_job),
            ("GET", "/api/jobs/{job_id}", self._handle_get_job),
            ("PATCH", "/api/jobs/{job_id}", self._handle_update_job),
            ("DELETE", "/api/jobs/{job_id}", self._handle_delete_job),
            ("POST", "/api/jobs/{job_id}/pause", self._handle_pause_job),
            ("POST", "/api/jobs/{job_id}/resume", self._handle_resume_job),
            ("POST", "/api/jobs/{job_id}/run", self._handle_run_job)]
        routes.extend(_room_grants._http_routes(self))
        routes.extend(_api_runs._http_routes(self))
        routes.extend(_api_server_converse._http_routes(self))
        if _CRON_AVAILABLE:
            # Chronos fire webhook (NAS -> agent): authenticated by a NAS-minted JWT.
            routes.append(("POST", "/api/cron/fire", self._handle_cron_fire))
        return routes

    # -- Session header helpers -------------------------------------------------------

    # Cap on session headers: above any realistic channel id, safe for Honcho / state.db.
    _MAX_SESSION_HEADER_LEN = 256
    # Source stamped on every session row this platform owns (also hardwired in
    # _bind_api_server_session and _create_agent) so peer lookups can filter on it.
    _SESSION_SOURCE = "api_server"

    def _declared_conversation_session(self, gateway_session_key: Optional[str]) -> Optional[str]:
        """Resolve the live session a client declared with ``X-Hermes-Session-Key`` (the key
        names the conversation, ``session_id`` its current transcript). Same reset-fenced
        recovery as ``SessionStore._recover_session_for_peer``; concurrent first requests
        converge (later row wins). None when undeclared, no live row, or DB error."""
        key = (gateway_session_key or "").strip()
        db = self._ensure_session_db() if key else None
        if db is None:
            return None
        try:
            row = db.find_latest_gateway_session_for_peer(
                source=self._SESSION_SOURCE, session_key=key)
        except Exception:
            logger.debug("[%s] declared-conversation lookup failed", self.name, exc_info=True)
            return None
        return str(row["id"]) if row and row.get("id") else None

    def _bind_declared_conversation(
        self, session_id: Optional[str], gateway_session_key: Optional[str]) -> None:
        """Record the declared conversation key on the session row (AIAgent writes it unkeyed);
        ``include_compression_ancestors`` covers a mid-turn rotation. UPDATE: no-op w/o a row.

        ``include_compression_ancestors`` carries the key up a mid-turn compression rotation so the pre- and
        post-rotation rows of one conversation share it, while that same walk deliberately stops at
        ``/branch``, delegate and tool children (#79161). The statement is an UPDATE, so it is a harmless
        no-op on a turn that failed before the row was created.
        """
        key = (gateway_session_key or "").strip()
        sid = str(session_id or "").strip()
        db = self._ensure_session_db() if key and sid else None
        if db is None:
            return
        try:
            # Never rewrite a row that already belongs to a different conversation
            # (record_gateway_session_peer does SET session_key = ?).
            existing = db.get_session(sid) or {}
            current = str(existing.get("session_key") or "").strip()
            if current and current != key:
                logger.debug(
                    "[%s] refusing to rebind session %s from a different declared conversation",
                    self.name, sid)
                return
            db.record_gateway_session_peer(
                sid, source=self._SESSION_SOURCE, session_key=key,
                include_compression_ancestors=True)
        except Exception:
            logger.debug(
                "[%s] declared-conversation bind failed for %s", self.name, sid, exc_info=True)

    def _parse_session_key_header(
        self, request: "web.Request") -> tuple[Optional[str], Optional["web.Response"]]:
        """Validate ``X-Hermes-Session-Key`` (per-channel memory scope) -> ``(key_or_None, None)``
        or ``(None, error)``. Requires API-key auth so a client can't guess another scope."""
        raw = request.headers.get("X-Hermes-Session-Key", "").strip()
        if not raw:
            return None, None
        if not self._api_key:
            logger.warning(
                "X-Hermes-Session-Key rejected: no API key configured. "
                "Set API_SERVER_KEY to enable long-term memory scoping.")
            return None, _error_response(
                "X-Hermes-Session-Key requires API key authentication. "
                "Configure API_SERVER_KEY to enable this feature.", 403)
        # Control characters could enable header injection on the echo path.
        if re.search(r'[\r\n\x00]', raw):
            return None, _invalid_request("Invalid session key")
        if len(raw) > self._MAX_SESSION_HEADER_LEN:
            return None, _invalid_request("Session key too long")
        return raw, None

    # -- Session DB -------------------------------------------------------------------

    def _open_and_cache_session_db(self, home) -> Optional[Any]:
        """Cached SessionDB for ``home`` (shared by both ``_ensure_session_db*``). Never writes
        ``self._session_db`` (explicit override only), so no profile pins later requests."""
        from hermes_state import SessionDB
        key = str(home)
        with self._session_db_cache_lock:
            if self._session_db_cache_closed:
                return None
            db = self._session_dbs.get(key)
            if db is None:
                db = SessionDB(db_path=home / "state.db")
                self._session_dbs[key] = db
            return db

    def _close_cached_session_dbs(self) -> None:
        """Close SessionDB handles owned by this adapter's profile cache."""
        with self._session_db_cache_lock:
            self._session_db_cache_closed = True
            cached = list(self._session_dbs.values())
            self._session_dbs.clear()
        shared_db = getattr(self, "_session_db", None)
        for db in cached:
            if db is shared_db:
                continue
            try:
                db.close()
            except Exception:
                logger.debug("Failed to close API-server SessionDB", exc_info=True)

    def _ensure_session_db(self):
        """SessionDB for the active profile home (the runtime scope redirects ``get_hermes_home()``
        per profile). Sync, for ``_create_agent``; handlers use ``_ensure_session_db_async``."""
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_constants import get_hermes_home
            return self._open_and_cache_session_db(get_hermes_home())
        except Exception as e:
            logger.debug("SessionDB unavailable for API server: %s", e)
            return None

    async def _ensure_session_db_async(self):
        """Async variant: the profile home is captured on the loop thread (its scope is invisible
        inside ``to_thread``), only the blocking open runs in the worker, single-flight locked."""
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
            key = str(home)
            with self._session_db_cache_lock:
                cached = self._session_dbs.get(key)
            if cached is not None:
                return cached
            if self._session_db_lock is None:
                self._session_db_lock = asyncio.Lock()
            async with self._session_db_lock:
                with self._session_db_cache_lock:
                    cached = self._session_dbs.get(key)
                if cached is not None:
                    return cached
                return await asyncio.to_thread(self._open_and_cache_session_db, home)
        except Exception as e:
            logger.debug("SessionDB unavailable for API server: %s", e)
            return None

    # -- Agent creation ---------------------------------------------------------------

    @staticmethod
    def _parse_model_routes(raw: Any) -> Dict[str, Dict[str, Any]]:
        """Validate ``model_routes`` (``alias -> {model, provider?, api_key?, base_url?}``); invalid
        shapes are dropped, never raised. Route ``api_key`` is an UPSTREAM credential: never log."""
        if not isinstance(raw, dict):
            if raw:
                logger.warning(
                    "api_server model_routes ignored: expected a mapping, got %s", type(raw).__name__)
            return {}
        allowed_keys = ("model", "provider", "api_key", "base_url")
        routes: Dict[str, Dict[str, Any]] = {}
        for alias, cfg in raw.items():
            alias_str = str(alias).strip()
            if not alias_str or not isinstance(cfg, dict):
                logger.warning(
                    "api_server model_routes: dropping invalid route entry %r", alias_str or alias)
                continue
            route = {
                key: str(cfg[key]).strip()
                for key in allowed_keys
                if cfg.get(key) is not None and str(cfg[key]).strip()}
            if not route.get("model"):
                logger.warning(
                    "api_server model_routes: route %r has no 'model'; dropping", alias_str)
                continue
            routes[alias_str] = route
        return routes

    def _resolve_route(self, model_alias: Any) -> Optional[Dict[str, Any]]:
        """Return the model_routes entry for *model_alias*, or None."""
        return self._model_routes.get(model_alias) if isinstance(model_alias, str) else None

    def _stored_session_model(self, session: Any) -> Optional[str]:
        """The model persisted on a session row, minus the virtual alias (replaying
        "hermes-agent" upstream as a provider model id 400s)."""
        stored = session.get("model") if isinstance(session, dict) else None
        if not stored or stored == self._model_name:
            return None
        return stored

    @staticmethod
    def _clean_runtime_id(value: Any, *, max_len: int = 200) -> str:
        text = "" if value is None else str(value).strip()
        return "" if len(text) > max_len or re.search(r"[\r\n\x00]", text) else text

    @classmethod
    def _split_provider_prefixed_model(cls, model: str) -> tuple[str, str]:
        text = cls._clean_runtime_id(model)
        if "::" in text:
            provider, raw = text.split("::", 1)
            if re.match(r"^[a-zA-Z0-9_.-]{2,64}$", provider) and raw.strip():
                return provider, raw.strip()
        return "", text

    @classmethod
    def _runtime_options_from_model_options(cls, model_options: Any) -> Dict[str, Any]:
        if not isinstance(model_options, dict):
            return {}
        runtime_options: Dict[str, Any] = {}
        reasoning = model_options.get("reasoning")
        if isinstance(reasoning, dict):
            enabled = reasoning.get("enabled")
            effort = cls._clean_runtime_id(reasoning.get("effort"), max_len=32)
            if enabled is False:
                runtime_options["reasoning_config"] = {"enabled": False}
            elif effort:
                runtime_options["reasoning_config"] = {"enabled": True, "effort": effort}
            elif enabled is True:
                runtime_options["reasoning_config"] = {"enabled": True}
        service_tier = cls._clean_runtime_id(model_options.get("service_tier"), max_len=32)
        if service_tier:
            runtime_options["service_tier"] = service_tier
        elif _coerce_request_bool(model_options.get("fast"), default=False):
            runtime_options["service_tier"] = "priority"
        return runtime_options

    def _session_runtime_request_from_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        raw_model = self._clean_runtime_id(body.get("model") or body.get("model_id"))
        raw_provider = self._clean_runtime_id(body.get("provider") or body.get("provider_id"), max_len=80)
        prefixed_provider, split_model = self._split_provider_prefixed_model(raw_model)
        provider = raw_provider or prefixed_provider
        model = split_model or raw_model
        alias_route = self._resolve_route(raw_model) or self._resolve_route(model)
        route = dict(alias_route) if isinstance(alias_route, dict) else None
        # The virtual alias is not a provider model id: null it upstream of route-building and
        # every "requested" dict so it is never persisted or misread as a raw override.
        if model == self._model_name:
            model = None
        route_source = "model_routes" if route else "global"
        if not route and model:
            route = {"model": model}
            if provider:
                route["provider"] = provider
            route_source = "raw_request"
        return {
            "requested": {"provider": provider, "model": model, "raw_model": raw_model},
            "route": route, "route_source": route_source,
            "runtime_options": self._runtime_options_from_model_options(body.get("model_options")),
            "require_model_lock": _coerce_request_bool(body.get("require_model_lock"), default=False),
            "model_options": (
                body.get("model_options") if isinstance(body.get("model_options"), dict) else {})}

    @classmethod
    def _requested_ids(cls, requested: Any) -> tuple[str, str]:
        """``(model, provider)`` cleaned from a ``requested``/lock mapping."""
        requested = requested or {}
        return (cls._clean_runtime_id(requested.get("model")),
                cls._clean_runtime_id(requested.get("provider"), max_len=80))

    def _runtime_lock_error(self, runtime_request: Dict[str, Any]) -> Optional["web.Response"]:
        if not runtime_request.get("require_model_lock"):
            return None
        model, provider = self._requested_ids(runtime_request.get("requested"))
        route = runtime_request.get("route")
        if not model and not provider:
            return _error_response(
                "require_model_lock was set but no model/provider was provided", 400, code="missing_model")
        if not route or runtime_request.get("route_source") == "global":
            return _error_response(
                "Requested Browser model lock cannot be routed; refusing silent global fallback",
                409, code="model_lock_unavailable")
        return None

    def _persist_session_runtime_lock(self, session_id: str, runtime_request: Dict[str, Any]) -> bool:
        # Persist only a newly confirmed lock: a reused stored lock must not be rewritten each
        # turn, and a one-off request override must not erase a confirmed lock.
        if runtime_request.get("persisted_lock") or not runtime_request.get("require_model_lock"):
            return True
        model, provider = self._requested_ids(runtime_request.get("requested"))
        if not model and not provider:
            return False
        db = self._ensure_session_db()
        if db is None:
            return False
        try:
            db.update_session_runtime_lock(
                session_id, model=model or None, provider=provider or None,
                model_options=runtime_request.get("model_options") or {},
                route_source=runtime_request.get("route_source") or "",
                confirmed=bool(runtime_request.get("require_model_lock")))
            return True
        except Exception:
            logger.warning("[%s] failed to persist session runtime lock for %s", self.name, session_id, exc_info=True)
            return False

    @staticmethod
    def _parse_session_model_config(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str) and raw.strip():
            with suppress(Exception):
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
        return {}

    def _runtime_request_from_persisted_session_lock(
        self, session: Optional[Dict[str, Any]], body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(session, dict):
            return None
        model_config = self._parse_session_model_config(session.get("model_config"))
        lock = model_config.get("browser_model_lock")
        if not isinstance(lock, dict) or not _coerce_request_bool(lock.get("confirmed"), default=False):
            return None
        model, provider = self._requested_ids(lock)
        if not model and not provider:
            return None
        if self._clean_runtime_id(lock.get("route_source"), max_len=64).lower() == "model_routes":
            route = self._resolve_route(model) if model else None
        else:
            route = {"model": model} if model else {}
            if provider:
                route["provider"] = provider
        model_options = body.get("model_options")
        if not isinstance(model_options, dict):
            model_options = lock.get("model_options")
        return {
            "requested": {"provider": provider, "model": model, "raw_model": model},
            "route": route or None, "route_source": "session_model_lock",
            "runtime_options": self._runtime_options_from_model_options(model_options),
            "require_model_lock": True,
            "model_options": model_options if isinstance(model_options, dict) else {},
            "persisted_lock": True}

    def _effective_session_runtime_request(
        self, *, session: Optional[Dict[str, Any]], body: Dict[str, Any]) -> Dict[str, Any]:
        runtime_request = self._session_runtime_request_from_body(body)
        requested = runtime_request.get("requested") or {}
        if requested.get("model") or requested.get("provider"):
            return runtime_request
        return self._runtime_request_from_persisted_session_lock(session, body) or runtime_request

    @classmethod
    def _sanitize_runtime_metadata(
        cls, *, runtime: Optional[Dict[str, Any]] = None, requested_runtime: Optional[Dict[str, Any]] = None,
        route_source: str = "global", model_lock: str = "") -> Dict[str, Any]:
        payload = dict(runtime or {})
        provider = cls._clean_runtime_id(
            payload.get("provider") or payload.get("provider_id") or payload.get("effective_provider"),
            max_len=80)
        model = cls._clean_runtime_id(payload.get("model") or payload.get("model_id") or payload.get("effective_model"))
        source = cls._clean_runtime_id(payload.get("route_source") or route_source, max_len=64)
        result: Dict[str, Any] = {
            "provider": provider, "model": model, "route_source": source or "global"}
        if requested_runtime or payload.get("requested"):
            model, provider = cls._requested_ids(requested_runtime or payload.get("requested"))
            result["requested"] = {"provider": provider, "model": model}
        if model_lock or payload.get("model_lock"):
            result["model_lock"] = cls._clean_runtime_id(model_lock or payload.get("model_lock"), max_len=32)
        return result

    @staticmethod
    def _normalize_session_source(value: Any) -> str:
        text = str(value or "").strip().lower()
        allowed = {"api_server", "hermes_browser", "browser", "cli", "telegram", "discord", "slack", "desktop", "dashboard"}
        if text not in allowed:
            return "api_server"
        return "hermes_browser" if text == "browser" else text

    def _session_model_override_for(self, session_key: Optional[str]) -> Optional[Dict[str, Any]]:
        """The gateway's per-session ``/model`` override for *session_key*, if any — a
        user-issued ``/model`` always wins over static route config."""
        if not session_key:
            return None
        try:
            from gateway.run import _gateway_runner_ref
            runner = _gateway_runner_ref()
            if runner is None:
                return None
            try:
                rehydrate = getattr(runner, "_rehydrate_session_model_override", None)
                if callable(rehydrate):
                    rehydrate(session_key)
            except Exception:
                logger.debug(
                    "api_server failed to rehydrate session /model override for %s", session_key, exc_info=True)
            override = runner._session_model_overrides.get(session_key)
            return dict(override) if isinstance(override, dict) else None
        except Exception:
            return None

    def _request_route_conflict_error(
        self, *, session_id: Optional[str], gateway_session_key: Optional[str], requested_model: Optional[str],
        requested_provider: Optional[str], route: Optional[Dict[str, Any]]) -> Optional[str]:
        """Return a 400-worthy conflict string for ambiguous route/provider mixes."""
        request_provider = _clean_request_string(requested_provider)
        if not request_provider or not isinstance(route, dict):
            return None
        if self._session_model_override_for(gateway_session_key or session_id):
            return None  # session /model wins over both, so nothing is ambiguous
        route_provider = _clean_request_string(route.get("provider"))
        route_api_key = _clean_request_string(route.get("api_key"))
        route_base_url = _clean_request_string(route.get("base_url"))
        route_alias = _clean_request_string(requested_model) or "requested model"
        if route_provider and request_provider != route_provider:
            return (
                f"Model route '{route_alias}' is pinned to provider '{route_provider}'. "
                f"Remove 'provider' or use '{route_provider}'.")
        if not route_provider and (route_api_key or route_base_url):
            return (
                f"Model route '{route_alias}' pins route credentials/base_url. "
                "Do not combine it with an explicit 'provider'.")
        return None

    @staticmethod
    def _resolve_provider_runtime(
        provider: Optional[str], *, target_model: Optional[str], required: bool,
    ) -> Optional[Dict[str, Any]]:
        """Runtime kwargs for ``provider``, falling back to the gateway's resolver; ``required``
        raises ``_ProviderAuthResolutionError`` (controlled response, not a raw 500), not None."""
        provider_name = _clean_request_string(provider)
        if not provider_name:
            return None
        try:
            return _resolve_request_runtime_agent_kwargs(provider_name, target_model=target_model or None)
        except Exception as exc:
            with suppress(Exception):
                from gateway.run import _resolve_runtime_agent_kwargs_for_provider
                return _resolve_runtime_agent_kwargs_for_provider(provider_name)
            if required:
                raise _ProviderAuthResolutionError(str(exc)) from exc
            logger.debug(
                "api_server provider-runtime refresh failed for provider=%s model=%s",
                provider_name, target_model or "", exc_info=True)
            return None

    def _apply_provider_runtime(
        self, runtime_kwargs: Dict[str, Any], provider: Optional[str], *,
        target_model: Optional[str], required: bool = False) -> bool:
        """Resolve ``provider``'s runtime and merge it into ``runtime_kwargs``; True if applied."""
        provider_runtime = self._resolve_provider_runtime(
            provider, target_model=target_model, required=required)
        if provider_runtime:
            _apply_runtime_agent_overrides(runtime_kwargs, provider_runtime)
        return bool(provider_runtime)

    def _recover_or_record_model(self, model: str, runtime_kwargs: Dict[str, Any], gateway_session_key) -> str:
        """Fill an empty resolved model: provider's default catalog model, then the last-known-good
        model for this key / process-wide. Non-empty non-virtual models are recorded instead."""
        # No model.default but a provider resolved (e.g. `hermes auth add` without `hermes model`).
        if not model and runtime_kwargs.get("provider"):
            with suppress(Exception):
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        model, runtime_kwargs["provider"])
        # Keyed by gateway_session_key only (session_id is per-request -> unbounded growth).
        _resolved_key = gateway_session_key or ""
        if not model:
            _recovered = (self._last_resolved_model.get(_resolved_key)
                          or self._last_resolved_model.get("*"))
            if _recovered and _recovered != self._model_name:
                logger.warning(
                    "Empty model resolved for session=%s — recovering "
                    "last-known-good model %s (config read likely returned "
                    "empty; see #35314)",
                    _resolved_key, _recovered)
                model = _recovered
        elif model != self._model_name:
            if _resolved_key:
                self._last_resolved_model[_resolved_key] = model
            self._last_resolved_model["*"] = model
        return model

    def _select_agent_runtime(
        self, runtime_kwargs: Dict[str, Any], model: str, *, requested_model: Optional[str],
        requested_provider: Optional[str], route: Optional[Dict[str, Any]], session_model: Optional[str],
        confirmed_runtime_lock: bool, gateway_session_key: Optional[str], session_id: Optional[str]) -> tuple:
        """Apply the model/provider precedence chain for one agent (mutates ``runtime_kwargs``):
        confirmed Browser lock > session ``/model`` override > session-persisted model >
        model_routes alias > per-request provider/model > global defaults. A confirmed lock
        bypasses the override and fails closed if its provider cannot be resolved.
        Returns ``(model, session_override, request_model, request_provider)``."""
        request_model = _clean_request_string(requested_model)
        request_provider = _clean_request_string(requested_provider)
        route_cfg = route if isinstance(route, dict) else {}
        route_model = _clean_request_string(route_cfg.get("model"))
        route_provider = _clean_request_string(route_cfg.get("provider"))
        session_key = gateway_session_key or session_id
        session_row_model = _clean_request_string(session_model)
        current_provider = _clean_request_string(runtime_kwargs.get("provider"))
        session_override = None if confirmed_runtime_lock else self._session_model_override_for(session_key)
        # Model-string precedence (override > session-persisted > global) is owned by
        # hermes_cli.model_switch.resolve_effective_model.
        from hermes_cli.model_switch import resolve_effective_model
        if session_override:
            model = resolve_effective_model(session_override, None, model)
            self._apply_provider_runtime(
                runtime_kwargs,
                _clean_request_string(session_override.get("provider")) or current_provider,
                target_model=model)
            _apply_runtime_agent_overrides(runtime_kwargs, session_override)
            if route or request_model or request_provider:
                logger.debug(
                    "api_server request selection skipped: session /model override wins for %s",
                    session_key or "")
        elif session_row_model and not confirmed_runtime_lock:
            # A session-persisted raw model (no route alias) is a standing selection that pins
            # this session's turns ahead of per-request body values.
            self._apply_provider_runtime(
                runtime_kwargs, current_provider, target_model=session_row_model)
            model = resolve_effective_model(None, session_row_model, model)
            if request_model or request_provider:
                logger.debug(
                    "api_server request selection skipped: session-persisted model wins for %s",
                    session_key or "")
        else:
            # The request's ``model`` selected the route, so its value is the ALIAS — never a
            # model name; a route with no ``model`` key keeps the global default.
            effective_model = (route_model or model) if route is not None else (request_model or model)
            effective_provider = request_provider or route_provider or current_provider
            applied = False
            if effective_provider and (bool(request_provider or route_provider) or effective_model != model):
                # A confirmed Browser lock fails closed: never fall through to the previous
                # global provider's credentials.
                applied = self._apply_provider_runtime(
                    runtime_kwargs, effective_provider, target_model=effective_model,
                    required=bool(request_provider) or confirmed_runtime_lock)
            if not applied and effective_provider and effective_provider != current_provider:
                runtime_kwargs["provider"] = effective_provider
            model = effective_model
            # Per-route explicit transport secrets/base URLs win after provider resolution.
            for key in ("api_key", "base_url"):
                value = _clean_request_string(route_cfg.get(key))
                if value:
                    runtime_kwargs[key] = value
            if route:
                logger.debug(
                    "api_server request selection applied: model=%s provider=%s route_provider=%s request_provider=%s",
                    model, runtime_kwargs.get("provider"), route_provider or "", request_provider or "")
        model = self._recover_or_record_model(model, runtime_kwargs, gateway_session_key)
        return model, session_override, request_model, request_provider

    def _create_agent(
        self, ephemeral_system_prompt: Optional[str] = None, session_id: Optional[str] = None,
        stream_delta_callback=None, tool_progress_callback=None, tool_start_callback=None,
        tool_complete_callback=None, gateway_session_key: Optional[str] = None,
        requested_model: Optional[str] = None, requested_provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None, route: Optional[Dict[str, Any]] = None,
        session_model: Optional[str] = None, confirmed_runtime_lock: bool = False,
        room_dispatch: Optional[Dict[str, Any]] = None,
        room_execution_policy: Optional[Dict[str, Any]] = None) -> Any:
        """Create an AIAgent from the gateway runtime config + platform toolsets.
        ``gateway_session_key`` persists across transcripts (memory scope), unlike ``session_id``;
        ``route`` / ``session_model`` are mutually exclusive; ``confirmed_runtime_lock`` beats the
        session ``/model`` override, disables the fallback chain and fails closed."""
        from run_agent import AIAgent
        from gateway.run import (
            _checkpoint_agent_kwargs, _current_max_iterations, _resolve_runtime_agent_kwargs,
            _resolve_gateway_model, _load_gateway_config, GatewayRunner)
        from hermes_cli.tools_config import _get_platform_tools
        # RuntimeError is caught ONLY here (sole provider-auth raiser); the typed subclass keeps
        # run_conversation() errors distinct.
        try:
            runtime_kwargs = _resolve_runtime_agent_kwargs()
        except RuntimeError as exc:
            raise _ProviderAuthResolutionError(str(exc)) from exc
        # A fallback-provider runtime carries its own ``model``: pop it (overrides config, and
        # must not collide with the ``**runtime_kwargs`` spread).
        model = runtime_kwargs.pop("model", None) or _resolve_gateway_model()
        request_reasoning_config = _request_reasoning_config(model_options)
        request_service_tier = _request_service_tier(model_options)
        model, session_override, request_model, request_provider = self._select_agent_runtime(
            runtime_kwargs, model,
            requested_model=requested_model, requested_provider=requested_provider, route=route,
            session_model=session_model, confirmed_runtime_lock=confirmed_runtime_lock,
            gateway_session_key=gateway_session_key, session_id=session_id)
        user_config = _load_gateway_config()
        enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))
        max_iterations = _current_max_iterations()
        if room_dispatch is not None:
            from gateway.hosted_room_execution_policy import RoomExecutionPolicy
            policy = RoomExecutionPolicy.from_mapping(room_execution_policy or {})
            enabled_toolsets = list(policy.enabled_toolsets)
            max_iterations = policy.max_iterations
        # Reasoning resolves against the model that actually runs (per-model overrides), so only
        # after the precedence chain settles; an explicit request wins.
        if request_reasoning_config is None:
            request_reasoning_config = GatewayRunner._load_reasoning_config(model)
        agent_kwargs = {
            "model": model, **runtime_kwargs, **_checkpoint_agent_kwargs(user_config),
            "max_iterations": max_iterations, "quiet_mode": True, "verbose_logging": False,
            "ephemeral_system_prompt": ephemeral_system_prompt or None,
            "enabled_toolsets": enabled_toolsets, "session_id": session_id,
            "platform": "api_server",
            "stream_delta_callback": stream_delta_callback,
            "tool_progress_callback": tool_progress_callback,
            "tool_start_callback": tool_start_callback,
            "tool_complete_callback": tool_complete_callback,
            "session_db": self._ensure_session_db(),
            # Same fallback provider chain as Telegram/Discord/Slack.
            "fallback_model": None if confirmed_runtime_lock else GatewayRunner._load_fallback_model(),
            "reasoning_config": request_reasoning_config,
            "gateway_session_key": gateway_session_key}
        if request_service_tier is not _REQUEST_OPTION_MISSING:
            agent_kwargs["service_tier"] = request_service_tier
        agent = AIAgent(**agent_kwargs)
        route_source = (
            "session_model_lock" if confirmed_runtime_lock
            else "session_model_override" if session_override
            else "raw_request" if route or request_model or request_provider else "global")
        agent._hermes_api_runtime = {
            "provider": runtime_kwargs.get("provider") or getattr(agent, "provider", "") or "",
            "model": getattr(agent, "model", None) or model,
            "route_source": route_source}
        return agent

    # -- HTTP handlers ----------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "hermes-agent", "version": _hermes_version()})

    @_require_auth
    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed — gateway state, platforms, PID for dashboard probing (Bearer auth)."""
        from gateway.status import (
            derive_gateway_busy, derive_gateway_drainable, normalize_updated_at, parse_active_agents,
            read_runtime_status)
        runtime = read_runtime_status() or {}
        gw_state = runtime.get("gateway_state")
        gw_active = parse_active_agents(runtime.get("active_agents", 0))
        # Served BY the gateway process, so gateway_running is True by definition; busy/
        # drainable use the same shared contract as /api/status so the two never disagree.
        active_api_runs, process_depth, active_delegations = self._readiness_work_counts()
        from gateway.run import _resolve_gateway_model
        readiness = collect_runtime_readiness(
            configured_model=_resolve_gateway_model(), runtime_status=runtime,
            active_api_runs=active_api_runs, process_completion_queue_depth=process_depth,
            active_delegations=active_delegations)
        return web.json_response({
            "status": readiness["status"], "readiness": readiness, "platform": "hermes-agent",
            "version": _hermes_version(), "gateway_state": gw_state,
            "platforms": runtime.get("platforms", {}), "active_agents": gw_active,
            "gateway_busy": derive_gateway_busy(
                gateway_running=True, gateway_state=gw_state, active_agents=gw_active),
            "gateway_drainable": derive_gateway_drainable(
                gateway_running=True, gateway_state=gw_state),
            "exit_reason": runtime.get("exit_reason"),
            # Contract: RFC3339 string | null, never a number (legacy epoch floats exist).
            "updated_at": normalize_updated_at(runtime.get("updated_at")), "pid": os.getpid()})

    @_require_auth
    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — hermes-agent plus configured model_routes aliases (alias + resolved
        model only, never credentials). Under /p/<profile>/ the primary id follows that profile."""
        now = int(time.time())
        # The middleware already entered the profile scope, so get_active_profile_name() resolves.
        model_name = self._resolve_model_name("") if _api_request_profile.get() else self._model_name

        def _model(mid: str, root: str, parent) -> Dict[str, Any]:
            return {"id": mid, "object": "model", "created": now, "owned_by": "hermes", "permission": [],
                    "root": root, "parent": parent}
        models = [_model(model_name, model_name, None)]
        models.extend(
            _model(alias, route_cfg.get("model", alias), model_name)
            for alias, route_cfg in self._model_routes.items() if alias != model_name)
        return web.json_response({"object": "list", "data": models})

    @_require_auth
    async def _handle_model_options(self, request: "web.Request") -> "web.Response":
        """GET /api/model/options — the dashboard/TUI model-picker inventory, so external clients
        can sync to the configured provider catalog instead of scraping /v1/models."""
        refresh = _coerce_request_bool(request.query.get("refresh"), default=False)
        try:
            from hermes_cli.inventory import build_model_options_payload, load_picker_context

            def _build_payload() -> Dict[str, Any]:
                return build_model_options_payload(
                    load_picker_context(), include_unconfigured=True, refresh=refresh)
            # Enrichment can fetch pricing/provider catalogs: keep it off the event loop.
            payload = await asyncio.to_thread(_build_payload)
            return web.json_response(payload)
        except Exception:
            logger.exception("[%s] GET /api/model/options failed", self.name)
            return _error_response("Failed to list model options.", 500, code="model_options_failed")

    @_require_auth
    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """GET /v1/capabilities — the stable, machine-readable API surface for external UIs."""
        return web.json_response({
            "object": "hermes.api_server.capabilities", "platform": "hermes-agent",
            "model": self._model_name,
            "auth": {"type": "bearer", "required": bool(self._api_key)},
            "runtime": {
                "mode": "server_agent", "tool_execution": "server", "split_runtime": False,
                "description": (
                    "The API server creates a server-side Hermes AIAgent; "
                    "tools execute on the API-server host unless a future "
                    "explicit split-runtime mode is enabled.")},
            "features": {
                "chat_completions": True, "chat_completions_streaming": True,
                "responses_api": True, "responses_streaming": True, "run_submission": True,
                "runs_idempotency": _api_runs._idempotency_capabilities(self, store_type=RunIdempotencyStore),
                **_STATIC_FEATURE_FLAGS,
                "cors": bool(self._cors_origins),
                # Always advertised for feature-detection; enabled follows config.
                "browser_extension_control": {
                    "enabled": self._browser_control_enabled(),
                    "protocol_version": _BROWSER_CONTROL_PROTOCOL_VERSION,
                    "capabilities": sorted(BROWSER_CONTROL_CAPABILITIES),
                    "artifact_capabilities": sorted(BROWSER_CONTROL_ARTIFACT_CAPABILITIES),
                    "developer_capabilities": sorted(BROWSER_CONTROL_DEVELOPER_CAPABILITIES),
                    "developer_mode": self._browser_control_developer_mode(),
                    "artifact_transport": {
                        "upload": {"method": "POST", "path": "/v1/artifacts/upload"},
                        "download": {
                            "method": "GET", "path": "/v1/artifacts/download/{artifact_id}"},
                        "max_bytes": DEFAULT_MAX_ARTIFACT_BYTES,
                        "ttl_seconds": DEFAULT_ARTIFACT_TTL_SECONDS,
                        "allowed_mime_types": sorted(DEFAULT_ALLOWED_MIME_TYPES)},
                    "real_browser_actions": True,
                    "transports": {
                        "local_vps": "websocket-subprotocol-ticket",
                        "cloud": "authenticated-gateway-rpc"}}},
            "endpoints": {name: {"method": m, "path": p} for name, (m, p) in _CAPABILITY_ENDPOINTS},
        })

    # -- Browser-extension control (authenticated local/VPS API) ----------------------

    async def _handle_browser_control_register(self, request: "web.Request") -> "web.Response":
        """POST /v1/browser-control/register — mint a short-lived single-use controller ticket.

        Identity is NOT taken from the body: the principal is a server-derived digest of the
        authenticated key/profile and capabilities are filtered to the allowlist (a spoofed
        ``principal_id`` / inflated list is ignored); the session must exist in the profile's
        SessionDB. Ladder: 404 disabled, 403 no API key, 401 bad Bearer, 201 success.
        """
        if not self._browser_control_enabled():
            return _error_response(
                "Browser control is not enabled on this server.", 404, code="browser_control_disabled")
        if not self._api_key:
            logger.warning(
                "browser-control registration rejected: no API key configured; "
                "set API_SERVER_KEY to enable authenticated browser control.")
            return _error_response(
                "Browser control registration requires a configured API key.", 403,
                err_type="gateway_auth_error", code="browser_control_auth_required")
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            payload = await request.json()
        except Exception:
            return _error_response("Request body must be valid JSON.", 400)
        if not isinstance(payload, dict):
            return _error_response("Request body must be a JSON object.", 400)
        if not browser_control_protocol_supported(payload.get("protocol_version")):
            return _error_response(
                "Unsupported browser-control protocol version.", 400, code="browser_control_protocol_unsupported")
        controller_id = str(payload.get("controller_id") or "").strip()
        browser_profile_id = str(payload.get("browser_profile_id") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        if not controller_id or not browser_profile_id or not session_id:
            return _error_response(
                "controller_id, browser_profile_id, and session_id are required.", 400,
                code="browser_control_invalid_registration")
        db = await self._ensure_session_db_async()
        if db is None:
            return _error_response("Session database unavailable.", 503, code="session_db_unavailable")
        if not await asyncio.to_thread(db.get_session, session_id):
            return _error_response(
                "Browser control may register only for an existing server session.", 403,
                err_type="gateway_auth_error", code="browser_control_session_forbidden")
        profile = _api_request_profile.get() or "default"
        developer_mode = self._browser_control_developer_mode()
        capabilities = filter_browser_control_capabilities(payload.get("capabilities"), developer_mode=developer_mode)
        if not capabilities:
            return _error_response(
                "At least one permitted browser-control capability is required.", 400,
                code="browser_control_no_capabilities")
        # Developer capabilities need broker Developer Mode (fail closed past the filter).
        if capabilities & BROWSER_CONTROL_DEVELOPER_CAPABILITIES and not developer_mode:
            return _error_response(
                "Developer Mode is required for browser_evaluate and raw CDP.", 403,
                code="browser_control_developer_mode_required")
        scope = ControllerScope(
            principal_id=self._derive_browser_control_principal(profile), profile_id=profile,
            session_id=session_id or None, controller_id=controller_id,
            browser_profile_id=browser_profile_id,
            transport_family=self._browser_control_transport_family(request),
            capabilities=capabilities)
        ticket = self._browser_control_broker.mint_ticket(scope)
        ticket_ttl = self._browser_control_broker.ticket_ttl_seconds
        return web.json_response(
            {
                "protocol_version": _BROWSER_CONTROL_PROTOCOL_VERSION, "ticket": ticket.value,
                # Best-effort wall-clock projection; the broker enforces expiry on its monotonic
                # clock, so after an NTP step trust ticket_expires_in_seconds.
                "ticket_expires_at": time.time() + ticket_ttl,
                "ticket_expires_in_seconds": ticket_ttl,
                "ws_path": "/v1/browser-control/ws",
                "scope": {
                    **{key: getattr(scope, key) for key in (
                        "principal_id", "profile_id", "session_id", "controller_id",
                        "browser_profile_id", "transport_family")},
                    "capabilities": sorted(scope.capabilities)}},
            status=201)

    async def _handle_browser_control_ws(self, request: "web.Request") -> "web.WebSocketResponse":
        """GET /v1/browser-control/ws — controller WebSocket (one-shot ticket).

        The ticket rides in ``Sec-WebSocket-Protocol`` (never the query string: targets land in
        access logs) and is exchanged once for the registration scope; bad/consumed/expired
        tickets 401 before upgrade. Owner-aware teardown cannot detach a newer generation.
        """
        # Re-checked at upgrade so disabling the feature closes the gate immediately.
        if not self._browser_control_enabled():
            raise web.HTTPNotFound()
        if request.query.get("ticket"):
            raise web.HTTPUnauthorized()
        requested_protocols = [
            value.strip() for value in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
            if value.strip()]
        ticket_protocols = [
            value for value in requested_protocols if value.startswith(_BROWSER_CONTROL_TICKET_PROTOCOL_PREFIX)]
        if _BROWSER_CONTROL_WS_PROTOCOL not in requested_protocols or len(ticket_protocols) != 1:
            raise web.HTTPUnauthorized()
        ticket_value = ticket_protocols[0][len(_BROWSER_CONTROL_TICKET_PROTOCOL_PREFIX) :]
        if not ticket_value:
            raise web.HTTPUnauthorized()
        try:
            scope = self._browser_control_broker.consume_ticket(ticket_value)
        except ControllerTicketInvalid:
            raise web.HTTPUnauthorized() from None
        except Exception:
            logger.exception("browser-control WS ticket consumption failed")
            raise web.HTTPUnauthorized() from None
        ws = web.WebSocketResponse(heartbeat=30.0, protocols=(_BROWSER_CONTROL_WS_PROTOCOL,))
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        _send = _browser_controller_ws_sender(ws, loop)

        # attach/disconnect take the controller send_lock, which a worker-thread dispatch may
        # hold while blocking on THIS loop: offload so the race parks a worker, not the loop.
        await asyncio.to_thread(self._browser_control_broker.attach, scope, _send, owner=ws)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        frame = msg.json()
                    except Exception:
                        continue
                    if isinstance(frame, dict):
                        reply = await asyncio.to_thread(
                            self._handle_browser_control_frame, scope, frame, owner=ws)
                        if isinstance(reply, dict):
                            await ws.send_json(reply)
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        finally:
            await asyncio.to_thread(self._browser_control_broker.disconnect, scope, owner=ws)
        return ws

    def _handle_browser_control_frame(
        self, scope: "ControllerScope", frame: dict, *, owner: Any = None) -> Optional[dict]:
        """Apply one controller->broker frame with exact-scope checks."""
        method = frame.get("method")
        params = frame.get("params")
        if not isinstance(params, dict) or owner is None or not self._browser_control_broker.is_owner(scope, owner):
            return
        if method == "browser.controller.heartbeat":
            nonce = str(params.get("nonce") or "").strip()
            if not nonce or len(nonce) > 128:
                return
            # Echoing the opaque nonce proves the socket is live without granting anything.
            return {"method": "browser.controller.heartbeat", "params": {"nonce": nonce, "ok": True}}
        if method == "browser.controller.detach":
            self._browser_control_broker.detach(scope, owner=owner, notify_controller=False)
            return {"method": "browser.controller.detach", "params": {"ok": True}}
        if method == "browser.controller.result":
            command_id = params.get("command_id")
            if isinstance(command_id, str) and command_id:
                # The broker resolves only a pending command with this socket's exact scope.
                ok = params.get("ok") is True
                self._browser_control_broker.complete(
                    command_id, scope=scope, ok=ok,
                    result=params.get("result") if ok else params.get("error"))
        elif method == "browser.controller.cancel":
            tool_call_id = params.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                self._browser_control_broker.cancel(scope, tool_call_id=tool_call_id)

    def _browser_control_enabled(self) -> bool:
        """``browser.extension_control.enabled`` (default False); tests monkeypatch this."""
        try:
            from gateway.browser_control_broker import browser_control_enabled as _flag
            return _flag()
        except Exception:
            return False

    def _derive_browser_control_principal(self, profile: str) -> str:
        """Non-reversible principal digest bound to the profile's expected API key, so a client
        cannot impersonate another controller by echoing an id."""
        key = self._expected_api_key() or self._api_key or ""
        digest = hashlib.sha256(f"{profile}\x00{key}".encode("utf-8")).hexdigest()
        return f"principal:{profile}:{digest[:32]}"

    def _browser_control_transport_family(self, request: "web.Request") -> str:
        """``local-api`` for a loopback peer, else ``remote-api``; the broker treats the family as
        part of exact identity, so a remote controller never satisfies a local-only dispatch."""
        host = None
        with suppress(Exception):
            transport = request.transport
            peer = transport.get_extra_info("peername") if transport is not None else None
            if isinstance(peer, tuple) and peer:
                host = peer[0]
            elif isinstance(peer, str):
                host = peer
        return "local-api" if host in ("127.0.0.1", "::1", "localhost") else "remote-api"

    def _browser_control_developer_mode(self) -> bool:
        """Broker Developer Mode gate for ``browser_evaluate`` / raw CDP; tests monkeypatch this."""
        try:
            return browser_control_developer_mode()
        except Exception:
            return False

    # -- One-shot artifact transport --------------------------------------------------

    def _artifact_store_for(self, profile: str) -> ArtifactStore:
        """Profile-scoped artifact store, lazily created under the profile's data dir and cached
        BY RESOLVED PROFILE (on a multiplex listener profile A must never pin B to A's root).

        The store root lives under the profile's data directory
        (``<HERMES_HOME>/plugin-data/.../artifacts``-style controlled root), so artifacts never escape the
        profile boundary. Stores are cached BY RESOLVED PROFILE — on a multiplex listener, profile A
        touching the artifact route first must never pin profile B to A's physical root (same frozen-handle
        class as the per-profile session-storage fix in #88734). The root itself is created on first use;
        TTL cleanup runs on every store/load/prune.
        """
        profile_key = str(profile or "default")
        store = self._browser_control_artifacts.get(profile_key)
        if store is not None:
            return store
        try:
            from hermes_cli.profiles import get_profile_dir
            root = Path(get_profile_dir(profile or "default")) / "artifacts" / "browser-control"
        except Exception:
            # Unscoped fallback (tests/manual wiring): controlled root under the Hermes home.
            try:
                from hermes_state import get_hermes_home
                root = Path(get_hermes_home()) / "artifacts" / "browser-control"
            except Exception:
                raise ArtifactError("no artifact root is resolvable") from None
        store = ArtifactStore(
            root, ttl_seconds=DEFAULT_ARTIFACT_TTL_SECONDS, max_bytes=DEFAULT_MAX_ARTIFACT_BYTES,
            allowed_mime_types=DEFAULT_ALLOWED_MIME_TYPES)
        store.prune_expired()
        self._browser_control_artifacts[profile_key] = store
        # Shared with the broker so dispatched artifact actions validate against the same
        # profile's controlled root ("approved artifact id only").
        try:
            self._browser_control_broker.attach_artifact_store(store, profile_id=profile_key)
        except Exception:
            logger.debug("could not attach artifact store to broker", exc_info=True)
        return store

    def _artifact_limiter(self) -> ArtifactRateLimiter:
        """Return the per-principal artifact route limiter (lazy)."""
        if self._browser_control_artifact_limiter is None:
            self._browser_control_artifact_limiter = ArtifactRateLimiter(window_seconds=60.0, max_requests=30)
        return self._browser_control_artifact_limiter

    def _inject_browser_control_artifacts(
        self, store: Optional[ArtifactStore], limiter: Optional[ArtifactRateLimiter] = None, *,
        profile: str = "default") -> None:
        """Inject a store/limiter (tests, diagnostics)."""
        if store is None:
            self._browser_control_artifacts.pop(profile, None)
        else:
            self._browser_control_artifacts[profile] = store
        if limiter is not None:
            self._browser_control_artifact_limiter = limiter

    def _artifact_route_prelude(self, request: "web.Request", action: str, *, check_enabled: bool = True) -> tuple:
        """Shared upload/download gate (feature flag -> API key -> Bearer -> per-principal rate
        limit) -> ``((profile, principal), None)`` or ``(None, error_response)``."""
        if check_enabled and not self._browser_control_enabled():
            return None, _error_response(
                "Browser control is not enabled on this server.", 404, code="browser_control_disabled")
        if not self._api_key:
            return None, _error_response(
                "Artifact transport requires a configured API key.", 403,
                err_type="gateway_auth_error", code="browser_control_auth_required")
        auth_err = self._check_auth(request)
        if auth_err:
            return None, auth_err
        profile = _api_request_profile.get() or "default"
        principal = self._derive_browser_control_principal(profile)
        if not self._artifact_limiter().allow(f"{action}:{principal}"):
            return None, _error_response(
                f"Artifact {action} rate limit exceeded.", 429, err_type="rate_limit_error",
                code="rate_limit_exceeded", headers={"Retry-After": "1"})
        return (profile, principal), None

    async def _handle_artifact_upload(self, request: "web.Request") -> "web.Response":
        """POST /v1/artifacts/upload — one-shot bounded raw upload; ``Content-Type`` must be an
        allowed MIME type, ``X-Artifact-Filename`` is display-only. Returns a provenance receipt
        (never a filesystem path). Ladder: 404 disabled, 403 no API key, 401 bad Bearer, 429
        rate limited, 413 too large, 415 MIME rejected, 400 missing filename/scope, 201."""
        ctx, err = self._artifact_route_prelude(request, "upload")
        if err is not None:
            return err
        profile, principal = ctx
        content_type = request.headers.get("Content-Type", "")
        filename = request.headers.get("X-Artifact-Filename", "").strip()
        if not filename:
            return _error_response("X-Artifact-Filename header is required.", 400)
        try:
            store = self._artifact_store_for(profile)
        except ArtifactError as exc:
            return _error_response(str(exc), 500, code="artifact_rejected")
        max_bytes = store.max_bytes
        try:
            # Read cap + 1 so an oversize body is rejected without unbounded buffering.
            data = await request.content.read(max_bytes + 1)
        except Exception:
            return _error_response("Failed to read request body.", 400)
        if len(data) > max_bytes:
            return _error_response(f"Artifact exceeds the {max_bytes}-byte cap.", 413, code="artifact_too_large")
        if not data:
            return _error_response("Empty artifact body.", 400)
        scope = _ArtifactScopeFacade(principal, transport_family=self._browser_control_transport_family(request))
        try:
            receipt = store.store(data, filename=filename, content_type=content_type, scope=scope)
        except ArtifactTooLarge as exc:
            return _error_response(str(exc), 413, code="artifact_too_large")
        except ArtifactError as exc:
            if "allowlist" in str(exc):
                return _error_response(str(exc), 415, code="artifact_mime_rejected")
            return _error_response(str(exc), 400, code="artifact_rejected")
        return web.json_response(
            receipt.to_dict(download_path=f"/v1/artifacts/download/{receipt.artifact_id}"), status=201)

    async def _handle_artifact_download(self, request: "web.Request") -> "web.Response":
        """GET /v1/artifacts/download/{artifact_id} — one-shot download (a second one 404s) with
        ``X-Artifact-Sha256``. Ladder: 404 disabled/unknown, 403 no API key, 401 bad Bearer, 429
        rate limited, 410 expired, 400 invalid id/scope mismatch, 200."""
        if not self._browser_control_enabled():
            raise web.HTTPNotFound()
        ctx, err = self._artifact_route_prelude(request, "download", check_enabled=False)
        if err is not None:
            return err
        profile, principal = ctx
        artifact_id = request.match_info.get("artifact_id", "")
        scope = _ArtifactScopeFacade(principal, transport_family=self._browser_control_transport_family(request))
        try:
            data, receipt = self._artifact_store_for(profile).load(artifact_id, scope=scope)
        except ArtifactError as exc:
            message = str(exc)
            if "expired" in message:
                return _error_response(message, 410, code="artifact_expired")
            status = 400 if "scope" in message or "invalid" in message else 404
            return _error_response(message, status, code="artifact_not_found")
        return web.Response(
            body=data, status=200, content_type=receipt.content_type,
            headers={"X-Artifact-Sha256": receipt.sha256, "X-Artifact-Id": receipt.artifact_id,
                     "Content-Disposition": f'attachment; filename="{receipt.filename}"'})

    @_require_auth
    async def _handle_skills(self, request: "web.Request") -> "web.Response":
        """GET /v1/skills — deterministic JSON listing of installed skills (name, description,
        category), the same set ``/skills list`` shows."""
        try:
            from tools.skills_tool import _find_all_skills, _sort_skills
            skills = _sort_skills(_find_all_skills(skip_disabled=False))
        except Exception:
            logger.exception("GET /v1/skills failed")
            return _error_response("Failed to enumerate skills", 500, err_type="server_error")
        return web.json_response({"object": "list", "data": skills})

    @_require_auth
    async def _handle_toolsets(self, request: "web.Request") -> "web.Response":
        """GET /v1/toolsets — each toolset the api_server agent exposes: enabled/configured state
        plus the concrete tool names it expands to."""
        try:
            from hermes_cli.config import load_config
            from hermes_cli.tools_config import (
                _get_effective_configurable_toolsets, _get_platform_tools, _toolset_has_keys,
                get_nous_subscription_features)
            from toolsets import resolve_toolset
            config = load_config()
            enabled_toolsets = _get_platform_tools(config, "api_server", include_default_mcp_servers=False)
            features = get_nous_subscription_features(config)
            data: List[Dict[str, Any]] = []
            for name, label, desc in _get_effective_configurable_toolsets():
                try:
                    tools = sorted(set(resolve_toolset(name)))
                except Exception:
                    tools = []
                data.append({
                    "name": name, "label": label, "description": desc,
                    "enabled": name in enabled_toolsets,
                    "configured": _toolset_has_keys(name, config, features=features),
                    "tools": tools})
        except Exception:
            logger.exception("GET /v1/toolsets failed")
            return _error_response("Failed to enumerate toolsets", 500, err_type="server_error")
        return web.json_response({"object": "list", "platform": "api_server", "data": data})

    # -- /api/sessions: thin client/session resource API -------------------------------

    @staticmethod
    def _parse_nonnegative_int(value: Any, default: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return default if parsed < 0 else min(parsed, maximum)

    @staticmethod
    def _session_db_unavailable() -> "web.Response":
        return _error_response("Session database unavailable", 503, code="session_db_unavailable")

    @staticmethod
    def _session_response(session: Dict[str, Any]) -> Dict[str, Any]:
        """Return a stable, client-safe session representation."""
        safe_keys = (
            "id", "source", "user_id", "model", "title", "started_at", "ended_at", "end_reason",
            "message_count", "tool_call_count", "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
            "actual_cost_usd", "api_call_count", "parent_session_id", "last_active", "preview",
            "_lineage_root_id", "pinned", "archived", "hidden")
        payload = {key: session.get(key) for key in safe_keys if key in session}
        # SQLite stores the flags as 0/1.
        payload.update(
            {f: bool(payload[f]) for f in ("pinned", "archived", "hidden") if f in payload})
        # Full system prompts / model_config never cross the client API; only their presence.
        payload["has_system_prompt"] = bool(session.get("system_prompt"))
        payload["has_model_config"] = bool(session.get("model_config"))
        return payload

    @staticmethod
    def _message_response(message: Dict[str, Any]) -> Dict[str, Any]:
        message = _project_client_message(message)
        safe_keys = (
            "id", "session_id", "role", "content", "tool_call_id", "tool_calls", "tool_name",
            "timestamp", "token_count", "finish_reason", "reasoning", "reasoning_content",
            "display_kind")
        return {key: message.get(key) for key in safe_keys if key in message}

    async def _read_json_body(self, request: "web.Request") -> tuple[Dict[str, Any], Optional["web.Response"]]:
        try:
            body = await request.json()
        except Exception:
            return {}, _error_response("Invalid JSON in request body", 400)
        if not isinstance(body, dict):
            return {}, _error_response("Request body must be a JSON object", 400)
        return body, None

    async def _get_existing_session_or_404(self, session_id: str) -> tuple[Optional[Dict[str, Any]], Optional["web.Response"]]:
        db = await self._ensure_session_db_async()
        if db is None:
            return None, self._session_db_unavailable()
        session = await asyncio.to_thread(db.get_session, session_id)
        if not session:
            return None, _error_response(f"Session not found: {session_id}", 404, code="session_not_found")
        return session, None

    async def _conversation_history_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        db = await self._ensure_session_db_async()
        if db is None:
            return []
        try:
            return await asyncio.to_thread(db.get_messages_as_conversation, session_id)
        except Exception as exc:
            logger.warning("Failed to load session history for %s: %s", session_id, exc)
            return []

    @_require_auth
    async def _handle_list_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions — list persisted Hermes sessions."""
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        limit = self._parse_nonnegative_int(request.query.get("limit"), default=50, maximum=200)
        offset = self._parse_nonnegative_int(request.query.get("offset"), default=0, maximum=1_000_000)
        source = request.query.get("source") or None
        include_children = _coerce_request_bool(request.query.get("include_children"), default=False)
        # Exact-title lookup (`hermes peer dm` -> canonical "Bot Chat"). include_hidden is honored
        # ONLY with a title filter: a blanket hidden listing stays off this client surface.
        title_filter = (request.query.get("title") or "").strip() or None
        include_hidden = bool(title_filter) and _coerce_request_bool(
            request.query.get("include_hidden"), default=False)

        async def _list() -> list:
            # include_pinned back-fills pins past the recency window; search_query pushes the
            # title needle into SQL (substring) so a hidden/old row is found, exact match below.
            rows = await asyncio.to_thread(
                db.list_sessions_rich, source=source, limit=limit, offset=offset,
                include_children=include_children, order_by_last_active=True, include_pinned=True,
                search_query=title_filter, include_hidden=include_hidden)
            if title_filter:
                rows = [s for s in rows if (s.get("title") or "").strip() == title_filter]
            return rows

        sessions = await _list()
        if title_filter and not sessions:
            # A canonical Bot Chat auto-archived by the orphan reaper would make `hermes peer dm`
            # mint transient sessions: resurrect and re-list; deliberate archives stay put.
            try:
                # Recoverable-archive resurrection (#92687): a canonical Bot Chat archived by the ws-orphan
                # reaper / older agent cleanup is invisible to list_sessions_rich (include_archived=False),
                # which would fail `hermes peer dm` resolution and mint transient sessions — same accident
                # the tui_gateway lookups heal.
                from tools.bot_mode_probe import BOT_CHAT_TITLE
                stale = db.get_session_by_title(title_filter) if title_filter == BOT_CHAT_TITLE else None
                if stale and stale.get("archived") and db.unarchive_recoverable_session(stale["id"]):
                    sessions = await _list()
            except Exception:
                pass  # resolution degrades to today's no-row behavior
        # Back-filled pins arrive PAST the limit, so counting them would report
        # another page that doesn't exist. Only the recency window decides.
        windowed = sum(1 for s in sessions if not s.get("pinned"))
        return web.json_response({
            "object": "list", "data": [self._session_response(s) for s in sessions],
            "limit": limit, "offset": offset, "has_more": windowed >= limit})

    @_require_auth
    async def _handle_create_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions -- create an empty Hermes session row. Existence check, insert and
        title handling run as ONE off-loop write so concurrent same-id creates can't both 201."""
        body, err = await self._read_json_body(request)
        if err:
            return err
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        raw_id = body.get("id") or body.get("session_id")
        session_id = str(raw_id).strip() if raw_id else f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        from gateway.session import _is_path_unsafe
        if not session_id or re.search(r'[\r\n\x00]', session_id) or _is_path_unsafe(session_id):
            return _error_response("Invalid session ID", 400, code="invalid_session_id")
        if len(session_id) > self._MAX_SESSION_HEADER_LEN:
            return _error_response("Session ID too long", 400, code="invalid_session_id")
        system_prompt = body.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return _error_response("system_prompt must be a string", 400, code="invalid_system_prompt")
        source = self._normalize_session_source(body.get("source") or "api_server")
        runtime_request = self._session_runtime_request_from_body(body)
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return lock_error
        requested = runtime_request.get("requested") or {}
        # The normalized requested["model"] (prefix split, virtual alias nulled) — the raw body
        # would persist "hermes-agent" and later send it to the provider literally.
        model_name = self._clean_runtime_id(requested.get("model")) or None
        model_config = None
        if requested.get("model") or requested.get("provider"):
            model_config = {"browser_model_lock": {
                "provider": requested.get("provider") or "", "model": requested.get("model") or "",
                "model_options": runtime_request.get("model_options") or {},
                "route_source": runtime_request.get("route_source") or "",
                "confirmed": bool(runtime_request.get("require_model_lock")),
                "updated_at": time.time()}}
        title = body.get("title")

        def _atomic(conn):
            # One BEGIN IMMEDIATE write: a concurrent same-id create blocks and sees the row.
            if conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone():
                return None, "exists"
            conn.execute(
                """INSERT INTO sessions (
                   id, source, model, model_config, system_prompt, started_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, source, model_name, json.dumps(model_config) if model_config else None,
                 system_prompt, time.time()))
            if title is not None:
                clean_title = db.sanitize_title(str(title))
                if clean_title:
                    conflict = conn.execute(
                        "SELECT id FROM sessions WHERE title = ? AND id != ?", (clean_title, session_id)).fetchone()
                    if conflict:
                        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                        return None, f"title:Title already in use by session {conflict['id']}"
                conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (clean_title, session_id))
            session_row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return (dict(session_row) if session_row else {
                "id": session_id, "source": source, "model": model_name, "title": title}), None
        session, err = await asyncio.to_thread(db._execute_write, _atomic)
        if err == "exists":
            return _error_response(f"Session already exists: {session_id}", 409, code="session_exists")
        if err and err.startswith("title:"):
            return _error_response(err[len("title:"):], 400, code="invalid_title")
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)}, status=201)

    @_require_auth
    async def _handle_get_session(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}."""
        session, err = await self._get_existing_session_or_404(request.match_info["session_id"])
        if err:
            return err
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)})

    @_require_auth
    async def _handle_patch_session(self, request: "web.Request") -> "web.Response":
        """PATCH /api/sessions/{session_id} — update client-safe session metadata."""
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        # pinned/archived/unread are durable desktop-sidebar flags.
        unknown = sorted(set(body) - {"title", "end_reason", "pinned", "archived", "hidden", "unread"})
        if unknown:
            return _error_response(
                f"Unsupported session fields: {', '.join(unknown)}", 400, code="unsupported_session_field")
        for flag in ("pinned", "archived", "hidden", "unread"):
            if flag in body and not isinstance(body[flag], bool):
                return _error_response(f"'{flag}' must be a boolean", 400, code="invalid_session_field")
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        if "title" in body:
            try:
                await asyncio.to_thread(
                    db.set_session_title, session_id, "" if body["title"] is None else str(body["title"]))
            except ValueError as exc:
                return _error_response(str(exc), 400, code="invalid_title")
        for flag, setter in (("pinned", db.set_session_pinned), ("archived", db.set_session_archived),
                             ("hidden", db.set_session_hidden)):
            if flag in body:
                await asyncio.to_thread(setter, session_id, body[flag])
        if "unread" in body:
            await asyncio.to_thread(db.set_session_read, session_id, read=not body["unread"])
        if body.get("end_reason"):
            await asyncio.to_thread(db.end_session, session_id, str(body["end_reason"]))
        session = await asyncio.to_thread(db.get_session, session_id) or session
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)})

    @_require_auth
    async def _handle_delete_session(self, request: "web.Request") -> "web.Response":
        """DELETE /api/sessions/{session_id}."""
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        deleted = await asyncio.to_thread(db.delete_session, session_id)
        return web.json_response({"object": "hermes.session.deleted", "id": session_id, "deleted": bool(deleted)})

    @_require_auth
    async def _handle_session_messages(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}/messages."""
        session_id = request.match_info["session_id"]
        _, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        resolved_id = await asyncio.to_thread(db.resolve_resume_session_id, session_id)
        raw_limit, raw_offset = request.query.get("limit"), request.query.get("offset", "0")
        order = request.query.get("order")
        if order not in (None, "oldest", "latest"):
            return _error_response("order must be one of: oldest, latest", 400, code="invalid_pagination")
        try:
            offset = int(raw_offset)
            requested_limit = None if raw_limit is None else int(raw_limit)
        except (TypeError, ValueError):
            offset = requested_limit = -1
        if offset < 0 or (requested_limit is not None and requested_limit < 0):
            return _error_response("limit and offset must be non-negative integers", 400, code="invalid_pagination")
        default_page = requested_limit is None
        latest_page = order == "latest" or (order is None and default_page)
        limit = 500 if default_page else min(requested_limit, 500)
        messages = await asyncio.to_thread(
            db.get_messages, resolved_id, limit=limit, offset=offset, latest=latest_page)
        return web.json_response({
            "object": "list", "session_id": resolved_id,
            "data": [self._message_response(m) for m in messages],
            "pagination": {
                "limit": limit, "offset": offset,
                "order": order or ("latest" if default_page else "oldest"),
                "returned": len(messages)}})

    @_require_auth
    async def _handle_fork_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/fork — branch via current SessionDB primitives."""
        source_id = request.match_info["session_id"]
        source, err = await self._get_existing_session_or_404(source_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        db = await self._ensure_session_db_async()
        fork_id = str(body.get("id") or body.get("session_id") or f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}").strip()
        if not fork_id or re.search(r'[\r\n\x00]', fork_id):
            return _error_response("Invalid session ID", 400, code="invalid_session_id")
        if await asyncio.to_thread(db.get_session, fork_id):
            return _error_response(f"Session already exists: {fork_id}", 409, code="session_exists")

        # CLI /branch semantics: end the original as branched, create a child with the transcript.
        await asyncio.to_thread(db.end_session, source_id, "branched")
        await asyncio.to_thread(
            db.create_session, fork_id, "api_server", model=source.get("model"),
            system_prompt=source.get("system_prompt"), parent_session_id=source_id)
        messages = await asyncio.to_thread(db.get_messages, source_id)
        await asyncio.to_thread(db.replace_messages, fork_id, messages)
        title = body.get("title")
        if title is None:
            base = source.get("title") or "fork"
            title = f"{base} fork"
            with suppress(Exception):
                title = await asyncio.to_thread(db.get_next_title_in_lineage, base)
        try:
            await asyncio.to_thread(db.set_session_title, fork_id, str(title))
        except ValueError as exc:
            return _error_response(str(exc), 400, code="invalid_title")
        fork = await asyncio.to_thread(db.get_session, fork_id) or {"id": fork_id, "parent_session_id": source_id}
        return web.json_response({"object": "hermes.session", "session": self._session_response(fork)}, status=201)

    async def _prepare_session_chat(self, request: "web.Request") -> tuple:
        """Shared prelude for /api/sessions/{id}/chat[/stream]: header/body validation, then
        runtime selection — a Browser model lock (body ``require_model_lock`` or a confirmed
        persisted lock) wins; else the session-persisted model routes via model_routes when an
        alias or threads through as ``session_model`` when raw, then body values.
        Returns ``(ctx, None)`` or ``(None, error)``; ``ctx["run_kwargs"]`` feeds ``_run_agent``."""
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return None, key_err
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return None, err
        body, err = await self._read_json_body(request)
        if err:
            return None, err
        user_message, err = _session_chat_user_message(body)
        if err is not None:
            return None, err
        system_prompt = body.get("system_message") or body.get("instructions")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return None, _error_response("system_message must be a string", 400, code="invalid_system_message")
        runtime_request = self._effective_session_runtime_request(session=session, body=body)
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return None, lock_error
        if not self._persist_session_runtime_lock(session_id, runtime_request):
            return None, _error_response(
                "Could not persist the requested session model lock", 500, code="model_lock_persistence_failed")
        lock_active = bool(runtime_request.get("require_model_lock"))
        if lock_active:
            route = runtime_request.get("route")
            session_model = None
            requested = runtime_request.get("requested") or {}
            agent_overrides: Dict[str, Any] = {}
            for src_key, dst_key in (("model", "requested_model"), ("provider", "requested_provider")):
                if requested.get(src_key):
                    agent_overrides[dst_key] = requested[src_key]
            if runtime_request.get("model_options"):
                agent_overrides["model_options"] = runtime_request["model_options"]
        else:
            stored_model = self._stored_session_model(session)
            stored_route = self._resolve_route(stored_model)
            route = stored_route or self._resolve_route(body.get("model"))
            session_model = stored_model if (stored_model and stored_route is None) else None
            agent_overrides = _request_agent_overrides(body, virtual_model=self._model_name)
            selection_error = self._request_route_conflict_error(
                session_id=session_id, gateway_session_key=gateway_session_key,
                requested_model=agent_overrides.get("requested_model"),
                requested_provider=agent_overrides.get("requested_provider"), route=route)
            if selection_error:
                return None, _error_response(selection_error, 400)
        run_kwargs = dict(
            user_message=user_message, ephemeral_system_prompt=system_prompt, session_id=session_id,
            gateway_session_key=gateway_session_key, route=route, session_model=session_model,
            requested_runtime=runtime_request.get("requested") or {},
            route_source=runtime_request.get("route_source") or "global",
            confirmed_runtime_lock=lock_active, **agent_overrides)
        return {
            "gateway_session_key": gateway_session_key, "session_id": session_id, "body": body,
            "user_message": user_message, "runtime_request": runtime_request,
            "lock_active": lock_active, "run_kwargs": run_kwargs}, None

    @staticmethod
    def _session_headers(session_id: str, gateway_session_key: Optional[str]) -> Dict[str, str]:
        """``X-Hermes-Session-Id`` (+ ``X-Hermes-Session-Key`` when declared) response headers."""
        headers = {"X-Hermes-Session-Id": session_id}
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        return headers

    def _effective_turn_runtime(self, runtime_request: Dict[str, Any], result: Any, usage: Any) -> Dict[str, Any]:
        """Sanitized runtime metadata for a finished session-chat turn."""
        runtime = self._result_runtime(result, usage)
        return self._sanitize_runtime_metadata(
            # Same shared ladder /api/status uses. Before this was unified, the two endpoints disagreed on
            # the same page load — the sidebar strip read "running" (it probed GATEWAY_HEALTH_URL and scoped
            # to the requested profile) while the Channels page rendered "The gateway is not running" (it
            # did neither). Cross-container, profile-scoped, and launch-service-managed deployments each hit
            # that split. profile_home is passed when the request was scoped to a named profile:
            # gateway/status readers resolve process-level paths and do NOT follow the HERMES_HOME
            # contextvar override (#56986 / #69143), so the profile's directory has to be handed over
            # explicitly or messaging silently reports another profile's gateway (#71211).
            runtime=runtime,
            requested_runtime=runtime_request.get("requested"),
            route_source=runtime_request.get("route_source") or "global",
            model_lock=self._model_lock_state(runtime_request, runtime))

    @staticmethod
    def _result_runtime(result: Any, usage: Any) -> Dict[str, Any]:
        """Runtime metadata from the result dict, falling back to the usage dict."""
        runtime = (result.get("runtime") or {}) if isinstance(result, dict) else {}
        return runtime or ((usage.get("runtime") or {}) if isinstance(usage, dict) else {})

    @staticmethod
    def _model_lock_state(runtime_request: Dict[str, Any], runtime: Any) -> str:
        """``confirmed`` once a runtime was observed under a lock, ``accepted`` before, else ``""``."""
        if not runtime_request.get("require_model_lock"):
            return ""
        return "confirmed" if runtime else "accepted"

    @_admit_api_agent_request
    async def _handle_session_chat(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/chat — one synchronous agent turn."""
        ctx, err = await self._prepare_session_chat(request)
        if err is not None:
            return err
        gateway_session_key = ctx["gateway_session_key"]
        session_id = ctx["session_id"]
        history = await self._conversation_history_for_session(session_id)
        result, usage = await self._run_agent(conversation_history=history, **ctx["run_kwargs"])
        is_dict = isinstance(result, dict)
        effective_session_id = result.get("session_id") if is_dict else session_id
        final_response = _resolve_media_to_data_urls(
            result.get("final_response", "") if is_dict else "")
        headers = self._session_headers(effective_session_id or session_id, gateway_session_key)
        return web.json_response(
            {"object": "hermes.session.chat.completion",
             "session_id": effective_session_id or session_id,
             "message": {"role": "assistant", "content": final_response}, "usage": usage,
             "runtime": self._effective_turn_runtime(ctx["runtime_request"], result, usage)},
            headers=headers)

    @_admit_api_agent_request
    async def _handle_session_chat_stream(self, request: "web.Request") -> "web.StreamResponse":
        """POST /api/sessions/{session_id}/chat/stream — SSE wrapper over _run_agent."""
        ctx, err = await self._prepare_session_chat(request)
        if err is not None:
            return err
        gateway_session_key, session_id = ctx["gateway_session_key"], ctx["session_id"]
        user_message, runtime_request = ctx["user_message"], ctx["runtime_request"]
        runtime_meta = self._sanitize_runtime_metadata(
            requested_runtime=runtime_request.get("requested"),
            route_source=runtime_request.get("route_source") or "global",
            model_lock=("accepted" if ctx["lock_active"] else ""))
        message_id = f"msg_{uuid.uuid4().hex}"
        run_id = f"run_{uuid.uuid4().hex}"
        events = _SessionEventQueue(session_id, run_id)
        queue, _event_payload = events.queue, events.payload
        # Claim ownership inside the request's profile scope before any run-keyed state
        # exists, so /v1/runs/{id}* control is confined to the starting profile.
        # See #93689.
        self._run_owners[run_id] = self._run_idempotency_scope(request)
        self._set_run_status(
            run_id, "queued", session_id=session_id, model=ctx["body"].get("model", self._model_name))

        def _delta(delta: str) -> None:
            if delta:
                events.enqueue("assistant.delta", {"message_id": message_id, "delta": delta})

        def _tool_progress(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs) -> None:
            if event_type == "reasoning.available":
                events.enqueue("tool.progress", {"message_id": message_id, "tool_name": tool_name or "_thinking", "delta": preview or ""})
            elif event_type in {"tool.started", "tool.completed", "tool.failed"}:
                events.enqueue(event_type, {"message_id": message_id, "tool_name": tool_name, "preview": preview, "args": args})

        async def _run_and_signal() -> None:
            try:
                await queue.put(_event_payload("run.started", {
                    "user_message": {"role": "user", "content": user_message},
                    "runtime": runtime_meta}))
                self._set_run_status(run_id, "running", last_event="run.started")
                await queue.put(_event_payload("message.started", {"message": {"id": message_id, "role": "assistant"}}))
                history = await self._conversation_history_for_session(session_id)
                result, usage = await self._run_agent(
                    conversation_history=history, stream_delta_callback=_delta,
                    tool_progress_callback=_tool_progress, active_run_id=run_id, **ctx["run_kwargs"])
                is_dict = isinstance(result, dict)
                final_response = _resolve_media_to_data_urls(result.get("final_response", "") if is_dict else "")
                effective_session_id = result.get("session_id", session_id) if is_dict else session_id
                turn_messages = self._turn_transcript_messages(history, user_message, result) if is_dict else []
                effective_runtime = self._effective_turn_runtime(runtime_request, result, usage)
                await queue.put(_event_payload("assistant.completed", {
                    "session_id": effective_session_id, "message_id": message_id,
                    "content": final_response, "completed": True,
                    "partial": bool(result.get("partial")) if is_dict else False,
                    "interrupted": False, "runtime": effective_runtime}))
                # A steer accepted after the final reply lands in result["pending_steer"]; surface
                # it so clients can replay it rather than lose it.
                pending_steer = result.get("pending_steer") if is_dict else None
                completed_payload = {
                    "session_id": effective_session_id, "message_id": message_id, "completed": True,
                    "messages": turn_messages, "usage": usage, "runtime": effective_runtime}
                if pending_steer:
                    completed_payload["pending_steer"] = pending_steer
                await queue.put(_event_payload("run.completed", completed_payload))
                self._set_run_status(
                    run_id, "completed", session_id=effective_session_id, usage=usage,
                    last_event="run.completed",
                    **({"pending_steer": pending_steer} if pending_steer else {}))
            except asyncio.CancelledError:
                self._set_run_status(run_id, "cancelled", last_event="run.cancelled")
                raise
            except Exception as exc:
                logger.exception("[api_server] session chat stream failed")
                self._set_run_status(
                    run_id, "failed", error=_redact_api_error_text(exc), last_event="run.failed")
                await queue.put(_event_payload("error", {"message": _redact_api_error_text(exc)}))
            finally:
                self._active_run_agents.pop(run_id, None)
                self._release_run_owner_if_forgotten(run_id)
                await queue.put(_event_payload("done", {}))
                await queue.put(None)

        # NOT in _active_run_tasks: _run_agent already counts this turn for the shutdown drain.
        task = asyncio.create_task(_run_and_signal())
        self._track_background_task(task)
        headers = {
            "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", **self._session_headers(session_id, gateway_session_key)}
        response = web.StreamResponse(status=200, headers=headers)
        await response.prepare(request)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if item is None:
                    break
                name, payload = item
                await response.write(_sse_frame(payload, event=name, ensure_ascii=False))
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            await self._drain_session_stream_task_on_disconnect(
                run_id, task, interrupt_message="SSE client disconnected", shield_wait=False)
            logger.info("Session SSE client disconnected; interrupted live run %s", run_id)
        except asyncio.CancelledError:
            await self._drain_session_stream_task_on_disconnect(
                run_id, task, interrupt_message="SSE task cancelled", shield_wait=True)
            logger.info("Session SSE task cancelled; drained live run %s", run_id)
            raise
        except Exception as exc:
            logger.debug("[api_server] session SSE stream error: %s", exc)
        return response

    async def _drain_session_stream_task_on_disconnect(
        self, run_id: str, task: "asyncio.Task", *, interrupt_message: str, shield_wait: bool
    ) -> None:
        """Preserve live run control refs until the executor-backed turn actually exits."""
        agent = self._active_run_agents.get(run_id)
        if agent is None:
            if not task.done():
                task.cancel()
                with suppress(Exception):
                    await task
            return
        with suppress(Exception):
            agent.interrupt(interrupt_message)
        if not task.done():
            with suppress(Exception):
                await (asyncio.shield(task) if shield_wait else task)

    @_require_auth
    async def _handle_session_model_lock(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/model — backend-ack a Browser model lock."""
        session_id = request.match_info["session_id"]
        _, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        runtime_request = self._session_runtime_request_from_body(body)
        runtime_request["require_model_lock"] = True
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return lock_error
        if not self._persist_session_runtime_lock(session_id, runtime_request):
            return _error_response(
                "Could not persist the requested session model lock", 500, code="model_lock_persistence_failed")
        requested = runtime_request.get("requested") or {}
        route = runtime_request.get("route") or {}
        runtime = self._sanitize_runtime_metadata(
            runtime={
                "provider": route.get("provider") or requested.get("provider") or "",
                "model": route.get("model") or requested.get("model") or "",
                "route_source": runtime_request.get("route_source") or "raw_request"},
            requested_runtime=requested,
            route_source=runtime_request.get("route_source") or "raw_request",
            model_lock="accepted")
        return web.json_response(
            {"object": "hermes.session.model_lock", "session_id": session_id, "runtime": runtime})

    # -- Cron jobs API ----------------------------------------------------------------

    _JOB_ID_RE = re.compile(r"[a-f0-9]{12}")
    # Update whitelist — prevents clients injecting arbitrary keys.
    _UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled"}
    _MAX_NAME_LENGTH = 200
    _MAX_PROMPT_LENGTH = 5000

    def _cron_request_guard(
        self, request: "web.Request", *, need_job_id: bool = False, check_draining: bool = False,
    ) -> tuple:
        """Shared /api/jobs prelude: auth → (drain) → cron available → (job_id). Returns (job_id, err)."""
        auth_err = self._check_auth(request)
        if auth_err:
            return None, auth_err
        if check_draining:
            draining = self._draining_response()
            if draining is not None:
                return None, draining
        if not _CRON_AVAILABLE:
            return None, web.json_response({"error": "Cron module not available"}, status=501)
        if not need_job_id:
            return None, None
        job_id = request.match_info["job_id"]
        if not self._JOB_ID_RE.fullmatch(job_id):
            logger.warning(
                "Cron jobs API rejected invalid job_id %r: %s", job_id, self._request_audit_log_suffix(request))
            return job_id, web.json_response({"error": "Invalid job ID format"}, status=400)
        return job_id, None

    @staticmethod
    def _cron_error_response(exc: BaseException) -> "web.Response":
        return web.json_response({"error": _redact_api_error_text(exc)}, status=500)

    def _validate_cron_prompt(self, prompt: str) -> Optional["web.Response"]:
        """Length cap + injection scan shared by create/update/run."""
        if len(prompt) > self._MAX_PROMPT_LENGTH:
            return web.json_response({"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400)
        if prompt and _scan_cron_prompt is not None:
            scan_error = _scan_cron_prompt(prompt)
            if scan_error:
                return web.json_response({"error": scan_error}, status=400)
        return None

    def _job_response(self, fn, job_id: str, *, notify: bool) -> "web.Response":
        """Run ``fn(job_id)``: 404 when falsy, else ``{"job": ...}``; exceptions -> 500."""
        try:
            job = fn(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            if notify:
                _notify_cron_provider_jobs_changed()
            return web.json_response({"job": job})
        except Exception as e:
            return self._cron_error_response(e)

    async def _job_lookup_or_mutate(self, request: "web.Request", fn, *, notify: bool) -> "web.Response":
        job_id, err = self._cron_request_guard(request, need_job_id=True)
        return err if err else self._job_response(fn, job_id, notify=notify)

    async def _handle_list_jobs(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs — list all cron jobs."""
        _, err = self._cron_request_guard(request)
        if err:
            return err
        try:
            include_disabled = request.query.get("include_disabled", "").lower() in {"true", "1"}
            return web.json_response({"jobs": _cron_list(include_disabled=include_disabled)})
        except Exception as e:
            return self._cron_error_response(e)

    async def _handle_create_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs — create a new cron job."""
        _, err = self._cron_request_guard(request)
        if err:
            return err
        try:
            body = await request.json()
            name = (body.get("name") or "").strip()
            schedule = (body.get("schedule") or "").strip()
            prompt = body.get("prompt", "")
            skills = body.get("skills")
            repeat = body.get("repeat")
            if not name:
                return web.json_response({"error": "Name is required"}, status=400)
            if len(name) > self._MAX_NAME_LENGTH:
                return web.json_response({"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400)
            if not schedule:
                return web.json_response({"error": "Schedule is required"}, status=400)
            prompt_err = self._validate_cron_prompt(prompt)
            if prompt_err:
                return prompt_err
            if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
                return web.json_response({"error": "Repeat must be a positive integer"}, status=400)
            kwargs = {
                "prompt": prompt, "schedule": schedule, "name": name,
                "deliver": body.get("deliver", "local"),
                "origin": self._cron_origin_from_request(request)}
            if skills:
                kwargs["skills"] = skills
            if repeat is not None:
                kwargs["repeat"] = repeat
            return web.json_response({"job": _cron_create(**kwargs)})
        except _CronSchedulerRegistrationError as e:
            return web.json_response(e.to_dict(), status=424)
        except Exception as e:
            return self._cron_error_response(e)

    async def _handle_get_job(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs/{job_id} — get a single cron job."""
        return await self._job_lookup_or_mutate(request, _cron_get, notify=False)

    async def _handle_update_job(self, request: "web.Request") -> "web.Response":
        """PATCH /api/jobs/{job_id} — update a cron job."""
        job_id, err = self._cron_request_guard(request, need_job_id=True)
        if err:
            return err
        try:
            body = await request.json()
            # Whitelist allowed fields to prevent arbitrary key injection
            sanitized = {k: v for k, v in body.items() if k in self._UPDATE_ALLOWED_FIELDS}
            if not sanitized:
                return web.json_response({"error": "No valid fields to update"}, status=400)
            if "name" in sanitized and len(sanitized["name"]) > self._MAX_NAME_LENGTH:
                return web.json_response({"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400)
            if "prompt" in sanitized:
                prompt_err = self._validate_cron_prompt(sanitized["prompt"])
                if prompt_err:
                    return prompt_err
        except Exception as e:
            return self._cron_error_response(e)
        return self._job_response(lambda jid: _cron_update(jid, sanitized), job_id, notify=True)

    async def _handle_delete_job(self, request: "web.Request") -> "web.Response":
        """DELETE /api/jobs/{job_id} — delete a cron job."""
        job_id, err = self._cron_request_guard(request, need_job_id=True)
        if err:
            return err
        try:
            if not _cron_remove(job_id):
                return web.json_response({"error": "Job not found"}, status=404)
            _notify_cron_provider_jobs_changed()
            return web.json_response({"ok": True})
        except Exception as e:
            return self._cron_error_response(e)

    async def _handle_pause_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/pause — pause a cron job."""
        return await self._job_lookup_or_mutate(request, _cron_pause, notify=True)

    async def _handle_resume_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/resume — resume a paused cron job."""
        return await self._job_lookup_or_mutate(request, _cron_resume, notify=True)

    async def _handle_run_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/run — trigger immediate execution."""
        job_id, err = self._cron_request_guard(request, need_job_id=True, check_draining=True)
        if err:
            return err
        # Optional transient per-run context (standalone `hermes cron run` /
        # cronjob(action='run', prompt=...)) — same cap + scan as a stored prompt.
        extra_prompt = body = None
        with suppress(Exception):
            body = await request.json()
        if isinstance(body, dict):
            raw_prompt = body.get("prompt")
            if raw_prompt is not None:
                extra_prompt = str(raw_prompt)
                prompt_err = self._validate_cron_prompt(extra_prompt)
                if prompt_err:
                    return prompt_err
                extra_prompt = extra_prompt or None
        return self._job_response(
            lambda jid: _cron_trigger(jid, extra_prompt=extra_prompt), job_id, notify=False)

    async def _handle_cron_fire(self, request: "web.Request") -> "web.Response":
        """POST /api/cron/fire — Chronos fire webhook (NAS -> agent), authenticated by a
        NAS-minted JWT via the pluggable verifier, NOT API_SERVER_KEY. 202 + background run so
        a long turn never trips NAS's timeout; the store CAS claim guards double-fire on retry."""
        from hermes_cli.config import cfg_get, load_config
        from plugins.cron_providers.chronos.verify import get_fire_verifier
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        cfg = load_config()
        verifier = get_fire_verifier()
        verify_kwargs = dict(
            token=token,
            expected_audience=cfg_get(cfg, "cron", "chronos", "expected_audience", default=""),
            jwks_or_key=cfg_get(cfg, "cron", "chronos", "nas_jwks_url", default="") or None,
            issuer=cfg_get(cfg, "cron", "chronos", "portal_url", default="") or None)
        try:
            claims = await _call_verifier(verifier, **verify_kwargs)
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
        *, chat_id: str = "", session_key: str = "", session_id: str = "",
        browser_control_principal: str = "", browser_control_transport_family: str = "") -> list:
        """Bind session contextvars for an API-server agent run — the SINGLE chokepoint for every
        agent-entry path. Hardwires ``platform="api_server"`` + ``async_delivery=False`` (HTTP
        can never wake the agent after the turn) so no route reintroduces the silent no-op bug.
        Returns reset tokens for ``clear_session_vars`` in a ``finally`` (request-scoped).

        See #10760.
        """
        from gateway.session_context import set_session_vars
        return set_session_vars(
            platform="api_server", chat_id=chat_id, session_key=session_key, session_id=session_id,
            browser_control_principal=browser_control_principal,
            browser_control_transport_family=browser_control_transport_family,
            async_delivery=False, cron_session="")

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
        for key, actual in (("provider", actual_provider), ("model", actual_model)):
            if actual:
                runtime[key] = actual
            else:
                runtime.setdefault(key, "")
        route = route or {}
        requested_runtime = requested_runtime or {}
        if confirmed_runtime_lock:
            expected_provider = self._clean_runtime_id(
                route.get("provider") or requested_runtime.get("provider"), max_len=80)
            expected_model = self._clean_runtime_id(route.get("model") or requested_runtime.get("model"))
            if (expected_provider and actual_provider != expected_provider) or (
                expected_model and actual_model != expected_model):
                raise RuntimeError(
                    "confirmed model lock runtime mismatch: "
                    f"expected provider={expected_provider or '<unspecified>'} "
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
        tool_complete_callback=None, agent_ref: Optional[list] = None, active_run_id: Optional[str] = None,
        gateway_session_key: Optional[str] = None, requested_model: Optional[str] = None,
        requested_provider: Optional[str] = None, model_options: Optional[Dict[str, Any]] = None,
        route: Optional[Dict[str, Any]] = None, session_model: Optional[str] = None,
        requested_runtime: Optional[Dict[str, Any]] = None, route_source: str = "global",
        confirmed_runtime_lock: bool = False, bind_declared_conversation: bool = False) -> tuple:
        """Create an agent and run one turn in a thread executor -> ``(result, usage)``.
        ``agent_ref[0]`` receives the agent so SSE writers can interrupt it; ``active_run_id``
        registers it in ``_active_run_agents``. Under a confirmed model lock the actual
        provider/model must match or the turn fails; ``runtime`` metadata is attached."""
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
                    session_id=session_id or "",
                    browser_control_principal=request_browser_control_principal,
                    browser_control_transport_family=request_browser_control_transport_family)
                agent = None
                try:
                    agent = self._create_agent(
                        ephemeral_system_prompt=ephemeral_system_prompt, session_id=session_id,
                        stream_delta_callback=stream_delta_callback, tool_progress_callback=tool_progress_callback,
                        tool_start_callback=tool_start_callback, tool_complete_callback=tool_complete_callback,
                        gateway_session_key=gateway_session_key, requested_model=requested_model,
                        requested_provider=requested_provider, model_options=model_options, route=route,
                        session_model=session_model, confirmed_runtime_lock=confirmed_runtime_lock)
                    if agent_ref is not None:
                        agent_ref[0] = agent
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
                    result = agent.run_conversation(
                        user_message=user_message, conversation_history=conversation_history,
                        task_id=effective_task_id)
                    return self._finish_turn_result(
                        agent, result, session_id, route=route, requested_runtime=requested_runtime,
                        route_source=route_source, confirmed_runtime_lock=confirmed_runtime_lock)
                except _ProviderAuthResolutionError as exc:
                    # Typed provider-auth failure only, handled once for every caller in
                    # run.py's response shape (text, no HTTP error).
                    logger.warning("Provider authentication failed for session=%s: %s",
                                   session_id or "", exc)
                    return (
                        {"final_response": f"⚠️ Provider authentication failed: {exc}", "messages": [],
                         "api_calls": 0, "tools": []},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                finally:
                    # Turn over (any outcome): clear ownership so a late disconnect can't reap
                    # background work this turn deliberately left running.
                    if active_run_id:
                        self._active_run_agents.pop(active_run_id, None)
                    if agent is not None:
                        _clear_turn_process_ownership(agent)
                        self._shutdown_interruptible_agents.pop(id(agent), None)
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
        try:
            return await loop.run_in_executor(None, _run)
        finally:
            self._inflight_agent_runs -= 1

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

    async def _handle_converse_ws(self, request: "web.Request") -> "web.WebSocketResponse":
        return await _api_server_converse._handle_converse_ws(self, request)

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
            # Bind directly (a pre-probe raced the bind, misreporting TIME_WAIT as "in use").
            # SO_REUSEADDR off on macOS (BSD can split traffic between two listeners).
            # Bind directly instead of probing 127.0.0.1 first — the old single-family pre-probe raced the
            # real bind and reported a TIME_WAIT socket as "in use" (#10297), failing gateway restarts for
            # up to ~60s. SO_REUSEADDR is platform-dependent (same rationale as the webhook adapter,
            # #65482): - macOS (BSD semantics): two sockets with SO_REUSEADDR can silently split traffic
            # while both report success — disable. - Linux: SO_REUSEADDR only permits rebinding past
            # TIME_WAIT (a second live listener needs SO_REUSEPORT, never set), so keep the default
            # (enabled) for instant restart rebinds.
            self._site = web.TCPSite(
                self._runner, self._host, self._port, reuse_address=False if sys.platform == "darwin" else None)
            try:
                await self._site.start()
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
                        # (observed: 1568+ retries over 5 days across multi-profile setups all defaulting to
                        # the same port, #52132), filling errors.log and leaking the adapter's ResponseStore
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
            self._mark_connected()
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                self.name, self._host, self._port, self._model_name)
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
        if self._response_store is not None:
            try:
                self._response_store.close()
            except Exception:
                logger.debug("Failed to close response store for %s", self.name, exc_info=True)
        _api_runs._close_run_state(self)
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
