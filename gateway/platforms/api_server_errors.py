"""API server error envelope: the OpenAI-compatible body plus the fields clients branch on.

Every JSON error the API server returns keeps the OpenAI shape existing SDKs parse
(``{"error": {"message", "type", "param", "code"}}``). ``error_body_middleware`` adds, on every
path that builds one (handlers, guards, sibling modules), the fields a client needs to decide
what to do next without string-matching messages:

- ``retryable``: from the closed ``ERROR_CODES`` table, else the response's own retry signals.
- ``retry_after_ms``: the ``Retry-After`` header in milliseconds, or null.
- ``field``: the offending input name (same value as ``param``).
- ``request_id``: the caller's ``X-Request-Id`` when well-formed, else a generated id; also sent
  back as the ``X-Request-Id`` response header so a log line and a bug report line up.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:  # the adapter reports the missing dependency; this module must still import
    web = None  # type: ignore[assignment]

from agent.redact import redact_sensitive_text

REQUEST_ID_HEADER = "X-Request-Id"
REQUEST_ID_KEY = "hermes_request_id"  # request-scoped storage slot set by error_body_middleware
_REQUEST_ID_RE = re.compile(r"[\x21-\x7e]{1,128}")
# Statuses a client may retry when the code is not in the table (or absent).
_RETRYABLE_STATUSES = frozenset({408, 429, 503, 504})


@dataclass(frozen=True)
class ErrorCode:
    status: int
    retryable: bool


def _codes(status: int, retryable: bool, *names: str) -> dict[str, ErrorCode]:
    return dict.fromkeys(names, ErrorCode(status, retryable))


# The codes the API server emits, published by /v1/schema. ``status`` is the code's HTTP status;
# ``retryable`` says whether resending the same request can succeed. A code missing here (a new
# ``RuntimeStoreError`` reason surfacing through ``store_error_response``) falls back to the
# response's own retry signals in ``_is_retryable``.
ERROR_CODES: dict[str, ErrorCode] = {
    **_codes(400, False, "invalid_request", "invalid_json", "invalid_content_length",
             "invalid_params", "missing_message", "missing_model", "invalid_author",
             "invalid_pagination", "invalid_platform", "invalid_session_field",
             "invalid_session_id", "invalid_system_message", "invalid_system_prompt",
             "invalid_title", "unsupported_session_field", "invalid_idempotency_key",
             "invalid_content_part", "invalid_image_url", "unsupported_content_type",
             "invalid_steer_input", "invalid_room_dispatch", "invalid_room_invitation",
             "invalid_room_grant_refresh", "invalid_room_grant_revoke",
             "invalid_room_run_close", "bot_chat_parameters_unsupported"),
    **_codes(401, False, "gateway_auth_failed"),
    **_codes(403, False, "room_grant_required"),
    **_codes(404, False, "session_not_found", "run_not_found", "profile_not_served"),
    **_codes(409, False, "session_exists", "model_lock_unavailable", "idempotency_key_conflict",
             "admission_conflict", "unknown_execution", "approval_not_pending",
             "run_not_active", "run_not_accepting_steer", "steer_not_accepted",
             # run-control refusals from ``store_error_response`` (documented as 409)
             "not_found", "stale_generation", "stale_epoch", "permission_denied",
             "profile_mismatch", "revision_conflict", "admission_started"),
    **_codes(409, True, "session_active_turn", "session_busy"),
    **_codes(413, False, "body_too_large"),
    **_codes(429, True, "rate_limit_exceeded"),
    **_codes(500, False, "model_options_failed", "model_lock_persistence_failed",
             "platform_event_dispatch_failed", "steer_failed"),
    # The agent run already executed: resending would run it (and its side effects) again.
    **_codes(502, False, "agent_incomplete"),
    **_codes(503, True, "gateway_draining", "platform_unavailable", "session_db_unavailable",
             "room_grant_store_unavailable", "run_store_unavailable", "runtime_draining",
             "storage_unavailable", "chain_moved", "config_backend_unavailable"),
    # Accepted and queued: a resend would be a second request.
    **_codes(503, False, "admission_queued", "platform_http_events_unsupported"),
}


def _redact_api_error_text(value: Any, *, limit: int | None = None) -> str:
    """Redact API-bound error text before it crosses the HTTP boundary."""
    redacted = redact_sensitive_text(str(value), force=True)
    return redacted[:limit] if limit is not None else redacted


def _openai_error(message: str, err_type: str = "invalid_request_error", param: str | None = None, code: str | None = None) -> dict[str, Any]:
    """OpenAI-style error envelope."""
    return {"error": {
        "message": _redact_api_error_text(message), "type": err_type, "param": param, "code": code}}


def _error_response(
    message: str, status: int, *, err_type: str = "invalid_request_error",
    param: str | None = None, code: str | None = None, headers: Optional[dict[str, str]] = None,
) -> web.Response:
    """``web.json_response(_openai_error(...), status=...)`` in one call."""
    return web.json_response(_openai_error(message, err_type, param, code), status=status, headers=headers)


def _invalid_request(message: str) -> web.Response:
    """400 with the bare ``{message, type}`` envelope the OpenAI-compatible validators use."""
    return web.json_response({"error": {"message": message, "type": "invalid_request_error"}}, status=400)


def request_id_for(request: web.Request) -> str:
    """The caller's ``X-Request-Id`` when it is 1-128 visible ASCII characters, else a new id."""
    supplied = request.headers.get(REQUEST_ID_HEADER, "").strip()
    return supplied if _REQUEST_ID_RE.fullmatch(supplied) else f"req_{uuid.uuid4().hex}"


def retry_after_ms(value: Optional[str]) -> Optional[int]:
    """A ``Retry-After`` header (delta-seconds or HTTP-date) in milliseconds; None if absent/bad."""
    text = (value or "").strip()
    if text.isdigit():
        return int(text) * 1000
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    return max(0, int((when.timestamp() - time.time()) * 1000))


def _is_retryable(code: Any, response: web.Response, delay_ms: Optional[int]) -> bool:
    entry = ERROR_CODES.get(code) if isinstance(code, str) else None
    if entry is not None:
        return entry.retryable
    if response.headers.get("x-should-retry", "").lower() == "false":
        return False
    return delay_ms is not None or response.status in _RETRYABLE_STATUSES


def _openai_error_body(response: web.Response) -> Optional[dict[str, Any]]:
    """The decoded body when ``response`` is an unsent JSON ``{"error": {...}}`` reply."""
    if response.status < 400 or response.prepared:
        return None
    if response.content_type != "application/json" or not isinstance(response.body, bytes):
        return None
    try:
        body = json.loads(response.body)
    except ValueError:
        return None
    return body if isinstance(body, dict) and isinstance(body.get("error"), dict) else None


def enrich_error_response(response: web.StreamResponse, request_id: str) -> None:
    """Add the retry/field/request-id fields to an OpenAI error body in place; keys a handler
    already set win, and every existing key keeps its value."""
    if not isinstance(response, web.Response):
        return  # an SSE/stream response: its body is not ours to rewrite
    body = _openai_error_body(response)
    if body is None:
        return
    error = body["error"]
    delay_ms = retry_after_ms(response.headers.get("Retry-After"))
    error.setdefault("retryable", _is_retryable(error.get("code"), response, delay_ms))
    error.setdefault("retry_after_ms", delay_ms)
    error.setdefault("field", error.get("param"))
    error.setdefault("request_id", request_id)
    response.body = json.dumps(body).encode()


async def _error_body_middleware(request: web.Request, handler):
    """Bind this request's id (read back by audit logs and ``stamp_request_id``) and enrich
    its error body."""
    request[REQUEST_ID_KEY] = request_id = request_id_for(request)
    response = await handler(request)
    enrich_error_response(response, request_id)
    return response


async def stamp_request_id(request: web.Request, response: web.StreamResponse) -> None:
    """``on_response_prepare`` hook: every response head, SSE streams included (their head is
    flushed inside the handler, before any middleware sees the response), carries the id."""
    response.headers.setdefault(REQUEST_ID_HEADER, request.get(REQUEST_ID_KEY) or request_id_for(request))


error_body_middleware = web.middleware(_error_body_middleware) if web is not None else None
