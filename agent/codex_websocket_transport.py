"""Opt-in WebSocket transport for ChatGPT Codex Responses.

Initial scope is intentionally narrow: ``provider == "openai-codex"`` using the
ChatGPT Codex backend URL (``https://chatgpt.com/backend-api/codex`` or the
resolved ``.../codex/responses`` equivalent). Generic OpenAI Responses endpoints
keep the existing SSE path because their ``store=False`` continuation semantics
are different from ChatGPT Codex's connection-scoped continuation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import ssl
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse, urlunparse

from utils import base_url_hostname

logger = logging.getLogger(__name__)

VALID_CODEX_RESPONSES_TRANSPORTS = {"sse", "websocket", "websocket-cached", "auto"}
OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
CODEX_RESPONSES_SDK_ONLY_FIELDS = frozenset({
    "extra_body",
    "extra_headers",
    "extra_query",
    "stream",
    "timeout",
})
SESSION_WEBSOCKET_CACHE_TTL_SECONDS = 5 * 60
WEBSOCKET_RECV_POLL_TIMEOUT_SECONDS = 1.0
WEBSOCKET_CONNECT_TIMEOUT_SECONDS = 15.0
WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE = "websocket_connection_limit_reached"
_TERMINAL_EVENTS = {
    "response.completed",
    "response.incomplete",
    "response.failed",
    "response.cancelled",
}


def normalize_codex_responses_transport(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        if normalized in VALID_CODEX_RESPONSES_TRANSPORTS:
            return normalized
    return "sse"


def set_agent_codex_responses_transport(agent: Any, value: Any) -> str:
    """Update an agent's Responses transport and retire stale session sockets."""
    normalized = normalize_codex_responses_transport(value)
    previous = normalize_codex_responses_transport(
        getattr(agent, "responses_transport", "sse")
    )
    if previous != normalized:
        cleanup_codex_websocket_session(getattr(agent, "session_id", None))
        agent._codex_websocket_auto_disabled_for = None
    agent.responses_transport = normalized
    return normalized


def resolve_codex_responses_url(base_url: str | None) -> str:
    """Resolve the HTTP Responses URL using Codex backend routing rules.

    ``base_url`` is the runtime endpoint from Hermes/OpenAI client state, not a
    request-payload field. The ChatGPT Codex backend accepts
    ``/backend-api/codex/responses``. If the endpoint already ends in
    ``/codex/responses`` it is used as-is; if it ends in ``/codex`` append
    ``/responses``; otherwise append ``/codex/responses``.
    """
    raw = (base_url or "https://chatgpt.com/backend-api/codex").strip().rstrip("/")
    if raw.endswith("/codex/responses"):
        return raw
    if raw.endswith("/codex"):
        return f"{raw}/responses"
    return f"{raw}/codex/responses"


def resolve_codex_websocket_url(base_url: str | None) -> str:
    """Resolve the WebSocket URL from the runtime Codex base URL.

    First resolve the HTTP Responses URL via ``resolve_codex_responses_url()``,
    then replace ``https`` with ``wss`` and ``http`` with ``ws``. Example:
    ``https://chatgpt.com/backend-api/codex`` ->
    ``wss://chatgpt.com/backend-api/codex/responses``.
    """
    parsed = urlparse(resolve_codex_responses_url(base_url))
    scheme = "wss" if parsed.scheme == "https" else "ws" if parsed.scheme == "http" else parsed.scheme
    return urlunparse(parsed._replace(scheme=scheme))


def is_supported_codex_websocket_backend(*, provider: str | None, base_url: str | None) -> bool:
    if (provider or "").strip().lower() != "openai-codex":
        return False
    resolved = resolve_codex_responses_url(base_url).lower().rstrip("/")
    return base_url_hostname(resolved) == "chatgpt.com" and resolved.endswith("/backend-api/codex/responses")


def request_body_without_input(body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in (body or {}).items()
        if k not in {"input", "previous_response_id", "client_metadata"}
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_bodies_match_except_input(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _stable_json(request_body_without_input(a)) == _stable_json(request_body_without_input(b))


def responses_inputs_equal(a: Any, b: Any) -> bool:
    return _stable_json(a) == _stable_json(b)


def build_codex_websocket_wire_body(api_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a WebSocket ``response.create`` body from SDK request kwargs.

    OpenAI SDK controls such as ``extra_headers``, ``extra_body``, ``timeout``,
    and ``stream`` belong to the client or handshake layer. ``extra_body`` is
    merged into the event body with the same precedence as the HTTP SDK path so
    provider extensions survive transport selection.
    """
    body = {
        key: value
        for key, value in api_kwargs.items()
        if key not in CODEX_RESPONSES_SDK_ONLY_FIELDS
    }
    extra_body = api_kwargs.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, Mapping):
            raise ValueError("Codex Responses WebSocket extra_body must be an object")
        body.update(dict(extra_body))
    unsupported = sorted(key for key in ("background", "stream") if key in body)
    if unsupported:
        raise ValueError(
            "Codex Responses WebSocket does not support body field(s): "
            + ", ".join(unsupported)
        )
    return body


@dataclass
class CachedWebSocketContinuationState:
    last_request_body: Dict[str, Any]
    last_response_id: str
    last_response_items: List[Dict[str, Any]]


@dataclass
class CachedWebSocketConnection:
    ws: Any
    cache_key: tuple[str, str] | None = None
    busy: bool = False
    continuation: CachedWebSocketContinuationState | None = None
    last_used_at: float = 0.0


class _WebSocketTransportError(RuntimeError):
    def __init__(self, message: str, *, cause: BaseException | None = None, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable
        if cause is not None:
            for attr in ("status_code", "body", "code", "param", "response"):
                if hasattr(cause, attr):
                    setattr(self, attr, getattr(cause, attr))
            if not hasattr(self, "status_code"):
                response = getattr(cause, "response", None)
                status = getattr(response, "status_code", None)
                if status is not None:
                    self.status_code = status


class WebSocketStartedError(_WebSocketTransportError):
    """Raised after the WebSocket request was sent; auto mode must not fallback."""

    request_replay_safe = False


class WebSocketNotStartedError(_WebSocketTransportError):
    """Raised when retry or fallback is known to be replay-safe."""

    request_replay_safe = True


class WebSocketRejectedError(_WebSocketTransportError):
    """Raised for an explicit terminal error event returned by the server."""

    request_replay_safe = True


class CodexWebSocketEventError(RuntimeError):
    """Structured provider error received from a Responses WebSocket."""

    def __init__(self, event: Any):
        body = _event_to_dict(event)
        nested = body.get("error") if isinstance(body.get("error"), dict) else {}
        message = nested.get("message") or body.get("message") or "Codex WebSocket error"
        status = (
            body.get("status")
            or body.get("status_code")
            or nested.get("status")
            or nested.get("status_code")
        )
        try:
            status_code = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_code = None
        self.body = body
        self.status_code = status_code
        self.code = nested.get("code") or body.get("code")
        self.param = nested.get("param") or body.get("param")
        super().__init__(str(message))


_websocket_session_cache: Dict[tuple[str, str], CachedWebSocketConnection] = {}
_pending_websocket_connections: Dict[tuple[str, str], threading.Event] = {}
_cache_generation = 0

# A process-wide lock is enough: entries are held briefly only while acquiring or
# releasing. The actual socket read/write stays outside this lock.
_cache_lock = threading.RLock()


def cleanup_codex_websocket_sessions() -> None:
    global _cache_generation
    with _cache_lock:
        _cache_generation += 1
        entries = list(_websocket_session_cache.items())
        _websocket_session_cache.clear()
        pending = list(_pending_websocket_connections.values())
        _pending_websocket_connections.clear()
    for event in pending:
        event.set()
    for _, entry in entries:
        _close_websocket_silently(entry.ws)


def cleanup_codex_websocket_session(session_id: str | None) -> None:
    """Close cached Codex WebSockets for one Hermes session."""
    if not session_id:
        return
    global _cache_generation
    stale: list[CachedWebSocketConnection] = []
    with _cache_lock:
        _cache_generation += 1
        for key, entry in list(_websocket_session_cache.items()):
            if key[0] == session_id:
                stale.append(entry)
                _websocket_session_cache.pop(key, None)
                entry.busy = False
                entry.continuation = None
        pending = []
        for key, event in list(_pending_websocket_connections.items()):
            if key[0] == session_id:
                pending.append(event)
                _pending_websocket_connections.pop(key, None)
    for event in pending:
        event.set()
    for entry in stale:
        _close_websocket_silently(entry.ws)


def cleanup_stale_codex_websocket_sessions(now: float | None = None) -> None:
    cutoff = (time.time() if now is None else now) - SESSION_WEBSOCKET_CACHE_TTL_SECONDS
    stale: list[CachedWebSocketConnection] = []
    with _cache_lock:
        for key, entry in list(_websocket_session_cache.items()):
            if not entry.busy and entry.last_used_at and entry.last_used_at < cutoff:
                stale.append(entry)
                _websocket_session_cache.pop(key, None)
    for entry in stale:
        _close_websocket_silently(entry.ws)


def _is_websocket_reusable(ws: Any) -> bool:
    connection = getattr(ws, "_connection", ws)
    closed = getattr(connection, "closed", False)
    if closed:
        return False
    state = getattr(connection, "state", None)
    state_name = str(getattr(state, "name", state) or "").upper()
    if state_name in {"CLOSED", "CLOSING"}:
        return False
    return True


def _close_websocket_silently(ws: Any) -> None:
    close = getattr(ws, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _abort_websocket_silently(ws: Any) -> None:
    """Unblock a cross-thread receive and leave final close to its owner."""
    connection = getattr(ws, "_connection", ws)
    sock = getattr(connection, "socket", None)
    shutdown = getattr(sock, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(socket.SHUT_RDWR)
            return
        except Exception:
            pass
    _close_websocket_silently(ws)


def _websocket_cache_key(
    session_id: str,
    url: str,
    headers: list[tuple[str, str]],
    extra_query: Mapping[str, Any] | None = None,
    client: Any = None,
) -> tuple[str, str]:
    """Key cached sockets by session plus endpoint/auth header fingerprint.

    Codex OAuth credentials can refresh while a Hermes session keeps the same
    ``session_id``. Including a stable fingerprint prevents reusing a socket
    authenticated for an older token or a different backend URL.
    """
    normalized_headers = sorted((str(k).lower(), str(v)) for k, v in headers)
    try:
        auth_headers = getattr(client, "auth_headers", None)
    except Exception:
        auth_headers = None
    normalized_auth_headers = (
        sorted((str(k).lower(), str(v)) for k, v in auth_headers.items())
        if isinstance(auth_headers, Mapping)
        else []
    )
    client_base_url = str(getattr(client, "base_url", "") or "")
    websocket_base_url = str(getattr(client, "websocket_base_url", "") or "")
    fingerprint = hashlib.sha256(
        _stable_json({
            "url": url,
            "headers": normalized_headers,
            "auth_headers": normalized_auth_headers,
            "client_base_url": client_base_url,
            "websocket_base_url": websocket_base_url,
            "query": dict(extra_query or {}),
        }).encode("utf-8")
    ).hexdigest()
    return session_id, fingerprint


def _connect_websocket(
    client: Any,
    headers: Mapping[str, str],
    *,
    extra_query: Mapping[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    # Keep connection establishment separate from the much longer model request
    # timeout. Until the handshake completes there is no socket object for the
    # Hermes watchdog to abort from its polling thread. A short bound therefore
    # guarantees that a stuck handshake remains a pre-send, replay-safe failure.
    open_timeout = WEBSOCKET_CONNECT_TIMEOUT_SECONDS
    if isinstance(timeout, (int, float)) and timeout > 0:
        open_timeout = min(open_timeout, float(timeout))
    options: Dict[str, Any] = {"open_timeout": open_timeout}

    # Proxy/SSL note: websockets.sync.client.connect uses environment proxy
    # support in modern websockets releases, but it cannot inherit custom httpx
    # transports mounted on the OpenAI SDK client. Keep support scoped to the
    # native ChatGPT Codex backend for this first PR. Users with enterprise
    # custom SDK transports should keep responses_transport=sse.
    if os.getenv("SSL_CERT_FILE"):
        ctx = ssl.create_default_context(cafile=os.getenv("SSL_CERT_FILE"))
        options["ssl"] = ctx

    # The SDK appends ``/responses`` to its WebSocket base URL. Older Hermes
    # configurations sometimes already include that suffix in ``base_url``;
    # give the SDK an explicit root so it does not connect to
    # ``.../responses/responses``.
    client_base_url = str(getattr(client, "base_url", "") or "").rstrip("/")
    copy_client = getattr(client, "copy", None)
    if client_base_url.endswith("/responses") and callable(copy_client):
        websocket_url = resolve_codex_websocket_url(client_base_url)
        websocket_root = websocket_url[: -len("/responses")]
        client = copy_client(websocket_base_url=websocket_root)

    responses = getattr(client, "responses", None)
    connect = getattr(responses, "connect", None)
    if not callable(connect):
        raise RuntimeError(
            "The installed OpenAI SDK does not expose responses.connect(); "
            "install the Hermes-pinned OpenAI SDK version."
        )
    connect_kwargs: Dict[str, Any] = {
        "extra_headers": dict(headers),
        "websocket_connection_options": options,
    }
    if extra_query:
        connect_kwargs["extra_query"] = dict(extra_query)
    manager = connect(**connect_kwargs)
    enter = getattr(manager, "enter", None) or getattr(manager, "__enter__", None)
    if not callable(enter):
        raise RuntimeError("OpenAI responses.connect() returned an invalid connection manager")
    return enter()


def _acquire_websocket(
    client: Any,
    url: str,
    headers: Mapping[str, str],
    session_id: str | None,
    *,
    extra_query: Mapping[str, Any] | None = None,
    timeout: float | None = None,
):
    cleanup_stale_codex_websocket_sessions()
    connect_options: Dict[str, Any] = {"timeout": timeout}
    if extra_query:
        connect_options["extra_query"] = extra_query
    if not session_id:
        ws = _connect_websocket(client, headers, **connect_options)
        return ws, None, False, lambda keep=True: _close_websocket_silently(ws)

    cache_key = _websocket_cache_key(
        session_id, url, list(headers.items()), extra_query, client
    )
    with _cache_lock:
        acquisition_generation = _cache_generation

    while True:
        stale_ws = None
        create_temporary = False
        creator = False
        with _cache_lock:
            if acquisition_generation != _cache_generation:
                create_temporary = True
                pending = None
            else:
                cached = _websocket_session_cache.get(cache_key)
                if cached and not cached.busy and _is_websocket_reusable(cached.ws):
                    cached.busy = True
                    return cached.ws, cached, True, _make_release(cache_key, cached)
                if cached and cached.busy:
                    # Parallel turns on one session use an uncached temporary
                    # connection and keep the cached continuation chain isolated.
                    create_temporary = True
                elif cached:
                    _websocket_session_cache.pop(cache_key, None)
                    stale_ws = cached.ws

                pending = _pending_websocket_connections.get(cache_key)
                if not create_temporary and pending is None:
                    pending = threading.Event()
                    _pending_websocket_connections[cache_key] = pending
                    creator = True

        if stale_ws is not None:
            _close_websocket_silently(stale_ws)

        if create_temporary:
            ws = _connect_websocket(client, headers, **connect_options)
            return ws, None, False, lambda keep=True: _close_websocket_silently(ws)

        if not creator:
            pending.wait()
            continue

        try:
            ws = _connect_websocket(client, headers, **connect_options)
        except Exception:
            with _cache_lock:
                if _pending_websocket_connections.get(cache_key) is pending:
                    _pending_websocket_connections.pop(cache_key, None)
                pending.set()
            raise

        entry = CachedWebSocketConnection(ws=ws, cache_key=cache_key, busy=True)
        with _cache_lock:
            published = (
                acquisition_generation == _cache_generation
                and _pending_websocket_connections.get(cache_key) is pending
            )
            if published:
                _websocket_session_cache[cache_key] = entry
                _pending_websocket_connections.pop(cache_key, None)
            pending.set()
        if published:
            return ws, entry, False, _make_release(cache_key, entry)
        return ws, None, False, lambda keep=True: _close_websocket_silently(ws)


def _make_release(cache_key: tuple[str, str], entry: CachedWebSocketConnection):
    def release(*, keep: bool = True) -> None:
        with _cache_lock:
            if not keep or not _is_websocket_reusable(entry.ws):
                if _websocket_session_cache.get(cache_key) is entry:
                    _websocket_session_cache.pop(cache_key, None)
                entry.busy = False
                entry.continuation = None
                close_ws = entry.ws
            else:
                entry.busy = False
                entry.last_used_at = time.time()
                close_ws = None
        if close_ws is not None:
            _close_websocket_silently(close_ws)
    return release


def _recv_websocket_event(ws: Any) -> Any:
    raw_connection = getattr(ws, "_connection", None)
    receiver = getattr(raw_connection, "recv", None) if raw_connection is not None else None
    if not callable(receiver):
        receiver = getattr(ws, "recv")
    try:
        raw = receiver(timeout=WEBSOCKET_RECV_POLL_TIMEOUT_SECONDS)
    except TypeError:
        raw = receiver()

    if not isinstance(raw, (str, bytes, dict)):
        return raw
    parse_event = getattr(ws, "parse_event", None)
    if callable(parse_event):
        return parse_event(raw)
    return websocket_event_to_object(raw)


def get_cached_websocket_input_delta(
    body: Dict[str, Any],
    continuation: CachedWebSocketContinuationState,
) -> List[Dict[str, Any]] | None:
    if not request_bodies_match_except_input(body, continuation.last_request_body):
        return None
    current_input = body.get("input") or []
    baseline = list(continuation.last_request_body.get("input") or []) + list(continuation.last_response_items or [])
    if len(current_input) < len(baseline):
        return None
    prefix = current_input[: len(baseline)]
    if not responses_inputs_equal(prefix, baseline):
        return None
    return current_input[len(baseline) :]


def build_cached_websocket_request_body(entry: CachedWebSocketConnection, body: Dict[str, Any]) -> Dict[str, Any]:
    continuation = entry.continuation
    if continuation is None:
        return dict(body)
    requested_previous_response_id = body.get("previous_response_id")
    if (
        requested_previous_response_id
        and requested_previous_response_id != continuation.last_response_id
    ):
        entry.continuation = None
        return dict(body)
    delta = get_cached_websocket_input_delta(body, continuation)
    if delta is None or not continuation.last_response_id:
        entry.continuation = None
        return dict(body)
    request_body = dict(body)
    request_body["previous_response_id"] = continuation.last_response_id
    request_body["input"] = delta
    return request_body


def _obj(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _obj(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_obj(v) for v in value]
    return value


def websocket_event_to_object(raw: Any) -> SimpleNamespace:
    if isinstance(raw, str):
        data = json.loads(raw)
    elif isinstance(raw, bytes):
        data = json.loads(raw.decode("utf-8"))
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError(f"Unsupported WebSocket event payload: {type(raw)!r}")
    event_type = data.get("type")
    if event_type == "response.done":
        data = dict(data)
        response = data.get("response")
        response_status = response.get("status") if isinstance(response, dict) else None
        response_status = str(response_status or "completed").strip().lower()
        terminal_types = {
            "completed": "response.completed",
            "failed": "response.failed",
            "incomplete": "response.incomplete",
            "cancelled": "response.cancelled",
            "canceled": "response.cancelled",
        }
        data["type"] = terminal_types.get(response_status, "response.completed")
        if isinstance(response, dict) and not response.get("status"):
            response = dict(response)
            response["status"] = "completed"
            data["response"] = response
    return _obj(data)


def _event_to_dict(event: Any) -> Dict[str, Any]:
    if isinstance(event, Mapping):
        return {str(key): _event_value_to_json(value) for key, value in event.items()}
    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json", by_alias=True, exclude_none=True))
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {
        key: _event_value_to_json(value)
        for key, value in vars(event).items()
        if isinstance(key, str) and not key.startswith("_")
    }


def _event_value_to_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _event_value_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_event_value_to_json(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", by_alias=True, exclude_none=True)
    if hasattr(value, "__dict__"):
        return {
            key: _event_value_to_json(item)
            for key, item in vars(value).items()
            if isinstance(key, str) and not key.startswith("_")
        }
    return value


def _turn_state_from_event(event: Any) -> str | None:
    if getattr(event, "type", "") not in {
        "response.metadata",
        "codex.response.metadata",
    }:
        return None
    headers = _event_to_dict(event).get("headers")
    if not isinstance(headers, Mapping):
        return None
    for name, value in headers.items():
        if str(name).lower() == "x-codex-turn-state" and value is not None:
            state = str(value).strip()
            return state or None
    return None


def _with_codex_turn_state(body: Dict[str, Any], turn_state: str | None) -> Dict[str, Any]:
    request_body = dict(body)
    if not turn_state:
        return request_body
    metadata = request_body.get("client_metadata")
    if metadata is None:
        metadata_dict: Dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        metadata_dict = dict(metadata)
    else:
        raise ValueError("Codex Responses WebSocket client_metadata must be an object")
    metadata_dict["x-codex-turn-state"] = turn_state
    request_body["client_metadata"] = metadata_dict
    return request_body


def _build_headers_from_client(client: Any, api_kwargs: Dict[str, Any], session_id: str | None) -> Dict[str, str]:
    headers: Dict[str, str] = {}

    canonical_names = {
        "authorization": "Authorization",
        "openai-beta": "OpenAI-Beta",
        "originator": "originator",
        "session-id": "session-id",
        "thread-id": "thread-id",
        "x-client-request-id": "x-client-request-id",
    }

    def put_header(name: Any, value: Any) -> None:
        raw_name = str(name)
        lower_name = raw_name.lower()
        for existing in list(headers):
            if existing.lower() == lower_name:
                headers.pop(existing, None)
        headers[canonical_names.get(lower_name, raw_name)] = str(value)

    # responses.connect() adds SDK auth headers, but it does not carry the
    # OpenAI client's caller-supplied default_headers into the WebSocket
    # handshake. Codex relies on those custom headers for Cloudflare identity
    # and ChatGPT account routing. Read the SDK's dedicated custom-header map
    # instead of default_headers, which also contains Omit sentinel values.
    custom = getattr(client, "_custom_headers", None)
    if isinstance(custom, Mapping):
        for key, value in custom.items():
            if value is not None:
                put_header(key, value)
    extra = api_kwargs.get("extra_headers")
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            if value is not None:
                put_header(key, value)
    request_id = str(session_id or uuid.uuid4())
    put_header("OpenAI-Beta", OPENAI_BETA_RESPONSES_WEBSOCKETS)
    put_header("x-client-request-id", request_id)
    put_header("session-id", request_id)
    put_header("thread-id", request_id)
    if not any(name.lower() == "originator" for name in headers):
        put_header("originator", "hermes-agent")
    return headers


def _is_retryable_transport_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionError, OSError, TimeoutError)):
            return True
        status = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status is None:
            status = getattr(response, "status_code", None)
        try:
            status_code = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_code = None
        if status_code in {408, 409, 425, 429} or (
            status_code is not None and status_code >= 500
        ):
            return True
        name = type(current).__name__.lower()
        if "connectionclosed" in name or "websockettimeout" in name:
            return True
        current = current.__cause__ or current.__context__
    return False


def _response_items_for_continuation(final_response: Any) -> List[Dict[str, Any]]:
    """Generate continuation items through Hermes' adapter conversion path."""
    from agent.codex_responses_adapter import _chat_messages_to_responses_input, _normalize_codex_response

    msg, _ = _normalize_codex_response(final_response)
    msg_dict: Dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
    reasoning = getattr(msg, "codex_reasoning_items", None)
    if reasoning:
        msg_dict["codex_reasoning_items"] = reasoning
    message_items = getattr(msg, "codex_message_items", None)
    if message_items:
        msg_dict["codex_message_items"] = message_items
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        converted = []
        for tc in tool_calls:
            fn = getattr(tc, "function", SimpleNamespace(name=getattr(tc, "name", ""), arguments=getattr(tc, "arguments", "{}")))
            converted.append({
                "id": getattr(tc, "id", None),
                "call_id": getattr(tc, "call_id", None),
                "function": {"name": getattr(fn, "name", ""), "arguments": getattr(fn, "arguments", "{}")},
            })
        msg_dict["tool_calls"] = converted
    return [item for item in _chat_messages_to_responses_input([msg_dict]) if item.get("type") != "function_call_output"]


def run_codex_websocket_stream(
    *,
    api_kwargs: Dict[str, Any],
    client: Any,
    provider: str,
    base_url: str,
    session_id: str | None,
    transport: str,
    collect_events: Callable[[Iterable[Any], Callable[[], Any] | None], Any],
    interrupted: Callable[[], bool] | None = None,
    timeout: float | None = None,
    register_connection_abort: (
        Callable[[Callable[[], None]], Callable[[], None]] | None
    ) = None,
    turn_state: str | None = None,
    record_turn_state: Callable[[str], None] | None = None,
) -> Any:
    if transport not in {"websocket", "websocket-cached", "auto"}:
        raise WebSocketNotStartedError(
            f"Unsupported Codex WebSocket transport: {transport}",
            retryable=False,
        )
    if not is_supported_codex_websocket_backend(provider=provider, base_url=base_url):
        raise WebSocketNotStartedError(
            "Codex WebSocket transport currently supports only provider=openai-codex "
            "with the ChatGPT Codex backend. Use responses_transport=sse for this endpoint.",
            retryable=False,
        )

    url = resolve_codex_websocket_url(base_url)
    headers = _build_headers_from_client(client, api_kwargs, session_id)
    extra_query = api_kwargs.get("extra_query")
    if extra_query is not None and not isinstance(extra_query, Mapping):
        raise WebSocketNotStartedError(
            "Codex Responses WebSocket extra_query must be an object",
            retryable=False,
        )
    ws = None
    entry = None
    release = None
    unregister_abort = None
    started = False
    keep_connection = True
    try:
        ws, entry, _reused, release = _acquire_websocket(
            client,
            url,
            headers,
            session_id,
            extra_query=extra_query,
            timeout=timeout,
        )
        if register_connection_abort is not None:
            unregister_abort = register_connection_abort(
                lambda: _abort_websocket_silently(ws)
            )
        wire_body = build_codex_websocket_wire_body(api_kwargs)
        body = dict(wire_body)
        if transport == "websocket-cached" and entry is not None:
            body = build_cached_websocket_request_body(entry, body)
        body = _with_codex_turn_state(body, turn_state)
        # Once send() starts, the frame may have reached the kernel or backend
        # even if the call raises before returning. Treat that boundary as an
        # ambiguous started request so no outer retry can duplicate execution.
        started = True
        ws.send({"type": "response.create", **body})

        def iter_events() -> Iterable[Any]:
            last_event_at = time.monotonic()
            while True:
                if interrupted and interrupted():
                    raise InterruptedError("Agent interrupted during Codex WebSocket stream")
                try:
                    event = _recv_websocket_event(ws)
                except TimeoutError:
                    if timeout is not None and time.monotonic() - last_event_at >= timeout:
                        raise TimeoutError(
                            f"Codex WebSocket stream was idle for {timeout:g} seconds"
                        )
                    continue
                last_event_at = time.monotonic()
                event_type = getattr(event, "type", "")
                if event_type == "response.done":
                    event = websocket_event_to_object(_event_to_dict(event))
                    event_type = getattr(event, "type", "")
                response_turn_state = _turn_state_from_event(event)
                if response_turn_state and record_turn_state is not None:
                    record_turn_state(response_turn_state)
                if event_type == "error":
                    raise CodexWebSocketEventError(event)
                yield event
                if event_type in _TERMINAL_EVENTS:
                    break

        final_response = collect_events(iter_events(), None)
        status = str(getattr(final_response, "status", "") or "").lower()
        if status in {"failed", "cancelled", "incomplete"}:
            keep_connection = False
            if entry is not None:
                entry.continuation = None
        elif transport == "websocket-cached" and entry is not None:
            response_id = getattr(final_response, "id", None)
            if isinstance(response_id, str) and response_id:
                try:
                    response_items = _response_items_for_continuation(final_response)
                    entry.continuation = CachedWebSocketContinuationState(
                        last_request_body=wire_body,
                        last_response_id=response_id,
                        last_response_items=response_items,
                    )
                except Exception:
                    # Continuation compaction is a cache optimization. A
                    # conversion failure after a completed response must keep
                    # the successful result and send full context next turn.
                    entry.continuation = None
                    logger.debug(
                        "Could not seed Codex WebSocket continuation state",
                        exc_info=True,
                    )
        return final_response
    except InterruptedError:
        if entry is not None:
            entry.continuation = None
        keep_connection = False
        raise
    except Exception as exc:
        if entry is not None:
            entry.continuation = None
        keep_connection = False
        retryable = _is_retryable_transport_error(exc)
        if (
            isinstance(exc, CodexWebSocketEventError)
            and exc.code == WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE
        ):
            # Codex explicitly reports this frame when the 60-minute socket
            # lifetime expires and guarantees that the caller should reconnect
            # and continue. This is the one post-send response that establishes
            # a replay-safe outcome.
            raise WebSocketNotStartedError(
                str(exc),
                cause=exc,
                retryable=True,
            ) from exc
        if isinstance(exc, CodexWebSocketEventError):
            # A structured server error is a definitive rejection, so normal
            # status-aware recovery remains safe. Transport/send/receive errors
            # still take the ambiguous WebSocketStartedError path below.
            raise WebSocketRejectedError(
                str(exc),
                cause=exc,
                retryable=retryable,
            ) from exc
        if started:
            raise WebSocketStartedError(str(exc), cause=exc, retryable=retryable) from exc
        raise WebSocketNotStartedError(str(exc), cause=exc, retryable=retryable) from exc
    finally:
        if unregister_abort is not None:
            unregister_abort()
        if release is not None:
            release(keep=keep_connection)
