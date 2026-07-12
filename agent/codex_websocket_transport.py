"""Opt-in WebSocket transport for ChatGPT Codex Responses.

Initial scope is intentionally narrow: ``provider == "openai-codex"`` using the
ChatGPT Codex backend URL (``https://chatgpt.com/backend-api/codex`` or the
resolved ``.../codex/responses`` equivalent). Generic OpenAI Responses endpoints
keep the existing SSE path because their ``store=False`` continuation semantics
are different from ChatGPT Codex's connection-scoped continuation.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
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
CODEX_RESPONSES_WIRE_FIELDS = frozenset({
    "model",
    "instructions",
    "input",
    "tools",
    "store",
    "reasoning",
    "include",
    "max_output_tokens",
    "temperature",
    "tool_choice",
    "parallel_tool_calls",
    "prompt_cache_key",
    "service_tier",
    "previous_response_id",
})
SESSION_WEBSOCKET_CACHE_TTL_SECONDS = 5 * 60
_TERMINAL_EVENTS = {
    "response.completed",
    "response.done",
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


def _require_websockets_connect() -> Callable[..., Any]:
    try:
        from websockets.sync.client import connect
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Codex Responses WebSocket transport requires the optional "
            "'websockets' package. Install Hermes with the WebSocket extra or "
            "run `pip install websockets`."
        ) from exc
    return connect


def resolve_codex_responses_url(base_url: str | None) -> str:
    """Resolve the HTTP Responses URL using pi-compatible Codex rules.

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

    Exact rule, matching pi v0.71.1: first resolve the HTTP Responses URL via
    ``resolve_codex_responses_url()``, then replace ``https`` with ``wss`` and
    ``http`` with ``ws``. Example:
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
    return {k: v for k, v in (body or {}).items() if k not in {"input", "previous_response_id"}}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_bodies_match_except_input(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _stable_json(request_body_without_input(a)) == _stable_json(request_body_without_input(b))


def responses_inputs_equal(a: Any, b: Any) -> bool:
    return _stable_json(a) == _stable_json(b)


def build_codex_websocket_wire_body(api_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return fields accepted inside a WebSocket ``response.create`` frame.

    OpenAI SDK controls such as ``extra_headers``, ``extra_body``, ``timeout``,
    and ``stream`` belong to the HTTP client or handshake layer and must never
    become JSON body fields on the ChatGPT Codex backend.
    """
    return {key: value for key, value in api_kwargs.items() if key in CODEX_RESPONSES_WIRE_FIELDS}


@dataclass
class CachedWebSocketContinuationState:
    last_request_body: Dict[str, Any]
    last_response_id: str
    last_response_items: List[Dict[str, Any]]


@dataclass
class CachedWebSocketConnection:
    ws: Any
    busy: bool = False
    continuation: CachedWebSocketContinuationState | None = None
    last_used_at: float = 0.0


class WebSocketStartedError(RuntimeError):
    """Raised after the WebSocket request was sent; auto mode must not fallback."""


class WebSocketNotStartedError(RuntimeError):
    """Raised before a WebSocket request was sent; auto mode may fallback."""


_websocket_session_cache: Dict[str, CachedWebSocketConnection] = {}

# A process-wide lock is enough: entries are held briefly only while acquiring or
# releasing. The actual socket read/write stays outside this lock.
import threading

_cache_lock = threading.RLock()


def cleanup_codex_websocket_sessions() -> None:
    with _cache_lock:
        entries = list(_websocket_session_cache.items())
        _websocket_session_cache.clear()
    for _, entry in entries:
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
    closed = getattr(ws, "closed", False)
    if closed:
        return False
    state = getattr(ws, "state", None)
    if isinstance(state, str) and state.upper() in {"CLOSED", "CLOSING"}:
        return False
    return True


def _close_websocket_silently(ws: Any) -> None:
    close = getattr(ws, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _connect_websocket(url: str, headers: list[tuple[str, str]], *, timeout: float | None = None) -> Any:
    connect = _require_websockets_connect()
    kwargs: Dict[str, Any] = {}
    if timeout is not None:
        kwargs["open_timeout"] = timeout

    # Proxy/SSL note: websockets.sync.client.connect uses environment proxy
    # support in modern websockets releases, but it cannot inherit custom httpx
    # transports mounted on the OpenAI SDK client. Keep support scoped to the
    # native ChatGPT Codex backend for this first PR. Users with enterprise
    # custom SDK transports should keep responses_transport=sse.
    if os.getenv("SSL_CERT_FILE"):
        ctx = ssl.create_default_context(cafile=os.getenv("SSL_CERT_FILE"))
        kwargs["ssl"] = ctx

    try:
        return connect(url, additional_headers=headers, **kwargs)
    except TypeError:
        return connect(url, extra_headers=headers, **kwargs)


def _acquire_websocket(url: str, headers: list[tuple[str, str]], session_id: str | None, *, timeout: float | None = None):
    cleanup_stale_codex_websocket_sessions()
    if not session_id:
        ws = _connect_websocket(url, headers, timeout=timeout)
        return ws, None, False, lambda keep=True: _close_websocket_silently(ws)

    with _cache_lock:
        cached = _websocket_session_cache.get(session_id)
        if cached and not cached.busy and _is_websocket_reusable(cached.ws):
            cached.busy = True
            return cached.ws, cached, True, _make_release(session_id, cached)
        if cached and cached.busy:
            ws = _connect_websocket(url, headers, timeout=timeout)
            return ws, None, False, lambda keep=True: _close_websocket_silently(ws)
        if cached:
            _websocket_session_cache.pop(session_id, None)
            _close_websocket_silently(cached.ws)

    ws = _connect_websocket(url, headers, timeout=timeout)
    entry = CachedWebSocketConnection(ws=ws, busy=True)
    with _cache_lock:
        _websocket_session_cache[session_id] = entry
    return ws, entry, False, _make_release(session_id, entry)


def _make_release(session_id: str, entry: CachedWebSocketConnection):
    def release(*, keep: bool = True) -> None:
        with _cache_lock:
            if not keep or not _is_websocket_reusable(entry.ws):
                if _websocket_session_cache.get(session_id) is entry:
                    _websocket_session_cache.pop(session_id, None)
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
        data["type"] = "response.completed"
        response = data.get("response")
        if isinstance(response, dict) and not response.get("status"):
            response = dict(response)
            response["status"] = "completed"
            data["response"] = response
    return _obj(data)


def _build_headers_from_client(client: Any, api_kwargs: Dict[str, Any], session_id: str | None) -> list[tuple[str, str]]:
    headers: Dict[str, str] = {}
    default_headers = getattr(client, "default_headers", None)
    if isinstance(default_headers, Mapping):
        headers.update({str(k): str(v) for k, v in default_headers.items() if v is not None})
    extra = api_kwargs.get("extra_headers")
    if isinstance(extra, dict):
        headers.update({str(k): str(v) for k, v in extra.items() if v is not None})
    for key in ("accept", "content-type", "OpenAI-Beta", "openai-beta"):
        headers.pop(key, None)
    api_key = getattr(client, "api_key", None)
    if api_key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"
    request_id = str(session_id or uuid.uuid4())
    headers["OpenAI-Beta"] = OPENAI_BETA_RESPONSES_WEBSOCKETS
    headers["x-client-request-id"] = request_id
    headers["session_id"] = request_id
    headers.setdefault("originator", "hermes-agent")
    return list(headers.items())


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
) -> Any:
    if transport not in {"websocket", "websocket-cached", "auto"}:
        raise WebSocketNotStartedError(f"Unsupported Codex WebSocket transport: {transport}")
    if not is_supported_codex_websocket_backend(provider=provider, base_url=base_url):
        raise WebSocketNotStartedError(
            "Codex WebSocket transport currently supports only provider=openai-codex "
            "with the ChatGPT Codex backend. Use responses_transport=sse for this endpoint."
        )

    url = resolve_codex_websocket_url(base_url)
    headers = _build_headers_from_client(client, api_kwargs, session_id)
    ws = None
    entry = None
    release = None
    started = False
    keep_connection = True
    try:
        ws, entry, _reused, release = _acquire_websocket(url, headers, session_id, timeout=timeout)
        wire_body = build_codex_websocket_wire_body(api_kwargs)
        body = dict(wire_body)
        if transport == "websocket-cached" and entry is not None:
            body = build_cached_websocket_request_body(entry, body)
        ws.send(json.dumps({"type": "response.create", **body}, ensure_ascii=False))
        started = True

        def iter_events() -> Iterable[Any]:
            while True:
                if interrupted and interrupted():
                    raise InterruptedError("Agent interrupted during Codex WebSocket stream")
                raw = _recv_websocket_frame(ws)
                event = websocket_event_to_object(raw)
                event_type = getattr(event, "type", "")
                if event_type == "error":
                    message = getattr(event, "message", None) or json.dumps(raw, ensure_ascii=False)
                    raise RuntimeError(f"Codex WebSocket error: {message}")
                yield event
                if event_type in _TERMINAL_EVENTS:
                    break

        final_response = collect_events(iter_events(), None)
        status = str(getattr(final_response, "status", "") or "").lower()
        if status in {"failed", "cancelled"}:
            keep_connection = False
            if entry is not None:
                entry.continuation = None
        elif transport == "websocket-cached" and entry is not None:
            response_id = getattr(final_response, "id", None)
            if isinstance(response_id, str) and response_id:
                entry.continuation = CachedWebSocketContinuationState(
                    last_request_body=wire_body,
                    last_response_id=response_id,
                    last_response_items=_response_items_for_continuation(final_response),
                )
        return final_response
    except Exception as exc:
        if entry is not None:
            entry.continuation = None
        keep_connection = False
        if started:
            raise WebSocketStartedError(str(exc)) from exc
        raise WebSocketNotStartedError(str(exc)) from exc
    finally:
        if release is not None:
            release(keep=keep_connection)
