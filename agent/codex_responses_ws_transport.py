"""Generic WebSocket transport for opt-in custom Responses API providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit


VALID_TRANSPORTS = frozenset({"sse", "websocket", "auto"})
_SDK_ONLY_FIELDS = frozenset({
    "extra_body",
    "extra_headers",
    "extra_query",
    "stream",
    "timeout",
})


class GenericWsNotStartedError(RuntimeError):
    """The WebSocket request was never sent and may safely fall back to SSE."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class GenericWsStartedError(RuntimeError):
    """The request may have reached the server and must never be replayed."""


class GenericWsRejectedError(RuntimeError):
    """The server explicitly rejected the WebSocket request."""


def normalize_responses_transport(value: Any) -> str:
    """Return a supported transport name, defaulting unknown values to SSE."""
    transport = str(value or "").strip().lower()
    return transport if transport in VALID_TRANSPORTS else "sse"


def is_generic_codex_ws_eligible(*, provider: Any, base_url: Any, api_mode: Any) -> bool:
    """Whether this is a named custom Codex Responses endpoint, never ChatGPT."""
    provider_name = str(provider or "").strip().lower()
    try:
        host = urlsplit(str(base_url or "").strip()).hostname or ""
    except ValueError:
        return False
    return (
        provider_name.startswith("custom:")
        and str(api_mode or "").strip().lower() == "codex_responses"
        and provider_name != "openai-codex"
        and host.lower() != "chatgpt.com"
    )


def resolve_responses_ws_url(base_url: Any, override: Any = None) -> str:
    """Derive a Responses WebSocket endpoint from an OpenAI-compatible base URL."""
    configured = str(override or "").strip()
    if configured:
        return configured

    parsed = urlsplit(str(base_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("A valid base_url is required for the Responses WebSocket transport")
    scheme = {"https": "wss", "http": "ws"}.get(parsed.scheme.lower(), parsed.scheme)
    path = parsed.path.rstrip("/")
    if not path.lower().endswith("/responses"):
        path = f"{path}/responses" if path else "/responses"
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, ""))


def build_ws_wire_body(api_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Remove OpenAI-SDK-only options and flatten ``extra_body`` into the payload."""
    body = {
        key: value
        for key, value in api_kwargs.items()
        if key not in _SDK_ONLY_FIELDS
    }
    extra_body = api_kwargs.get("extra_body")
    if isinstance(extra_body, Mapping):
        body.update(extra_body)
    for key in _SDK_ONLY_FIELDS:
        body.pop(key, None)
    return body


def _connect_websocket(url: str, *, headers: Mapping[str, str], timeout: float):
    from websockets.sync.client import connect

    return connect(url, additional_headers=dict(headers) or None, open_timeout=timeout)


def _event_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _event_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_event_namespace(item) for item in value]
    return value


def _event_value(event: Mapping[str, Any], name: str) -> Any:
    value = event.get(name)
    if value is None:
        response = event.get("response")
        if isinstance(response, Mapping):
            value = response.get(name)
    return value


def _normalize_terminal_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") != "response.done":
        return event
    status = str(_event_value(event, "status") or "").strip().lower()
    if status == "completed":
        event["type"] = "response.completed"
    elif status == "failed":
        event["type"] = "response.failed"
    else:
        event["type"] = "response.incomplete"
    return event


def _server_error_message(event: Mapping[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, Mapping):
        message = error.get("message") or error.get("code")
    else:
        message = event.get("message") or error
    return str(message or "WebSocket server rejected the response request")


def _build_headers(
    *,
    api_kwargs: Mapping[str, Any],
    client: Any,
    api_key: Any,
    headers: Mapping[str, Any] | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in (getattr(client, "default_headers", None), headers, api_kwargs.get("extra_headers")):
        if isinstance(candidate, Mapping):
            result.update({str(key): str(value) for key, value in candidate.items()})

    key = api_key if api_key is not None else getattr(client, "api_key", None)
    if isinstance(key, str) and key and key != "no-key-required":
        if not any(name.lower() == "authorization" for name in result):
            result["Authorization"] = f"Bearer {key}"
    return result


def run_generic_codex_ws_stream(
    *,
    api_kwargs: Mapping[str, Any],
    client: Any = None,
    api_key: Any = None,
    headers: Mapping[str, Any] | None = None,
    provider: Any,
    base_url: Any,
    responses_ws_url: Any = None,
    session_id: Any = None,
    transport: Any,
    collect_events: Callable[[Any, Any], Any],
    interrupted: Callable[[], bool] | None,
    timeout: float = 15.0,
    register_connection_abort: Callable[[Callable[[str], None]], None] | None = None,
) -> Any:
    """Send one generic Responses request over WebSocket and collect its events.

    ``collect_events`` owns all Responses event semantics; this module only
    converts the wire frames and enforces the no-replay boundary at ``send``.
    """
    del session_id  # Generic providers do not share a session header contract.
    normalized_transport = normalize_responses_transport(transport)
    if normalized_transport == "sse":
        raise GenericWsNotStartedError("Responses WebSocket transport is disabled", retryable=False)
    if not is_generic_codex_ws_eligible(
        provider=provider,
        base_url=base_url,
        api_mode="codex_responses",
    ):
        raise GenericWsNotStartedError(
            "Responses WebSocket transport is only available for named custom Codex providers",
            retryable=False,
        )

    started = False
    try:
        url = resolve_responses_ws_url(base_url, responses_ws_url)
        connection = _connect_websocket(
            url,
            headers=_build_headers(
                api_kwargs=api_kwargs,
                client=client,
                api_key=api_key,
                headers=headers,
            ),
            timeout=float(timeout or 15.0),
        )
        with connection as websocket:
            def _abort(_reason: str) -> None:
                close = getattr(websocket, "close", None)
                if callable(close):
                    close()

            if register_connection_abort is not None:
                register_connection_abort(_abort)

            started = True
            websocket.send(json.dumps({"type": "response.create", **build_ws_wire_body(api_kwargs)}))

            def _events():
                while True:
                    if interrupted is not None and interrupted():
                        raise InterruptedError("Agent interrupted during Responses WebSocket stream")
                    frame = websocket.recv()
                    if isinstance(frame, bytes):
                        frame = frame.decode("utf-8")
                    event = json.loads(frame)
                    if not isinstance(event, dict):
                        continue
                    event = _normalize_terminal_event(event)
                    if event.get("type") == "error":
                        raise GenericWsRejectedError(_server_error_message(event))
                    yield _event_namespace(event)
                    if event.get("type") in {
                        "response.completed",
                        "response.failed",
                        "response.incomplete",
                    }:
                        return

            return collect_events(_events(), None)
    except (GenericWsNotStartedError, GenericWsStartedError, GenericWsRejectedError, InterruptedError):
        raise
    except Exception as exc:
        if started:
            raise GenericWsStartedError(f"Responses WebSocket stream failed after request start: {exc}") from exc
        raise GenericWsNotStartedError(f"Responses WebSocket connection failed: {exc}") from exc
