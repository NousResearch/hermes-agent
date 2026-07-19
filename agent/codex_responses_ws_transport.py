"""Generic WebSocket transport for opt-in custom Responses API providers."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

VALID_TRANSPORTS = frozenset({"sse", "websocket", "auto"})
_SDK_ONLY_FIELDS = frozenset(
    {
        "extra_body",
        "extra_headers",
        "extra_query",
        "stream",
        "timeout",
    }
)
_TERMINAL_EVENT_TYPES = frozenset(
    {
        "response.completed",
        "response.failed",
        "response.incomplete",
        "response.cancelled",
    }
)
DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
DEFAULT_RECV_POLL_SECONDS = 1.0
DEFAULT_IDLE_TIMEOUT_SECONDS = 180.0


class GenericWsNotStartedError(RuntimeError):
    """The WebSocket request was never sent and may safely fall back to SSE."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status_code: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.body = body


class GenericWsStartedError(RuntimeError):
    """The request may have reached the server and must never be replayed."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.body = body


class GenericWsRejectedError(RuntimeError):
    """The server explicitly rejected the WebSocket request."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.body = body


def normalize_responses_transport(value: Any) -> str:
    """Return a supported transport name, defaulting unknown values to SSE."""
    transport = str(value or "").strip().lower().replace("_", "-")
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


def build_generic_ws_identity(
    *,
    session_id: Any,
    transport_provider: Any,
    base_url: Any,
    model: Any,
    responses_ws_url: Any = None,
    transport: Any = None,
) -> tuple[Any, ...]:
    """Build a sticky-disable identity that includes WS endpoint and mode."""
    return (
        session_id,
        str(transport_provider or "").strip().lower(),
        str(base_url or "").rstrip("/"),
        str(model or ""),
        str(responses_ws_url or "").strip(),
        normalize_responses_transport(transport),
    )


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

    return connect(
        url,
        additional_headers=dict(headers) or None,
        open_timeout=timeout,
    )


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
    event_type = str(event.get("type") or "")
    if event_type == "response.canceled":
        event = dict(event)
        event["type"] = "response.cancelled"
        return event
    if event_type != "response.done":
        return event

    event = dict(event)
    status = str(_event_value(event, "status") or "").strip().lower()
    if status == "completed":
        event["type"] = "response.completed"
    elif status == "failed":
        event["type"] = "response.failed"
    elif status in {"cancelled", "canceled"}:
        event["type"] = "response.cancelled"
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


def _server_error_status(event: Mapping[str, Any]) -> int | None:
    candidates = [
        event.get("status"),
        event.get("status_code"),
    ]
    error = event.get("error")
    if isinstance(error, Mapping):
        candidates.extend([error.get("status"), error.get("status_code")])
    response = event.get("response")
    if isinstance(response, Mapping):
        candidates.extend([response.get("status_code"), response.get("status")])
    for candidate in candidates:
        if candidate is None or isinstance(candidate, bool):
            continue
        try:
            code = int(candidate)
        except (TypeError, ValueError):
            continue
        if 100 <= code < 600:
            return code
    return None


def _build_headers(
    *,
    api_kwargs: Mapping[str, Any],
    client: Any,
    api_key: Any,
    headers: Mapping[str, Any] | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in (
        getattr(client, "default_headers", None),
        headers,
        api_kwargs.get("extra_headers"),
    ):
        if isinstance(candidate, Mapping):
            result.update({str(key): str(value) for key, value in candidate.items()})

    key = api_key if api_key is not None else getattr(client, "api_key", None)
    if isinstance(key, str) and key and key != "no-key-required":
        if not any(name.lower() == "authorization" for name in result):
            result["Authorization"] = f"Bearer {key}"
    return result


def _recv_frame(websocket: Any, *, poll_timeout: float) -> Any:
    try:
        return websocket.recv(timeout=poll_timeout)
    except TypeError:
        # Some test doubles / older wrappers accept no timeout kwarg.
        return websocket.recv()


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
    timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    idle_timeout: float | None = None,
    recv_poll_timeout: float = DEFAULT_RECV_POLL_SECONDS,
    register_connection_abort: Callable[[Callable[[str], None]], None] | None = None,
) -> Any:
    """Send one generic Responses request over WebSocket and collect its events.

    ``collect_events`` owns all Responses event semantics; this module only
    converts the wire frames and enforces the no-replay boundary at ``send``.
    """
    del session_id  # Generic providers do not share a session header contract.
    normalized_transport = normalize_responses_transport(transport)
    if normalized_transport == "sse":
        raise GenericWsNotStartedError(
            "Responses WebSocket transport is disabled",
            retryable=False,
        )
    if not is_generic_codex_ws_eligible(
        provider=provider,
        base_url=base_url,
        api_mode="codex_responses",
    ):
        raise GenericWsNotStartedError(
            "Responses WebSocket transport is only available for named custom Codex providers",
            retryable=False,
        )

    connect_timeout = float(timeout or DEFAULT_CONNECT_TIMEOUT_SECONDS)
    poll_timeout = float(recv_poll_timeout or DEFAULT_RECV_POLL_SECONDS)
    if idle_timeout is None:
        # Prefer a generous idle budget, but never shorter than the connect timeout.
        idle_limit = max(float(DEFAULT_IDLE_TIMEOUT_SECONDS), connect_timeout)
    else:
        idle_limit = float(idle_timeout)
    if idle_limit <= 0:
        idle_limit = DEFAULT_IDLE_TIMEOUT_SECONDS

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
            timeout=connect_timeout,
        )
        with connection as websocket:
            def _abort(_reason: str) -> None:
                close = getattr(websocket, "close", None)
                if callable(close):
                    close()

            if register_connection_abort is not None:
                register_connection_abort(_abort)

            wire_body = build_ws_wire_body(api_kwargs)
            payload = json.dumps({"type": "response.create", **wire_body})
            # Mark started at the send boundary: once send is invoked the frame
            # may have left the process even if the call later raises.
            started = True
            websocket.send(payload)

            def _events():
                last_event_at = time.monotonic()
                while True:
                    if interrupted is not None and interrupted():
                        raise InterruptedError(
                            "Agent interrupted during Responses WebSocket stream"
                        )
                    try:
                        frame = _recv_frame(websocket, poll_timeout=poll_timeout)
                    except TimeoutError:
                        if time.monotonic() - last_event_at >= idle_limit:
                            raise TimeoutError(
                                f"Responses WebSocket stream idle for {idle_limit:g}s"
                            )
                        continue
                    except Exception as exc:
                        # websockets raises TimeoutError subclasses in some versions;
                        # also tolerate bare timeout-like messages from fakes.
                        if type(exc).__name__ in {"TimeoutError", "TimeoutException"}:
                            if time.monotonic() - last_event_at >= idle_limit:
                                raise TimeoutError(
                                    f"Responses WebSocket stream idle for {idle_limit:g}s"
                                ) from exc
                            continue
                        raise

                    last_event_at = time.monotonic()
                    if isinstance(frame, bytes):
                        frame = frame.decode("utf-8")
                    event = json.loads(frame)
                    if not isinstance(event, dict):
                        continue
                    event = _normalize_terminal_event(event)
                    if event.get("type") == "error":
                        raise GenericWsRejectedError(
                            _server_error_message(event),
                            status_code=_server_error_status(event),
                            body=event,
                        )
                    yield _event_namespace(event)
                    if event.get("type") in _TERMINAL_EVENT_TYPES:
                        return

            return collect_events(_events(), None)
    except (
        GenericWsNotStartedError,
        GenericWsStartedError,
        GenericWsRejectedError,
        InterruptedError,
    ):
        raise
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            status_code = None
        if started:
            raise GenericWsStartedError(
                f"Responses WebSocket stream failed after request start: {exc}",
                status_code=status_code,
            ) from exc
        raise GenericWsNotStartedError(
            f"Responses WebSocket connection failed: {exc}",
            status_code=status_code,
        ) from exc
