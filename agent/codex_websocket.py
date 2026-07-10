"""Responses WebSocket transport for the OAuth-backed Codex endpoint.

The public OpenAI Python client only knows about the HTTP/SSE Responses
endpoint.  ChatGPT subscription models can advertise a WebSocket-first
transport, while still emitting the same ``response.*`` events.  This module
keeps the wire-specific code separate and feeds the existing Hermes event
consumer from :mod:`agent.codex_runtime`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

CODEX_RESPONSES_LITE_HEADER = "X-OpenAI-Internal-Codex-Responses-Lite"
CODEX_RESPONSES_LITE_CLIENT_METADATA_KEY = (
    "ws_request_header_x_openai_internal_codex_responses_lite"
)
CODEX_RESPONSES_WEBSOCKET_BETA = "responses_websockets=2026-02-06"


class CodexWebSocketError(ConnectionError):
    """A WebSocket failure with enough context to decide whether to fall back."""

    def __init__(
        self,
        message: str,
        *,
        started: bool = False,
        status_code: Optional[int] = None,
        safe_to_fallback: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.started = bool(started)
        self.status_code = status_code
        self.safe_to_fallback = (
            not self.started if safe_to_fallback is None else bool(safe_to_fallback)
        )


def responses_websocket_url(base_url: str) -> str:
    """Convert a Codex HTTP base URL into its Responses WebSocket URL."""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("Codex WebSocket transport requires a base URL")
    if url.endswith("/responses"):
        response_url = url
    else:
        response_url = f"{url}/responses"
    if response_url.startswith("https://"):
        return "wss://" + response_url[len("https://") :]
    if response_url.startswith("http://"):
        return "ws://" + response_url[len("http://") :]
    if response_url.startswith(("wss://", "ws://")):
        return response_url
    raise ValueError(f"Unsupported Codex WebSocket base URL: {base_url!r}")


def _set_header(headers: Dict[str, str], name: str, value: str) -> None:
    """Set a header without leaving a differently-cased duplicate behind."""
    for existing in list(headers):
        if existing.lower() == name.lower() and existing != name:
            del headers[existing]
    headers[name] = value


def _get_header(headers: Dict[str, str], name: str) -> Optional[str]:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value)
    return None


@lru_cache(maxsize=1)
def _codex_client_version() -> str:
    """Return the installed Codex CLI version for the WS edge header."""
    override = os.getenv("CODEX_CLI_VERSION", "").strip()
    if override:
        return override
    binary = shutil.which("codex")
    if binary:
        try:
            completed = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            match = re.search(r"\b(\d+\.\d+\.\d+)\b", completed.stdout or "")
            if match:
                return match.group(1)
        except Exception:
            pass
    return "0.0.0"


def build_codex_websocket_headers(
    agent: Any,
    api_kwargs: Dict[str, Any],
    *,
    use_responses_lite: bool,
) -> Dict[str, str]:
    """Build the first-party headers required by the Codex WS handshake."""
    headers: Dict[str, str] = {}
    client_kwargs = getattr(agent, "_client_kwargs", {})
    if isinstance(client_kwargs, dict):
        defaults = client_kwargs.get("default_headers")
        if isinstance(defaults, dict):
            headers.update({str(k): str(v) for k, v in defaults.items() if v is not None})

    # The default header builder is deliberately imported lazily: this module
    # must remain importable in tests and on installations that never use
    # OpenAI/Codex.
    try:
        from agent.auxiliary_client import _codex_cloudflare_headers

        first_party = _codex_cloudflare_headers(str(getattr(agent, "api_key", "") or ""))
        for key, value in first_party.items():
            if _get_header(headers, key) is None:
                _set_header(headers, key, str(value))
    except Exception:
        logger.debug("Could not build Codex first-party headers", exc_info=True)

    request_headers = api_kwargs.get("extra_headers")
    if isinstance(request_headers, dict):
        for key, value in request_headers.items():
            if value is not None:
                _set_header(headers, str(key), str(value))

    api_key = getattr(agent, "api_key", None)
    if (
        isinstance(api_key, str)
        and api_key.strip()
        and _get_header(headers, "Authorization") is None
    ):
        _set_header(headers, "Authorization", f"Bearer {api_key.strip()}")

    # The upstream Codex client sends this beta marker on its Responses
    # WebSocket handshake.  Preserve any caller-supplied beta values while
    # ensuring the transport marker is present.
    existing_beta = _get_header(headers, "OpenAI-Beta") or ""
    if "responses_websockets=" not in existing_beta:
        existing_beta = (
            f"{existing_beta},{CODEX_RESPONSES_WEBSOCKET_BETA}"
            if existing_beta
            else CODEX_RESPONSES_WEBSOCKET_BETA
        )
        _set_header(headers, "OpenAI-Beta", existing_beta)

    # Codex's edge expects a client version header on the WebSocket path.  The
    # exact CLI build is not semantically important to Hermes' wire protocol;
    # use the installed Codex CLI version when one is present so model catalog
    # minimum-version gates behave like they do for the first-party client.
    if _get_header(headers, "version") is None:
        _set_header(headers, "version", _codex_client_version())

    if use_responses_lite:
        _set_header(headers, CODEX_RESPONSES_LITE_HEADER, "true")

    return headers


def build_codex_websocket_request(
    api_kwargs: Dict[str, Any],
    *,
    use_responses_lite: bool = False,
) -> Dict[str, Any]:
    """Build the ``response.create`` JSON frame from Hermes request kwargs."""
    excluded = {"stream", "extra_headers", "extra_body", "timeout"}
    payload = {
        key: value
        for key, value in api_kwargs.items()
        if key not in excluded
    }
    extra_body = api_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        payload.update(extra_body)

    # Responses Lite is connection-scoped on the WebSocket upgrade, but the
    # upstream Codex client also records that choice in client metadata on the
    # response.create frame.  Preserve caller metadata while adding the exact
    # marker used by the first-party transport.
    if use_responses_lite:
        from agent.codex_runtime import _prepare_responses_lite_request_kwargs

        payload = _prepare_responses_lite_request_kwargs(payload)
        client_metadata = payload.get("client_metadata")
        if isinstance(client_metadata, dict):
            client_metadata = dict(client_metadata)
        else:
            client_metadata = {}
        client_metadata[CODEX_RESPONSES_LITE_CLIENT_METADATA_KEY] = "true"
        payload["client_metadata"] = client_metadata

    # The upstream wire shape is a tagged ``response.create`` frame whose
    # remaining fields are the regular Responses request body.
    payload["type"] = "response.create"
    payload["stream"] = True
    return payload


def _stream_idle_timeout(api_kwargs: Dict[str, Any]) -> float:
    configured = api_kwargs.get("timeout")
    if isinstance(configured, (int, float)) and not isinstance(configured, bool) and configured > 0:
        return max(30.0, min(float(configured), 300.0))
    return 120.0


def run_codex_websocket(
    agent: Any,
    api_kwargs: Dict[str, Any],
    *,
    use_responses_lite: bool,
    on_text_delta=None,
    on_reasoning_delta=None,
    on_first_delta=None,
    on_event=None,
    interrupt_check=None,
) -> Any:
    """Run one Codex Responses request over WebSocket.

    The returned object has the same ``SimpleNamespace`` shape as the HTTP
    path because both paths share ``_consume_codex_event_stream``.
    """
    try:
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect
    except ImportError as exc:  # pragma: no cover - dependency is core
        raise CodexWebSocketError("websockets dependency is unavailable") from exc

    from agent.codex_runtime import _consume_codex_event_stream

    base_url = getattr(agent, "base_url", None) or getattr(agent, "_base_url", None)
    ws_url = responses_websocket_url(str(base_url or ""))
    headers = build_codex_websocket_headers(
        agent,
        api_kwargs,
        use_responses_lite=use_responses_lite,
    )
    wire_request = build_codex_websocket_request(
        api_kwargs,
        use_responses_lite=use_responses_lite,
    )
    idle_timeout = _stream_idle_timeout(api_kwargs)
    state = {"started": False, "request_sent": False}

    def _events() -> Iterator[Dict[str, Any]]:
        connection = None
        last_event_at = time.monotonic()
        try:
            connection = connect(
                ws_url,
                additional_headers=headers,
                user_agent_header=None,
                compression="deflate",
                open_timeout=min(30.0, idle_timeout),
                ping_interval=20.0,
                ping_timeout=20.0,
                close_timeout=10.0,
            )
            setattr(agent, "_codex_active_websocket", connection)
            # Be conservative around partial writes: once send() is attempted,
            # the server may have accepted the request even if the client gets
            # an exception before send() returns.
            state["request_sent"] = True
            connection.send(json.dumps(wire_request, ensure_ascii=False, separators=(",", ":")))

            while True:
                if interrupt_check is not None and interrupt_check():
                    raise InterruptedError("Agent interrupted during Codex WebSocket stream")
                if getattr(agent, "_interrupt_requested", False):
                    raise InterruptedError("Agent interrupted during Codex WebSocket stream")
                if time.monotonic() - last_event_at > idle_timeout:
                    raise CodexWebSocketError(
                        f"Codex WebSocket idle timeout after {int(idle_timeout)}s",
                        started=state["started"],
                    )

                try:
                    raw = connection.recv(timeout=1.0)
                except TimeoutError:
                    continue
                except ConnectionClosed as exc:
                    raise CodexWebSocketError(
                        f"Codex WebSocket closed before response.completed: {exc}",
                        started=state["started"],
                        safe_to_fallback=not state["request_sent"],
                    ) from exc

                if raw is None:
                    raise CodexWebSocketError(
                        "Codex WebSocket closed before response.completed",
                        started=state["started"],
                        safe_to_fallback=not state["request_sent"],
                    )
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.debug("Ignoring non-UTF-8 Codex WebSocket frame")
                        continue
                if not isinstance(raw, str):
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("Ignoring malformed Codex WebSocket event: %r", raw[:200])
                    continue
                if not isinstance(event, dict):
                    continue

                event_type = event.get("type")
                if event_type == "error":
                    error = event.get("error")
                    if isinstance(error, dict):
                        message = error.get("message") or error.get("code") or str(error)
                    else:
                        message = str(error or "Codex WebSocket returned an error")
                    status = event.get("status", event.get("status_code"))
                    status_code = int(status) if isinstance(status, (int, float)) else None
                    prefix = f"HTTP {status_code}: " if status_code else ""
                    raise CodexWebSocketError(
                        f"{prefix}{message}",
                        started=state["started"],
                        status_code=status_code,
                        # An explicit error frame means the server rejected
                        # this request; replaying it over HTTP is safe when
                        # no response event has started, even though the
                        # response.create frame was sent.
                        safe_to_fallback=not state["started"],
                    )

                state["started"] = True
                last_event_at = time.monotonic()
                yield event
        except CodexWebSocketError:
            raise
        except InterruptedError:
            raise
        except Exception as exc:
            if getattr(agent, "_interrupt_requested", False):
                raise InterruptedError("Agent interrupted during Codex WebSocket stream") from exc
            raise CodexWebSocketError(
                f"Codex WebSocket transport failed: {exc}",
                started=state["started"],
                safe_to_fallback=not state["request_sent"],
            ) from exc
        finally:
            if getattr(agent, "_codex_active_websocket", None) is connection:
                setattr(agent, "_codex_active_websocket", None)
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    event_iter = _events()
    try:
        return _consume_codex_event_stream(
            event_iter,
            model=api_kwargs.get("model"),
            on_text_delta=on_text_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_first_delta=on_first_delta,
            on_event=on_event,
            interrupt_check=interrupt_check,
        )
    finally:
        close_iter = getattr(event_iter, "close", None)
        if callable(close_iter):
            close_iter()


def close_active_codex_websocket(agent: Any) -> None:
    """Abort a request-local WebSocket from Hermes' watchdog/interrupt thread."""
    connection = getattr(agent, "_codex_active_websocket", None)
    if connection is None:
        return
    try:
        connection.close()
    except Exception:
        logger.debug("Codex WebSocket abort failed", exc_info=True)


__all__ = [
    "CODEX_RESPONSES_LITE_HEADER",
    "CODEX_RESPONSES_LITE_CLIENT_METADATA_KEY",
    "CODEX_RESPONSES_WEBSOCKET_BETA",
    "CodexWebSocketError",
    "build_codex_websocket_headers",
    "build_codex_websocket_request",
    "close_active_codex_websocket",
    "responses_websocket_url",
    "run_codex_websocket",
]
