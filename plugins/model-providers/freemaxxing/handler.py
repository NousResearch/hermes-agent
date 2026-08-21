"""Authenticated OpenAI-compatible request handler."""

from __future__ import annotations

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Set

from .policy import (
    _MAX_REQUEST_BODY_BYTES,
    _MAX_SSE_EVENT_BYTES,
    _MAX_SSE_LINE_BYTES,
    AuthError,
    ClientRequestError,
    ModelNotFoundError,
    RateLimitError,
    TransientError,
)
from .pool import pool
from .upstream import _exhausted_message, _forward, _open_stream

logger = logging.getLogger("freemaxxing.proxy")

def _expected_token(handler: BaseHTTPRequestHandler) -> str:
    return str(getattr(handler.server, "auth_token", "") or "")


def _authorized(handler: BaseHTTPRequestHandler) -> bool:
    expected = _expected_token(handler)
    supplied = handler.headers.get("Authorization", "")
    return bool(expected) and hmac.compare_digest(
        supplied,
        f"Bearer {expected}",
    )


def _ensure_server_pool(handler: BaseHTTPRequestHandler) -> None:
    server = handler.server
    if bool(getattr(server, "pool_ready", False)):
        return
    lock = getattr(server, "pool_init_lock")
    with lock:
        if bool(getattr(server, "pool_ready", False)):
            return
        initializer = getattr(server, "pool_initializer", None)
        if initializer is None:
            server.pool_ready = True
            return
        initializer()
        server.pool_ready = True


class ChatCompletionsHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - %s", self.client_address[0], format % args)

    def _send_json(self, code: int, body: Any) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except OSError:
            pass
        self.close_connection = True

    def _send_error(
        self,
        code: int,
        message: str,
        error_type: str = "freemaxxing_error",
    ) -> None:
        self._send_json(
            code,
            {"error": {"message": message, "type": error_type}},
        )

    def _require_auth(self) -> bool:
        if _authorized(self):
            return True
        self._send_error(401, "Unauthorized", error_type="unauthorized")
        return False

    def _require_runtime(self) -> bool:
        try:
            _ensure_server_pool(self)
            return True
        except RuntimeError as exc:
            self._send_error(
                409,
                str(exc),
                error_type="profile_isolation",
            )
            return False
        except Exception as exc:
            logger.exception("freemaxxing: pool initialization failed")
            self._send_error(
                503,
                f"Freemaxxing runtime unavailable: {exc}",
                error_type="runtime_unavailable",
            )
            return False

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/v1/healthz":
            self._send_json(
                200,
                {"service": "freemaxxing", "status": "ok"},
            )
            return

        if not self._require_auth():
            return
        if path == "/healthz":
            self._send_json(
                200,
                {"service": "freemaxxing", "health": pool.health()},
            )
            return
        if path == "/v1/models":
            if not self._require_runtime():
                return
            self._send_json(
                200,
                {"object": "list", "data": pool.get_aggregated_models()},
            )
            return
        self._send_error(404, f"Unknown path: {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._send_error(404, f"Unknown path: {path}")
            return
        if not self._require_auth() or not self._require_runtime():
            return
        self._handle_chat_completions()

    def _read_body(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            self._send_error(400, "Invalid Content-Length")
            return None
        if length > _MAX_REQUEST_BODY_BYTES:
            self._send_error(413, "Request body too large")
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error(400, "Invalid JSON body")
            return None
        if not isinstance(payload, dict):
            self._send_error(400, "JSON body must be an object")
            return None
        return payload

    def _handle_chat_completions(self) -> None:
        body = self._read_body()
        if body is None:
            return
        model = str(body.get("model", "") or "")
        if bool(body.get("stream", False)):
            self._handle_streaming(body, model)
        else:
            self._handle_nonstreaming(body, model)

    def _handle_nonstreaming(
        self,
        body: Dict[str, Any],
        model: str,
    ) -> None:
        tried: Set[str] = set()
        last_error: Optional[str] = None

        while len(tried) < max(1, pool.count()):
            backend = pool.next(model, exclude=tried)
            if backend is None:
                break
            if backend.name in tried:
                continue
            tried.add(backend.name)
            try:
                response = _forward(backend, body)
                self._send_json(200, response)
                backend.record_success()
                logger.info(
                    "freemaxxing: model=%s selected=%s tier=%d attempted=%d",
                    model,
                    backend.name,
                    backend.tier,
                    len(tried),
                )
                return
            except RateLimitError as exc:
                backend.record_failure(exc.retry_after, "rate_limit")
                last_error = (
                    f"rate limited on {backend.name} "
                    f"(retry after {exc.retry_after:.0f}s)"
                )
            except ModelNotFoundError as exc:
                last_error = str(exc)
            except AuthError as exc:
                last_error = str(exc)
            except ClientRequestError as exc:
                self._send_error(400, str(exc), error_type="invalid_request")
                return
            except TransientError as exc:
                backend.record_failure(10.0, "transient")
                last_error = str(exc)
            except Exception:
                logger.exception(
                    "freemaxxing: internal failure while using %s",
                    backend.name,
                )
                self._send_error(
                    500,
                    "Freemaxxing internal proxy error",
                    error_type="internal_error",
                )
                return

        self._send_error(503, _exhausted_message(last_error))

    def _handle_streaming(
        self,
        body: Dict[str, Any],
        model: str,
    ) -> None:
        tried: Set[str] = set()
        last_error: Optional[str] = None

        while len(tried) < max(1, pool.count()):
            backend = pool.next(model, exclude=tried)
            if backend is None:
                break
            if backend.name in tried:
                continue
            tried.add(backend.name)
            try:
                upstream = _open_stream(backend, body)
            except RateLimitError as exc:
                backend.record_failure(exc.retry_after, "rate_limit")
                last_error = f"rate limited on {backend.name}"
                continue
            except ModelNotFoundError as exc:
                last_error = str(exc)
                continue
            except AuthError as exc:
                last_error = str(exc)
                continue
            except ClientRequestError as exc:
                self._send_error(400, str(exc), error_type="invalid_request")
                return
            except TransientError as exc:
                backend.record_failure(10.0, "transient")
                last_error = str(exc)
                continue
            except Exception:
                logger.exception(
                    "freemaxxing: internal stream-open failure on %s",
                    backend.name,
                )
                self._send_error(
                    500,
                    "Freemaxxing internal proxy error",
                    error_type="internal_error",
                )
                return

            # Do not commit a 200 to the downstream client until the upstream
            # proves it can produce the first bounded SSE line. A backend that
            # closes or corrupts the stream immediately is still eligible for
            # transparent failover.
            try:
                first_chunk = upstream.readline(_MAX_SSE_LINE_BYTES + 1)
            except Exception as exc:
                try:
                    upstream.close()
                except Exception:
                    pass
                backend.record_failure(10.0, "stream_interrupted")
                last_error = (
                    f"backend {backend.name} stream failed before commit: {exc}"
                )
                continue
            if not first_chunk or len(first_chunk) > _MAX_SSE_LINE_BYTES:
                try:
                    upstream.close()
                except Exception:
                    pass
                backend.record_failure(10.0, "stream_interrupted")
                last_error = (
                    f"backend {backend.name} stream ended before commit"
                )
                continue

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            completed = False
            upstream_interrupted = False
            downstream_cancelled = False
            event_bytes = 0
            pending: Optional[bytes] = first_chunk
            try:
                while True:
                    if pending is not None:
                        chunk = pending
                        pending = None
                    else:
                        try:
                            chunk = upstream.readline(_MAX_SSE_LINE_BYTES + 1)
                        except Exception as exc:
                            upstream_interrupted = True
                            logger.warning(
                                "freemaxxing: upstream stream read failed on %s: %s",
                                backend.name,
                                exc,
                            )
                            break
                    if not chunk:
                        if not completed:
                            upstream_interrupted = True
                        break
                    if len(chunk) > _MAX_SSE_LINE_BYTES:
                        upstream_interrupted = True
                        logger.warning(
                            "freemaxxing: oversized SSE line from %s",
                            backend.name,
                        )
                        break

                    if chunk in {b"\n", b"\r\n"}:
                        event_bytes = 0
                    else:
                        event_bytes += len(chunk)
                        if event_bytes > _MAX_SSE_EVENT_BYTES:
                            upstream_interrupted = True
                            logger.warning(
                                "freemaxxing: oversized SSE event from %s",
                                backend.name,
                            )
                            break

                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except OSError:
                        downstream_cancelled = True
                        break
                    if b"data: [DONE]" in chunk:
                        completed = True
            finally:
                try:
                    upstream.close()
                except Exception:
                    pass

            if downstream_cancelled:
                logger.info(
                    "freemaxxing: downstream cancelled stream from %s; "
                    "backend health unchanged",
                    backend.name,
                )
            elif completed and not upstream_interrupted:
                backend.record_success()
                logger.info(
                    "freemaxxing: model=%s selected=%s tier=%d (streaming)",
                    model,
                    backend.name,
                    backend.tier,
                )
            else:
                backend.record_failure(10.0, "stream_interrupted")
            return

        self._send_error(503, _exhausted_message(last_error))
