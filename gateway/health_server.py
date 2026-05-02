"""
Gateway HTTP health server.

Provides a lightweight HTTP server exposing /health and /health/detailed
endpoints for Docker health checks, monitoring, and dashboard liveness
detection.

Usage:
    The server starts automatically when the gateway runs.  Configure the
    bind address and port via config.yaml or environment variables:

    config.yaml:
        gateway_health:
          enabled: true
          host: "127.0.0.1"
          port: 8642

    Environment variables (override config.yaml):
        HERMES_HEALTH_ENABLED=1
        HERMES_HEALTH_HOST=0.0.0.0
        HERMES_HEALTH_PORT=8642

Endpoints:
    GET /health
        Returns 200 with {"status": "ok", "pid": <pid>}

    GET /health/detailed
        Returns 200 with full gateway runtime state including:
        - gateway_state, pid, start_time
        - active_agents count
        - platform connection statuses
        - updated_at timestamp

    GET /ready
        Returns 200 only when the gateway is fully started and at least
        one platform is connected.  Returns 503 otherwise.  Useful for
        Kubernetes readiness probes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8642


class _HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("health: %s", format % args)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/health/detailed":
            self._handle_health_detailed()
        elif self.path == "/ready":
            self._handle_ready()
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "not found"}).encode())

    def _handle_health(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = json.dumps({
            "status": "ok",
            "pid": os.getpid(),
        })
        self.wfile.write(body.encode())

    def _handle_health_detailed(self) -> None:
        state_fn = getattr(self.server, "_get_state", None)
        if state_fn is None:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "state unavailable"}).encode())
            return

        state = state_fn()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(state).encode())

    def _handle_ready(self) -> None:
        ready_fn = getattr(self.server, "_is_ready", None)
        if ready_fn is None or not ready_fn():
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "not ready"}).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ready"}).encode())


class HealthServer:
    """Lightweight HTTP health server for the gateway.

    Thread-safe — the state callbacks are invoked from the HTTP server
    thread, so callers should ensure their callbacks are thread-safe or
    use atomic reads.
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        get_state: Optional[Callable[[], Dict[str, Any]]] = None,
        is_ready: Optional[Callable[[], bool]] = None,
    ):
        self.host = host
        self.port = port
        self._get_state = get_state or (lambda: {})
        self._is_ready = is_ready or (lambda: False)
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the health server in a background daemon thread."""
        if self._server is not None:
            logger.warning("health server already started")
            return

        self._server = HTTPServer((self.host, self.port), _HealthHandler)
        self._server._get_state = self._get_state
        self._server._is_ready = self._is_ready
        self._server.daemon_threads = True

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="hermes-health-server",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Health server listening on http://%s:%d", self.host, self.port
        )

    def stop(self) -> None:
        """Shut down the health server."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Health server stopped")

    @property
    def is_running(self) -> bool:
        return self._server is not None


def resolve_health_config() -> tuple[bool, str, int]:
    """Resolve health server config from env vars and return (enabled, host, port).

    Priority:
    1. HERMES_HEALTH_* env vars
    2. Caller defaults (127.0.0.1:8642, disabled)
    """
    enabled = os.getenv("HERMES_HEALTH_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    host = os.getenv("HERMES_HEALTH_HOST", _DEFAULT_HOST).strip()
    raw_port = os.getenv("HERMES_HEALTH_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else _DEFAULT_PORT
    except (TypeError, ValueError):
        port = _DEFAULT_PORT
    return enabled, host, port
