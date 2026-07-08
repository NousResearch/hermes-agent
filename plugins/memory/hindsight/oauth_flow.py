"""Browser "Connect" flow for the Hindsight memory provider — no CLI step.

This is the counter-proposal to a JWT key-minting endpoint. Hermes never touches
OAuth or JWTs. It opens the browser to the Hindsight UI — where the user is
already signed in — and the UI creates an API key through its existing
key-creation path and hands it back to the desktop over a ``127.0.0.1`` loopback
redirect (the ``gh`` / ``gcloud auth login`` pattern). A Hindsight API key
authenticates exactly like a pasted key, so it is stored as the provider's
``apiKey``.

Plugs into the generic memory-provider connect framework in
``hermes_cli.memory_oauth`` via the ``start_loopback_flow_background`` /
``get_flow_status`` convention — same wiring Honcho uses.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urlparse, parse_qs

logger = logging.getLogger(__name__)

LOOPBACK_HOST = "127.0.0.1"

# The Hindsight UI route that mints a key for the desktop and redirects it back
# to the loopback. Env-overridable for dev / self-hosted / tests.
_CLOUD_CONNECT_URL = "https://ui.hindsight.vectorize.io/connect/desktop"

# Pending connects live only until their callback returns; the loopback is
# single-use with a bounded wait, so there's no long-lived pending registry.
_DEFAULT_TIMEOUT = 300.0


def resolve_connect_url() -> str:
    """The UI /connect/desktop route (env-overridable)."""
    return os.environ.get("HINDSIGHT_CONNECT_URL", _CLOUD_CONNECT_URL).rstrip("/")


def resolve_config_path() -> Path:
    """The Hindsight provider's config file (where it reads ``apiKey``)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "hindsight" / "config.json"


def build_connect_url(
    port: int,
    state: str,
    *,
    source: str = "hermes-desktop",
    hostname: str | None = None,
) -> str:
    """The URL the desktop opens: UI mints a key and redirects it to the loopback."""
    params = {"port": str(port), "state": state, "source": source}
    if hostname:
        params["host"] = hostname
    return f"{resolve_connect_url()}?{urlencode(params)}"


def _store_api_key(config_path: Path, api_key: str) -> None:
    """Persist the minted key as the provider's ``apiKey`` (0600)."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    data["apiKey"] = api_key
    data.setdefault("mode", "cloud")
    try:
        from utils import atomic_json_write

        atomic_json_write(config_path, data, mode=0o600)
    except Exception:
        config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(config_path, 0o600)
        except OSError:
            pass


_CALLBACK_HTML = (
    b"<!doctype html><meta charset=utf-8><title>Hindsight connected</title>"
    b"<body style='font:14px ui-monospace,monospace;background:#0b0e14;color:#c9d1d9;"
    b"display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
    b"<div>Connected to Hindsight. You can close this tab and return to Hermes.</div>"
)


def _bind_loopback_server() -> tuple[HTTPServer, dict[str, str]]:
    """Bind a one-shot ``127.0.0.1`` callback server on a random port."""
    captured: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib API name
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            params = parse_qs(parsed.query)
            captured["key"] = (params.get("key") or [""])[0]
            captured["state"] = (params.get("state") or [""])[0]
            captured["error"] = (params.get("error") or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_CALLBACK_HTML)

        def log_message(self, *args):  # silence stdlib request logging
            return

    server = HTTPServer((LOOPBACK_HOST, 0), _Handler)
    return server, captured


def capture_loopback_key(
    server: HTTPServer, captured: dict[str, str], *, timeout: float = _DEFAULT_TIMEOUT
) -> tuple[str, str]:
    """Serve a single ``/callback`` GET and return ``(key, state)``."""
    server.timeout = timeout
    try:
        deadline = time.monotonic() + timeout
        while "key" not in captured and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if captured.get("error"):
        raise ValueError(f"connect denied: {captured['error']}")
    if not captured.get("key"):
        raise TimeoutError("no connect callback received before timeout")
    return captured["key"], captured.get("state", "")


def connect_via_loopback(
    *,
    config_path: Path | None = None,
    open_url: Callable[[str], None] | None = None,
    source: str = "hermes-desktop",
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """Drive the full connect flow: open browser → capture minted key → store it.

    ``open_url`` defaults to the system browser; tests inject a driver that
    follows the UI redirect into the loopback callback.
    """
    server, captured = _bind_loopback_server()
    port = server.server_address[1]
    state = secrets.token_urlsafe(32)
    url = build_connect_url(port, state, source=source, hostname=socket.gethostname())

    if open_url is None:
        import webbrowser

        open_url = webbrowser.open

    # Browser opens from a short-lived thread; the socket is already bound, so a
    # fast redirect can't beat it.
    threading.Thread(target=lambda: open_url(url), daemon=True).start()

    key, returned_state = capture_loopback_key(server, captured, timeout=timeout)
    if returned_state != state:
        raise ValueError("connect state mismatch — possible CSRF, aborting")
    if not key.startswith("hsk_"):
        raise ValueError(
            "connect returned an unexpected credential (not a Hindsight API key)"
        )

    _store_api_key(config_path or resolve_config_path(), key)
    logger.info("Hindsight connect: stored a browser-provisioned API key")
    return key


# — Background launcher + status, for the desktop "Connect" button —
# The flow blocks on a browser round-trip, so the connect endpoint kicks it off
# in a thread and the UI polls status rather than holding the request open.


@dataclass
class FlowStatus:
    state: str = "idle"  # idle | pending | connected | error
    detail: str = ""


_status = FlowStatus()
_status_lock = threading.Lock()
_flow_thread: threading.Thread | None = None


def _detect_connection() -> tuple[bool, str | None]:
    """Whether a Hindsight credential is already stored."""
    try:
        path = resolve_config_path()
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and (data.get("apiKey") or data.get("api_key")):
                return True, "apikey"
    except Exception:
        pass
    if os.environ.get("HINDSIGHT_API_KEY"):
        return True, "apikey"
    return False, None


def _set_status(state: str, detail: str = "") -> None:
    with _status_lock:
        _status.state, _status.detail = state, detail


def get_flow_status() -> dict[str, object]:
    with _status_lock:
        state, detail = _status.state, _status.detail
    connected, auth = _detect_connection()
    return {"state": state, "detail": detail, "connected": connected, "auth": auth}


def start_loopback_flow_background(
    *,
    config_path: Path | None = None,
    source: str = "hermes-desktop",
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Launch the connect flow in a daemon thread; returns the initial status.

    Idempotent while a flow is pending — a second call is a no-op so a
    double-clicked button can't open two browser tabs.
    """
    global _flow_thread
    config_path = config_path or resolve_config_path()
    with _status_lock:
        if _status.state == "pending" and _flow_thread and _flow_thread.is_alive():
            return get_flow_status()
        _status.state, _status.detail = "pending", "waiting for browser consent"

    def _run() -> None:
        try:
            connect_via_loopback(
                config_path=config_path, source=source, timeout=timeout
            )
            _set_status("connected", "Hindsight connected")
        except Exception as exc:
            logger.warning("Hindsight connect loopback flow failed: %s", exc)
            _set_status("error", str(exc))

    _flow_thread = threading.Thread(
        target=_run, name="hindsight-connect-loopback", daemon=True
    )
    _flow_thread.start()
    return get_flow_status()
