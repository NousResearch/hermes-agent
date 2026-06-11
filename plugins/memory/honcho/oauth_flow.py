"""Zero-CLI OAuth authorization flow for the Honcho memory provider.

The user clicks "Connect to Honcho" in the desktop app; this module runs the
RFC 8252 native-app authorization-code + PKCE flow and writes the resulting
grant to honcho.json — no terminal step. ``begin_authorization`` /
``complete_authorization`` are the transport-agnostic core: the code can arrive
via the loopback listener here or (later) an Electron ``hermes://`` handler;
both funnel into ``complete_authorization``.

Endpoints are config-driven (env first, local-dev defaults) because the
dashboard hosts ``/authorize`` while the API hosts ``/oauth/token`` — two
different origins. Production deployments set the env overrides.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlparse

from plugins.memory.honcho import oauth
from plugins.memory.honcho.client import resolve_active_host, resolve_config_path

logger = logging.getLogger(__name__)

# The loopback redirect registered for the Hermes OAuth client.
LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_PORT = 8765
LOOPBACK_REDIRECT_URI = "http://localhost:8765/callback"

# Pending authorizations live only until their callback returns; keyed by the
# CSRF ``state`` so a stray/forged callback can't complete a grant.
_PENDING_TTL_SECONDS = 600


@dataclass(frozen=True)
class OAuthEndpoints:
    """Resolved authorization-server URLs and client identity."""

    authorize_url: str  # dashboard /authorize
    token_url: str  # API /oauth/token
    client_id: str
    scope: str


# Cloud (production) hosts; dashboard serves /authorize, API serves /oauth/token.
_CLOUD_DASHBOARD = "https://app.honcho.dev"
_CLOUD_TOKEN_URL = "https://api.honcho.dev/oauth/token"
_LOCAL_DASHBOARD = "http://localhost:3000"
_LOCAL_TOKEN_URL = "http://localhost:8000/oauth/token"


def _is_loopback_url(url: str | None) -> bool:
    return bool(url) and any(h in url for h in ("localhost", "127.0.0.1", "::1"))


def resolve_endpoints(
    environment: str | None = None, base_url: str | None = None
) -> OAuthEndpoints:
    """Resolve OAuth endpoints, zero-config by default.

    Keys off the host's honcho ``environment`` (production → cloud, local →
    localhost); a self-hosted ``base_url`` derives the token endpoint from the
    API host. Env vars override every field for unusual deployments.
    """
    if environment is None or base_url is None:
        try:
            from plugins.memory.honcho.client import HonchoClientConfig

            cfg = HonchoClientConfig.from_global_config()
            environment = environment or cfg.environment
            base_url = base_url if base_url is not None else cfg.base_url
        except Exception:
            environment = environment or "production"

    is_local = (environment or "").lower() == "local" or _is_loopback_url(base_url)
    default_dashboard = _LOCAL_DASHBOARD if is_local else _CLOUD_DASHBOARD
    default_token = _LOCAL_TOKEN_URL if is_local else _CLOUD_TOKEN_URL
    # Self-hosted API (non-loopback base_url): token rides the same host.
    if base_url and not is_local:
        default_token = f"{base_url.rstrip('/')}/oauth/token"

    dashboard = os.environ.get("HONCHO_OAUTH_DASHBOARD", default_dashboard).rstrip("/")
    return OAuthEndpoints(
        authorize_url=os.environ.get("HONCHO_OAUTH_AUTHORIZE_URL", f"{dashboard}/authorize"),
        token_url=os.environ.get("HONCHO_OAUTH_TOKEN_URL", default_token),
        client_id=os.environ.get("HONCHO_OAUTH_CLIENT_ID", "hermes-desktop"),
        scope=os.environ.get("HONCHO_OAUTH_SCOPE", "write"),
    )


@dataclass
class _Pending:
    verifier: str
    redirect_uri: str
    created_at: float


_pending: dict[str, _Pending] = {}
_pending_lock = threading.Lock()


def _pkce() -> tuple[str, str]:
    """Return (verifier, S256 challenge) for an authorization-code request."""
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


def _prune_pending(now: float) -> None:
    expired = [s for s, p in _pending.items() if now - p.created_at > _PENDING_TTL_SECONDS]
    for state in expired:
        _pending.pop(state, None)


def begin_authorization(
    endpoints: OAuthEndpoints,
    redirect_uri: str = LOOPBACK_REDIRECT_URI,
    *,
    source: str | None = None,
    now: float | None = None,
) -> tuple[str, str]:
    """Start an authorization: return ``(authorize_url, state)`` and stash PKCE.

    ``source`` tags the authorize link with the initiating surface
    (``hermes-desktop`` / ``hermes-cli``) so the consent side can attribute
    connects and vary behavior per surface.
    """
    now = time.time() if now is None else now
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(32)
    with _pending_lock:
        _prune_pending(now)
        _pending[state] = _Pending(verifier=verifier, redirect_uri=redirect_uri, created_at=now)
    params = {
        "client_id": endpoints.client_id,
        "redirect_uri": redirect_uri,
        "scope": endpoints.scope,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "response_type": "code",
        "state": state,
    }
    if source:
        params["source"] = source
    return f"{endpoints.authorize_url}?{urlencode(params)}", state


def complete_authorization(
    endpoints: OAuthEndpoints,
    code: str,
    state: str,
    *,
    config_path: Path | None = None,
    host: str | None = None,
    apply_config: bool = True,
    now: float | None = None,
) -> oauth.OAuthCredential:
    """Exchange ``code`` for a grant and persist it. Raises on bad state/exchange.

    ``apply_config=False`` stores the tokens only, skipping the grant's config
    block — the CLI path, where settings stay wizard-owned.
    """
    with _pending_lock:
        pending = _pending.pop(state, None)
    if pending is None:
        raise ValueError("unknown or expired authorization state")

    grant = oauth._http_post_form(
        endpoints.token_url,
        {
            "grant_type": "authorization_code",
            "client_id": endpoints.client_id,
            "code": code,
            "redirect_uri": pending.redirect_uri,
            "code_verifier": pending.verifier,
        },
        oauth._REFRESH_TIMEOUT_SECONDS,
    )

    path = config_path or resolve_config_path()
    target_host = host or resolve_active_host()
    cred = oauth.install_grant(
        path,
        target_host,
        grant,
        client_id=endpoints.client_id,
        token_endpoint=endpoints.token_url,
        apply_config=apply_config,
        now=now,
    )
    # Drop the singleton so the next acquisition builds with the new token.
    from plugins.memory.honcho.client import reset_honcho_client

    reset_honcho_client()
    logger.info("Honcho OAuth grant installed for host %s", target_host)
    return cred


_CALLBACK_HTML = (
    b"<!doctype html><meta charset=utf-8>"
    b"<title>Honcho connected</title>"
    b"<body style='font:14px ui-monospace,monospace;background:#0b0e14;color:#c9d1d9;"
    b"display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
    b"<div>Connected to Honcho. You can close this tab and return to Hermes.</div>"
)


def capture_loopback_code(*, port: int = LOOPBACK_PORT, timeout: float = 300.0) -> tuple[str, str]:
    """Run a one-shot loopback server and return ``(code, state)`` from the callback.

    Serves a single ``/callback`` GET, replies with a close-this-tab page, then
    stops. Raises ``TimeoutError`` if no callback arrives within ``timeout``.
    """
    captured: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib API name
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            params = parse_qs(parsed.query)
            captured["code"] = (params.get("code") or [""])[0]
            captured["state"] = (params.get("state") or [""])[0]
            captured["error"] = (params.get("error") or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_CALLBACK_HTML)

        def log_message(self, *args):  # silence stdlib request logging
            return

    server = HTTPServer((LOOPBACK_HOST, port), _Handler)
    server.timeout = timeout
    try:
        # handle_request honors server.timeout; loop until our callback lands so a
        # stray probe to another path doesn't end the wait empty-handed.
        deadline = time.monotonic() + timeout
        while "code" not in captured and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()

    if captured.get("error"):
        raise ValueError(f"authorization denied: {captured['error']}")
    if "code" not in captured:
        raise TimeoutError("no OAuth callback received before timeout")
    return captured["code"], captured.get("state", "")


def authorize_via_loopback(
    *,
    config_path: Path | None = None,
    host: str | None = None,
    source: str | None = None,
    apply_config: bool = True,
    open_url: Callable[[str], None] | None = None,
    timeout: float = 300.0,
) -> oauth.OAuthCredential:
    """Drive the full loopback flow: open browser → capture code → exchange → persist.

    ``open_url`` defaults to the system browser; tests inject a driver that
    follows the authorize redirect into the loopback callback. It always
    receives the authorize URL, so a CLI caller can also print it for
    browserless environments.
    """
    endpoints = resolve_endpoints()
    authorize_url, state = begin_authorization(endpoints, LOOPBACK_REDIRECT_URI, source=source)

    if open_url is None:
        import webbrowser

        open_url = webbrowser.open

    # Open the browser only after the listener is up, so a fast redirect can't
    # beat the socket; the listener starts inside capture, so kick the browser
    # from a short-lived thread that fires once we begin serving.
    opener = threading.Thread(target=lambda: open_url(authorize_url), daemon=True)
    opener.start()

    code, returned_state = capture_loopback_code(timeout=timeout)
    if returned_state != state:
        raise ValueError("OAuth state mismatch — possible CSRF, aborting")
    return complete_authorization(
        endpoints,
        code,
        returned_state,
        config_path=config_path,
        host=host,
        apply_config=apply_config,
    )


# — Background launcher + status, for the desktop "Connect" button —
# The flow blocks on a browser round-trip, so the web_server endpoint kicks it
# off in a thread and the UI polls status rather than holding the request open.


@dataclass
class FlowStatus:
    state: str = "idle"  # idle | pending | connected | error
    detail: str = ""


_status = FlowStatus()
_status_lock = threading.Lock()
_flow_thread: threading.Thread | None = None


def _detect_connection() -> tuple[bool, str | None]:
    """Report whether a credential is already stored: 'oauth', 'apikey', or none."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig.from_global_config()
        block = (cfg.raw.get("hosts") or {}).get(cfg.host) or {}
        if oauth.OAuthCredential.from_host_block(block) is not None:
            return True, "oauth"
        if cfg.api_key:
            return True, "apikey"
    except Exception:
        pass
    return False, None


def get_flow_status() -> dict[str, object]:
    with _status_lock:
        state, detail = _status.state, _status.detail
    connected, auth = _detect_connection()
    return {"state": state, "detail": detail, "connected": connected, "auth": auth}


def _set_status(state: str, detail: str = "") -> None:
    with _status_lock:
        _status.state, _status.detail = state, detail


def start_loopback_flow_background(
    *,
    config_path: Path | None = None,
    host: str | None = None,
    source: str = "hermes-desktop",
    timeout: float = 300.0,
) -> dict[str, str]:
    """Launch the loopback flow in a daemon thread; returns the initial status.

    Idempotent while a flow is pending — a second call is a no-op so a
    double-clicked button can't open two browser tabs / bind :8765 twice.
    """
    global _flow_thread
    with _status_lock:
        if _status.state == "pending" and _flow_thread and _flow_thread.is_alive():
            return {"state": _status.state, "detail": _status.detail}
        _status.state, _status.detail = "pending", "waiting for browser consent"

    def _run() -> None:
        try:
            authorize_via_loopback(config_path=config_path, host=host, source=source, timeout=timeout)
            _set_status("connected", "Honcho connected")
        except Exception as exc:
            logger.warning("Honcho OAuth loopback flow failed: %s", exc)
            _set_status("error", str(exc))

    _flow_thread = threading.Thread(target=_run, name="honcho-oauth-loopback", daemon=True)
    _flow_thread.start()
    return get_flow_status()
