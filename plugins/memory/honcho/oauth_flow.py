"""Browser sign-in flow for the Honcho memory provider — no CLI step.

``begin_authorization`` / ``complete_authorization`` are the transport-agnostic
core (the code can arrive via the loopback listener here or a ``hermes://``
handler). Endpoints are env-overridable because ``/authorize`` (dashboard) and
``/oauth/token`` (API) live on different origins.
"""

from __future__ import annotations

import base64
import hashlib
import html
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
from plugins.memory.honcho.client import HonchoClientConfig, resolve_active_host, resolve_config_path

logger = logging.getLogger(__name__)

# Loopback redirect registered for the Hermes OAuth client. IP-literal so the browser can't resolve the
# advertised host to ::1 and miss the IPv4 bind.
LOOPBACK_HOST = "127.0.0.1"
LOOPBACK_PORT = 8765
LOOPBACK_REDIRECT_URI = f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}/callback"
# Pending authorizations are keyed by the CSRF ``state`` so a forged callback can't complete a grant;
# stale entries are swept after this TTL.
_PENDING_TTL_SECONDS = 600
# Dashboard serves /authorize, API serves /oauth/token.
_CLOUD_DASHBOARD = "https://app.honcho.dev"
_CLOUD_TOKEN_URL = "https://api.honcho.dev/oauth/token"
_LOCAL_DASHBOARD = "http://localhost:3000"
_LOCAL_TOKEN_URL = "http://localhost:8000/oauth/token"
# One OAuth client for every surface (consent branding varies via ``source``), so there is a single
# grant identity to refresh — no clientId/refresh-token desync.
_DEFAULT_CLIENT_ID = "hermes-agent"

def _display_config_path(path: object) -> str:
    """Home-relative display string for the consent screen (never the write path); outside ``$HOME``, the bare name."""
    p = Path(str(path))
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return p.name

@dataclass(frozen=True)
class OAuthEndpoints:
    """Resolved authorization-server URLs and client identity."""

    authorize_url: str  # dashboard /authorize
    token_url: str  # API /oauth/token
    client_id: str
    scope: str
    device_authorization_url: str = ""  # API /oauth/device_authorization

def resolve_endpoints(environment: str | None = None, base_url: str | None = None) -> OAuthEndpoints:
    """Resolve OAuth endpoints, zero-config by default: the host's honcho ``environment`` picks cloud vs
    localhost, a self-hosted ``base_url`` derives the token endpoint from the API host, env vars override all."""
    if environment is None or base_url is None:
        try:
            cfg = HonchoClientConfig.from_global_config()
            environment = environment or cfg.environment
            base_url = base_url if base_url is not None else cfg.base_url
        except Exception:
            environment = environment or "production"

    is_loopback = bool(base_url) and any(h in base_url for h in ("localhost", "127.0.0.1", "::1"))
    is_local = (environment or "").lower() == "local" or is_loopback
    default_token = _LOCAL_TOKEN_URL if is_local else _CLOUD_TOKEN_URL
    if base_url and not is_local:  # self-hosted API: token rides the same host
        default_token = f"{base_url.rstrip('/')}/oauth/token"
    dashboard = os.environ.get("HONCHO_OAUTH_DASHBOARD", _LOCAL_DASHBOARD if is_local else _CLOUD_DASHBOARD).rstrip("/")
    token_url = os.environ.get("HONCHO_OAUTH_TOKEN_URL", default_token)
    default_device = f"{token_url.rsplit('/', 1)[0]}/device_authorization"  # rides the token endpoint's origin
    return OAuthEndpoints(
        authorize_url=os.environ.get("HONCHO_OAUTH_AUTHORIZE_URL", f"{dashboard}/authorize"),
        token_url=token_url,
        client_id=os.environ.get("HONCHO_OAUTH_CLIENT_ID", _DEFAULT_CLIENT_ID),
        scope=os.environ.get("HONCHO_OAUTH_SCOPE", "write"),
        device_authorization_url=os.environ.get("HONCHO_OAUTH_DEVICE_AUTH_URL", default_device),
    )

_pending: dict[str, tuple[str, str, float]] = {}  # state -> (verifier, redirect_uri, created_at)
_pending_lock = threading.Lock()

def begin_authorization(
    endpoints: OAuthEndpoints, redirect_uri: str = LOOPBACK_REDIRECT_URI, *,
    source: str | None = None, config_path: str | None = None, now: float | None = None,
) -> tuple[str, str]:
    """Start an authorization: return ``(authorize_url, state)`` and stash PKCE. ``source`` tags the initiating
    surface for consent branding; ``config_path`` is the home-relative *display* string (the real write path
    goes to ``complete_authorization``)."""
    now = time.time() if now is None else now
    verifier = secrets.token_urlsafe(64)  # PKCE: S256 challenge of a fresh verifier
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(32)
    with _pending_lock:
        for stale in [s for s, p in _pending.items() if now - p[2] > _PENDING_TTL_SECONDS]:
            _pending.pop(stale, None)
        _pending[state] = (verifier, redirect_uri, now)
    params = {
        "client_id": endpoints.client_id, "redirect_uri": redirect_uri, "scope": endpoints.scope,
        "code_challenge": challenge, "code_challenge_method": "S256", "response_type": "code", "state": state,
    }
    params.update({k: v for k, v in (("source", source), ("config_path", config_path)) if v})
    return f"{endpoints.authorize_url}?{urlencode(params)}", state

def _install(
    endpoints: OAuthEndpoints, grant: dict, *, path: Path | None, host: str | None,
    apply_config: bool, now: float | None, kind: str,
) -> oauth.OAuthCredential:
    """Persist ``grant`` for the target host; drop the cached client so the next acquisition uses the new token."""
    target_host = host or resolve_active_host()
    cred = oauth.install_grant(
        path or resolve_config_path(), target_host, grant,
        client_id=endpoints.client_id, token_endpoint=endpoints.token_url, apply_config=apply_config, now=now,
    )
    from plugins.memory.honcho.client import reset_honcho_client
    reset_honcho_client()
    logger.info("Honcho OAuth %sgrant installed for host %s", kind, target_host)
    return cred

def complete_authorization(
    endpoints: OAuthEndpoints, code: str, state: str, *,
    config_path: Path | None = None, host: str | None = None, apply_config: bool = True, now: float | None = None,
) -> oauth.OAuthCredential:
    """Exchange ``code`` for a grant and persist it. Raises on bad state/exchange. ``apply_config=False``
    stores tokens only (CLI path: settings stay wizard-owned)."""
    with _pending_lock:
        pending = _pending.pop(state, None)
    if pending is None:
        raise ValueError("unknown or expired authorization state")
    verifier, redirect_uri, _ = pending
    form = {"grant_type": "authorization_code", "client_id": endpoints.client_id, "code": code,
            "redirect_uri": redirect_uri, "code_verifier": verifier}
    _, grant = oauth._http_json("POST", endpoints.token_url, timeout=oauth._REFRESH_TIMEOUT_SECONDS, data=form)
    return _install(endpoints, grant, path=config_path, host=host, apply_config=apply_config, now=now, kind="")

_CALLBACK_PAGE = (
    "<!doctype html><meta charset=utf-8><title>{title}</title>"
    "<body style='font:14px ui-monospace,monospace;background:#0b0e14;color:#c9d1d9;"
    "display:flex;align-items:center;justify-content:center;height:100vh;margin:0'><div>{body}</div>"
)
_CALLBACK_HTML = _CALLBACK_PAGE.format(
    title="Honcho connected", body="Connected to Honcho. You can close this tab and return to Hermes."
).encode()
_CALLBACK_ERROR_HTML = _CALLBACK_PAGE.format(  # ``{error}`` is filled per request
    title="Honcho sign-in failed", body="Sign-in was not completed ({error}). You can close this tab and re-run setup."
)

def _bind_loopback_server() -> tuple[HTTPServer, dict[str, str]]:
    """Bind the one-shot callback server, returning it and its capture dict. Prefers :8765, else an
    OS-assigned port (the AS relaxes the port for loopback redirect URIs; the caller advertises the bound port)."""
    captured: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib API name
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            params = parse_qs(parsed.query)
            for k in ("code", "state", "error", "error_description"):
                captured[k] = (params.get(k) or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            error = captured["error"]
            self.wfile.write(_CALLBACK_ERROR_HTML.format(error=html.escape(error)).encode() if error else _CALLBACK_HTML)

        def log_message(self, *args):  # silence stdlib request logging
            return

    try:
        return HTTPServer((LOOPBACK_HOST, LOOPBACK_PORT), _Handler), captured
    except OSError:
        return HTTPServer((LOOPBACK_HOST, 0), _Handler), captured

def capture_loopback_code(server: HTTPServer, captured: dict[str, str], *, timeout: float = 300.0) -> tuple[str, str]:
    """Serve ``/callback`` until our code lands; return ``(code, state)``. Loops so a stray probe to another
    path doesn't end the wait; raises ``TimeoutError`` if nothing arrives within ``timeout``."""
    server.timeout = timeout
    deadline = time.monotonic() + timeout
    try:
        while "code" not in captured and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()

    if error := captured.get("error"):
        detail = captured.get("error_description")
        raise ValueError(f"authorization denied: {error}{f' ({detail})' if detail else ''}")
    if "code" not in captured:
        raise TimeoutError("no OAuth callback received before timeout")
    return captured["code"], captured.get("state", "")

def authorize_via_loopback(
    *, config_path: Path | None = None, host: str | None = None, source: str | None = None,
    apply_config: bool = True, open_url: Callable[[str], None] | None = None, timeout: float = 300.0,
) -> oauth.OAuthCredential:
    """Full loopback flow: open browser → capture code → exchange → persist. ``open_url`` (default: system
    browser) always receives the authorize URL, so a CLI caller can print it for browserless setups."""
    # Bind first so the advertised redirect_uri carries the actual bound port.
    server, captured = _bind_loopback_server()
    redirect_uri = f"http://{LOOPBACK_HOST}:{server.server_address[1]}/callback"
    endpoints = resolve_endpoints()
    path = config_path or resolve_config_path()
    authorize_url, state = begin_authorization(endpoints, redirect_uri, source=source,
                                               config_path=_display_config_path(path))
    if open_url is None:
        import webbrowser
        open_url = webbrowser.open
    # Socket is already bound, so a fast redirect can't beat the browser thread.
    threading.Thread(target=lambda: open_url(authorize_url), daemon=True).start()
    code, returned_state = capture_loopback_code(server, captured, timeout=timeout)
    if returned_state != state:
        raise ValueError("OAuth state mismatch — possible CSRF, aborting")
    return complete_authorization(endpoints, code, returned_state, config_path=path, host=host,
                                  apply_config=apply_config)


# — Device authorization grant (RFC 8628), for headless / remote-VM clients —

DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
# RFC 8628 §3.5: slow_down adds 5s per response; cap matches the server's DEVICE_POLL_INTERVAL_MAX.
_SLOW_DOWN_STEP = 5
_POLL_INTERVAL_CAP = 60
# RFC 8414 metadata; advertising the device grant marks a host as device-login capable.
_AS_METADATA_PATH = "/.well-known/oauth-authorization-server"

class DeviceFlowError(RuntimeError):
    """A device-flow request failed. ``error`` is the RFC error code when known."""

    def __init__(self, error: str, description: str | None = None):
        self.error, self.description = error, description
        super().__init__(f"{error}: {description}" if description else error)

class AccessDenied(DeviceFlowError):
    """The user denied the authorization request."""

class DeviceCodeExpired(DeviceFlowError):
    """The device code expired before the user approved it."""

class AuthorizationTimeout(DeviceFlowError):
    """Polling ran past the device code's lifetime with no decision."""

# Terminal server outcomes for a device-token poll (RFC 8628 §3.5).
_DEVICE_POLL_ERRORS = {"access_denied": AccessDenied, "expired_token": DeviceCodeExpired}

@dataclass(frozen=True)
class DeviceCode:
    """RFC 8628 §3.2 device authorization response."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int

def supports_device_login(endpoints: OAuthEndpoints, *, timeout: float = 5.0) -> bool:
    """Whether the host advertises the device grant in its RFC 8414 metadata. Fails closed on any error."""
    origin = endpoints.token_url.rsplit("/oauth/", 1)[0]
    try:
        body = oauth._http_json("GET", f"{origin}{_AS_METADATA_PATH}", timeout=timeout)[1]
    except Exception:
        return False
    grants = body.get("grant_types_supported") if isinstance(body, dict) else None
    return isinstance(grants, list) and DEVICE_GRANT_TYPE in grants

def request_device_code(endpoints: OAuthEndpoints, *, source: str | None = None) -> DeviceCode:
    """Request a device + user code pair (RFC 8628 §3.1)."""
    if not endpoints.device_authorization_url:
        raise ValueError("no device authorization endpoint resolved")
    data = {"client_id": endpoints.client_id, "scope": endpoints.scope, **({"source": source} if source else {})}
    url = endpoints.device_authorization_url
    status, body = oauth._http_post_form_status(url, data, oauth._REFRESH_TIMEOUT_SECONDS)
    if status != 200:
        raise DeviceFlowError(str(body.get("error") or f"http_{status}"), body.get("error_description"))
    try:
        uri = body["verification_uri"]
        return DeviceCode(
            device_code=body["device_code"], user_code=body["user_code"], verification_uri=uri,
            verification_uri_complete=body.get("verification_uri_complete", f"{uri}?user_code={body['user_code']}"),
            expires_in=int(body["expires_in"]), interval=int(body.get("interval", 5)),  # §3.2: default 5s
        )
    except (KeyError, TypeError, ValueError) as e:
        raise DeviceFlowError("invalid_response", f"malformed device authorization response: {e}") from e

def poll_for_token(
    endpoints: OAuthEndpoints, device: DeviceCode, *, on_poll: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Poll the token endpoint until approved (RFC 8628 §3.4/§3.5). Sleeps ``interval`` before each poll,
    bumping it on ``slow_down``. Raises ``AccessDenied`` / ``DeviceCodeExpired`` on terminal outcomes and
    ``AuthorizationTimeout`` when ``expires_in`` elapses with no decision."""
    import httpx

    form = {"grant_type": DEVICE_GRANT_TYPE, "device_code": device.device_code, "client_id": endpoints.client_id}
    interval = max(1, min(device.interval, _POLL_INTERVAL_CAP))
    deadline = monotonic() + max(1, device.expires_in)
    while True:
        if monotonic() + interval >= deadline:
            raise AuthorizationTimeout("expired_token", "timed out waiting for approval")
        sleep(interval)
        if on_poll:
            on_poll()
        try:
            status, body = oauth._http_post_form_status(endpoints.token_url, form, oauth._REFRESH_TIMEOUT_SECONDS)
        except httpx.TransportError as e:  # a network blip mid-poll shouldn't kill a 10-minute wait
            logger.debug("device token poll transport error, retrying: %s", e)
            continue

        if status == 200:
            if not body.get("access_token"):
                raise DeviceFlowError("invalid_response", "token response missing access_token")
            return body
        error = str(body.get("error") or f"http_{status}")
        if error == "slow_down":
            interval = min(interval + _SLOW_DOWN_STEP, _POLL_INTERVAL_CAP)
        elif error != "authorization_pending":
            raise _DEVICE_POLL_ERRORS.get(error, DeviceFlowError)(error, body.get("error_description"))

def authorize_via_device_code(
    *, config_path: Path | None = None, host: str | None = None, source: str | None = None,
    apply_config: bool = True, display: Callable[[DeviceCode], None] | None = None,
    open_url: Callable[[str], None] | None = None, on_poll: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> oauth.OAuthCredential:
    """Full device flow: request codes → show user code → poll → persist. ``open_url`` (if given) receives
    ``verification_uri_complete``; no default browser open, since the approving browser may be on another machine."""
    endpoints = resolve_endpoints()
    path = config_path or resolve_config_path()  # resolve NOW so a later ambient lookup can't drift
    target_host = host or resolve_active_host()

    device = request_device_code(endpoints, source=source)
    if display:
        display(device)
    if open_url:
        open_url(device.verification_uri_complete)
    grant = poll_for_token(endpoints, device, on_poll=on_poll, sleep=sleep)
    return _install(endpoints, grant, path=path, host=target_host, apply_config=apply_config, now=None,
                    kind="device ")


# — Background launcher + status, for the desktop "Connect" button — the flow
# blocks on a browser round-trip, so web_server runs it in a thread and the UI polls.

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
        cfg = HonchoClientConfig.from_global_config()
        block = (cfg.raw.get("hosts") or {}).get(cfg.host) or {}
        auth = "oauth" if oauth.OAuthCredential.from_host_block(block) is not None else "apikey" if cfg.api_key else None
    except Exception:
        auth = None
    return auth is not None, auth

def get_flow_status() -> dict[str, object]:
    with _status_lock:
        state, detail = _status.state, _status.detail
    connected, auth = _detect_connection()
    return {"state": state, "detail": detail, "connected": connected, "auth": auth}

def _set_status(state: str, detail: str = "") -> None:
    with _status_lock:
        _status.state, _status.detail = state, detail

def start_loopback_flow_background(
    *, config_path: Path | None = None, host: str | None = None, source: str = "hermes-desktop",
    timeout: float = 300.0,
) -> dict[str, str]:
    """Launch the loopback flow in a daemon thread; returns the initial status.
    Idempotent while pending, so a double-click can't open two tabs / bind :8765 twice."""
    global _flow_thread
    # Resolve under the caller's profile scope NOW — a context-local HERMES_HOME override can't reach the worker.
    config_path = config_path or resolve_config_path()
    host = host or resolve_active_host()
    with _status_lock:
        if _status.state == "pending" and _flow_thread and _flow_thread.is_alive():
            return {"state": _status.state, "detail": _status.detail}
        _status.state, _status.detail = "pending", "waiting for browser consent"

    def _run() -> None:
        try:
            authorize_via_loopback(
                config_path=config_path, host=host, source=source, apply_config=False, timeout=timeout
            )
            _set_status("connected", "Honcho connected")
        except Exception as exc:
            logger.warning("Honcho OAuth loopback flow failed: %s", exc)
            _set_status("error", str(exc))

    _flow_thread = threading.Thread(target=_run, name="honcho-oauth-loopback", daemon=True)
    _flow_thread.start()
    return get_flow_status()
