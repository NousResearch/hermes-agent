"""Solstice OAuth flow — PKCE loopback login, NAS-brokered exchange.

Solstice authenticates with the user's Solstice account. The flow:

1. Hermes generates a PKCE pair + state nonce and opens the authorize
   URL in the browser, with ``redirect_uri`` pointing at a local loopback.
2. The provider redirects the code back to the loopback listener.
3. Hermes POSTs ``{code, code_verifier, redirect_uri}`` to a **NAS broker
   endpoint** which holds the ``client_secret`` (never shipped to
   Hermes) and performs the token exchange.
4. Tokens are stored in ``auth.json`` ``providers.solstice`` (same custody
   model as xai-oauth / qwen-oauth), and refreshed through the NAS broker
   whenever the access token nears expiry.

Inference traffic goes DIRECT from Hermes to
``generativelanguage.googleapis.com`` (the per-user-quota endpoint) with
the access token — NAS is out of the prompt path entirely.

Secret-launch mechanics: the provider is `hidden=True` (out of pickers and auth
lists) but enabled by default once a logged-in Nous user has a reachable NAS
broker; ``hermes auth add solstice`` surfaces it.  The module defers to the
single auth-side gate ``solstice_enabled`` (see that function).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-sourced — NO defaults baked into the tree)
# ---------------------------------------------------------------------------

# Base URL of the NAS broker that holds the client_secret and performs the
# code exchange / refresh, and owns the client config (discovered at
# login).  Defaults to the public portal origin.
SOLSTICE_NAS_BASE_URL_ENV = "SOLSTICE_NAS_BASE_URL"
DEFAULT_SOLSTICE_NAS_BASE_URL = "https://portal.nousresearch.com"

SOLSTICE_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 300  # refresh 5 min before expiry

# NAS broker endpoints.
SOLSTICE_NAS_EXCHANGE_PATH = "/api/oauth/gemini-auth/exchange"
SOLSTICE_NAS_REFRESH_PATH = "/api/oauth/gemini-auth/refresh"
SOLSTICE_NAS_CONFIG_PATH = "/api/oauth/gemini-auth/config"

# Inference base URL — per-user-quota host.  SINGLE SOURCE OF TRUTH for
# the inference host: the provider profile (plugins/model-providers/solstice)
# and the transport (agent/solstice_adapter.py) import this constant, so a
# launch-time host correction is a one-line change in exactly one place.
SOLSTICE_INFERENCE_BASE_URL = "https://generativelanguage.googleapis.com/v1alpha"


class _AuthError(Exception):
    """Local stand-in for ``hermes_cli.auth.AuthError``.

    Deliberately NOT imported from ``hermes_cli.auth`` at module top: this
    module must stay light-importable (std + httpx) because the solstice
    provider plugin imports it during provider discovery, and pulling in
    ``hermes_cli.auth`` there would drag the whole auth subsystem into every
    ``list_providers()`` call.  The exception is converted to the real
    ``AuthError`` at the module boundary (``resolve_solstice_runtime_credentials``),
    so callers outside this module always see the canonical type.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "solstice",
        code: str = "solstice_error",
        relogin_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.relogin_required = relogin_required


def solstice_enabled() -> bool:
    """True when the Solstice provider is surfaced/usable.

    With NAS owning the client config (discovered at login), there is
    no local enable-gate env var — the provider is enabled by default and
    usable whenever a logged-in Nous user has a reachable NAS broker. The
    provider stays `hidden=True` (out of the default picker) until configured,
    matching the dark-launch intent; once a user runs `hermes auth add
    solstice` it becomes the active provider.
    """
    return True


def solstice_nas_base_url() -> str:
    """Return the NAS broker base URL (env override or portal default)."""
    return (
        os.getenv(SOLSTICE_NAS_BASE_URL_ENV, "") or ""
    ).strip() or DEFAULT_SOLSTICE_NAS_BASE_URL


def _validate_broker_url(url: str) -> str:
    """Refuse a non-HTTPS broker URL (MITM → refresh-token leak guard)."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise _AuthError(
            f"Solstice NAS broker URL must be HTTPS: {url!r}",
            code="solstice_broker_invalid",
        )
    if not parsed.hostname:
        raise _AuthError(
            f"Solstice NAS broker URL is missing a hostname: {url!r}",
            code="solstice_broker_invalid",
        )
    return url.rstrip("/")


# ---------------------------------------------------------------------------
# Token store (auth.json providers.solstice)
# ---------------------------------------------------------------------------


def _solstice_state_from_store(
    auth_store: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    from hermes_cli.auth import _load_provider_state

    state = _load_provider_state(auth_store, "solstice")
    if isinstance(state, dict):
        return state
    credential_pool = auth_store.get("credential_pool")
    entries = (
        credential_pool.get("solstice")
        if isinstance(credential_pool, dict)
        else None
    )
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            access_token = str(entry.get("access_token", "") or "").strip()
            refresh_token = str(entry.get("refresh_token", "") or "").strip()
            if not access_token or not refresh_token:
                continue
            merged = dict(state or {})
            merged["tokens"] = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": str(entry.get("token_type") or "Bearer"),
            }
            if entry.get("last_refresh"):
                merged["last_refresh"] = entry.get("last_refresh")
            merged.setdefault("auth_mode", "oauth_pkce")
            return merged
    return state if isinstance(state, dict) else None


def _solstice_state_has_usable_tokens(state: Optional[Dict[str, Any]]) -> bool:
    tokens = state.get("tokens") if isinstance(state, dict) else None
    return (
        isinstance(tokens, dict)
        and bool(str(tokens.get("access_token", "") or "").strip())
        and bool(str(tokens.get("refresh_token", "") or "").strip())
    )


def _read_solstice_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _load_global_auth_store,
    )

    if _lock:
        with _auth_store_lock():
            auth_store = _load_auth_store()
    else:
        auth_store = _load_auth_store()
    state = _solstice_state_from_store(auth_store)
    if not _solstice_state_has_usable_tokens(state):
        global_state = _solstice_state_from_store(_load_global_auth_store())
        if _solstice_state_has_usable_tokens(global_state):
            state = global_state
    if not state:
        raise _AuthError(
            "No Solstice credentials stored. Sign in with `hermes auth add solstice`.",
            code="solstice_missing",
            relogin_required=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise _AuthError(
            "Solstice auth state is missing tokens. Re-authenticate.",
            code="solstice_invalid_shape",
            relogin_required=True,
        )
    return {
        "tokens": tokens,
        "last_refresh": state.get("last_refresh"),
        "redirect_uri": state.get("redirect_uri"),
        "auth_mode": state.get("auth_mode", "oauth_pkce"),
    }


def _save_solstice_tokens(
    tokens: Dict[str, Any],
    *,
    redirect_uri: str = "",
    last_refresh: Optional[str] = None,
    set_active: bool = True,
) -> None:
    """Persist Solstice tokens into auth.json (with global-root write-through)."""
    from hermes_cli.auth import (
        _auth_store_lock,
        _global_auth_file_path,
        _load_auth_store,
        _load_provider_state_with_source,
        _persist_provider_state_to_store,
        _same_path,
        _save_auth_store,
        _store_provider_state,
    )

    if last_refresh is None:
        last_refresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state, source_path = _load_provider_state_with_source(auth_store, "solstice")
        if state is None:
            state = {}
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = "oauth_pkce"
        if redirect_uri:
            state["redirect_uri"] = redirect_uri
        global_root = _global_auth_file_path()
        is_from_root = bool(
            source_path is not None
            and global_root is not None
            and _same_path(source_path, global_root)
        )
        if is_from_root and global_root is not None:
            _persist_provider_state_to_store(
                "solstice", state, global_root, set_active=False
            )
        else:
            _store_provider_state(
                auth_store, "solstice", state, set_active=set_active
            )
            _save_auth_store(auth_store)


def _solstice_access_token_is_expiring(
    access_token: str, skew_seconds: int = 0
) -> bool:
    """True when a JWT access token's ``exp`` is within *skew_seconds*."""
    if not isinstance(access_token, str) or "." not in access_token:
        return False
    try:
        import base64
        import json

        parts = access_token.split(".")
        if len(parts) < 2:
            return False
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")
        )
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return float(exp) <= (time.time() + max(0, int(skew_seconds)))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NAS broker calls
# ---------------------------------------------------------------------------


def _nous_bearer_header() -> Dict[str, str]:
    """Resolve the caller's Nous access token and build a Bearer auth header.

    NAS gates the Solstice broker behind a valid Nous token from a logged-in
    user (no specific scope). Lazily imported so this module stays
    light-importable during provider discovery.
    """
    from hermes_cli.auth import resolve_nous_access_token

    token = resolve_nous_access_token()
    return {"Authorization": f"Bearer {token}"}


def _exchange_code(
    code: str, code_verifier: str, redirect_uri: str, *, timeout: float = 30.0
) -> Dict[str, Any]:
    """POST the authorize code to the NAS broker for token exchange."""
    broker = _validate_broker_url(solstice_nas_base_url())
    url = f"{broker}{SOLSTICE_NAS_EXCHANGE_PATH}"
    headers = {"Accept": "application/json", **_nous_bearer_header()}
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        response = client.post(
            url,
            json={
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": redirect_uri,
            },
            headers=headers,
        )
    if response.status_code != 200:
        raise _AuthError(
            f"Solstice OAuth exchange failed (HTTP {response.status_code})."
            + (f" {response.text.strip()}" if response.text else ""),
            code="solstice_exchange_failed",
        )
    payload = response.json()
    for key in ("access_token", "refresh_token"):
        if not str(payload.get(key, "") or "").strip():
            raise _AuthError(
                f"Solstice OAuth exchange response missing {key}.",
                code="solstice_exchange_invalid",
            )
    return payload


def _refresh_tokens(refresh_token: str, *, timeout: float = 30.0) -> Dict[str, Any]:
    """POST the refresh token to the NAS broker for rotation."""
    broker = _validate_broker_url(solstice_nas_base_url())
    url = f"{broker}{SOLSTICE_NAS_REFRESH_PATH}"
    headers = {"Accept": "application/json", **_nous_bearer_header()}
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        response = client.post(
            url,
            json={"refresh_token": refresh_token},
            headers=headers,
        )
    if response.status_code != 200:
        body = response.text or response.reason_phrase
        relogin = "invalid_grant" in body.lower() or "refresh_token" in body.lower()
        raise _AuthError(
            f"Solstice token refresh failed: {body}",
            code="solstice_refresh_failed",
            relogin_required=relogin,
        )
    payload = response.json()
    if not str(payload.get("access_token", "") or "").strip():
        raise _AuthError(
            "Solstice refresh response missing access_token.",
            code="solstice_refresh_invalid",
            relogin_required=True,
        )
    return payload


def _fetch_solstice_config(*, timeout: float = 30.0) -> Dict[str, Any]:
    """GET the client config from NAS (client_id, scope, authorize_url).

    NAS is the single source of truth for the client config; the local agent
    builds its authorize URL from this (adding only its own PKCE challenge,
    state, and loopback redirect_uri).
    """
    broker = _validate_broker_url(solstice_nas_base_url())
    url = f"{broker}{SOLSTICE_NAS_CONFIG_PATH}"
    headers = {"Accept": "application/json", **_nous_bearer_header()}
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        response = client.get(url, headers=headers)
    if response.status_code != 200:
        raise _AuthError(
            f"Solstice config fetch failed (HTTP {response.status_code})."
            + (f" {response.text.strip()}" if response.text else ""),
            code="solstice_config_failed",
        )
    payload = response.json()
    for key in ("client_id", "authorize_url", "scope"):
        if not str(payload.get(key, "") or "").strip():
            raise _AuthError(
                f"Solstice config response missing {key}.",
                code="solstice_config_invalid",
            )
    return payload


# ---------------------------------------------------------------------------
# Login flow (PKCE loopback)
# ---------------------------------------------------------------------------


def _solstice_pkce_pair() -> tuple[str, str, str]:
    """Return (code_verifier, code_challenge_S256, state) for the Solstice login.

    Self-contained rather than shared: PKCE generation is per-provider on this tree (see
    ``auth_minimax._minimax_pkce_pair`` / ``auth_spotify._spotify_code_verifier``), and this module
    must stay light-importable during provider discovery.
    """
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge, secrets.token_urlsafe(24)


def _make_solstice_callback_handler(
    expected_path: str,
) -> tuple[type[BaseHTTPRequestHandler], Dict[str, Any]]:
    result: Dict[str, Any] = {
        "code": None,
        "state": None,
        "error": None,
        "error_description": None,
    }

    class _SolsticeCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return
            params = parse_qs(parsed.query)
            result["code"] = params.get("code", [None])[0]
            result["state"] = params.get("state", [None])[0]
            result["error"] = params.get("error", [None])[0]
            result["error_description"] = params.get("error_description", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if result["error"]:
                body = "<html><body><h1>Solstice authorization failed.</h1>You can close this tab.</body></html>"
            else:
                body = "<html><body><h1>Solstice authorization received.</h1>You can close this tab.</body></html>"
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _SolsticeCallbackHandler, result


def _wait_for_loopback_callback(
    port: int, path: str, *, timeout_seconds: float = 180.0
) -> Dict[str, Any]:
    """Run the loopback listener until the code arrives (or timeout/error)."""
    import threading

    handler_cls, result = _make_solstice_callback_handler(path)

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        server = _ReuseHTTPServer(("127.0.0.1", port), handler_cls)
    except OSError as exc:
        raise _AuthError(
            f"Could not bind Solstice callback server on 127.0.0.1:{port}: {exc}",
            code="solstice_callback_bind_failed",
        ) from exc

    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if result["code"] or result["error"]:
                return result
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise _AuthError(
        "Solstice authorization timed out waiting for the local callback.",
        code="solstice_callback_timeout",
    )


def _pick_loopback_port() -> int:
    """Ask the OS for a free loopback port, then release it for the listener."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _login_solstice(
    args: Any,
    pconfig: Any,
    *,
    force_new_login: bool = False,
) -> None:
    """Run the Solstice PKCE loopback login and persist tokens."""
    del pconfig  # parity with other provider login helpers

    if not solstice_enabled():
        raise _AuthError(
            "Solstice is not configured. Set SOLSTICE_CLIENT_ID in "
            "~/.hermes/.env and try again.",
            code="solstice_not_configured",
        )

    if not force_new_login:
        try:
            existing = resolve_solstice_runtime_credentials(
                refresh_if_expiring=False
            )
            api_key = existing.get("api_key", "")
            if api_key and not _solstice_access_token_is_expiring(api_key, 60):
                print("Existing Solstice credentials found in Hermes auth store.")
                try:
                    reuse = input("Use existing credentials? [Y/n]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    reuse = "y"
                if reuse in {"", "y", "yes"}:
                    from hermes_cli.auth import _update_config_for_provider

                    _update_config_for_provider(
                        "solstice",
                        existing.get("base_url", SOLSTICE_INFERENCE_BASE_URL),
                    )
                    print()
                    print("Login successful!")
                    return
        except Exception:
            pass  # no existing creds → fall through to a fresh login

    cfg = _fetch_solstice_config()
    verifier, challenge, state = _solstice_pkce_pair()
    port = _pick_loopback_port()
    redirect_uri = f"http://127.0.0.1:{port}/gemini-auth/callback"

    print()
    print("Signing in to Solstice...")
    print("(Hermes creates its own local OAuth session — tokens stay on this machine)")
    print()

    authorize_params = {
        "client_id": cfg["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": cfg["scope"],
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    authorize_url = f"{cfg['authorize_url']}?{urlencode(authorize_params)}"

    open_browser = not bool(getattr(args, "no_browser", False))
    can_open_browser = True
    try:
        from hermes_cli.auth import _can_open_graphical_browser, _is_remote_session

        if _is_remote_session():
            open_browser = False
        if not _can_open_graphical_browser():
            can_open_browser = False
    except Exception:
        pass

    print(f"  1. Open: {authorize_url}")
    if open_browser and can_open_browser:
        import webbrowser

        if webbrowser.open(authorize_url):
            print("  (Opened browser for sign-in)")
        else:
            print("  Could not open a browser automatically — open the URL above.")
    else:
        print("  (Open the URL above and authorize.)")
    print("Waiting for authorization...")

    callback = _wait_for_loopback_callback(port, "/gemini-auth/callback")
    if callback.get("error"):
        raise _AuthError(
            f"Solstice authorization failed: {callback.get('error_description') or callback.get('error')}",
            code="solstice_denied",
        )
    if callback.get("state") != state:
        raise _AuthError(
            "Solstice callback state mismatch — the response may have been tampered with.",
            code="solstice_state_mismatch",
        )
    code = callback.get("code")
    if not code:
        raise _AuthError(
            "Solstice callback did not include an authorization code.",
            code="solstice_callback_no_code",
        )

    token_payload = _exchange_code(code, verifier, redirect_uri)
    expires_in = _coerce_ttl_seconds(token_payload.get("expires_in", 0))
    now = datetime.now(timezone.utc)
    tokens = {
        "access_token": str(token_payload["access_token"]).strip(),
        "refresh_token": str(token_payload["refresh_token"]).strip(),
        "token_type": str(
            token_payload.get("token_type", "Bearer") or "Bearer"
        ).strip(),
        "scope": str(token_payload.get("scope") or "").strip(),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(
            now.timestamp() + expires_in, tz=timezone.utc
        ).isoformat(),
        "expires_in": expires_in,
    }
    _save_solstice_tokens(tokens, redirect_uri=redirect_uri, set_active=True)

    from hermes_cli.auth import (
        _update_config_for_provider,
        unsuppress_credential_source,
    )

    unsuppress_credential_source("solstice", "oauth_pkce")
    _update_config_for_provider("solstice", SOLSTICE_INFERENCE_BASE_URL)
    print()
    print("✓ Solstice login successful.")
    print(f"  Tokens: {_auth_file_hint()}")


def _coerce_ttl_seconds(value: Any) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _auth_file_hint() -> str:
    try:
        from hermes_constants import display_hermes_home

        return f"{display_hermes_home()}/auth.json"
    except Exception:
        return "auth.json"


# ---------------------------------------------------------------------------
# Runtime resolution + status
# ---------------------------------------------------------------------------


def resolve_solstice_runtime_credentials(
    *,
    refresh_if_expiring: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Resolve usable Solstice credentials, refreshing via NAS when needed.

    Mirrors resolve_xai_oauth_runtime_credentials: read store → refresh if
    expiring → return dict with api_key/base_url.  The ``api_key`` is the
    OAuth access token (Bearer).
    """
    from hermes_cli.auth import AuthError

    try:
        stored = _read_solstice_tokens()
    except _AuthError as exc:
        raise AuthError(
            str(exc),
            provider="solstice",
            code=getattr(exc, "code", "solstice_missing"),
            relogin_required=getattr(exc, "relogin_required", True),
        ) from exc

    tokens = stored["tokens"]
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()

    should_refresh = force_refresh or (
        refresh_if_expiring
        and _solstice_access_token_is_expiring(
            access_token, SOLSTICE_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
        )
    )
    if should_refresh:
        if not refresh_token:
            raise AuthError(
                "Solstice access token is expired and no refresh token is stored. Re-login.",
                provider="solstice",
                code="solstice_no_refresh_token",
                relogin_required=True,
            )
        try:
            rotated = _refresh_tokens(refresh_token)
            new_tokens = dict(tokens)
            new_tokens["access_token"] = str(rotated["access_token"]).strip()
            if str(rotated.get("refresh_token", "") or "").strip():
                new_tokens["refresh_token"] = str(rotated["refresh_token"]).strip()
            expires_in = _coerce_ttl_seconds(rotated.get("expires_in", 0))
            now = datetime.now(timezone.utc)
            new_tokens["obtained_at"] = now.isoformat()
            new_tokens["expires_at"] = datetime.fromtimestamp(
                now.timestamp() + expires_in, tz=timezone.utc
            ).isoformat()
            new_tokens["expires_in"] = expires_in
            _save_solstice_tokens(new_tokens, set_active=False)
            tokens = new_tokens
        except _AuthError as exc:
            # Terminal refresh failure (invalid_grant / revoked) — clear the
            # dead tokens from auth.json so subsequent resolutions fail fast
            # without a network retry (mirrors the xai-oauth quarantine).
            try:
                from hermes_cli.auth import (
                    _auth_store_lock,
                    _load_auth_store,
                    _load_provider_state,
                    _save_auth_store,
                    _store_provider_state,
                )

                with _auth_store_lock():
                    _q_store = _load_auth_store()
                    _q_state = _load_provider_state(_q_store, "solstice") or {}
                    _q_tokens = dict(_q_state.get("tokens") or {})
                    _q_tokens.pop("access_token", None)
                    _q_tokens.pop("refresh_token", None)
                    _q_state["tokens"] = _q_tokens
                    _q_state["last_auth_error"] = {
                        "provider": "solstice",
                        "code": getattr(exc, "code", "solstice_refresh_failed"),
                        "message": str(exc),
                        "reason": "runtime_refresh_failure",
                        "relogin_required": True,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                    _store_provider_state(
                        _q_store, "solstice", _q_state, set_active=False
                    )
                    _save_auth_store(_q_store)
            except Exception as _save_exc:  # pragma: no cover - best effort
                logger.debug(
                    "Solstice OAuth: failed to persist quarantined state: %s",
                    _save_exc,
                )
            raise AuthError(
                str(exc),
                provider="solstice",
                code=getattr(exc, "code", "solstice_refresh_failed"),
                relogin_required=getattr(exc, "relogin_required", True),
            ) from exc

    return {
        "provider": "solstice",
        "api_mode": "chat_completions",
        "base_url": SOLSTICE_INFERENCE_BASE_URL,
        "api_key": str(tokens.get("access_token", "") or "").strip(),
        "source": "hermes-auth-store",
        "last_refresh": stored.get("last_refresh"),
        "expires_at_ms": _token_expiry_ms(tokens.get("expires_at")),
    }


def _token_expiry_ms(expires_at: Any) -> Optional[int]:
    try:
        dt = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def get_solstice_status() -> Dict[str, Any]:
    """Return auth status dict (logged_in, account hints, error)."""
    try:
        creds = resolve_solstice_runtime_credentials(refresh_if_expiring=False)
        access = creds.get("api_key", "")
        expired = _solstice_access_token_is_expiring(access, 0)
        return {
            "provider": "solstice",
            "logged_in": bool(access),
            "configured": True,
            "access_token_expired": expired,
            "expires_at_ms": creds.get("expires_at_ms"),
        }
    except Exception as exc:
        return {
            "provider": "solstice",
            "logged_in": False,
            "configured": True,
            "error": str(exc),
        }
