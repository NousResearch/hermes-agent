"""Spotify OAuth (loopback PKCE) login, refresh and runtime credentials.

Re-exported from ``hermes_cli/auth.py`` (patch targets unchanged); origin helpers are imported
lazily per function so ``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations

import logging
import base64
import hashlib
import os
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse
from hermes_cli.auth_constants import (
    AuthError, DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL, DEFAULT_SPOTIFY_API_BASE_URL, DEFAULT_SPOTIFY_REDIRECT_URI,
    DEFAULT_SPOTIFY_SCOPE, SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, SPOTIFY_DASHBOARD_URL, SPOTIFY_DOCS_URL,
    _spotify_err, httpx,
)

logger = logging.getLogger("hermes_cli.auth")

_CALLBACK_HTML = "<html><body><h1>Spotify authorization {}.</h1>You can close this tab.</body></html>"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _spotify_scope_string(raw_scope: Optional[str] = None) -> str:
    """Requested scope, whitespace-normalized and de-duplicated (order kept)."""
    return " ".join(dict.fromkeys((raw_scope or DEFAULT_SPOTIFY_SCOPE).split()))


def _spotify_setting(
    state: Optional[Dict[str, Any]], state_key: str, env_vars: Tuple[str, ...], default: str, *,
    explicit: Optional[str] = None, strip_slash: bool = False,
) -> str:
    """First non-empty of explicit arg, env vars (``.env`` aware), stored state, then *default*."""
    from hermes_cli.config import get_env_value
    candidates = (
        explicit, *(get_env_value(var) for var in env_vars),
        state.get(state_key) if isinstance(state, dict) else None, default,
    )
    for candidate in candidates:
        cleaned = _clean(candidate)
        if strip_slash:
            cleaned = cleaned.rstrip("/")
        if cleaned:
            return cleaned
    return default


def _spotify_client_id(explicit: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> str:
    client_id = _spotify_setting(
        state, "client_id", ("HERMES_SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_ID"), "", explicit=explicit,
    )
    if client_id:
        return client_id
    raise _spotify_err(
        "Spotify client_id is required. Set HERMES_SPOTIFY_CLIENT_ID or pass --client-id.",
        "spotify_client_id_missing",
    )


def _spotify_redirect_uri(explicit: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> str:
    return _spotify_setting(
        state, "redirect_uri", ("HERMES_SPOTIFY_REDIRECT_URI", "SPOTIFY_REDIRECT_URI"),
        DEFAULT_SPOTIFY_REDIRECT_URI, explicit=explicit,
    )


def _spotify_api_base_url(state: Optional[Dict[str, Any]] = None) -> str:
    return _spotify_setting(
        state, "api_base_url", ("HERMES_SPOTIFY_API_BASE_URL",), DEFAULT_SPOTIFY_API_BASE_URL, strip_slash=True,
    )


def _spotify_accounts_base_url(state: Optional[Dict[str, Any]] = None) -> str:
    return _spotify_setting(
        state, "accounts_base_url", ("HERMES_SPOTIFY_ACCOUNTS_BASE_URL",), DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
        strip_slash=True,
    )


def _spotify_code_verifier(length: int = 64) -> str:
    return base64.urlsafe_b64encode(os.urandom(length)).decode("ascii").rstrip("=")[:128]


def _spotify_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _spotify_build_authorize_url(
    *, client_id: str, redirect_uri: str, scope: str, state: str, code_challenge: str,
    accounts_base_url: str,
) -> str:
    query = urlencode({
        "client_id": client_id, "response_type": "code", "redirect_uri": redirect_uri,
        "scope": scope, "state": state, "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    })
    return f"{accounts_base_url}/authorize?{query}"


def _spotify_validate_redirect_uri(redirect_uri: str) -> tuple[str, int, str]:
    parsed = urlparse(redirect_uri)
    host = parsed.hostname or ""
    problem = (
        "must use http://localhost or http://127.0.0.1." if parsed.scheme != "http"
        else "must point to localhost or 127.0.0.1." if host not in {"127.0.0.1", "localhost"}
        else "must include an explicit localhost port." if not parsed.port
        else None
    )
    if problem:
        raise _spotify_err(f"Spotify PKCE redirect_uri {problem}", "spotify_redirect_invalid")
    return host, parsed.port, parsed.path or "/"


def _make_spotify_callback_handler(expected_path: str) -> tuple[type[BaseHTTPRequestHandler], dict[str, Any]]:
    result: dict[str, Any] = {"code": None, "state": None, "error": None, "error_description": None}

    class _SpotifyCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return

            params = parse_qs(parsed.query)
            for key in result:
                result[key] = params.get(key, [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_CALLBACK_HTML.format("failed" if result["error"] else "received").encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _SpotifyCallbackHandler, result


def _spotify_wait_for_callback(redirect_uri: str, *, timeout_seconds: float = 180.0) -> dict[str, Any]:
    host, port, path = _spotify_validate_redirect_uri(redirect_uri)
    handler_cls, result = _make_spotify_callback_handler(path)

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        server = _ReuseHTTPServer((host, port), handler_cls)
    except OSError as exc:
        raise _spotify_err(
            f"Could not bind Spotify callback server on {host}:{port}: {exc}", "spotify_callback_bind_failed",
        ) from exc

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
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
    raise _spotify_err("Spotify authorization timed out waiting for the local callback.", "spotify_callback_timeout")


def _spotify_token_payload_to_state(
    token_payload: Dict[str, Any], *, client_id: str, redirect_uri: str, requested_scope: str,
    accounts_base_url: str, api_base_url: str, previous_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from hermes_cli.auth import _coerce_ttl_seconds
    now = datetime.now(timezone.utc)
    expires_in = _coerce_ttl_seconds(token_payload.get("expires_in", 0))
    expires_at = datetime.fromtimestamp(now.timestamp() + expires_in, tz=timezone.utc)
    state = dict(previous_state or {})
    state.update({
        "client_id": client_id, "redirect_uri": redirect_uri,
        "accounts_base_url": accounts_base_url, "api_base_url": api_base_url,
        "scope": requested_scope,
        "granted_scope": str(token_payload.get("scope") or requested_scope).strip(),
        "token_type": _clean(token_payload.get("token_type", "Bearer") or "Bearer") or "Bearer",
        "access_token": _clean(token_payload.get("access_token")),
        "refresh_token": _clean(token_payload.get("refresh_token") or state.get("refresh_token")),
        "obtained_at": now.isoformat(), "expires_at": expires_at.isoformat(),
        "expires_in": expires_in, "auth_type": "oauth_pkce",
    })
    return state


def _spotify_token_post(
    accounts_base_url: str, data: Dict[str, str], *, timeout_seconds: float, what: str,
    failed_code: str, invalid_code: str, invalid_message: str, failed_suffix: str = "",
    relogin_required: bool = False,
) -> Dict[str, Any]:
    """POST to Spotify's ``/api/token`` and return the JSON payload, or raise a shaped AuthError."""
    try:
        response = httpx.post(
            f"{accounts_base_url}/api/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=data,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise _spotify_err(f"Spotify {what} failed: {exc}", failed_code) from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise _spotify_err(
            f"Spotify {what} failed.{failed_suffix}" + (f" Response: {detail}" if detail else ""),
            failed_code, relogin=relogin_required,
        )
    payload = response.json()
    if not isinstance(payload, dict) or not _clean(payload.get("access_token")):
        raise _spotify_err(invalid_message, invalid_code, relogin=relogin_required)
    return payload


def _refresh_spotify_oauth_state(state: Dict[str, Any], *, timeout_seconds: float = 20.0) -> Dict[str, Any]:
    refresh_token = _clean(state.get("refresh_token"))
    if not refresh_token:
        raise _spotify_err(
            "Spotify refresh token missing. Run `hermes auth spotify` again.",
            "spotify_refresh_token_missing", relogin=True,
        )

    client_id = _spotify_client_id(state=state)
    accounts_base_url = _spotify_accounts_base_url(state)
    payload = _spotify_token_post(
        accounts_base_url,
        {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": client_id},
        timeout_seconds=timeout_seconds, what="token refresh", failed_code="spotify_refresh_failed",
        invalid_code="spotify_refresh_invalid",
        invalid_message="Spotify refresh response did not include an access_token.",
        failed_suffix=" Run `hermes auth spotify` again.", relogin_required=True,
    )

    return _spotify_token_payload_to_state(
        payload, client_id=client_id, redirect_uri=_spotify_redirect_uri(state=state),
        requested_scope=str(state.get("scope") or DEFAULT_SPOTIFY_SCOPE),
        accounts_base_url=accounts_base_url, api_base_url=_spotify_api_base_url(state),
        previous_state=state,
    )


def resolve_spotify_runtime_credentials(
    *, force_refresh: bool = False, refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    from hermes_cli.auth import _auth_store_lock, _is_expiring, _load_auth_store, _load_provider_state, _quarantine_flat_oauth_state, _refresh_spotify_oauth_state, _save_auth_store, _store_provider_state
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "spotify")
        if not state:
            raise _spotify_err(
                "Spotify is not authenticated. Run `hermes auth spotify` first.", "spotify_auth_missing", relogin=True,
            )

        should_refresh = bool(force_refresh)
        if not should_refresh and refresh_if_expiring:
            should_refresh = _is_expiring(state.get("expires_at"), refresh_skew_seconds)
        if should_refresh:
            try:
                state = _refresh_spotify_oauth_state(state)
                _store_provider_state(auth_store, "spotify", state, set_active=False)
                _save_auth_store(auth_store)
            except AuthError as exc:
                if exc.relogin_required and state.get("refresh_token"):
                    _quarantine_flat_oauth_state(state, "spotify", exc)
                    try:
                        _store_provider_state(auth_store, "spotify", state, set_active=False)
                        _save_auth_store(auth_store)
                    except Exception as _save_exc:
                        logger.debug("Spotify OAuth: failed to persist quarantined state: %s", _save_exc)
                raise

    access_token = _clean(state.get("access_token"))
    if not access_token:
        raise _spotify_err(
            "Spotify access token missing. Run `hermes auth spotify` again.",
            "spotify_access_token_missing", relogin=True,
        )

    return {
        "provider": "spotify", "access_token": access_token, "api_key": access_token,
        "token_type": str(state.get("token_type", "Bearer") or "Bearer"),
        "base_url": _spotify_api_base_url(state),
        "scope": _clean(state.get("granted_scope") or state.get("scope")),
        "client_id": _spotify_client_id(state=state),
        "redirect_uri": _spotify_redirect_uri(state=state), "expires_at": state.get("expires_at"),
        "refresh_token": _clean(state.get("refresh_token")),
    }


def get_spotify_auth_status() -> Dict[str, Any]:
    from hermes_cli.auth import _is_expiring, get_provider_auth_state
    state = get_provider_auth_state("spotify")
    if not state:
        return {"logged_in": False}

    expires_at = state.get("expires_at")
    refresh_token = _clean(state.get("refresh_token"))
    return {
        "logged_in": bool(refresh_token or not _is_expiring(expires_at, 0)),
        "auth_type": state.get("auth_type", "oauth_pkce"), "client_id": state.get("client_id"),
        "redirect_uri": state.get("redirect_uri"),
        "scope": state.get("granted_scope") or state.get("scope"), "expires_at": expires_at,
        "api_base_url": state.get("api_base_url"), "has_refresh_token": bool(refresh_token),
    }


def _spotify_interactive_setup(redirect_uri_hint: str) -> str:
    """Walk the user through creating a Spotify developer app; persist the client_id to ~/.hermes/.env."""
    from hermes_cli.auth import _is_remote_session
    from hermes_cli.config import save_env_value
    print(
        f"\n{'=' * 70}\nSpotify first-time setup\n{'=' * 70}\n\n"
        "Spotify requires every user to register their own lightweight\n"
        "developer app. This takes about two minutes and only has to be\n"
        "done once per machine.\n\n"
        f"Full guide: {SPOTIFY_DOCS_URL}\n\n"
        "Steps:\n"
        f"  1. Opening {SPOTIFY_DASHBOARD_URL} in your browser...\n"
        "  2. Click 'Create app' and fill in:\n"
        "       App name:     anything (e.g. hermes-agent)\n"
        "       Description:  anything\n"
        f"       Redirect URI: {redirect_uri_hint}\n"
        "       API/SDK:      Web API\n"
        "  3. Agree to the terms, click Save.\n"
        "  4. Open the app's Settings page and copy the Client ID.\n"
        "  5. Paste it below.\n"
    )

    if not _is_remote_session():
        try:
            webbrowser.open(SPOTIFY_DASHBOARD_URL)
        except Exception:
            pass

    from hermes_cli.cli_output import line_input
    try:
        raw = line_input("Spotify Client ID: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Spotify setup cancelled.")

    if not raw:
        print(f"\nNo Client ID entered. See {SPOTIFY_DOCS_URL} for the full guide.")
        raise SystemExit("Spotify setup cancelled: empty Client ID.")

    # Persist so later runs skip the wizard; only pin a NON-default redirect URI.
    save_env_value("HERMES_SPOTIFY_CLIENT_ID", raw)
    if redirect_uri_hint and redirect_uri_hint != DEFAULT_SPOTIFY_REDIRECT_URI:
        save_env_value("HERMES_SPOTIFY_REDIRECT_URI", redirect_uri_hint)

    print("\nSaved HERMES_SPOTIFY_CLIENT_ID to ~/.hermes/.env\n")
    return raw


def login_spotify_command(args) -> None:
    from hermes_cli.auth import _auth_store_lock, _can_open_graphical_browser, _is_remote_session, _load_auth_store, _print_loopback_ssh_hint, _save_auth_store, _store_provider_state, get_provider_auth_state
    existing_state = get_provider_auth_state("spotify") or {}

    # No client_id anywhere -> wizard instead of "HERMES_SPOTIFY_CLIENT_ID is required".
    try:
        client_id = _spotify_client_id(getattr(args, "client_id", None), existing_state)
    except AuthError as exc:
        if getattr(exc, "code", "") != "spotify_client_id_missing":
            raise
        client_id = _spotify_interactive_setup(
            redirect_uri_hint=getattr(args, "redirect_uri", None) or DEFAULT_SPOTIFY_REDIRECT_URI,
        )

    redirect_uri = _spotify_redirect_uri(getattr(args, "redirect_uri", None), existing_state)
    scope = _spotify_scope_string(getattr(args, "scope", None) or existing_state.get("scope"))
    accounts_base_url = _spotify_accounts_base_url(existing_state)
    api_base_url = _spotify_api_base_url(existing_state)
    open_browser = not getattr(args, "no_browser", False)

    code_verifier = _spotify_code_verifier()
    state_nonce = uuid.uuid4().hex
    authorize_url = _spotify_build_authorize_url(
        client_id=client_id, redirect_uri=redirect_uri, scope=scope, state=state_nonce,
        code_challenge=_spotify_code_challenge(code_verifier), accounts_base_url=accounts_base_url,
    )

    print(
        f"Starting Spotify PKCE login...\nClient ID: {client_id}\nRedirect URI: {redirect_uri}\n"
        "Make sure this redirect URI is allow-listed in your Spotify app settings.\n\n"
        f"Open this URL to authorize Hermes:\n{authorize_url}\n\nFull setup guide: {SPOTIFY_DOCS_URL}\n"
    )

    _print_loopback_ssh_hint(redirect_uri, docs_url=SPOTIFY_DOCS_URL)

    if open_browser and not _is_remote_session() and _can_open_graphical_browser():
        try:
            opened = webbrowser.open(authorize_url)
        except Exception:
            opened = False
        print(
            "Browser opened for Spotify authorization." if opened
            else "Could not open the browser automatically; use the URL above."
        )

    callback = _spotify_wait_for_callback(redirect_uri, timeout_seconds=float(getattr(args, "timeout", None) or 180.0))
    if callback.get("error"):
        raise SystemExit(f"Spotify authorization failed: {callback.get('error_description') or callback['error']}")
    if callback.get("state") != state_nonce:
        raise SystemExit("Spotify authorization failed: state mismatch.")

    token_payload = _spotify_token_post(
        accounts_base_url,
        {
            "client_id": client_id, "grant_type": "authorization_code",
            "code": str(callback.get("code") or ""), "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
        timeout_seconds=float(getattr(args, "timeout", None) or 20.0),
        what="token exchange", failed_code="spotify_token_exchange_failed",
        invalid_code="spotify_token_exchange_invalid",
        invalid_message="Spotify token response did not include an access_token.",
    )
    spotify_state = _spotify_token_payload_to_state(
        token_payload, client_id=client_id, redirect_uri=redirect_uri, requested_scope=scope,
        accounts_base_url=accounts_base_url, api_base_url=api_base_url,
    )

    with _auth_store_lock():
        auth_store = _load_auth_store()
        _store_provider_state(auth_store, "spotify", spotify_state, set_active=False)
        saved_to = _save_auth_store(auth_store)

    print(
        f"Spotify login successful!\n  Auth state: {saved_to}\n"
        f"  Provider state saved under providers.spotify\n  Docs: {SPOTIFY_DOCS_URL}"
    )
