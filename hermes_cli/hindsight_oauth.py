"""Hindsight Cloud OAuth 2.1 client — mirrors the MCP OAuth flow.

Runs the same dynamic-client-registration + PKCE + loopback flow that
Claude Code's MCP client uses against Hindsight Cloud's OAuth server, so
``hermes memory setup hindsight`` can authorize in the browser instead of
asking the user to paste an API key.

Endpoints (relative to the configured ``api_url``):
  POST {api_url}/oauth/register   — RFC 7591 dynamic client registration
  GET  {api_url}/oauth/authorize  — authorization (browser, PKCE S256)
  POST {api_url}/oauth/token      — authorization_code exchange + refresh

A loopback ``http://127.0.0.1:<port>/callback`` redirect URI is classified
as a native client by the server (no client secret, ``token_endpoint_auth_
method=none``). Credentials are stored in ``~/.hermes/.hindsight_oauth.json``
(0600), mirroring the Anthropic OAuth credential file in ``anthropic_adapter``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_CLIENT_NAME = "Hermes Agent"
_SCOPE = "openid profile email offline_access"
_CALLBACK_PATH = "/callback"
_CALLBACK_TIMEOUT_SECONDS = 180.0
_HTTP_TIMEOUT = 30.0
# Refresh the access token when it is within this window of expiry.
_REFRESH_SKEW_MS = 60_000


class HindsightOAuthError(Exception):
    """Raised when the Hindsight OAuth flow fails."""


class HindsightReauthRequired(HindsightOAuthError):
    """Raised when stored credentials can no longer be refreshed (re-login needed)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _endpoint(api_url: str | None, path: str) -> str:
    return (api_url or _DEFAULT_API_URL).rstrip("/") + path


def credentials_path() -> Path:
    return get_hermes_home() / ".hindsight_oauth.json"


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE ``code_verifier`` and ``code_challenge`` (S256)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


def _read_json(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise HindsightOAuthError(f"{exc.code} {exc.reason}: {detail}".strip()) from exc
    except urllib.error.URLError as exc:
        raise HindsightOAuthError(
            f"Could not reach the Hindsight OAuth server: {exc.reason}"
        ) from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HindsightOAuthError(
            f"Invalid JSON from {req.full_url}: {body[:200]}"
        ) from exc
    if not isinstance(data, dict):
        raise HindsightOAuthError(f"Unexpected response from {req.full_url}: {body[:200]}")
    return data


def _post_json(url: str, payload: dict[str, Any], timeout: float = _HTTP_TIMEOUT) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    return _read_json(req, timeout)


def _post_form(url: str, fields: dict[str, str], timeout: float = _HTTP_TIMEOUT) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=urlencode(fields).encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    return _read_json(req, timeout)


def _jwt_claims(token: str) -> dict[str, Any]:
    """Best-effort decode of a JWT payload (unverified — informational only)."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _org_id_from_jwt(token: str) -> str:
    """Best-effort read of the ``org_id`` claim (informational only, unverified)."""
    return str(_jwt_claims(token).get("org_id") or "")


# ---------------------------------------------------------------------------
# Loopback callback server (mirrors hermes_cli.auth Spotify/xAI pattern)
# ---------------------------------------------------------------------------

class _ReuseHTTPServer(HTTPServer):
    allow_reuse_address = True


def _make_callback_handler() -> tuple[type[BaseHTTPRequestHandler], dict[str, Any]]:
    result: dict[str, Any] = {"code": None, "state": None, "error": None, "error_description": None}

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != _CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return
            params = parse_qs(parsed.query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            error = params.get("error", [None])[0]
            error_description = params.get("error_description", [None])[0]

            # Send the full response BEFORE publishing the result. The waiter
            # tears the server down the instant ``code``/``error`` is set, which
            # would otherwise race this write and leave the browser tab hanging.
            if error:
                body = (
                    "<html><body><h1>Hindsight authorization failed.</h1>"
                    "You can close this tab and return to your terminal.</body></html>"
                )
            else:
                body = (
                    "<html><body><h1>Connected to Hindsight.</h1>"
                    "You can close this tab and return to your terminal.</body></html>"
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            try:
                self.wfile.flush()
            except Exception:
                pass

            # Publish last — this is what unblocks _wait_for_callback.
            result["state"] = state
            result["error"] = error
            result["error_description"] = error_description
            result["code"] = code

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

    return _CallbackHandler, result


def _start_loopback_server() -> tuple[HTTPServer, dict[str, Any], threading.Thread, int]:
    handler_cls, result = _make_callback_handler()
    try:
        server = _ReuseHTTPServer(("127.0.0.1", 0), handler_cls)
    except OSError as exc:
        raise HindsightOAuthError(f"Could not bind a local callback server: {exc}") from exc
    port = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()
    return server, result, thread, port


def _stop_loopback_server(server: HTTPServer, thread: threading.Thread) -> None:
    try:
        server.shutdown()
    except Exception:
        pass
    try:
        server.server_close()
    except Exception:
        pass
    thread.join(timeout=1.0)


def _wait_for_callback(result: dict[str, Any], *, timeout_seconds: float = _CALLBACK_TIMEOUT_SECONDS) -> dict[str, Any]:
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    while time.monotonic() < deadline:
        if result["code"] or result["error"]:
            return result
        time.sleep(0.1)
    raise HindsightOAuthError("Timed out waiting for the browser authorization callback.")


def _open_browser(auth_url: str, open_browser: bool) -> None:
    print()
    print("  Opening your browser to authorize Hermes with Hindsight Cloud.")
    print("  If it doesn't open automatically, visit:")
    print(f"\n  {auth_url}\n")
    if not open_browser:
        return
    try:
        from hermes_cli.auth import _can_open_graphical_browser as _can_open
    except Exception:
        _can_open = lambda: True  # noqa: E731 — degrade to prior behavior
    if _can_open():
        try:
            import webbrowser

            webbrowser.open(auth_url)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Credential storage
# ---------------------------------------------------------------------------

def read_credentials() -> dict[str, Any] | None:
    """Read stored OAuth credentials, or ``None`` if absent/unusable."""
    path = credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read Hindsight OAuth credentials: %s", exc)
        return None
    if not isinstance(data, dict) or not data.get("access_token"):
        return None
    return data


def write_credentials(creds: dict[str, Any]) -> None:
    """Persist OAuth credentials to ``~/.hermes/.hindsight_oauth.json`` (0600)."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from utils import atomic_json_write

        atomic_json_write(path, creds, mode=0o600)
    except Exception:
        # Fallback: write directly with restrictive permissions.
        path.write_text(json.dumps(creds, indent=2), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def clear_credentials() -> None:
    """Remove stored OAuth credentials (used when a refresh token is dead)."""
    try:
        credentials_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Failed to remove Hindsight OAuth credentials: %s", exc)


def _build_credentials(api_url: str, client_id: str, token: dict[str, Any]) -> dict[str, Any]:
    access_token = token.get("access_token")
    if not access_token:
        raise HindsightOAuthError(f"Token response missing access_token: {token}")
    try:
        expires_in = int(token.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600
    claims = _jwt_claims(access_token)
    return {
        "client_id": client_id,
        "access_token": access_token,
        "refresh_token": token.get("refresh_token") or "",
        "expires_at_ms": _now_ms() + expires_in * 1000,
        "scope": token.get("scope") or _SCOPE,
        "api_url": api_url,
        "org_id": str(claims.get("org_id") or ""),
        "org_name": str(claims.get("org_name") or ""),
    }


# ---------------------------------------------------------------------------
# OAuth flow
# ---------------------------------------------------------------------------

def register_client(api_url: str, redirect_uri: str) -> str:
    """Register a dynamic OAuth client (RFC 7591) and return its ``client_id``."""
    resp = _post_json(
        _endpoint(api_url, "/oauth/register"),
        {"client_name": _CLIENT_NAME, "redirect_uris": [redirect_uri]},
    )
    client_id = resp.get("client_id")
    if not client_id:
        raise HindsightOAuthError(f"Registration response missing client_id: {resp}")
    return str(client_id)


def run_hindsight_oauth_login(api_url: str | None = None, *, open_browser: bool = True) -> dict[str, Any]:
    """Run the full browser OAuth flow and persist credentials.

    Returns the stored credential dict. Raises :class:`HindsightOAuthError`
    on any failure (registration, authorization, state mismatch, token
    exchange, timeout).
    """
    api_url = (api_url or _DEFAULT_API_URL).rstrip("/")
    server, result, thread, port = _start_loopback_server()
    try:
        redirect_uri = f"http://127.0.0.1:{port}{_CALLBACK_PATH}"
        client_id = register_client(api_url, redirect_uri)
        verifier, challenge = _generate_pkce()
        state = secrets.token_urlsafe(32)
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "scope": _SCOPE,
        }
        auth_url = _endpoint(api_url, "/oauth/authorize") + "?" + urlencode(params)
        _open_browser(auth_url, open_browser)
        callback = _wait_for_callback(result)
    finally:
        _stop_loopback_server(server, thread)

    if callback.get("error"):
        detail = callback.get("error_description") or ""
        raise HindsightOAuthError(f"Authorization failed: {callback['error']} {detail}".strip())
    if callback.get("state") != state:
        raise HindsightOAuthError("Authorization state mismatch — aborting (possible CSRF).")
    code = callback.get("code")
    if not code:
        raise HindsightOAuthError("No authorization code returned.")

    token = _post_form(
        _endpoint(api_url, "/oauth/token"),
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
    )
    creds = _build_credentials(api_url, client_id, token)
    write_credentials(creds)
    return creds


def refresh_access_token(api_url: str, client_id: str, refresh_token: str) -> dict[str, Any]:
    """Exchange a refresh token for a new access/refresh token pair."""
    return _post_form(
        _endpoint(api_url, "/oauth/token"),
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        },
    )


def get_valid_access_token(
    api_url: str | None = None, *, refresh_skew_ms: int = _REFRESH_SKEW_MS
) -> str | None:
    """Return a non-expired access token, refreshing transparently if needed.

    Returns ``None`` when no credentials are stored. Raises
    :class:`HindsightReauthRequired` when the refresh token is dead (the
    caller should prompt the user to re-run setup).
    """
    creds = read_credentials()
    if not creds:
        return None
    target_api_url = (api_url or creds.get("api_url") or _DEFAULT_API_URL).rstrip("/")

    try:
        expires_at_ms = int(creds.get("expires_at_ms") or 0)
    except (TypeError, ValueError):
        expires_at_ms = 0
    if _now_ms() < expires_at_ms - refresh_skew_ms:
        return creds.get("access_token")

    refresh_token = creds.get("refresh_token")
    client_id = creds.get("client_id")
    if not refresh_token or not client_id:
        # Cannot refresh — hand back whatever we have and let the API reject it.
        return creds.get("access_token")

    try:
        token = refresh_access_token(target_api_url, client_id, refresh_token)
    except HindsightOAuthError as exc:
        if "invalid_grant" in str(exc):
            clear_credentials()
            raise HindsightReauthRequired(
                "Hindsight session expired. Re-run `hermes memory setup hindsight` to sign in again."
            ) from exc
        logger.warning("Hindsight token refresh failed: %s", exc)
        return creds.get("access_token")

    new_creds = _build_credentials(target_api_url, client_id, token)
    if not new_creds.get("org_id"):
        new_creds["org_id"] = creds.get("org_id", "")
    if not new_creds.get("org_name"):
        new_creds["org_name"] = creds.get("org_name", "")
    write_credentials(new_creds)
    return new_creds.get("access_token")
