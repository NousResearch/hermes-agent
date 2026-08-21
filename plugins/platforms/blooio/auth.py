"""
Blooio Apps OAuth 2.0 client for Hermes Agent.

Blooio's OAuth server (``https://api.blooio.com/oauth``) implements the
Authorization Code grant with PKCE (S256) plus rotating refresh tokens — the
standard native-app flow (RFC 8252). This module drives that flow from the CLI:

  * ``login()``   opens the browser to the consent screen, captures the code on
                  a loopback redirect, exchanges it (PKCE) for an access +
                  refresh token, and persists them.
  * ``logout()``  revokes + clears the stored tokens.
  * ``status()``  reports the current auth state.

Runtime auth is resolved by :func:`resolve_auth`, which returns a
:class:`BlooioAuth` that yields a fresh bearer token (auto-refreshing the OAuth
access token as it nears expiry) and the organization id to scope requests
with. An API key (``BLOOIO_API_KEY``) is honored as a headless/CI fallback.

Tokens are stored in ``~/.hermes/auth.json`` under ``credential_pool.blooio``
(0o600, atomic write), mirroring the Photon plugin's credential handling.
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
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The public, official "Blooio for Hermes" OAuth app. Overridable for staging
# via BLOOIO_CLIENT_ID. Public client => PKCE, no client secret is shipped.
DEFAULT_CLIENT_ID = "bloapp_iJGDMbM_iT9i7HH_cVr55GXI"

DEFAULT_API_HOST = "https://api.blooio.com"

# Scopes the iMessage gateway needs: send/read messages + reactions, manage
# chats (typing/read/contact-card), read channels, and manage the inbound
# webhook (create + rotate signing secret).
DEFAULT_SCOPES = [
    "messages:read",
    "messages:write",
    "chats:read",
    "chats:write",
    "channels:read",
    "webhooks:read",
    "webhooks:write",
]

# Loopback redirect ports registered on the app (exact-match). The CLI binds
# the first free one. Keep in sync with the app's registered redirect_uris.
LOOPBACK_PORTS = [8765, 8766, 8767]
REDIRECT_PATH = "/callback"

# Refresh the access token this many seconds before its stated expiry.
_REFRESH_SKEW_SECONDS = 120


def _api_host() -> str:
    return (os.getenv("BLOOIO_API_HOST") or DEFAULT_API_HOST).rstrip("/")


def _client_id() -> str:
    return os.getenv("BLOOIO_CLIENT_ID") or DEFAULT_CLIENT_ID


def authorize_url() -> str:
    return f"{_api_host()}/oauth/authorize"


def token_url() -> str:
    return f"{_api_host()}/oauth/token"


def revoke_url() -> str:
    return f"{_api_host()}/oauth/revoke"


# ---------------------------------------------------------------------------
# Token storage (auth.json credential_pool.blooio)
# ---------------------------------------------------------------------------

def _load_auth() -> Dict[str, Any]:
    try:
        from hermes_cli.auth import _load_auth_store

        return _load_auth_store() or {}
    except Exception:
        path = _auth_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8")) or {}
            except Exception:
                return {}
        return {}


def _auth_path():
    from pathlib import Path

    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "auth.json"
    except Exception:
        return Path.home() / ".hermes" / "auth.json"


def _save_auth(data: Dict[str, Any]) -> None:
    try:
        from hermes_cli.auth import _save_auth_store

        _save_auth_store(data)
        return
    except Exception:
        pass
    # Fallback: atomic 0o600 write.
    import stat
    import uuid

    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _store_tokens(record: Dict[str, Any]) -> None:
    auth = _load_auth()
    auth.setdefault("credential_pool", {})["blooio"] = [record]
    _save_auth(auth)


def _load_tokens() -> Optional[Dict[str, Any]]:
    auth = _load_auth()
    pool = auth.get("credential_pool", {}).get("blooio") or []
    if isinstance(pool, list) and pool:
        return pool[0]
    return None


def clear_tokens() -> None:
    auth = _load_auth()
    pool = auth.get("credential_pool", {})
    if pool.get("blooio"):
        pool["blooio"] = []
        _save_auth(auth)


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_pkce() -> Tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for PKCE S256."""
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


# ---------------------------------------------------------------------------
# Runtime auth resolution
# ---------------------------------------------------------------------------

class BlooioAuth:
    """Resolved runtime auth: yields a bearer token + organization scope.

    ``mode`` is ``"api_key"`` (static bearer) or ``"oauth"`` (auto-refreshing
    access token). ``organization_id`` scopes requests via ``X-Organization-Id``
    when set (required for OAuth tokens installed on multiple orgs).
    """

    def __init__(
        self,
        mode: str,
        *,
        api_key: str = "",
        organization_id: str = "",
    ) -> None:
        self.mode = mode
        self._api_key = api_key
        self.organization_id = organization_id or os.getenv("BLOOIO_ORG_ID", "") or ""
        self._lock = threading.Lock()

    async def bearer(self) -> str:
        if self.mode == "api_key":
            return self._api_key
        return await self._oauth_bearer()

    async def _oauth_bearer(self) -> str:
        record = _load_tokens()
        if not record:
            raise BlooioAuthError(
                "Not authenticated with Blooio. Run `hermes blooio login`."
            )
        now = time.time()
        expires_at = float(record.get("expires_at", 0) or 0)
        if record.get("access_token") and expires_at - now > _REFRESH_SKEW_SECONDS:
            return str(record["access_token"])
        refreshed = await _refresh_tokens(record)
        return str(refreshed["access_token"])


class BlooioAuthError(Exception):
    """Raised when Blooio auth is missing or cannot be refreshed."""


def resolve_auth(config: Any = None) -> Optional[BlooioAuth]:
    """Determine runtime auth: API key first (headless), else stored OAuth."""
    extra = getattr(config, "extra", {}) or {}
    api_key = os.getenv("BLOOIO_API_KEY") or extra.get("api_key", "")
    if api_key:
        org = os.getenv("BLOOIO_ORG_ID") or extra.get("organization_id", "") or ""
        return BlooioAuth("api_key", api_key=api_key, organization_id=org)
    if _load_tokens():
        org = (
            os.getenv("BLOOIO_ORG_ID")
            or extra.get("organization_id", "")
            or (_load_tokens() or {}).get("organization_id", "")
            or ""
        )
        return BlooioAuth("oauth", organization_id=org)
    return None


def has_credentials() -> bool:
    return bool(os.getenv("BLOOIO_API_KEY") or _load_tokens())


# ---------------------------------------------------------------------------
# Token exchange / refresh (async, httpx)
# ---------------------------------------------------------------------------

async def _post_token(form: Dict[str, str]) -> Dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            token_url(),
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code >= 400:
        raise BlooioAuthError(f"token endpoint {resp.status_code}: {resp.text[:200]}")
    return resp.json() or {}


def _token_record(
    payload: Dict[str, Any],
    *,
    code_verifier: str,
    organization_id: str = "",
) -> Dict[str, Any]:
    now = time.time()
    expires_in = float(payload.get("expires_in", 3600) or 3600)
    return {
        "access_token": payload.get("access_token"),
        "refresh_token": payload.get("refresh_token"),
        "expires_at": now + expires_in,
        "scope": payload.get("scope", ""),
        "code_verifier": code_verifier,
        "organization_id": organization_id,
        "issued_at": int(now),
    }


async def _refresh_tokens(record: Dict[str, Any]) -> Dict[str, Any]:
    refresh_token = record.get("refresh_token")
    if not refresh_token:
        raise BlooioAuthError("No refresh token; run `hermes blooio login` again.")
    # Public-client refresh requires a code_verifier to be *present* (PKCE was
    # validated at code exchange; refresh only checks presence). Reuse the one
    # captured at login.
    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": _client_id(),
        "code_verifier": record.get("code_verifier") or "pkce",
    }
    payload = await _post_token(form)
    updated = _token_record(
        payload,
        code_verifier=record.get("code_verifier") or "pkce",
        organization_id=record.get("organization_id", ""),
    )
    _store_tokens(updated)
    return updated


# ---------------------------------------------------------------------------
# Interactive login (loopback PKCE)
# ---------------------------------------------------------------------------

def _pick_free_port() -> Tuple[int, Any]:
    """Bind the first free registered loopback port; return (port, socket)."""
    import socket

    last_err: Optional[Exception] = None
    for port in LOOPBACK_PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
            return port, sock
        except OSError as exc:
            last_err = exc
            sock.close()
    raise BlooioAuthError(
        f"No registered loopback port is free ({LOOPBACK_PORTS}): {last_err}"
    )


def _capture_code(sock: Any, expected_state: str, timeout: float = 300.0) -> str:
    """Accept one loopback request, return the ?code=, and close the browser tab."""
    sock.settimeout(timeout)
    conn, _ = sock.accept()
    try:
        conn.settimeout(10.0)
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        request_line = data.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        path = request_line.split(" ")[1] if " " in request_line else ""
        query = urllib.parse.urlparse(path).query
        params = urllib.parse.parse_qs(query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]
        err = (params.get("error") or [""])[0]

        if err:
            body = f"Authorization failed: {err}. You can close this tab."
        elif not code or state != expected_state:
            body = "Authorization failed (state mismatch). You can close this tab."
            code = ""
        else:
            body = "Blooio connected to Hermes. You can close this tab and return to the terminal."
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
            b"Connection: close\r\n\r\n"
            + f"<html><body style='font-family:sans-serif'>{body}</body></html>".encode("utf-8")
        )
        if not code:
            raise BlooioAuthError(body)
        return code
    finally:
        conn.close()


async def login(
    *,
    scopes: Optional[List[str]] = None,
    open_browser: bool = True,
    organization_id: str = "",
) -> Dict[str, Any]:
    """Run the loopback PKCE authorization-code flow and persist tokens."""
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(24)
    port, sock = _pick_free_port()
    try:
        redirect_uri = f"http://127.0.0.1:{port}{REDIRECT_PATH}"
        params = {
            "response_type": "code",
            "client_id": _client_id(),
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": " ".join(scopes or DEFAULT_SCOPES),
            "state": state,
        }
        url = f"{authorize_url()}?{urllib.parse.urlencode(params)}"
        print("\nOpen this URL to connect Blooio to Hermes:\n")
        print(f"  {url}\n")
        if open_browser:
            try:
                import webbrowser

                webbrowser.open(url, new=2)
            except Exception:
                pass
        print("Waiting for authorization…")

        import asyncio

        code = await asyncio.get_event_loop().run_in_executor(
            None, _capture_code, sock, state
        )
    finally:
        sock.close()

    payload = await _post_token(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"http://127.0.0.1:{port}{REDIRECT_PATH}",
            "client_id": _client_id(),
            "code_verifier": verifier,
        }
    )
    record = _token_record(payload, code_verifier=verifier, organization_id=organization_id)
    # Resolve the org the token acts on (auto for single-install) unless given.
    if not record["organization_id"]:
        record["organization_id"] = await _discover_organization(record["access_token"])
    _store_tokens(record)
    return record


async def _discover_organization(access_token: str) -> str:
    """Best-effort: read the token's org from GET /v4/me (single-install auto-resolves)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_api_host()}/v4/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if resp.status_code == 400:
            # Multi-org install: server returns the list; caller must pick one.
            body = resp.json()
            orgs = body.get("organization_ids") or []
            if orgs:
                logger.warning(
                    "[blooio] token spans multiple orgs %s — set BLOOIO_ORG_ID.", orgs
                )
            return ""
        data = (resp.json() or {}).get("data") or {}
        return data.get("organization_id") or (data.get("organization") or {}).get(
            "organization_id", ""
        ) or ""
    except Exception as exc:
        logger.debug("[blooio] org discovery failed: %s", exc)
        return ""


async def logout() -> None:
    """Revoke the stored refresh token and clear local credentials."""
    record = _load_tokens()
    if record and record.get("refresh_token"):
        import httpx

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.post(
                    revoke_url(),
                    data={
                        "token": record["refresh_token"],
                        "client_id": _client_id(),
                        "code_verifier": record.get("code_verifier") or "pkce",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
        except Exception as exc:
            logger.debug("[blooio] revoke failed: %s", exc)
    clear_tokens()


def status() -> Dict[str, Any]:
    """Return a non-secret snapshot of the current auth state."""
    if os.getenv("BLOOIO_API_KEY"):
        return {"mode": "api_key", "organization_id": os.getenv("BLOOIO_ORG_ID", "")}
    record = _load_tokens()
    if not record:
        return {"mode": "none"}
    now = time.time()
    return {
        "mode": "oauth",
        "organization_id": record.get("organization_id", ""),
        "scope": record.get("scope", ""),
        "access_expires_in": max(0, int(float(record.get("expires_at", 0) or 0) - now)),
        "has_refresh_token": bool(record.get("refresh_token")),
    }
