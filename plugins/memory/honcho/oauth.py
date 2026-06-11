"""OAuth 2.1 credential storage and refresh for the Honcho memory provider.

Honcho's OAuth 2.1 authorization server issues opaque ``hch-at-`` access tokens
and ``hch-rt-`` refresh tokens; an access token authenticates the Honcho API
*exactly like a scoped API key*, so it is stored as the host's ``apiKey`` and
sent as the Bearer. Access tokens live one hour, so this module keeps the
stored token fresh by exchanging the refresh token before expiry.

Refresh uses rotation with single-use reuse detection: each refresh returns a
new refresh token and replaying a superseded one revokes the whole grant. So
two invariants matter here — every refresh must (1) persist the rotated refresh
token atomically and (2) be serialized, or concurrent refreshes would replay
the same token and self-revoke. A failed refresh never raises into the agent;
the stale token is left in place and the existing fail-open path handles the
eventual 401.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

ACCESS_TOKEN_PREFIX = "hch-at-"
REFRESH_TOKEN_PREFIX = "hch-rt-"

# Refresh this many seconds before the access token actually expires, so an
# in-flight request never races the expiry boundary.
_REFRESH_SKEW_SECONDS = 120

# Default HTTP timeout for the token exchange. Kept short — the refresh happens
# on the path to a memory call, and a stalled auth server must not hang it.
_REFRESH_TIMEOUT_SECONDS = 15.0

# Serializes refresh across threads sharing one process's config. Re-checked
# under the lock (double-checked) so racing callers don't replay a rotated
# refresh token and trip reuse detection.
_refresh_lock = threading.Lock()


def is_oauth_access_token(value: str | None) -> bool:
    """True when ``value`` is an OAuth access token (vs a static API key)."""
    return bool(value) and value.startswith(ACCESS_TOKEN_PREFIX)


@dataclass
class OAuthCredential:
    """An OAuth grant as stored in a honcho.json host block.

    ``access_token`` mirrors the host's ``apiKey``; the remaining fields live in
    the host's ``oauth`` sub-block. ``expires_at`` is absolute epoch seconds.
    """

    access_token: str
    refresh_token: str
    expires_at: float
    client_id: str
    token_endpoint: str
    scope: str = "write"
    token_type: str = "Bearer"

    @classmethod
    def from_host_block(cls, block: dict[str, Any]) -> "OAuthCredential | None":
        """Build a credential from a honcho.json host block, or None if incomplete."""
        oauth = block.get("oauth")
        access = block.get("apiKey")
        if not isinstance(oauth, dict) or not is_oauth_access_token(access):
            return None
        refresh = oauth.get("refreshToken")
        endpoint = oauth.get("tokenEndpoint")
        client_id = oauth.get("clientId")
        if not (refresh and endpoint and client_id):
            return None
        try:
            expires_at = float(oauth.get("expiresAt", 0))
        except (TypeError, ValueError):
            expires_at = 0.0
        return cls(
            access_token=access,
            refresh_token=str(refresh),
            expires_at=expires_at,
            client_id=str(client_id),
            token_endpoint=str(endpoint),
            scope=str(oauth.get("scope", "write")),
            token_type=str(oauth.get("tokenType", "Bearer")),
        )

    def oauth_block(self) -> dict[str, Any]:
        """The ``oauth`` sub-block to persist (the access token lives in apiKey)."""
        return {
            "refreshToken": self.refresh_token,
            "expiresAt": int(self.expires_at),
            "clientId": self.client_id,
            "tokenEndpoint": self.token_endpoint,
            "scope": self.scope,
            "tokenType": self.token_type,
        }

    def is_expired(self, *, now: float, skew: float = _REFRESH_SKEW_SECONDS) -> bool:
        """True when the access token is within ``skew`` seconds of expiry."""
        return now >= (self.expires_at - skew)


# Indirection so tests can drive the exchange without a live server.
def _http_post_form(url: str, data: dict[str, str], timeout: float) -> dict[str, Any]:
    """POST form-encoded ``data`` to ``url`` and return the parsed JSON body."""
    import httpx

    resp = httpx.post(url, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _exchange_refresh_token(cred: OAuthCredential, *, now: float) -> OAuthCredential:
    """Run the refresh_token grant and return the rotated credential.

    Raises on any transport/protocol failure; callers fail open.
    """
    body = _http_post_form(
        cred.token_endpoint,
        {
            "grant_type": "refresh_token",
            "client_id": cred.client_id,
            "refresh_token": cred.refresh_token,
        },
        _REFRESH_TIMEOUT_SECONDS,
    )
    access = body.get("access_token")
    refresh = body.get("refresh_token")
    if not is_oauth_access_token(access) or not refresh:
        raise ValueError("refresh response missing access_token/refresh_token")
    try:
        expires_in = int(body.get("expires_in", 0))
    except (TypeError, ValueError):
        expires_in = 0
    return OAuthCredential(
        access_token=access,
        refresh_token=str(refresh),
        expires_at=now + expires_in,
        client_id=cred.client_id,
        token_endpoint=cred.token_endpoint,
        scope=str(body.get("scope", cred.scope)),
        token_type=str(body.get("token_type", cred.token_type)),
    )


def _read_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write_config(path: Path, raw: dict[str, Any]) -> None:
    """Write ``raw`` to ``path`` atomically, preserving 0600 on the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    text = json.dumps(raw, indent=2) + "\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (overlay wins on scalars/lists)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _persist_credential(path: Path, host: str, cred: OAuthCredential) -> None:
    """Persist ``cred`` into ``host``'s block (apiKey + oauth), leaving all else intact."""
    raw = _read_config(path)
    hosts = raw.setdefault("hosts", {})
    block = hosts.setdefault(host, {})
    block["apiKey"] = cred.access_token
    block["oauth"] = cred.oauth_block()
    _atomic_write_config(path, raw)


def ensure_fresh_token(
    path: Path,
    host: str,
    raw: dict[str, Any] | None = None,
    *,
    now: float | None = None,
) -> tuple[str | None, bool]:
    """Return ``(access_token, refreshed)`` for ``host``, refreshing if near expiry.

    Returns ``(None, False)`` when the host has no OAuth credential (e.g. a plain
    API key) so callers leave the existing token untouched. Refresh failures are
    swallowed: the current (possibly stale) token is returned with
    ``refreshed=False`` and the fail-open path handles any resulting 401.
    """
    now = time.time() if now is None else now
    source = raw if raw is not None else _read_config(path)
    block = (source.get("hosts") or {}).get(host) or {}
    cred = OAuthCredential.from_host_block(block)
    if cred is None:
        return None, False

    if not cred.is_expired(now=now):
        return cred.access_token, False

    with _refresh_lock:
        # Re-read under the lock: another thread may have just rotated the token.
        fresh_block = (_read_config(path).get("hosts") or {}).get(host) or {}
        current = OAuthCredential.from_host_block(fresh_block) or cred
        if not current.is_expired(now=now):
            return current.access_token, current.access_token != cred.access_token
        try:
            rotated = _exchange_refresh_token(current, now=now)
        except Exception as exc:
            logger.warning("Honcho OAuth refresh failed for host %s: %s", host, exc)
            return current.access_token, False
        _persist_credential(path, host, rotated)
        logger.info("Honcho OAuth token refreshed for host %s", host)
        return rotated.access_token, True


def install_grant(
    path: Path,
    host: str,
    grant: dict[str, Any],
    *,
    client_id: str,
    token_endpoint: str,
    now: float | None = None,
) -> OAuthCredential:
    """Apply a fresh OAuth grant to ``path`` for ``host``.

    Deep-merges the grant's ``config`` (the manifest default_config) into the
    file root — preserving other hosts and root keys — then writes the host's
    ``apiKey`` and ``oauth`` block. ``grant`` is an OAuthTokenResponse dict
    (access_token, refresh_token, expires_in, scope, config).
    """
    now = time.time() if now is None else now
    access = grant.get("access_token")
    refresh = grant.get("refresh_token")
    if not is_oauth_access_token(access) or not refresh:
        raise ValueError("grant missing access_token/refresh_token")
    try:
        expires_in = int(grant.get("expires_in", 0))
    except (TypeError, ValueError):
        expires_in = 0

    cred = OAuthCredential(
        access_token=access,
        refresh_token=str(refresh),
        expires_at=now + expires_in,
        client_id=client_id,
        token_endpoint=token_endpoint,
        scope=str(grant.get("scope", "write")),
        token_type=str(grant.get("token_type", "Bearer")),
    )

    raw = _read_config(path)
    granted_config = grant.get("config")
    if isinstance(granted_config, dict):
        _deep_merge(raw, granted_config)
    hosts = raw.setdefault("hosts", {})
    block = hosts.setdefault(host, {})
    block["apiKey"] = cred.access_token
    block["oauth"] = cred.oauth_block()
    _atomic_write_config(path, raw)
    return cred


def apply_token_to_client(client: Any, token: str) -> bool:
    """Rotate the live Honcho client's Bearer in place. Returns success.

    The SDK builds its auth header per request from the HTTP client's
    ``api_key``, so mutating it rotates every holder of the singleton without a
    rebuild. Guarded: an SDK shape change degrades to False and the caller can
    fall back to resetting the client.
    """
    http = getattr(client, "_http", None)
    if http is None or not hasattr(http, "api_key"):
        return False
    http.api_key = token
    return True
