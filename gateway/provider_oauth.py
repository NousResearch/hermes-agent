"""Headless provider OAuth (PKCE) for the API server.

The dashboard already implements a paste-a-code Anthropic login
(``hermes_cli/web_server.py``), but that lives in the ``hermes web``
process. Deployments that run only ``hermes gateway`` — headless servers,
containers, k8s pods — have no way to (re)authenticate a provider except
an interactive TTY inside the process's host, which is exactly the thing
those deployments are built to avoid.

This module holds the transport-agnostic half of that flow so the API
server can expose it (see ``api_server._handle_provider_oauth_*``):

    start()  -> {session_id, auth_url}      # human opens the URL
    submit() -> {ok: True}                  # human pastes back "code#state"

No callback listener is required: Anthropic's redirect target is its own
hosted page, so the code travels via copy/paste and the pod never needs
inbound reachability from the browser.

The PKCE verifier must survive between the two calls, so sessions are held
in memory with a TTL. That is deliberate: a gateway is a single long-lived
process, and an unfinished login is worthless after the TTL anyway.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# An unfinished login is worthless after this; also bounds memory.
SESSION_TTL_SECONDS = 600
_MAX_SESSIONS = 32

AUTHORIZE_URL = "https://claude.ai/oauth/authorize"

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


class ProviderOAuthError(Exception):
    """Flow could not proceed. ``status`` is the HTTP status to return."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _anthropic_constants() -> Tuple[Any, ...]:
    """Import the adapter's OAuth constants, or fail with a clear error."""
    try:
        from agent.anthropic_adapter import (  # noqa: WPS433 - optional dep
            _OAUTH_CLIENT_ID,
            _OAUTH_REDIRECT_URI,
            _OAUTH_SCOPES,
            _OAUTH_TOKEN_URLS,
            _generate_pkce,
        )
    except Exception as exc:  # pragma: no cover - adapter always ships
        raise ProviderOAuthError(
            f"Anthropic OAuth unavailable: {exc}", status=501
        ) from exc
    return (
        _OAUTH_CLIENT_ID,
        _OAUTH_REDIRECT_URI,
        _OAUTH_SCOPES,
        _OAUTH_TOKEN_URLS,
        _generate_pkce,
    )


def _prune_locked(now: float) -> None:
    expired = [
        sid for sid, s in _sessions.items() if now - s["created_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _sessions.pop(sid, None)
    # Hard cap: drop oldest first if a caller spams start(). ``>=`` because
    # pruning runs before the caller inserts — leave room for one more.
    while len(_sessions) >= _MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k]["created_at"])
        _sessions.pop(oldest, None)


def start(provider: str = "anthropic") -> Dict[str, Any]:
    """Begin a PKCE login. Returns the auth URL for a human to open."""
    if provider != "anthropic":
        raise ProviderOAuthError(f"Unsupported provider: {provider}", status=400)
    client_id, redirect_uri, scopes, _token_urls, generate_pkce = _anthropic_constants()

    verifier, challenge = generate_pkce()
    session_id = uuid.uuid4().hex
    now = time.time()
    with _sessions_lock:
        _prune_locked(now)
        _sessions[session_id] = {
            "provider": provider,
            "verifier": verifier,
            # Anthropic round-trips the verifier as `state`.
            "state": verifier,
            "created_at": now,
        }

    params = {
        "code": "true",
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    return {
        "session_id": session_id,
        "provider": provider,
        "flow": "pkce",
        "auth_url": f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}",
        "expires_in": SESSION_TTL_SECONDS,
    }


def _exchange(token_urls, payload: bytes) -> Dict[str, Any]:
    """POST the code exchange, trying each known token host in order."""
    last_exc: Optional[Exception] = None
    for endpoint in token_urls:
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "hermes-gateway/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: PERF203 - fallback host is the point
            last_exc = exc
    raise ProviderOAuthError(f"Token exchange failed: {last_exc}", status=502)


def submit(session_id: str, code_input: str, provider: str = "anthropic") -> Dict[str, Any]:
    """Exchange the pasted code for tokens and persist the credential."""
    if provider != "anthropic":
        raise ProviderOAuthError(f"Unsupported provider: {provider}", status=400)
    client_id, redirect_uri, _scopes, token_urls, _gen = _anthropic_constants()

    now = time.time()
    with _sessions_lock:
        _prune_locked(now)
        sess = _sessions.get(session_id)
    if not sess or sess["provider"] != provider:
        raise ProviderOAuthError("Unknown or expired session", status=404)

    # Anthropic's callback page renders the code as `<code>#<state>`.
    parts = (code_input or "").strip().split("#", 1)
    code = parts[0].strip()
    if not code:
        raise ProviderOAuthError("No code provided", status=400)
    state = parts[1].strip() if len(parts) > 1 else sess["state"]

    result = _exchange(
        token_urls,
        json.dumps(
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": sess["verifier"],
            }
        ).encode(),
    )

    access_token = result.get("access_token") or ""
    refresh_token = result.get("refresh_token") or ""
    if not access_token:
        raise ProviderOAuthError("No access token returned", status=502)
    expires_at_ms = int(now * 1000) + int(result.get("expires_in") or 3600) * 1000

    _persist_anthropic(access_token, refresh_token, expires_at_ms)

    # One-shot: a consumed code cannot be replayed.
    with _sessions_lock:
        _sessions.pop(session_id, None)

    logger.info("[api_server] anthropic OAuth login completed (session=%s)", session_id)
    return {"ok": True, "provider": provider, "expires_at_ms": expires_at_ms}


def _persist_anthropic(access_token: str, refresh_token: str, expires_at_ms: int) -> None:
    """Write the credential file and register it in the pool.

    Mirrors ``hermes auth add anthropic`` (and the dashboard flow) so a
    gateway-side login leaves the system in the same state. The running
    gateway picks the new credential up without a restart: the auth store
    is re-read from disk per resolution.
    """
    from agent.anthropic_adapter import _get_hermes_oauth_file
    from utils import atomic_json_write

    atomic_json_write(
        _get_hermes_oauth_file(),
        {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        },
        indent=2,
        mode=0o600,
    )

    # Best-effort pool registration: the file write above is what runtime
    # credential resolution reads; the pool only drives rotation.
    try:
        from agent.credential_pool import (
            AUTH_TYPE_OAUTH,
            SOURCE_MANUAL,
            PooledCredential,
            load_pool,
        )

        pool = load_pool("anthropic")
        source = f"{SOURCE_MANUAL}:gateway_pkce"
        for entry in [
            e for e in pool.entries() if getattr(e, "source", "").startswith(source)
        ]:
            try:
                pool.remove_entry(getattr(entry, "id", ""))
            except Exception:
                pass
        pool.add_entry(
            PooledCredential(
                provider="anthropic",
                id=secrets.token_hex(3),
                label="gateway PKCE",
                auth_type=AUTH_TYPE_OAUTH,
                priority=0,
                source=source,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at_ms=expires_at_ms,
            )
        )
    except Exception as exc:
        logger.warning("[api_server] anthropic pool add failed: %s", exc)
