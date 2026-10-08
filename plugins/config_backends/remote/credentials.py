"""The plane credential (design §4.4 step 2, D26, D31, D32).

Cloud: the Nous access token from the profile's ``auth.json`` (``resolve_nous_access_token``).
On-prem: an IdP workload token from the OAuth2 client-credentials grant, configured by
``GATEWAY_RELAY_IDP_*`` values in ``.env``. Never from config.yaml (the config is what this token
fetches) and never from a secret source: :data:`PLANE_CREDENTIAL_ENV_NAMES` lists every env name
that decides which token is sent where, and boot refuses a secret source that supplies one.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

from hermes_cli.urllib_security import open_credentialed_url

IDP_ENV_PREFIX = "GATEWAY_RELAY_IDP_"
_IDP_KEYS = ("token_url", "client_id", "client_secret", "scope")

PLANE_CREDENTIAL_ENV_NAMES = frozenset({
    # which backend, which plane, which agent record
    "HERMES_CONFIG_BACKEND", "HERMES_CONFIG_REMOTE_URL", "HERMES_CONFIG_INSTANCE_ID",
    # on-prem workload identity
    *(IDP_ENV_PREFIX + k.upper() for k in _IDP_KEYS),
    # Cloud: where auth.json lives and where its refresh token is sent
    "HERMES_SHARED_AUTH_DIR", "HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL",
})

_HTTP_TIMEOUT_S = 15.0


class PlaneCredentialError(RuntimeError):
    """No plane credential could be obtained. ``retryable`` is False for a missing or partial
    configuration (retrying cannot help)."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


_IDP_CACHE: Dict[Tuple[str, str, str], Tuple[float, str]] = {}
_IDP_LOCK = threading.Lock()


def _idp_env() -> Dict[str, str]:
    return {k: os.environ.get(IDP_ENV_PREFIX + k.upper(), "").strip() for k in _IDP_KEYS}


def credential_kind() -> str:
    return "idp" if _idp_env()["token_url"] else "nous"


def plane_token(home: Path) -> str:
    """A fresh bearer for ``home``'s requests (re-resolved per request, contract §10.2)."""
    return _idp_token() if credential_kind() == "idp" else _nous_token(home)


def _stored_nous_token(home: Path) -> str:
    """The profile's stored Nous access token while it is not near expiry, read straight from
    ``auth.json``, else ``""``. The state is the profile's own when its store has a ``nous`` entry
    (a guest identity saved before its first token exchange included) and the root's only when it
    has none, like ``hermes_cli.auth._load_provider_state_with_source``.

    Config-independent on purpose: importing ``hermes_cli.auth`` reads config (plugin discovery),
    and in remote mode that read is this very fetch, so the boot fetch cannot need that module
    while the token is still good. Anything else is resolved through the full resolver."""
    from hermes_cli.auth_constants import ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    from hermes_constants import get_default_hermes_root
    for path in dict.fromkeys((Path(home) / "auth.json", get_default_hermes_root() / "auth.json")):
        try:
            providers = json.loads(path.read_text(encoding="utf-8-sig")).get("providers")
        except (OSError, ValueError, AttributeError):
            continue
        state = providers.get("nous") if isinstance(providers, dict) else None
        if not isinstance(state, dict):
            continue
        try:  # this store owns the login from here on: a gap means the resolver, never the root
            token, expires = state["access_token"], datetime.fromisoformat(state["expires_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError, AttributeError):
            return ""
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        fresh = expires.timestamp() > time.time() + ACCESS_TOKEN_REFRESH_SKEW_SECONDS
        return token if fresh and isinstance(token, str) else ""
    return ""


def _nous_token(home: Path) -> str:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    stored = _stored_nous_token(home)
    if stored:
        return stored
    try:
        from hermes_cli.auth import resolve_nous_access_token
    except Exception as exc:  # noqa: BLE001 — an import failure is a broken install, not transient
        raise PlaneCredentialError(f"cannot load the Nous auth module: {exc}", retryable=False) from exc
    token = set_hermes_home_override(home)  # auth.json is per profile home
    try:
        return resolve_nous_access_token()
    except Exception as exc:  # noqa: BLE001 — AuthError and network errors alike
        # A missing login or a revoked session needs a human; a refresh that hit the network can retry.
        fatal = bool(getattr(exc, "relogin_required", False)) or getattr(exc, "code", None) == "nous_auth_missing"
        raise PlaneCredentialError(
            f"no Nous access token for {home} (auth.json): {exc}", retryable=not fatal) from exc
    finally:
        reset_hermes_home_override(token)


def _idp_token() -> str:
    env = _idp_env()
    token_url, client_id, client_secret, scope = (env[k] for k in _IDP_KEYS)
    if not client_id or not client_secret:
        raise PlaneCredentialError(
            f"{IDP_ENV_PREFIX}TOKEN_URL is set but {IDP_ENV_PREFIX}CLIENT_ID / {IDP_ENV_PREFIX}CLIENT_SECRET "
            "are not: Remote Config needs the OAuth2 client-credentials grant (both values in .env).",
            retryable=False)
    key = (token_url, client_id, scope)
    now = time.monotonic()
    with _IDP_LOCK:
        cached = _IDP_CACHE.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
    form = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    if scope:
        form["scope"] = scope
    req = urllib.request.Request(
        token_url, data=urllib.parse.urlencode(form).encode(), method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
    try:
        with open_credentialed_url(req, timeout=_HTTP_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode() or "{}")
    except Exception as exc:  # noqa: BLE001 — HTTP and network errors alike; the message has no secret
        raise PlaneCredentialError(f"IdP token request to {token_url} failed: {exc}", retryable=True) from exc
    access_token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise PlaneCredentialError("IdP client-credentials response had no access_token", retryable=True)
    expires_in = payload.get("expires_in")
    ttl = float(expires_in) if isinstance(expires_in, int | float) and expires_in > 0 else 60.0
    with _IDP_LOCK:
        _IDP_CACHE[key] = (now + max(ttl - 30.0, 0.0), access_token)
    return access_token
