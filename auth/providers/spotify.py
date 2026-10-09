"""Canonical spotify authentication mechanics; no CLI dependencies."""

from __future__ import annotations

import logging


from datetime import datetime, timezone

from typing import Any, Dict, Optional, Tuple

from urllib.parse import urlencode, urlparse

from auth.errors import AuthError

from auth.constants import (
    DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
    DEFAULT_SPOTIFY_API_BASE_URL,
    DEFAULT_SPOTIFY_REDIRECT_URI,
    DEFAULT_SPOTIFY_SCOPE,
    SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    _spotify_err,
    httpx,
)

from auth.oauth import (
    _bind_loopback_callback_server,
    _make_loopback_callback_handler,
    _serve_loopback_callback,
)

logger = logging.getLogger("hermes_cli.auth")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _spotify_scope_string(raw_scope: Optional[str] = None) -> str:
    """Requested scope, whitespace-normalized and de-duplicated (order kept)."""
    return " ".join(dict.fromkeys((raw_scope or DEFAULT_SPOTIFY_SCOPE).split()))


def _spotify_setting(
    state: Optional[Dict[str, Any]],
    state_key: str,
    env_vars: Tuple[str, ...],
    default: str,
    *,
    explicit: Optional[str] = None,
    strip_slash: bool = False,
    environment,
) -> str:
    """First non-empty of explicit arg, env vars (``.env`` aware), stored state, then *default*."""
    environment.require_current_scope()
    pass
    candidates = (
        explicit,
        *(environment.read_env().get(var) for var in env_vars),
        state.get(state_key) if isinstance(state, dict) else None,
        default,
    )
    for candidate in candidates:
        cleaned = _clean(candidate)
        if strip_slash:
            cleaned = cleaned.rstrip("/")
        if cleaned:
            return cleaned
    return default


def _spotify_client_id(
    explicit: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    *,
    environment,
) -> str:
    environment.require_current_scope()
    client_id = _spotify_setting(
        state,
        "client_id",
        ("HERMES_SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_ID"),
        "",
        explicit=explicit,
        environment=environment,
    )
    if client_id:
        return client_id
    raise _spotify_err(
        "Spotify client_id is required. Set HERMES_SPOTIFY_CLIENT_ID or pass --client-id.",
        "spotify_client_id_missing",
    )


def _spotify_redirect_uri(
    explicit: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    *,
    environment,
) -> str:
    environment.require_current_scope()
    return _spotify_setting(
        state,
        "redirect_uri",
        ("HERMES_SPOTIFY_REDIRECT_URI", "SPOTIFY_REDIRECT_URI"),
        DEFAULT_SPOTIFY_REDIRECT_URI,
        explicit=explicit,
        environment=environment,
    )


def _spotify_api_base_url(
    state: Optional[Dict[str, Any]] = None, *, environment
) -> str:
    environment.require_current_scope()
    return _spotify_setting(
        state,
        "api_base_url",
        ("HERMES_SPOTIFY_API_BASE_URL",),
        DEFAULT_SPOTIFY_API_BASE_URL,
        strip_slash=True,
        environment=environment,
    )


def _spotify_accounts_base_url(
    state: Optional[Dict[str, Any]] = None, *, environment
) -> str:
    environment.require_current_scope()
    return _spotify_setting(
        state,
        "accounts_base_url",
        ("HERMES_SPOTIFY_ACCOUNTS_BASE_URL",),
        DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
        strip_slash=True,
        environment=environment,
    )


def _spotify_build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
    code_challenge: str,
    accounts_base_url: str,
) -> str:
    query = urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    })
    return f"{accounts_base_url}/authorize?{query}"


def _spotify_validate_redirect_uri(redirect_uri: str) -> tuple[str, int, str]:
    parsed = urlparse(redirect_uri)
    host = parsed.hostname or ""
    problem = (
        "must use http://localhost or http://127.0.0.1."
        if parsed.scheme != "http"
        else "must point to localhost or 127.0.0.1."
        if host not in {"127.0.0.1", "localhost"}
        else "must include an explicit localhost port."
        if not parsed.port
        else None
    )
    if problem:
        raise _spotify_err(
            f"Spotify PKCE redirect_uri {problem}", "spotify_redirect_invalid"
        )
    return host, parsed.port, parsed.path or "/"


def _spotify_wait_for_callback(
    redirect_uri: str, *, timeout_seconds: float = 180.0
) -> dict[str, Any]:
    host, port, path = _spotify_validate_redirect_uri(redirect_uri)
    handler_cls, result = _make_loopback_callback_handler(path, display_name="Spotify")
    server = _bind_loopback_callback_server(
        host,
        port,
        handler_cls,
        err=_spotify_err,
        bind_failed_code="spotify_callback_bind_failed",
    )
    return _serve_loopback_callback(
        server,
        result,
        timeout_seconds=timeout_seconds,
        err=_spotify_err,
        timeout_code="spotify_callback_timeout",
    )


def _spotify_token_payload_to_state(
    token_payload: Dict[str, Any],
    *,
    client_id: str,
    redirect_uri: str,
    requested_scope: str,
    accounts_base_url: str,
    api_base_url: str,
    previous_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from auth.oauth import _coerce_ttl_seconds

    now = datetime.now(timezone.utc)
    expires_in = _coerce_ttl_seconds(token_payload.get("expires_in", 0))
    expires_at = datetime.fromtimestamp(now.timestamp() + expires_in, tz=timezone.utc)
    state = dict(previous_state or {})
    state.update({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "accounts_base_url": accounts_base_url,
        "api_base_url": api_base_url,
        "scope": requested_scope,
        "granted_scope": str(token_payload.get("scope") or requested_scope).strip(),
        "token_type": _clean(token_payload.get("token_type", "Bearer") or "Bearer")
        or "Bearer",
        "access_token": _clean(token_payload.get("access_token")),
        "refresh_token": _clean(
            token_payload.get("refresh_token") or state.get("refresh_token")
        ),
        "obtained_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "expires_in": expires_in,
        "auth_type": "oauth_pkce",
    })
    return state


def _spotify_token_post(
    accounts_base_url: str,
    data: Dict[str, str],
    *,
    timeout_seconds: float,
    what: str,
    failed_code: str,
    invalid_code: str,
    invalid_message: str,
    failed_suffix: str = "",
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
            f"Spotify {what} failed.{failed_suffix}"
            + (f" Response: {detail}" if detail else ""),
            failed_code,
            relogin=relogin_required,
        )
    payload = response.json()
    if not isinstance(payload, dict) or not _clean(payload.get("access_token")):
        raise _spotify_err(invalid_message, invalid_code, relogin=relogin_required)
    return payload


def _refresh_spotify_oauth_state(
    state: Dict[str, Any], *, timeout_seconds: float = 20.0, environment
) -> Dict[str, Any]:
    environment.require_current_scope()
    refresh_token = _clean(state.get("refresh_token"))
    if not refresh_token:
        raise _spotify_err(
            "Spotify refresh token missing. Run `hermes auth spotify` again.",
            "spotify_refresh_token_missing",
            relogin=True,
        )

    client_id = _spotify_client_id(state=state, environment=environment)
    accounts_base_url = _spotify_accounts_base_url(state, environment=environment)
    payload = _spotify_token_post(
        accounts_base_url,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
        timeout_seconds=timeout_seconds,
        what="token refresh",
        failed_code="spotify_refresh_failed",
        invalid_code="spotify_refresh_invalid",
        invalid_message="Spotify refresh response did not include an access_token.",
        failed_suffix=" Run `hermes auth spotify` again.",
        relogin_required=True,
    )

    return _spotify_token_payload_to_state(
        payload,
        client_id=client_id,
        redirect_uri=_spotify_redirect_uri(state=state, environment=environment),
        requested_scope=str(state.get("scope") or DEFAULT_SPOTIFY_SCOPE),
        accounts_base_url=accounts_base_url,
        api_base_url=_spotify_api_base_url(state, environment=environment),
        previous_state=state,
    )


def resolve_spotify_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    environment,
) -> Dict[str, Any]:
    environment.require_current_scope()
    from auth.store import _auth_store_lock, _load_auth_store, _save_auth_store
    from auth.token_validation import _is_expiring
    from auth.oauth import _quarantine_flat_oauth_state
    from auth.providers.spotify import _refresh_spotify_oauth_state
    from auth.provider_state import _load_provider_state, _store_provider_state

    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "spotify")
        if not state:
            raise _spotify_err(
                "Spotify is not authenticated. Run `hermes auth spotify` first.",
                "spotify_auth_missing",
                relogin=True,
            )

        should_refresh = bool(force_refresh)
        if not should_refresh and refresh_if_expiring:
            should_refresh = _is_expiring(state.get("expires_at"), refresh_skew_seconds)
        if should_refresh:
            try:
                state = _refresh_spotify_oauth_state(state, environment=environment)
                _store_provider_state(auth_store, "spotify", state, set_active=False)
                _save_auth_store(auth_store)
            except AuthError as exc:
                if exc.relogin_required and state.get("refresh_token"):
                    _quarantine_flat_oauth_state(state, "spotify", exc)
                    try:
                        _store_provider_state(
                            auth_store, "spotify", state, set_active=False
                        )
                        _save_auth_store(auth_store)
                    except Exception as _save_exc:
                        logger.debug(
                            "Spotify OAuth: failed to persist quarantined state: %s",
                            _save_exc,
                        )
                raise

    access_token = _clean(state.get("access_token"))
    if not access_token:
        raise _spotify_err(
            "Spotify access token missing. Run `hermes auth spotify` again.",
            "spotify_access_token_missing",
            relogin=True,
        )

    return {
        "provider": "spotify",
        "access_token": access_token,
        "api_key": access_token,
        "token_type": str(state.get("token_type", "Bearer") or "Bearer"),
        "base_url": _spotify_api_base_url(state, environment=environment),
        "scope": _clean(state.get("granted_scope") or state.get("scope")),
        "client_id": _spotify_client_id(state=state, environment=environment),
        "redirect_uri": _spotify_redirect_uri(state=state, environment=environment),
        "expires_at": state.get("expires_at"),
        "refresh_token": _clean(state.get("refresh_token")),
    }


def get_spotify_auth_status() -> Dict[str, Any]:
    from auth.token_validation import _is_expiring
    from auth.provider_state import get_provider_auth_state

    state = get_provider_auth_state("spotify")
    if not state:
        return {"logged_in": False}

    expires_at = state.get("expires_at")
    refresh_token = _clean(state.get("refresh_token"))
    return {
        "logged_in": bool(refresh_token or not _is_expiring(expires_at, 0)),
        "auth_type": state.get("auth_type", "oauth_pkce"),
        "client_id": state.get("client_id"),
        "redirect_uri": state.get("redirect_uri"),
        "scope": state.get("granted_scope") or state.get("scope"),
        "expires_at": expires_at,
        "api_base_url": state.get("api_base_url"),
        "has_refresh_token": bool(refresh_token),
    }
