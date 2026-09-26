"""Meta Muse subscription OAuth helpers.

The subscription login uses Meta's device-code identity token to mint a short-lived
Model API key.  Secrets are kept in the normal Hermes auth store and are never
included in log messages.
"""

from __future__ import annotations

import time
from typing import Any, Optional
from urllib.parse import urlparse

from hermes_cli.auth_constants import (
    AuthError,
    DEVICE_CODE_GRANT_TYPE,
    META_DEVICE_AUTHORIZATION_URL,
    META_DEVICE_TOKEN_URL,
    META_OAUTH_CLIENT_ID,
    _FORM_JSON_HEADERS,
    httpx,
)


_RELOGIN = "Re-authenticate with `hermes model`."

META_OAUTH_PROVIDER_ID = "meta-oauth"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _meta_error(message: str, code: Optional[str] = None, *, relogin: bool = False) -> AuthError:
    return AuthError(message, provider="meta-oauth", code=code, relogin_required=relogin)


def _safe_json(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _meta_client() -> Any:
    return httpx.Client(timeout=30.0, headers={"Accept": "application/json"})


def start_device_authorization(client: Any = None) -> dict[str, Any]:
    """Request a Meta device code and validate its user-facing URL."""
    own_client = client is None
    client = client or _meta_client()
    try:
        response = client.post(
            META_DEVICE_AUTHORIZATION_URL,
            headers=_FORM_JSON_HEADERS,
            data={"client_id": META_OAUTH_CLIENT_ID},
        )
        payload = _safe_json(response)
        if response.status_code >= 400:
            raise _meta_error(
                f"Meta device authorization failed (HTTP {response.status_code}{_error_detail(payload)}).",
                "device_code_request_failed",
            )
        device_code = _clean(payload.get("device_code"))
        user_code = _clean(payload.get("user_code"))
        verification_uri = (
            _trusted_http_url(payload.get("verification_uri_complete"))
            or _trusted_http_url(payload.get("verification_uri"))
        )
        interval = _positive_number(payload.get("interval"))
        expires_in = _positive_number(payload.get("expires_in"))
        if not device_code or not user_code or not verification_uri or not interval or not expires_in:
            raise _meta_error("Meta device authorization response missing fields.", "device_code_invalid")
        return {
            "device_code": device_code,
            "user_code": user_code,
            "verification_uri": verification_uri,
            "interval": int(interval),
            "expires_in": int(expires_in),
        }
    finally:
        if own_client:
            client.close()


def poll_for_identity_token(
    client: Any, device_code: str, *, expires_in: int, poll_interval: int,
) -> str:
    """Poll Meta's device token endpoint and return the identity bearer token."""
    from hermes_cli.auth_device_flow import _poll_device_token_generic

    def _validate(payload: dict[str, Any]) -> None:
        if not _clean(payload.get("access_token")):
            raise _meta_error(
                "Meta device-code token response did not include an access_token.",
                "device_token_invalid", relogin=True,
            )

    def _error(response: Any, payload: dict[str, Any]) -> Exception:
        error = _clean(payload.get("error")) or _clean(response.text) or "unknown error"
        terminal = error in {"access_denied", "expired_token", "invalid_grant"}
        return _meta_error(
            f"Meta device-code token polling failed: {error}." + (f" {_RELOGIN}" if terminal else ""),
            "device_token_denied" if terminal else "device_token_failed",
            relogin=terminal,
        )

    result = _poll_device_token_generic(
        lambda: client.post(
            META_DEVICE_TOKEN_URL,
            headers=_FORM_JSON_HEADERS,
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "device_code": device_code,
                "client_id": META_OAUTH_CLIENT_ID,
            },
        ),
        expires_in=max(1, int(expires_in)),
        poll_interval=max(1, int(poll_interval)),
        validate_success=_validate,
        on_non_json_error=lambda _response: _meta_error(
            "Meta device-code token polling returned a non-JSON error response.",
            "device_token_failed",
        ),
        on_error=_error,
        on_timeout=lambda: _meta_error(
            "Timed out waiting for Meta device authorization.", "device_code_timeout",
        ),
    )
    return _clean(result.get("access_token"))


def mint_meta_api_key(identity_token: str, client: Any = None) -> dict[str, Any]:
    """Mint a 24-hour Meta Model API key from the device-flow identity token."""
    from hermes_cli.auth_constants import META_API_KEY_LIFETIME_MS, META_API_KEY_MINT_URL

    identity_token = _clean(identity_token)
    if not identity_token:
        raise _meta_error(f"Meta OAuth identity token is missing. {_RELOGIN}", "identity_token_missing", relogin=True)
    own_client = client is None
    client = client or _meta_client()
    try:
        response = client.post(
            META_API_KEY_MINT_URL,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {identity_token}",
                "Content-Type": "application/json",
                "x-api-version": "1.0.0",
            },
            content=b"{}",
        )
        payload = _safe_json(response)
        if response.status_code in {401, 403}:
            raise _meta_error(
                f"Meta session expired (HTTP {response.status_code}{_error_detail(payload)}). {_RELOGIN}",
                "meta_session_expired", relogin=True,
            )
        if response.status_code >= 400:
            raise _meta_error(
                f"Meta API key mint failed (HTTP {response.status_code}{_error_detail(payload)}).",
                "api_key_mint_failed",
            )
        key = _clean(payload.get("api_key"))
        if not key:
            action_url = _trusted_http_url(payload.get("action_url"))
            suffix = f" Complete setup at {action_url}." if action_url else ""
            raise _meta_error(
                f"Meta did not issue an API key.{suffix}", "api_key_missing",
            )
        return {
            "access_token": key,
            "refresh_token": identity_token,
            "expires_at_ms": int(time.time() * 1000) + META_API_KEY_LIFETIME_MS,
        }
    finally:
        if own_client:
            client.close()


def _trusted_http_url(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlparse(value)
    except Exception:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value


def _error_detail(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("error_description", "detail", "message", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return f": {value.strip()}"
    return ""


def _positive_number(value: Any) -> Optional[float]:
    return value if isinstance(value, (int, float)) and value > 0 else None


def _token_pair(tokens: Any) -> tuple[str, str]:
    if not isinstance(tokens, dict):
        return "", ""
    return _clean(tokens.get("access_token")), _clean(tokens.get("refresh_token"))


def _meta_oauth_state_from_store(auth_store: dict[str, Any]) -> Optional[dict[str, Any]]:
    from hermes_cli.auth import _load_provider_state

    state = _load_provider_state(auth_store, META_OAUTH_PROVIDER_ID)
    return state if isinstance(state, dict) else None


def _read_meta_oauth_tokens(*, _lock: bool = True) -> dict[str, Any]:
    from hermes_cli.auth_codex import _load_auth_store_maybe_locked

    state = _meta_oauth_state_from_store(_load_auth_store_maybe_locked(_lock))
    if not state:
        raise _meta_error(
            "No Meta OAuth credentials stored. Select Meta (Muse subscription) in `hermes model`.",
            "meta_auth_missing", relogin=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise _meta_error(
            f"Meta OAuth state is missing tokens. {_RELOGIN}",
            "meta_auth_invalid_shape", relogin=True,
        )
    access_token, refresh_token = _token_pair(tokens)
    for value, field in ((access_token, "access_token"), (refresh_token, "refresh_token")):
        if not value:
            raise _meta_error(
                f"Meta OAuth state is missing {field}. {_RELOGIN}",
                f"meta_auth_missing_{field}", relogin=True,
            )
    return {"tokens": tokens, "last_refresh": state.get("last_refresh")}


def _save_meta_oauth_tokens(
    tokens: dict[str, Any], *, last_refresh: Optional[str] = None,
    auth_mode: str = "oauth_device_code", set_active: bool = True,
) -> None:
    from hermes_cli.auth import (
        _auth_store_lock, _load_auth_store, _load_provider_state,
        _save_auth_store, _store_provider_state, _utc_now_z,
    )

    if last_refresh is None:
        last_refresh = _utc_now_z()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, META_OAUTH_PROVIDER_ID) or {}
        state.update(tokens=dict(tokens), last_refresh=last_refresh, auth_mode=auth_mode)
        _store_provider_state(auth_store, META_OAUTH_PROVIDER_ID, state, set_active=set_active)
        _save_auth_store(auth_store)


def _quarantine_meta_oauth_tokens(exc: AuthError) -> None:
    import logging
    from hermes_cli.auth import (
        _last_auth_error_marker, _load_auth_store, _load_provider_state,
        _save_auth_store, _store_provider_state,
    )

    logger = logging.getLogger("hermes_cli.auth")
    try:
        store = _load_auth_store()
        state = _load_provider_state(store, META_OAUTH_PROVIDER_ID) or {}
        tokens = dict(state.get("tokens") or {})
        tokens.pop("access_token", None)
        tokens.pop("refresh_token", None)
        state["tokens"] = tokens
        state["last_auth_error"] = _last_auth_error_marker(
            META_OAUTH_PROVIDER_ID, exc, reason="runtime_refresh_failure",
            default_code="meta_refresh_failed",
        )
        _store_provider_state(store, META_OAUTH_PROVIDER_ID, state, set_active=False)
        _save_auth_store(store)
    except Exception as save_exc:
        logger.debug("Meta OAuth: failed to persist quarantined state: %s", save_exc)


def _meta_oauth_key_is_expiring(tokens: dict[str, Any], skew_seconds: int = 0) -> bool:
    try:
        expires_ms = int(tokens.get("expires_at_ms") or 0)
    except Exception:
        return True
    if expires_ms <= 0:
        return True
    return expires_ms <= int(time.time() * 1000) + max(0, int(skew_seconds)) * 1000


def _meta_access_token_is_expiring(access_token: str, skew_seconds: int = 0) -> bool:
    if not _clean(access_token):
        return True
    try:
        data = _read_meta_oauth_tokens()
    except Exception:
        return True
    if _clean(data["tokens"].get("access_token")) != _clean(access_token):
        return False
    return _meta_oauth_key_is_expiring(data["tokens"], skew_seconds)


def _meta_oauth_device_code_login(
    *, timeout_seconds: float = 20.0, open_browser: bool = True,
) -> dict[str, Any]:
    from hermes_cli.auth_constants import httpx as _httpx
    from hermes_cli.auth_device_flow import _print_device_code_instructions
    from hermes_cli.auth import (
        _can_open_graphical_browser, _is_remote_session, _utc_now_z,
    )

    timeout = _httpx.Timeout(max(20.0, float(timeout_seconds or 20.0)))
    with _httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        device = start_device_authorization(client)
        _print_device_code_instructions(
            str(device["verification_uri"]), str(device["user_code"]),
            open_browser=open_browser and not _is_remote_session() and _can_open_graphical_browser(),
            swallow_open_errors=True,
        )
        print(f"Waiting for approval (polling every {max(1, int(device['interval']))}s)...")
        identity = poll_for_identity_token(
            client, str(device["device_code"]),
            expires_in=int(device["expires_in"]), poll_interval=int(device["interval"]),
        )
        minted = mint_meta_api_key(identity, client)
    return {
        "tokens": {
            "access_token": minted["access_token"], "refresh_token": minted["refresh_token"],
            "expires_at_ms": minted["expires_at_ms"],
        },
        "base_url": _meta_oauth_inference_base_url(),
        "last_refresh": _utc_now_z(), "source": "oauth-device-code",
    }


def _login_meta_oauth(args, pconfig=None, *, force_new_login: bool = False) -> None:
    from hermes_cli.auth import (
        _is_remote_session, _print_login_success, _update_config_for_provider,
        resolve_meta_oauth_runtime_credentials, unsuppress_credential_source,
    )
    from hermes_cli.auth_device_flow import _offer_existing_oauth_credentials
    from hermes_cli.auth_constants import DEFAULT_META_OAUTH_BASE_URL

    del pconfig
    if not force_new_login and _offer_existing_oauth_credentials(
        META_OAUTH_PROVIDER_ID,
        resolve=resolve_meta_oauth_runtime_credentials,
        is_expiring=_meta_access_token_is_expiring,
        display_name="Meta (Muse subscription)",
        default_base_url=DEFAULT_META_OAUTH_BASE_URL,
    ):
        return

    print()
    print("Signing in to Meta (Muse subscription)...")
    print("(Hermes creates its own local OAuth session)")
    print()

    timeout_seconds = float(getattr(args, "timeout", None) or 20.0)
    open_browser = not getattr(args, "no_browser", False)
    if _is_remote_session():
        open_browser = False

    creds = _meta_oauth_device_code_login(timeout_seconds=timeout_seconds, open_browser=open_browser)
    _save_meta_oauth_tokens(
        creds["tokens"], last_refresh=creds.get("last_refresh"), auth_mode="oauth_device_code",
    )
    unsuppress_credential_source(META_OAUTH_PROVIDER_ID, "device_code")
    config_path = _update_config_for_provider(
        META_OAUTH_PROVIDER_ID, creds.get("base_url", DEFAULT_META_OAUTH_BASE_URL))
    _print_login_success(META_OAUTH_PROVIDER_ID, config_path, show_auth_state=True)


def refresh_meta_oauth_pure(access_token: str, refresh_token: str, *, client: Any = None) -> dict[str, Any]:
    """Re-mint the daily Model API key from the stored identity token.

    Meta has no refresh_token grant (it 404s) — refresh always means re-mint.
    Signature matches the pool convention ``refresh_fn(access_token, refresh_token)``;
    only the identity (refresh) token is sent to the mint endpoint.
    """
    from hermes_cli.auth import _utc_now_z

    del access_token
    refresh_token = _clean(refresh_token)
    if not refresh_token:
        raise _meta_error(
            f"Meta OAuth is missing refresh_token. {_RELOGIN}",
            "meta_auth_missing_refresh_token", relogin=True,
        )
    minted = mint_meta_api_key(refresh_token, client)
    return {**minted, "last_refresh": _utc_now_z()}


def _meta_oauth_inference_base_url() -> str:
    import os

    from hermes_cli.auth_constants import DEFAULT_META_OAUTH_BASE_URL

    candidate = (
        os.getenv("HERMES_META_BASE_URL", "").strip().rstrip("/")
        or os.getenv("META_BASE_URL", "").strip().rstrip("/")
    )
    return candidate or DEFAULT_META_OAUTH_BASE_URL


def _is_terminal_meta_oauth_refresh_error(exc: Exception) -> bool:
    return (
        isinstance(exc, AuthError) and exc.provider == META_OAUTH_PROVIDER_ID
        and bool(exc.relogin_required)
    )


_META_ENV_KEY_VARS = ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY")


def _meta_env_api_key() -> str:
    """Explicit Meta API key from the environment (Meta's documented ``MODEL_API_KEY`` first).

    Explicit keys win over subscription state; a miss is silent so callers fall
    through to the pooled OAuth credential. Never logs the key.
    """
    from agent.secret_scope import UnscopedSecretError, get_secret_str

    for var in _META_ENV_KEY_VARS:
        try:
            token = _clean(get_secret_str(var, ""))
        except UnscopedSecretError:
            raise
        except Exception:
            token = ""
        if token:
            return token
    return ""


def _refresh_meta_oauth_tokens(tokens: dict[str, Any]) -> dict[str, Any]:
    from hermes_cli.auth import _load_auth_store, _load_provider_state

    try:
        state = _load_provider_state(_load_auth_store(), META_OAUTH_PROVIDER_ID) or {}
        auth_mode = str(state.get("auth_mode") or "oauth_device_code")
    except Exception:
        auth_mode = "oauth_device_code"
    refreshed = refresh_meta_oauth_pure(
        _clean(tokens.get("access_token")), _clean(tokens.get("refresh_token")))
    updated = dict(tokens)
    updated["access_token"] = refreshed["access_token"]
    updated["refresh_token"] = refreshed["refresh_token"]
    if refreshed.get("expires_at_ms") is not None:
        updated["expires_at_ms"] = refreshed["expires_at_ms"]
    _save_meta_oauth_tokens(
        updated, last_refresh=refreshed.get("last_refresh"),
        auth_mode=auth_mode, set_active=False,
    )
    return updated


def resolve_meta_oauth_runtime_credentials(
    *, force_refresh: bool = False, refresh_if_expiring: bool = True,
    refresh_skew_seconds: Optional[int] = None,
) -> dict[str, Any]:
    from hermes_cli.auth_constants import AUTH_LOCK_TIMEOUT_SECONDS, META_OAUTH_REFRESH_SKEW_SECONDS

    from hermes_cli.auth import _auth_store_lock

    skew = (
        int(refresh_skew_seconds) if refresh_skew_seconds is not None
        else META_OAUTH_REFRESH_SKEW_SECONDS
    )

    # Precedence: explicit env key wins, then the pooled subscription credential
    # from the auth store. No keychain read (Muse login stays the plugin's own
    # rule on the ``meta-ai`` provider); a miss raises re-login AuthError below.
    env_key = _meta_env_api_key()
    if env_key:
        return {
            "provider": META_OAUTH_PROVIDER_ID,
            "base_url": _meta_oauth_inference_base_url(),
            "api_key": env_key,
            "source": "env",
            "last_refresh": None,
            "auth_mode": "api_key",
        }

    def _should_refresh(data: dict[str, Any]) -> bool:
        return bool(force_refresh) or bool(
            refresh_if_expiring and _meta_oauth_key_is_expiring(data["tokens"], skew)
        )

    data = _read_meta_oauth_tokens()
    tokens = dict(data["tokens"])
    if _should_refresh(data):
        with _auth_store_lock(timeout_seconds=float(AUTH_LOCK_TIMEOUT_SECONDS)):
            data = _read_meta_oauth_tokens(_lock=False)
            tokens = dict(data["tokens"])
            if _should_refresh(data):
                try:
                    tokens = _refresh_meta_oauth_tokens(tokens)
                except AuthError as exc:
                    if _is_terminal_meta_oauth_refresh_error(exc):
                        _quarantine_meta_oauth_tokens(exc)
                    raise
    return {
        "provider": META_OAUTH_PROVIDER_ID,
        "base_url": _meta_oauth_inference_base_url(),
        "api_key": _clean(tokens.get("access_token")),
        "source": "hermes-auth-store",
        "last_refresh": data.get("last_refresh"),
        "auth_mode": "oauth_device_code",
    }


def get_meta_oauth_auth_status() -> dict[str, Any]:
    from hermes_cli.auth import _auth_file_path

    try:
        creds = resolve_meta_oauth_runtime_credentials(refresh_if_expiring=False)
        return {
            "logged_in": True, "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"), "source": creds.get("source"),
            "api_key": creds.get("api_key"),
        }
    except AuthError as exc:
        return {"logged_in": False, "auth_store": str(_auth_file_path()), "error": str(exc)}
