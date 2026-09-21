"""xAI Grok OAuth: token store, discovery, refresh, device-code login.

Split out of ``hermes_cli/auth.py``; origin helpers are imported lazily per function so
``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations

import logging
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import urlparse
from hermes_cli.auth_codex import _load_auth_store_maybe_locked, _refresh_payload_access_token
from hermes_cli.auth_constants import (
    AUTH_LOCK_TIMEOUT_SECONDS, AuthError, DEFAULT_XAI_OAUTH_BASE_URL, DEVICE_CODE_GRANT_TYPE,
    XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, XAI_OAUTH_CLIENT_ID, XAI_OAUTH_DEVICE_CODE_URL,
    XAI_OAUTH_DISCOVERY_URL, XAI_OAUTH_SCOPE, _FORM_JSON_HEADERS, _xai_err, httpx,
)
from utils import env_float

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig
logger = logging.getLogger("hermes_cli.auth")

_RELOGIN = "Re-authenticate with `hermes model`."


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _token_pair(tokens: Any) -> tuple[str, str]:
    """``(access_token, refresh_token)`` stripped; empty strings when *tokens* is not a dict."""
    if not isinstance(tokens, dict):
        return "", ""
    return _clean(tokens.get("access_token")), _clean(tokens.get("refresh_token"))


def _xai_oauth_state_from_store(auth_store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return usable xAI OAuth state from provider state or credential pool."""
    from hermes_cli.auth import _load_provider_state
    state = _load_provider_state(auth_store, "xai-oauth")
    if isinstance(state, dict) and all(_token_pair(state.get("tokens"))):
        return state

    credential_pool = auth_store.get("credential_pool")
    entries = credential_pool.get("xai-oauth") if isinstance(credential_pool, dict) else None
    for entry in entries if isinstance(entries, list) else ():
        access_token, refresh_token = _token_pair(entry)
        if not access_token or not refresh_token:
            continue
        merged = dict(state or {})
        merged["tokens"] = {
            "access_token": access_token, "refresh_token": refresh_token,
            "token_type": str(entry.get("token_type") or "Bearer"),
        }
        if entry.get("last_refresh"):
            merged["last_refresh"] = entry.get("last_refresh")
        merged.setdefault("auth_mode", "oauth_pkce")
        return merged

    return state if isinstance(state, dict) else None


def _xai_oauth_state_has_usable_tokens(state: Optional[Dict[str, Any]]) -> bool:
    return isinstance(state, dict) and all(_token_pair(state.get("tokens")))


def _read_xai_oauth_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    from hermes_cli.auth import _load_global_auth_store
    state = _xai_oauth_state_from_store(_load_auth_store_maybe_locked(_lock))
    if not _xai_oauth_state_has_usable_tokens(state):
        global_state = _xai_oauth_state_from_store(_load_global_auth_store())
        if _xai_oauth_state_has_usable_tokens(global_state):
            state = global_state
    if not state:
        raise _xai_err(
            "No xAI OAuth credentials stored. Select xAI Grok OAuth (SuperGrok / Premium+) in `hermes model`.",
            "xai_auth_missing", relogin=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise _xai_err(f"xAI OAuth state is missing tokens. {_RELOGIN}", "xai_auth_invalid_shape", relogin=True)
    access_token, refresh_token = _token_pair(tokens)
    for value, field in ((access_token, "access_token"), (refresh_token, "refresh_token")):
        if not value:
            raise _xai_err(
                f"xAI OAuth state is missing {field}. {_RELOGIN}", f"xai_auth_missing_{field}", relogin=True,
            )
    return {
        "tokens": tokens, "last_refresh": state.get("last_refresh"),
        "discovery": state.get("discovery") or {}, "redirect_uri": state.get("redirect_uri"),
    }


def _write_through_xai_oauth_to_global_root(state: Dict[str, Any]) -> None:
    """Best-effort persist of a rotated xAI grant into the global-root auth.json.

    xAI rotates refresh_token on every refresh, so a profile that refreshed a root-resolved grant
    must write the chain back to root. Touches only root ``providers.xai-oauth``; swallows all
    errors (root-stale is better than breaking the profile's own save).
    """
    from hermes_cli.auth import _global_auth_file_path, _persist_provider_state_to_store
    global_path = _global_auth_file_path()
    if global_path is None:  # classic mode (profile == root); the profile save already hit root
        return
    # Seat belt: under pytest never write the real ~/.hermes/auth.json (mirrors the read-side guard
    # in _load_global_auth_store). Uses raw HOME, not Path.home(), which fixtures may monkeypatch.
    real_home_env = os.environ.get("HOME", "") if os.environ.get("PYTEST_CURRENT_TEST") else ""
    if real_home_env:
        real_root = Path(real_home_env) / ".hermes" / "auth.json"
        try:
            if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                return
        except Exception:
            return
    try:
        _persist_provider_state_to_store("xai-oauth", state, global_path, set_active=False)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("xAI OAuth: write-through to global root failed: %s", exc)


def _save_xai_oauth_tokens(
    tokens: Dict[str, Any], *, discovery: Optional[Dict[str, Any]] = None, redirect_uri: str = "",
    last_refresh: Optional[str] = None, auth_mode: str = "oauth_device_code",
    set_active: bool = True,
) -> None:
    """Persist xAI OAuth tokens; *set_active* also promotes ``xai-oauth`` to ``active_provider``.

    Pass ``set_active=False`` for side-tool bootstrap (TTS/setup, tools config, dashboard, refresh)
    so inference routing is unchanged.
    """
    from hermes_cli.auth import _auth_store_lock, _global_auth_file_path, _load_auth_store, _load_provider_state_with_source, _same_path, _save_auth_store, _store_provider_state, _utc_now_z, _write_through_xai_oauth_to_global_root
    if last_refresh is None:
        last_refresh = _utc_now_z()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        # A profile lacking its own xai-oauth block reads root's grant via fallback; refreshing it
        # must write the rotated chain back to root or root keeps a revoked refresh token. Decide by
        # where the grant was resolved FROM (key presence lies: _store_provider_state creates it).
        state, source_path = _load_provider_state_with_source(auth_store, "xai-oauth")
        state = state if state is not None else {}
        state.update(tokens=tokens, last_refresh=last_refresh, auth_mode=auth_mode)
        if discovery:
            state["discovery"] = discovery
        if redirect_uri:
            state["redirect_uri"] = redirect_uri
        global_root = _global_auth_file_path()
        if source_path is not None and global_root is not None and _same_path(source_path, global_root):
            # Root-only write-back: a profile copy would shadow root and disable write-through.
            _write_through_xai_oauth_to_global_root(state)
        else:
            _store_provider_state(auth_store, "xai-oauth", state, set_active=set_active)
            _save_auth_store(auth_store)


def _xai_jwt_exp(access_token: Any) -> Optional[float]:
    """``exp`` claim of a JWT-shaped access token, or None when absent/undecodable."""
    if not isinstance(access_token, str) or "." not in access_token:
        return None
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")).get("exp")
        return float(exp) if isinstance(exp, (int, float)) else None
    except Exception:
        return None


def _xai_access_token_is_expiring(access_token: str, skew_seconds: int = 0) -> bool:
    exp = _xai_jwt_exp(access_token)
    return exp is not None and exp <= time.time() + max(0, int(skew_seconds))


def _xai_proactive_refresh_skew_seconds(access_token: str) -> int:
    """Proactive-refresh lead time before JWT ``exp``.

    Device-code logins often return ~15-minute JWTs; the full hour-long skew would refresh on every
    resolution, burning single-use refresh tokens and racing callers into ``invalid_grant``.
    """
    max_skew = XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    exp = _xai_jwt_exp(access_token)
    if exp is None:
        return max_skew
    remaining = exp - time.time()
    return min(120, max_skew) if 0 < remaining <= 45 * 60 else max_skew


def _is_xai_origin_host(host: str) -> bool:
    """``x.ai`` is the bare apex, so an exact match or any ``.x.ai`` suffix is accepted."""
    return host == "x.ai" or host.endswith(".x.ai")


def _xai_url_problem(url: str) -> tuple[Optional[str], str]:
    """``(problem, host)`` — problem is ``"scheme"``, ``"host"``, ``"origin"`` or None when *url* is HTTPS on x.ai."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        return "scheme", host
    if not host:
        return "host", host
    if not _is_xai_origin_host(host):
        return "origin", host
    return None, host


def _xai_validate_oauth_endpoint(url: str, *, field: str) -> str:
    """Refuse a discovery endpoint that isn't HTTPS on the xAI origin.

    Discovery is cached in auth.json, so one MITM at login could plant a ``token_endpoint`` that
    receives the refresh_token forever; pinning scheme + host (RFC 8414 §2) removes that.
    """
    problem, host = _xai_url_problem(url)
    if problem is None:
        return url
    message = {
        "scheme": f"xAI OIDC discovery returned a non-HTTPS {field}: {url!r}.",
        "host": f"xAI OIDC discovery {field} is missing a hostname: {url!r}.",
        "origin": (
            f"xAI OIDC discovery {field} host {host!r} is not on the xAI origin "
            f"(expected x.ai or a *.x.ai subdomain). Refusing to use a cached "
            f"endpoint that may have been substituted by a MITM during initial "
            f"discovery; re-authenticate with `hermes model` to re-fetch."
        ),
    }[problem]
    raise _xai_err(message, "xai_discovery_invalid")


def _xai_validate_inference_base_url(value: str, *, fallback: str) -> str:
    """Pin the OAuth inference base_url to ``*.x.ai``; warn and use *fallback* on rejection.

    Warn-not-raise: a bad env var must not deadlock auth, but the bearer must never leak elsewhere.
    """
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        return fallback
    try:
        problem, host = _xai_url_problem(candidate)
    except Exception:
        logger.warning("Ignoring malformed xAI base_url override %r; using %s instead.", candidate, fallback)
        return fallback
    if problem is None:
        return candidate
    if problem == "scheme":
        logger.warning(
            "Refusing non-HTTPS xAI base_url override %r (xai-oauth bearer would "
            "be sent in cleartext); falling back to %s.",
            candidate, fallback,
        )
    elif problem == "host":
        logger.warning("Ignoring xAI base_url override %r with no hostname; using %s instead.", candidate, fallback)
    else:
        logger.warning(
            "Refusing xAI base_url override %r — host %r is not on the xAI origin "
            "(expected x.ai or a *.x.ai subdomain). The xai-oauth bearer is only "
            "valid against xAI's inference API; sending it elsewhere would leak "
            "the credential. Falling back to %s.",
            candidate, host, fallback,
        )
    return fallback


def _xai_oauth_discovery(timeout_seconds: float = 15.0) -> Dict[str, str]:
    try:
        response = httpx.get(XAI_OAUTH_DISCOVERY_URL, headers={"Accept": "application/json"}, timeout=timeout_seconds)
    except Exception as exc:
        raise _xai_err(f"xAI OIDC discovery failed: {exc}", "xai_discovery_failed") from exc
    if response.status_code != 200:
        raise _xai_err(f"xAI OIDC discovery returned status {response.status_code}.", "xai_discovery_failed")
    try:
        payload = response.json()
    except Exception as exc:
        raise _xai_err(f"xAI OIDC discovery returned invalid JSON: {exc}", "xai_discovery_invalid_json") from exc
    if not isinstance(payload, dict):
        raise _xai_err("xAI OIDC discovery response was not a JSON object.", "xai_discovery_incomplete")
    endpoints = {k: _clean(payload.get(k)) for k in ("authorization_endpoint", "token_endpoint")}
    if not all(endpoints.values()):
        raise _xai_err("xAI OIDC discovery response was missing required endpoints.", "xai_discovery_incomplete")
    for field, url in endpoints.items():
        _xai_validate_oauth_endpoint(url, field=field)
    return endpoints


def _xai_tokens_from_payload(payload: Dict[str, Any], access_token: str, fallback_refresh: str) -> Dict[str, Any]:
    """Token block persisted for xAI OAuth; falls back to *fallback_refresh* when none is rotated in."""
    return {
        "access_token": access_token,
        "refresh_token": str(payload.get("refresh_token") or fallback_refresh).strip(),
        "id_token": _clean(payload.get("id_token")), "expires_in": payload.get("expires_in"),
        "token_type": _clean(payload.get("token_type") or "Bearer") or "Bearer",
    }


def refresh_xai_oauth_pure(
    access_token: str, refresh_token: str, *, token_endpoint: str = "",
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    from hermes_cli.auth import _nonempty_str, _utc_now_z, _xai_oauth_discovery
    del access_token
    if not _nonempty_str(refresh_token):
        raise _xai_err(
            f"xAI OAuth is missing refresh_token. {_RELOGIN}", "xai_auth_missing_refresh_token", relogin=True,
        )
    endpoint = token_endpoint.strip() or _xai_oauth_discovery(timeout_seconds)["token_endpoint"]
    # Re-validate cached endpoints: an old/hand-edited auth.json may carry a non-xAI token_endpoint
    # that would otherwise receive every future refresh_token.
    _xai_validate_oauth_endpoint(endpoint, field="token_endpoint")
    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            endpoint, headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "refresh_token", "client_id": XAI_OAUTH_CLIENT_ID, "refresh_token": refresh_token},
        )
    if response.status_code != 200:
        detail = response.text.strip()
        suffix = f" Response: {detail}" if detail else ""
        # 403 is almost always a tier/entitlement gate; re-login won't fix it, so use a separate
        # code and format_auth_error skips the re-authenticate hint.
        # ``403`` from xAI's token endpoint is almost always a tier / entitlement gate (the OAuth grant
        # exists but the account isn't on the allowlist for API access). Re-running ``hermes model`` won't
        # fix that — surface a separate error code so ``format_auth_error`` doesn't append a misleading
        # re-authenticate hint, and point users at the ``XAI_API_KEY`` fallback. See #26847.
        if response.status_code == 403:
            raise _xai_err(
                "xAI token refresh failed with HTTP 403." + suffix
                + " This OAuth account is not authorized for xAI API"
                  " access — xAI may be restricting API/OAuth use to"
                  " specific SuperGrok tiers despite the in-app"
                  " subscription being active. Re-logging in won't"
                  " change that; set ``XAI_API_KEY`` and switch to"
                  " ``provider: xai`` (API-key path) if available, or"
                  " upgrade your subscription at https://x.ai/grok.",
                "xai_oauth_tier_denied", relogin=False,
            )
        raise _xai_err(
            "xAI token refresh failed." + suffix, "xai_refresh_failed", relogin=response.status_code in {400, 401},
        )
    payload, refreshed_access = _refresh_payload_access_token(
        response, provider="xai-oauth",
        invalid_json=("xAI token refresh returned invalid JSON: {exc}", "xai_refresh_invalid_json"),
        invalid_json_relogin=False, strict_str=False,
        invalid_response=("xAI token refresh response was not a JSON object.", "xai_refresh_invalid_response"),
        missing_access=("xAI token refresh response was missing access_token.", "xai_refresh_missing_access_token"),
    )
    return {**_xai_tokens_from_payload(payload, refreshed_access, refresh_token), "last_refresh": _utc_now_z()}


def _refresh_xai_oauth_tokens(
    tokens: Dict[str, Any], *, token_endpoint: str, redirect_uri: str = "", timeout_seconds: float
) -> Dict[str, Any]:
    # Keep the stored auth_mode (legacy logins may carry ``oauth_pkce``): refresh must not relabel it.
    from hermes_cli.auth import _load_auth_store, _load_provider_state, refresh_xai_oauth_pure
    try:
        state = _load_provider_state(_load_auth_store(), "xai-oauth") or {}
        auth_mode = str(state.get("auth_mode") or "oauth_device_code")
    except Exception:
        auth_mode = "oauth_device_code"
    refreshed = refresh_xai_oauth_pure(
        _clean(tokens.get("access_token")), _clean(tokens.get("refresh_token")),
        token_endpoint=token_endpoint, timeout_seconds=timeout_seconds,
    )
    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = refreshed["access_token"]
    updated_tokens["refresh_token"] = refreshed["refresh_token"]
    if refreshed.get("id_token"):
        updated_tokens["id_token"] = refreshed["id_token"]
    if refreshed.get("expires_in") is not None:
        updated_tokens["expires_in"] = refreshed["expires_in"]
    if refreshed.get("token_type"):
        updated_tokens["token_type"] = refreshed["token_type"]
    # set_active=False: side tools (TTS) refresh xAI tokens while chat routes elsewhere.
    _save_xai_oauth_tokens(
        updated_tokens, discovery={"token_endpoint": token_endpoint}, redirect_uri=redirect_uri,
        last_refresh=refreshed["last_refresh"], auth_mode=auth_mode, set_active=False,
    )
    return updated_tokens


def _quarantine_xai_oauth_tokens(exc: AuthError) -> None:
    """Clear dead xAI tokens after a terminal (400/401/403) refresh failure so later sessions fail fast.

    Best-effort: persistence failures are logged and swallowed; the caller re-raises regardless.
    """
    from hermes_cli.auth import _last_auth_error_marker, _load_auth_store, _load_provider_state, _save_auth_store, _store_provider_state
    try:
        store = _load_auth_store()
        state = _load_provider_state(store, "xai-oauth") or {}
        tokens = dict(state.get("tokens") or {})
        tokens.pop("access_token", None)
        tokens.pop("refresh_token", None)
        # Capture the previous singleton tokens BEFORE overwriting them. The pool-sync step uses this to
        # distinguish legacy singleton-aliases (which should be refreshed) from independent accounts that
        # ``hermes auth add openai-codex`` created (which must not be overwritten — see #39236).
        state["tokens"] = tokens
        state["last_auth_error"] = _last_auth_error_marker(
            "xai-oauth", exc, reason="runtime_refresh_failure", default_code="xai_refresh_failed",
        )
        _store_provider_state(store, "xai-oauth", state, set_active=False)
        _save_auth_store(store)
    except Exception as save_exc:
        logger.debug("xAI OAuth: failed to persist quarantined state: %s", save_exc)


def _xai_oauth_inference_base_url() -> str:
    return _xai_validate_inference_base_url(
        os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/") or os.getenv("XAI_BASE_URL", "").strip().rstrip("/"),
        fallback=DEFAULT_XAI_OAUTH_BASE_URL,
    )


def resolve_xai_oauth_runtime_credentials(
    *, force_refresh: bool = False, refresh_if_expiring: bool = True,
    refresh_skew_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    from hermes_cli.auth import _auth_store_lock, _is_terminal_xai_oauth_refresh_error, _refresh_xai_oauth_tokens, _xai_oauth_discovery

    def _should_refresh(data: Dict[str, Any]) -> bool:
        access_token = _clean(data["tokens"].get("access_token"))
        skew = (
            int(refresh_skew_seconds) if refresh_skew_seconds is not None
            else _xai_proactive_refresh_skew_seconds(access_token)
        )
        return bool(force_refresh) or bool(
            refresh_if_expiring and _xai_access_token_is_expiring(access_token, skew)
        )

    data = _read_xai_oauth_tokens()
    tokens = dict(data["tokens"])
    refresh_timeout_seconds = env_float("HERMES_XAI_REFRESH_TIMEOUT_SECONDS", 20)
    if _should_refresh(data):
        with _auth_store_lock(timeout_seconds=max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)):
            # Re-read under the lock: a concurrent caller may already have rotated the grant.
            data = _read_xai_oauth_tokens(_lock=False)
            tokens = dict(data["tokens"])
            if _should_refresh(data):
                token_endpoint = (
                    _clean(dict(data.get("discovery") or {}).get("token_endpoint"))
                    or _xai_oauth_discovery(refresh_timeout_seconds)["token_endpoint"]
                )
                try:
                    tokens = _refresh_xai_oauth_tokens(
                        tokens, token_endpoint=token_endpoint, redirect_uri=_clean(data.get("redirect_uri")),
                        timeout_seconds=refresh_timeout_seconds,
                    )
                except AuthError as exc:
                    if _is_terminal_xai_oauth_refresh_error(exc):
                        _quarantine_xai_oauth_tokens(exc)
                    raise

    return {
        "provider": "xai-oauth",
        "base_url": _xai_oauth_inference_base_url(),
        "api_key": _clean(tokens.get("access_token")),
        "source": "hermes-auth-store",
        "last_refresh": data.get("last_refresh"),
        # Display only; auth.json may still carry a legacy ``oauth_pkce`` label.
        "auth_mode": "oauth_device_code",
    }


def _login_xai_oauth(args, pconfig: ProviderConfig, *, force_new_login: bool = False) -> None:
    from hermes_cli.auth import _is_remote_session, _offer_existing_oauth_credentials, _print_login_success, _update_config_for_provider, _xai_oauth_device_code_login, resolve_xai_oauth_runtime_credentials, unsuppress_credential_source
    del pconfig

    if not force_new_login and _offer_existing_oauth_credentials(
        "xai-oauth",
        resolve=resolve_xai_oauth_runtime_credentials,
        is_expiring=_xai_access_token_is_expiring,
        display_name="xAI OAuth",
        default_base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    ):
        return

    print()
    print("Signing in to xAI Grok OAuth (SuperGrok / Premium+)...")
    print("(Hermes creates its own local OAuth session)")
    print()

    timeout_seconds = float(getattr(args, "timeout", None) or 20.0)
    open_browser = not getattr(args, "no_browser", False)
    if _is_remote_session():
        open_browser = False

    creds = _xai_oauth_device_code_login(timeout_seconds=timeout_seconds, open_browser=open_browser)
    _save_xai_oauth_tokens(
        creds["tokens"], discovery=creds.get("discovery"),
        redirect_uri=creds.get("redirect_uri", ""), last_refresh=creds.get("last_refresh"),
        auth_mode="oauth_device_code",
    )
    # Explicit re-login re-enables the credential: clear the ``device_code`` suppression marker left
    # by ``hermes auth remove xai-oauth``. Deliberately NOT inside _save_xai_oauth_tokens — the
    # refresh hot path shares that helper and must never mutate suppression state.
    unsuppress_credential_source("xai-oauth", "device_code")
    config_path = _update_config_for_provider("xai-oauth", creds.get("base_url", DEFAULT_XAI_OAUTH_BASE_URL))
    _print_login_success("xai-oauth", config_path, show_auth_state=True)


def _xai_oauth_request_device_code(client: httpx.Client, *, scope: str = XAI_OAUTH_SCOPE) -> Dict[str, Any]:
    response = client.post(
        XAI_OAUTH_DEVICE_CODE_URL, headers=_FORM_JSON_HEADERS, data={"client_id": XAI_OAUTH_CLIENT_ID, "scope": scope},
    )
    if response.status_code != 200:
        raise _xai_err(
            f"xAI device-code request failed (HTTP {response.status_code})."
            + (f" Response: {response.text.strip()}" if response.text else ""),
            "device_code_request_failed",
        )
    payload = response.json()
    required = ("device_code", "user_code", "verification_uri", "verification_uri_complete", "expires_in", "interval")
    missing = [key for key in required if key not in payload]
    if missing:
        raise _xai_err(f"xAI device-code response missing fields: {', '.join(missing)}", "device_code_invalid")
    return payload


def _xai_oauth_poll_device_token(
    client: httpx.Client, *, token_endpoint: str, device_code: str, expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    from hermes_cli.auth import _poll_device_token_generic
    def _validate(payload: Dict[str, Any]) -> None:
        for field_name, article in (("access_token", "an"), ("refresh_token", "a")):
            if not payload.get(field_name):
                raise _xai_err(
                    f"xAI device-code token response did not include {article} {field_name}.",
                    "xai_device_token_invalid",
                )

    def _error(response, error_payload) -> Exception:
        description = error_payload.get("error_description") or error_payload.get("error") or response.text
        return _xai_err(f"xAI device-code token polling failed: {description}", "xai_device_token_failed")

    return _poll_device_token_generic(
        lambda: client.post(
            token_endpoint, headers=_FORM_JSON_HEADERS,
            data={"grant_type": DEVICE_CODE_GRANT_TYPE, "client_id": XAI_OAUTH_CLIENT_ID, "device_code": device_code},
        ),
        expires_in=int(expires_in),
        poll_interval=max(1, int(poll_interval)),
        validate_success=_validate,
        on_non_json_error=lambda _r: _xai_err(
            "xAI device-code token polling returned a non-JSON error response.", "xai_device_token_failed",
        ),
        on_error=_error,
        on_timeout=lambda: _xai_err("Timed out waiting for xAI device authorization.", "device_code_timeout"),
    )


def _xai_oauth_device_code_login(*, timeout_seconds: float = 20.0, open_browser: bool = True) -> Dict[str, Any]:
    from hermes_cli.auth import _can_open_graphical_browser, _is_remote_session, _print_device_code_instructions, _utc_now_z, _xai_oauth_discovery, _xai_oauth_poll_device_token
    discovery = _xai_oauth_discovery(timeout_seconds)
    timeout = httpx.Timeout(max(20.0, timeout_seconds))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        device_data = _xai_oauth_request_device_code(client)
        interval = int(device_data["interval"])
        _print_device_code_instructions(
            str(device_data.get("verification_uri_complete") or device_data["verification_uri"]),
            str(device_data["user_code"]),
            open_browser=open_browser and not _is_remote_session() and _can_open_graphical_browser(),
            swallow_open_errors=True,
        )
        print(f"Waiting for approval (polling every {max(1, interval)}s)...")
        payload = _xai_oauth_poll_device_token(
            client, token_endpoint=discovery["token_endpoint"],
            device_code=str(device_data["device_code"]), expires_in=int(device_data["expires_in"]),
            poll_interval=interval,
        )

    access_token, refresh_token = _token_pair(payload)
    if not access_token or not refresh_token:
        raise _xai_err("xAI device-code token response was missing required tokens.", "xai_device_token_invalid")
    return {
        "tokens": _xai_tokens_from_payload(payload, access_token, refresh_token),
        "discovery": discovery, "redirect_uri": "", "base_url": _xai_oauth_inference_base_url(),
        "last_refresh": _utc_now_z(), "source": "oauth-device-code",
    }
