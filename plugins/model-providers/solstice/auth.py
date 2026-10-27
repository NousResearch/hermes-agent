"""Solstice sign-in: browser PKCE with a loopback redirect, token exchange and refresh brokered by the
Nous portal (NAS), which holds the confidential client secret. Hermes keeps its own PKCE verifier and
the user's tokens; NAS never sees a prompt.

The generic ``hermes_cli.auth_oauth_pkce_plugin`` flow runs the browser leg, the credential-pool row
and the single-use-safe refresh. This module supplies only what is NAS-specific: the client config it
discovers and the token POSTs it brokers. Every NAS call carries the user's Nous access token.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

DEFAULT_NAS_BASE_URL = "https://portal.nousresearch.com"
_CONFIG_PATH = "/api/oauth/gemini-auth/config"
_TOKEN_PATHS = {"authorization_code": "/api/oauth/gemini-auth/exchange", "refresh_token": "/api/oauth/gemini-auth/refresh"}
_TIMEOUT_SECONDS = 30.0
# Google token-endpoint codes NAS relays when a refresh token can never work again: revoked or expired
# (invalid_grant), or issued to an OAuth client NAS no longer uses (unauthorized_client).
_DEAD_GRANT_ERRORS = frozenset({"invalid_grant", "unauthorized_client"})


def _err(message: str, code: str, *, relogin: bool = False):
    from hermes_cli.auth_constants import AuthError
    return AuthError(f"solstice: {message}", provider="solstice", code=code, relogin_required=relogin)


def _nas_base_url() -> str:
    """The Nous portal the user is signed into (``HERMES_PORTAL_BASE_URL`` override first)."""
    from hermes_cli.auth import _nous_portal_base_url, get_provider_auth_state
    return _nous_portal_base_url(get_provider_auth_state("nous") or {})


def _nas_request(method: str, path: str, *, json: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from hermes_cli.auth import _default_verify, resolve_nous_access_token
    from hermes_cli.auth_constants import httpx

    try:
        token = resolve_nous_access_token()
    except Exception as exc:
        # Not a re-login of Solstice: its grant is intact, and marking it dead would outlive the Nous fix.
        raise _err("sign in to Nous Portal first (`hermes auth add nous`).", "solstice_nous_login_required") from exc
    try:
        response = httpx.request(method, _nas_base_url() + path, json=json, timeout=_TIMEOUT_SECONDS,
                                 headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
                                 verify=_default_verify())
    except Exception as exc:
        raise _err(f"Nous portal request failed: {type(exc).__name__}", "solstice_broker_unreachable") from exc
    if response.status_code == 404:
        raise _err("not available on this Nous account yet.", "solstice_not_configured")
    if response.status_code >= 400:
        error = _error_code(response)
        exc = _err(f"Nous portal returned HTTP {response.status_code}" + (f" ({error})." if error else "."),
                   "solstice_broker_failed")
        exc.broker_error = error
        raise exc
    return response.json()


def _error_code(response: Any) -> str:
    try:
        payload = response.json()
    except ValueError:  # a non-JSON error body (proxy page): no code to relay
        return ""
    return str(payload.get("error") or "").strip() if isinstance(payload, dict) else ""


def discover_client() -> dict[str, str]:
    """``client_id`` / ``authorize_url`` / ``scope`` NAS serves for the shared Google client."""
    payload = _nas_request("GET", _CONFIG_PATH)
    missing = [k for k in ("client_id", "authorize_url", "scope") if not str(payload.get(k) or "").strip()]
    if missing:
        raise _err(f"Nous portal config is missing {', '.join(missing)}.", "solstice_config_invalid")
    return {k: str(payload[k]).strip() for k in ("client_id", "authorize_url", "scope")}


def broker_token_request(grant: Mapping[str, str]) -> Mapping[str, Any]:
    """``OAuthPKCEConfig.token_request``: send the grant to NAS, which adds the client secret.

    A refresh is a dead sign-in only when NAS relays Google's verdict that the grant itself is dead
    (``_DEAD_GRANT_ERRORS``). Anything else (a Google or portal 5xx, the portal's rate limit, a Nous
    token the portal rejects) leaves the sign-in in place for the pool to retry after a cooldown.
    """
    path = _TOKEN_PATHS[grant["grant_type"]]
    body = {k: v for k, v in grant.items() if k in ("code", "code_verifier", "redirect_uri", "refresh_token")}
    try:
        return _nas_request("POST", path, json=body)
    except Exception as exc:
        if grant["grant_type"] == "refresh_token" and getattr(exc, "broker_error", "") in _DEAD_GRANT_ERRORS:
            raise _err(f"Google no longer accepts this sign-in ({exc.broker_error}); run `hermes auth add solstice` "
                       "again.", "invalid_grant", relogin=True) from exc
        raise
