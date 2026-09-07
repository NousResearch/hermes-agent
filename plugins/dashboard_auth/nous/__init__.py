"""NousDashboardAuthProvider — Nous Portal OAuth (authorization-code + PKCE).

Implements ``nous-account-service/docs/agent-dashboard-oauth-contract.md``; registers only
when a client_id (``dashboard.oauth.client_id`` / ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``, shape
``agent:{instance_id}``) is configured. Access tokens are RS256 JWTs verified against the
Portal JWKS with ``aud`` = bare client_id. Portal issues a 24h *rotating* refresh token with
reuse detection: the middleware MUST persist ``Session.refresh_token`` back to the cookie on
every refresh or the next refresh replays a rotated token and revokes the whole session.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from hermes_cli.dashboard_auth import LoginStart, ProviderError, Session
from plugins.dashboard_auth._shared import (
    JwtOAuthProvider,
    SkipRegistration,
    exchange_token,
    load_config_section,
    pkce_login_start,
    refresh_token_from,
    register_provider,
    resolve_env_or_cfg,
    session_from_claims,
    validate_redirect_uri,
    verify_jwt)

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-nous"

_DEFAULT_PORTAL_URL = "https://portal.nousresearch.com"
_SCOPE = "agent_dashboard:access"  # contract C3
_EXPECTED_CONTRACT_VERSION = 1  # contract C11

LAST_SKIP_REASON: str = ""  # cleared on every register() so restarts don't leak stale reasons


class NousDashboardAuthProvider(JwtOAuthProvider):
    """Nous Portal OAuth via authorization-code + PKCE (S256)."""

    name = "nous"
    display_name = "Nous Research"

    def __init__(self, *, client_id: str, portal_url: str) -> None:
        # Defense-in-depth: register() filters too, but a malformed id must never construct a provider.
        if not client_id.startswith("agent:"):
            raise ValueError(f"client_id must match contract shape 'agent:{{instance_id}}', got {client_id!r}")
        self._client_id = client_id
        self._agent_instance_id = client_id[len("agent:") :]
        self._portal_url = portal_url.rstrip("/")
        self._jwks_url = f"{self._portal_url}/.well-known/jwks.json"
        self._authorize_url = f"{self._portal_url}/oauth/authorize"
        self._token_url = f"{self._portal_url}/api/oauth/token"
        self._jwks_client: Any = None  # lazily built (crypto import cost)

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        validate_redirect_uri(redirect_uri)
        return pkce_login_start(self._authorize_url, client_id=self._client_id, scope=_SCOPE, redirect_uri=redirect_uri)

    def revoke_session(self, *, refresh_token: str) -> None:
        # Portal exposes no token-endpoint revocation grant; logout is client-side cookie
        # clearing and the RT expires within its 24h TTL.
        return None

    # ---- JwtOAuthProvider hooks -------------------------------------------

    def _jwks_uri(self) -> str:
        return self._jwks_url

    def _refresh_request(self, refresh_token: str) -> tuple[Dict[str, str], Optional[Dict[str, str]]]:
        # The RT goes in BOTH the body (Portal's request schema requires it) and the
        # ``x-nous-refresh-token`` header (Portal reconciles the two and keeps the value
        # out of body access logs). Header-only → 400.
        return (
            {"grant_type": "refresh_token", "client_id": self._client_id, "refresh_token": refresh_token},
            {"x-nous-refresh-token": refresh_token})

    def _grant(
        self, data: Dict[str, str], *, bad_request_exc: type[Exception], headers: Optional[Dict[str, str]] = None,
        previous_refresh_token: str = "",
    ) -> Session:
        access_token, payload = exchange_token(
            self._token_url, data, headers=headers, bad_request_exc=bad_request_exc,
            idp="Portal", endpoint="Portal token endpoint", token_key="access_token",
            missing_msg="Portal token response missing access_token")
        # Rotating RT the caller MUST persist back to the cookie.
        return self._session(access_token, refresh_token_from(payload), self._claims_for(access_token))


    def _claims_for(self, access_token: str) -> Dict[str, Any]:
        claims = verify_jwt(
            access_token, self._get_jwks_client(), algorithms=["RS256"],
            audience=self._client_id,  # contract C2: bare client_id
            issuer=self._portal_url, label="access token")
        # Contract C9: agent_instance_id is "should" not "must" — tolerated when absent
        # (the aud check already binds the token to this instance).
        token_instance_id = claims.get("agent_instance_id")
        if token_instance_id is not None and token_instance_id != self._agent_instance_id:
            raise ProviderError(
                f"agent_instance_id mismatch: token={token_instance_id!r} vs configured={self._agent_instance_id!r}")
        contract_version = claims.get("oauth_contract_version")
        if contract_version is None:
            logger.warning(
                "Nous Portal token missing oauth_contract_version claim (contract says it should be %d); proceeding anyway.",
                _EXPECTED_CONTRACT_VERSION)
        elif contract_version != _EXPECTED_CONTRACT_VERSION:
            raise ProviderError(
                f"unsupported oauth_contract_version={contract_version!r}, expected {_EXPECTED_CONTRACT_VERSION}")
        return claims


    def _session(self, access_token: str, refresh_token: str, claims: Dict[str, Any]) -> Session:
        # Contract C4: no email / display_name in tokens.
        return session_from_claims(
            self.name, claims, access_token=access_token, refresh_token=refresh_token, org_id=str(claims.get("org_id") or ""))


# ---- Plugin entry point ----

def _load_config_oauth_section() -> dict:
    return load_config_section(logger, _TAG, "dashboard", "oauth")


def _settings() -> dict:
    """Resolve NousDashboardAuthProvider kwargs; the skip reason names BOTH configuration surfaces."""
    section = _load_config_oauth_section()
    client_id = resolve_env_or_cfg("HERMES_DASHBOARD_OAUTH_CLIENT_ID", section.get("client_id", ""))
    portal_url = resolve_env_or_cfg("HERMES_DASHBOARD_PORTAL_URL", section.get("portal_url", "")) or _DEFAULT_PORTAL_URL
    if not client_id:
        raise SkipRegistration(
            "HERMES_DASHBOARD_OAUTH_CLIENT_ID is not set (and dashboard.oauth.client_id "
            "in config.yaml is empty). The Nous Portal provisions this env var (shape "
            "'agent:{instance_id}') when it deploys a Hermes Agent instance — set it to "
            "your provisioned client id (either as an env var or under "
            "dashboard.oauth.client_id in config.yaml), or pass --insecure to skip the "
            "OAuth gate entirely.")
    if not client_id.startswith("agent:"):
        raise SkipRegistration(
            f"HERMES_DASHBOARD_OAUTH_CLIENT_ID={client_id!r} doesn't match the contract "
            f"shape 'agent:{{instance_id}}'. The Nous Portal provisions this value at deploy "
            f"time; check your Fly app's secrets or override with the value from the Portal admin UI.",
            level="warning")
    return {"client_id": client_id, "portal_url": portal_url}


def register(ctx) -> None:
    """Register ``NousDashboardAuthProvider`` when a client_id is configured."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(ctx, logger, _TAG, NousDashboardAuthProvider, _settings)
    if kwargs is not None:
        logger.info(
            "dashboard-auth-nous: registered provider (client_id=%s, portal=%s)", kwargs["client_id"], kwargs["portal_url"])


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import base64  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import httpx  # noqa: F401,E402
import os  # noqa: F401,E402
import secrets  # noqa: F401,E402
import urllib.parse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'DashboardAuthProvider': ('hermes_cli.dashboard_auth', 'DashboardAuthProvider'),
    'InvalidCodeError': ('hermes_cli.dashboard_auth', 'InvalidCodeError'),
    'RefreshExpiredError': ('hermes_cli.dashboard_auth', 'RefreshExpiredError'),
    'classify_jwks_lookup_error': ('hermes_cli.dashboard_auth', 'classify_jwks_lookup_error'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
