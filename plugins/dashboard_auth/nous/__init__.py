"""NousDashboardAuthProvider — Nous Portal OAuth (authorization-code + PKCE).

Implements ``nous-account-service/docs/agent-dashboard-oauth-contract.md``; registers only
when a client_id (``dashboard.oauth.client_id`` / ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``, shape
``agent:{instance_id}``) is configured. Access tokens are RS256 JWTs verified against the
Portal JWKS with ``aud`` = bare client_id. Portal issues a 24h *rotating* refresh token with
reuse detection: the middleware MUST persist ``Session.refresh_token`` back to the cookie on
every refresh or the next refresh replays a rotated token and revokes the whole session.

While the gateway serves the Webapp, its sign-in uses the client Web Access registered for the
Webapp instead of the configured one (``hermes_cli.dashboard_auth.listener_client``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import httpx

from hermes_cli.dashboard_auth import (
    InvalidCodeError, LoginStart, ProviderError, Session, TokenPrincipal, listener_client)
from plugins.dashboard_auth._shared import (
    DEFAULT_TOKEN_LEEWAY_SECONDS,
    JSON_HEADERS,
    JwtOAuthProvider,
    SkipRegistration,
    exchange_token,
    load_config_section,
    parse_json_body,
    parse_leeway,
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

# Human-readable identity enrichment (display/attribution only — never auth).
# The Portal has no OIDC discovery doc, but its account endpoint (the same call
# the CLI makes, ``hermes_cli/nous_account._fetch_nous_account_info``) answers
# with ``user.email`` for a Bearer access token. Enrichment is POST-verification:
# the JWT checks above stay the sole gate, and any enrichment failure — network,
# 401, unexpected shape — leaves the session exactly as it was before.
_ACCOUNT_PATH = "/api/oauth/account"
_ACCOUNT_TIMEOUT_SEC = 4.0
_IDENTITY_TTL_SECONDS = 900.0  # success cache, further bounded by the token's exp
_IDENTITY_FAILURE_TTL_SECONDS = 60.0  # a down/rejecting Portal is not hammered per request
_IDENTITY_CACHE_MAX = 256

LAST_SKIP_REASON: str = ""  # cleared on every register() so restarts don't leak stale reasons


class NousDashboardAuthProvider(JwtOAuthProvider):
    """Nous Portal OAuth via authorization-code + PKCE (S256)."""

    name = "nous"
    display_name = "Nous Research"
    supports_peer_pass = True

    def __init__(self, *, client_id: str, portal_url: str, token_leeway: float = DEFAULT_TOKEN_LEEWAY_SECONDS) -> None:
        # Defense-in-depth: register() filters too, but a malformed id must never construct a provider.
        if not client_id.startswith("agent:"):
            raise ValueError(f"client_id must match contract shape 'agent:{{instance_id}}', got {client_id!r}")
        self._client_id = client_id
        self._agent_instance_id = client_id[len("agent:") :]
        self._portal_url = portal_url.rstrip("/")
        # Clock-skew tolerance (seconds) for the access token's exp/nbf/iat claims;
        # the default absorbs Portal/host clock skew (#47815), 0 restores strict mode.
        self._token_leeway = parse_leeway(token_leeway)
        self._jwks_url = f"{self._portal_url}/.well-known/jwks.json"
        self._authorize_url = f"{self._portal_url}/oauth/authorize"
        self._token_url = f"{self._portal_url}/api/oauth/token"
        self._jwks_client: Any = None  # lazily built (crypto import cost)
        # jti -> (monotonic expiry, (display_name, email)). verify_session rebuilds a
        # Session on EVERY gated request; the cache keeps enrichment at one Portal
        # call per token per window instead of one per click.
        self._identity_cache: dict[str, tuple[float, tuple[str, str]]] = {}

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

    def _refresh_request(self, refresh_token: str) -> tuple[dict[str, str], Optional[dict[str, str]]]:
        # The RT goes in BOTH the body (Portal's request schema requires it) and the
        # ``x-nous-refresh-token`` header (Portal reconciles the two and keeps the value
        # out of body access logs). Header-only → 400.
        return (
            {"grant_type": "refresh_token", "client_id": self._client_id, "refresh_token": refresh_token},
            {"x-nous-refresh-token": refresh_token})

    def _grant(
        self, data: dict[str, str], *, bad_request_exc: type[Exception], headers: Optional[dict[str, str]] = None,
        previous_refresh_token: str = "",
    ) -> Session:
        access_token, payload = exchange_token(
            self._token_url, data, headers=headers, bad_request_exc=bad_request_exc,
            idp="Portal", endpoint="Portal token endpoint", token_key="access_token",
            missing_msg="Portal token response missing access_token")
        # Rotating RT the caller MUST persist back to the cookie.
        return self._session(access_token, refresh_token_from(payload), self._claims_for(access_token))


    def _claims_for(self, access_token: str) -> dict[str, Any]:
        # Every dashboard grant mints ``agent_dashboard:access``. A token for this agent without it,
        # a peer pass or one minted for another purpose with no ``scope`` claim, is no dashboard session.
        claims = self._agent_claims(access_token)
        if _SCOPE not in _scopes(claims):
            raise InvalidCodeError(f"token does not grant {_SCOPE}")
        return claims

    def verify_peer_pass(self, *, token: str) -> Optional[TokenPrincipal]:
        try:
            claims = self._agent_claims(token)
        except InvalidCodeError:
            return None
        user_id = str(claims.get("sub") or "")
        if not user_id:
            return None
        return TokenPrincipal(principal=user_id, provider=self.name, scopes=tuple(sorted(_scopes(claims))))

    def _agent_claims(self, access_token: str) -> dict[str, Any]:
        """Claims of a Portal access token minted for this agent, whatever it grants."""
        claims = verify_jwt(
            access_token, self._get_jwks_client(), algorithms=["RS256"],
            audience=self._client_id,  # contract C2: bare client_id
            issuer=self._portal_url, label="access token", leeway=self._token_leeway)
        # Contract C9: agent_instance_id is "should" not "must" — tolerated when absent
        # (the aud check already binds the token to this instance). A token for another
        # instance is not ours, like a foreign aud: a rejection, never a 503.
        token_instance_id = claims.get("agent_instance_id")
        if token_instance_id is not None and token_instance_id != self._agent_instance_id:
            raise InvalidCodeError(
                f"agent_instance_id mismatch: token={token_instance_id!r} vs configured={self._agent_instance_id!r}")
        # A contract version this agent does not support stays a 503: every token Portal
        # mints would fail it, so treating it as "not ours" would only log everyone out.
        contract_version = claims.get("oauth_contract_version")
        if contract_version is None:
            logger.warning(
                "Nous Portal token missing oauth_contract_version claim (contract says it should be %d); proceeding anyway.",
                _EXPECTED_CONTRACT_VERSION)
        elif contract_version != _EXPECTED_CONTRACT_VERSION:
            raise ProviderError(
                f"unsupported oauth_contract_version={contract_version!r}, expected {_EXPECTED_CONTRACT_VERSION}")
        return claims


    def _session(self, access_token: str, refresh_token: str, claims: dict[str, Any]) -> Session:
        # Contract C4: today's dashboard tokens carry no email / display_name, so the
        # author of a comment would fall back to the raw ``sub``. Claim mapping stays
        # forward-compatible (a contract extension that adds email/name is used with
        # no extra HTTP), and otherwise the Portal account endpoint supplies a
        # human-readable identity POST-VERIFICATION only: ``user_id`` stays the raw
        # ``sub``, and any enrichment failure degrades to exactly the pre-enrichment
        # session — login and comments never depend on it.
        display_name = str(claims.get("name") or claims.get("display_name") or "")
        email = str(claims.get("email") or "")
        if not (display_name or email):
            display_name, email = self._account_identity(access_token, claims)
        return session_from_claims(
            self.name, claims, access_token=access_token, refresh_token=refresh_token, email=email,
            display_name=display_name, org_id=str(claims.get("org_id") or ""))

    def _account_identity(self, access_token: str, claims: dict[str, Any]) -> tuple[str, str]:
        """``(display_name, email)`` from the Portal account endpoint, cached per ``jti``.

        Failures are swallowed to ``("", "")`` and cached briefly so a rejecting or
        unreachable Portal costs one retry per minute, not one per click. The success
        cache never outlives the access token itself.
        """
        key = str(claims.get("jti") or claims.get("sid") or "")
        now = time.monotonic()
        if key:
            hit = self._identity_cache.get(key)
            if hit is not None and hit[0] > now:
                return hit[1]
        identity: tuple[str, str] = ("", "")
        try:
            response = httpx.get(
                f"{self._portal_url}{_ACCOUNT_PATH}",
                headers={**JSON_HEADERS, "Authorization": f"Bearer {access_token}",
                         "User-Agent": "HermesAgent/1.0"},
                timeout=_ACCOUNT_TIMEOUT_SEC)
            if response.status_code == 200:
                user = parse_json_body(response).get("user")
                if isinstance(user, dict):
                    identity = (
                        str(user.get("name") or user.get("full_name") or user.get("display_name") or ""),
                        str(user.get("email") or ""))
            else:
                logger.debug("dashboard-auth-nous: account enrichment got %s; session keeps raw sub",
                             response.status_code)
        except Exception as exc:  # enrichment must never break a session
            logger.debug("dashboard-auth-nous: account enrichment unavailable (%s); session keeps raw sub", exc)
        if key:
            ttl = _IDENTITY_FAILURE_TTL_SECONDS if not any(identity) else _IDENTITY_TTL_SECONDS
            if any(identity):
                exp = claims.get("exp")
                if isinstance(exp, (int, float)):
                    ttl = max(1.0, min(ttl, float(exp) - time.time()))
            if len(self._identity_cache) >= _IDENTITY_CACHE_MAX:
                self._identity_cache.clear()
            self._identity_cache[key] = (time.monotonic() + ttl, identity)
        return identity


def _scopes(claims: dict[str, Any]) -> set[str]:
    raw = claims.get("scope")
    return set(raw.split()) if isinstance(raw, str) else set()


# ---- Plugin entry point ----

def _load_config_oauth_section() -> dict:
    return load_config_section(logger, _TAG, "dashboard", "oauth")


def _settings() -> dict:
    """Resolve NousDashboardAuthProvider kwargs; the skip reason names BOTH configuration surfaces."""
    section = _load_config_oauth_section()
    own = listener_client.current()  # a plugin reload while the listener uses its own client keeps it
    if own is not None:
        return {"client_id": own[0], "portal_url": own[1],
                "token_leeway": parse_leeway(section.get("token_leeway"))}
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
    return {"client_id": client_id, "portal_url": portal_url,
            # Clock-skew tolerance for access-token exp/nbf/iat (config.yaml only; default 60s, 0 = strict).
            "token_leeway": parse_leeway(section.get("token_leeway"))}


def register(ctx) -> None:
    """Register ``NousDashboardAuthProvider`` when a client_id is configured."""
    global LAST_SKIP_REASON
    listener_client.offer(_swap_provider)
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(ctx, logger, _TAG, NousDashboardAuthProvider, _settings)
    if kwargs is not None:
        logger.info(
            "dashboard-auth-nous: registered provider (client_id=%s, portal=%s, token_leeway=%ss)",
            kwargs["client_id"], kwargs["portal_url"], kwargs["token_leeway"])


def _swap_provider() -> Callable[[], None]:
    """Put a provider on the listener's own client (``listener_client.use`` has made it current) in
    place of the registered one; the returned undo puts that one, or none, back."""
    from hermes_cli.dashboard_auth.registry import (
        register_global_provider, restore_registration, snapshot_registration)

    previous = snapshot_registration(NousDashboardAuthProvider.name)
    provider = NousDashboardAuthProvider(**_settings())
    register_global_provider(provider)
    return lambda: restore_registration(NousDashboardAuthProvider.name, provider, previous)
