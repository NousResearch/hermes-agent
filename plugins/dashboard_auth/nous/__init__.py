"""NousDashboardAuthProvider — Nous Portal OAuth (authorization-code + PKCE).

Implements ``nous-account-service/docs/agent-dashboard-oauth-contract.md``
(PR #180). The plugin auto-loads (bundled, kind=backend) but only registers
its provider when a client_id is configured — either via ``config.yaml`` or
via the Portal-injected env var — so loopback / ``--insecure`` operators
are unaffected.

Configuration surfaces (env wins over config.yaml when set non-empty):

  ``config.yaml`` — canonical surface::

      dashboard:
        oauth:
          client_id: agent:{agent_instance_id}   # required
          portal_url: https://portal.example     # optional

  Environment overrides — used by Fly.io's platform-secret injection so
  per-deploy values don't need to bake into ``config.yaml``:

      HERMES_DASHBOARD_OAUTH_CLIENT_ID  — shape ``agent:{agent_instance_id}``
      HERMES_DASHBOARD_PORTAL_URL       — defaults to
                                          ``https://portal.nousresearch.com``
                                          (production Portal). Override only
                                          for staging (``portal.rewbs.uk``)
                                          or a custom deployment.

Empty env var values are treated as unset so a provisioned-but-not-populated
Fly secret can't shadow a valid config.yaml entry.

Key contract points encoded here:

  - client_id is per-instance (``agent:{instance_id}``); the suffix is also
    cross-checked against the token's ``agent_instance_id`` claim as
    defense-in-depth.
  - scope is ``agent_dashboard:access`` only (no OIDC scopes).
  - tokens are RS256 JWTs verified against ``/.well-known/jwks.json``;
    JWKS is cached for 5 minutes.
  - V1 has NO refresh tokens — ``refresh_session`` always raises
    ``RefreshExpiredError`` so the middleware redirects to ``/auth/login``.
  - audience claim is the bare ``client_id`` (no ``hermes-cli:`` prefix).
  - tolerant ``oauth_contract_version`` check: missing → warn + proceed;
    present and ``!= 1`` → refuse.

The cookie payload returned by ``start_login`` stashes the PKCE
``code_verifier`` and the OAuth ``state`` parameter for the
``/auth/callback`` handler to retrieve. The auth-route layer is the owner
of cookie names; this provider just hands back ``{"code_verifier": …,
"state": …}`` and the route serializes those into the ``hermes_session_pkce``
cookie.

Forward compatibility: if a future Portal contract starts issuing refresh
tokens, ``complete_login`` already captures the value forward-compatibly
(populates ``Session.refresh_token``). Wiring the RT cookie back into the
middleware's near-expiry refresh path lives in the host application, not
here.

Skip reasons:
  The plugin exposes a module-level ``LAST_SKIP_REASON`` that the gate's
  fail-closed branch reads to surface a useful operator error message
  ("Set HERMES_DASHBOARD_OAUTH_CLIENT_ID …") instead of the bare "no
  providers registered" the gate would otherwise emit.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import urllib.parse
from typing import Any, Dict, Optional

import httpx

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    InvalidCodeError,
    LoginStart,
    ProviderError,
    RefreshExpiredError,
    Session,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Production Portal URL. Override via HERMES_DASHBOARD_PORTAL_URL for
# staging (portal.rewbs.uk) or a custom deployment. Contract docs name
# this as the production issuer.
_DEFAULT_PORTAL_URL = "https://portal.nousresearch.com"


# ---------------------------------------------------------------------------
# Skip-reason channel for operator-friendly error messages
# ---------------------------------------------------------------------------
#
# When the plugin loads but refuses to register (missing / malformed
# env vars), the auth gate downstream just sees "zero providers" and
# emits a generic "install a provider" error. That's misleading for the
# common case where the provider IS installed but mis-configured. The
# plugin writes the *specific* reason to this module-level slot; the
# gate reads it back when building its fail-closed SystemExit message.
#
# Cleared on every register() call so repeated dashboard starts in the
# same process (tests, hot-reload) don't leak stale reasons.

LAST_SKIP_REASON: str = ""


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

# Contract C3: scope name for the dashboard flow.
_SCOPE = "agent_dashboard:access"

# Contract C11: emitted claim should equal 1; tolerant (warn) if missing.
_EXPECTED_CONTRACT_VERSION = 1

# Contract C7: JWKS Cache-Control max-age=300.
_JWKS_CACHE_SECONDS = 300

# httpx timeout for the token endpoint POST.
_TOKEN_ENDPOINT_TIMEOUT_SEC = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64url_no_pad(raw: bytes) -> str:
    """Base64url-encode without ``=`` padding (RFC 7636 §4)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class NousDashboardAuthProvider(DashboardAuthProvider):
    """Nous Portal OAuth via authorization-code + PKCE (S256)."""

    name = "nous"
    display_name = "Nous Research"

    def __init__(self, *, client_id: str, portal_url: str) -> None:
        if not client_id.startswith("agent:"):
            # Defense-in-depth. The plugin entry point already filters, but
            # the provider should never be constructible with a malformed id.
            raise ValueError(
                "client_id must match contract shape 'agent:{instance_id}', "
                f"got {client_id!r}"
            )
        self._client_id = client_id
        self._agent_instance_id = client_id[len("agent:") :]
        self._portal_url = portal_url.rstrip("/")
        self._jwks_url = f"{self._portal_url}/.well-known/jwks.json"
        self._authorize_url = f"{self._portal_url}/oauth/authorize"
        self._token_url = f"{self._portal_url}/api/oauth/token"
        # PyJWKClient is lazily imported so plugin discovery doesn't pay the
        # crypto-import cost when the provider isn't activated.
        self._jwks_client: Any = None

    # ---- public API (DashboardAuthProvider) -------------------------------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        self._validate_redirect_uri(redirect_uri)

        code_verifier = _b64url_no_pad(secrets.token_bytes(64))  # ~86 chars
        code_challenge = _b64url_no_pad(
            hashlib.sha256(code_verifier.encode("ascii")).digest()
        )
        state = _b64url_no_pad(secrets.token_bytes(32))

        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "scope": _SCOPE,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        redirect_url = f"{self._authorize_url}?{urllib.parse.urlencode(params)}"
        # The auth-route layer expects ``cookie_payload[\"hermes_session_pkce\"]``
        # as a single semicolon-delimited string of ``key=value`` segments,
        # matching the stub provider's shape. The route handler prepends
        # ``provider=`` so the callback knows which plugin to dispatch to.
        cookie_payload = {
            "hermes_session_pkce": f"state={state};verifier={code_verifier}",
        }
        return LoginStart(redirect_url=redirect_url, cookie_payload=cookie_payload)

    def complete_login(
        self,
        *,
        code: str,
        state: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Session:
        # ``state`` is verified by the auth-route layer before this call
        # (it checks the cookie-stashed state matches the query-param state);
        # we just receive it for symmetry with the protocol. Nous Portal
        # doesn't re-check state at the token endpoint, so we ignore it here.
        _ = state

        try:
            response = httpx.post(
                self._token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self._client_id,
                    "code_verifier": code_verifier,
                },
                headers={"Accept": "application/json"},
                timeout=_TOKEN_ENDPOINT_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            raise ProviderError(f"Portal token endpoint unreachable: {exc}") from exc

        if response.status_code == 400:
            # Contract: invalid_code, invalid_grant, redirect_uri_mismatch all
            # surface as 400 with an OAuth-shaped JSON error envelope.
            body = self._parse_json_body(response)
            error_code = body.get("error", "invalid_request")
            raise InvalidCodeError(f"Portal rejected code: {error_code}")
        if response.status_code != 200:
            raise ProviderError(
                f"Portal token endpoint returned {response.status_code}: "
                f"{response.text[:200]!r}"
            )

        payload = self._parse_json_body(response)
        access_token = payload.get("access_token")
        if not access_token or not isinstance(access_token, str):
            raise ProviderError("Portal token response missing access_token")

        token_type = str(payload.get("token_type", "")).lower()
        if token_type and token_type != "bearer":
            raise ProviderError(f"unexpected token_type={token_type!r}")

        claims = self._verify_jwt(access_token)
        # Contract V1: no refresh token expected. If a future Portal ever
        # adds one, capture it forward-compatibly.
        refresh_token = payload.get("refresh_token") or ""
        if not isinstance(refresh_token, str):
            refresh_token = ""
        return self._session_from_claims(access_token, refresh_token, claims)

    def refresh_session(self, *, refresh_token: str) -> Session:
        # Contract V1 has no refresh tokens — always force re-auth. If a
        # future Portal contract starts issuing them, this method needs to
        # be re-implemented; until then it's an unconditional refusal.
        raise RefreshExpiredError(
            "Nous Portal does not issue refresh tokens in OAuth contract v1; "
            "user must re-authenticate via /auth/login."
        )

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # Contract: returns None on expiry/invalidity (middleware then
        # triggers redirect-to-login since refresh_session can never succeed
        # under V1); raises ProviderError if the IDP is unreachable.
        try:
            claims = self._verify_jwt(access_token)
        except InvalidCodeError:
            # Expired/invalid token — middleware contract is None, not raise.
            return None
        except ProviderError:
            # JWKS unreachable, etc. Bubble up so middleware emits 503.
            raise
        # verify_session has no access to the original refresh_token; pass
        # "" because in contract V1 there is none anyway.
        return self._session_from_claims(access_token, "", claims)

    def revoke_session(self, *, refresh_token: str) -> None:
        # Contract V1: no refresh tokens to revoke, and no Portal revocation
        # endpoint documented for dashboard tokens. Logout is purely
        # client-side cookie clearing; this is a best-effort no-op.
        _ = refresh_token
        return None

    # ---- internals --------------------------------------------------------

    def _validate_redirect_uri(self, redirect_uri: str) -> None:
        """Surface obviously-broken redirect_uris before bouncing to Portal.

        The Portal-side check (``agent-redirect-uri.ts``) is authoritative;
        this is a fast-fail for the common operator-error case.
        """
        parsed = urllib.parse.urlparse(redirect_uri)
        if parsed.scheme not in ("https", "http"):
            raise ProviderError(
                f"redirect_uri must be http(s), got {redirect_uri!r}"
            )
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost",
            "127.0.0.1",
        ):
            raise ProviderError(
                "redirect_uri may only use http:// for localhost/127.0.0.1, "
                f"got {redirect_uri!r}"
            )
        if not parsed.path or not parsed.path.endswith("/auth/callback"):
            raise ProviderError(
                "redirect_uri path must end with '/auth/callback', "
                f"got {redirect_uri!r}"
            )

    def _parse_json_body(self, response: httpx.Response) -> Dict[str, Any]:
        ctype = response.headers.get("content-type", "")
        if not ctype.startswith("application/json"):
            return {}
        try:
            body = response.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}

    def _get_jwks_client(self) -> Any:
        if self._jwks_client is None:
            from jwt import PyJWKClient  # lazy import

            self._jwks_client = PyJWKClient(
                self._jwks_url,
                cache_keys=True,
                lifespan=_JWKS_CACHE_SECONDS,
            )
        return self._jwks_client

    def _verify_jwt(self, access_token: str) -> Dict[str, Any]:
        # Lazy import — keeps startup fast for operators who never trigger
        # the gated path.
        import jwt

        try:
            signing_key = self._get_jwks_client().get_signing_key_from_jwt(
                access_token
            )
        except jwt.PyJWKClientError as exc:
            raise ProviderError(f"JWKS lookup failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(f"JWKS lookup failed: {exc!r}") from exc

        try:
            claims = jwt.decode(
                access_token,
                signing_key.key,
                algorithms=["RS256"],
                # Contract C2: aud is the bare client_id.
                audience=self._client_id,
                # Contract: issuer is the Portal base URL.
                issuer=self._portal_url,
                options={"require": ["exp", "iat", "aud", "iss", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            # verify_session() catches this and returns None per protocol.
            raise InvalidCodeError(f"access token expired: {exc}") from exc
        except jwt.InvalidTokenError as exc:
            # Surface the actual claim values that failed verification so
            # operators don't have to dig into the JWT to debug config drift
            # between HERMES_DASHBOARD_PORTAL_URL / HERMES_DASHBOARD_OAUTH_CLIENT_ID
            # and what Portal is actually emitting. Decoding without verification
            # is safe here: we've already failed to verify, and we never trust
            # these values — they're surfaced for diagnostics only.
            details = ""
            try:
                unverified = jwt.decode(
                    access_token,
                    options={"verify_signature": False, "verify_exp": False},
                )
                details = (
                    f" [token iss={unverified.get('iss')!r} "
                    f"aud={unverified.get('aud')!r}; "
                    f"expected iss={self._portal_url!r} "
                    f"aud={self._client_id!r}]"
                )
            except Exception:
                pass
            raise ProviderError(
                f"access token verification failed: {exc}{details}"
            ) from exc

        self._check_agent_instance_id(claims)
        self._check_contract_version(claims)
        return claims

    def _check_agent_instance_id(self, claims: Dict[str, Any]) -> None:
        """Contract C9: cross-check agent_instance_id against our config."""
        token_instance_id = claims.get("agent_instance_id")
        if token_instance_id is None:
            # Tolerated — the claim is documented as "should" not "must".
            # Our audience check on the bare client_id already binds the
            # token to this instance; agent_instance_id is defense-in-depth.
            return
        if token_instance_id != self._agent_instance_id:
            raise ProviderError(
                f"agent_instance_id mismatch: token={token_instance_id!r} "
                f"vs configured={self._agent_instance_id!r}"
            )

    def _check_contract_version(self, claims: Dict[str, Any]) -> None:
        """Contract C11 — tolerant treatment per OQ-C2."""
        contract_version = claims.get("oauth_contract_version")
        if contract_version is None:
            logger.warning(
                "Nous Portal token missing oauth_contract_version claim "
                "(contract says it should be %d); proceeding anyway.",
                _EXPECTED_CONTRACT_VERSION,
            )
            return
        if contract_version != _EXPECTED_CONTRACT_VERSION:
            raise ProviderError(
                f"unsupported oauth_contract_version={contract_version!r}, "
                f"expected {_EXPECTED_CONTRACT_VERSION}"
            )

    def _session_from_claims(
        self,
        access_token: str,
        refresh_token: str,
        claims: Dict[str, Any],
    ) -> Session:
        # Contract C4: no email / display_name in tokens. AuthWidget will
        # show user_id (truncated). Session fields kept for forward-compat.
        user_id = str(claims.get("sub", ""))
        if not user_id:
            raise ProviderError("token missing 'sub' (user_id) claim")
        return Session(
            user_id=user_id,
            email="",
            display_name="",
            org_id=str(claims.get("org_id") or ""),
            provider=self.name,
            expires_at=int(claims["exp"]),
            access_token=access_token,
            refresh_token=refresh_token,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _load_config_oauth_section() -> dict:
    """Return the ``dashboard.oauth`` block from ``config.yaml`` if it
    exists and is a dict; otherwise an empty dict.

    Robust to (a) load_config() raising (malformed YAML, IO error,
    config.yaml absent — common in fresh installs), (b) the
    ``dashboard`` key being absent or non-dict, and (c) the ``oauth``
    sub-key being present but not a dict (user typo). Each shape falls
    through to ``{}`` so register() can rely on `.get(...)` access.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        logger.debug(
            "dashboard-auth-nous: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg_get(cfg, "dashboard", "oauth", default=None)
    return section if isinstance(section, dict) else {}


def _resolve_client_id() -> str:
    """Resolve the OAuth client_id with env-overrides-config precedence.

    Order:
      1. ``HERMES_DASHBOARD_OAUTH_CLIENT_ID`` env var (when non-empty
         after strip — empty values are treated as unset so a
         provisioned-but-not-populated Fly secret can't shadow a valid
         config.yaml entry).
      2. ``dashboard.oauth.client_id`` in ``config.yaml``.
      3. Empty string — signals "no client_id configured" to the caller.
    """
    env = os.environ.get("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "").strip()
    if env:
        return env
    cfg_value = _load_config_oauth_section().get("client_id", "")
    return str(cfg_value).strip()


def _resolve_portal_url() -> str:
    """Resolve the Portal URL with env-overrides-config precedence.

    Order:
      1. ``HERMES_DASHBOARD_PORTAL_URL`` env var (non-empty after strip).
      2. ``dashboard.oauth.portal_url`` in ``config.yaml``.
      3. :data:`_DEFAULT_PORTAL_URL` (production Portal).
    """
    env = os.environ.get("HERMES_DASHBOARD_PORTAL_URL", "").strip()
    if env:
        return env
    cfg_value = str(
        _load_config_oauth_section().get("portal_url", "")
    ).strip()
    return cfg_value or _DEFAULT_PORTAL_URL


def register(ctx) -> None:
    """Plugin entry — called by the plugin loader at startup.

    Registers ``NousDashboardAuthProvider`` only when a client_id is
    configured (either via ``HERMES_DASHBOARD_OAUTH_CLIENT_ID`` env var
    or via ``dashboard.oauth.client_id`` in ``config.yaml``). The env
    var wins when set non-empty — Fly.io's platform-secret injection
    pushes the per-deploy value through this path.

    When skipping, writes a short human-readable reason to the module-
    level :data:`LAST_SKIP_REASON` so the dashboard's fail-closed branch
    can surface "Set HERMES_DASHBOARD_OAUTH_CLIENT_ID …" instead of the
    bare "no providers registered" the gate would otherwise emit. The
    reason mentions BOTH configuration surfaces so operators don't
    guess wrong about which one to populate.

    Operator-owned dashboards (loopback / ``--insecure``) leave both
    surfaces unset, so this plugin is a no-op for them. The gate-
    engagement layer (``hermes_cli.web_server.should_require_auth`` +
    the fail-closed check in ``start_server``) handles the "public bind
    with zero providers" case independently.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    client_id = _resolve_client_id()
    portal_url = _resolve_portal_url()

    if not client_id:
        LAST_SKIP_REASON = (
            "HERMES_DASHBOARD_OAUTH_CLIENT_ID is not set (and "
            "dashboard.oauth.client_id in config.yaml is empty). The "
            "Nous Portal provisions this env var (shape "
            "'agent:{instance_id}') when it deploys a Hermes Agent "
            "instance — set it to your provisioned client id (either "
            "as an env var or under dashboard.oauth.client_id in "
            "config.yaml), or pass --insecure to skip the OAuth gate "
            "entirely."
        )
        logger.debug("dashboard-auth-nous: %s", LAST_SKIP_REASON)
        return

    if not client_id.startswith("agent:"):
        LAST_SKIP_REASON = (
            f"HERMES_DASHBOARD_OAUTH_CLIENT_ID={client_id!r} doesn't match "
            f"the contract shape 'agent:{{instance_id}}'. The Nous Portal "
            f"provisions this value at deploy time; check your Fly app's "
            f"secrets or override with the value from the Portal admin UI."
        )
        logger.warning("dashboard-auth-nous: %s", LAST_SKIP_REASON)
        return

    try:
        provider = NousDashboardAuthProvider(
            client_id=client_id, portal_url=portal_url
        )
    except ValueError as exc:
        LAST_SKIP_REASON = f"NousDashboardAuthProvider construction failed: {exc}"
        logger.warning("dashboard-auth-nous: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)
    logger.info(
        "dashboard-auth-nous: registered provider (client_id=%s, portal=%s)",
        client_id,
        portal_url,
    )
