"""SelfHostedOIDCProvider — generic self-hosted OpenID Connect dashboard auth.

A standards-compliant OpenID Connect Relying Party for the ``hermes dashboard``
OAuth gate. Unlike the bundled ``nous`` provider (which encodes Nous Portal's
bespoke contract — ``agent:{instance_id}`` client ids, a custom access-token
JWT, the ``x-nous-refresh-token`` header, an ``oauth_contract_version`` claim),
this provider speaks **plain OIDC** so it works against any conformant
self-hosted identity provider:

    Authentik · Keycloak · Zitadel · Authelia · Auth0 · Okta · Google · …

It is a pure drop-in plugin: it implements the five
:class:`~hermes_cli.dashboard_auth.DashboardAuthProvider` methods and touches
nothing in core auth/runtime/login. The HTTP round trip, cookies, CSRF
``state`` check and ``redirect_uri`` reconstruction are all owned by
``hermes_cli/dashboard_auth/routes.py``; this provider only:

  1. discovers the IDP's endpoints from ``{issuer}/.well-known/openid-configuration``,
  2. builds the ``/authorize`` URL with PKCE (S256),
  3. exchanges the authorization code for tokens at the discovered
     ``token_endpoint``,
  4. verifies the **ID token** (RS256/ES256) against the discovered
     ``jwks_uri`` with ``iss`` / ``aud`` pinned to the configured issuer /
     client id, and maps standard OIDC claims (``sub``, ``email``, ``name``)
     onto a :class:`~hermes_cli.dashboard_auth.Session`.

Why the ID token (not the access token)? OIDC guarantees the ID token is a
signed JWT carrying identity claims — that is its entire purpose. The access
token's format is opaque to the client per the spec; many IDPs issue random
opaque strings the client cannot verify locally. Verifying the ID token is the
only choice that is universally correct across self-hosted IDPs. (The ``nous``
provider verifies its *access* token because Nous Portal mints a custom JWT
access token with the dashboard claims baked in — a non-OIDC shortcut.)

Public PKCE clients only. Confidential clients (with a ``client_secret``) are
not yet supported — see the ``# TODO(confidential-client)`` seam in
``complete_login`` / ``refresh_session``. Self-hosters configuring a CLI/SPA
client almost always register a public + PKCE client, which is the smaller,
simpler surface.

Configuration surfaces (env wins over config.yaml when set non-empty, so a
provisioned-but-not-populated secret can't shadow a valid config.yaml entry —
same precedence convention as the ``nous`` plugin)::

    # config.yaml — canonical surface
    dashboard:
      oauth:
        provider: self-hosted
        self_hosted:
          issuer: https://auth.example.com/application/o/hermes/   # required
          client_id: hermes-dashboard                              # required
          scopes: "openid profile email"                           # optional

    # Environment overrides (Docker/Fly secret injection)
    HERMES_DASHBOARD_OIDC_ISSUER
    HERMES_DASHBOARD_OIDC_CLIENT_ID
    HERMES_DASHBOARD_OIDC_SCOPES        # optional; defaults to "openid profile email"

Skip reasons: when the plugin loads but can't register (missing issuer /
client_id), it writes a human-readable reason to the module-level
:data:`LAST_SKIP_REASON` so the gate's fail-closed branch can surface a useful
operator error instead of the bare "no providers registered".
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import threading
import time
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
# Defaults / constants
# ---------------------------------------------------------------------------

# OIDC core scopes. ``openid`` is mandatory (without it the IDP won't issue an
# ID token); ``profile``/``email`` populate the Session's display_name/email.
_DEFAULT_SCOPES = "openid profile email"

# Signing algorithms we accept on the ID token. RS256 is the OIDC default;
# ES256 is common on modern self-hosted IDPs (Zitadel, newer Keycloak realms).
# HS256 is deliberately excluded — it implies a shared secret we don't have in
# the public-client model and is a well-known JWT confusion footgun.
_ALLOWED_ID_TOKEN_ALGS = ("RS256", "ES256", "RS384", "RS512", "ES384", "ES512")

# httpx timeouts.
_DISCOVERY_TIMEOUT_SEC = 10.0
_TOKEN_ENDPOINT_TIMEOUT_SEC = 10.0

# OIDC discovery is low-frequency and the document is effectively static;
# cache it for the process lifetime with a soft TTL so a long-running
# dashboard picks up an IDP endpoint migration within the hour.
_DISCOVERY_CACHE_TTL_SEC = 3600

# JWKS cache (PyJWKClient handles its own caching; this mirrors the nous
# provider's 5-minute lifespan so key rotation is picked up promptly).
_JWKS_CACHE_SECONDS = 300


# ---------------------------------------------------------------------------
# Skip-reason channel (mirrors the nous plugin)
# ---------------------------------------------------------------------------

LAST_SKIP_REASON: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64url_no_pad(raw: bytes) -> str:
    """Base64url-encode without ``=`` padding (RFC 7636 §4)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _require_https_or_loopback(url: str, *, field: str) -> str:
    """Reject an endpoint URL that isn't HTTPS (loopback http is allowed).

    OAuth credentials (codes, tokens) flow over these URLs. We require HTTPS
    for everything except an explicit loopback host so a misconfigured issuer
    can't ship the authorization code / refresh token in cleartext. Returns
    the URL unchanged on success; raises :class:`ProviderError` otherwise.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https":
        return url
    if parsed.scheme == "http" and (parsed.hostname or "") in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        return url
    raise ProviderError(
        f"OIDC {field} must be https:// (or http on localhost), got {url!r}"
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class SelfHostedOIDCProvider(DashboardAuthProvider):
    """Generic self-hosted OpenID Connect provider (authorization-code + PKCE)."""

    name = "self-hosted"
    display_name = "Self-Hosted OIDC"

    def __init__(
        self,
        *,
        issuer: str,
        client_id: str,
        scopes: str = _DEFAULT_SCOPES,
    ) -> None:
        if not issuer:
            raise ValueError("issuer is required")
        if not client_id:
            raise ValueError("client_id is required")
        # ``issuer`` is the OIDC issuer identifier. Normalise the trailing
        # slash for stable string compares (the ``iss`` claim must match the
        # issuer the IDP advertises in discovery — we pin against the
        # discovered value, not this normalised one, to be tolerant of a
        # trailing-slash mismatch between config and the IDP).
        self._issuer = issuer.rstrip("/")
        _require_https_or_loopback(self._issuer, field="issuer")
        self._client_id = client_id
        self._scopes = scopes.strip() or _DEFAULT_SCOPES

        # Discovery + JWKS are lazily resolved on first use so plugin
        # registration never makes a network call (the IDP may be down at
        # boot; the gate should still come up and fail per-request).
        self._discovery: Dict[str, Any] | None = None
        self._discovery_fetched_at: float = 0.0
        self._discovery_lock = threading.Lock()
        self._jwks_client: Any = None

    # ---- public API (DashboardAuthProvider) -------------------------------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        self._validate_redirect_uri(redirect_uri)
        disco = self._get_discovery()

        code_verifier = _b64url_no_pad(secrets.token_bytes(64))  # ~86 chars
        code_challenge = _b64url_no_pad(
            hashlib.sha256(code_verifier.encode("ascii")).digest()
        )
        state = _b64url_no_pad(secrets.token_bytes(32))

        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "scope": self._scopes,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        redirect_url = (
            f"{disco['authorization_endpoint']}?{urllib.parse.urlencode(params)}"
        )
        # Same flat ``state=…;verifier=…`` cookie shape every provider uses;
        # the auth-route layer prepends ``provider=`` and parses it back out.
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
        # ``state`` is verified by the auth-route layer before this call.
        _ = state
        disco = self._get_discovery()

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self._client_id,
            "code_verifier": code_verifier,
        }
        # TODO(confidential-client): when client_secret support lands, add it
        # here (and switch to HTTP Basic auth if the IDP's
        # token_endpoint_auth_methods_supported prefers client_secret_basic).
        return self._exchange(
            disco["token_endpoint"], data, bad_request_exc=InvalidCodeError
        )

    def refresh_session(self, *, refresh_token: str) -> Session:
        if not refresh_token:
            raise RefreshExpiredError("no refresh token present in session")
        disco = self._get_discovery()

        data = {
            "grant_type": "refresh_token",
            "client_id": self._client_id,
            "refresh_token": refresh_token,
            # Re-request the same scopes so the rotated ID token keeps the
            # identity claims (some IDPs narrow scope on refresh otherwise).
            "scope": self._scopes,
        }
        # TODO(confidential-client): add client_secret here when supported.
        return self._exchange(
            disco["token_endpoint"],
            data,
            bad_request_exc=RefreshExpiredError,
            previous_refresh_token=refresh_token,
        )

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # The session cookie stores the ID token in the access-token slot (see
        # ``_session_from_tokens``) precisely so this per-request check can
        # verify a real JWT. Returns None on expiry/invalidity (middleware
        # then refreshes or logs out); raises ProviderError if the IDP/JWKS is
        # unreachable.
        try:
            claims = self._verify_id_token(access_token)
        except InvalidCodeError:
            # Expired / invalid token — protocol says return None, not raise.
            return None
        except ProviderError:
            raise
        # No refresh token available on this path; "" is fine — the middleware
        # re-reads the refresh-token cookie separately for refresh_session.
        return self._session_from_tokens(
            id_token=access_token, refresh_token="", claims=claims
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        # Best-effort RFC 7009 revocation if the IDP advertised an endpoint.
        # Must never raise — logout is client-side cookie clearing regardless.
        if not refresh_token:
            return None
        try:
            disco = self._get_discovery()
        except ProviderError:
            return None
        endpoint = str(disco.get("revocation_endpoint") or "").strip()
        if not endpoint:
            return None
        try:
            httpx.post(
                endpoint,
                data={
                    "token": refresh_token,
                    "token_type_hint": "refresh_token",
                    "client_id": self._client_id,
                },
                headers={"Accept": "application/json"},
                timeout=_TOKEN_ENDPOINT_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug("self-hosted OIDC: revoke failed (ignored): %s", exc)
        return None

    # ---- internals: token exchange ----------------------------------------

    def _exchange(
        self,
        token_endpoint: str,
        data: Dict[str, str],
        *,
        bad_request_exc: type[Exception],
        previous_refresh_token: str = "",
    ) -> Session:
        """POST the token endpoint and turn the response into a Session.

        Shared by ``complete_login`` (auth-code grant) and ``refresh_session``
        (refresh grant). ``bad_request_exc`` is raised on a 400 —
        ``InvalidCodeError`` for the auth-code path, ``RefreshExpiredError``
        for the refresh path — preserving the middleware's distinct handling.
        """
        try:
            response = httpx.post(
                token_endpoint,
                data=data,
                headers={"Accept": "application/json"},
                timeout=_TOKEN_ENDPOINT_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            raise ProviderError(
                f"OIDC token endpoint unreachable: {exc}"
            ) from exc

        if response.status_code == 400:
            body = self._parse_json_body(response)
            error_code = body.get("error", "invalid_request")
            raise bad_request_exc(
                f"IDP rejected token request: {error_code}"
            )
        if response.status_code != 200:
            raise ProviderError(
                f"OIDC token endpoint returned {response.status_code}: "
                f"{response.text[:200]!r}"
            )

        payload = self._parse_json_body(response)

        id_token = payload.get("id_token")
        if not id_token or not isinstance(id_token, str):
            raise ProviderError(
                "OIDC token response missing id_token — ensure the 'openid' "
                "scope is configured and the client is allowed to receive an "
                "ID token."
            )

        token_type = str(payload.get("token_type", "")).lower()
        if token_type and token_type != "bearer":
            raise ProviderError(f"unexpected token_type={token_type!r}")

        claims = self._verify_id_token(id_token)

        # Refresh-token rotation: prefer a freshly-issued one, else keep the
        # previous (some IDPs don't rotate). Empty string if neither — the
        # session then behaves as ID-token-only until expiry.
        refresh_token = payload.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            refresh_token = previous_refresh_token or ""

        return self._session_from_tokens(
            id_token=id_token, refresh_token=refresh_token, claims=claims
        )

    # ---- internals: discovery ---------------------------------------------

    def _get_discovery(self) -> Dict[str, Any]:
        """Return the cached OIDC discovery document, fetching if stale."""
        now = time.time()
        if (
            self._discovery is not None
            and (now - self._discovery_fetched_at) < _DISCOVERY_CACHE_TTL_SEC
        ):
            return self._discovery
        with self._discovery_lock:
            now = time.time()
            if (
                self._discovery is not None
                and (now - self._discovery_fetched_at) < _DISCOVERY_CACHE_TTL_SEC
            ):
                return self._discovery
            disco = self._fetch_discovery()
            self._discovery = disco
            self._discovery_fetched_at = now
            # New issuer/keys → drop the JWKS client so it re-binds to the
            # freshly-discovered jwks_uri.
            self._jwks_client = None
            return disco

    def _discovery_url(self) -> str:
        # RFC 8414 / OIDC Discovery: ``{issuer}/.well-known/openid-configuration``.
        return f"{self._issuer}/.well-known/openid-configuration"

    def _fetch_discovery(self) -> Dict[str, Any]:
        url = self._discovery_url()
        try:
            # follow_redirects=True: many IDPs answer the discovery GET with a
            # 3xx rather than a direct 200 — Authentik canonicalises the
            # ``.well-known`` path, and any IDP behind a reverse proxy doing an
            # http→https upgrade redirects too. httpx (unlike curl -L or the
            # requests library) defaults to follow_redirects=False, so without
            # this the redirect comes back as a bare 3xx with an empty body and
            # the ``status != 200`` check below raises "discovery returned 302"
            # → provider_unreachable → 503. Following the redirect is safe: the
            # issuer-pin check and _require_https_or_loopback below still
            # validate the *resolved* document and every endpoint in it, so a
            # redirect to a hostile location can't smuggle in a bad issuer or a
            # cleartext endpoint. (The token/revocation POSTs deliberately do
            # NOT follow redirects — see _exchange — because they carry an auth
            # code / refresh token and the endpoint is already the canonical
            # absolute URL resolved here.)
            response = httpx.get(
                url,
                headers={"Accept": "application/json"},
                timeout=_DISCOVERY_TIMEOUT_SEC,
                follow_redirects=True,
            )
        except httpx.RequestError as exc:
            raise ProviderError(f"OIDC discovery unreachable: {exc}") from exc
        if response.status_code != 200:
            raise ProviderError(
                f"OIDC discovery returned {response.status_code} for {url!r}"
            )
        payload = self._parse_json_body(response)
        if not payload:
            raise ProviderError("OIDC discovery returned a non-JSON body")

        authorization_endpoint = str(
            payload.get("authorization_endpoint", "") or ""
        ).strip()
        token_endpoint = str(payload.get("token_endpoint", "") or "").strip()
        jwks_uri = str(payload.get("jwks_uri", "") or "").strip()
        if not authorization_endpoint or not token_endpoint or not jwks_uri:
            raise ProviderError(
                "OIDC discovery missing one of authorization_endpoint / "
                "token_endpoint / jwks_uri"
            )

        # Pin the discovered issuer: a mismatch between the configured issuer
        # and the ``issuer`` the IDP advertises means the discovery document
        # was served from the wrong place (proxy/MITM/misconfig). We tolerate
        # only a trailing-slash difference.
        advertised_issuer = str(payload.get("issuer", "") or "").strip()
        if advertised_issuer and advertised_issuer.rstrip("/") != self._issuer:
            raise ProviderError(
                f"OIDC discovery issuer mismatch: document advertises "
                f"{advertised_issuer!r} but configured issuer is "
                f"{self._issuer!r}"
            )

        _require_https_or_loopback(
            authorization_endpoint, field="authorization_endpoint"
        )
        _require_https_or_loopback(token_endpoint, field="token_endpoint")
        _require_https_or_loopback(jwks_uri, field="jwks_uri")

        revocation_endpoint = str(
            payload.get("revocation_endpoint", "") or ""
        ).strip()

        return {
            "issuer": advertised_issuer or self._issuer,
            "authorization_endpoint": authorization_endpoint,
            "token_endpoint": token_endpoint,
            "jwks_uri": jwks_uri,
            "revocation_endpoint": revocation_endpoint,
        }

    # ---- internals: JWT verification --------------------------------------

    def _get_jwks_client(self) -> Any:
        if self._jwks_client is None:
            from jwt import PyJWKClient  # lazy import

            disco = self._get_discovery()
            self._jwks_client = PyJWKClient(
                disco["jwks_uri"],
                cache_keys=True,
                lifespan=_JWKS_CACHE_SECONDS,
            )
        return self._jwks_client

    def _verify_id_token(self, id_token: str) -> Dict[str, Any]:
        import jwt  # lazy import — keeps startup fast for the ungated path

        disco = self._get_discovery()

        try:
            signing_key = self._get_jwks_client().get_signing_key_from_jwt(
                id_token
            )
        except jwt.PyJWKClientError as exc:
            raise ProviderError(f"JWKS lookup failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(f"JWKS lookup failed: {exc!r}") from exc

        try:
            claims = jwt.decode(
                id_token,
                signing_key.key,
                algorithms=list(_ALLOWED_ID_TOKEN_ALGS),
                audience=self._client_id,
                issuer=disco["issuer"],
                options={"require": ["exp", "iat", "aud", "iss", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            # verify_session() catches this and returns None per protocol.
            raise InvalidCodeError(f"ID token expired: {exc}") from exc
        except jwt.InvalidTokenError as exc:
            # Surface the actual iss/aud the token carried so operators can
            # debug config drift between the configured issuer/client_id and
            # what the IDP emits. Decoding-without-verification is safe here:
            # we already failed verification and never trust these values.
            details = ""
            try:
                unverified = jwt.decode(
                    id_token,
                    options={"verify_signature": False, "verify_exp": False},
                )
                details = (
                    f" [token iss={unverified.get('iss')!r} "
                    f"aud={unverified.get('aud')!r}; "
                    f"expected iss={disco['issuer']!r} "
                    f"aud={self._client_id!r}]"
                )
            except Exception:
                pass
            raise ProviderError(
                f"ID token verification failed: {exc}{details}"
            ) from exc

        return claims

    # ---- internals: mapping + misc ----------------------------------------

    def _session_from_tokens(
        self,
        *,
        id_token: str,
        refresh_token: str,
        claims: Dict[str, Any],
    ) -> Session:
        """Map verified OIDC claims onto a Session.

        The verified ID token is stored in ``Session.access_token`` so the
        per-request ``verify_session`` re-verifies a real JWT. The opaque
        OAuth access token is intentionally NOT stored — Hermes does not call
        any resource API with it; the dashboard only needs identity.
        """
        user_id = str(claims.get("sub", ""))
        if not user_id:
            raise ProviderError("ID token missing 'sub' (user_id) claim")

        email = str(claims.get("email", "") or "")
        # Standard OIDC display claims, in preference order.
        display_name = str(
            claims.get("name")
            or claims.get("preferred_username")
            or claims.get("nickname")
            or email
            or ""
        )
        # Org/tenant is non-standard; accept the common spellings. Groups, if
        # present as a list, are joined so multi-tenant IDPs surface *something*
        # rather than dropping the info — org_id is a free-form string.
        org_id = claims.get("org_id") or claims.get("organization") or ""
        if not org_id:
            groups = claims.get("groups")
            if isinstance(groups, list) and groups:
                org_id = ",".join(str(g) for g in groups)
        org_id = str(org_id or "")

        return Session(
            user_id=user_id,
            email=email,
            display_name=display_name,
            org_id=org_id,
            provider=self.name,
            expires_at=int(claims["exp"]),
            access_token=id_token,
            refresh_token=refresh_token,
        )

    def _validate_redirect_uri(self, redirect_uri: str) -> None:
        """Fast-fail obviously-broken redirect_uris before bouncing to the IDP.

        The IDP's own allowlist is authoritative; this just catches the common
        operator-error case with a clear message. Mirrors the nous provider.
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


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _load_config_oauth_section() -> dict:
    """Return the ``dashboard.oauth`` block from config.yaml, or ``{}``.

    Robust to load_config() raising, the ``dashboard`` key being absent or
    non-dict, and ``oauth`` being present but not a dict — each falls through
    to ``{}`` so callers can rely on ``.get(...)``.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        logger.debug(
            "dashboard-auth-self-hosted: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg_get(cfg, "dashboard", "oauth", default=None)
    return section if isinstance(section, dict) else {}


def _oidc_subsection(oauth_section: dict) -> dict:
    """Return the ``dashboard.oauth.self_hosted`` sub-block, or ``{}``."""
    sub = oauth_section.get("self_hosted")
    return sub if isinstance(sub, dict) else {}


def _resolve_setting(env_var: str, cfg_value: Any) -> str:
    """env-wins-config with empty-is-unset precedence.

    1. ``env_var`` when non-empty after strip (an empty provisioned secret
       must not shadow a valid config.yaml entry).
    2. ``cfg_value`` from config.yaml.
    3. Empty string.
    """
    env = os.environ.get(env_var, "").strip()
    if env:
        return env
    return str(cfg_value or "").strip()


def register(ctx) -> None:
    """Plugin entry — called by the plugin loader at startup.

    Registers :class:`SelfHostedOIDCProvider` only when both an issuer and a
    client_id are configured (via ``HERMES_DASHBOARD_OIDC_*`` env vars or the
    ``dashboard.oauth.self_hosted`` block in config.yaml). Operator-owned
    loopback / ``--insecure`` dashboards leave these unset, so the plugin is a
    no-op for them.

    On skip, writes a reason to :data:`LAST_SKIP_REASON` that names BOTH
    configuration surfaces so operators don't guess wrong about which to set.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    oauth_section = _load_config_oauth_section()
    oidc_cfg = _oidc_subsection(oauth_section)

    issuer = _resolve_setting(
        "HERMES_DASHBOARD_OIDC_ISSUER", oidc_cfg.get("issuer")
    )
    client_id = _resolve_setting(
        "HERMES_DASHBOARD_OIDC_CLIENT_ID", oidc_cfg.get("client_id")
    )
    scopes = (
        _resolve_setting("HERMES_DASHBOARD_OIDC_SCOPES", oidc_cfg.get("scopes"))
        or _DEFAULT_SCOPES
    )

    if not issuer or not client_id:
        LAST_SKIP_REASON = (
            "Self-hosted OIDC dashboard auth is not configured. Set both an "
            "issuer and a client_id — either as env vars "
            "(HERMES_DASHBOARD_OIDC_ISSUER + HERMES_DASHBOARD_OIDC_CLIENT_ID) "
            "or under dashboard.oauth.self_hosted.{issuer,client_id} in "
            "config.yaml — or pass --insecure to skip the OAuth gate "
            "entirely. (issuer set: %s; client_id set: %s)"
            % (bool(issuer), bool(client_id))
        )
        logger.debug("dashboard-auth-self-hosted: %s", LAST_SKIP_REASON)
        return

    try:
        provider = SelfHostedOIDCProvider(
            issuer=issuer, client_id=client_id, scopes=scopes
        )
    except (ValueError, ProviderError) as exc:
        LAST_SKIP_REASON = (
            f"SelfHostedOIDCProvider construction failed: {exc}"
        )
        logger.warning("dashboard-auth-self-hosted: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)
    logger.info(
        "dashboard-auth-self-hosted: registered provider "
        "(issuer=%s, client_id=%s, scopes=%r)",
        issuer,
        client_id,
        scopes,
    )
