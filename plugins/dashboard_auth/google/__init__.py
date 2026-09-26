"""Google OAuth — dashboard auth provider for Hermes Agent.

Implements the ``DashboardAuthProvider`` interface backed by Google's
OpenID Connect / OAuth 2.0 authorization-code flow with PKCE (RFC 6749
§4.1, RFC 7636). Registers itself when ``HERMES_DASHBOARD_GOOGLE_CLIENT_ID``
and ``HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET`` are both set (either as env
vars or under ``dashboard.oauth.google_*`` in config.yaml).

Endpoints are resolved from Google's OpenID Connect discovery document
(``https://accounts.google.com/.well-known/openid-configuration``, cached
with a soft TTL) rather than hardcoded, so a Google-side endpoint migration
is picked up without a code change. The ID token returned by the token
endpoint is verified locally against Google's published JWKS (issuer +
audience pinned, signature checked) and stored as the session's access
token, so every ``verify_session()`` call re-verifies a real JWT instead of
making a network round trip to Google on every dashboard request — the same
approach the self-hosted OIDC provider uses.

Configuration
-------------
Env vars (highest precedence):
  HERMES_DASHBOARD_GOOGLE_CLIENT_ID       — Google OAuth client ID
  HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET   — Google OAuth client secret

config.yaml (fallback):
  dashboard:
    oauth:
      google_client_id: "..."
      google_client_secret: "..."

Optional allowlist (either env var; comma/whitespace-separated email list):
  DASHBOARD_ALLOWED_USERS   — dashboard-specific allowlist
  GATEWAY_ALLOWED_USERS     — shared with the API gateway's allowlist

An empty/missing allowlist means any authenticated Google account may sign
in (backward-compatible default). When set, ``complete_login`` rejects any
email not on the list with ``InvalidCodeError``.

No client_id/client_secret configured → plugin skips registration silently
(zero impact on loopback or --insecure users).

Usage
-----
1. Create an OAuth 2.0 Web Application client in the Google Cloud Console.
   - Authorized redirect URI: https://<your-dashboard>/auth/callback
2. Set the env vars above (or add them to config.yaml).
3. Restart the dashboard. Google will appear as a login option.

Standards referenced
---------------------
- RFC 6749 (OAuth 2.0 Authorization Framework) — authorization-code grant
  (§4.1), refresh-token grant (§6), client authentication (§2.3.1).
- RFC 7636 (PKCE) — code_verifier / code_challenge (S256).
- OpenID Connect Core 1.0 — ID token claims, discovery document shape.
- OpenID Connect Discovery 1.0 — {issuer}/.well-known/openid-configuration.
- Google Identity Platform docs — ``email_verified`` semantics, the
  optional ``hd`` (hosted domain) claim for Google Workspace accounts, and
  the ``prompt=select_account`` authorization parameter.
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
# Constants
# ---------------------------------------------------------------------------

_GOOGLE_DISCOVERY_URL = (
    "https://accounts.google.com/.well-known/openid-configuration"
)
_OAUTH_SCOPE = "openid email profile"

# Google's discovery document does not advertise a revocation_endpoint (it's
# absent from accounts.google.com's .well-known/openid-configuration), so —
# unlike the self-hosted OIDC provider, which reads one from discovery —
# this is a fixed, Google-documented endpoint.
_GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# Signing algorithms accepted on the ID token. Google currently only signs
# with RS256, but we accept the same breadth as the self-hosted OIDC
# provider for consistency. This doesn't weaken verification: the key is
# always looked up from Google's JWKS by 'kid', so a wider algorithm
# allowlist doesn't let an attacker supply their own verification key (the
# precondition for an "alg confusion" attack).
_ALLOWED_ID_TOKEN_ALGS = ("RS256", "ES256", "RS384", "RS512", "ES384", "ES512")

# httpx timeouts.
_DISCOVERY_TIMEOUT_SEC = 10.0
_TOKEN_ENDPOINT_TIMEOUT_SEC = 10.0

# OIDC discovery is low-frequency and the document is effectively static;
# cache it for the process lifetime with a soft TTL so a long-running
# dashboard picks up a Google-side endpoint migration within the hour.
_DISCOVERY_CACHE_TTL_SEC = 3600

# JWKS cache (PyJWKClient handles its own caching internally; this mirrors
# the sibling providers' 5-minute lifespan so key rotation is picked up
# promptly).
_JWKS_CACHE_SECONDS = 300

# Module-level skip reason — read by the dashboard's fail-closed branch
# when zero providers are registered, so the operator sees a useful
# message instead of a generic "install a provider" error.
LAST_SKIP_REASON: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64url_no_pad(raw: bytes) -> str:
    """Base64url-encode without '=' padding (RFC 7636 §4)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _require_https_or_loopback(url: str, *, field: str) -> str:
    """Reject an endpoint URL that isn't HTTPS (loopback http is allowed).

    Defense in depth: Google's discovery document is always served over
    HTTPS and always advertises HTTPS endpoints in practice, but pinning
    the scheme here means a corrupted or redirected discovery response
    can't silently downgrade the authorization/token/JWKS endpoints to
    plaintext.
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
    raise ProviderError(f"Google OIDC {field} must be https://, got {url!r}")


class _EmailNotVerifiedError(InvalidCodeError):
    """ID token has ``email_verified=False``.

    Google's own guidance: only trust the email address for identity
    purposes when ``email_verified`` is true. A false value means Google
    itself won't vouch for the address (e.g. an unmanaged Workspace account
    on an unverified custom domain) — accepting it anyway would let a
    caller authenticate under an email they don't control. Subclassing
    ``InvalidCodeError`` (rather than ``ProviderError``) means both
    ``complete_login`` and ``verify_session`` treat this the same way they
    treat an expired or otherwise invalid token: reject the credential
    rather than surface a 503.
    """


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GoogleDashboardAuthProvider(DashboardAuthProvider):
    """Google OAuth 2.0 / OpenID Connect via authorization-code + PKCE (S256)."""

    name = "google"
    display_name = "Google"

    def __init__(self, *, client_id: str, client_secret: str) -> None:
        if not client_id:
            raise ValueError("client_id is required")
        if not client_secret:
            raise ValueError("client_secret is required")
        self._client_id = client_id
        self._client_secret = client_secret

        # Discovery + JWKS are lazily resolved on first use so plugin
        # registration never makes a network call (Google may be
        # unreachable at boot; the gate should still come up and fail
        # per-request instead of blocking startup).
        self._discovery: Dict[str, Any] | None = None
        self._discovery_fetched_at: float = 0.0
        self._discovery_lock = threading.Lock()
        self._jwks_client: Any = None

    # ---- public API (DashboardAuthProvider) --------------------------------

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
            "scope": _OAUTH_SCOPE,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            # Google-specific: forces the account picker even if the user
            # is already signed in to exactly one Google account, so a
            # shared/kiosk browser doesn't silently reuse the wrong
            # identity.
            "prompt": "select_account",
            # Google omits refresh_token from the token response for
            # web-application clients unless the authorization request
            # carries access_type=offline. Without this, refresh_session()
            # never receives a refresh token to work with and every
            # session silently falls back to full re-login once the ID
            # token expires (~1h).
            "access_type": "offline",
        }
        redirect_url = (
            f"{disco['authorization_endpoint']}?{urllib.parse.urlencode(params)}"
        )
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
        _ = state  # verified by the auth-route layer before dispatch
        disco = self._get_discovery()

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "code_verifier": code_verifier,
        }
        session = self._exchange(
            disco["token_endpoint"], data, bad_request_exc=InvalidCodeError
        )

        # ── Allowlist check ──────────────────────────────────────────────
        # Dashboard inherits the gateway's allowlist convention:
        # DASHBOARD_ALLOWED_USERS (or GATEWAY_ALLOWED_USERS) controls who
        # can authenticate. An empty/missing allowlist means "any
        # authenticated user" (backward-compatible default).
        self._check_allowlist(session.email)
        # ─────────────────────────────────────────────────────────────────
        return session

    def refresh_session(self, *, refresh_token: str) -> Session:
        if not refresh_token:
            raise RefreshExpiredError("no refresh token in session")
        disco = self._get_discovery()

        data = {
            "grant_type": "refresh_token",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": refresh_token,
        }
        session = self._exchange(
            disco["token_endpoint"],
            data,
            bad_request_exc=RefreshExpiredError,
            previous_refresh_token=refresh_token,
        )
        # Sessions are stateless JWTs, so without this re-check a user
        # removed from the allowlist keeps a working dashboard until their
        # current ID token expires rather than being cut off at the next
        # refresh.
        self._check_allowlist(session.email, denied_exc=RefreshExpiredError)
        return session

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # The session cookie stores the ID token in the access-token slot
        # (see _session_from_tokens) precisely so this per-request check
        # can verify a real JWT locally instead of calling Google's
        # userinfo endpoint on every dashboard request.
        try:
            claims = self._verify_id_token(access_token)
        except InvalidCodeError:
            # Expired / invalid / email_verified=False — protocol says
            # return None, not raise (middleware then refreshes or logs
            # out).
            return None
        return self._session_from_tokens(
            id_token=access_token, refresh_token="", claims=claims
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        # Best-effort: POST to Google's revocation endpoint. Must never
        # raise — logout is client-side cookie clearing regardless.
        if not refresh_token:
            return None
        try:
            response = httpx.post(
                _GOOGLE_REVOKE_URL,
                data={"token": refresh_token},
                timeout=_TOKEN_ENDPOINT_TIMEOUT_SEC,
            )
            if response.status_code != 200:
                # Non-fatal (e.g. an already-expired token yields 400) —
                # logged so logout auditing can distinguish "revoked" from
                # "Google rejected the revoke call" without changing
                # behavior.
                logger.debug(
                    "dashboard-auth-google: revoke returned status_code=%s",
                    response.status_code,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(
                "dashboard-auth-google: revoke failed (ignored): %s", exc
            )
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

        Shared by ``complete_login`` (auth-code grant) and
        ``refresh_session`` (refresh grant). ``bad_request_exc`` is raised
        on a 400 — ``InvalidCodeError`` for the auth-code path,
        ``RefreshExpiredError`` for the refresh path.
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
                f"Google token endpoint unreachable: {exc}"
            ) from exc

        if response.status_code == 400:
            body = self._parse_json_body(response)
            error_code = body.get("error", "invalid_grant")
            raise bad_request_exc(f"Google rejected token request: {error_code}")
        if response.status_code != 200:
            raise ProviderError(
                f"Google token endpoint returned {response.status_code}: "
                f"{response.text[:200]!r}"
            )

        payload = self._parse_json_body(response)

        id_token = payload.get("id_token")
        if not id_token or not isinstance(id_token, str):
            raise ProviderError(
                "Google token response missing id_token — ensure the "
                "'openid' scope is included in the authorization request."
            )

        token_type = str(payload.get("token_type", "")).lower()
        if token_type and token_type != "bearer":
            raise ProviderError(f"unexpected token_type={token_type!r}")

        claims = self._verify_id_token(id_token)

        # Google rotates the refresh token only in rare cases; prefer a
        # freshly-issued one, else keep the previous.
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
                and (now - self._discovery_fetched_at)
                < _DISCOVERY_CACHE_TTL_SEC
            ):
                return self._discovery
            disco = self._fetch_discovery()
            self._discovery = disco
            self._discovery_fetched_at = now
            # New keys → drop the JWKS client so it re-binds to the
            # freshly-discovered jwks_uri.
            self._jwks_client = None
            return disco

    def _fetch_discovery(self) -> Dict[str, Any]:
        try:
            response = httpx.get(
                _GOOGLE_DISCOVERY_URL,
                headers={"Accept": "application/json"},
                timeout=_DISCOVERY_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            raise ProviderError(
                f"Google discovery endpoint unreachable: {exc}"
            ) from exc
        if response.status_code != 200:
            raise ProviderError(
                f"Google discovery returned {response.status_code}"
            )
        payload = self._parse_json_body(response)
        if not payload:
            raise ProviderError("Google discovery returned a non-JSON body")

        authorization_endpoint = str(
            payload.get("authorization_endpoint", "") or ""
        ).strip()
        token_endpoint = str(payload.get("token_endpoint", "") or "").strip()
        jwks_uri = str(payload.get("jwks_uri", "") or "").strip()
        issuer = str(payload.get("issuer", "") or "").strip()
        if not (authorization_endpoint and token_endpoint and jwks_uri and issuer):
            raise ProviderError(
                "Google discovery missing one of authorization_endpoint / "
                "token_endpoint / jwks_uri / issuer"
            )

        _require_https_or_loopback(
            authorization_endpoint, field="authorization_endpoint"
        )
        _require_https_or_loopback(token_endpoint, field="token_endpoint")
        _require_https_or_loopback(jwks_uri, field="jwks_uri")

        return {
            "issuer": issuer,
            "authorization_endpoint": authorization_endpoint,
            "token_endpoint": token_endpoint,
            "jwks_uri": jwks_uri,
        }

    # ---- internals: JWT verification --------------------------------------

    def _get_jwks_client(self) -> Any:
        if self._jwks_client is None:
            from jwt import PyJWKClient  # lazy import — keeps startup fast

            disco = self._get_discovery()
            self._jwks_client = PyJWKClient(
                disco["jwks_uri"],
                cache_keys=True,
                lifespan=_JWKS_CACHE_SECONDS,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "HermesAgent/1.0",
                },
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
            raise InvalidCodeError(f"id_token expired: {exc}") from exc
        except jwt.InvalidTokenError as exc:
            # Surface the actual iss/aud the token carried so operators can
            # debug config drift between the configured client and what
            # Google emits. Decoding without verification is safe here: we
            # already failed verification and never trust these values.
            details = ""
            try:
                unverified = jwt.decode(
                    id_token,
                    options={"verify_signature": False, "verify_exp": False},
                )
                details = (
                    f" [token iss={unverified.get('iss')!r} "
                    f"aud={unverified.get('aud')!r}; expected "
                    f"iss={disco['issuer']!r} aud={self._client_id!r}]"
                )
            except Exception:
                pass
            raise ProviderError(
                f"id_token verification failed: {exc}{details}"
            ) from exc

        # Google's own guidance: only trust the email address for identity
        # purposes when email_verified is true. Missing is tolerated
        # (some flows omit the claim entirely); explicitly false is not.
        email_verified = claims.get("email_verified")
        if email_verified is False:
            raise _EmailNotVerifiedError(
                f"Google id_token has email_verified=False for "
                f"{claims.get('email')!r}; refusing to trust it"
            )
        if email_verified is None:
            logger.debug("id_token missing email_verified claim; proceeding")

        return claims

    # ---- internals: mapping + misc ----------------------------------------

    def _session_from_tokens(
        self, *, id_token: str, refresh_token: str, claims: Dict[str, Any]
    ) -> Session:
        """Map verified ID-token claims onto a Session.

        The verified ID token is stored in ``Session.access_token`` so the
        per-request ``verify_session`` re-verifies a real JWT. The opaque
        OAuth access token is intentionally not stored — Hermes never
        calls a Google resource API with it; the dashboard only needs
        identity.
        """
        user_id = str(claims.get("sub", ""))
        if not user_id:
            raise ProviderError("id_token missing 'sub' (user_id) claim")

        email = str(claims.get("email", "") or "")
        display_name = str(claims.get("name") or email or "")
        # 'hd' (hosted domain) is Google's own claim identifying the
        # Google Workspace domain the account belongs to — the closest
        # analog to an org/tenant id for a Google-backed login. Absent for
        # personal @gmail.com accounts.
        org_id = str(claims.get("hd", "") or "")

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

    @staticmethod
    def _resolve_allowlist() -> list[str]:
        """Return the effective allowlist from env vars, falling back to
        config.yaml (dashboard.oauth.google_allowed_users) — mirrors the
        env-wins-over-config precedence used by ``_resolve_config``."""
        raw = os.environ.get("DASHBOARD_ALLOWED_USERS", "") or os.environ.get(
            "GATEWAY_ALLOWED_USERS", ""
        )
        if not raw.strip():
            try:
                from hermes_cli.config import cfg_get, load_config

                oauth = cfg_get(load_config(), "dashboard", "oauth", default=None)
                if isinstance(oauth, dict):
                    raw = str(oauth.get("google_allowed_users", "") or "")
            except Exception:
                raw = ""
        if not raw.strip():
            return []
        return [
            entry.strip()
            for entry in raw.replace(",", " ").split()
            if entry.strip()
        ]

    def _check_allowlist(
        self, email: str, *, denied_exc: type[Exception] = InvalidCodeError
    ) -> None:
        """Raise ``denied_exc`` if the allowlist is set and email is not on it."""
        allowed = self._resolve_allowlist()
        if not allowed:
            # Empty allowlist = any authenticated user (backward-compat)
            return
        if email.lower() in (a.lower() for a in allowed):
            return
        raise denied_exc(
            f"Access denied: {email} is not on the dashboard allowlist. "
            f"Set DASHBOARD_ALLOWED_USERS in .env to grant access."
        )

    def _validate_redirect_uri(self, redirect_uri: str) -> None:
        """Fast-fail on an obviously-broken redirect_uri before bouncing to
        Google. Google's own check is authoritative (redirect_uri_mismatch),
        but that error is opaque to the operator; this catches the common
        misconfiguration case (bad HERMES_DASHBOARD_PUBLIC_URL / proxy
        headers) with a clear message instead.
        """
        parsed = urllib.parse.urlparse(redirect_uri)
        if parsed.scheme not in ("https", "http"):
            raise ProviderError(
                f"redirect_uri must be http(s), got {redirect_uri!r}"
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


def _resolve_config() -> tuple[str, str]:
    """Return (client_id, client_secret) — env wins over config.yaml."""
    # Env vars (highest precedence — empty after strip = unset)
    env_id = os.environ.get("HERMES_DASHBOARD_GOOGLE_CLIENT_ID", "").strip()
    env_secret = os.environ.get(
        "HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET", ""
    ).strip()

    if env_id and env_secret:
        return env_id, env_secret

    # Fall back to config.yaml
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception:
        return env_id, env_secret  # config.yaml unreadable — keep env

    oauth = cfg_get(cfg, "dashboard", "oauth", default=None)
    if not isinstance(oauth, dict):
        return env_id, env_secret

    cfg_id = str(oauth.get("google_client_id", "")).strip()
    cfg_secret = str(oauth.get("google_client_secret", "")).strip()

    # Prefer env if set (partial env = partial fallback)
    final_id = env_id or cfg_id
    final_secret = env_secret or cfg_secret
    return final_id, final_secret


def register(ctx) -> None:
    """Plugin entry — called by the plugin loader at startup."""

    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    client_id, client_secret = _resolve_config()

    if not client_id:
        LAST_SKIP_REASON = (
            "Google OAuth dashboard auth provider not configured: "
            "set HERMES_DASHBOARD_GOOGLE_CLIENT_ID + "
            "HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET (env vars), or "
            "dashboard.oauth.google_client_id + "
            "dashboard.oauth.google_client_secret in config.yaml."
        )
        logger.debug("dashboard-auth-google: %s", LAST_SKIP_REASON)
        return

    if not client_secret:
        LAST_SKIP_REASON = (
            "Google OAuth client ID is set but client SECRET is "
            "missing. Set HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET "
            "or dashboard.oauth.google_client_secret."
        )
        logger.debug("dashboard-auth-google: %s", LAST_SKIP_REASON)
        return

    try:
        provider = GoogleDashboardAuthProvider(
            client_id=client_id, client_secret=client_secret
        )
    except ValueError as exc:
        LAST_SKIP_REASON = (
            f"GoogleDashboardAuthProvider construction failed: {exc}"
        )
        logger.warning("dashboard-auth-google: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)
    logger.info(
        "dashboard-auth-google: registered provider "
        "(client_id=%s…)", client_id[:16]
    )
