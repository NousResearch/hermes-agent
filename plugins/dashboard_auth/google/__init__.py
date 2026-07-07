"""GoogleOAuthDashboardAuthProvider — Google OAuth 2.0 dashboard auth.

Implements the authorization-code + PKCE flow for the Hermes dashboard,
following the protocol defined in ``hermes_cli.dashboard_auth.DashboardAuthProvider``.
The provider uses the Google **access token** as the session handle and verifies
it through Google's ``tokeninfo`` endpoint, so transparent refresh works.

Configuration (env wins over config.yaml when set non-empty):

  Environment:
    HERMES_DASHBOARD_PUBLIC_URL
    HERMES_DASHBOARD_GOOGLE_CLIENT_ID
    HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET
    HERMES_DASHBOARD_GOOGLE_ALLOWED_EMAILS  # comma-separated whitelist

  config.yaml:
    dashboard:
      public_url: "https://hermes.example.com"
      oauth:
        google:
          client_id: "...apps.googleusercontent.com"
          client_secret: "..."
          allowed_emails: "user@example.com"  # optional; prefer env var

The plugin is a no-op when any required value is missing, so loopback / ``--insecure``
operators are unaffected.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
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

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"

_SCOPES = "openid email profile"
_TOKEN_TIMEOUT_SEC = 10.0
_TOKENINFO_CACHE_TTL_SEC = 300.0  # 5 minutes, matching Nous JWKS cache

LAST_SKIP_REASON: str = ""


def _b64url_no_pad(raw: bytes) -> str:
    """Base64url-encode without ``=`` padding (RFC 7636 §4)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class GoogleOAuthDashboardAuthProvider(DashboardAuthProvider):
    """Google OAuth 2.0 dashboard auth provider (authorization-code + PKCE)."""

    name = "google-oauth"
    display_name = "Google"

    supports_password: bool = False
    supports_token: bool = False
    supports_session: bool = True

    def __init__(
        self, *, client_id: str, client_secret: str, public_url: str
    ) -> None:
        if not client_id:
            raise ValueError("Google client_id is required")
        self._client_id = client_id
        self._client_secret = client_secret
        self._public_url = public_url.rstrip("/")
        self._tokeninfo_cache: Dict[str, Dict[str, Any]] = {}
        self._allowed_emails = _resolve_allowed_emails()
        if not self._allowed_emails:
            logger.warning(
                "google-oauth-auth-provider: No allowed emails configured. "
                "ANY Google account will be able to log in to the dashboard!"
            )

    # ---- public API (DashboardAuthProvider) -------------------------------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        self._validate_redirect_uri(redirect_uri)

        code_verifier = _b64url_no_pad(secrets.token_bytes(64))
        code_challenge = _b64url_no_pad(
            hashlib.sha256(code_verifier.encode("ascii")).digest()
        )
        state = _b64url_no_pad(secrets.token_bytes(32))

        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "scope": _SCOPES,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
        }

        redirect_url = f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
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
        _ = state
        tokens = self._exchange_code(
            code=code,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
        )
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token", "")
        if not access_token:
            raise InvalidCodeError("Google token response missing access_token")

        claims = self._verify_access_token(access_token)
        return self._claims_to_session(
            claims=claims,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        if not access_token:
            return None
        try:
            claims = self._verify_access_token(access_token)
            return self._claims_to_session(
                claims=claims,
                access_token=access_token,
                refresh_token="",
            )
        except InvalidCodeError:
            return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        if not refresh_token:
            raise RefreshExpiredError("no refresh token present in session")

        try:
            response = httpx.post(
                _GOOGLE_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": refresh_token,
                },
                headers={"Accept": "application/json"},
                timeout=_TOKEN_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            raise ProviderError(f"Google token endpoint unreachable: {exc}") from exc

        try:
            tokens = self._parse_token_response(response)
            new_access_token = tokens.get("access_token")
            if not new_access_token:
                raise RefreshExpiredError("Google refresh response missing access_token")
            claims = self._verify_access_token(new_access_token)
            return self._claims_to_session(
                claims=claims,
                access_token=new_access_token,
                refresh_token=tokens.get("refresh_token") or refresh_token,
            )
        except InvalidCodeError as exc:
            raise RefreshExpiredError(str(exc)) from exc

    def revoke_session(self, *, refresh_token: str) -> None:
        if not refresh_token:
            return
        try:
            httpx.post(
                _GOOGLE_REVOKE_URL,
                data={"token": refresh_token},
                timeout=_TOKEN_TIMEOUT_SEC,
            )
        except Exception:  # noqa: BLE001
            pass

    # ---- helpers -----------------------------------------------------------

    def _validate_redirect_uri(self, redirect_uri: str) -> None:
        if not redirect_uri.startswith("https://") and not redirect_uri.startswith("http://"):
            raise ProviderError(f"invalid redirect_uri: {redirect_uri!r}")

    def _exchange_code(
        self, *, code: str, code_verifier: str, redirect_uri: str
    ) -> Dict[str, Any]:
        try:
            response = httpx.post(
                _GOOGLE_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code_verifier": code_verifier,
                },
                headers={"Accept": "application/json"},
                timeout=_TOKEN_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            raise ProviderError(f"Google token endpoint unreachable: {exc}") from exc
        return self._parse_token_response(response)

    def _parse_token_response(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except Exception as exc:
            raise ProviderError(f"invalid JSON from Google token endpoint: {exc}") from exc

        if response.status_code == 200:
            return payload

        # 400 / 401 / 403 from token endpoint means the code/refresh token is bad.
        if response.status_code in (400, 401, 403):
            error = payload.get("error", "unknown")
            error_desc = payload.get("error_description", "")
            raise InvalidCodeError(f"Google token error: {error} {error_desc}")

        raise ProviderError(
            f"Google token endpoint returned HTTP {response.status_code}: {payload}"
        )

    def _verify_access_token(self, access_token: str) -> Dict[str, Any]:
        now = time.time()
        cached = self._tokeninfo_cache.get(access_token)
        if cached and now - cached["_cached_at"] < _TOKENINFO_CACHE_TTL_SEC:
            return cached["claims"]

        try:
            response = httpx.get(
                _GOOGLE_TOKENINFO_URL,
                params={"access_token": access_token},
                timeout=_TOKEN_TIMEOUT_SEC,
            )
        except httpx.RequestError as exc:
            raise ProviderError(f"Google tokeninfo endpoint unreachable: {exc}") from exc

        try:
            claims = response.json()
        except Exception as exc:
            raise ProviderError(f"invalid JSON from Google tokeninfo: {exc}") from exc

        if response.status_code in (400, 401, 403):
            raise InvalidCodeError(
                f"Google tokeninfo rejected token: {claims.get('error', '')}"
            )
        if response.status_code >= 500:
            raise ProviderError(
                f"Google tokeninfo unavailable: HTTP {response.status_code}"
            )
        if response.status_code != 200:
            raise ProviderError(
                f"Google tokeninfo returned HTTP {response.status_code}"
            )

        aud = claims.get("aud")
        if aud != self._client_id:
            raise InvalidCodeError(
                f"tokeninfo aud mismatch: expected {self._client_id}, got {aud}"
            )

        self._tokeninfo_cache[access_token] = {"claims": claims, "_cached_at": now}
        return claims

    def _claims_to_session(
        self,
        *,
        claims: Dict[str, Any],
        access_token: str,
        refresh_token: str,
    ) -> Session:
        user_id = str(claims.get("sub", ""))
        if not user_id:
            raise ProviderError("tokeninfo response missing 'sub' (user_id)")

        email = str(claims.get("email") or "")
        allowed = self._allowed_emails
        if allowed and email.lower() not in allowed:
            logger.warning(
                "google-oauth-auth-provider: login denied for email %s (not in allowlist)",
                email or "<empty>",
            )
            err_msg = (
                f"Google account {email} is not authorized to access this dashboard"
                if email
                else "Google account is not authorized to access this dashboard"
            )
            raise InvalidCodeError(err_msg)

        display_name = str(claims.get("name") or claims.get("given_name") or email)

        exp = claims.get("exp")
        if not exp:
            exp = int(time.time()) + 3600

        return Session(
            user_id=user_id,
            email=email,
            display_name=display_name,
            org_id="",
            provider=self.name,
            expires_at=int(exp),
            access_token=access_token,
            refresh_token=refresh_token,
        )


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _load_config_oauth_section() -> dict:
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "google-oauth-auth-provider: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg_get(cfg, "dashboard", "oauth", default=None)
    return section if isinstance(section, dict) else {}


def _resolve_client_id() -> str:
    env = os.environ.get("HERMES_DASHBOARD_GOOGLE_CLIENT_ID", "").strip()
    if env:
        return env
    cfg_value = _load_config_oauth_section().get("google", {})
    if isinstance(cfg_value, dict):
        return str(cfg_value.get("client_id", "")).strip()
    return ""


def _resolve_client_secret() -> str:
    env = os.environ.get("HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET", "").strip()
    if env:
        return env
    cfg_value = _load_config_oauth_section().get("google", {})
    if isinstance(cfg_value, dict):
        return str(cfg_value.get("client_secret", "")).strip()
    return ""


def _resolve_public_url() -> str:
    env = os.environ.get("HERMES_DASHBOARD_PUBLIC_URL", "").strip()
    if env:
        return env
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        value = cfg_get(cfg, "dashboard", "public_url", default="")
        if value:
            return str(value).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _resolve_allowed_emails() -> set[str]:
    """Return the set of allowed Google email addresses.

    The env var ``HERMES_DASHBOARD_GOOGLE_ALLOWED_EMAILS`` is the source of truth.
    If it is non-empty, it is split on commas and lower-cased.  An empty value
    means "allow all" (no whitelist) to preserve backwards compatibility for
    development and loopback deployments.
    """
    env = os.environ.get("HERMES_DASHBOARD_GOOGLE_ALLOWED_EMAILS", "").strip()
    if env:
        return {e.strip().lower() for e in env.split(",") if e.strip()}

    # Optional config.yaml fallback (also comma-separated or list)
    try:
        google_cfg = _load_config_oauth_section().get("google")
        if isinstance(google_cfg, dict):
            cfg_value = google_cfg.get("allowed_emails")
            if isinstance(cfg_value, str):
                return {e.strip().lower() for e in cfg_value.split(",") if e.strip()}
            if isinstance(cfg_value, list):
                return {e.strip().lower() for e in cfg_value if isinstance(e, str) and e.strip()}
    except Exception:  # noqa: BLE001
        pass

    return set()


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry — called by the Hermes plugin loader at startup.

    Registers ``GoogleOAuthDashboardAuthProvider`` only when the required
    Google OAuth credentials and a public callback URL are configured.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    client_id = _resolve_client_id()
    client_secret = _resolve_client_secret()
    public_url = _resolve_public_url()

    if not client_id:
        LAST_SKIP_REASON = (
            "HERMES_DASHBOARD_GOOGLE_CLIENT_ID is not set (and "
            "dashboard.oauth.google.client_id in config.yaml is empty). "
            "Create OAuth 2.0 credentials in Google Cloud Console and set "
            "the client id (either as an env var or under "
            "dashboard.oauth.google.client_id in config.yaml), or pass "
            "--insecure to skip the OAuth gate entirely."
        )
        logger.debug("google-oauth-auth-provider: %s", LAST_SKIP_REASON)
        return

    if not client_secret:
        LAST_SKIP_REASON = (
            "HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET is not set (and "
            "dashboard.oauth.google.client_secret in config.yaml is empty). "
            "Google OAuth for web server-side apps requires a client secret."
        )
        logger.warning("google-oauth-auth-provider: %s", LAST_SKIP_REASON)
        return

    if not public_url:
        LAST_SKIP_REASON = (
            "HERMES_DASHBOARD_PUBLIC_URL is not set and dashboard.public_url "
            "in config.yaml is empty. Google OAuth requires a stable public "
            "callback URL (<public_url>/auth/callback)."
        )
        logger.warning("google-oauth-auth-provider: %s", LAST_SKIP_REASON)
        return

    if not public_url.startswith("https://") and not public_url.startswith("http://"):
        LAST_SKIP_REASON = (
            f"HERMES_DASHBOARD_PUBLIC_URL={public_url!r} must start with "
            "https:// or http://."
        )
        logger.warning("google-oauth-auth-provider: %s", LAST_SKIP_REASON)
        return

    try:
        provider = GoogleOAuthDashboardAuthProvider(
            client_id=client_id,
            client_secret=client_secret,
            public_url=public_url,
        )
    except ValueError as exc:
        LAST_SKIP_REASON = f"GoogleOAuthDashboardAuthProvider construction failed: {exc}"
        logger.warning("google-oauth-auth-provider: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)
    logger.info(
        "google-oauth-auth-provider: registered provider (client_id=%s, public_url=%s)",
        client_id,
        public_url,
    )
