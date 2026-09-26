"""Helpers shared by the bundled dashboard-auth providers.

Each provider module keeps its own ``logger`` / ``LAST_SKIP_REASON`` (the gate reads
those by module) and its ``register(ctx)``; the config/env resolution, skip/register
bookkeeping, PKCE login start, token-endpoint exchange and JWT verification live here.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import urllib.parse
from typing import Any, Callable, Dict, Optional

import httpx

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider, InvalidCodeError, LoginStart, ProviderError, RefreshExpiredError, Session,
    classify_jwks_lookup_error)

# JWKS Cache-Control max-age (nous contract C7); self-hosted mirrors it.
JWKS_CACHE_SECONDS = 300
TOKEN_ENDPOINT_TIMEOUT_SEC = 10.0
JSON_HEADERS = {"Accept": "application/json"}


# ---- Config / env resolution ----

def load_config_section(logger: logging.Logger, tag: str, *path: str) -> dict:
    """The ``config.yaml`` block at ``path`` as a dict, or ``{}`` — robust to load_config()
    raising (fresh install, malformed YAML), absent keys, or a non-dict value."""
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        logger.debug("%s: load_config() raised %s; falling back to env-only configuration", tag, exc)
        return {}
    section = cfg_get(cfg, *path, default=None)
    return section if isinstance(section, dict) else {}


def resolve_env_or_cfg(env_name: str, cfg_value: Any) -> str:
    """Env-wins-over-config; an empty env value is treated as unset so a
    provisioned-but-blank secret can't shadow a valid config.yaml entry."""
    return os.environ.get(env_name, "").strip() or str(cfg_value or "").strip()


# ---- register() bookkeeping ----

class SkipRegistration(Exception):
    """Raised by a provider's settings resolver to decline registration; ``reason`` is
    the operator-facing text stored in the module's ``LAST_SKIP_REASON``."""

    def __init__(self, reason: str, level: str = "debug") -> None:
        super().__init__(reason)
        self.reason, self.level = reason, level


def register_provider(
    ctx, logger: logging.Logger, tag: str, provider_cls: type, settings: Callable[[], dict],
) -> tuple[Optional[dict], str]:
    """Build ``provider_cls(**settings())`` and register it on ``ctx``.

    Returns ``(kwargs, "")`` on success, ``(None, skip_reason)`` when ``settings`` raised
    ``SkipRegistration`` (logged at its level) or construction raised ``ValueError`` /
    ``ProviderError`` (logged as a warning). Callers store the reason in ``LAST_SKIP_REASON``.
    """
    try:
        kwargs = settings()
        provider = provider_cls(**kwargs)
    except SkipRegistration as skip:
        getattr(logger, skip.level)("%s: %s", tag, skip.reason)
        return None, skip.reason
    except (ValueError, ProviderError) as exc:
        reason = f"{provider_cls.__name__} construction failed: {exc}"
        logger.warning("%s: %s", tag, reason)
        return None, reason
    ctx.register_dashboard_auth_provider(provider)
    return kwargs, ""


# ---- OAuth / PKCE ----

def b64url_no_pad(raw: bytes) -> str:
    """Base64url-encode without ``=`` padding (RFC 7636 §4)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def validate_redirect_uri(redirect_uri: str) -> None:
    """Fast-fail obviously-broken redirect_uris before bouncing to the IDP (whose allowlist
    is authoritative). Any ``http://`` host is allowed so dashboards behind TLS-terminating
    proxies / on LAN IPs aren't rejected."""
    parsed = urllib.parse.urlparse(redirect_uri)
    if parsed.scheme not in ("https", "http"):
        raise ProviderError(f"redirect_uri must be http(s), got {redirect_uri!r}")
    if not parsed.path or not parsed.path.endswith("/auth/callback"):
        raise ProviderError(f"redirect_uri path must end with '/auth/callback', got {redirect_uri!r}")


def pkce_login_start(authorize_url: str, *, client_id: str, scope: str, redirect_uri: str) -> LoginStart:
    """Build the authorization-code + PKCE (S256) redirect and cookie payload. Callers
    validate ``redirect_uri`` first. The auth-route layer expects
    ``cookie_payload["hermes_session_pkce"]`` as a flat ``state=…;verifier=…`` string
    (it prepends ``provider=``)."""
    code_verifier = b64url_no_pad(secrets.token_bytes(64))  # ~86 chars
    state = b64url_no_pad(secrets.token_bytes(32))
    params = {
        "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri, "scope": scope, "state": state,
        "code_challenge": b64url_no_pad(hashlib.sha256(code_verifier.encode("ascii")).digest()),
        "code_challenge_method": "S256"}
    return LoginStart(
        redirect_url=f"{authorize_url}?{urllib.parse.urlencode(params)}",
        cookie_payload={"hermes_session_pkce": f"state={state};verifier={code_verifier}"})


def parse_json_body(response: httpx.Response) -> Dict[str, Any]:
    """JSON object body, or ``{}`` for non-JSON content-type / parse error / non-dict."""
    if not response.headers.get("content-type", "").startswith("application/json"):
        return {}
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def exchange_token(
    url: str, data: Dict[str, str], *, headers: Optional[Dict[str, str]] = None, bad_request_exc: type[Exception],
    idp: str, endpoint: str, token_key: str, missing_msg: str) -> tuple[str, Dict[str, Any]]:
    """POST a token grant and return ``(token, payload)``.

    A 400 (OAuth-shaped error envelope) raises ``bad_request_exc`` — ``InvalidCodeError``
    for the auth-code path, ``RefreshExpiredError`` for refresh — so the middleware's
    distinct handling is preserved. Any other non-200, transport failure, missing
    ``token_key`` or non-bearer ``token_type`` raises ``ProviderError``. Redirects are
    deliberately NOT followed: the body carries an auth code / refresh token.
    """
    try:
        response = httpx.post(url, data=data, headers={**JSON_HEADERS, **(headers or {})}, timeout=TOKEN_ENDPOINT_TIMEOUT_SEC)
    except httpx.RequestError as exc:
        raise ProviderError(f"{endpoint} unreachable: {exc}") from exc
    if response.status_code == 400:
        error_code = parse_json_body(response).get("error", "invalid_request")
        raise bad_request_exc(f"{idp} rejected token request: {error_code}")
    if response.status_code != 200:
        raise ProviderError(f"{endpoint} returned {response.status_code}: {response.text[:200]!r}")
    payload = parse_json_body(response)
    token = payload.get(token_key)
    if not token or not isinstance(token, str):
        raise ProviderError(missing_msg)
    token_type = str(payload.get("token_type", "")).lower()
    if token_type and token_type != "bearer":
        raise ProviderError(f"unexpected token_type={token_type!r}")
    return token, payload


def refresh_token_from(payload: Dict[str, Any], fallback: str = "") -> str:
    """The token response's refresh token, or ``fallback`` when absent/non-string
    (the session then behaves as access-token-only until expiry)."""
    rt = payload.get("refresh_token")
    return rt if isinstance(rt, str) and rt else fallback


def session_from_claims(
    provider: str, claims: Dict[str, Any], *, access_token: str, refresh_token: str,
    label: str = "token", email: str = "", display_name: str = "", org_id: str = "") -> Session:
    """Map verified JWT claims onto a Session; ``sub`` is mandatory."""
    user_id = str(claims.get("sub", ""))
    if not user_id:
        raise ProviderError(f"{label} missing 'sub' (user_id) claim")
    return Session(
        user_id=user_id, email=email, display_name=display_name, org_id=org_id, provider=provider,
        expires_at=int(claims["exp"]), access_token=access_token, refresh_token=refresh_token)


# ---- JWT verification ----

def make_jwks_client(jwks_url: str) -> Any:
    """PyJWKClient with explicit Accept/User-Agent (some WAFs block the library default).
    Imported lazily so plugin discovery stays cheap."""
    from jwt import PyJWKClient

    return PyJWKClient(
        jwks_url, cache_keys=True, lifespan=JWKS_CACHE_SECONDS,
        headers={"Accept": "application/json", "User-Agent": "HermesAgent/1.0"})


def verify_jwt(
    token: str, jwks_client: Any, *, algorithms: list[str], audience: str, issuer: str, label: str) -> Dict[str, Any]:
    """Verify ``token`` against ``jwks_client`` with pinned ``aud``/``iss``.

    Unreachable JWKS → ``ProviderError`` (503); a bearer that is not one of our JWTs
    (opaque peer key, foreign kid) → ``InvalidCodeError`` (None / next provider); folding
    both into 503 broke peer-key bearers. Expiry raises ``InvalidCodeError`` (verify_session
    maps it to None); any other claim failure raises ``ProviderError`` with the unverified
    iss/aud appended so operators can spot config drift.
    """
    import jwt  # lazy — keeps startup fast for the ungated path

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
    except Exception as exc:
        # Unreachable JWKS -> ProviderError (503); a bearer that is not one of our JWTs (opaque peer key,
        # foreign kid) -> InvalidCodeError (None / next provider). Folding both into 503 produced #94558.
        # Unreachable JWKS -> ProviderError (503); a bearer that is not one of our JWTs (opaque peer key,
        # foreign kid) -> InvalidCodeError (None / next provider). Folding both into 503 produced #94558.
        raise classify_jwks_lookup_error(exc) from exc
    try:
        return jwt.decode(
            token, signing_key.key, algorithms=algorithms, audience=audience, issuer=issuer,
            options={"require": ["exp", "iat", "aud", "iss", "sub"]})
    except jwt.ExpiredSignatureError as exc:
        raise InvalidCodeError(f"{label} expired: {exc}") from exc
    except jwt.InvalidTokenError as exc:
        # Decoding without verification is safe here: verification already failed and
        # these values are surfaced for diagnostics only, never trusted.
        details = ""
        try:
            unverified = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
            details = (
                f" [token iss={unverified.get('iss')!r} aud={unverified.get('aud')!r}; "
                f"expected iss={issuer!r} aud={audience!r}]")
        except Exception:
            pass
        raise ProviderError(f"{label} verification failed: {exc}{details}") from exc


# ---- Shared provider skeletons ----

class NonInteractiveMixin:
    """OAuth-redirect stubs for providers without a browser login flow (password / service
    credential). ``_NOT_INTERACTIVE`` is the operator-facing reason; ``_NO_START_LOGIN``
    optionally overrides the ``start_login`` message."""

    _NOT_INTERACTIVE: str = ""
    _NO_START_LOGIN: str = ""

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise NotImplementedError(self._NO_START_LOGIN or self._NOT_INTERACTIVE)

    def complete_login(self, *, code: str, state: str, code_verifier: str, redirect_uri: str) -> Session:
        raise NotImplementedError(self._NOT_INTERACTIVE)


class JwtOAuthProvider(DashboardAuthProvider):
    """Authorization-code + PKCE provider whose session token is a JWT we verify ourselves
    (nous: Portal access token; self-hosted: OIDC ID token). Subclasses set ``_client_id`` and
    implement: ``_jwks_uri() -> str``; ``_claims_for(token) -> claims`` (raises
    ``InvalidCodeError`` on expiry/foreign token, ``ProviderError`` otherwise);
    ``_grant(data, *, bad_request_exc, headers=None, previous_refresh_token="") -> Session``;
    ``_refresh_request(refresh_token) -> (form_data, extra_headers)``;
    ``_session(token, refresh_token, claims) -> Session``."""

    _jwks_client: Any = None
    _client_id: str = ""

    def _get_jwks_client(self) -> Any:
        if self._jwks_client is None:
            self._jwks_client = make_jwks_client(self._jwks_uri())
        return self._jwks_client

    def complete_login(self, *, code: str, state: str, code_verifier: str, redirect_uri: str) -> Session:
        # ``state`` is verified by the auth-route layer before this call.
        return self._grant(
            {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
             "client_id": self._client_id, "code_verifier": code_verifier},
            bad_request_exc=InvalidCodeError)

    def refresh_session(self, *, refresh_token: str) -> Session:
        if not refresh_token:
            raise RefreshExpiredError("no refresh token present in session")
        data, headers = self._refresh_request(refresh_token)
        return self._grant(
            data, headers=headers, bad_request_exc=RefreshExpiredError, previous_refresh_token=refresh_token)

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # None on expiry/invalidity (middleware then tries refresh); a ProviderError
        # (JWKS unreachable) bubbles up so middleware emits 503.
        try:
            claims = self._claims_for(access_token)
        except InvalidCodeError:
            return None
        return self._session(access_token, "", claims)
