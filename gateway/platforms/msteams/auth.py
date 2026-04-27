"""Bot Framework auth helpers for the Microsoft Teams adapter.

This first Teams slice intentionally keeps auth small and dependency-light:
inbound activities are validated against Microsoft's Bot Framework OpenID
metadata with PyJWT, and outbound replies use the OAuth2 client-credentials
flow through httpx.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx
import jwt

logger = logging.getLogger(__name__)

BOT_FRAMEWORK_SCOPE = "https://api.botframework.com/.default"
BOT_FRAMEWORK_OPENID_METADATA_URL = (
    "https://login.botframework.com/v1/.well-known/openidconfiguration"
)
BOT_FRAMEWORK_VALID_ISSUERS = (
    "https://api.botframework.com",
    "https://api.botframework.com/",
)
DEFAULT_AUTH_CACHE_TTL_SECONDS = 3600
_TOKEN_REFRESH_LEEWAY_SECONDS = 300


class AuthError(Exception):
    """Raised when Bot Framework auth setup or token acquisition fails."""


class BotFrameworkJWTValidator:
    """Validate Bot Framework bearer tokens on inbound activities."""

    def __init__(
        self,
        app_id: str,
        http_client: httpx.AsyncClient,
        *,
        metadata_url: str = BOT_FRAMEWORK_OPENID_METADATA_URL,
        cache_ttl_seconds: int = DEFAULT_AUTH_CACHE_TTL_SECONDS,
    ):
        self._app_id = str(app_id or "").strip()
        self._http_client = http_client
        self._metadata_url = metadata_url
        self._cache_ttl_seconds = max(300, int(cache_ttl_seconds))
        self._openid_config: Optional[dict[str, Any]] = None
        self._openid_config_expiry = 0.0
        self._jwks: Optional[dict[str, Any]] = None
        self._jwks_expiry = 0.0
        self._lock = asyncio.Lock()

    async def validate_authorization_header(
        self,
        authorization: str,
        *,
        service_url: str | None = None,
    ) -> bool:
        """Return True when *authorization* contains a valid Bot Framework JWT."""
        scheme, _, token = str(authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return False
        try:
            await self.validate(token.strip(), service_url=service_url)
            return True
        except Exception as exc:
            logger.warning("msteams: Bot Framework JWT validation failed: %s", exc)
            return False

    async def validate(
        self,
        token: str,
        *,
        service_url: str | None = None,
    ) -> dict[str, Any]:
        if not self._app_id:
            raise AuthError("MSTEAMS_APP_ID is required for JWT validation")

        header = jwt.get_unverified_header(token)
        algorithm = str(header.get("alg") or "")
        if algorithm != "RS256":
            raise jwt.InvalidAlgorithmError(f"unsupported JWT alg: {algorithm!r}")

        kid = str(header.get("kid") or "").strip()
        jwks = await self._get_jwks()
        signing_key = self._resolve_signing_key(kid, jwks)
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=self._app_id,
            issuer=BOT_FRAMEWORK_VALID_ISSUERS,
            options={"require": ["exp", "iss", "aud"]},
        )

        if service_url:
            token_service_url = str(
                payload.get("serviceurl") or payload.get("serviceUrl") or ""
            ).strip()
            if (
                token_service_url
                and token_service_url.rstrip("/") != service_url.rstrip("/")
            ):
                raise jwt.InvalidTokenError("Bot Framework token serviceUrl mismatch")

        return payload

    async def _get_openid_config(self) -> dict[str, Any]:
        now = time.time()
        async with self._lock:
            if self._openid_config and now < self._openid_config_expiry:
                return self._openid_config

            response = await self._http_client.get(self._metadata_url)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise AuthError("Bot Framework OpenID metadata was not a JSON object")
            self._openid_config = payload
            self._openid_config_expiry = now + self._cache_ttl_seconds
            return payload

    async def _get_jwks(self) -> dict[str, Any]:
        now = time.time()
        async with self._lock:
            if self._jwks and now < self._jwks_expiry:
                return self._jwks

        openid_config = await self._get_openid_config()
        jwks_uri = str(openid_config.get("jwks_uri") or "").strip()
        if not jwks_uri:
            raise AuthError("Bot Framework OpenID metadata missing jwks_uri")

        async with self._lock:
            now = time.time()
            if self._jwks and now < self._jwks_expiry:
                return self._jwks

            response = await self._http_client.get(jwks_uri)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("keys"), list):
                raise AuthError("Bot Framework JWKS payload missing keys")
            self._jwks = payload
            self._jwks_expiry = now + self._cache_ttl_seconds
            return payload

    @staticmethod
    def _resolve_signing_key(kid: str, jwks: dict[str, Any]) -> Any:
        keys = jwks.get("keys") or []
        for key_payload in keys:
            if not isinstance(key_payload, dict):
                continue
            if kid and str(key_payload.get("kid") or "") != kid:
                continue
            return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key_payload))
        raise AuthError(f"Bot Framework signing key not found for kid={kid!r}")


class BotFrameworkTokenProvider:
    """Acquire Bot Framework access tokens for outbound activity posts."""

    def __init__(
        self,
        *,
        app_id: str,
        app_password: str,
        tenant_id: str = "botframework.com",
        http_client: httpx.AsyncClient,
        authority_host: str = "https://login.microsoftonline.com",
    ):
        self.app_id = str(app_id or "").strip()
        self.tenant_id = str(tenant_id or "botframework.com").strip()
        self._app_password = str(app_password or "")
        self._http_client = http_client
        self._authority_host = authority_host.rstrip("/")
        self._cache: dict[str, tuple[str, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

        if not self.app_id:
            raise AuthError("MSTEAMS_APP_ID is required")
        if not self._app_password:
            raise AuthError("MSTEAMS_APP_PASSWORD is required")

    def _lock_for(self, scope: str) -> asyncio.Lock:
        lock = self._locks.get(scope)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[scope] = lock
        return lock

    async def get_token(self, scope: str = BOT_FRAMEWORK_SCOPE) -> str:
        now = time.time()
        cached = self._cache.get(scope)
        if cached and cached[1] - _TOKEN_REFRESH_LEEWAY_SECONDS > now:
            return cached[0]

        async with self._lock_for(scope):
            cached = self._cache.get(scope)
            now = time.time()
            if cached and cached[1] - _TOKEN_REFRESH_LEEWAY_SECONDS > now:
                return cached[0]

            token_url = (
                f"{self._authority_host}/{self.tenant_id}/oauth2/v2.0/token"
            )
            response = await self._http_client.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.app_id,
                    "client_secret": self._app_password,
                    "scope": scope,
                },
            )
            try:
                payload = response.json()
            except Exception:
                payload = {}
            if response.status_code >= 400 or "access_token" not in payload:
                detail = (
                    payload.get("error_description")
                    or payload.get("error")
                    or getattr(response, "text", "")
                    or f"HTTP {response.status_code}"
                )
                raise AuthError(f"Bot Framework token acquisition failed: {detail}")

            expires_in = int(payload.get("expires_in") or 3600)
            expires_at = time.time() + expires_in
            access_token = str(payload["access_token"])
            self._cache[scope] = (access_token, expires_at)
            return access_token
