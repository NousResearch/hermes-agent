"""Inbound cron-fire token verification for Chronos.

NAS POSTs ``/api/cron/fire`` with a short-lived NAS-minted JWT, verified via PyJWT (never
hand-rolled) before any job runs. ``get_fire_verifier`` is the pluggable seam so an alternative
auth mode (direct per-job cron-key) can swap in without a handler change.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("cron.chronos.verify")

# Purpose claim scoping a token to the fire endpoint (a general agent JWT must not replay here).
_FIRE_PURPOSE = "cron_fire"

# Process-wide PyJWKClient cache keyed by JWKS URL: PyJWKClient caches keys on the INSTANCE, so a
# fresh client per fire re-fetched JWKS every time (portal rate-limits 403->401; relay 504s).
_JWK_CLIENTS: Dict[str, Any] = {}
_JWK_CLIENTS_LOCK = threading.Lock()


def _get_jwk_client(jwks_url: str) -> Any:
    """Return the process-cached PyJWKClient for ``jwks_url`` (double-checked lock)."""
    client = _JWK_CLIENTS.get(jwks_url)
    if client is not None:
        return client
    with _JWK_CLIENTS_LOCK:
        client = _JWK_CLIENTS.get(jwks_url)
        if client is None:
            from jwt import PyJWKClient
            # Explicit Accept + User-Agent: the portal WAF 403s the default urllib fingerprint.
            client = PyJWKClient(
                jwks_url, headers={"Accept": "application/json", "User-Agent": "HermesAgent/1.0"})
            _JWK_CLIENTS[jwks_url] = client
        return client


def verify_nas_fire_token(*, token: str, expected_audience: str, jwks_or_key: Optional[str] = None,
                          issuer: Optional[str] = None, leeway_seconds: int = 30) -> Optional[Dict[str, Any]]:
    """Verify a NAS-minted cron-fire JWT; return decoded claims or None (never raises, so the
    handler answers 401 without leaking which check failed). Checks asymmetric signature (JWKS
    URL or inline PEM; symmetric rejected), ``aud``, ``exp``/``nbf`` with leeway, ``iss`` when
    configured, and ``purpose == "cron_fire"``."""
    if not token or not expected_audience:
        return None
    if not jwks_or_key:  # never fall back to unsigned decode on a security boundary
        logger.warning("cron fire: no JWKS/key configured; refusing token")
        return None
    try:
        import jwt
        if jwks_or_key.startswith(("http://", "https://")):
            signing_key = _get_jwk_client(jwks_or_key).get_signing_key_from_jwt(token).key
        else:
            signing_key = jwks_or_key  # inline PEM public key (test / pinned-key deployments)
        decode_kwargs: Dict[str, Any] = dict(
            algorithms=["RS256", "RS384", "RS512", "ES256", "ES384"], audience=expected_audience,
            leeway=leeway_seconds, options={"require": ["exp", "aud"]})
        if issuer:
            decode_kwargs["issuer"] = issuer
        claims = jwt.decode(token, signing_key, **decode_kwargs)
    except Exception as e:
        logger.warning("cron fire: token verification failed: %s", e)
        return None
    if claims.get("purpose") != _FIRE_PURPOSE:
        logger.warning("cron fire: token missing/!=%s purpose claim", _FIRE_PURPOSE)
        return None
    return claims


def get_fire_verifier() -> Callable[..., Optional[Dict[str, Any]]]:
    """Return the active inbound-fire verifier (default: the NAS-JWT verifier)."""
    return verify_nas_fire_token
