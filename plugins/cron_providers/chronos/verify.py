"""Inbound cron-fire token verification for Chronos (Phase 4E.1).

When NAS relays an external scheduler fire to the agent, it POSTs
``/api/cron/fire`` with a short-lived NAS-minted JWT. This module verifies that
JWT before any job runs — the security boundary for remotely-triggered job
execution.

We verify a NAS-minted JWT (the trust path the agent already has) rather than
let an external scheduler call the agent directly: the scheduler signs with
NAS's keys, which the agent doesn't (and shouldn't) hold. See the plan's DQ-4.

The verifier is pluggable (``get_fire_verifier``) so the escape-hatch mode
(direct per-job cron-key) can swap in later with no handler change.

Crypto is delegated to PyJWT (already a declared dependency) — we do NOT
hand-roll JWT verification.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("cron.chronos.verify")

# The purpose claim that scopes a token to the fire endpoint. A general agent
# JWT (without this claim) must NOT be replayable against /api/cron/fire.
_FIRE_PURPOSE = "cron_fire"

# Process-wide cache of PyJWKClient instances, keyed by JWKS URL.
#
# WHY THIS EXISTS: a PyJWKClient caches the fetched JWKS (signing keys) on the
# INSTANCE. Constructing a fresh client per fire therefore threw that cache
# away and forced a synchronous JWKS HTTP GET to the portal on EVERY fire. Under
# a burst of concurrent fires (an instance with several cron jobs firing in the
# same window) that fanned out into N simultaneous JWKS fetches, which the
# portal rate-limited (HTTP 403) — verification then failed and the agent
# answered 401. When a fetch was merely slow rather than rate-limited, it blocked
# the event loop long enough that the fire webhook could not return its 202
# before the relay's 30s timeout (observed in prod as relay 504s concentrated on
# high-job-count instances). Reusing one client per URL keeps the signing keys
# cached (NAS keys rotate rarely), so the steady state is zero JWKS fetches per
# fire. See docs/chronos-managed-cron-contract.md and the betterstack triage.
_JWK_CLIENTS: Dict[str, Any] = {}
_JWK_CLIENTS_LOCK = threading.Lock()


def _get_jwk_client(jwks_url: str) -> Any:
    """Return a process-cached PyJWKClient for ``jwks_url`` (one per URL).

    PyJWKClient does its own key caching internally (``cache_keys``/``lifespan``);
    the whole point here is to reuse the SAME instance across fires so that cache
    is actually hit instead of discarded. Double-checked-locked so concurrent
    fires resolve to a single shared client without racing.
    """
    client = _JWK_CLIENTS.get(jwks_url)
    if client is not None:
        return client
    with _JWK_CLIENTS_LOCK:
        client = _JWK_CLIENTS.get(jwks_url)
        if client is None:
            from jwt import PyJWKClient

            client = PyJWKClient(jwks_url)
            _JWK_CLIENTS[jwks_url] = client
        return client


def verify_nas_fire_token(
    *,
    token: str,
    expected_audience: str,
    jwks_or_key: Optional[str] = None,
    issuer: Optional[str] = None,
    leeway_seconds: int = 30,
) -> Optional[Dict[str, Any]]:
    """Verify a NAS-minted cron-fire JWT. Return decoded claims, or None.

    Checks (all must pass):
      - signature against the NAS JWKS (``jwks_or_key`` is a JWKS URL) — RS256
        family; symmetric secrets are rejected (NAS signs asymmetrically).
      - ``aud`` == ``expected_audience`` (this agent: ``agent:{instance_id}``).
      - ``exp`` / ``nbf`` within ``leeway_seconds``.
      - ``iss`` == ``issuer`` when an issuer is configured.
      - ``purpose`` == ``"cron_fire"`` — so a general agent JWT can't be
        replayed against the fire endpoint.

    Returns None (never raises) on any failure, so the handler can answer 401
    without leaking which check failed.
    """
    if not token or not expected_audience:
        return None
    if not jwks_or_key:
        # No verification key configured → cannot verify → refuse. We never
        # fall back to unsigned decode for a security boundary.
        logger.warning("cron fire: no JWKS/key configured; refusing token")
        return None

    try:
        import jwt

        # Resolve the signing key from the JWKS endpoint by the token's kid.
        signing_key = None
        if jwks_or_key.startswith("http://") or jwks_or_key.startswith("https://"):
            # Reuse a process-cached client so the JWKS fetch is amortised across
            # fires (a fresh client per fire re-fetched the JWKS every time and,
            # under concurrent fires, tripped the portal's rate limit → 403 →
            # 401, or blocked the event loop past the relay's 30s timeout → 504).
            jwk_client = _get_jwk_client(jwks_or_key)
            signing_key = jwk_client.get_signing_key_from_jwt(token).key
        else:
            # A PEM public key passed inline (test / pinned-key deployments).
            signing_key = jwks_or_key

        options = {"require": ["exp", "aud"]}
        decode_kwargs: Dict[str, Any] = dict(
            algorithms=["RS256", "RS384", "RS512", "ES256", "ES384"],
            audience=expected_audience,
            leeway=leeway_seconds,
            options=options,
        )
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
    """Return the active inbound-fire verifier.

    Default = the NAS-JWT verifier. The DQ-4 escape hatch (direct per-job
    cron-key) would return a cron-key verifier here instead, selected by config
    — so the webhook handler never changes when the auth mode is swapped.
    """
    return verify_nas_fire_token
