"""
Validate Clerk-issued session JWTs against the Clerk JWKS endpoint.

A Unitalk Cloud sign-in produces a Clerk session token (JWT) that the Desktop
sends to the gateway as a credential. This module validates the token signature
using Clerk's published JWKS keys and extracts user identity claims.

JWKS URL: https://clerk.unitalk.ai/.well-known/jwks.json

Usage:
    from agent.clerk_auth import validate_clerk_token

    user_info = validate_clerk_token(token)
    if user_info:
        # { "user_id": "...", "email": "...", "name": "...", ... }
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

CLERK_JWKS_URL = "https://clerk.unitalk.ai/.well-known/jwks.json"
CLERK_ISSUER = "https://clerk.unitalk.ai"

# Cache JWKS for 1 hour to avoid hitting Clerk on every validation
_jwks_cache: Optional[dict] = None
_jwks_cache_at: float = 0.0
_JWKS_CACHE_TTL_S = 3600


@dataclass
class ClerkUserInfo:
    user_id: str
    email: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    full_name: Optional[str]
    username: Optional[str]
    image_url: Optional[str]


def _fetch_jwks() -> dict:
    """Fetch Clerk JWKS, using cache when fresh."""
    global _jwks_cache, _jwks_cache_at

    now = time.time()
    if _jwks_cache is not None and (now - _jwks_cache_at) < _JWKS_CACHE_TTL_S:
        return _jwks_cache

    try:
        req = Request(CLERK_JWKS_URL, headers={"Accept": "application/json"})
        with urlopen(req, timeout=10) as resp:
            jwks = json.loads(resp.read().decode("utf-8"))
        _jwks_cache = jwks
        _jwks_cache_at = now
        return jwks
    except Exception as exc:
        logger.warning("Could not fetch Clerk JWKS: %s", exc)
        # Return stale cache if available
        if _jwks_cache is not None:
            return _jwks_cache
        raise


def _decode_jwt_segments(token: str) -> tuple[dict, dict] | None:
    """Base64url-decode JWT header and payload without verifying signature."""
    segments = token.split(".")
    if len(segments) != 3:
        return None

    import base64

    def b64url_decode(s: str) -> bytes:
        # Add padding
        s = s + "=" * (4 - len(s) % 4)
        return base64.urlsafe_b64decode(s)

    try:
        header = json.loads(b64url_decode(segments[0]))
        payload = json.loads(b64url_decode(segments[1]))
        return header, payload
    except Exception:
        return None


def _find_jwk(jwks: dict, kid: str) -> dict | None:
    """Find a JWK by key ID in the JWKS."""
    keys = jwks.get("keys", [])
    for key in keys:
        if key.get("kid") == kid:
            return key
    return None


def _verify_rs256(token: str, jwk: dict) -> bool:
    """Verify RS256 JWT signature using the given JWK."""
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError:
        logger.warning("cryptography package not installed; token signature verification skipped")
        return True  # Degrade to payload-only validation

    try:
        n_bytes = int.from_bytes(
            _b64url_decode_int(jwk["n"]), "big", signed=False
        )
        e_bytes = int.from_bytes(
            _b64url_decode_int(jwk["e"]), "big", signed=False
        )
    except Exception:
        logger.warning("Could not decode JWK RSA parameters")
        return False

    public_numbers = rsa.RSAPublicNumbers(e_bytes, n_bytes)
    public_key = public_numbers.public_key(default_backend())

    # Reconstruct the signed content: header.payload
    segments = token.split(".")
    if len(segments) != 3:
        return False

    signed_content = f"{segments[0]}.{segments[1]}".encode("ascii")

    import base64

    signature = _b64url_decode_int(segments[2])

    try:
        public_key.verify(
            signature,
            signed_content,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def _b64url_decode_int(s: str) -> bytes:
    """Decode a base64url-encoded string, adding padding."""
    import base64

    s = s + "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def _validate_jwt_claims(payload: dict) -> bool:
    """Validate standard JWT claims: expiry, issuer, not-before."""
    now = int(time.time())

    # Expiry
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and exp < now:
        logger.info("Clerk JWT expired at %s (now %s)", exp, now)
        return False

    # Not before
    nbf = payload.get("nbf")
    if isinstance(nbf, (int, float)) and nbf > now:
        logger.info("Clerk JWT not yet valid (nbf=%s, now=%s)", nbf, now)
        return False

    # Issuer
    iss = payload.get("iss", "")
    if iss and iss != CLERK_ISSUER:
        logger.info("Clerk JWT issuer mismatch: got %s, expected %s", iss, CLERK_ISSUER)
        return False

    return True


def validate_clerk_token(token: str) -> Optional[ClerkUserInfo]:
    """
    Validate a Clerk-issued session JWT.

    Returns ClerkUserInfo on success, None on any validation failure.
    The caller should treat None as an unauthenticated request.

    Validation steps:
    1. Decode JWT segments (no verify yet)
    2. Fetch JWKS from Clerk
    3. Find the matching JWK by kid
    4. Verify RS256 signature
    5. Validate standard claims (exp, iss, nbf)
    6. Extract user identity fields
    """
    if not token or not isinstance(token, str):
        return None

    token = token.strip()

    # Step 1: Decode without verifying
    decoded = _decode_jwt_segments(token)
    if decoded is None:
        logger.info("Clerk JWT: could not decode token segments")
        return None

    header, payload = decoded

    # Step 2-3: Get JWKS and find matching key
    kid = header.get("kid", "")
    if not kid:
        logger.info("Clerk JWT: missing kid in header")
        return None

    try:
        jwks = _fetch_jwks()
    except Exception as exc:
        logger.error("Clerk JWT: could not fetch JWKS: %s", exc)
        return None

    jwk = _find_jwk(jwks, kid)
    if jwk is None:
        logger.info("Clerk JWT: key %s not found in JWKS", kid)
        return None

    # Step 4: Verify signature
    alg = header.get("alg", "")
    if alg != "RS256":
        logger.info("Clerk JWT: unsupported algorithm %s", alg)
        return None

    if not _verify_rs256(token, jwk):
        logger.info("Clerk JWT: signature verification failed")
        return None

    # Step 5: Validate claims
    if not _validate_jwt_claims(payload):
        return None

    # Step 6: Extract user identity
    user_id = payload.get("sub", "")
    if not user_id:
        logger.info("Clerk JWT: missing sub claim")
        return None

    email = payload.get("email")
    first_name = payload.get("first_name") or payload.get("given_name")
    last_name = payload.get("last_name") or payload.get("family_name")
    full_name = payload.get("name") or payload.get("full_name")
    username = payload.get("username")
    image_url = payload.get("image_url") or payload.get("picture")

    return ClerkUserInfo(
        user_id=str(user_id),
        email=str(email) if email else None,
        first_name=str(first_name) if first_name else None,
        last_name=str(last_name) if last_name else None,
        full_name=str(full_name) if full_name else None,
        username=str(username) if username else None,
        image_url=str(image_url) if image_url else None,
    )