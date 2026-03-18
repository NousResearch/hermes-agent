"""HMAC-SHA256 signature verification for Mission Control webhooks."""

import hmac
import hashlib
import logging
import os

logger = logging.getLogger(__name__)


def verify_signature(body: bytes, signature_header: str, secret: str) -> bool:
    """
    Verify X-MC-Signature header from Mission Control.
    
    MC computes HMAC-SHA256(secret, raw_utf8_body) and sends it as:
    X-MC-Signature: sha256=<hex>
    
    Args:
        body: Raw request body bytes
        signature_header: Value of X-MC-Signature header (e.g., "sha256=abc123...")
        secret: Shared webhook secret configured in MC
        
    Returns:
        True if signature valid, False otherwise
        
    Example:
        >>> verify_signature(
        ...     b'{"event": "test"}',
        ...     "sha256=aabbcc...",
        ...     "my-secret"
        ... )
        True
    """
    if not secret:
        # In production, require a secret. Allow unauthenticated only in explicit dev mode.
        if os.getenv("MC_ALLOW_UNAUTHENTICATED", "").lower() != "true":
            logger.error(
                "[mc] No webhook secret configured. Set MC_WEBHOOK_SECRET "
                "or MC_ALLOW_UNAUTHENTICATED=true for development"
            )
            return False
        logger.warning("[mc] Dev mode: accepting unauthenticated requests (MC_ALLOW_UNAUTHENTICATED=true)")
        return True
        
    if not signature_header:
        logger.warning("[mc] Missing X-MC-Signature header")
        return False
        
    # Compute expected signature
    expected_hmac = hmac.new(
        secret.encode('utf-8'),
        body,
        hashlib.sha256
    ).hexdigest()
    expected = f"sha256={expected_hmac}"
    
    # Constant-time comparison to prevent timing attacks
    is_valid = hmac.compare_digest(expected, signature_header)
    
    if not is_valid:
        logger.warning("[mc] Signature mismatch")
        
    return is_valid