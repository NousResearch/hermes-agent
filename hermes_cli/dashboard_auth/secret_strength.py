"""Shared fail-closed strength checks for dashboard service secrets."""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional


# Default entropy bar: 43 url-safe-base64 chars ~= 256 bits. token_urlsafe(32)
# produces 43 chars, so a correctly-provisioned secret clears this exactly.
_DEFAULT_MIN_SECRET_CHARS = 43
_MIN_DISTINCT_CHARS = 16
_MIN_SHANNON_BITS = 128.0


def _shannon_bits(value: str) -> float:
    """Return the total Shannon entropy (bits) of ``value``."""
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    per_char = -sum((count / n) * math.log2(count / n) for count in counts.values())
    return per_char * n


def assess_secret_strength(
    secret: str, *, min_chars: int = _DEFAULT_MIN_SECRET_CHARS
) -> Optional[str]:
    """Return a rejection reason for a weak secret, else ``None``."""
    if not secret:
        return "secret is empty"
    if len(secret) < min_chars:
        return (
            f"secret too short: {len(secret)} chars (need >= {min_chars}; "
            "use a >=256-bit value, e.g. `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"`)"
        )
    distinct = len(set(secret))
    if distinct < _MIN_DISTINCT_CHARS:
        return (
            f"secret has only {distinct} distinct characters (need >= "
            f"{_MIN_DISTINCT_CHARS}); looks structured/low-entropy"
        )
    bits = _shannon_bits(secret)
    if bits < _MIN_SHANNON_BITS:
        return (
            f"secret entropy too low: {bits:.0f} bits (need >= "
            f"{_MIN_SHANNON_BITS:.0f}); looks structured/repeated"
        )
    return None
