"""Fail-closed entropy gate for service-credential shared secrets.

Shared by every non-interactive bearer-secret auth plugin (drain control,
the kanban REST API credential, ...): a weak/short/low-entropy secret must
fail CLOSED at plugin load — it is never silently accepted. Bar: >= 256 bits
of entropy / >= 43 url-safe-base64 chars, and the value must not be obviously
structured (all-one-character, too few distinct characters).

Extracted from the drain plugin (the first consumer) so each new credential
plugin reuses one gate instead of growing its own slightly-different copy.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

# Default entropy bar: 43 url-safe-base64 chars ~= 256 bits. token_urlsafe(32)
# produces 43 chars, so a correctly-provisioned secret clears this exactly.
DEFAULT_MIN_SECRET_CHARS = 43
# A secret must contain at least this many DISTINCT characters — rejects
# degenerate values like "aaaa..." that are long but trivially low-entropy.
_MIN_DISTINCT_CHARS = 16
# Shannon entropy floor (bits) over the secret's characters — a second,
# distribution-aware guard on top of the length + distinct-count checks.
_MIN_SHANNON_BITS = 128.0


def _shannon_bits(value: str) -> float:
    """Total Shannon entropy (bits) of ``value`` over its character distribution.

    H = len * sum(-p_i * log2(p_i)). A long string drawn from a wide alphabet
    scores high; a long run of one character scores ~0.
    """
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    per_char = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return per_char * n


def assess_secret_strength(
    secret: str, *, min_chars: int = DEFAULT_MIN_SECRET_CHARS
) -> Optional[str]:
    """Return a rejection reason if ``secret`` is too weak, else ``None``.

    Fail-closed entropy gate (decisions.md Q-A). Checks, in order:
      * length >= ``min_chars`` (default 43 url-safe-b64 chars ~= 256 bits),
      * at least ``_MIN_DISTINCT_CHARS`` distinct characters,
      * Shannon entropy >= ``_MIN_SHANNON_BITS`` bits.

    A ``None`` return means the secret passes. Any string return is a
    human-readable reason the caller logs + records as the skip reason.
    """
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
