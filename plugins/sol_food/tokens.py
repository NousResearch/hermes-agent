"""Opaque Sol-food callback tokens (``sf1:`` scheme).

A token is exactly ``sf1:`` + an unpadded 22-character base64url encoding
of 128 cryptographically random bits — 26 ASCII bytes total, below
Telegram's 1–64-byte callback-data limit. The token is fully opaque: it
carries no food text, Telegram identity, Supabase identity, action, or
serialized payload. The server-side row keyed by the token carries the
action and all origin bindings.
"""

from __future__ import annotations

import base64
import re
import secrets
from typing import Optional

from plugins.sol_food.limits import (
    FOOD_CALLBACK_B64_CHARS,
    FOOD_CALLBACK_PREFIX,
    FOOD_CALLBACK_RANDOM_BYTES,
    FOOD_CALLBACK_TOKEN_BYTES,
)

__all__ = ["mint_token", "parse_token"]

_TOKEN_RE = re.compile(
    r"\A" + re.escape(FOOD_CALLBACK_PREFIX) + r"[A-Za-z0-9_-]{%d}\Z" % FOOD_CALLBACK_B64_CHARS
)


def mint_token() -> str:
    """Mint a fresh opaque callback token."""
    raw = secrets.token_bytes(FOOD_CALLBACK_RANDOM_BYTES)
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    token = FOOD_CALLBACK_PREFIX + encoded
    # Invariants, cheap enough to assert on every mint.
    if len(token) != FOOD_CALLBACK_TOKEN_BYTES or not _TOKEN_RE.match(token):
        raise AssertionError("sf1 token invariant violation")
    return token


def parse_token(callback_data: object) -> Optional[str]:
    """Return the token if ``callback_data`` is a well-formed sf1 token.

    Strict: anything that is not exactly 26 ASCII characters of the sf1
    grammar (including padded, over-long, truncated, or non-str input)
    returns None. Callers treat None as a fail-closed malformed callback.
    """
    if not isinstance(callback_data, str):
        return None
    if len(callback_data) != FOOD_CALLBACK_TOKEN_BYTES:
        return None
    if not _TOKEN_RE.match(callback_data):
        return None
    return callback_data
