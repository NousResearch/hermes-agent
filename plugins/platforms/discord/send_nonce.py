"""Deterministic Discord message nonces for re-sent ledgered finals.

discord.py gives every ``Messageable.send`` a fresh random nonce and sets ``enforce_nonce``,
so its own internal retries are deduplicated but a second Hermes-level send of the same reply
(``_send_with_retry``, the plain-text fallback, the delivery ledger's reconnect/boot
redelivery) creates a second visible message whenever the first one was created at Discord and
only its response was lost. Deriving the nonce from the delivery-ledger obligation id makes
every such re-send of the same chunk carry the same nonce, so Discord returns the message it
already created instead of posting a copy (enforce_nonce dedups within a few minutes).
"""

import hashlib
from typing import Any, Optional

# Discord's Create Message ``nonce``: "integer or string, up to 25 characters".
_NONCE_MAX_CHARS = 25


def chunk_nonce(idempotency_key: str, index: int) -> str:
    """The nonce for chunk ``index`` of the send identified by ``idempotency_key``."""
    digest = hashlib.sha256(f"{idempotency_key}:{index}".encode("utf-8", "replace")).hexdigest()
    return digest[:_NONCE_MAX_CHARS]


def nonce_kwargs(metadata: Optional[dict[str, Any]], index: int) -> dict[str, str]:
    """``{"nonce": ...}`` for ``channel.send`` when ``metadata`` carries an idempotency key;
    empty otherwise, so discord.py keeps its random per-call nonce."""
    key = (metadata or {}).get("idempotency_key")
    if not isinstance(key, str) or not key:
        return {}
    return {"nonce": chunk_nonce(key, index)}
