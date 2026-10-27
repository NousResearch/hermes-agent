"""Secret-safe digest of the pool key(s) behind an auxiliary client-cache hint.

Split from ``agent.auxiliary_client`` (facade size cap).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _pool_credential_digest(pool: Any, entry: Any = None) -> str:
    """Secret-safe digest over the runtime key(s) backing the cache hint.

    When ``entry`` (the peeked selection) carries a key, only THAT key is digested: rotating a
    sibling account must not churn this entry's cached client. Otherwise every pool entry
    contributes, taken WITHOUT availability filters (``pool.entries()``, not ``peek()``): an entry
    benched by a per-model cooldown still counts, so a token rotated while the cooldown is active
    changes the digest too (#113022 point 5). Only the blake2b of the keys enters the cache key.
    """
    from agent.auxiliary_client import _pool_runtime_api_key

    keys = [k for k in (_pool_runtime_api_key(entry),) if k] if entry is not None else []
    if not keys:
        try:
            entries_fn = getattr(pool, "entries", None)
            entries = entries_fn() if callable(entries_fn) else []
        except Exception as exc:
            logger.debug("Auxiliary client: could not list pool entries for digest: %s", exc)
            return ""
        keys = sorted(k for k in (_pool_runtime_api_key(e) for e in entries or ()) if k)
    if not keys:
        return ""
    digest = hashlib.blake2b(digest_size=16)
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
