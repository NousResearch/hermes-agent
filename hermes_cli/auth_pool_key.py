"""API keys held in the credential pool, read together with the endpoint each one was stored for.

``auth._resolve_api_key_provider_secret_and_url`` falls back here when no env key is set. The
entry that supplies the key also supplies the endpoint, in the same read: a key stored for a
provider-compatible host is never paired with the registry default or with another entry's
endpoint (#42269).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pool_key_and_endpoint(provider_id: str) -> tuple[str, str]:
    """``(api_key, base_url)`` of the first usable pool entry, ``("", "")`` when there is none.

    The pool's own pick (peek) goes first, then the other rows, so one malformed entry does not
    block a valid one. ``base_url`` is "" when the entry stores none.
    """
    from agent.credential_pool import load_pool
    from hermes_cli.auth import _usable_declared_secret

    try:
        pool = load_pool(provider_id)
        if not (pool and pool.has_credentials()):
            return "", ""
        first = pool.peek()
        # Lightweight pool adapters (old plugins) may expose only peek(); real pools list every row.
        rows = pool.entries() if callable(getattr(pool, "entries", None)) else []
    except Exception:  # health: allow BLE001 -- an unreadable pool reads as "no pooled key", never a crash
        logger.debug("credential pool lookup failed for provider %r", provider_id, exc_info=True)
        return "", ""
    candidates = [first] if first is not None else []
    candidates += [entry for entry in rows if entry is not None and entry is not first]
    source = f"credential_pool:{provider_id}"
    for entry in candidates:
        key = getattr(entry, "access_token", "") or getattr(entry, "runtime_api_key", "")
        if val := _usable_declared_secret(provider_id, key, source):
            url = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None)
            return val, url.strip().rstrip("/") if isinstance(url, str) else ""
    return "", ""
