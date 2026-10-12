"""Exact-value redaction of secrets that transited a server-side vault fill (browser_vault).

Re-exported by ``agent.redact``; moved out of it unchanged.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Vault-value redaction registry (profile-scoped, bounded)
# ---------------------------------------------------------------------------
# Exact secret values that transited a server-side vault fill (browser_vault). Generic
# credential-shaped regexes cannot catch an arbitrary user password, so the fill path registers
# the exact bytes and every browser_* tool result (including browser_cdp Runtime.evaluate
# passthrough) is scrubbed against them before it can reach the model. Memory only: never
# persisted or logged. Keyed by profile home so a multiplex gateway never scrubs profile B's
# output with profile A's passwords (which would also confirm to B that the bytes exist), and
# bounded per profile: a fill-heavy session evicts its oldest entries rather than growing forever.
_VAULT_REDACTION_MAX_PER_PROFILE = 64
_VAULT_REDACTION_VALUES: dict = {}  # profile home → ordered {unit key: tuple of exact values}
_VAULT_REDACTION_LOCK = threading.Lock()


def _vault_scope() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def register_vault_redaction_value(value) -> None:
    """Register an exact vault secret value for model-facing redaction.

    Called by the vault fill path BEFORE the injection happens, so no later browser tool result
    can echo the value back into model context. Also registers the form a text input normalizes
    it to (CR/LF stripped), since that is what the page holds.
    """
    register_vault_redaction_values((value,))


def register_vault_redaction_values(values) -> None:
    """Atomically register one or more related exact values.

    A multiline secret is treated as one bounded registration unit: when its
    fragments exceed the per-profile capacity, retain evenly spaced fragments
    (including the first and last) rather than evicting the secret's own early
    lines one by one.
    """
    if not values:
        return
    normalized_values = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        normalized = value.replace("\r", "").replace("\n", "")
        normalized_values.append((value, normalized))
    if not normalized_values:
        return
    with _VAULT_REDACTION_LOCK:
        bucket = _VAULT_REDACTION_VALUES.setdefault(_vault_scope(), OrderedDict())
        candidates = []
        for value, normalized in normalized_values:
            lines = [line for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line]
            candidates.extend(v for v in (value, normalized, *lines) if v)
        # One source application is one bounded unit. A multiline secret may
        # contribute many exact fragments, but those fragments must be evicted
        # together rather than evicting the secret's own early lines first.
        unit_key = normalized_values[0][0]
        bucket.pop(unit_key, None)  # re-registering refreshes recency
        bucket[unit_key] = tuple(dict.fromkeys(candidates))
        while len(bucket) > _VAULT_REDACTION_MAX_PER_PROFILE:
            del bucket[next(iter(bucket))]


def clear_vault_redaction_values() -> None:
    """Drop the current profile's registered values (profile teardown / explicit lock)."""
    with _VAULT_REDACTION_LOCK:
        _VAULT_REDACTION_VALUES.pop(_vault_scope(), None)


def redact_registered_vault_values(text: str) -> str:
    """Exact-substring scrub of every vault secret value registered for the current profile."""
    if not isinstance(text, str) or not text:
        return text
    with _VAULT_REDACTION_LOCK:
        bucket = _VAULT_REDACTION_VALUES.get(_vault_scope())
        values = sorted(
            (value for unit in bucket.values() for value in unit),
            key=len,
            reverse=True,
        ) if bucket else ()  # longest first: a substring never shadows its superstring
    for value in values:
        if value in text:
            text = text.replace(value, "«redacted-vault-secret»")
    return text
