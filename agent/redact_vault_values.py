"""Exact-value redaction of secrets that transited a server-side vault fill (browser_vault), plus the
values a plugin registers through ``register_plugin_redaction_value``.

Re-exported by ``agent.redact``; moved out of it unchanged.
"""

from __future__ import annotations

import threading

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
_VAULT_REDACTION_VALUES: dict = {}  # profile home → ordered {value: None}
_VAULT_REDACTION_LOCK = threading.Lock()
# Values a plugin registers live in their own bucket with their own cap, so a plugin can neither evict
# a vault-fill value nor flood the scrub list with short strings that would mangle every result.
PLUGIN_REDACTION_MIN_LENGTH = 8
_PLUGIN_REDACTION_MAX_PER_PROFILE = 64
_PLUGIN_REDACTION_VALUES: dict = {}  # profile home → ordered {value: None}


def _vault_scope() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def register_vault_redaction_value(value) -> None:
    """Register an exact vault secret value for model-facing redaction.

    Called by the vault fill path BEFORE the injection happens, so no later browser tool result
    can echo the value back into model context. Also registers the form a text input normalizes
    it to (CR/LF stripped), since that is what the page holds.
    """
    if not isinstance(value, str) or not value:
        return
    normalized = value.replace("\r", "").replace("\n", "")
    with _VAULT_REDACTION_LOCK:
        bucket = _VAULT_REDACTION_VALUES.setdefault(_vault_scope(), {})
        for v in (value, normalized):
            if v:
                bucket.pop(v, None)  # re-registering refreshes recency
                bucket[v] = None
        while len(bucket) > _VAULT_REDACTION_MAX_PER_PROFILE:
            del bucket[next(iter(bucket))]


def register_plugin_redaction_value(value) -> None:
    """Register an exact secret value a plugin holds, so Hermes scrubs it from model-facing output.

    The public entry for plugins. The value is kept in the plugin bucket of the current profile, never
    in the vault-fill bucket, so it cannot evict a vault value, and it is scrubbed for that profile
    only. ``ValueError`` for a value shorter than ``PLUGIN_REDACTION_MIN_LENGTH`` characters (a short
    string would blank ordinary text) or one that is not a string. Past the per-profile cap the oldest
    plugin value drops.
    """
    if not isinstance(value, str) or len(value) < PLUGIN_REDACTION_MIN_LENGTH:
        raise ValueError(f"a redaction value must be a string of at least {PLUGIN_REDACTION_MIN_LENGTH} characters")
    with _VAULT_REDACTION_LOCK:
        bucket = _PLUGIN_REDACTION_VALUES.setdefault(_vault_scope(), {})
        bucket.pop(value, None)  # re-registering refreshes recency
        bucket[value] = None
        while len(bucket) > _PLUGIN_REDACTION_MAX_PER_PROFILE:
            del bucket[next(iter(bucket))]


def clear_vault_redaction_values() -> None:
    """Drop the current profile's registered values (profile teardown / explicit lock)."""
    with _VAULT_REDACTION_LOCK:
        _VAULT_REDACTION_VALUES.pop(_vault_scope(), None)
        _PLUGIN_REDACTION_VALUES.pop(_vault_scope(), None)


def redact_registered_vault_values(text: str) -> str:
    """Exact-substring scrub of every vault secret value registered for the current profile."""
    if not isinstance(text, str) or not text:
        return text
    with _VAULT_REDACTION_LOCK:
        scope = _vault_scope()
        merged = {**(_VAULT_REDACTION_VALUES.get(scope) or {}), **(_PLUGIN_REDACTION_VALUES.get(scope) or {})}
        values = sorted(merged, key=len, reverse=True)  # longest first: a substring never shadows its superstring
    for value in values:
        if value in text:
            text = text.replace(value, "«redacted-vault-secret»")
    return text
