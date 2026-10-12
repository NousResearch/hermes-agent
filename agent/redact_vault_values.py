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
_MIN_SECRET_LINE_CHARS = 24
_VAULT_REDACTION_VALUES: dict = {}  # profile home key → ordered {value: tuple of exact fragments}
_VAULT_REDACTION_LOCK = threading.Lock()


def _vault_scope(home=None) -> str:
    from hermes_constants import hermes_home_key
    return hermes_home_key(home)


def register_vault_redaction_value(value, *, home=None) -> None:
    """Register an exact secret value for model-facing redaction.

    Called by the vault fill path BEFORE the injection happens, so no later browser tool result
    can echo the value back into model context, and by the secret-source apply funnel. Also
    registers the form a text input normalizes it to (CR/LF stripped), since that is what the
    page holds, and each line of a multi-line value: line-oriented output (``env | sort``,
    ``grep -v key``, ``sed -n 2,4p`` of a key file) separates a PEM body from its markers, so
    only the exact line can still be recognised. Armor lines (``-----BEGIN …``) and lines under
    ``_MIN_SECRET_LINE_CHARS`` are never registered on their own: the scrub is an unanchored
    substring replace, and those would blank out innocent text (key body lines are 64 chars).

    All fragments form ONE unit for the per-profile bound, so a 50-line key never evicts its own
    early lines or older entries one at a time. ``home`` scopes the unit to a profile that is not
    the ambient one (multiplex hydration resolves profile B with no home override bound).
    """
    if not isinstance(value, str) or not value:
        return
    lines = [line.strip() for line in value.splitlines()] if ("\n" in value or "\r" in value) else []
    fragments = tuple(dict.fromkeys(
        f for f in (value, value.replace("\r", "").replace("\n", ""),
                    *(ln for ln in lines if len(ln) >= _MIN_SECRET_LINE_CHARS and not ln.startswith("-----")))
        if f
    ))
    with _VAULT_REDACTION_LOCK:
        bucket = _VAULT_REDACTION_VALUES.setdefault(_vault_scope(home), OrderedDict())
        bucket.pop(value, None)  # re-registering refreshes recency
        bucket[value] = fragments
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
