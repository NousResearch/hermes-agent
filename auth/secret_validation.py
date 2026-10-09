"""Shared provider credential validation, without source or route selection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PLACEHOLDER_SECRET_VALUES = {
    "*", "**", "***", "changeme", "your_api_key", "your_api_key_here", "your-api-key",
    "placeholder", "example", "dummy", "null", "none"}


_PLACEHOLDER_KEY_PREFIXES = ("sk-", "ghp_", "hf_")


def _is_placeholder_shape(value: str) -> bool:
    """True for the placeholder shapes shipped in .env.example and the docs."""
    lowered = value.lower()
    if lowered.startswith("your_") and lowered.endswith("_here"):
        return True
    for prefix in _PLACEHOLDER_KEY_PREFIXES:
        if lowered.startswith(prefix):
            tail = lowered[len(prefix):]
            if tail and all(c == "x" for c in tail):
                return True
    stripped = lowered.replace(" ", "").replace("-", "").replace("_", "")
    return bool(stripped) and all(c == "x" for c in stripped)


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    """Return True when a configured secret looks usable, not empty/placeholder."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    return (len(cleaned) >= min_length
            and cleaned.lower() not in _PLACEHOLDER_SECRET_VALUES
            and not _is_placeholder_shape(cleaned))


KNOWN_PROVIDER_KEY_PREFIXES: Dict[str, tuple] = {
    "openrouter": ("sk-or-",),  # all OpenRouter keys are sk-or-... (currently sk-or-v1-)
}


def _matches_key_prefix(provider_id: str, val: str) -> bool:
    """True when *val* starts with one of *provider_id*'s declared key prefixes (False when the
    provider declares none)."""
    return val.startswith(KNOWN_PROVIDER_KEY_PREFIXES.get(provider_id, ()))


def looks_like_openrouter_key(value: Any) -> bool:
    """True when *value* carries an OpenRouter key prefix. OPENAI_API_KEY is a legacy home for an
    OpenRouter key, so only a value shaped like one may be read as an OpenRouter credential: a real
    OpenAI key must never be auto-routed to, or sent to, openrouter.ai."""
    return _matches_key_prefix("openrouter", str(value or "").strip())


def _usable_declared_secret(provider_id: str, value: Any, source: str) -> Optional[str]:
    """*value* stripped when it is a usable, prefix-valid secret; None (after warning on a provable
    prefix mismatch, so it never shadows a later credential source) otherwise. Providers without a
    declared prefix are fail-open."""
    val = str(value or "").strip()
    if not has_usable_secret(val):
        return None
    prefixes = KNOWN_PROVIDER_KEY_PREFIXES.get(provider_id)
    if prefixes and not _matches_key_prefix(provider_id, val):
        logger.warning(
            "Ignoring %s for provider %r: value does not match the expected key "
            "prefix (%s). Falling back to the next credential source. Fix or "
            "remove the malformed key to silence this warning.",
            source, provider_id, " or ".join(prefixes))
        return None
    return val
