"""Normalize the provider error context a credential-pool row records on exhaustion."""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from agent.credential_pool_timestamps import _parse_absolute_timestamp
from agent.redact import redact_sensitive_text
from agent.retry_utils import reset_delay_from_message

# Shorter values are too likely to collide with ordinary words in the error text.
_MIN_KNOWN_SECRET_LEN = 8


def redact_error_text(text: str, known_secrets: Iterable[Optional[str]] = ()) -> str:
    """Mask *text* for persistence: the row's own secrets verbatim, then the pattern redactor.

    The pattern pass alone misses opaque keys (custom providers, random hex), so a provider echoing
    the rejected key back would still reach auth.json. force=True: a persistence boundary must not
    depend on security.redact_secrets.
    """
    secrets = {s.strip() for s in known_secrets if isinstance(s, str) and len(s.strip()) >= _MIN_KNOWN_SECRET_LEN}
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "***")
    return redact_sensitive_text(text, force=True)


def _normalize_error_context(
    error_context: Optional[dict[str, Any]], known_secrets: Iterable[Optional[str]] = ()
) -> dict[str, Any]:
    if not isinstance(error_context, dict):
        return {}
    known_secrets = tuple(known_secrets)
    normalized: dict[str, Any] = {}
    # Providers echo the rejected key back ("Invalid API key: sk-..."), and both fields land in
    # auth.json (last_error_reason / last_error_message), which the sanitizer keeps (#71994);
    # "reason" carries the whole message when a body's "error" is a bare string.
    for key in ("reason", "message"):
        value = error_context.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = redact_error_text(value.strip(), known_secrets)
    reset_at = (
        error_context.get("reset_at")
        or error_context.get("resets_at")
        or error_context.get("retry_until")
    )
    parsed_reset_at = _parse_absolute_timestamp(reset_at)
    message = error_context.get("message")
    if parsed_reset_at is None and isinstance(message, str):
        retry_delay_seconds = reset_delay_from_message(message)
        if retry_delay_seconds is not None:
            parsed_reset_at = time.time() + retry_delay_seconds
    if parsed_reset_at is not None:
        normalized["reset_at"] = parsed_reset_at
    return normalized
