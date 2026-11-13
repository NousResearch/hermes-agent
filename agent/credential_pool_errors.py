"""Normalize the provider error context a credential-pool row records on exhaustion."""

from __future__ import annotations

import time
from typing import Any, Optional

from agent.credential_pool_timestamps import _parse_absolute_timestamp
from agent.redact import redact_sensitive_text
from agent.retry_utils import reset_delay_from_message


def _normalize_error_context(error_context: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(error_context, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in ("reason", "message"):
        value = error_context.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    if "message" in normalized:
        # Providers echo the rejected key back ("Invalid API key: sk-..."), and this text lands in
        # last_error_message, which the auth.json sanitizer keeps (#71994). force=True: a
        # persistence boundary must not depend on security.redact_secrets.
        normalized["message"] = redact_sensitive_text(normalized["message"], force=True)
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
