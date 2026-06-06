"""Behavior-preserving gateway status-message filter seam."""
from __future__ import annotations

import re
from typing import Any

from gateway.runtime.delivery_redaction import redact_gateway_user_facing_secrets
from gateway.runtime.provider_error_sanitizer import (
    gateway_platform_value,
    gateway_provider_error_reply,
    looks_like_gateway_provider_error,
)

TELEGRAM_NOISY_STATUS_RE = re.compile(
    r"("  # transient/auxiliary status that should stay in logs, not Telegram chat
    r"auxiliary\s+.+\s+failed"
    r"|compression\s+summary\s+failed"
    r"|fallback\s+context\s+marker"
    r"|configured\s+compression\s+model\s+.+\s+failed"
    r"|no\s+auxiliary\s+llm\s+provider\s+configured"
    r"|auto-lowered\s+compression\s+threshold"
    r"|preflight\s+compression"
    r"|rate\s+limited\.\s+waiting\s+\d"
    r"|retrying\s+in\s+\d"
    r"|max\s+retries\s+\(\d+\).*(?:trying\s+fallback|exhausted|invalid\s+responses)"
    r"|stream\s+(?:drop|drop\s+mid\s+tool-call).+retry\s+\d"
    r"|stale\s+connections\s+from\s+a\s+previous\s+provider\s+issue"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def prepare_gateway_status_message(platform: Any, event_type: str, message: str) -> str | None:
    """Filter/sanitize agent status callbacks before platform delivery."""
    text = str(message or "").strip()
    if not text:
        return None
    if gateway_platform_value(platform) != "telegram":
        return text

    text = redact_gateway_user_facing_secrets(text)
    if TELEGRAM_NOISY_STATUS_RE.search(text):
        return None
    if looks_like_gateway_provider_error(text):
        return gateway_provider_error_reply(text)
    return text
