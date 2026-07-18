"""Deterministic redaction helpers for Truth Ledger pre-persistence gates."""

from __future__ import annotations

from typing import Any

from agent.redact import redact_sensitive_text

_DROP_KEYS = {
    "conversation_history",
    "raw_tool_output",
    "tool_output_raw",
    "assistant_reasoning",
    "chain_of_thought",
    "cot",
}


def redact_text(text: str) -> tuple[str, bool]:
    """Redact text and report whether sensitive content was detected."""
    redacted = redact_sensitive_text(text, force=True)
    return redacted, redacted != text


def contains_sensitive_material(text: str) -> bool:
    """True when redaction alters the text, indicating sensitive data."""
    _, changed = redact_text(text)
    return changed


def sanitize_payload(value: Any) -> Any:
    """Recursively remove disallowed raw fields and redact sensitive strings."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in _DROP_KEYS:
                continue
            sanitized[key] = sanitize_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str):
        redacted, _ = redact_text(value)
        return redacted
    return value
