"""Minimize untrusted local material before auxiliary model egress."""

from __future__ import annotations

import re
from typing import Any

from agent.redact import redact_sensitive_text

_PRIVATE_LOCATIONS = (
    re.compile(r"/(?:home|Users)/[^\s\"'<>]+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"\b[a-z0-9.-]+\.(?:internal|corp|local|lan)\b", re.IGNORECASE),
    re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2,3}\b"),
)


def model_safe_text(value: str) -> str:
    value = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
        "[REDACTED PRIVATE KEY]", value, flags=re.DOTALL,
    )
    # If a secret is detected, remove the complete line, not a masked prefix
    # and suffix intended for diagnostic logs.
    lines = []
    for line in value.splitlines(keepends=True):
        redacted = redact_sensitive_text(line, force=True, redact_url_credentials=True)
        lines.append("[REDACTED]\n" if redacted != line else line)
    text = "".join(lines)
    for pattern in _PRIVATE_LOCATIONS:
        text = pattern.sub("[PRIVATE LOCATION]", text)
    return text


def model_safe_data(value: Any) -> Any:
    if isinstance(value, str):
        return model_safe_text(value)
    if isinstance(value, list):
        return [model_safe_data(item) for item in value]
    if isinstance(value, dict):
        return {model_safe_text(str(key)): model_safe_data(item) for key, item in value.items()}
    return value
