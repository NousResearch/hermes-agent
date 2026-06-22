"""Shared redaction helpers for Memory v2.

These helpers are intentionally conservative. Memory v2 stores raw evidence and
retrieval logs locally, so every persistence boundary should redact common
credential shapes even when the caller already tried to sanitize input.
"""

from __future__ import annotations

import re
from typing import Any

REDACTION = "[REDACTED]"
SENSITIVE_QUERY = "[REDACTED sensitive query]"

_PRIVATE_KEY_RE = re.compile(r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----")
_URI_CREDENTIAL_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s:/?#]+:)([^\s@/?#]+)(@)")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_TOKEN_PREFIX_RE = re.compile(
    r"(?i)\b(gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{8,}|sk-[A-Za-z0-9][A-Za-z0-9_-]{8,}|sk-proj-[A-Za-z0-9_-]{8,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16})\b"
)
_LABELED_SECRET_RE = re.compile(
    r"(?is)"
    r"("
    r"(?:authorization\s*:\s*bearer\s+)|"
    r"(?:bearer\s+)|"
    r"(?:[A-Z0-9_]*(?:API[_-]?KEY|PRIVATE[_-]?KEY|SECRET[_-]?ACCESS[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|CLIENT[_-]?SECRET|TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIAL)\s*(?:=|:|is)?\s*)|"
    r"(?:api[_ -]?key\s*(?:=|:|is)?\s*)|"
    r"(?:private\s+key\s*(?:=|:|is)?\s*)|"
    r"(?:client\s+secret\s*(?:=|:|is)?\s*)|"
    r"(?:password\s*(?:=|:|is)?\s*)|"
    r"(?:passwd\s*(?:=|:|is)?\s*)|"
    r"(?:token\s*(?:=|:|is)?\s*)|"
    r"(?:secret\s*(?:=|:|is)?\s*)|"
    r"(?:credential\s*(?:=|:|is)?\s*)|"
    r"(?:(?:openai|anthropic|github|gitlab|aws|azure|google|gcp|slack|discord|stripe|huggingface|hf)\s+(?:api\s+)?key\s*(?:=|:|is)?\s*)"
    r")"
    r"([^\s,;\]\}\)]+)"
)
_HIGH_ENTROPY_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*\s*(?:=|:|is)\s*)([^\s,;\]\}\)]+)"
)


def redact_text(text: str) -> str:
    """Redact common credential forms from text."""

    redacted = str(text or "")
    redacted = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", redacted)
    redacted = _URI_CREDENTIAL_RE.sub(lambda match: f"{match.group(1)}{REDACTION}{match.group(3)}", redacted)
    redacted = _LABELED_SECRET_RE.sub(lambda match: match.group(0) if match.group(2).startswith(REDACTION) else f"{match.group(1)}{REDACTION}", redacted)
    redacted = _HIGH_ENTROPY_ASSIGNMENT_RE.sub(lambda match: match.group(0) if match.group(2).startswith(REDACTION) else f"{match.group(1)}{REDACTION}", redacted)
    redacted = _TOKEN_PREFIX_RE.sub(REDACTION, redacted)
    redacted = _JWT_RE.sub(REDACTION, redacted)
    return redacted


def contains_sensitive_text(text: str) -> bool:
    """Return true if text appears to contain a credential-like value."""

    raw = str(text or "")
    return redact_text(raw) != raw


def redacted_query_for_log(query: str) -> str:
    """Return a safe retrieval-log query string."""

    text = str(query or "")
    if contains_sensitive_text(text):
        return SENSITIVE_QUERY
    return text[:500]


def redacted_query_hash_input(query: str) -> str:
    """Return canonical query text to hash for retrieval logs.

    Sensitive queries intentionally hash the redacted sentinel, not the raw
    secret-bearing string, so low-entropy secret guesses cannot be verified by
    comparing hashes in exported logs.
    """

    return redacted_query_for_log(query)


def redact_data(value: Any) -> Any:
    """Recursively redact strings in JSON/YAML-like data."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    if isinstance(value, dict):
        return {key: redact_data(item) for key, item in value.items()}
    return value
