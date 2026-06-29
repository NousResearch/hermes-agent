"""Credential-pool disk-boundary sanitization helpers.

These helpers define which credential-pool entries are references to borrowed
runtime secrets and strip raw values before those entries are written to
``auth.json``.  They intentionally have no dependency on ``hermes_cli.auth`` so
both the pool model and the final auth-store write boundary can share the same
policy without import cycles.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Mapping


# Sources Hermes owns and can intentionally persist in auth.json.  Everything
# else with a non-empty source is treated as borrowed/reference-only by default
# so future external secret providers fail closed at the disk boundary.
_PERSISTABLE_PROVIDER_SOURCES = frozenset({
    ("anthropic", "hermes_pkce"),
    ("minimax-oauth", "oauth"),
    ("nous", "device_code"),
    ("openai-codex", "device_code"),
    ("xai-oauth", "loopback_pkce"),
})

_SAFE_SECRETISH_METADATA_KEYS = frozenset({
    "secret_fingerprint",
    "secret_source",
    "token_type",
    "scope",
    "client_id",
    "agent_key_id",
    "agent_key_expires_at",
    "agent_key_expires_in",
    "agent_key_reused",
    "agent_key_obtained_at",
    "expires_at",
    "expires_at_ms",
    "expires_in",
    "last_refresh",
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
})

_SECRET_VALUE_KEYS = frozenset({
    "access_token",
    "refresh_token",
    "agent_key",
    "api_key",
    "apikey",
    "api_token",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "id_token",
    "oauth_token",
    "private_key",
    "secret_key",
    "session_token",
    "password",
    "secret",
    "token",
    "tokens",
})

_SECRET_VALUE_SUFFIXES = (
    "_api_key",
    "_api_token",
    "_access_token",
    "_auth_token",
    "_refresh_token",
    "_bearer_token",
    "_client_secret",
    "_id_token",
    "_oauth_token",
    "_private_key",
    "_session_token",
    "_secret_key",
    "_password",
    "_secret",
    "_token",
    "_key",
)

_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_COMMAND_API_KEY_HEAD_RE = re.compile(
    r"^\s*(?:[$>]\s*)?(?:hermes|openclaw)\b",
    re.IGNORECASE,
)
_COMMAND_API_KEY_AUTH_MARKER_RE = re.compile(
    r"(?:^|\s)(?:auth|setup|model|portal|onboard)\b"
    r"|(?:^|\s)--(?:api-key|auth-choice|auth-type|type|provider|model|label|non-interactive)\b",
    re.IGNORECASE,
)


def _normalize_key(key: Any) -> str:
    raw = str(key or "").strip()
    raw = _CAMEL_CASE_BOUNDARY.sub("_", raw)
    return raw.lower().replace("-", "_").replace(".", "_")


def is_borrowed_credential_source(source: Any, provider_id: Any = None) -> bool:
    """Return True when ``source`` points at a borrowed/reference-only secret."""
    normalized_source = str(source or "").strip().lower()
    if not normalized_source:
        return False
    if normalized_source == "manual" or normalized_source.startswith("manual:"):
        return False
    normalized_provider = str(provider_id or "").strip().lower()
    return (normalized_provider, normalized_source) not in _PERSISTABLE_PROVIDER_SOURCES


def is_command_shaped_api_key(value: Any) -> bool:
    """Return True when an API-key input is clearly a copied CLI command.

    This deliberately stays narrow: it only rejects Hermes/OpenClaw command
    lines that also contain auth/setup/model/onboard markers or auth flags.
    Normal secrets that merely contain words like "hermes" remain accepted.
    """
    text = str(value or "").strip()
    if not text:
        return False
    return bool(
        _COMMAND_API_KEY_HEAD_RE.search(text)
        and _COMMAND_API_KEY_AUTH_MARKER_RE.search(text)
    )


def _is_secret_payload_key(key: Any) -> bool:
    normalized = _normalize_key(key)
    if not normalized or normalized in _SAFE_SECRETISH_METADATA_KEYS:
        return False
    if normalized in _SECRET_VALUE_KEYS:
        return True
    return normalized.endswith(_SECRET_VALUE_SUFFIXES)


def _fingerprint_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"sha256:{digest[:16]}"


def _credential_secret_fingerprint(payload: Mapping[str, Any]) -> str | None:
    for key in ("agent_key", "access_token", "refresh_token", "api_key", "token", "secret"):
        fingerprint = _fingerprint_value(payload.get(key))
        if fingerprint:
            return fingerprint

    for key, value in payload.items():
        if _is_secret_payload_key(key):
            fingerprint = _fingerprint_value(value)
            if fingerprint:
                return fingerprint

    existing = payload.get("secret_fingerprint")
    if isinstance(existing, str) and existing.startswith("sha256:"):
        return existing
    return None


def sanitize_borrowed_credential_payload(
    payload: Mapping[str, Any],
    provider_id: Any = None,
) -> Dict[str, Any]:
    """Return a disk-safe credential-pool payload.

    Owned sources (manual entries and Hermes-owned OAuth/device-code state)
    pass through unchanged.  Borrowed/reference-only sources keep labels,
    source refs, status/cooldown metadata, counters, and a non-reversible
    fingerprint, but raw secret value fields are removed.
    """
    result = dict(payload)
    if not is_borrowed_credential_source(result.get("source"), provider_id):
        return result

    fingerprint = _credential_secret_fingerprint(result)
    sanitized = {
        key: value
        for key, value in result.items()
        if not _is_secret_payload_key(key)
    }
    if fingerprint:
        sanitized["secret_fingerprint"] = fingerprint
    return sanitized
