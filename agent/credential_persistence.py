"""Credential-pool disk-boundary sanitization: strip raw secrets from *borrowed*
pool entries before they reach ``auth.json``, and never write a spent token pair
over a newer rotation. Deliberately free of ``hermes_cli.auth`` imports so the
pool model and the auth-store write boundary share one policy without import
cycles."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Mapping, Optional, Tuple


# Sources Hermes owns and may persist with secrets.  Any other non-empty,
# non-manual source is borrowed/reference-only so new external providers fail
# closed at the disk boundary.
_PERSISTABLE_PROVIDER_SOURCES = frozenset({
    ("anthropic", "hermes_pkce"),
    ("minimax-oauth", "oauth"),
    ("nous", "device_code"),
    ("openai-codex", "device_code"),
    ("xai-oauth", "device_code"),
})

# Metadata keys that look secret-ish by suffix but are safe to persist.
_SAFE_SECRETISH_METADATA_KEYS = frozenset({
    "secret_fingerprint", "secret_source", "token_type", "scope", "client_id",
    "agent_key_id", "agent_key_expires_at", "agent_key_expires_in",
    "agent_key_reused", "agent_key_obtained_at", "expires_at", "expires_at_ms",
    "expires_in", "last_refresh", "last_status", "last_status_at",
    "last_error_code", "last_error_reason", "last_error_message",
    "last_error_reset_at",
})

_SECRET_VALUE_KEYS = frozenset({
    "access_token", "refresh_token", "agent_key", "api_key", "apikey",
    "api_token", "auth_token", "authorization", "bearer_token", "client_secret",
    "credential", "credentials", "id_token", "oauth_token", "private_key",
    "secret_key", "session_token", "password", "secret", "token", "tokens",
})

_SECRET_VALUE_SUFFIXES = (
    "_api_key", "_api_token", "_access_token", "_auth_token", "_refresh_token",
    "_bearer_token", "_client_secret", "_id_token", "_oauth_token",
    "_private_key", "_session_token", "_secret_key", "_password", "_secret",
    "_token", "_key",
)

_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalize_key(key: Any) -> str:
    raw = _CAMEL_CASE_BOUNDARY.sub("_", str(key or "").strip())
    return raw.lower().replace("-", "_").replace(".", "_")


def is_borrowed_credential_source(source: Any, provider_id: Any = None) -> bool:
    """Return True when ``source`` points at a borrowed/reference-only secret."""
    normalized_source = str(source or "").strip().lower()
    if not normalized_source or normalized_source == "manual" or normalized_source.startswith("manual:"):
        return False
    normalized_provider = str(provider_id or "").strip().lower()
    return (normalized_provider, normalized_source) not in _PERSISTABLE_PROVIDER_SOURCES


def _is_secret_payload_key(key: Any) -> bool:
    normalized = _normalize_key(key)
    if not normalized or normalized in _SAFE_SECRETISH_METADATA_KEYS:
        return False
    return normalized in _SECRET_VALUE_KEYS or normalized.endswith(_SECRET_VALUE_SUFFIXES)


def fingerprint_secret_value(value: Any) -> str | None:
    """Non-reversible ``sha256:<16 hex>`` fingerprint of one secret value.

    Callers comparing a live secret against the ``secret_fingerprint`` left on
    a sanitized (borrowed) pool row need exactly the digest this module writes.
    """
    text = "" if value is None else str(value)
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"sha256:{digest[:16]}"


def _credential_secret_fingerprint(payload: Mapping[str, Any]) -> str | None:
    preferred = ("agent_key", "access_token", "refresh_token", "api_key", "token", "secret")
    candidates = [payload.get(k) for k in preferred]
    candidates += [v for k, v in payload.items() if _is_secret_payload_key(k)]
    for value in candidates:
        fingerprint = fingerprint_secret_value(value)
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

    Owned sources pass through unchanged.  Borrowed sources keep labels,
    source refs, status/cooldown metadata, counters and a fingerprint, but
    every raw secret value field is removed.
    """
    result = dict(payload)
    if not is_borrowed_credential_source(result.get("source"), provider_id):
        return result
    fingerprint = _credential_secret_fingerprint(result)
    sanitized = {k: v for k, v in result.items() if not _is_secret_payload_key(k)}
    if fingerprint:
        sanitized["secret_fingerprint"] = fingerprint
    return sanitized


# Fields one token refresh mints together; a row takes all of them from one side, never a mix.
_TOKEN_FIELDS = (
    "access_token", "refresh_token", "expires_at", "expires_at_ms", "expires_in", "obtained_at",
    "last_refresh", "agent_key", "agent_key_expires_at", "agent_key_expires_in", "agent_key_id",
    "agent_key_obtained_at", "agent_key_reused",
)


def token_pair(row: Any) -> Tuple[Any, Any]:
    """``(access_token, refresh_token)`` of a pool row; ``(None, None)`` for a non-mapping."""
    if not isinstance(row, Mapping):
        return (None, None)
    return (row.get("access_token"), row.get("refresh_token"))


def keep_rotated_disk_tokens(
    entry: Dict[str, Any], disk_entry: Any, base: Optional[Tuple[Any, Any]],
) -> Dict[str, Any]:
    """Keep the disk's token fields when its pair moved since the writer's *base*.

    Pools write back a whole in-memory snapshot. When another pool instance (a second cached
    session, another process) rotated this row's single-use refresh token after the snapshot was
    read, the snapshot's pair is already spent upstream, and writing it back loses the login.
    *base* is the pair the writer last read from or wrote to disk, so a disk pair that differs
    from it was minted by someone else, while a pair the writer minted itself still lands. A
    disk row without token material is never a rotation (a borrowed row keeps none on disk)."""
    disk_pair = token_pair(disk_entry)
    if base is None or not any(disk_pair) or disk_pair == base:
        return entry
    kept = {k: v for k, v in entry.items() if k not in _TOKEN_FIELDS}
    kept.update((k, disk_entry[k]) for k in _TOKEN_FIELDS if k in disk_entry)
    return kept
