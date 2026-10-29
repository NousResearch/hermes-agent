"""Timestamp parsing shared by credential-pool status, refresh, and cooldown bookkeeping."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse of epoch seconds / epoch ms / ISO-8601 into epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0:
            return None
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            return None
    return None


def token_expires_at(entry: PooledCredential) -> Optional[float]:
    """When a pool entry's access token expires (epoch seconds), from the row's own expiry fields
    or, for a JWT (Codex), its ``exp`` claim; None when nothing says. No network, no refresh."""
    for value in (entry.expires_at_ms, entry.expires_at):
        expires = _parse_absolute_timestamp(value)
        if expires is not None:
            return expires
    from hermes_cli.auth_constants import _decode_jwt_claims

    return _parse_absolute_timestamp(_decode_jwt_claims(entry.access_token).get("exp"))
