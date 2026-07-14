"""Broker account constants, grant model, and redacted error types.

Every error and repr in this module is safe to log: identifiers and
categories only, secret values replaced with ``[REDACTED]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

ACCOUNT_ALIASES = ("A", "B", "C")

OAUTH_GRANT_KEYCHAIN_SERVICE = "ai.hermes.oauth-broker.openai-codex"
CLIENT_KEY_KEYCHAIN_SERVICE = "ai.hermes.oauth-broker.client"
CLIENT_KEY_KEYCHAIN_ACCOUNT = "local"

GRANT_SCHEMA_VERSION = 1

REDACTED = "[REDACTED]"


class GrantStoreError(RuntimeError):
    """Grant store failure exposing alias and category, never token text."""

    def __init__(self, *, alias, category: str, detail: str = "") -> None:
        self.alias = alias
        self.category = category
        message = f"oauth grant store {category} for account {alias!r}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


def _require_non_empty_str(name: str, value) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"OAuthGrant.{name} must be a non-empty string, got {REDACTED}"
        )


@dataclass(frozen=True)
class OAuthGrant:
    """One account's Codex OAuth grant as stored in the Keychain item."""

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float
    account_id: str

    def __post_init__(self) -> None:
        _require_non_empty_str("access_token", self.access_token)
        _require_non_empty_str("refresh_token", self.refresh_token)
        _require_non_empty_str("account_id", self.account_id)
        if isinstance(self.expires_at, bool) or not isinstance(
            self.expires_at, (int, float)
        ):
            raise ValueError(
                f"OAuthGrant.expires_at must be an epoch number, got {REDACTED}"
            )
        expires_at = float(self.expires_at)
        if not math.isfinite(expires_at):
            raise ValueError(
                f"OAuthGrant.expires_at must be finite, got {REDACTED}"
            )
        object.__setattr__(self, "expires_at", expires_at)


@dataclass(frozen=True)
class RedactedAccountStatus:
    """Loggable per-account state: identifiers, booleans, timestamps, and a
    one-way fingerprint — never token material."""

    alias: str
    present: bool
    healthy: bool
    terminal_category: Optional[str]
    expires_at: Optional[float]
    last_refresh_at: Optional[float]
    last_refresh_result: Optional[str]
    access_token_fingerprint: Optional[str]
    # True while a rotated grant lives only in memory because the Keychain
    # write keeps failing (design §八.7). Requires re-OAuth if the process
    # dies before persistence succeeds.
    persistence_degraded: bool = False


__all__ = [
    "ACCOUNT_ALIASES",
    "CLIENT_KEY_KEYCHAIN_ACCOUNT",
    "CLIENT_KEY_KEYCHAIN_SERVICE",
    "GRANT_SCHEMA_VERSION",
    "OAUTH_GRANT_KEYCHAIN_SERVICE",
    "REDACTED",
    "GrantStoreError",
    "OAuthGrant",
    "RedactedAccountStatus",
]
