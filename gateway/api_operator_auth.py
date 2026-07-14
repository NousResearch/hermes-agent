"""Hermes operator authorization vocabulary.

Defines the scope domains, scope normalization, and the credential/principal
types shared by the canonical API server's operator-token authentication and
credential-issuance paths (pairing, storage, and per-request scope checks).

Kept independent of aiohttp so it can be imported by storage and enrollment
modules without pulling in the HTTP server.
"""

from dataclasses import dataclass
from typing import Iterable

VALID_SCOPE_DOMAINS = frozenset({
    "chat", "sessions", "profiles", "providers", "skills",
    "tools", "memory", "tasks", "gateway", "settings",
})


def normalize_scopes(values: Iterable[str]) -> tuple[str, ...]:
    scopes = {str(value).strip().lower() for value in values if str(value).strip()}
    if "*" in scopes:
        return ("*",)
    for scope in scopes:
        domain, separator, access = scope.partition(":")
        if separator != ":" or domain not in VALID_SCOPE_DOMAINS or access not in {"read", "write"}:
            raise ValueError(f"unknown scope: {scope}")
    return tuple(sorted(scopes))


@dataclass(frozen=True)
class AuthPrincipal:
    credential_id: str
    scopes: tuple[str, ...]
    is_superuser: bool = False

    def allows(self, required: str) -> bool:
        return self.is_superuser or "*" in self.scopes or required in self.scopes


@dataclass(frozen=True)
class IssuedCredential:
    credential_id: str
    token: str
    label: str
    scopes: tuple[str, ...]
    created_at: float


@dataclass(frozen=True)
class CredentialSummary:
    credential_id: str
    label: str
    scopes: tuple[str, ...]
    created_at: float
    revoked_at: float | None
