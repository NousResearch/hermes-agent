"""Keychain-only storage for broker OAuth grants.

One generic-password item per account alias under
``ai.hermes.oauth-broker.openai-codex``. No ordinary credential file is
ever created; the only serialization target is the Keychain item payload.
"""

from __future__ import annotations

import json
from typing import Optional

from agent.keychain_secret import (
    KeychainBackend,
    KeychainError,
    KeychainNotFound,
    KeychainRef,
    delete_keychain_secret,
    read_keychain_secret,
    write_keychain_secret,
)
from agent.oauth_broker.models import (
    ACCOUNT_ALIASES,
    GRANT_SCHEMA_VERSION,
    OAUTH_GRANT_KEYCHAIN_SERVICE,
    REDACTED,
    GrantStoreError,
    OAuthGrant,
)

_GRANT_FIELDS = ("access_token", "refresh_token", "expires_at", "account_id")
_PAYLOAD_FIELDS = frozenset(("schema_version", *_GRANT_FIELDS))


class _DuplicateGrantField(ValueError):
    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(field)


def _strict_object_pairs(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateGrantField(str(key))
        payload[key] = value
    return payload


def grant_to_payload(grant: OAuthGrant) -> str:
    return json.dumps(
        {
            "schema_version": GRANT_SCHEMA_VERSION,
            "access_token": grant.access_token,
            "refresh_token": grant.refresh_token,
            "expires_at": grant.expires_at,
            "account_id": grant.account_id,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def grant_from_payload(alias: str, raw: str) -> OAuthGrant:
    # ``from None`` everywhere: chained JSON/validation errors could embed
    # payload fragments (token substrings) in the traceback.
    try:
        payload = json.loads(raw, object_pairs_hook=_strict_object_pairs)
    except _DuplicateGrantField as exc:
        raise GrantStoreError(
            alias=alias,
            category="duplicate_field",
            detail=f"duplicate field {exc.field!r} ({REDACTED})",
        ) from None
    except (TypeError, ValueError):
        raise GrantStoreError(
            alias=alias,
            category="invalid_payload",
            detail=f"stored payload is not JSON ({REDACTED})",
        ) from None
    if not isinstance(payload, dict):
        raise GrantStoreError(
            alias=alias,
            category="invalid_payload",
            detail=f"stored payload is not an object ({REDACTED})",
        )
    version = payload.get("schema_version")
    if type(version) is not int or version != GRANT_SCHEMA_VERSION:
        raise GrantStoreError(
            alias=alias,
            category="schema_version",
            detail=f"expected schema_version {GRANT_SCHEMA_VERSION}",
        )
    unknown_fields = set(payload) - _PAYLOAD_FIELDS
    if unknown_fields:
        key = sorted(str(field) for field in unknown_fields)[0]
        raise GrantStoreError(
            alias=alias,
            category="unknown_field",
            detail=f"unknown field {key!r} with value {REDACTED}",
        )
    try:
        return OAuthGrant(
            access_token=payload.get("access_token"),
            refresh_token=payload.get("refresh_token"),
            expires_at=payload.get("expires_at"),
            account_id=payload.get("account_id"),
        )
    except ValueError as exc:
        # OAuthGrant validation messages already redact values.
        raise GrantStoreError(
            alias=alias, category="invalid_grant_payload", detail=str(exc)
        ) from None


class KeychainGrantStore:
    """Load/replace/delete OAuth grants for aliases A, B, C."""

    def __init__(
        self,
        *,
        backend: Optional[KeychainBackend] = None,
        service: str = OAUTH_GRANT_KEYCHAIN_SERVICE,
    ) -> None:
        self._backend = backend
        self._service = service

    def _ref(self, alias) -> KeychainRef:
        if alias not in ACCOUNT_ALIASES:
            raise GrantStoreError(
                alias=alias,
                category="invalid_alias",
                detail=f"alias must be one of {ACCOUNT_ALIASES}",
            )
        return KeychainRef(service=self._service, account=alias)

    def load(self, alias: str) -> OAuthGrant:
        ref = self._ref(alias)
        try:
            raw = read_keychain_secret(ref, backend=self._backend)
        except KeychainNotFound:
            raise GrantStoreError(
                alias=alias, category="not_found", detail="no grant provisioned"
            ) from None
        except KeychainError as exc:
            raise GrantStoreError(alias=alias, category=exc.category) from exc
        return grant_from_payload(alias, raw)

    def replace(self, alias: str, grant: OAuthGrant) -> None:
        """Atomically overwrite the alias's single Keychain item."""
        ref = self._ref(alias)
        try:
            write_keychain_secret(ref, grant_to_payload(grant), backend=self._backend)
        except KeychainError as exc:
            raise GrantStoreError(alias=alias, category=exc.category) from exc

    def delete(self, alias: str) -> None:
        ref = self._ref(alias)
        try:
            delete_keychain_secret(ref, backend=self._backend)
        except KeychainNotFound:
            raise GrantStoreError(
                alias=alias, category="not_found", detail="no grant provisioned"
            ) from None
        except KeychainError as exc:
            raise GrantStoreError(alias=alias, category=exc.category) from exc


__all__ = [
    "KeychainGrantStore",
    "grant_from_payload",
    "grant_to_payload",
]
