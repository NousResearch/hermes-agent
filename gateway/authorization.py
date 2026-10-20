"""Fail-closed authorization primitives for gateway ingress and actions."""

from __future__ import annotations

import asyncio
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Protocol, runtime_checkable


class PrincipalKind(str, Enum):
    """Trusted origin of a verified principal."""

    USER = "user"
    SERVICE = "service"
    ON_BEHALF_OF = "on_behalf_of"


class PermissionDecision(str, Enum):
    """The only decisions exposed by the authorization boundary."""

    ALLOW = "allow"
    DENY = "deny"


_PRINCIPAL_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPrincipal:
    """Identity established by trusted gateway code, never by message payload."""

    principal_id: str
    kind: PrincipalKind
    authz_version: int
    scopes: frozenset[str]
    initiating_principal_id: str | None = None

    def __init__(
        self,
        *,
        _construction_token: object,
        principal_id: str,
        kind: PrincipalKind,
        authz_version: int,
        scopes: Iterable[str],
        initiating_principal_id: str | None = None,
    ) -> None:
        if _construction_token is not _PRINCIPAL_CONSTRUCTION_TOKEN:
            raise ValueError("VerifiedPrincipal requires trusted gateway construction")
        object.__setattr__(self, "principal_id", principal_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "authz_version", authz_version)
        object.__setattr__(self, "scopes", frozenset(scopes))
        object.__setattr__(self, "initiating_principal_id", initiating_principal_id)

    @classmethod
    def _create(
        cls,
        *,
        principal_id: str,
        kind: PrincipalKind,
        authz_version: int,
        scopes: Iterable[str],
        initiating_principal_id: str | None = None,
    ) -> VerifiedPrincipal:
        return cls(
            _construction_token=_PRINCIPAL_CONSTRUCTION_TOKEN,
            principal_id=principal_id,
            kind=kind,
            authz_version=authz_version,
            scopes=scopes,
            initiating_principal_id=initiating_principal_id,
        )


@dataclass(frozen=True, slots=True)
class AuthorizedEnvelope:
    """Authorization metadata kept separate from message and tool payloads."""

    principal: VerifiedPrincipal
    authz_version: int

    def __post_init__(self) -> None:
        if self.authz_version != self.principal.authz_version:
            raise ValueError("envelope authz_version must match its principal")


@runtime_checkable
class AuthorizationProvider(Protocol):
    """Backend contract used by fail-closed authorization callers."""

    async def readiness(self) -> bool:
        """Return whether the backend can safely answer authorization checks."""
        ...

    async def authorize(
        self,
        *,
        principal: VerifiedPrincipal,
        permission: str,
        resource: Mapping[str, object] | None = None,
    ) -> PermissionDecision:
        """Return ALLOW only for an explicitly authorized request."""
        ...


class UnavailableAuthorizationProvider:
    """Default provider used when no authorization backend is configured."""

    async def readiness(self) -> bool:
        return False

    async def authorize(
        self,
        *,
        principal: VerifiedPrincipal,
        permission: str,
        resource: Mapping[str, object] | None = None,
    ) -> PermissionDecision:
        return PermissionDecision.DENY


async def authorize_fail_closed(
    provider: AuthorizationProvider,
    *,
    principal: VerifiedPrincipal,
    permission: str,
    known_permissions: AbstractSet[str],
    resource: Mapping[str, object] | None = None,
    timeout_seconds: float = 1.0,
) -> PermissionDecision:
    """Authorize a known permission, reducing every failure to DENY.

    This boundary deliberately emits no logs. Callers may audit safe identifiers and
    reason codes, but must never log the protected message or tool payload here.
    """

    if permission not in known_permissions:
        return PermissionDecision.DENY
    try:
        ready = await asyncio.wait_for(
            provider.readiness(), timeout=timeout_seconds
        )
        if ready is not True:
            return PermissionDecision.DENY
        decision = await asyncio.wait_for(
            provider.authorize(
                principal=principal,
                permission=permission,
                resource=resource,
            ),
            timeout=timeout_seconds,
        )
    except Exception:
        return PermissionDecision.DENY
    return (
        PermissionDecision.ALLOW
        if decision is PermissionDecision.ALLOW
        else PermissionDecision.DENY
    )


def verified_user_principal(
    *,
    principal_id: str,
    authz_version: int,
    scopes: Iterable[str] = (),
) -> VerifiedPrincipal:
    """Create a verified user principal with a non-overridable USER kind."""

    return VerifiedPrincipal._create(
        principal_id=principal_id,
        kind=PrincipalKind.USER,
        authz_version=authz_version,
        scopes=scopes,
    )


def verified_service_principal(
    *,
    principal_id: str,
    authz_version: int,
    scopes: Iterable[str] = (),
) -> VerifiedPrincipal:
    """Create a verified service principal with a non-overridable SERVICE kind."""

    return VerifiedPrincipal._create(
        principal_id=principal_id,
        kind=PrincipalKind.SERVICE,
        authz_version=authz_version,
        scopes=scopes,
    )


def attenuated_on_behalf_of_principal(
    *,
    parent: VerifiedPrincipal,
    scopes: Iterable[str],
) -> VerifiedPrincipal:
    """Create a delegated user context whose scopes cannot exceed its parent."""

    delegated_scopes = frozenset(scopes)
    if not delegated_scopes.issubset(parent.scopes):
        raise ValueError("on-behalf-of scopes must be a subset of parent scopes")
    if parent.kind is PrincipalKind.SERVICE:
        raise ValueError("service principals cannot create on-behalf-of user contexts")
    initiating_principal_id = parent.initiating_principal_id or parent.principal_id
    return VerifiedPrincipal._create(
        principal_id=parent.principal_id,
        kind=PrincipalKind.ON_BEHALF_OF,
        authz_version=parent.authz_version,
        scopes=delegated_scopes,
        initiating_principal_id=initiating_principal_id,
    )
