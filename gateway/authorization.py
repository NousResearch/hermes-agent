"""Fail-closed authorization primitives for gateway ingress and actions."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Iterable, Mapping, Set as AbstractSet
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Never, Protocol, TypeVar, runtime_checkable


_MAX_IDENTITY_LENGTH = 256
_MAX_SCOPE_LENGTH = 256
_T = TypeVar("_T")


class _DeadlineExpired:
    __slots__ = ()


_DEADLINE_EXPIRED = _DeadlineExpired()


class PrincipalKind(str, Enum):
    """Trusted origin of a verified principal."""

    USER = "user"
    SERVICE = "service"
    ON_BEHALF_OF = "on_behalf_of"


class PermissionDecision(str, Enum):
    """The only decisions exposed by the authorization boundary."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPrincipal:
    """Identity established by a trusted issuer, never by message payload."""

    principal_id: str
    kind: PrincipalKind
    authz_version: int
    scopes: frozenset[str]
    initiating_principal_id: str | None = None

    def __init__(self) -> None:
        raise TypeError("VerifiedPrincipal requires a trusted issuer capability")


class TrustedPrincipalIssuer:
    """Non-serializable capability held and injected only by core bootstrap.

    This is an in-process trust boundary, not a Python sandbox: arbitrary malicious
    code already executing in this interpreter can subvert Python objects. Model,
    message, and tool payloads cannot mint a principal unless trusted core code gives
    them this capability. Never register an issuer as a tool or plugin API.
    """

    __slots__ = ()

    def __new__(cls) -> TrustedPrincipalIssuer:
        raise TypeError("issuer capabilities are created only by core bootstrap")

    def __reduce__(self) -> Never:
        raise TypeError("issuer capabilities cannot be serialized")

    def issue_user(
        self,
        *,
        principal_id: str,
        authz_version: int,
        scopes: Iterable[str] = (),
    ) -> VerifiedPrincipal:
        return self.__issue(
            principal_id=principal_id,
            kind=PrincipalKind.USER,
            authz_version=authz_version,
            scopes=scopes,
        )

    def issue_service(
        self,
        *,
        principal_id: str,
        authz_version: int,
        scopes: Iterable[str] = (),
    ) -> VerifiedPrincipal:
        return self.__issue(
            principal_id=principal_id,
            kind=PrincipalKind.SERVICE,
            authz_version=authz_version,
            scopes=scopes,
        )

    def issue_on_behalf_of(
        self,
        *,
        parent: VerifiedPrincipal,
        scopes: Iterable[str],
    ) -> VerifiedPrincipal:
        if not isinstance(parent, VerifiedPrincipal):
            raise TypeError("parent must be a VerifiedPrincipal")
        delegated_scopes = _validated_scopes(scopes)
        if not delegated_scopes.issubset(parent.scopes):
            raise ValueError("on-behalf-of scopes must be a subset of parent scopes")
        if parent.kind is PrincipalKind.SERVICE:
            raise ValueError("service principals cannot create on-behalf-of user contexts")
        return self.__issue(
            principal_id=parent.principal_id,
            kind=PrincipalKind.ON_BEHALF_OF,
            authz_version=parent.authz_version,
            scopes=delegated_scopes,
            initiating_principal_id=(
                parent.initiating_principal_id or parent.principal_id
            ),
        )

    def __issue(
        self,
        *,
        principal_id: str,
        kind: PrincipalKind,
        authz_version: int,
        scopes: Iterable[str],
        initiating_principal_id: str | None = None,
    ) -> VerifiedPrincipal:
        principal_id = _validated_text(
            principal_id, field_name="principal_id", max_length=_MAX_IDENTITY_LENGTH
        )
        if type(authz_version) is not int:
            raise TypeError("authz_version must be an exact int")
        if authz_version < 0:
            raise ValueError("authz_version must be non-negative")
        validated_scopes = _validated_scopes(scopes)
        if kind is PrincipalKind.ON_BEHALF_OF:
            if initiating_principal_id is None:
                raise ValueError("on-behalf-of principals require an initiating identity")
            initiating_principal_id = _validated_text(
                initiating_principal_id,
                field_name="initiating_principal_id",
                max_length=_MAX_IDENTITY_LENGTH,
            )
        elif initiating_principal_id is not None:
            raise ValueError("direct principals cannot have an initiating identity")
        principal = object.__new__(VerifiedPrincipal)
        object.__setattr__(principal, "principal_id", principal_id)
        object.__setattr__(principal, "kind", kind)
        object.__setattr__(principal, "authz_version", authz_version)
        object.__setattr__(principal, "scopes", validated_scopes)
        object.__setattr__(
            principal, "initiating_principal_id", initiating_principal_id
        )
        return principal


def _validated_text(value: object, *, field_name: str, max_length: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact str")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty and non-whitespace")
    if len(value) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length {max_length}")
    return value


def _validated_scopes(scopes: object) -> frozenset[str]:
    if isinstance(scopes, (str, bytes, Mapping)) or not isinstance(scopes, Iterable):
        raise TypeError("scopes must be an iterable of strings, not text or a mapping")
    return frozenset(
        _validated_text(scope, field_name="scope", max_length=_MAX_SCOPE_LENGTH)
        for scope in scopes
    )


def _bootstrap_principal_issuer() -> TrustedPrincipalIssuer:
    """Create the capability for core bootstrap dependency injection only."""

    return object.__new__(TrustedPrincipalIssuer)


@dataclass(frozen=True, slots=True)
class AuthorizedEnvelope:
    """Authorization metadata kept separate from message and tool payloads."""

    principal: VerifiedPrincipal
    authz_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.principal, VerifiedPrincipal):
            raise TypeError("envelope principal must be a VerifiedPrincipal")
        if type(self.authz_version) is not int:
            raise TypeError("envelope authz_version must be an exact int")
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
    timeout_seconds: int | float = 1.0,
) -> PermissionDecision:
    """Authorize a known permission, reducing every failure to DENY.

    ``timeout_seconds`` must be a finite exact ``int`` or ``float`` (not ``bool``)
    greater than or equal to zero. Invalid deadlines deny before provider methods run.

    This boundary deliberately emits no logs. Callers may audit safe identifiers and
    reason codes, but must never log the protected message or tool payload here.
    """

    if permission not in known_permissions:
        return PermissionDecision.DENY
    deadline_seconds = _validated_timeout_seconds(timeout_seconds)
    if deadline_seconds is None:
        return PermissionDecision.DENY
    try:
        ready = await _await_with_deadline(provider.readiness(), deadline_seconds)
        if ready is _DEADLINE_EXPIRED or ready is not True:
            return PermissionDecision.DENY
        decision = await _await_with_deadline(
            provider.authorize(
                principal=principal,
                permission=permission,
                resource=resource,
            ),
            deadline_seconds,
        )
    except Exception:
        return PermissionDecision.DENY
    return (
        PermissionDecision.ALLOW
        if decision is PermissionDecision.ALLOW
        else PermissionDecision.DENY
    )


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


def _validated_timeout_seconds(value: object) -> int | float | None:
    if type(value) is int:
        return value if value >= 0 else None
    if type(value) is float and math.isfinite(value) and value >= 0:
        return value
    return None


async def _await_with_deadline(
    awaitable: Awaitable[_T], timeout_seconds: object
) -> _T | _DeadlineExpired:
    """Race an operation against a deadline without trusting cancellation."""

    deadline_seconds = _validated_timeout_seconds(timeout_seconds)
    if deadline_seconds is None:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        return _DEADLINE_EXPIRED

    operation = asyncio.ensure_future(awaitable)
    deadline = asyncio.create_task(asyncio.sleep(deadline_seconds))
    try:
        done, _pending = await asyncio.wait(
            (operation, deadline), return_when=asyncio.FIRST_COMPLETED
        )
    except BaseException:
        operation.cancel()
        deadline.cancel()
        operation.add_done_callback(_consume_task_result)
        deadline.add_done_callback(_consume_task_result)
        raise

    if deadline in done:
        operation.cancel()
        operation.add_done_callback(_consume_task_result)
        _consume_task_result(deadline)
        return _DEADLINE_EXPIRED

    deadline.cancel()
    deadline.add_done_callback(_consume_task_result)
    try:
        return operation.result()
    except asyncio.CancelledError:
        return _DEADLINE_EXPIRED
