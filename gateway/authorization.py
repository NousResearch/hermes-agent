"""Fail-closed authorization primitives for gateway ingress and actions."""

from __future__ import annotations

import asyncio
import math
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Never, Protocol, TypeVar, runtime_checkable


_MAX_IDENTITY_LENGTH = 256
_MAX_SCOPE_LENGTH = 256
# Permissions are protocol identifiers, not payloads. 256 characters leaves ample
# room for namespaced identifiers while bounding attacker-controlled input.
_MAX_PERMISSION_LENGTH = 256
# The permission catalog is trusted configuration, but still crosses this public
# boundary. Exact built-in sets make length and iteration finite and side-effect
# free; the cap bounds validation work before any provider access.
_MAX_KNOWN_PERMISSIONS = 512
# Keep authorization deadlines bounded to values every supported event loop can
# represent operationally, and to a duration appropriate for an ingress check.
_MAX_AUTHORIZATION_TIMEOUT_SECONDS = 60
_QUARANTINE_POLL_INTERVAL_SECONDS = 0.01
_T = TypeVar("_T")
_NATIVE_ASYNCIO_TASK_TYPE = asyncio.Task
_NATIVE_ASYNCIO_TASK = asyncio.Task
_QUARANTINED_AUTHORIZATION_TASKS: set[asyncio.Task[Any]] = set()
_AUTHORIZATION_REAPER_TASKS: set[asyncio.Task[Any]] = set()
_AUTHORIZATION_POLL_HANDLES: dict[
    asyncio.Task[Any], asyncio.TimerHandle
] = {}


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
        if type(parent) is not VerifiedPrincipal:
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
        if type(self.principal) is not VerifiedPrincipal:
            raise TypeError("envelope principal must be a VerifiedPrincipal")
        if type(self.authz_version) is not int:
            raise TypeError("envelope authz_version must be an exact int")
        if self.authz_version != self.principal.authz_version:
            raise ValueError("envelope authz_version must match its principal")


@runtime_checkable
class AuthorizationProvider(Protocol):
    """Trusted first-party, in-process authorization backend contract.

    Methods must return native coroutines created for the current event loop and
    perform only nonblocking work there. Implementations must be
    cancellation-cooperative or independently bounded; providers that indefinitely
    suppress cancellation are outside this contract. This in-process boundary cannot
    guarantee availability against malicious code executing in the same interpreter.
    """

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
    known_permissions: set[str] | frozenset[str],
    resource: Mapping[str, object] | None = None,
    timeout_seconds: int | float = 1.0,
) -> PermissionDecision:
    """Authorize a known permission, reducing every failure to DENY.

    The principal must be a ``VerifiedPrincipal`` and the permission must be an
    exact, non-whitespace ``str`` of at most 256 characters. ``known_permissions``
    must be an exact built-in ``set`` or ``frozenset`` (subclasses are rejected),
    contain no more than 512 exact valid strings, and include ``permission``.
    Invalid inputs deny before provider methods run.

    ``timeout_seconds`` must be a finite exact ``int`` or ``float`` (not ``bool``)
    between zero and 60 seconds, inclusive. Invalid deadlines also deny before
    provider methods run.

    Provider methods are trusted first-party, nonblocking, current event loop code.
    They must be cancellation-cooperative or independently bounded. Cancellation is
    cooperative in Python, so finite cancellation suppression is strongly tracked in
    a private quarantine until completion; the eventual result is consumed and can
    never authorize a request that already returned or was cancelled. Providers that
    indefinitely suppress cancellation are outside the contract. Malicious code in
    the same interpreter can still defeat availability; this is not a Python sandbox.

    If the host event loop rejects the final internal cleanup timer after callback
    registration and native reaper construction both fail, the task remains strongly
    quarantined as a fail-safe. That broken-loop condition is a host-runtime failure
    outside the provider contract.

    This boundary deliberately emits no logs. Callers may audit safe identifiers and
    reason codes, but must never log the protected message or tool payload here.
    """

    if type(principal) is not VerifiedPrincipal:
        return PermissionDecision.DENY
    if not _is_valid_permission(permission):
        return PermissionDecision.DENY
    if type(known_permissions) not in (set, frozenset):
        return PermissionDecision.DENY
    if len(known_permissions) > _MAX_KNOWN_PERMISSIONS:
        return PermissionDecision.DENY
    if not all(_is_valid_permission(item) for item in known_permissions):
        return PermissionDecision.DENY
    if permission not in known_permissions:
        return PermissionDecision.DENY
    deadline_seconds = _validated_timeout_seconds(timeout_seconds)
    if deadline_seconds is None:
        return PermissionDecision.DENY
    try:
        readiness_coroutine = provider.readiness()
        if type(readiness_coroutine) is not types.CoroutineType:
            return PermissionDecision.DENY
        ready = await _await_with_deadline(readiness_coroutine, deadline_seconds)
        if ready is _DEADLINE_EXPIRED or ready is not True:
            return PermissionDecision.DENY
        authorization_coroutine = provider.authorize(
            principal=principal,
            permission=permission,
            resource=resource,
        )
        if type(authorization_coroutine) is not types.CoroutineType:
            return PermissionDecision.DENY
        decision = await _await_with_deadline(
            authorization_coroutine,
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
    except BaseException:
        pass


def _is_valid_permission(value: object) -> bool:
    return (
        type(value) is str
        and bool(value.strip())
        and len(value) <= _MAX_PERMISSION_LENGTH
    )


def _discard_quarantined_task(task: asyncio.Task[Any]) -> None:
    handle = _AUTHORIZATION_POLL_HANDLES.pop(task, None)
    if handle is not None:
        handle.cancel()
    _QUARANTINED_AUTHORIZATION_TASKS.discard(task)


def _authorization_task_done(task: asyncio.Task[Any]) -> bool:
    return _NATIVE_ASYNCIO_TASK_TYPE.done(task)


def _consume_authorization_task_result(task: asyncio.Task[Any]) -> None:
    try:
        _NATIVE_ASYNCIO_TASK_TYPE.result(task)
    except BaseException:
        pass


def _consume_quarantined_task(task: asyncio.Task[Any]) -> None:
    try:
        _consume_authorization_task_result(task)
    finally:
        _discard_quarantined_task(task)


def _poll_quarantined_task(
    task: asyncio.Task[Any], loop: asyncio.AbstractEventLoop
) -> None:
    _AUTHORIZATION_POLL_HANDLES.pop(task, None)
    if task not in _QUARANTINED_AUTHORIZATION_TASKS:
        return
    if _authorization_task_done(task):
        try:
            _consume_authorization_task_result(task)
        finally:
            _discard_quarantined_task(task)
        return
    _schedule_quarantine_poll(task, loop)


def _schedule_quarantine_poll(
    task: asyncio.Task[Any], loop: asyncio.AbstractEventLoop
) -> None:
    if (
        task not in _QUARANTINED_AUTHORIZATION_TASKS
        or task in _AUTHORIZATION_POLL_HANDLES
    ):
        return
    try:
        handle = loop.call_later(
            _QUARANTINE_POLL_INTERVAL_SECONDS,
            _poll_quarantined_task,
            task,
            loop,
        )
    except BaseException:
        # The host loop is broken beyond the provider contract. Retain the exact
        # owned task strongly in quarantine rather than risk an unconsumed result.
        return
    _AUTHORIZATION_POLL_HANDLES[task] = handle


async def _reap_quarantined_task(task: asyncio.Task[Any]) -> None:
    try:
        try:
            await task
        except BaseException:
            pass
    finally:
        _discard_quarantined_task(task)
        reaper = asyncio.current_task()
        if reaper is not None:
            _AUTHORIZATION_REAPER_TASKS.discard(reaper)


def _start_quarantine_reaper(
    task: asyncio.Task[Any], loop: asyncio.AbstractEventLoop
) -> None:
    reaper_coroutine = _reap_quarantined_task(task)
    try:
        reaper = _NATIVE_ASYNCIO_TASK(reaper_coroutine, loop=loop)
    except BaseException:
        reaper_coroutine.close()
        if _authorization_task_done(task):
            try:
                _consume_authorization_task_result(task)
            finally:
                _discard_quarantined_task(task)
        else:
            _schedule_quarantine_poll(task, loop)
        return
    _AUTHORIZATION_REAPER_TASKS.add(reaper)


def _cancel_and_quarantine(
    task: asyncio.Task[Any], loop: asyncio.AbstractEventLoop
) -> None:
    if _authorization_task_done(task):
        _consume_authorization_task_result(task)
        return
    _QUARANTINED_AUTHORIZATION_TASKS.add(task)
    try:
        task.add_done_callback(_consume_quarantined_task)
    except BaseException:
        try:
            task.cancel()
        except BaseException:
            pass
        _start_quarantine_reaper(task, loop)
        return
    try:
        task.cancel()
    except BaseException:
        # Ownership and result consumption were established before cancellation.
        # A one-shot cancellation failure must not escape the fail-closed boundary;
        # the still-running task remains quarantined until it settles.
        pass


def _quarantined_task_count() -> int:
    """Return the owned-task quarantine size for focused invariant tests."""

    return len(_QUARANTINED_AUTHORIZATION_TASKS)


def _reaper_task_count() -> int:
    """Return the quarantine-reaper registry size for focused invariant tests."""

    return len(_AUTHORIZATION_REAPER_TASKS)


def _polling_handle_count() -> int:
    """Return the quarantine-poll handle registry size for invariant tests."""

    return len(_AUTHORIZATION_POLL_HANDLES)


def _expire_deadline(deadline: asyncio.Future[None]) -> None:
    if not deadline.done():
        deadline.set_result(None)


def _cancel_and_consume_deadline(
    deadline: asyncio.Future[None], handle: asyncio.TimerHandle
) -> None:
    handle.cancel()
    if not deadline.done():
        deadline.cancel()
    _consume_task_result(deadline)


def _validated_timeout_seconds(value: object) -> int | float | None:
    if type(value) is int:
        return value if 0 <= value <= _MAX_AUTHORIZATION_TIMEOUT_SECONDS else None
    if (
        type(value) is float
        and math.isfinite(value)
        and 0 <= value <= _MAX_AUTHORIZATION_TIMEOUT_SECONDS
    ):
        return value
    return None


async def _await_with_deadline(
    coroutine: types.CoroutineType[Any, Any, _T], timeout_seconds: object
) -> _T | _DeadlineExpired:
    """Race an owned native coroutine against a deadline."""

    deadline_seconds = _validated_timeout_seconds(timeout_seconds)
    if deadline_seconds is None:
        coroutine.close()
        return _DEADLINE_EXPIRED

    loop = asyncio.get_running_loop()
    try:
        operation = asyncio.Task(coroutine, loop=loop)
    except BaseException:
        coroutine.close()
        raise
    try:
        deadline = loop.create_future()
        try:
            handle = loop.call_later(deadline_seconds, _expire_deadline, deadline)
        except BaseException:
            deadline.cancel()
            _consume_task_result(deadline)
            raise
    except BaseException:
        _cancel_and_quarantine(operation, loop)
        raise

    try:
        done, _pending = await asyncio.wait(
            (operation, deadline), return_when=asyncio.FIRST_COMPLETED
        )
    except BaseException:
        _cancel_and_consume_deadline(deadline, handle)
        _cancel_and_quarantine(operation, loop)
        raise

    if deadline in done:
        _cancel_and_consume_deadline(deadline, handle)
        _cancel_and_quarantine(operation, loop)
        return _DEADLINE_EXPIRED

    _cancel_and_consume_deadline(deadline, handle)
    try:
        return operation.result()
    except asyncio.CancelledError:
        return _DEADLINE_EXPIRED
